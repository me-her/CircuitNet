"""
ViT + 2D RoPE with Temporal Cross-Attention for IR Drop Prediction.

Architecture:
  - Plain ViT encoder with 2D Rotary Position Embeddings (no learned pos embed)
  - Shared encoder across T power timesteps
  - Temporal cross-attention fuses multi-timestep features
  - Lightweight conv decoder upsamples back to full resolution
  - Output gated by input power maps (same protocol as MAVI baseline)
  - Flash Attention 3 on Hopper GPUs (H100), SDPA fallback elsewhere

References:
  - "Rotary Position Embedding for Vision Transformer" (ECCV 2024)
  - FA3 kernel: varunneal/flash-attention-3 via HuggingFace kernels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Flash Attention 3 loader (same pattern as karpathy/nanochat)
# ---------------------------------------------------------------------------

def _load_flash_attention_3():
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        if major != 9:  # FA3 requires Hopper (sm90) — H100, H200
            return None
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel("varunneal/flash-attention-3").flash_attn_interface
    except Exception:
        return None

_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None


def flash_attn_func(q, k, v):
    """Unified attention: FA3 on Hopper, F.scaled_dot_product_attention elsewhere.

    FA3 expects (B, N, H, D) layout.
    SDPA expects (B, H, N, D) layout.
    Input q/k/v here are (B, H, N, D).
    """
    if HAS_FA3:
        # Transpose to FA3 layout: (B, H, N, D) -> (B, N, H, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = _fa3.flash_attn_func(q, k, v, causal=False)
        return out.transpose(1, 2)  # back to (B, H, N, D)
    else:
        return F.scaled_dot_product_attention(q, k, v)


# ---------------------------------------------------------------------------
# 2D RoPE (Axial variant — split head dim in half for x and y axes)
# ---------------------------------------------------------------------------

def build_2d_rope_freqs(head_dim, h, w, theta=10000.0, device=None):
    """Precompute 2D RoPE frequency tensors for a grid of size (h, w).

    Returns complex-valued freqs of shape (h*w, head_dim//2).
    """
    half = head_dim // 2
    # Split half into two equal parts for h-axis and w-axis
    dim_h = half // 2
    dim_w = half - dim_h

    freqs_h = 1.0 / (theta ** (torch.arange(0, dim_h, dtype=torch.float32, device=device) / dim_h))
    freqs_w = 1.0 / (theta ** (torch.arange(0, dim_w, dtype=torch.float32, device=device) / dim_w))

    pos_h = torch.arange(h, dtype=torch.float32, device=device)
    pos_w = torch.arange(w, dtype=torch.float32, device=device)

    # Outer products
    angles_h = torch.outer(pos_h, freqs_h)  # (h, dim_h)
    angles_w = torch.outer(pos_w, freqs_w)  # (w, dim_w)

    # Broadcast to grid
    angles_h = angles_h[:, None, :].expand(h, w, dim_h).reshape(h * w, dim_h)
    angles_w = angles_w[None, :, :].expand(h, w, dim_w).reshape(h * w, dim_w)

    angles = torch.cat([angles_h, angles_w], dim=-1)  # (h*w, half)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)  # complex
    return freqs_cis


def apply_rope(x, freqs_cis):
    """Apply rotary embeddings to query or key tensor.

    x: (B, num_heads, N, head_dim)
    freqs_cis: (N, head_dim//2) complex
    """
    # Reshape x to pairs
    B, H, N, D = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(B, H, N, D // 2, 2))
    # Broadcast freqs
    freqs = freqs_cis[None, None, :N, :]  # (1, 1, N, D//2)
    x_rotated = torch.view_as_real(x_complex * freqs).reshape(B, H, N, D)
    return x_rotated.type_as(x)


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1 - self.p
        mask = x.new_empty(x.shape[0], *((1,) * (x.ndim - 1))).bernoulli_(keep).div_(keep)
        return x * mask


class RoPEAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, freqs_cis):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, heads, N, head_dim)

        # Apply RoPE to q and k
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        # FA3 on Hopper, SDPA fallback
        x = flash_attn_func(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RoPEAttention(dim, num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x, freqs_cis):
        x = x + self.drop_path(self.attn(self.norm1(x), freqs_cis))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # (B, C, H/p, W/p)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = self.norm(x)
        return x, H, W


# ---------------------------------------------------------------------------
# Temporal Cross-Attention
# ---------------------------------------------------------------------------

class TemporalCrossAttention(nn.Module):
    """Fuses T timestep features via cross-attention."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, x):
        """x: (B, T, N, C) -> (B, N, C)"""
        B, T, N, C = x.shape

        # Query from temporal mean, keys/values from all timesteps
        q_in = self.norm_q(x.mean(dim=1))  # (B, N, C)
        kv_in = self.norm_kv(x.reshape(B, T * N, C))

        q = self.q_proj(q_in).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(kv_in).reshape(B, T * N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        out = flash_attn_func(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out) + q_in


# ---------------------------------------------------------------------------
# Conv Decoder
# ---------------------------------------------------------------------------

class ConvUpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(self.up(x))


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class ViTRoPEIRDrop(nn.Module):
    """
    ViT + 2D RoPE encoder → Temporal Cross-Attention → Conv Decoder.

    Input:  (B, 1, T, H, W)  — same format as MAVI baseline
    Output: (B, H, W)        — predicted IR drop map

    Model sizes:
        tiny:  embed_dim=384,  depth=12, heads=6   (~22M params)
        small: embed_dim=512,  depth=12, heads=8   (~38M params)
        base:  embed_dim=768,  depth=12, heads=12  (~86M params)
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=4,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch embed (shared across timesteps)
        self.patch_embed = PatchEmbed(patch_size, in_channels, embed_dim)

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Temporal fusion
        self.temporal_attn = TemporalCrossAttention(embed_dim, num_heads)

        # Decoder: upsample from (H/patch_size, W/patch_size) to (H, W)
        # For patch_size=16: need 4 x2 upsamples (16x)
        # For patch_size=8: need 3 x2 upsamples (8x)
        n_ups = {4: 2, 8: 3, 16: 4}[patch_size]
        decoder_layers = []
        ch = embed_dim
        for i in range(n_ups):
            ch_out = max(ch // 2, out_channels)
            if i == n_ups - 1:
                ch_out = out_channels
            decoder_layers.append(ConvUpsampleBlock(ch, ch_out))
            ch = ch_out
        self.decoder = nn.ModuleList(decoder_layers)

        # Cached RoPE frequencies
        self._cached_freqs = None
        self._cached_hw = None

    def _get_freqs(self, h, w, device):
        if self._cached_hw != (h, w) or self._cached_freqs is None or self._cached_freqs.device != device:
            head_dim = self.embed_dim // self.blocks[0].attn.num_heads
            self._cached_freqs = build_2d_rope_freqs(head_dim, h, w, device=device)
            self._cached_hw = (h, w)
        return self._cached_freqs

    def _encode(self, x_2d):
        """Encode a single 2D frame through ViT."""
        x, H, W = self.patch_embed(x_2d)
        freqs = self._get_freqs(H, W, x.device)
        for blk in self.blocks:
            x = blk(x, freqs)
        x = self.norm(x)
        return x, H, W

    def forward(self, x):
        """x: (B, 1, T, H, W)"""
        B, _, T, H_in, W_in = x.shape
        x_in = x[:, :, :self.out_channels, :, :]  # for gating

        # Encode each timestep (shared weights)
        encoded = []
        for t in range(T):
            frame = x[:, 0, t, :, :].unsqueeze(1)  # (B, 1, H, W)
            enc, H_enc, W_enc = self._encode(frame)
            encoded.append(enc)

        # Temporal fusion
        stacked = torch.stack(encoded, dim=1)  # (B, T, N, C)
        fused = self.temporal_attn(stacked)     # (B, N, C)

        # Reshape to 2D
        feat = fused.transpose(1, 2).reshape(B, self.embed_dim, H_enc, W_enc)

        # Decode
        for layer in self.decoder:
            feat = layer(feat)

        logits = feat  # (B, out_channels, H, W)

        # Gate by input power (same as MAVI)
        logits = x_in.squeeze(1) * logits
        return torch.sum(logits, dim=1)

    def init_weights(self, pretrained=None, **kwargs):
        if isinstance(pretrained, str):
            state = torch.load(pretrained, map_location="cpu")
            key = "state_dict" if "state_dict" in state else "model"
            missing = self.load_state_dict(state[key], strict=False)
            print(f"Loaded pretrained, missing: {missing}")
        else:
            self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
