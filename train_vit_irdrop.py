#!/usr/bin/env python3
"""
Train ViT-RoPE + Temporal Cross-Attention for IR Drop Prediction.
Optimized for H100 — bf16, tf32, FlashAttention (via F.scaled_dot_product_attention),
torch.compile, persistent_workers.

Usage:
    # Full training run
    python train_vit_irdrop.py \
        --dataroot /path/to/training_set/IR_drop \
        --save_path ./work_dir/vit_irdrop

    # Smoke test
    python train_vit_irdrop.py \
        --dataroot /path/to/training_set/IR_drop \
        --save_path ./work_dir/smoke \
        --max_iters 100 --model_size tiny --batch_size 2

    # Full base model on H100
    python train_vit_irdrop.py \
        --dataroot /path/to/training_set/IR_drop \
        --save_path ./work_dir/vit_base \
        --model_size base --batch_size 8 --compile
"""

import os
import sys
import json
import time
import argparse
from math import cos, pi

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "routability_ir_drop_prediction"))
from datasets.irdrop_dataset import IRDropDataset
from models.vit_rope_irdrop import ViTRoPEIRDrop


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "tiny":  dict(embed_dim=384,  depth=12, num_heads=6,  patch_size=16, mlp_ratio=4.0),
    "small": dict(embed_dim=512,  depth=12, num_heads=8,  patch_size=16, mlp_ratio=4.0),
    "base":  dict(embed_dim=768,  depth=12, num_heads=12, patch_size=16, mlp_ratio=4.0),
}


# ---------------------------------------------------------------------------
# LR scheduler
# ---------------------------------------------------------------------------

class CosineRestartLr:
    def __init__(self, base_lr, max_iters, min_lr=1e-7):
        self.base_lr = base_lr
        self.max_iters = max_iters
        self.min_lr = min_lr

    def get_lr(self, iter_num):
        alpha = min(iter_num / self.max_iters, 1.0)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (cos(pi * alpha) + 1)


# ---------------------------------------------------------------------------
# Infinite data loader
# ---------------------------------------------------------------------------

class IterLoader:
    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)
        return data

    def __iter__(self):
        return self


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="ViT-RoPE IR Drop Training (H100)")

    # Data
    p.add_argument("--dataroot", required=True)
    p.add_argument("--ann_file_train", default="routability_ir_drop_prediction/files/train_N28.csv")
    p.add_argument("--ann_file_test", default="routability_ir_drop_prediction/files/test_N28.csv")

    # Model
    p.add_argument("--model_size", default="tiny", choices=["tiny", "small", "base"])
    p.add_argument("--in_channels", type=int, default=1)
    p.add_argument("--out_channels", type=int, default=4)
    p.add_argument("--drop_path_rate", type=float, default=0.1)

    # Training
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_iters", type=int, default=200000)
    p.add_argument("--loss_weight", type=float, default=100.0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--save_path", default="work_dir/vit_irdrop")
    p.add_argument("--save_freq", type=int, default=10000)
    p.add_argument("--print_freq", type=int, default=100)
    p.add_argument("--pretrained", default=None)

    # H100
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    p.add_argument("--no_bf16", action="store_true")
    p.add_argument("--no_tf32", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    # ---- H100 setup ----
    if not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    use_bf16 = not args.no_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"BF16: {use_bf16} | TF32: {not args.no_tf32}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ---- Save config ----
    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ---- Dataset ----
    print("===> Loading dataset")
    dataset = IRDropDataset(
        ann_file=args.ann_file_train,
        dataroot=args.dataroot,
        test_mode=False,
    )
    loader = IterLoader(DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    ))
    print(f"Dataset: {len(dataset)} samples | batch_size={args.batch_size}")

    # ---- Model ----
    print("===> Building model")
    cfg = MODEL_CONFIGS[args.model_size].copy()
    model = ViTRoPEIRDrop(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        drop_path_rate=args.drop_path_rate,
        **cfg,
    )
    if args.pretrained:
        model.init_weights(pretrained=args.pretrained)
    else:
        model.init_weights()

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: ViT-RoPE-{args.model_size} | {n_params / 1e6:.1f}M params")
    print(f"  embed_dim={cfg['embed_dim']} depth={cfg['depth']} heads={cfg['num_heads']} patch={cfg['patch_size']}")

    if args.compile and hasattr(torch, "compile"):
        print("===> torch.compile enabled")
        model = torch.compile(model)

    # ---- Loss / Optimizer ----
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = CosineRestartLr(args.lr, args.max_iters)

    # ---- Train ----
    print(f"===> Training for {args.max_iters} iterations\n")
    model.train()
    running_loss = 0.0
    iter_num = 0
    t0 = time.time()

    while iter_num < args.max_iters:
        with tqdm(total=args.print_freq, desc=f"iter {iter_num}", leave=False) as bar:
            for feature, label, _ in loader:
                feature = feature.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                lr = scheduler.get_lr(iter_num)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_bf16):
                    pred = model(feature)
                    loss = args.loss_weight * criterion(pred, label)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                iter_num += 1
                bar.update(1)

                if iter_num % args.print_freq == 0:
                    break

        avg_loss = running_loss / args.print_freq
        elapsed = time.time() - t0
        speed = iter_num / elapsed
        eta_h = (args.max_iters - iter_num) / max(speed, 1e-9) / 3600

        print(f"[{iter_num}/{args.max_iters}] loss={avg_loss:.4f} lr={lr:.2e} {speed:.1f} it/s ETA={eta_h:.1f}h")
        running_loss = 0.0

        if iter_num % args.save_freq == 0:
            path = os.path.join(args.save_path, f"model_iters_{iter_num}.pth")
            torch.save({"state_dict": model.state_dict(), "iter_num": iter_num, "args": vars(args)}, path)
            print(f"  -> Checkpoint: {path}")

    # Final checkpoint
    path = os.path.join(args.save_path, "model_final.pth")
    torch.save({"state_dict": model.state_dict(), "iter_num": iter_num, "args": vars(args)}, path)
    print(f"\n===> Done. Final checkpoint: {path}")


if __name__ == "__main__":
    main()
