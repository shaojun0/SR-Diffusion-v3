"""
SR-Diffusion v2 Training

Training loop:
  1. Load HR images → random crop 512×512
  2. Forward: fp32 autocast → noise prediction loss
  3. Backward: raw gradients (8-bit AdamW handles precision internally)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  4. Step: gradient accumulation × 4, cosine annealing LR
  5. Checkpoint: save every N steps + best loss

全参数训练 (2.09B), fp32 + 8-bit AdamW + gradient checkpointing.
"""

import os
import sys
import time
import random
import argparse
from glob import glob

import torch
import torch.nn as nn
import bitsandbytes as bnb
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ── Local imports ───────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model_v2 import Config, build_model


# ═══════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════

class HRDataset(Dataset):
    """Load HR images from directory, random crop to size×size."""

    def __init__(self, root: str, size: int = 512):
        self.size = size
        self.files = []
        for ext in ("png", "jpg", "jpeg", "webp"):
            self.files.extend(
                glob(os.path.join(root, f"**/*.{ext}"), recursive=True)
            )
        if not self.files:
            raise RuntimeError(f"No images found in {root}")
        print(f"  [Data] {len(self.files)} images from {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        img = Image.open(self.files[idx]).convert("RGB")
        w, h = img.size

        if w < self.size or h < self.size:
            img = img.resize(
                (max(w, self.size), max(h, self.size)), Image.BICUBIC
            )
            w, h = img.size

        left = random.randint(0, w - self.size)
        top  = random.randint(0, h - self.size)
        img  = img.crop((left, top, left + self.size, top + self.size))

        return transforms.ToTensor()(img) * 2.0 - 1.0          # → [-1, 1]


# ═══════════════════════════════════════════════════════════════
# Training Entry
# ═══════════════════════════════════════════════════════════════

def train(args):
    device = torch.device("cuda")
    print(f"[Train] Device: {device} | fp32 (no autocast) precision | 8-bit AdamW")

    # ── Config ──────────────────────────────────────────────────
    cfg = Config()
    cfg.dino_dir    = "/root/autodl-tmp/sr_dinov2/dinov2-giant"
    cfg.sd_model_id = '/root/autodl-tmp/sd_models/AI-ModelScope/stable-diffusion-2-1'
    cfg.output_dir  = "/root/sr_diffusion_output"
    cfg.lr          = args.lr
    cfg.batch_size  = args.batch_size
    cfg.grad_accum  = args.grad_accum
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Data ────────────────────────────────────────────────────
    dataset = HRDataset(args.data_dir, cfg.image_size)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── Model ───────────────────────────────────────────────────
    model = build_model(cfg, device=device)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    print(f"  [Train] Optimizer params: {sum(p.numel() for p in params)/1e9:.2f}B")

    # ── Optimizer & Scheduler ───────────────────────────────────
    opt = bnb.optim.AdamW8bit(params, lr=cfg.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=args.t0, T_mult=2, eta_min=cfg.lr * 0.01
    )

    # ── Resume ──────────────────────────────────────────────────
    step, epoch_start, best_loss = 0, 0, float("inf")
    if args.resume and os.path.exists(args.resume):
        print(f"  [Resume] Loading {args.resume}")
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        step       = ck.get("step", 0)
        epoch_start = ck.get("epoch", 0) + 1
        best_loss  = ck.get("best_loss", float("inf"))
        print(
            f"  [Resume] epoch={epoch_start}  step={step}  "
            f"best_loss={best_loss:.6f}"
        )

    # ── Logging ─────────────────────────────────────────────────
    writer = SummaryWriter(os.path.join(cfg.output_dir, "tb"))

    print(f"\n{'='*60}")
    print(
        f"Training: {len(dataset)} imgs | BS={cfg.batch_size}×{cfg.grad_accum} "
        f"| LR={cfg.lr}"
    )
    print(f"Resolution: {cfg.image_size}² | max_eig: {cfg.max_eig} | fp32")
    print(f"{'='*60}\n")

    # ── Cleanup stale checkpoints ────
    import glob
    all_pts = glob.glob(os.path.join(cfg.output_dir, "*.pt"))
    keep = set()
    if args.resume and os.path.exists(args.resume):
        keep.add(os.path.basename(args.resume))
    for f in all_pts:
        bn = os.path.basename(f)
        if bn == "best.pt" or bn in keep:
            continue
        os.remove(f)
        print(f"  [Cleanup] Removed stale {bn}")

    # ── Loop ────────────────────────────────────────────────────
    for epoch in range(epoch_start, args.epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0

        for i, hr in enumerate(pbar):
            hr = hr.to(device, non_blocking=True)

            # Forward (fp32 autocast)
            # fp32 forward for 1024^2 stability
            loss, pred, noise = model(hr)
            loss = loss / cfg.grad_accum
            loss.backward()

            if (i + 1) % cfg.grad_accum == 0:
                nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                opt.step()
                opt.zero_grad()
                sched.step()
                step += 1

                writer.add_scalar("loss", loss.item() * cfg.grad_accum, step)
                writer.add_scalar("lr", sched.get_last_lr()[0], step)

                pbar.set_postfix(
                    loss=f"{loss.item() * cfg.grad_accum:.4f}",
                    lr=f"{sched.get_last_lr()[0]:.2e}",
                    step=step,
                )

                current = loss.item() * cfg.grad_accum
                ckpt = {
                    "step": step,
                    "epoch": epoch,
                    "best_loss": best_loss,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(),
                    "loss": current,
                }

                # ── Periodic checkpoint ──
                if step % args.save_every == 0:
                    path = os.path.join(cfg.output_dir, f"ckpt_step{step}.pt")
                    torch.save(ckpt, path)
                    old = os.path.join(
                        cfg.output_dir, f"ckpt_step{step - args.save_every}.pt"
                    )
                    if os.path.exists(old):
                        os.remove(old)

                # ── Best checkpoint ──
                if current < best_loss:
                    best_loss = current
                    torch.save(ckpt, os.path.join(cfg.output_dir, "best.pt"))
                    print(f"  best={best_loss:.6f} @ step {step}")

            epoch_loss += loss.item() * cfg.grad_accum

        avg = epoch_loss / len(loader)
        gpu = torch.cuda.max_memory_allocated() / 1e9
        print(f"[Epoch {epoch+1}] loss={avg:.4f} | GPU peak={gpu:.1f}GB")
        writer.add_scalar("epoch_loss", avg, epoch)

    writer.close()
    print("\nTraining complete.")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SR-Diffusion v2 Training")
    p.add_argument("--data_dir",   default="/root/autodl-tmp/DIV2K_train_HR")
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch_size", type=int,   default=1)
    p.add_argument("--grad_accum", type=int,   default=4)
    p.add_argument("--lr",         type=float, default=5e-5)
    p.add_argument("--num_workers", type=int,  default=2)
    p.add_argument("--save_every", type=int,   default=500)
    p.add_argument("--t0",         type=int,   default=1000)
    p.add_argument("--resume",     default=None)
    train(p.parse_args())
