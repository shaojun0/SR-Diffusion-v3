"""
SR-Diffusion v2: DINOv2-giant + SD 2.1 U-Net Diffusion 超分辨率.

Pipeline:
  1. HR image → SVD → DINOv2 Encoder → cross-attention tokens
  2. LR bicubic → VAE Encoder → condition latent
  3. HR → VAE Encoder → +noise → U-Net(cross_attn) → x₀ pred
  4. Loss: MSE(x₀_pred, hr_z) — pure VAE latent reconstruction

全参数训练 (2.09B), fp32 + 8-bit AdamW + gradient checkpointing.
"""

import os, sys
import torch
import bitsandbytes as bnb
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from glob import glob
from tqdm.auto import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from model_v2 import Config, build_model

# ── Paths ──────────────────────────────────────────────────────
DATA_DIR      = "/root/autodl-tmp/DIV2K_train_HR"
OUTPUT_DIR    = "/root/sr_diffusion_output"
DINO_DIR      = "/root/autodl-tmp/sr_dinov2/dinov2-giant"
SD_MODEL      = "/root/autodl-tmp/sd_models/AI-ModelScope/stable-diffusion-2-1"

# ── Hyperparams ────────────────────────────────────────────────
EPOCHS        = 100
BATCH_SIZE    = 1
GRAD_ACCUM    = 4
LR            = 5e-5
WEIGHT_DECAY  = 0.01
NUM_WORKERS   = 2
SAVE_EVERY    = 500
T0            = 1000
MAX_GRAD_NORM = 1.0

# ── Config ─────────────────────────────────────────────────────
cfg = Config()
cfg.dino_dir    = DINO_DIR
cfg.sd_model_id = SD_MODEL
cfg.output_dir  = OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Dataset ────────────────────────────────────────────────────

class HRDataset(Dataset):
    """Load HR images, random crop to cfg.image_size × cfg.image_size."""

    def __init__(self, root: str, size: int):
        import random as _rnd
        self.size = size
        self.files = []
        for ext in ("png", "jpg", "jpeg", "webp"):
            self.files.extend(glob(os.path.join(root, f"**/*.{ext}"), recursive=True))
        if not self.files:
            raise RuntimeError(f"No images found in {root}")
        print(f"  [Data] {len(self.files)} images from {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        import random as _rnd
        img = Image.open(self.files[idx]).convert("RGB")
        w, h = img.size
        if w < self.size or h < self.size:
            img = img.resize((max(w, self.size), max(h, self.size)), Image.BICUBIC)
            w, h = img.size
        left = _rnd.randint(0, w - self.size)
        top  = _rnd.randint(0, h - self.size)
        img  = img.crop((left, top, left + self.size, top + self.size))
        return transforms.ToTensor()(img) * 2.0 - 1.0  # → [-1, 1]

# ── Model ──────────────────────────────────────────────────────
model = build_model(cfg)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  [Train] Trainable params: {trainable/1e9:.2f}B")

# ── DataLoader ─────────────────────────────────────────────────
dataset = HRDataset(DATA_DIR, cfg.image_size)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,
)

# ── Optimizer & Scheduler ──────────────────────────────────────
opt = bnb.optim.AdamW8bit(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    opt, T_0=T0, T_mult=2, eta_min=LR * 0.01
)

# ── Accelerator (handles device, grad_accum, backward) ─────────
accelerator = Accelerator(
    gradient_accumulation_steps=GRAD_ACCUM,
    mixed_precision="no",
)
model, opt, loader, sched = accelerator.prepare(model, opt, loader, sched)

# ── Training ───────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"Training: {len(dataset)} imgs | BS={BATCH_SIZE}×{GRAD_ACCUM} | LR={LR}")
    print(f"Resolution: {cfg.image_size}² | max_eig: {cfg.max_eig} | fp32")
    print(f"{'='*60}\n")

    best_loss = float("inf")
    step = 0

    for epoch in range(EPOCHS):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for hr in pbar:
            with accelerator.accumulate(model):
                loss, _, _ = model(hr)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                opt.step()
                sched.step()
                opt.zero_grad()

            if accelerator.sync_gradients:
                step += 1
                loss_val = loss.detach().item()
                pbar.set_postfix(loss=f"{loss_val:.4f}", step=step)

                # ── Best checkpoint ──
                if loss_val < best_loss:
                    best_loss = loss_val
                    accelerator.save(
                        accelerator.unwrap_model(model).state_dict(),
                        os.path.join(OUTPUT_DIR, "best.pt"),
                    )
                    if accelerator.is_main_process:
                        print(f"  best={best_loss:.6f} @ step {step}")

                # ── Periodic checkpoint ──
                if step % SAVE_EVERY == 0:
                    accelerator.save(
                        {
                            "model": accelerator.unwrap_model(model).state_dict(),
                            "optimizer": opt.state_dict(),
                            "scheduler": sched.state_dict(),
                            "step": step,
                            "epoch": epoch,
                            "best_loss": best_loss,
                        },
                        os.path.join(OUTPUT_DIR, f"ckpt_step{step}.pt"),
                    )

    accelerator.end_training()
    print("Training complete.")
