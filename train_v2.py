"""
SR-Diffusion v2: DINOv2-giant + SD 2.1 U-Net Diffusion 超分辨率.

Pipeline:
  1. HR image → SVD → DINOv2 Encoder → cross-attention tokens
  2. LR bicubic → VAE Encoder → condition latent
  3. HR → VAE Encoder → +noise → U-Net(cross_attn) → x₀ pred
  4. Loss: ε-MSE + 0.5·x₀-MSE — noise pred + latent KL constraint
  5. CFG-style conditioning dropout (15%) to prevent mode collapse

全参数训练 (2.09B), fp32 + 8-bit AdamW + gradient checkpointing.
"""

import os, sys
import torch
import bitsandbytes as bnb
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from glob import glob

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

def collate_fn(batch):
    return {"hr": torch.stack(batch, dim=0)}

# ── Model ──────────────────────────────────────────────────────
model = build_model(cfg)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  [Train] Trainable params: {trainable/1e9:.2f}B")

# ── Dataset ────────────────────────────────────────────────────
train_dataset = HRDataset(DATA_DIR, cfg.image_size)

# ── Optimizer & Scheduler ──────────────────────────────────────
opt = bnb.optim.AdamW8bit(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    opt, T_0=T0, T_mult=2, eta_min=LR * 0.01
)

# ── Training ───────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    logging_steps=1,
    save_steps=SAVE_EVERY,
    save_total_limit=3,
    max_grad_norm=MAX_GRAD_NORM,
    weight_decay=WEIGHT_DECAY,
    fp16=False,
    dataloader_num_workers=NUM_WORKERS,
    remove_unused_columns=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
    optimizers=(opt, sched),
)

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"Training: {len(train_dataset)} imgs | BS={BATCH_SIZE}×{GRAD_ACCUM} | LR={LR}")
    print(f"Resolution: {cfg.image_size}² | max_eig: {cfg.max_eig} | fp32")
    print(f"{'='*60}\n")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print("Training complete.")
