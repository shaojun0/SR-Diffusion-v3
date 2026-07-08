"""
SR-Diffusion v2: DINOv2-giant + SD 2.1 U-Net Diffusion 超分辨率.
Resume version — auto-detects latest checkpoint.
With skip_pretrained support to save disk space.
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
from model_v2 import SRDiffusionConfig as Config, build_model

DATA_DIR      = "/root/autodl-tmp/DIV2K/HighResolution/DIV2K_train_HR"
OUTPUT_DIR    = "/root/autodl-tmp/sr_diffusion_output"
DINO_DIR      = "/root/autodl-tmp/sr_dinov2/dinov2-giant"
SD_MODEL      = "/root/autodl-tmp/sd_models/AI-ModelScope/stable-diffusion-2-1"

EPOCHS        = 100
BATCH_SIZE    = 1
GRAD_ACCUM    = 4
LR            = 5e-5
WEIGHT_DECAY  = 0.01
NUM_WORKERS   = 2
SAVE_EVERY    = 500
T0            = 1000
MAX_GRAD_NORM = 1.0

cfg = Config()
cfg.dino_dir    = DINO_DIR
cfg.sd_model_id = SD_MODEL
cfg.output_dir  = OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

class HRDataset(Dataset):
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
        return transforms.ToTensor()(img) * 2.0 - 1.0

def collate_fn(batch):
    return {"hr": torch.stack(batch, dim=0)}

# ── Detect checkpoint ──
# Search checkpoints
ckpt_dirs = sorted(glob(os.path.join(OUTPUT_DIR, "checkpoint-*")))
resume_ckpt = ckpt_dirs[-1] if ckpt_dirs else None
skip_pretrained = resume_ckpt is not None

model = build_model(cfg, skip_pretrained=skip_pretrained)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  [Train] Trainable params: {trainable/1e9:.2f}B")

train_dataset = HRDataset(DATA_DIR, cfg.image_size)

opt = bnb.optim.AdamW8bit(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    opt, T_0=T0, T_mult=2, eta_min=LR * 0.01
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    logging_steps=1,
    save_steps=SAVE_EVERY,
    save_total_limit=2,
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
    print(f"Training: {len(train_dataset)} imgs | BS={BATCH_SIZE}x{GRAD_ACCUM} | LR={LR}")
    print(f"Resolution: {cfg.image_size}² | fp32")
    if skip_pretrained:
        print(f"Mode: config-init + checkpoint resume (disk-saving)")
    print(f"{'='*60}\n")

    if resume_ckpt:
        print(f"Resuming from: {resume_ckpt}")
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        trainer.train()

    trainer.save_model(OUTPUT_DIR)
    print("Training complete.")
