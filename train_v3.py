"""
SR-Diffusion v3: Train from saved weights.
Loads from /root/autodl-tmp/sr_diffusion_v3_weights and resumes training.
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

# ── 关键: v3 的 SRDiffusionConfig 需要导入为当前模块的类 ──
# from_pretrained 会 import class from model source
# 所以我们直接用 in-project import
from model_v3_test import SRDiffusionConfig, SRDiffusion

DATA_DIR    = "/root/autodl-tmp/DIV2K/HighResolution/DIV2K_train_HR"
OUTPUT_DIR  = "/root/autodl-tmp/sr_diffusion_v3_output"
WEIGHTS_DIR = "/root/autodl-tmp/sr_diffusion_v3_weights"

EPOCHS        = 100
BATCH_SIZE    = 1
GRAD_ACCUM    = 4
LR            = 5e-5
WEIGHT_DECAY  = 0.01
NUM_WORKERS   = 2
SAVE_EVERY    = 500
T0            = 1000
MAX_GRAD_NORM = 1.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load model from saved weights ──
print("=" * 60)
print("Loading SR-Diffusion v3 from saved weights...")
print(f"  Source: {WEIGHTS_DIR}")
model = SRDiffusion.from_pretrained(WEIGHTS_DIR)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Trainable params: {trainable/1e9:.2f}B")
print("=" * 60)

# ── Dataset ──
class HRDataset(Dataset):
    def __init__(self, root: str, size: int):
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

train_dataset = HRDataset(DATA_DIR, 1024)

# ── Optimizer (8-bit AdamW) ──
optimizer = bnb.optim.AdamW8bit(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

# ── Training args ──
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=1,
    save_steps=SAVE_EVERY,
    save_total_limit=3,
    fp16=False,
    dataloader_num_workers=NUM_WORKERS,
    report_to=[],
    max_grad_norm=MAX_GRAD_NORM,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    gradient_checkpointing=False,
    remove_unused_columns=True,
    ddp_find_unused_parameters=False,
)

# ── Trainer ──
class V3Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        hr = inputs["hr"].cuda()
        outputs = model(hr=hr)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

trainer = V3Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
    optimizers=(optimizer, None),
)

print("\n" + "=" * 60)
print("Starting v3 training...")
print("=" * 60)
trainer.train()
