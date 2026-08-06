"""
SR-Qwen-VL v12: Training script (multi-GPU via HF Trainer DataParallel)

Usage: python train.py [--test] [--epochs N]
  --test   Quick smoke-test: 50 steps, no save
  --epochs N   Override default 3 epochs
"""
import os, sys, io, argparse
import torch
import numpy as np
from PIL import Image
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer
from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from model import SRQwenVLConfig, SRQwenVLv12

# ═════════════════════════════════════════
# Paths
# ═════════════════════════════════════════

OUTPUT_DIR  = "output/sr_qwen_vl_v12"
DINOV2_DIR  = "/root/autodl-tmp/sr_dinov2/models/AI-ModelScope--dinov2-giant/snapshots/master/"
QWEN_DIR    = "/root/autodl-tmp/qwen3_5_4B/"
DATA_PATH   = "/root/autodl-tmp/construction_site_zh/"

# ═════════════════════════════════════════
# SVD + Training Hyperparams
# ═════════════════════════════════════════

SVD_MAX_EIG = 128
SVD_ENERGY_THRESHOLD = 0.99
SVD_IMAGE_SIZE = 1024
MAX_LENGTH = 256

EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 4
LR = 1e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.03
NUM_WORKERS = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ═════════════════════════════════════════
# Image -> feature map helper
# ═════════════════════════════════════════

def image_to_feature(image_input, image_size=1024):
    """bytes/Image -> (1024,1024) centered float tensor."""
    if isinstance(image_input, Image.Image):
        img = image_input.convert("L")
    else:
        img = Image.open(io.BytesIO(image_input)).convert("L")
    img = img.resize((image_size, image_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr)
    return t - t.mean()

# ═════════════════════════════════════════
# Dataset (raw bytes + text)
# ═════════════════════════════════════════

class ParquetDataset(Dataset):
    def __init__(self, hf_dataset):
        self.ds = hf_dataset

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        caption = row.get("zh_caption", row["image_caption"])
        violations = row.get("violations", "")
        base = f"描述这张建筑工地图片：{caption}"
        text = base if violations is None else f"{base}\n隐患：{violations}"
        return {"image_bytes": row["image"], "text": text}

# ═════════════════════════════════════════
# Collator: SVD + tokenize
# ═════════════════════════════════════════

class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        texts = [b["text"] for b in batch]
        images = tuple(b["image_bytes"] for b in batch)

        # Batched SVD
        features = torch.stack([image_to_feature(b) for b in images])
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(features, full_matrices=False)
        S_sq = S * S
        total_e = S_sq.sum(dim=1, keepdim=True)
        cumsum = torch.cumsum(S_sq, dim=1)
        n_per = (cumsum / total_e < SVD_ENERGY_THRESHOLD).sum(dim=1) + 1
        n_per = n_per.clamp(min=32, max=SVD_MAX_EIG)

        svd_mats = []
        max_n = 0
        for k in range(len(images)):
            n = n_per[k].item()
            m = torch.cat([U[k, :, :n].T, Vh[k, :n, :]], dim=0)
            svd_mats.append(m)
            max_n = max(max_n, m.size(0))

        # Pad to batch max
        full_dim = max_n
        padded = []
        for m in svd_mats:
            if m.size(0) < full_dim:
                m = torch.cat([m, torch.zeros(full_dim - m.size(0), m.size(1))], dim=0)
            padded.append(m)
        svd_matrix = torch.stack(padded, dim=0)

        # Tokenize
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100

        return {"svd_matrix": svd_matrix, "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"], "labels": labels}

# ═════════════════════════════════════════
# Trainer (fixed compute_loss — no model.device)
# ═════════════════════════════════════════

class SRTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Get device from input tensors (not model.device — DataParallel-safe)
        device = inputs["svd_matrix"].device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        outputs = model(**inputs)
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]

# ═════════════════════════════════════════
# Main
# ═════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Quick smoke test: 50 steps, no save")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint dir")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"SR-Qwen-VL v12 Training")
    print(f"  Mode: {'TEST (50 steps)' if args.test else 'FULL TRAIN'}")
    print(f"  Epochs: {args.epochs}  Batch: {BATCH_SIZE}×{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}")
    print(f"  LR: {LR}  Warmup: {WARMUP_RATIO*100:.0f}%")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(QWEN_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    ds = load_dataset("parquet", data_files={
        "train": os.path.join(DATA_PATH, "train.parquet"),
        "test": os.path.join(DATA_PATH, "test.parquet"),
    }, cache_dir="./cache")
    train_ds = ParquetDataset(ds["train"])
    test_ds = ParquetDataset(ds["test"])
    print(f"Train: {len(train_ds)}  Test: {len(test_ds)}")

    # Model — pre-load pretrained weights BEFORE Trainer wraps with DataParallel
    # This avoids StopIteration on replicate (model.parameters() empty on replicas)
    config = SRQwenVLConfig(dino_dir=DINOV2_DIR, qwen_dir=QWEN_DIR,
                            svd_max_eig=SVD_MAX_EIG, svd_energy_threshold=SVD_ENERGY_THRESHOLD)
    model = SRQwenVLv12(config)
    model._enable_gradient_checkpointing()
    model._ensure_pretrained()  # ⚡ load DINO+Qwen before DataParallel wrapping
    model = model.bfloat16()    # Match DINO/Qwen dtype (DataParallel threads lack autocast)
    model.log_params()

    # Collator
    collator = Collator(tokenizer)

    # Training args
    save_steps = 1000000 if args.test else 880
    eval_strategy = "no" if args.test else "epoch"
    save_strategy = "no" if args.test else "epoch"

    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=50 if args.test else -1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=10,
        save_steps=save_steps,
        save_total_limit=3,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        bf16=True,
        dataloader_num_workers=NUM_WORKERS,
        report_to=[],
        max_grad_norm=MAX_GRAD_NORM,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        remove_unused_columns=False,
    )

    trainer = SRTrainer(model=model, args=training_args,
                        train_dataset=train_ds, eval_dataset=test_ds,
                        data_collator=collator)

    trainer.train()

    if not args.test:
        final_dir = os.path.join(OUTPUT_DIR, "final")
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"\n✅ Final model saved to {final_dir}")
    else:
        print("\n✅ Smoke test complete")

if __name__ == "__main__":
    main()
