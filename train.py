"""
SR-Qwen-VL v10: Training with HF Trainer + bitsandbytes 8-bit AdamW

SVD(1024×1024) → (2n,1024) matrix → DINO Encoder → MLP → Qwen → Text

v4: Dataset 从 Parquet 加载（image_bytes + text），
    SVD + tokenize 全部在 collate_fn 里完成，batched SVD。
"""
import os, sys, io
import torch
import numpy as np
import pandas as pd
import bitsandbytes as bnb
from PIL import Image
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from model import SRQwenVLConfig, SRQwenVLv10

# ═══════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════

PARQUET_TRAIN   = "/root/autodl-tmp/construction_site_zh/train.parquet"
PARQUET_TEST    = "/root/autodl-tmp/construction_site_zh/test.parquet"
OUTPUT_DIR      = "/root/autodl-tmp/sr_qwen_vl_v10_output"
DINOV2_DIR      = "/root/autodl-tmp/sr_dinov2/models/AI-ModelScope--dinov2-giant"
QWEN_DIR        = "/root/autodl-tmp/qwen3.5-4B"

# ═══════════════════════════════════════════════════════════════
# SVD 参数
# ═══════════════════════════════════════════════════════════════

SVD_MAX_EIG          = 128
SVD_ENERGY_THRESHOLD = 0.99
SVD_IMAGE_SIZE       = 1024
SVD_PATCH_SIZE       = 32

# ═══════════════════════════════════════════════════════════════
# Training Hyperparams
# ═══════════════════════════════════════════════════════════════

EPOCHS              = 3
BATCH_SIZE          = 2
GRAD_ACCUM          = 4
LR                  = 1e-4
WEIGHT_DECAY        = 0.01
NUM_WORKERS         = 4
SAVE_EVERY          = 880
MAX_LENGTH          = 256
MAX_GRAD_NORM       = 1.0
WARMUP_RATIO        = 0.03

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# Image helper — handles both PIL Image and raw bytes
# ═══════════════════════════════════════════════════════════════

def load_image_as_feature(image_input,
                          image_size: int = 1024) -> torch.Tensor:
    """Load single image → grayscale → (image_size, image_size) matrix.

    Accepts PIL.Image or raw bytes (JPEG/PNG).
    HF datasets auto-decodes image columns to PIL.Image.
    """
    if isinstance(image_input, Image.Image):
        img = image_input.convert("L")
    else:
        img = Image.open(io.BytesIO(image_input)).convert("L")
    img = img.resize((image_size, image_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr)
    return t - t.mean()


# ═══════════════════════════════════════════════════════════════
# Dataset — reads Parquet, returns raw image + text
# ═══════════════════════════════════════════════════════════════

class RawParquetDataset(Dataset):
    """__getitem__ → {"image": PIL.Image | bytes, "text": str}.

    Reads from a single .parquet file (path or pre-loaded DataFrame).
    Columns: image (JPEG/PNG bytes), image_caption (str), violations (str or None).
    """

    def __init__(self, dataset):
        if isinstance(dataset, str):
            self.dataset = pd.read_parquet(dataset, engine="pyarrow")
        elif isinstance(dataset, pd.DataFrame):
            self.dataset = dataset
        else:
            self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        row = self.dataset.iloc[idx]
        image = row["image"]  # bytes or PIL.Image (HF datasets auto-decodes)
        caption: str = row["image_caption"]
        violations = row["violations"]

        base = f"描述这张建筑工地图片：{caption}"
        text = base if violations is None else f"{base}\n隐患：{violations}"

        return {"image_bytes": image, "text": text}


# ═══════════════════════════════════════════════════════════════
# Collate Function — batched SVD + tokenize
# ═══════════════════════════════════════════════════════════════

class SVDCollector:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        return self.collate_fn(batch)

    def collate_fn(self, batch: list) -> dict:
        texts = [item["text"] for item in batch]
        images = [item["image_bytes"] for item in batch]
        svd_matrix = self._build_svd(images)
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "svd_matrix": svd_matrix,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _build_svd(self, images):
        """Batched SVD: (B, H, W) → (B, 2n, 1024)."""
        features_batch = torch.stack([
            load_image_as_feature(b, SVD_IMAGE_SIZE)
            for b in images
        ])
        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(features_batch, full_matrices=False)
        S_sq = S * S
        total_e = S_sq.sum(dim=1, keepdim=True)
        cumsum = torch.cumsum(S_sq, dim=1)
        n_per_img = (cumsum / total_e < SVD_ENERGY_THRESHOLD).sum(dim=1) + 1
        n_per_img = n_per_img.clamp(min=32, max=SVD_MAX_EIG)
        svd_list = []
        for k in range(len(images)):
            n = n_per_img[k].item()
            U_top = U[k, :, :n].T  # (n, 1024)
            Vh_top = Vh[k, :n, :]  # (n, 1024)
            mat = torch.cat([U_top, Vh_top], dim=0)  # (2n, 1024)
            svd_list.append(mat)
        return self._pad(svd_list)

    def _pad(self, mats: list[torch.Tensor]) -> torch.Tensor:
        """Pad to max 2n in batch."""
        full_dim = max(len(m) for m in mats)
        pad_mats = []
        for mat in mats:
            d = mat.size(0)
            if d < full_dim:
                pad = torch.zeros(full_dim - d, mat.size(1))
                mat = torch.cat([mat, pad], dim=0)
            pad_mats.append(mat)
        return torch.stack(pad_mats, dim=0)


# ══ Step 1: Model ══
config = SRQwenVLConfig(
    dino_dir=DINOV2_DIR,
    qwen_dir=QWEN_DIR,
    svd_max_eig=SVD_MAX_EIG,
    svd_energy_threshold=SVD_ENERGY_THRESHOLD,
)
model = SRQwenVLv10(config)
model.build_model()
model._enable_gradient_checkpointing()

# ══ Step 2: Dataset (parquet, raw bytes + text) ══
train_dataset = RawParquetDataset(PARQUET_TRAIN)
test_dataset  = RawParquetDataset(PARQUET_TEST)
print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

# ══ Step 3: Collator (SVD + tokenize happens here) ══
collator = SVDCollector(model.tokenizer)

# ══ Step 4: Optimizer ══
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ══ Step 5: Trainer ══
training_args = TrainingArguments(
    output_dir=os.path.join(OUTPUT_DIR, "checkpoints"),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=3,
    eval_strategy="epoch",
    bf16=True,
    dataloader_num_workers=NUM_WORKERS,
    report_to=[],
    max_grad_norm=MAX_GRAD_NORM,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)


class V10Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        inputs["svd_matrix"] = inputs["svd_matrix"].bfloat16()
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


trainer = V10Trainer(
    model=model, args=training_args,
    train_dataset=train_dataset, eval_dataset=test_dataset,
    data_collator=collator, optimizers=(optimizer, None),
)

print(f"\n{'='*60}")
print(f"Starting v10 training (v4 parquet SVD)...")
print(f"  SVD: max_eig={SVD_MAX_EIG}, energy={SVD_ENERGY_THRESHOLD}")
print(f"  Epochs: {EPOCHS}  Batch: {BATCH_SIZE}×{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}")
print(f"  LR: {LR}  Warmup: {WARMUP_RATIO*100:.0f}%")
print(f"  Output: {OUTPUT_DIR}")
print(f"{'='*60}\n")

trainer.train()

final_dir = os.path.join(OUTPUT_DIR, "final")
model.save_pretrained(final_dir)
print(f"\nFinal model saved to {final_dir}")
