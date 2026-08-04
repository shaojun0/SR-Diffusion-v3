"""
SR-Qwen-VL v10: Training with HF Trainer + bitsandbytes 8-bit AdamW

SVD(1024×1024) → (2n,1024) matrix → DINO Encoder → MLP → Qwen → Text

v4: Dataset 从 Parquet 加载（image_bytes + text），
    SVD + tokenize 全部在 collate_fn 里完成，batched SVD + LRU 缓存。
"""
import os, sys, io, hashlib
from functools import lru_cache
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

FULL_DIM = 2 * SVD_MAX_EIG  # 256


# ═══════════════════════════════════════════════════════════════
# Image Bytes → Patches helper (CPU, pure function)
# ═══════════════════════════════════════════════════════════════

def bytes_to_patches(
    image_bytes: bytes,
    image_size: int = SVD_IMAGE_SIZE,
    patch_size: int = SVD_PATCH_SIZE,
) -> torch.Tensor:
    """Decode JPEG/PNG bytes → grayscale → patchify → (n_patches, patch_dim).

    For default params: (1024, 1024) — 1024 patches, 1024-dimensional.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = img.resize((image_size, image_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr)

    patches = t.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    n_patches = patches.shape[0] * patches.shape[1]
    patch_dim = patch_size * patch_size
    patches = patches.reshape(n_patches, patch_dim)
    patches = patches - patches.mean(dim=1, keepdim=True)
    return patches  # (1024, 1024)


# ═══════════════════════════════════════════════════════════════
# SVD LRU Cache — per-image, keyed by content hash (md5)
# ═══════════════════════════════════════════════════════════════

# Shared dict — populated by batched SVD in collate_fn.
# Keys: md5_hex of image_bytes  →  Values: (U_cpu, S_cpu, Vh_cpu)
_svd_cache_dict: dict = {}


@lru_cache(maxsize=4096)
def _svd_single_lru(content_hash: str) -> tuple:
    """Get SVD result by content hash.  Shared-dict hit → instant.

    Returns (U, S, Vh) as CPU tensors: U(1024,1024), S(1024,), Vh(1024,1024).

    The shared _svd_cache_dict is populated by batched collate_fn;
    this function is the LRU-cached retrieval path.
    On a miss (should not happen under normal collate_fn flow) it raises.
    """
    if content_hash in _svd_cache_dict:
        return _svd_cache_dict[content_hash]

    raise KeyError(
        f"SVD not computed for hash {content_hash}. "
        f"This should not happen — collate_fn always batches uncached images first."
    )


# ═══════════════════════════════════════════════════════════════
# Dataset — reads Parquet, returns raw image_bytes + text
# ═══════════════════════════════════════════════════════════════

class RawParquetDataset(Dataset):
    """__getitem__ → { "image_bytes": bytes, "text": str }.

    Reads from a single .parquet file with columns:
      image        — bytes (JPEG/PNG)
      zh_caption   — str (中文描述)
      violations   — str or None/NaN

    No SVD, no tokenize — all heavy lifting deferred to collate_fn.
    """

    def __init__(self, parquet_path: str):
        self.parquet_path = parquet_path
        self.df = pd.read_parquet(parquet_path)

        # ── Pre-extract strings from DataFrame for speed ──
        self._zh_captions = self.df["zh_caption"].tolist()

        raw_violations = self.df["violations"].tolist()
        self._violations: list = []
        for v in raw_violations:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                self._violations.append(None)
            else:
                self._violations.append(str(v))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        image_bytes: bytes = row["image"]
        caption: str = self._zh_captions[idx]
        violations = self._violations[idx]

        base = f"描述这张建筑工地图片：{caption}"
        text = base if violations is None else f"{base}\n隐患：{violations}"

        return {"image_bytes": image_bytes, "text": text}


# ═══════════════════════════════════════════════════════════════
# Collate Function Factory — owns tokenizer, does SVD + tokenize
# ═══════════════════════════════════════════════════════════════

def make_collator(tokenizer):
    """Returns a collate_fn that does batched SVD + batch tokenize."""

    def collate_fn(batch: list) -> dict:
        texts = [item["text"] for item in batch]
        image_bytes_list: list[bytes] = [item["image_bytes"] for item in batch]

        # ── 1. Compute content hashes & identify uncached images ──
        content_hashes = [hashlib.md5(b).hexdigest() for b in image_bytes_list]
        uncached_hashes = [h for h in content_hashes if h not in _svd_cache_dict]

        # ── 2. Batched SVD for uncached images on GPU ──
        if uncached_hashes:
            # Build mapping: hash → image_bytes for uncached only
            hash_to_bytes = {
                content_hashes[i]: image_bytes_list[i]
                for i, h in enumerate(content_hashes)
                if h in uncached_hashes
            }

            # Patchify all uncached images → stack → GPU
            uncached_bytes = [hash_to_bytes[h] for h in uncached_hashes]
            patches_batch = torch.stack([
                bytes_to_patches(b, SVD_IMAGE_SIZE, SVD_PATCH_SIZE)
                for b in uncached_bytes
            ]).cuda()  # (U, 1024, 1024)

            with torch.no_grad():
                U, S, Vh = torch.linalg.svd(patches_batch, full_matrices=False)
                # U  (U, 1024, 1024)   S  (U, 1024)   Vh (U, 1024, 1024)

            # Energy-based truncation  (per-batch vector — no .item())
            S_sq = S * S
            total_e = S_sq.sum(dim=1, keepdim=True)
            cumsum = torch.cumsum(S_sq, dim=1)
            n_per_img = (cumsum / total_e < SVD_ENERGY_THRESHOLD).sum(dim=1) + 1
            n_per_img = n_per_img.clamp(min=32, max=SVD_MAX_EIG).cpu()

            # Store full (U, S, Vh) in shared dict
            for k, ch in enumerate(uncached_hashes):
                _svd_cache_dict[ch] = (
                    U[k].cpu(),   # (1024, 1024)
                    S[k].cpu(),   # (1024,)
                    Vh[k].cpu(),  # (1024, 1024)
                )

        # ── 3. Retrieve SVD results via _svd_single_lru (LRU-cached) ──
        svd_results = [_svd_single_lru(h) for h in content_hashes]

        # ── 4. Build padded SVD matrices from (U, S, Vh) ──
        svd_matrices: list[torch.Tensor] = []

        for U_img, S_img, Vh_img in svd_results:  # all on CPU
            S_sq = S_img * S_img
            total_e = S_sq.sum()
            cumsum = torch.cumsum(S_sq, dim=0)
            n = (cumsum / total_e < SVD_ENERGY_THRESHOLD).sum() + 1
            n = n.clamp(32, SVD_MAX_EIG)         # 0-dim tensor

            U_top = U_img[:, :n].T               # (n, 1024)  — U^T
            Vh_top = Vh_img[:n, :]               # (n, 1024)  — Vh rows
            mat = torch.cat([U_top, Vh_top], dim=0)  # (2n, 1024)

            # Pad to FULL_DIM = 2 * SVD_MAX_EIG
            d = mat.size(0)
            if d < FULL_DIM:
                padding = torch.zeros(FULL_DIM - d, mat.size(1))
                mat = torch.cat([mat, padding], dim=0)

            svd_matrices.append(mat)

        # ── 5. Tokenize entire batch (batched, not per-sample) ──
        encodings = tokenizer(
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
            "svd_matrix":    torch.stack(svd_matrices),
            "input_ids":     input_ids,
            "attention_mask": attention_mask,
            "labels":        labels,
        }

    return collate_fn


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("SR-Qwen-VL v10: SVD → DINO Encoder → MLP → Qwen")
print("  v4: Parquet dataset, SVD + tokenize in collate_fn, LRU cache")
print("=" * 60)

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
collator = make_collator(model.tokenizer)

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
    save_steps=SAVE_EVERY,
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
