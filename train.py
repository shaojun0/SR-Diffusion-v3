"""
SR-Qwen-VL v10: Training with HF Trainer + bitsandbytes 8-bit AdamW

SVD(1024×1024) → (2n,1024) matrix → DINO Encoder → MLP → Qwen → Text

v2: SVD 预处理移入 SVDDataset.__init__，职责分明。
    Dataset 自行加载图片 → 计算 SVD → 磁盘缓存 + 内存缓存。
"""
import os, sys, json, hashlib, time
import torch
import numpy as np
import bitsandbytes as bnb
from PIL import Image
from tqdm import tqdm
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from model import SRQwenVLConfig, SRQwenVLv10

# ═══════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════

IMAGE_DIR       = "/root/autodl-tmp/construction_site/images"
CAPTIONS_FILE   = "/root/autodl-tmp/construction_captions_zh.json"
SVD_CACHE_DIR   = "/root/autodl-tmp/sr_qwen_vl_v10_output/svd_cache"
DINOV2_DIR      = "/root/autodl-tmp/sr_dinov2/models/AI-ModelScope--dinov2-giant"
QWEN_DIR        = "/root/autodl-tmp/qwen3.5-4B"
OUTPUT_DIR      = "/root/autodl-tmp/sr_qwen_vl_v10_output"

# ═══════════════════════════════════════════════════════════════
# SVD 参数
# ═══════════════════════════════════════════════════════════════

SVD_MAX_EIG          = 128
SVD_ENERGY_THRESHOLD = 0.99
SVD_IMAGE_SIZE       = 1024
SVD_PATCH_SIZE       = 32
SVD_BATCH_SIZE       = 8

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
# Image → Patches helper (pure function, GPU-free)
# ═══════════════════════════════════════════════════════════════

def load_image_as_patches(img_path: str,
                          image_size: int = 1024,
                          patch_size: int = 32) -> torch.Tensor:
    """Load single image → grayscale → patchify → (n_patches, patch_dim)."""
    img = Image.open(img_path).convert("L")
    img = img.resize((image_size, image_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr)
    patches = t.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    n_patches = patches.shape[0] * patches.shape[1]
    patch_dim = patch_size * patch_size
    patches = patches.reshape(n_patches, patch_dim)
    patches = patches - patches.mean(dim=1, keepdim=True)
    return patches  # (n_patches, patch_dim)   → (1024, 1024) for default params


# ═══════════════════════════════════════════════════════════════
# Dataset — owns its SVD preprocessing
# ═══════════════════════════════════════════════════════════════

class SVDDataset(Dataset):
    """
    自包含 Dataset：
      __init__   → 加载图片、计算 SVD、存入磁盘缓存 + 内存
      __getitem__ → 直接返回预计算好的 SVD 矩阵 + tokenized caption

    模块级 _svd_memory_cache 利用 Linux fork 的 copy-on-write，
    避免 DataLoader worker 重复拷贝大张量。
    """

    # split → list of (2n, 1024) tensors, already padded to full_dim
    _svd_memory_cache: dict = {}

    def __init__(self,
                 split: str,
                 tokenizer,
                 image_dir: str = IMAGE_DIR,
                 captions_file: str = CAPTIONS_FILE,
                 svd_cache_dir: str = SVD_CACHE_DIR,
                 # ── SVD params ──
                 max_eig: int = SVD_MAX_EIG,
                 energy_threshold: float = SVD_ENERGY_THRESHOLD,
                 image_size: int = SVD_IMAGE_SIZE,
                 patch_size: int = SVD_PATCH_SIZE,
                 batch_size: int = SVD_BATCH_SIZE,
                 device: str = "cuda",
                 # ── text params ──
                 max_length: int = MAX_LENGTH,
                 ):
        self.split = split
        self.tokenizer = tokenizer
        self.max_eig = max_eig
        self.energy_threshold = energy_threshold
        self.image_size = image_size
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length
        self.full_dim = 2 * max_eig

        # ── Load captions ──
        with open(captions_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.data = data[split]

        # ── Build or load SVD ──
        if split not in SVDDataset._svd_memory_cache:
            self._build_svd(image_dir, svd_cache_dir)
        self.svd_matrices = SVDDataset._svd_memory_cache[split]

    # ── internal: disk cache key ──
    def _cache_hash(self) -> str:
        key = f"{self.max_eig}_{self.energy_threshold}_{self.image_size}_{self.patch_size}"
        return hashlib.md5(key.encode()).hexdigest()[:8]

    # ── internal: SVD computation ──
    def _build_svd(self, image_dir: str, svd_cache_dir: str):
        """Load images → batched SVD → disk cache + memory cache."""
        os.makedirs(svd_cache_dir, exist_ok=True)

        hash_file = os.path.join(svd_cache_dir, ".svd_config_hash")
        current_hash = self._cache_hash()

        # ── Try disk cache first ──
        cached_hash = None
        if os.path.exists(hash_file):
            with open(hash_file) as f:
                cached_hash = f.read().strip()

        if cached_hash == current_hash:
            print(f"  [Dataset:{self.split}] Loading SVD from disk cache ...")
            svd_list = []
            for item in tqdm(self.data, desc=f"  Load SVD ({self.split})"):
                idx = item["index"]
                pt_path = os.path.join(svd_cache_dir, f"{idx}.pt")
                mat = torch.load(pt_path, map_location="cpu", weights_only=True)
                svd_list.append(self._pad(mat))
            SVDDataset._svd_memory_cache[self.split] = svd_list
            print(f"  [Dataset:{self.split}] Loaded {len(svd_list)} SVD matrices to memory")
            return

        # ── Compute from scratch ──
        if cached_hash:
            print(f"  [Dataset:{self.split}] SVD config changed ({cached_hash} → {current_hash})")
        else:
            print(f"  [Dataset:{self.split}] No SVD cache, computing (hash={current_hash})")

        # 1. Collect image paths
        index_to_path = {}
        for item in self.data:
            idx = item["index"]
            path = os.path.join(image_dir, f"{idx}.jpg")
            if not os.path.exists(path):
                for ext in (".png", ".jpeg", ".JPG", ".JPEG"):
                    alt = os.path.join(image_dir, f"{idx}{ext}")
                    if os.path.exists(alt):
                        path = alt
                        break
            if os.path.exists(path):
                index_to_path[idx] = path
            else:
                print(f"  [WARN] Image not found: {idx}")

        # 2. CPU: load & patchify all images
        print(f"  [Dataset:{self.split}] Loading {len(index_to_path)} images & patchifying ...")
        patches_list = []
        indices = []
        t0 = time.time()
        for idx, path in tqdm(index_to_path.items(), desc=f"  Patchify ({self.split})"):
            try:
                patches = load_image_as_patches(path, self.image_size, self.patch_size)
                patches_list.append(patches)
                indices.append(idx)
            except Exception as e:
                print(f"  [SKIP] {idx}: {e}")

        if not patches_list:
            raise RuntimeError(f"No valid images for split '{self.split}'!")

        all_patches = torch.stack(patches_list)  # (N, 1024, 1024)
        n_images = all_patches.size(0)
        print(f"  [Dataset:{self.split}] Patches: {all_patches.shape}, "
              f"~{all_patches.numel() * 4 / 1e9:.1f}GB, "
              f"patchify time: {time.time() - t0:.1f}s")

        # 3. GPU: batched SVD
        svd_list = [None] * n_images  # pre-allocate by batch position
        n_batches = (n_images + self.batch_size - 1) // self.batch_size
        print(f"  [Dataset:{self.split}] Running batched SVD "
              f"(batch={self.batch_size}, {n_batches} batches) ...")
        t0 = time.time()

        for b in range(n_batches):
            b_start = b * self.batch_size
            b_end = min(b_start + self.batch_size, n_images)
            b_actual = b_end - b_start

            M = all_patches[b_start:b_end].to(self.device)  # (B, 1024, 1024)

            with torch.no_grad():
                U, S, Vh = torch.linalg.svd(M, full_matrices=False)

            # Energy truncation
            S_sq = S * S
            total_e = S_sq.sum(dim=1, keepdim=True)
            cumsum = torch.cumsum(S_sq, dim=1)
            n_per_img = (cumsum / total_e < self.energy_threshold).sum(dim=1) + 1
            n_per_img = n_per_img.clamp(min=32, max=self.max_eig)

            for k in range(b_actual):
                n = n_per_img[k].item()
                idx = indices[b_start + k]
                U_top = U[k, :, :n].T           # (n, 1024)
                Vh_top = Vh[k, :n, :]           # (n, 1024)
                mat = torch.cat([U_top, Vh_top], dim=0)  # (2n, 1024)

                # Save to disk
                torch.save(mat.cpu(), os.path.join(svd_cache_dir, f"{idx}.pt"))

                # Store in memory (padded)
                svd_list[b_start + k] = self._pad(mat.cpu())

            if (b + 1) % 20 == 0:
                elapsed = time.time() - t0
                done = b_end
                rate = done / elapsed
                eta = (n_images - done) / rate / 60 if rate > 0 else 0
                print(f"    [{b_end}/{n_images}] {rate:.1f} img/s, ETA: {eta:.1f}min")

        elapsed = time.time() - t0
        print(f"  [Dataset:{self.split}] SVD done: {n_images} images in "
              f"{elapsed/60:.1f}min ({n_images/elapsed:.1f} img/s)")

        # Write hash file
        with open(hash_file, "w") as f:
            f.write(current_hash)

        SVDDataset._svd_memory_cache[self.split] = svd_list
        print(f"  [Dataset:{self.split}] {len(svd_list)} matrices in memory "
              f"(~{sum(m.numel() for m in svd_list) * 4 / 1e9:.1f}GB)")

    def _pad(self, mat: torch.Tensor) -> torch.Tensor:
        """Pad or truncate to full_dim."""
        d = mat.size(0)
        if d < self.full_dim:
            pad = torch.zeros(self.full_dim - d, mat.size(1))
            return torch.cat([mat, pad], dim=0)
        elif d > self.full_dim:
            return mat[:self.full_dim]
        return mat

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Pre-computed SVD matrix (already padded)
        svd_matrix = self.svd_matrices[idx]

        # Tokenize
        text = f"描述这张建筑工地图片：{item['zh']}"
        encoding = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "svd_matrix": svd_matrix,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch):
    return {
        "svd_matrix": torch.stack([item["svd_matrix"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("SR-Qwen-VL v10: SVD → DINO Encoder → MLP → Qwen")
print("=" * 60)

# ══ Step 1: Model ══
config = SRQwenVLConfig(
    dino_dir=DINOV2_DIR, qwen_dir=QWEN_DIR,
    svd_max_eig=SVD_MAX_EIG,
    svd_energy_threshold=SVD_ENERGY_THRESHOLD,
)
model = SRQwenVLv10(config)
model.build_model()
model._enable_gradient_checkpointing()

# ══ Step 2: Data — Dataset handles its own SVD ══
train_dataset = SVDDataset(
    "train", model.tokenizer,
    image_dir=IMAGE_DIR, captions_file=CAPTIONS_FILE,
    svd_cache_dir=SVD_CACHE_DIR,
    max_eig=SVD_MAX_EIG, energy_threshold=SVD_ENERGY_THRESHOLD,
    image_size=SVD_IMAGE_SIZE, patch_size=SVD_PATCH_SIZE,
    batch_size=SVD_BATCH_SIZE, device="cuda",
)
test_dataset = SVDDataset(
    "test", model.tokenizer,
    image_dir=IMAGE_DIR, captions_file=CAPTIONS_FILE,
    svd_cache_dir=SVD_CACHE_DIR,
    max_eig=SVD_MAX_EIG, energy_threshold=SVD_ENERGY_THRESHOLD,
    image_size=SVD_IMAGE_SIZE, patch_size=SVD_PATCH_SIZE,
    batch_size=SVD_BATCH_SIZE, device="cuda",
)
print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

# ══ Step 3: Optimizer ══
optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ══ Step 4: Trainer ══
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
    data_collator=collate_fn, optimizers=(optimizer, None),
)

print(f"\n{'='*60}")
print(f"Starting v10 training ...")
print(f"  SVD: max_eig={SVD_MAX_EIG}, energy={SVD_ENERGY_THRESHOLD}")
print(f"  Epochs: {EPOCHS}  Batch: {BATCH_SIZE}×{GRAD_ACCUM}={BATCH_SIZE*GRAD_ACCUM}")
print(f"  LR: {LR}  Warmup: {WARMUP_RATIO*100:.0f}%")
print(f"  Output: {OUTPUT_DIR}")
print(f"{'='*60}\n")

trainer.train()

final_dir = os.path.join(OUTPUT_DIR, "final")
model.save_pretrained(final_dir)
print(f"\nFinal model saved to {final_dir}")
