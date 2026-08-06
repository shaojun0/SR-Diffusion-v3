"""
SR-Qwen-VL v11: Inference Test Script

用法:
    python test.py                          # 加载最终模型，对数据集第一条做推理
    python test.py --prompt "描述..."        # 自定义 prompt
"""

import io
import sys
import argparse
import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from transformers import AutoTokenizer

from model import SRQwenVLv10

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

DEFAULT_IMAGE_SIZE    = 1024
SVD_ENERGY_THRESHOLD  = 0.99
SVD_MAX_EIG           = 128
MODEL_PATH            = "output/sr_qwen_vl_v10_output/final/"
QWEN_DIR              = "models/qwen3.5-4B"
DATA_PATH             = "translated_dataset_with_new_fields"

# ═══════════════════════════════════════════════════════════════
# Image → SVD Matrix
# ═══════════════════════════════════════════════════════════════

def load_image_as_feature(image, image_size: int = DEFAULT_IMAGE_SIZE) -> torch.Tensor:
    """单张图 → 灰度 → resize → (image_size, image_size) 矩阵."""
    if isinstance(image, Image.Image):
        img = image.convert("L")
    else:
        img = Image.open(io.BytesIO(image)).convert("L")
    img = img.resize((image_size, image_size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr)
    return t - t.mean()


def build_svd(image,
              image_size: int = DEFAULT_IMAGE_SIZE,
              energy_threshold: float = SVD_ENERGY_THRESHOLD,
              max_eig: int = SVD_MAX_EIG) -> torch.Tensor:
    """单张图 → SVD → (2n, 1024) eigen-matrix."""
    feature = load_image_as_feature(image, image_size).unsqueeze(0)  # (1, H, W)

    with torch.no_grad():
        U, S, Vh = torch.linalg.svd(feature, full_matrices=False)

    S_sq = S * S
    total_e = S_sq.sum(dim=1, keepdim=True)
    cumsum = torch.cumsum(S_sq, dim=1)
    n = (cumsum / total_e < energy_threshold).sum(dim=1) + 1
    n = n.clamp(min=32, max=max_eig).item()

    U_top = U[0, :, :n].T       # (n, H)
    Vh_top = Vh[0, :n, :]       # (n, W)
    mat = torch.cat([U_top, Vh_top], dim=0)    # (2n, H)

    return mat  # (2n, 1024)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SR-Qwen-VL v10 inference")
    parser.add_argument("--model-path", default=MODEL_PATH, help="模型路径")
    parser.add_argument("--qwen-dir", default=QWEN_DIR, help="Qwen/tokenizer 目录")
    parser.add_argument("--data-path", default=DATA_PATH, help="数据集路径")
    parser.add_argument("--prompt", default="描述这张建筑工地图片：", help="推理 prompt")
    parser.add_argument("--idx", type=int, default=0, help="数据集索引")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}")

    # ── 加载数据集 ──
    print(f"Loading dataset: {args.data_path}")
    dataset = load_dataset(args.data_path, cache_dir="./cache")
    splits = list(dataset.keys())
    train_split = "train" if "train" in splits else splits[0]
    print(f"  Splits: {splits}, using {train_split}[{args.idx}]")
    sample = dataset[train_split][args.idx]
    image = sample["image"]

    # ── 加载模型 ──
    print(f"Loading model: {args.model_path}")
    model = SRQwenVLv10.from_pretrained(
        args.model_path,
        device_map=device if device == "cuda" else None,
    )

    # ── 加载 tokenizer（独立，不绑定模型）──
    print(f"Loading tokenizer: {args.qwen_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  vocab_size = {tokenizer.vocab_size}")

    # ── SVD ──
    print("Computing SVD...")
    svd_mat = build_svd(image)
    print(f"  svd_mat.shape = {svd_mat.shape}  (2n={svd_mat.shape[0]})")

    # ── Generate ──
    print(f"\nPrompt: {args.prompt}")
    svd_input = svd_mat.unsqueeze(0).to(device).bfloat16()  # (1, 2n, 1024)

    result = model.generate(
        svd_matrix=svd_input,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"\n{'='*60}")
    print(f"Generated (zh): {result}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
