"""
Precompute SVD (2n, 1024) matrices — batched GPU with torch.linalg.svd.

Process:
  1. Load B images → resize 1024×1024 → grayscale → patchify → stack (B, 1024, 1024)
  2. Batched SVD: torch.linalg.svd(M, full_matrices=False) on GPU
  3. Per-image energy truncation (99%) → n components
  4. Build (2n, 1024) = [U[:,:n]^T; Vh[:n,:]]
  5. Save individual .pt files (float32)
"""
import torch
import os
import json
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse


def load_image_as_patches(
    img_path: str,
    image_size: int = 1024,
    patch_size: int = 32,
) -> torch.Tensor:
    """Load single image → grayscale → patchify → (n_patches, patch_dim)."""
    img = Image.open(img_path).convert("L")
    img = img.resize((image_size, image_size), Image.LANCZOS)

    # Fast numpy conversion
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr)

    # Unfold into patches
    patches = t.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    n_patches = patches.shape[0] * patches.shape[1]
    patch_dim = patch_size * patch_size
    patches = patches.reshape(n_patches, patch_dim)
    patches = patches - patches.mean(dim=1, keepdim=True)
    return patches  # (1024, 1024)


def precompute_batched(
    image_dir: str,
    captions_file: str,
    output_dir: str,
    max_eig: int = 128,
    energy_threshold: float = 0.99,
    image_size: int = 1024,
    patch_size: int = 32,
    batch_size: int = 8,
    device: str = "cuda",
    splits: list = None,
):
    """Batched GPU SVD: load B images, stack → torch.linalg.svd batch → save."""
    os.makedirs(output_dir, exist_ok=True)

    if splits is None:
        splits = ["train", "test"]

    with open(captions_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Collect all indices with valid image paths
    all_indices = sorted(set(
        item["index"] for split in splits if split in data
        for item in data[split]
    ))

    # Pre-resolve image paths
    index_to_path = {}
    for idx in all_indices:
        path = os.path.join(image_dir, f"{idx}.jpg")
        if not os.path.exists(path):
            for ext in (".png", ".jpeg", ".JPG", ".JPEG"):
                alt = os.path.join(image_dir, f"{idx}{ext}")
                if os.path.exists(alt):
                    path = alt
                    break
        if os.path.exists(path):
            index_to_path[idx] = path

    # Filter to unprocessed
    pending = []
    for idx, path in index_to_path.items():
        if not os.path.exists(os.path.join(output_dir, f"{idx}.pt")):
            pending.append((idx, path))

    total = len(pending)
    print(f"Total: {len(all_indices)}, Already done: {len(all_indices) - total}, Pending: {total}")
    if total == 0:
        print("All done!")
        return

    stats = {"total": total, "success": 0, "failed": 0, "n_dist": {}}
    start_time = time.time()

    # ── 1. CPU: load & patchify all pending images ──
    print("Loading images and computing patches on CPU...")
    all_patches = []
    valid_indices = []
    for idx, path in tqdm(pending, desc="Patchify"):
        try:
            patches = load_image_as_patches(path, image_size, patch_size)
            all_patches.append(patches)
            valid_indices.append(idx)
        except Exception as e:
            print(f"  [SKIP] {idx}: {e}")
            stats["failed"] += 1

    if not all_patches:
        print("No valid images to process!")
        return

    # Stack all into one big tensor (move to GPU in batches)
    all_patches = torch.stack(all_patches)  # (N, 1024, 1024)
    n_images = all_patches.size(0)
    print(f"Patches tensor: {all_patches.shape}, ~{all_patches.numel() * 4 / 1e9:.1f}GB")

    # ── 2. GPU: batched SVD ──
    n_batches = (n_images + batch_size - 1) // batch_size
    print(f"Running batched SVD on {device} (batch_size={batch_size}, {n_batches} batches)...")

    for b in range(n_batches):
        b_start = b * batch_size
        b_end = min(b_start + batch_size, n_images)
        batch_indices = valid_indices[b_start:b_end]
        b_actual = b_end - b_start

        # Move batch to GPU
        M = all_patches[b_start:b_end].to(device)  # (B, 1024, 1024)

        with torch.no_grad():
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            # U: (B, 1024, 1024), S: (B, 1024), Vh: (B, 1024, 1024)  [Vh = V^T]

        # ── Per-image truncation & save ──
        S_sq = S * S
        total_e = S_sq.sum(dim=1, keepdim=True)  # (B, 1)
        cumsum = torch.cumsum(S_sq, dim=1)       # (B, 1024)
        n_per_img = (cumsum / total_e < energy_threshold).sum(dim=1) + 1  # (B,)
        n_per_img = n_per_img.clamp(min=32, max=max_eig)

        for k in range(b_actual):
            n = n_per_img[k].item()
            idx = batch_indices[k]

            # Build (2n, 1024): [U[:,:n]^T; Vh[:n,:]]
            U_top = U[k, :, :n].T                   # (n, 1024) — note: U has vectors in columns
            Vh_top = Vh[k, :n, :]                    # (n, 1024) — Vh rows = V cols
            svd_matrix = torch.cat([U_top, Vh_top], dim=0)  # (2n, 1024)

            torch.save(svd_matrix.cpu(), os.path.join(output_dir, f"{idx}.pt"))
            stats["success"] += 1
            stats["n_dist"].setdefault(str(n), 0)
            stats["n_dist"][str(n)] += 1

        if (b + 1) % 10 == 0:
            elapsed = time.time() - start_time
            done = stats["success"]
            rate = done / elapsed
            remaining = n_images - b_end
            eta = remaining / rate / 60 if rate > 0 else 0
            print(f"  [{b_end}/{n_images}] {rate:.1f} img/s, ETA: {eta:.1f}min")

    elapsed = time.time() - start_time
    print(f"\nDone! {stats['success']} images in {elapsed/60:.1f}min "
          f"({stats['success']/elapsed:.1f} img/s)")
    print(f"n distribution: {stats['n_dist']}")

    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="/root/autodl-tmp/construction_site/images")
    parser.add_argument("--captions_file", default="/root/autodl-tmp/construction_captions_zh.json")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/sr_qwen_vl_v10_output/svd_cache")
    parser.add_argument("--max_eig", type=int, default=128)
    parser.add_argument("--energy_threshold", type=float, default=0.99)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    args = parser.parse_args()

    precompute_batched(**vars(args))
