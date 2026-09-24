"""DIV2K (2K 数据集) -> 定分辨率 uint8 npy，供 12 组实验共用。

DIV2K_train_HR: 800 张  -> train
DIV2K_valid_HR: 100 张  -> val

预处理：短边缩放到 S，再中心裁剪 S x S（保持长宽比，不拉伸）。
输出：<out>/div2k_train_{S}.npy  (N,S,S,3) uint8
      <out>/div2k_val_{S}.npy    (N,S,S,3) uint8
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image


def collect(root: str) -> list[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = [os.path.join(root, f) for f in sorted(os.listdir(root)) if f.lower().endswith(exts)]
    return files


def to_square(path: str, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    nw, nh = max(size, round(w * scale)), max(size, round(h * scale))
    img = img.resize((nw, nh), Image.BICUBIC)
    left, top = (nw - size) // 2, (nh - size) // 2
    img = img.crop((left, top, left + size, top + size))
    return np.asarray(img, dtype=np.uint8)


def build(files: list[str], size: int, out_path: str) -> None:
    arr = np.zeros((len(files), size, size, 3), dtype=np.uint8)
    for i, f in enumerate(files):
        arr[i] = to_square(f, size)
        if (i + 1) % 100 == 0 or i + 1 == len(files):
            print(f"  [{size}] {i+1}/{len(files)}", flush=True)
    np.save(out_path, arr)
    print(f"  saved {out_path} shape={arr.shape} {arr.nbytes/1e6:.1f} MB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr_dir", default="/root/autodl-tmp/rrnet/data/DIV2K_HR")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/rrnet/data")
    ap.add_argument("--sizes", type=int, nargs="+", default=[224, 448])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_files = collect(os.path.join(args.hr_dir, "DIV2K_train_HR"))
    val_files = collect(os.path.join(args.hr_dir, "DIV2K_valid_HR"))
    print(f"train={len(train_files)} val={len(val_files)}", flush=True)
    assert train_files and val_files, f"no images under {args.hr_dir}"

    for s in args.sizes:
        build(train_files, s, os.path.join(args.out_dir, f"div2k_train_{s}.npy"))
        build(val_files, s, os.path.join(args.out_dir, f"div2k_val_{s}.npy"))


if __name__ == "__main__":
    main()
