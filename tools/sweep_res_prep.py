"""sweep_res_prep.py — 把 DIV2K 预 resize 成 S×S 的 .npy（uint8），训练时 mmap 直读

动机（实测）：DIV2K 原图是 2040×1404 PNG，每个 epoch 重新解码 + BICUBIC resize
是纯 CPU 瓶颈 —— 28px 训练时 GPU util 只有 6%。预处理好之后 dataloader 只做
一次 index + 归一化，训练吞吐可提高数倍，才有本钱把大分辨率的预算加上去。

用法:
  python sweep_res_prep.py --size 224            # train+val
  python sweep_res_prep.py --sizes 28,56,112     # 批量
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image


def build(files, size, out):
    arr = np.zeros((len(files), size, size, 3), np.uint8)
    for i, fp in enumerate(files):
        im = Image.open(fp).convert("RGB").resize((size, size), Image.BICUBIC)
        arr[i] = np.asarray(im, np.uint8)
    np.save(out, arr)
    return arr.shape, os.path.getsize(out) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int)
    ap.add_argument("--sizes", default="")
    ap.add_argument("--root", default="/root/autodl-tmp/srres/data")
    args = ap.parse_args()
    sizes = [int(v) for v in args.sizes.split(",") if v] or [args.size]

    tr_dir = os.path.join(args.root, "DIV2K_train_HR")
    va_dir = os.path.join(args.root, "DIV2K_valid_HR")
    tr = sorted(os.path.join(tr_dir, f) for f in os.listdir(tr_dir) if f.endswith(".png"))
    va = sorted(os.path.join(va_dir, f) for f in os.listdir(va_dir) if f.endswith(".png"))
    print(f"[prep] train={len(tr)} val={len(va)}")
    for s in sizes:
        ot = os.path.join(args.root, f"S{s}_train.npy")
        ov = os.path.join(args.root, f"S{s}_val.npy")
        if os.path.exists(ot) and os.path.exists(ov):
            print(f"[prep] S={s} 已存在，跳过")
            continue
        sh, mb = build(tr, s, ot)
        print(f"[prep] S={s} train {sh} {mb:.1f}MB", flush=True)
        sh, mb = build(va, s, ov)
        print(f"[prep] S={s} val   {sh} {mb:.1f}MB", flush=True)
    print("PREP_DONE")


if __name__ == "__main__":
    main()
