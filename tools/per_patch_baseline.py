#!/usr/bin/env python3
"""per_patch_baseline.py — 「每块预测自己的 DC」平凡解基线（0-255 空间 L1/PSNR）

判据用途：高分辨率臂若 best PSNR **低于**本基线，说明模型连「每块涂平均色」都没学会
（见 doc/2026-09-23/ANALYSIS_res_sweep_rootcause.md §1）。

口径与 `sweep_res_train.py::evaluate` 一致：patch = 14×14×3（patch_px=588）、
行主序平方块、0-255 空间、MSE 对 全图所有像素/通道 取平均、PSNR=10log10(255²/MSE)。

用法:
    python tools/per_patch_baseline.py --data_root /root/autodl-tmp/srres/data \
        --sizes 28,56,112,224,336 --out <json>
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

PATCH = 14


def baseline(size, npy):
    a = np.load(npy)                                   # (M,S,S,3) uint8
    assert a.shape[1] == size, (a.shape, size)
    p = size // PATCH
    assert p * PATCH == size
    b = a.reshape(a.shape[0], p, PATCH, p, PATCH, 3).astype(np.float64)
    m = b.mean(axis=(2, 4), keepdims=True)             # 每块 DC
    d = b - m
    mse = float((d ** 2).mean())
    return {"size": size, "n_val": int(a.shape[0]), "patch": PATCH,
            "l1_255": float(np.abs(d).mean()), "mse": mse,
            "psnr": 10.0 * math.log10(255.0 ** 2 / mse)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/root/autodl-tmp/srres/data")
    ap.add_argument("--sizes", default="28,56,112,224,336")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    res = {}
    for s in [int(v) for v in args.sizes.split(",") if v]:
        npy = os.path.join(args.data_root, f"S{s}_val.npy")
        if not os.path.exists(npy):
            print(f"[skip] {npy} 不存在")
            continue
        r = baseline(s, npy)
        res[str(s)] = r
        print(f"S={s:>4} val={r['n_val']}  per-patch DC 基线: "
              f"L1={r['l1_255']:.2f}  PSNR={r['psnr']:.2f} dB", flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
