#!/usr/bin/env python3
"""plot_rd_compare.py — 把所有模型的 RD 曲线与传统 codec 画在同一张 bpp–PSNR 图上

用法:
    python tools/plot_rd_compare.py --out rd_compare.png \
        --ours  "448x252 BPTT=doc/2026-09-22/data/../..." \
        --base  ".../baseline_classic.json"
更简单的做法是用内置清单（本仓库当前实验）:
    python tools/plot_rd_compare.py --preset --out rd_compare.png

⚠️ 口径提醒: bpp 是「总比特/像素」, 所以不同分辨率可以同轴比较; 但小画布会被
文件头开销污染（224×126 上 JPEG 的最低操作点高达 0.36 bpp, 448×252 只有 0.21）。
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PRESET_OURS = [
    ("ours 224x126 (K=144, d4, BPTT, fp32)", "/root/autodl-tmp/cot_l1/eval_224_d4/test_224_d4_bptt.json"),
    ("ours 448x252 (K=576, d2, BPTT, fp16+8bit)",
     "/root/autodl-tmp/cot_l1/eval_448_d2/test_448_d2_bptt_fp16_8bit.json"),
    ("ours 896x504 (K=168, d4, BPTT, fp16+8bit)",
     "/root/autodl-tmp/cot_l1/eval_896_d4/test_896_d4_bptt_slice012.json"),
]
PRESET_BASE = [
    ("224x126", "/root/autodl-tmp/cot_l1/eval_224_d4/baseline_classic_224.json"),
    ("448x252", "/root/autodl-tmp/cot_l1/eval_448_d2/baseline_classic_448.json"),
    ("896x504", "/root/autodl-tmp/cot_l1/eval_896_d4/baseline_classic_896.json"),
]


def load_ours(path):
    with open(path) as f:
        d = json.load(f)
    b = np.array(d.get("step_bpp_beta1") or [], float)
    p = np.array(d.get("step_psnr") or [], float)
    if len(b) == 0:      # 老 schema: 自己折算（β=1）
        px = int(d.get("bpp_px") or 448 * 252)
        dim = 1024
        t = np.array(d["decoder_steps"], float)
        b = (t + 1) * dim / px
        p = np.array(d["step_psnr"], float)
    return b, p


def load_base(path):
    with open(path) as f:
        d = json.load(f)
    out = {}
    for r in d["rows"]:
        out.setdefault(r["codec"], []).append((r["bpp"], r["psnr"]))
    return {k: (np.array([x[0] for x in v]), np.array([x[1] for x in v]))
            for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", action="store_true")
    ap.add_argument("--ours", nargs="*", default=[], help="name=path")
    ap.add_argument("--base", nargs="*", default=[], help="name=path")
    a = ap.parse_args()

    ours = PRESET_OURS if a.preset else [tuple(s.split("=", 1)) for s in a.ours]
    base = PRESET_BASE if a.preset else [tuple(s.split("=", 1)) for s in a.base]

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    colors = ["tab:red", "tab:purple", "tab:orange", "tab:brown"]
    for i, (name, path) in enumerate(ours):
        if not os.path.isfile(path):
            print(f"[skip] {name}: {path} 不存在")
            continue
        b, p = load_ours(path)
        ax.plot(b, p, "-o", ms=5, color=colors[i % len(colors)], label=name, zorder=5)

    styles = [("JPEG", "-s"), ("WebP", "-^"), ("AVIF", "-d")]
    base_colors = ["tab:gray", "tab:olive", "tab:cyan", "tab:pink"]
    for j, (name, path) in enumerate(base):
        if not os.path.isfile(path):
            print(f"[skip] {name}: {path} 不存在")
            continue
        ax.plot([], [], " ", label=f"—— classic @{name} ——")
        for codec, (b, p) in load_base(path).items():
            mk = dict(styles).get(codec, "-x")
            ax.plot(b, p, mk, ms=4, lw=1.2, alpha=.85,
                    color=base_colors[j % len(base_colors)],
                    linestyle="-" if j % 2 == 0 else "--",
                    label=f"{codec} @{name}")

    ax.set_xscale("log")
    ax.set_xlabel("bpp (bits/pixel, β=1 estimated for ours)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Rate–Distortion: ours vs classic codecs (construction_site test)")
    ax.grid(alpha=.3, which="both")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(f"[plot] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
