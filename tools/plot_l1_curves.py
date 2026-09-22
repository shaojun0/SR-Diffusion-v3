#!/usr/bin/env python3
"""plot_l1_curves.py — 把若干臂的「逐 step 像素 L1」画在一起（容忍老 schema）

动机（2026-09-23）: 448×252 / K=576 / 24 步 / d2 / BPTT 这个**完全相同的配方**有三次运行，
只差精度/优化器，需要一张图直接看「精修有没有发生」:

  · `doc/2026-09-16/data/bptt24_bptt_infer_test.json`   fp32 + adamw_torch_fused（历史 BPTT）
  · `doc/2026-09-16/data/bptt24_detach_infer_test.json` fp32 + adamw_torch_fused（历史 detach 对照）
  · 本轮 `test_448_d2_bptt_fp16_8bit.json`              fp16 + adamw_bnb_8bit

⚠️ 老 schema（2026-09-16 那两个）**只有** `step_pixel_l1_255`/`full_pixel_l1_255`，
没有 `step_psnr`/`step_ms_ssim`/`per_image` ⇒ 本脚本只用 L1 曲线（两边都有的量），
不做 PSNR/自适应分析（那些由 analyze_rd_earlystop.py 负责, 但它吃不了老 schema 的 PSNR）。

用法:
  python tools/plot_l1_curves.py --out fig.png \
    --arm "448 fp32 BPTT（历史）=/path/bptt24_bptt_infer_test.json" \
    --arm "448 fp16+8bit（本轮）=/path/test_448_d2_bptt_fp16_8bit.json"
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", action="append", required=True, help="name=path.json")
    ap.add_argument("--title", default=None)
    ap.add_argument("--ylabel", default="pixel L1 (0-255, full canvas)")
    a = ap.parse_args()

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    colors = ["tab:blue", "tab:gray", "tab:red", "tab:green", "tab:purple"]
    for k, spec in enumerate(a.arm):
        name, path = spec.split("=", 1)
        d = json.load(open(path))
        y = d["step_pixel_l1_255"]
        t = d.get("decoder_steps") or list(range(1, len(y) + 1))
        full = d.get("full_pixel_l1_255")
        style = "-o"
        if "detach" in name.lower():
            style = "--s"
        ax.plot(t, y, style, ms=4.5, color=colors[k % len(colors)],
                label=f"{name}  (full={full:.2f}, span={y[0]-y[-1]:+.2f})")
    ax.set_xlabel("sampling step t  (tokens = t+1)")
    ax.set_ylabel(a.ylabel)
    ax.set_title(a.title or "448x252 / K=576 / 24 steps / d2 / BPTT — same recipe, "
                            "different precision & optimizer")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(f"[plot] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
