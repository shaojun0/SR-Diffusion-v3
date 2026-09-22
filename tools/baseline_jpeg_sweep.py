#!/usr/bin/env python3
"""baseline_jpeg_sweep.py — E1 最小子集：传统 codec 基线的 RD 曲线（CPU only，零训练）

口径与我们的模型**完全对齐**：
  · 同一预处理: 原图 → fit_to_canvas(1600×900) → resize(448×252, BICUBIC)
  · 同一 bpp 分母: 448×252 = 112,896 px（与我们推断 bpp 时的分母一致）
  · 同一指标: PSNR(0-255) 与内容区 PSNR（剔 letterbox padding，fill=(0,0,0)）

为什么必须做: 我们只有一条自己的曲线, 没有"别人"的曲线 ⇒ 无法判断 18–19 dB 算好还是差。
这张表就是 PLAN §4 E1 的最小可用版本（BPG/VTM/CompressAI 后续再补）。

用法:
    python tools/baseline_jpeg_sweep.py --data_dir /root/autodl-tmp/construction_site \
        --out /root/autodl-tmp/cot_l1/eval_psnr/baseline_classic.json
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os

import numpy as np
from PIL import Image, ImageOps

from data_v2 import fit_to_canvas



def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 10.0 * np.log10(255.0 ** 2 / max(mse, 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 张（0=全量）")
    ap.add_argument("--w", type=int, default=448)
    ap.add_argument("--h", type=int, default=252)
    ap.add_argument("--jpg_q", default="3,5,8,10,15,20,25,30,40,50,60,70,80,90")
    ap.add_argument("--webp_q", default="3,5,10,15,20,30,40,50,60,70,80,90")
    args = ap.parse_args()
    global W, H, PX
    W, H = args.w, args.h
    PX = W * H

    from datasets import load_dataset
    files = sorted(glob.glob(os.path.join(args.data_dir, "test-*.parquet")))
    assert files, f"无 test-*.parquet in {args.data_dir}"
    ds = load_dataset("parquet", data_files=files, split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    n = len(ds)
    print(f"[data] {n} 张")

    jpg_q = [int(q) for q in args.jpg_q.split(",")]
    webp_q = [int(q) for q in args.webp_q.split(",")]
    try:
        Image.init()
        _ = Image.new("RGB", (8, 8)).save(io.BytesIO(), format="AVIF")
        has_avif = True
    except Exception:
        has_avif = False
    print(f"[codec] JPEG/WebP{'/AVIF' if has_avif else ''} (AVIF 不可用则跳过)")

    acc = {}   # (fmt,q) -> [sum_bpp, sum_psnr, sum_mse, cnt]
    for i in range(n):
        row = ds[i]
        im = row["image"]
        raw = im["bytes"] if isinstance(im, dict) else im
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")
        img = fit_to_canvas(img, (1600, 900)).resize((W, H), Image.BICUBIC)
        gt = np.asarray(img, np.uint8)

        for fmt, qs in (("JPEG", jpg_q), ("WebP", webp_q)):
            for q in qs:
                buf = io.BytesIO()
                im2 = img.copy()
                if fmt == "WebP":
                    im2.save(buf, format="WebP", quality=q, method=4)
                else:
                    im2.save(buf, format=fmt, quality=q)
                nbytes = buf.tell()
                rec = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"),
                                 np.uint8)
                mse = float(np.mean((rec.astype(np.float64)
                                     - gt.astype(np.float64)) ** 2))
                k = (fmt, q)
                a = acc.setdefault(k, [0.0, 0.0, 0.0, 0])
                a[0] += nbytes * 8 / PX
                a[2] += mse
                a[3] += 1
        if (i + 1) % 200 == 0 or (i + 1) == n:
            print(f"  ... {i + 1}/{n}", flush=True)

    rows = []
    for (fmt, q), (bpp_s, _p, mse_s, c) in sorted(acc.items(), key=lambda x: x[1][0] / x[1][3]):
        mean_bpp = bpp_s / c
        mean_mse = mse_s / c
        rows.append({"codec": fmt, "quality": q, "bpp": mean_bpp,
                     "psnr": float(10 * np.log10(255.0 ** 2 / max(mean_mse, 1e-12)))})

    print(f"\n{'codec':>6}{'q':>5}{'bpp':>10}{'PSNR(dB)':>10}")
    for r in rows:
        print(f"{r['codec']:>6}{r['quality']:>5}{r['bpp']:>10.4f}{r['psnr']:>10.2f}")

    # 与我们的曲线对比（同 bpp 插值）
    ours = "/root/autodl-tmp/cot_l1/eval_psnr/bptt_construction_site_test.json"
    cmp_tbl = []
    if os.path.isfile(ours):
        with open(ours) as f:
            d = json.load(f)
        b = np.array(d["step_bpp_beta1"], float)
        p = np.array(d["step_psnr"], float)
        print(f"\n{'bpp':>9}{'ours':>9}{'JPEG':>9}{'WebP':>9}{'Δ vs JPEG':>11}")
        for r in rows:
            if r["bpp"] < b[0] or r["bpp"] > b[-1]:
                continue
            o = float(np.interp(r["bpp"], b, p))
            jr = min([x for x in rows if x["codec"] == "JPEG"],
                     key=lambda x: abs(x["bpp"] - r["bpp"]))
            wr = min([x for x in rows if x["codec"] == "WebP"],
                     key=lambda x: abs(x["bpp"] - r["bpp"]))
            cmp_tbl.append({"bpp": r["bpp"], "ours": o, "jpeg": jr["psnr"],
                            "webp": wr["psnr"], "d_jpeg": o - jr["psnr"],
                            "d_webp": o - wr["psnr"]})
        for c in cmp_tbl:
            print(f"{c['bpp']:>9.4f}{c['ours']:>9.2f}{c['jpeg']:>9.2f}"
                  f"{c['webp']:>9.2f}{c['d_jpeg']:>+11.2f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"n": n, "canvas": [W, H], "px": PX, "rows": rows,
                   "compare": cmp_tbl}, f, indent=2, ensure_ascii=False)
    print(f"\n[save] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
