"""sweep_res_jpeg.py — 与 sweep_res_train.py 完全同口径的 JPEG/WebP 基线

同图、同分辨率（直接 resize 到 S×S）、同 PSNR 口径（0-255、由聚合 MSE 反推）、
同 bpp 分母（S×S）。零训练，CPU 跑。

用法:
  python sweep_res_jpeg.py --size 112 --out /root/autodl-tmp/srres/out/112_jpeg.json
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os

import numpy as np
from PIL import Image, ImageOps

JPG_Q = [5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90]
WEBP_Q = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--val_dir", default="/root/autodl-tmp/srres/data/DIV2K_valid_HR")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jpg_q", default=",".join(map(str, JPG_Q)))
    ap.add_argument("--webp_q", default=",".join(map(str, WEBP_Q)))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.val_dir, "*.png")))
    if args.limit:
        files = files[:args.limit]
    S = args.size
    px = S * S
    jq = [int(v) for v in args.jpg_q.split(",")]
    wq = [int(v) for v in args.webp_q.split(",")]

    acc = {}
    for i, fp in enumerate(files):
        img = ImageOps.exif_transpose(Image.open(fp)).convert("RGB").resize(
            (S, S), Image.BICUBIC)
        gt = np.asarray(img, np.uint8)
        for fmt, qs in (("JPEG", jq), ("WebP", wq)):
            for q in qs:
                buf = io.BytesIO()
                if fmt == "JPEG":
                    img.save(buf, format="JPEG", quality=q, subsampling=0)
                else:
                    img.save(buf, format="WebP", quality=q, method=6)
                nbytes = buf.tell()
                rec = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
                mse = float(np.mean((rec.astype(np.float64) - gt.astype(np.float64)) ** 2))
                a = acc.setdefault((fmt, q), [0.0, 0.0, 0])
                a[0] += nbytes * 8 / px
                a[1] += mse
                a[2] += 1
        if (i + 1) % 25 == 0:
            print(f"[jpeg S={S}] {i+1}/{len(files)}", flush=True)

    out = {"size": S, "px": px, "n": len(files), "codecs": {}}
    for (fmt, q), (bpp, mse, n) in sorted(acc.items()):
        mse /= n
        out["codecs"].setdefault(fmt, []).append(
            {"q": q, "bpp": bpp / n, "mse": mse,
             "psnr": 10.0 * np.log10(255.0 ** 2 / max(mse, 1e-12))})
    for fmt in out["codecs"]:
        out["codecs"][fmt].sort(key=lambda r: r["bpp"])
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"-> {args.out}")
    for fmt, rows in out["codecs"].items():
        print(f"  {fmt}: " + " ".join(f"{r['bpp']:.3f}bpp/{r['psnr']:.1f}dB"
                                      for r in rows))


if __name__ == "__main__":
    main()
