"""diag_step_spread.py — 逐步输出到底有没有变化？（BPTT 精修是否真的发生）

对一个已训好的产物目录，测三条「步间差异」曲线（都在 0-255 像素空间、全画布）:

  · |img_t − img_last|   第 t 步输出与**末步**输出的平均绝对差  ← "carry 到底把输出推动了多少"
  · |img_t − img_{t−1}|  相邻步之间的平均绝对差                ← "这一步比上一步改了多少"
  · 以及 |img_1 − img_last| / |img_last| 的相对量

判据: 若 |img_t − img_last| 全程 ≈ 0（远小于模型自身的 L1），
⇒ 该模型**推理时忽略 carry**（所有采样步输出同一张图）——这正是"平坦 RD 曲线"的机制级证据，
而不是"曲线平但每步仍在动、只是变好得少"。

用法:
  python tools/diag_step_spread.py --out_dir <训练产物目录> [--limit 8] [--dino_dir ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_v2 import DINO_MEAN, DINO_STD, ParquetImageDataset, V2Collator  # noqa: E402
from model_v2 import SRPhase1V2, patches_to_image  # noqa: E402
from transformers import Dinov2Model  # noqa: E402

MEAN = [float(v) for v in DINO_MEAN]
STD = [float(v) for v in DINO_STD]


def build(out_dir, dino_dir):
    info = json.load(open(os.path.join(out_dir, "model_info.json")))
    dino = Dinov2Model.from_pretrained(dino_dir)
    if getattr(dino.config, "use_mask_token", False):
        dino.config.use_mask_token = False
        del dino.embeddings.mask_token
    m = SRPhase1V2(dinov2=dino, num_patches=info["num_patches"], dim=info["dim"],
                   heads=info["heads"], mlp_ratio=info["mlp_ratio"],
                   decoder_steps=info["decoder_steps"],
                   decoder_depth=info["decoder_depth"],
                   num_specials=info["num_specials"],
                   stack_dim=int(info.get("stack_dim") or 0),
                   decoder_dropout=float(info.get("decoder_dropout") or 0.0))
    sd = torch.load(os.path.join(out_dir, "final_model.pt"), map_location="cpu")
    m.load_state_dict(sd)
    return m, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--data_dir", default="/root/autodl-tmp/construction_site")
    ap.add_argument("--dino_dir", default="/root/autodl-tmp/models/dinov2-large")
    ap.add_argument("--limit", type=int, default=8)
    a = ap.parse_args()

    m, info = build(a.out_dir, a.dino_dir)
    W, H = info["input_size"]
    steps = info["decoder_steps"]
    m = m.cuda().eval()
    files = sorted(f for f in os.listdir(a.data_dir) if f.startswith("test-") and
                   f.endswith(".parquet"))
    ds = ParquetImageDataset([os.path.join(a.data_dir, f) for f in files],
                             limit=a.limit)
    coll = V2Collator(model_size=(W, H))
    batch = coll([ds[i] for i in range(len(ds))])
    x = batch["pixel_values"].cuda()
    with torch.no_grad():
        out = m(x)
        Y, tgt = out["Y_pix"], out["target_pix"]        # (B,T,N,588) / (B,N,588)
        B, T, N, _ = Y.shape
        # ⚠️ patches_to_image(pixels, H, W, ...) —— 是 (H, W) 不是 (W, H);
        #    且它返回 **Tensor**（消费方要自己 .cpu().numpy()）。
        rec = patches_to_image(Y.reshape(B * T, N, 588), H, W, MEAN, STD) \
            .cpu().numpy().astype(np.float32)
        rec = rec.reshape(B, T, H, W, 3)                        # (B,T,H,W,3)
        gt = patches_to_image(tgt, H, W, MEAN, STD) \
            .cpu().numpy().astype(np.float32)                   # (B,H,W,3)

    l1_last = float(np.abs(rec[:, -1] - gt).mean())
    l1_all = [float(np.abs(rec[:, i] - gt).mean()) for i in range(T)]
    print(f"[cfg] {a.out_dir}")
    print(f"      input={W}x{H} N={info['num_patches']} K={info['num_specials']} "
          f"steps={len(steps)} depth={info['decoder_depth']} "
          f"K={info.get('num_specials')} n_img={B}")
    print(f"{'t':>6} {'tokens':>7} {'L1_t(GT)':>9} {'|img_t-img_last|':>18} "
          f"{'|img_t-img_t-1|':>17}")
    for i, t in enumerate(steps):
        d_last = float(np.abs(rec[:, i] - rec[:, -1]).mean())
        d_prev = (float(np.abs(rec[:, i] - rec[:, i - 1]).mean()) if i else 0.0)
        print(f"{t:>6} {t + 1:>7} {l1_all[i]:>9.3f} {d_last:>18.4f} {d_prev:>17.4f}")
    span = float(np.abs(rec[:, 0] - rec[:, -1]).mean())
    gain = l1_all[0] - l1_all[-1]          # 正 = 末步比首步好
    r = span / max(l1_last, 1e-9)
    print(f"[结论] 对 GT 的 L1: 首步 {l1_all[0]:.3f} → 末步 {l1_all[-1]:.3f} "
          f"(改善 {gain:+.3f} px) | |首步−末步| = {span:.4f} px (占末步 L1 的 {r:.3f})")
    if r < 0.05:
        print("       ⇒ 输出**几乎不动**（轨迹退化到首步）")
    elif abs(gain) < 0.10 * max(l1_last, 1e-9):
        print("       ⇒ 轨迹**在动但不降误差**（各步质量基本相同）—— 注意这与"
              "「输出不动」是两种不同机制, 别混为一谈")
    else:
        pct = 100.0 * gain / max(l1_all[0], 1e-9)
        print(f"       ⇒ 轨迹在动**且降误差**（首→末 −{pct:.1f}%）= 真的在精修")


if __name__ == "__main__":
    main()
