"""
相似度扫描: 输入 patch 特征 vs 输出 F_hat 各个采样步 t 的平均相似度
=================================================
指标（每采样步 t 独立, 512 张 test, fp32）:
  1) 原始余弦:  cos(P, Y_t) —— 质心主导, 恒≈1.0, 区分度低
  2) 结构余弦:  cos(P-P̄, Y_t-Ȳ_t) —— 去质心后只比空间结构; ≈0 ⇒ 结构未还原
  3) 原始 L1 / 结构 L1 对照
注意: 余弦必须用 F.cosine_similarity; Tensor.norm(-1) 的 -1 是 p 阶数不是 dim!
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Dinov2Model

from data_v2 import ParquetImageDataset, V2Collator
from model_v2 import SRPhase1V2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--dino_dir", default="models/dinov2-large")
    p.add_argument("--final_model", required=True)
    p.add_argument("--model_input", default="448x252")
    p.add_argument("--limit", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--json_out", default="sim_vs_t.json")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(0)
    W, H = (int(v) for v in args.model_input.lower().split("x"))
    N = (W // 14) * (H // 14)

    dino = Dinov2Model.from_pretrained(args.dino_dir)
    if getattr(dino.config, "use_mask_token", False):
        dino.config.use_mask_token = False
        del dino.embeddings.mask_token
    model = SRPhase1V2(dinov2=dino, num_patches=N,
                       dim=dino.config.hidden_size, reencoder_depth=4)
    sd = torch.load(args.final_model, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval().cuda()
    T_steps = model.decoder.steps
    NT = len(T_steps)

    files = sorted(glob.glob(os.path.join(args.data_dir, "test-*.parquet")))
    ds = ParquetImageDataset(files, limit=args.limit)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=8, collate_fn=V2Collator(model_size=(W, H)))

    c_struct = np.zeros(NT); c_raw = np.zeros(NT)
    l1_struct = np.zeros(NT); l1_raw = np.zeros(NT)
    n_pairs = 0; n_imgs = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["pixel_values"].cuda()
            feats = model.dinov2(x).last_hidden_state
            cls, patch = feats[:, 0:1], feats[:, 1:]
            sp = model.special_bank(x.shape[0], x.device)
            z = model.re_encoder(torch.cat([cls, sp, patch], dim=1))
            _ = model.decoder(z[:, 0:1], z[:, 1:1 + N])
            Y = model.decoder.last_Y                    # (B,NT,N,D)
            P = patch.float(); Yf = Y.float()
            B = P.shape[0]
            C_raw = F.cosine_similarity(
                P.unsqueeze(1).expand_as(Yf), Yf, dim=-1)          # (B,NT,N)
            c_raw += C_raw.sum((0, 2)).cpu().numpy()
            l1_raw += (Yf - P.unsqueeze(1)).abs().mean((2, 3)).sum(0).cpu().numpy()
            P_c = P - P.mean(1, keepdim=True)
            Yc = Yf - Yf.mean(2, keepdim=True)
            C_struct = F.cosine_similarity(
                P_c.unsqueeze(1).expand_as(Yc), Yc, dim=-1)
            c_struct += C_struct.sum((0, 2)).cpu().numpy()
            l1_struct += (Yc - P_c.unsqueeze(1)).abs().mean((2, 3)).sum(0).cpu().numpy()
            n_pairs += B * N; n_imgs += B

    c_r = c_raw / n_pairs; c_s = c_struct / n_pairs
    l1r = l1_raw / n_imgs; l1s = l1_struct / n_imgs

    print(f"=== 输入 patch vs 输出 patch 各采样步 t 相似度 "
          f"({n_imgs} 图 × {N} patch) ===")
    print(f"{'t':>5} {'原始余弦':>10} {'结构余弦':>10} {'原始L1':>10} {'结构L1':>10}")
    for i, t in enumerate(T_steps):
        print(f"{t:>5} {c_r[i]:>10.6f} {c_s[i]:>10.6f} {l1r[i]:>10.6f} {l1s[i]:>10.6f}")
    print(f"\nt=0: 原始 {c_r[0]:.6f} 结构 {c_s[0]:.6f} | "
          f"t={T_steps[-1]}: 原始 {c_r[-1]:.6f} 结构 {c_s[-1]:.6f}")
    print(f"结构余弦 |max| = {np.abs(c_s).max():.6f} (≈0 ⇒ 结构未还原)")

    with open(args.json_out, "w") as f:
        json.dump({"steps": T_steps, "n_imgs": n_imgs,
                   "raw_cos": [float(v) for v in c_r],
                   "struct_cos": [float(v) for v in c_s],
                   "raw_l1": [float(v) for v in l1r],
                   "struct_l1": [float(v) for v in l1s]},
                  f, indent=2, ensure_ascii=False)
    print(f"[save] {args.json_out}")


if __name__ == "__main__":
    main()
