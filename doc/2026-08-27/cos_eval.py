"""
逐 patch 余弦相似度评估 — 输出 F_hat vs 原始 DINO patch 特征
=================================================
用户要看: 每个输出 patch 与原始 patch 向量的余弦相似度, 判断训练效果。

核心坑(勿踩): DINO patch 特征每图几乎恒定(跨位置 std≈5e-5, 全部 patch 几乎
同向) ⇒ 原始余弦 cos(F_hat[k], patch[k]) 天然≈1.0, 无区分度。
因此必须同时给出:
  · 目标平度: patch 行间余弦(目标本身有多"平") —— 决定"原始余弦"的下限
  · 原始余弦: cos(F_hat[k], patch[k]) —— 表面相似度
  · 结构余弦: 去质心后 cos(F_hat[k]-μ, patch[k]-μ) —— 是否还原了真实的
    空间差异(这才是"还原效果"); ≈0 ⇒ 输出是常数/质心, 未还原结构
  · 质心基线: 输出=每图 patch 均值向量时的余弦 —— "输出常数"能达到的水平
  · 随机基线: 输出=随机同分布向量的余弦 —— 无信息下限
逐 patch 位置 k=0..575 全部输出, 供用户自行判断。

用法:
    python cos_eval.py --final_model output/phase1_v2_fp32_uniform/final_model.pt \
        --tag fp32_uniform --output cos_fp32_uniform.json
    python cos_eval.py --final_model output/phase1_v2_test/final_model.pt \
        --tag bf16_density --output cos_bf16_density.json
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
    p.add_argument("--data_dir", default="/root/autodl-tmp/construction_site")
    p.add_argument("--dino_dir", default="/root/autodl-tmp/models/dinov2-large")
    p.add_argument("--final_model", required=True)
    p.add_argument("--tag", default="model", help="结果标识")
    p.add_argument("--output", default="cos_eval.json")
    p.add_argument("--model_input", default="448x252")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=512, help="test 样本数(0=全量3004)")
    return p.parse_args()


def main():
    args = parse_args()
    W, H = (int(v) for v in args.model_input.lower().split("x"))
    N = (W // 14) * (H // 14)               # 576

    dino = Dinov2Model.from_pretrained(args.dino_dir)
    if getattr(dino.config, "use_mask_token", False):
        dino.config.use_mask_token = False
        del dino.embeddings.mask_token
    model = SRPhase1V2(dinov2=dino, num_patches=N, dim=dino.config.hidden_size,
                       reencoder_depth=4)
    sd = torch.load(args.final_model, map_location="cpu")
    miss, unexp = model.load_state_dict(sd, strict=True)
    assert not miss and not unexp, (miss, unexp)
    model.eval().cuda()
    print(f"[model] {args.tag}: {args.final_model} (全 fp32 前向)")

    files = sorted(glob.glob(os.path.join(args.data_dir, "test-*.parquet")))
    ds = ParquetImageDataset(files, limit=args.limit)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        collate_fn=V2Collator(model_size=(W, H)))

    # 逐位置累加器 (k=0..N-1)
    cos_raw_sum = np.zeros(N)
    cos_struct_sum = np.zeros(N)
    cos_centroid_sum = np.zeros(N)   # 质心基线(按图)
    cos_rand_sum = np.zeros(N)
    # 目标平度: patch 行间余弦(按图, 随机采样 64 对)
    cos_pp_sum = 0.0
    n_pairs_pp = 0
    # 输出平度: F_hat 行间余弦
    cos_ff_sum = 0.0
    n_pairs_ff = 0
    n_imgs = 0

    rng = np.random.default_rng(0)
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x = batch["pixel_values"].cuda()
            B = x.shape[0]
            feats = model.dinov2(x).last_hidden_state          # (B,257,D)
            patch = feats[:, 1:].float()                       # (B,N,D)
            cls = feats[:, 0:1]
            sp = model.special_bank(B, x.device)
            z = model.re_encoder(torch.cat([cls, sp, patch], dim=1))
            F_hat = model.decoder(z[:, 0:1], z[:, 1:1 + N]).float()   # (B,N,D)

            # ── 原始余弦: 逐位置 cos(F_hat[:,k], patch[:,k]) ──
            C = F.cosine_similarity(F_hat, patch, dim=-1)      # (B,N)
            cos_raw_sum += C.sum(0).cpu().numpy()

            # ── 结构余弦: 去每图质心后逐位置 ──
            pc = patch - patch.mean(1, keepdim=True)
            fc = F_hat - F_hat.mean(1, keepdim=True)
            Cs = F.cosine_similarity(fc, pc, dim=-1)           # (B,N)
            cos_struct_sum += Cs.sum(0).cpu().numpy()

            # ── 质心基线: F=每图 patch 均值 ⇒ cos(μ, patch[k]) ──
            mu = patch.mean(1, keepdim=True).expand_as(patch)
            Cc = F.cosine_similarity(mu, patch, dim=-1)
            cos_centroid_sum += Cc.sum(0).cpu().numpy()

            # ── 随机基线: 同图同分布随机向量 ──
            noise = torch.randn_like(F_hat)
            noise = noise / noise.norm(dim=-1, keepdim=True)
            noise = noise * F_hat.norm(dim=-1, keepdim=True)   # 与输出同模长
            Cr = F.cosine_similarity(noise, patch, dim=-1)
            cos_rand_sum += Cr.sum(0).cpu().numpy()

            # ── 目标平度: 每图随机 64 对 patch 行间余弦(逐图采样, 避免索引 bug) ──
            idx = rng.integers(0, N, size=(B, 64, 2))
            for b in range(B):
                pi = patch[b, idx[b, :, 0]]          # (64,D)
                pj = patch[b, idx[b, :, 1]]
                cos_pp_sum += F.cosine_similarity(pi, pj, dim=-1).sum().item()
                n_pairs_pp += 64
                fi = F_hat[b, idx[b, :, 0]]
                fj = F_hat[b, idx[b, :, 1]]
                cos_ff_sum += F.cosine_similarity(fi, fj, dim=-1).sum().item()
                n_pairs_ff += 64

            # ── 跨位置 std 补充证据: 输出是否比目标更"平" ──
            std_p = patch.std(dim=1).mean().item()
            std_f = F_hat.std(dim=1).mean().item()
            std_p_acc = getattr(globals(), "_std_p_acc", 0.0) + std_p
            std_f_acc = getattr(globals(), "_std_f_acc", 0.0) + std_f
            globals()["_std_p_acc"] = std_p_acc
            globals()["_std_f_acc"] = std_f_acc

            n_imgs += B
            if (bi + 1) % 5 == 0 or (bi + 1) == len(loader):
                print(f"  ... {n_imgs}/{len(ds)}", flush=True)

    # 汇总
    def norm(x):
        return x / n_imgs
    cos_raw = norm(cos_raw_sum)
    cos_struct = norm(cos_struct_sum)
    cos_centroid = norm(cos_centroid_sum)
    cos_rand = norm(cos_rand_sum)
    cos_pp = cos_pp_sum / n_pairs_pp
    cos_ff = cos_ff_sum / n_pairs_ff
    std_p_mean = globals().get("_std_p_acc", 0.0) / len(loader)
    std_f_mean = globals().get("_std_f_acc", 0.0) / len(loader)

    print(f"\n=== {args.tag} 逐 patch 余弦相似度 ({n_imgs} 图 × {N} patch) ===")
    print(f"[平度] 目标 patch 行间余弦 = {cos_pp:.6f}  (≈1 ⇒ 目标几乎同向, "
          f"原始余弦天然≈1, 无区分度)")
    print(f"[平度] 输出 F_hat 行间余弦 = {cos_ff:.6f}  (≈1 ⇒ 输出也是常数)")
    print(f"[平度] 跨位置 std: 目标 {std_p_mean:.6f} / 输出 {std_f_mean:.6f} "
          f"(输出 std << 目标 std ⇒ 输出比目标更平, 未还原差异)")
    print(f"[对照] 随机基线(同模长噪声) 余弦 = {cos_rand.mean():.6f}")
    print(f"[对照] 质心基线(输出=均值)  余弦 = {cos_centroid.mean():.6f}")
    print(f"[表面] 原始余弦 cos(F_hat[k], patch[k]) = {cos_raw.mean():.6f} "
          f"(min {cos_raw.min():.6f} / max {cos_raw.max():.6f})")
    print(f"[真实] 结构余弦(去质心) = {cos_struct.mean():.6f} "
          f"(min {cos_struct.min():.6f} / max {cos_struct.max():.6f})")
    print(f"[判据] 结构余弦 ≈ 0 ⇒ 模型输出≈质心常数, 未还原空间差异; "
          f"> 0.3 才说明还原了结构")

    # 逐位置分布摘要 (按空间位置排序的前后段)
    print(f"\n[位置分布] 原始余弦 前8: {np.round(cos_raw[:8], 4)}")
    print(f"[位置分布] 结构余弦 前8: {np.round(cos_struct[:8], 4)}")

    out = {
        "tag": args.tag, "final_model": args.final_model,
        "n_imgs": n_imgs, "num_patches": N,
        "target_rowwise_cos": cos_pp,
        "output_rowwise_cos": cos_ff,
        "target_crosspos_std": std_p_mean,
        "output_crosspos_std": std_f_mean,
        "rand_baseline_cos": float(cos_rand.mean()),
        "centroid_baseline_cos": float(cos_centroid.mean()),
        "raw_cos_per_patch": [float(v) for v in cos_raw],
        "struct_cos_per_patch": [float(v) for v in cos_struct],
        "raw_cos_mean": float(cos_raw.mean()),
        "struct_cos_mean": float(cos_struct.mean()),
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[save] {args.output}")


if __name__ == "__main__":
    main()
