"""
SR-Diffusion Phase 1 v2 — test 分支推理测试（像素目标版, 2026-08-27）
=================================================
架构（model_v2.py, test 分支）: DINOv2-large → ReEncoder → OutputQueryDecoder
    （输出查询注意力, KV 因果: 每采样步 t 只见前缀 s≤t）→ PixelHead →
    像素 patch 预测。

测试项（像素目标 = 最终判据）:
    1) 全量重建像素 L1（归一化空间 + 反归一化 0-255 空间双口径）——
       对照: 全图平均色 baseline / 每 patch 平均色 baseline。
       像素 L1 必须显著优于"平均色"才有还原意义（特征空间 L1 是假象,
       已证实特征目标退化）。
    2) 渐进重建曲线（前缀扫描的新架构等价物）—— 解码器 KV 因果: 采样步
       t 只见前缀 s≤t。对每步 t∈T_sub 度量像素 L1(Y_t_pix, target_pix),
       得到"前缀越长重建越精"的渐进曲线。

用法:
    python infer_v2_test.py \
        --data_dir /root/autodl-tmp/construction_site \
        --dino_dir /root/autodl-tmp/models/dinov2-large \
        --final_model output/phase1_v2_pixelfp32/final_model.pt \
        --output output/phase1_v2_pixelfp32/infer_test.json
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Dinov2Model

from data_v2 import ParquetImageDataset, V2Collator, DINO_MEAN, DINO_STD
from model_v2 import SRPhase1V2


def parse_args():
    p = argparse.ArgumentParser(description="infer test (pixel target) for OutputQueryDecoder v2")
    p.add_argument("--data_dir", required=True, help="parquet 目录(test-*.parquet)")
    p.add_argument("--dino_dir", default="models/dinov2-large")
    p.add_argument("--final_model", required=True, help="训练好的权重(final_model.pt)")
    p.add_argument("--output", default="output/phase1_v2_pixelfp32/infer_test.json")
    p.add_argument("--model_input", default="448x252")
    p.add_argument("--canvas", default="1600x900")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="只用前 N 条 test(0=全量)")
    p.add_argument("--reencoder_depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--decoder_steps", default=None,
                   help="必须与训练一致(逗号分隔); 默认 square_step_schedule(N)")
    return p.parse_args()


def _patch_to_img(pix_patches, H, W):
    """(B,N,14,14,3) 归一化像素 patch → (B,H,W,3) 反归一化 0-255 图像。

    布局与 DINO 一致: row-major (先 y 后 x), N = (H/14)*(W/14)。
    """
    B = pix_patches.shape[0]
    img = pix_patches.reshape(B, H // 14, W // 14, 14, 14, 3) \
                     .permute(0, 1, 3, 2, 4, 5) \
                     .reshape(B, H, W, 3)
    img = img.cpu().numpy() * DINO_STD + DINO_MEAN
    return np.clip(img, 0, 255)


def main():
    args = parse_args()
    W, H = (int(v) for v in args.model_input.lower().split("x"))
    num_patches = (W // 14) * (H // 14)
    PATCH_PX = 14 * 14 * 3

    steps = None
    if args.decoder_steps:
        steps = [int(s) for s in args.decoder_steps.split(",") if s.strip()]

    # ── 模型: 训练好的重建权重 ──
    dino = Dinov2Model.from_pretrained(args.dino_dir)
    if getattr(dino.config, "use_mask_token", False):
        dino.config.use_mask_token = False
        del dino.embeddings.mask_token
    model = SRPhase1V2(dinov2=dino, num_patches=num_patches,
                       dim=dino.config.hidden_size,
                       reencoder_depth=args.reencoder_depth,
                       heads=args.heads, mlp_ratio=args.mlp_ratio,
                       decoder_steps=steps)
    sd = torch.load(args.final_model, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval().cuda()
    T_steps = model.decoder.steps
    print(f"[model] loaded {args.final_model}: N={num_patches}, "
          f"decoder 采样 {len(T_steps)} 步 {T_steps[:6]}...{T_steps[-3:]}")

    # ── 数据: test 分片, 与训练同预处理（1600:900 画布 → 448x252）──
    test_files = sorted(glob.glob(os.path.join(args.data_dir, "test-*.parquet")))
    assert test_files, f"无 test-*.parquet in {args.data_dir}"
    coll = V2Collator(model_size=(W, H), canvas=(1600, 900))
    ds = ParquetImageDataset(test_files, limit=args.limit)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=coll)
    n_total = len(ds)
    print(f"[data] {n_total} 条 test")

    # ── 前向: 全量 test, 像素 L1（归一化空间 + 0-255 空间）──
    norm_sum, norm_sq, pix_sum, pix_sq, n = 0.0, 0.0, 0.0, 0.0, 0
    step_pix_sum = np.zeros(len(T_steps), np.float64)   # 每采样步像素 L1 (0-255)
    step_pix_sq = np.zeros(len(T_steps), np.float64)
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x = batch["pixel_values"].cuda()            # (B,3,H,W) 归一化
            B, C, Hh, Ww = x.shape
            feats = model.dinov2(x).last_hidden_state
            cls, patch_feat = feats[:, 0:1], feats[:, 1:]
            specials = model.special_bank(B, x.device)
            z = model.re_encoder(torch.cat([cls, specials, patch_feat], dim=1))
            z_cls, z_s = z[:, 0:1], z[:, 1:1 + num_patches]
            F_hat = model.decoder(z_cls, z_s)           # (B,N,D) 特征(采样步平均)
            Y = model.decoder.last_Y                    # (B,|T|,N,D) 特征
            Y_pix = model.pixel_head(Y)                 # (B,|T|,N,588) 像素(归一化)
            F_pix = model.pixel_head(F_hat)             # (B,N,588) 采样步平均
            target = x.reshape(B, C, Hh // 14, 14, Ww // 14, 14) \
                      .permute(0, 2, 4, 1, 3, 5).reshape(B, num_patches, PATCH_PX)
            # 归一化空间 L1
            l1_norm = (F_pix - target).abs().mean(dim=(1, 2))          # (B,)
            norm_sum += l1_norm.sum().item()
            norm_sq += (l1_norm ** 2).sum().item()
            # 0-255 空间: 反归一化 (patch 级, 与 pixel_recon_check 同口径)
            gt_255 = _patch_to_img(target, Hh, Ww)      # (B,H,W,3)
            recon_255 = _patch_to_img(F_pix, Hh, Ww)
            l1_pix = np.abs(recon_255 - gt_255).mean(axis=(1, 2, 3))   # (B,)
            pix_sum += l1_pix.sum().item()
            pix_sq += (l1_pix ** 2).sum().item()
            # 每采样步像素 L1 (0-255)
            for i in range(len(T_steps)):
                step_img = _patch_to_img(Y_pix[:, i], Hh, Ww)
                sl1 = np.abs(step_img - gt_255).mean(axis=(1, 2, 3))   # (B,)
                step_pix_sum[i] += sl1.sum().item()
                step_pix_sq[i] += (sl1 ** 2).sum().item()
            n += B
            if (bi + 1) % 10 == 0 or (bi + 1) == len(loader):
                print(f"  ... {n}/{n_total} ({time.time() - t0:.0f}s)", flush=True)

    norm_mean = norm_sum / n
    pix_mean = pix_sum / n
    pix_std = np.sqrt(max(pix_sq / n - pix_mean ** 2, 0.0))
    step_pix_mean = step_pix_sum / n
    step_pix_std = np.sqrt(np.maximum(step_pix_sq / n - step_pix_mean ** 2, 0.0))

    print(f"\n[full] 全量重建像素 L1 (归一化空间) = {norm_mean:.6f}")
    print(f"[full] 全量重建像素 L1 (0-255 空间) = {pix_mean:.2f} ± {pix_std:.2f}")
    print(f"       参照(旧实验): 全图平均色≈61, 每patch平均色≈?, 质心基线见 pixel_recon_check")
    print(f"[steps] 渐进曲线 ({len(T_steps)} 步, 0-255 像素 L1):")
    for i, t in enumerate(T_steps):
        print(f"    t={t:4d} (前缀 {t + 1:4d} 键) L1 = {step_pix_mean[i]:.2f} ± {step_pix_std[i]:.2f}")
    head = step_pix_mean[:min(4, len(step_pix_mean))]
    tail = step_pix_mean[max(0, len(step_pix_mean) - 4):]
    print(f"[steps] 前段(短前缀) {head.mean():.2f} | 后段(长前缀) {tail.mean():.2f} | "
          f"最短/最长 = {step_pix_mean[0]:.2f}/{step_pix_mean[-1]:.2f}")
    print(f"[time] {(time.time() - t0):.0f}s | {n} 图")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "n": n, "num_patches": num_patches, "input": [W, H],
            "decoder_steps": T_steps,
            "full_norm_l1": float(norm_mean),
            "full_pixel_l1_255": float(pix_mean), "full_pixel_std_255": float(pix_std),
            "step_pixel_l1_255": [float(v) for v in step_pix_mean],
            "step_pixel_std_255": [float(v) for v in step_pix_std],
            "head_mean": float(head.mean()), "tail_mean": float(tail.mean()),
            "shortest_l1": float(step_pix_mean[0]), "longest_l1": float(step_pix_mean[-1]),
            "time_s": round(time.time() - t0, 1),
        }, f, indent=2, ensure_ascii=False)
    print(f"[save] {args.output}")


if __name__ == "__main__":
    main()
