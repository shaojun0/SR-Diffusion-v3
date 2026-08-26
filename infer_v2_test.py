"""
SR-Diffusion Phase 1 v2 — test 分支推理测试（OutputQueryDecoder 版）
=================================================
架构（model_v2.py, test 分支）: DINOv2-large → ReEncoder → OutputQueryDecoder
    （输出查询注意力, KV 因果: 每采样步 t 只见前缀 s≤t）→ F_hat = 采样步平均。

测试项（对照 main 文档的 infer_k_sweep 精神, 适配新架构）:
    1) 全量重建 fp32 L1 —— 与旧 FeatureDecoder 基线（fp32 L1≈0.00114）对比。
       bf16 eval 低估（旧文档: 0.000995 vs fp32 0.00114）, 这里统一 fp32 度量。
    2) 渐进重建曲线（前缀扫描的新架构等价物）—— 解码器 KV 因果: 采样步 t
       只见前缀 s≤t。对每步 t∈T_sub 度量 L1(Y_t, patch), 得到"前缀越长重建
       越精"的渐进曲线; 前段(短前缀)应明显粗于后段(长前缀), 即信息前置。
       （旧 FeatureDecoder 版用"只喂前 k 个 z_s"做 k 扫描; 新解码器无该
       接口, 改为按采样步切分 last_Y —— 语义等价: 每步的可见键数=前缀长度。）

用法:
    python infer_v2_test.py \
        --data_dir /root/autodl-tmp/construction_site \
        --dino_dir /root/autodl-tmp/models/dinov2-large \
        --final_model output/phase1_v2_test/final_model.pt \
        --output output/phase1_v2_test/infer_test.json
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

from data_v2 import ParquetImageDataset, V2Collator
from model_v2 import SRPhase1V2


def parse_args():
    p = argparse.ArgumentParser(description="infer test for OutputQueryDecoder v2")
    p.add_argument("--data_dir", required=True, help="parquet 目录(test-*.parquet)")
    p.add_argument("--dino_dir", default="models/dinov2-large")
    p.add_argument("--final_model", required=True, help="训练好的权重(final_model.pt)")
    p.add_argument("--output", default="output/phase1_v2_test/infer_test.json")
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
    p.add_argument("--decoder_loss_weight", default="density")
    return p.parse_args()


def main():
    args = parse_args()
    W, H = (int(v) for v in args.model_input.lower().split("x"))
    num_patches = (W // 14) * (H // 14)

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
                       decoder_steps=steps,
                       decoder_loss_weight=args.decoder_loss_weight)
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

    # ── 单次前向同时取 Y 与 patch: 直接调模型的内部管线（fp32）──
    # 注意: 不走 model(x) 而是复刻其内部, 是为了同一次前向拿到
    # last_Y(每采样步) + patch(真值), 避免重复前向; 与 model.forward
    # 完全同路径（同一模块调用顺序）, 度量口径一致。
    def forward_once(x):
        feats = model.dinov2(x).last_hidden_state      # (B,257,D)
        cls, patch = feats[:, 0:1], feats[:, 1:]
        specials = model.special_bank(x.shape[0], x.device)
        z = model.re_encoder(torch.cat([cls, specials, patch], dim=1))
        z_cls, z_s = z[:, 0:1], z[:, 1:1 + num_patches]
        F_hat = model.decoder(z_cls, z_s)
        Y = model.decoder.last_Y
        return F_hat, Y, patch

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x = batch["pixel_values"].cuda()
            B = x.shape[0]
            F_hat, Y, patch = forward_once(x)          # fp32
            l1_full = (F_hat - patch).abs().mean(dim=(1, 2))          # (B,)
            l1_steps = (Y - patch.unsqueeze(1)).abs().mean(dim=(2, 3))  # (B,|T|)
            full_sum += l1_full.sum().item()
            full_sq += (l1_full ** 2).sum().item()
            step_sum += l1_steps.sum(dim=0).cpu().numpy()
            step_sq += (l1_steps ** 2).sum(dim=0).cpu().numpy()
            n += B
            if (bi + 1) % 10 == 0 or (bi + 1) == len(loader):
                print(f"  ... {n}/{n_total} ({time.time() - t0:.0f}s)", flush=True)

    full_mean = full_sum / n
    full_std = np.sqrt(max(full_sq / n - full_mean ** 2, 0.0))
    step_mean = step_sum / n
    step_std = np.sqrt(np.maximum(step_sq / n - step_mean ** 2, 0.0))
    # density 权重下 F_hat 平均与加权损失口径; 这里报告每步原始 L1 曲线
    # （采样步 t 的可见键数 = t+1, 即前缀长度; t=N 步=全前缀）
    head = step_mean[:min(4, len(step_mean))]
    tail = step_mean[max(0, len(step_mean) - 4):]
    print(f"\n[full] 全量重建 fp32 L1 = {full_mean:.6f} ± {full_std:.6f} (n={n})")
    print(f"[steps] 渐进曲线 ({len(T_steps)} 步):")
    for i, t in enumerate(T_steps):
        print(f"    t={t:4d} (前缀 {t + 1:4d} 键) L1 = {step_mean[i]:.6f} ± {step_std[i]:.6f}")
    print(f"[steps] 前段(短前缀) {head.mean():.6f} | 后段(长前缀) {tail.mean():.6f} | "
          f"最短/最长 = {step_mean[0]:.6f}/{step_mean[-1]:.6f}")
    print(f"[time] {(time.time() - t0):.0f}s | {n} 图")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "n": n, "num_patches": num_patches, "input": [W, H],
            "decoder_steps": T_steps,
            "full_fp32_l1": float(full_mean), "full_fp32_std": float(full_std),
            "step_l1": [float(v) for v in step_mean],
            "step_std": [float(v) for v in step_std],
            "head_mean": float(head.mean()), "tail_mean": float(tail.mean()),
            "shortest_l1": float(step_mean[0]), "longest_l1": float(step_mean[-1]),
            "time_s": round(time.time() - t0, 1),
        }, f, indent=2, ensure_ascii=False)
    print(f"[save] {args.output}")


if __name__ == "__main__":
    main()
