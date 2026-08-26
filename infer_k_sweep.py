"""
SR-Diffusion Phase 1 v2 — 还原精度 vs z_s 前缀长度 k 扫描（推理）
=================================================
只喂前 k 个 z_s（+z_cls）给 FeatureDecoder, 重建全量 N 个 DINO patch 特征,
度量 = L1(F_hat, patch)（与训练同一口径）。k=1..kmax 扫描, 附 k=0 与
全量 k=N 作参考锚点。

两种模式:
    1) k 扫描（默认）: 前缀长度扫描。训练后预期 k=1..32 平台期消失。
    2) 滑窗探针（--window N）: 扫描所有连续 N-token 窗口的还原 L1,
       验证"信息前置"（DESIGN §7.2）——前段窗口应明显好于后段；
       训练前实测任意 32-token 窗口 ≈0.0033 无差异（信息摊匀）。

用法:
    python infer_k_sweep.py \
        --data_dir /root/autodl-tmp/construction_site \
        --dino_dir /root/autodl-tmp/models/dinov2-large \
        --final_model output/phase1_v2/final_model.pt \
        --output output/phase1_v2/k_sweep.json

    滑窗探针（N=576, 32-token 窗口）:
    python infer_k_sweep.py --data_dir ... --dino_dir ... \
        --final_model output/phase1_v2/final_model.pt \
        --window 32 --output output/phase1_v2/window_probe.json
"""
import argparse
import glob
import json
import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import Dinov2Model

from data_v2 import ParquetImageDataset, V2Collator
from model_v2 import SRPhase1V2


def parse_args():
    p = argparse.ArgumentParser(description="recon L1 vs z_s prefix k")
    p.add_argument("--data_dir", required=True, help="parquet 目录(test-*.parquet)")
    p.add_argument("--dino_dir", default="models/dinov2-large")
    p.add_argument("--final_model", required=True, help="训练好的权重(final_model.pt)")
    p.add_argument("--output", default="output/phase1_v2/k_sweep.json")
    p.add_argument("--model_input", default="448x252")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=500, help="只用前 N 条 test(0=全量)")
    p.add_argument("--kmax", type=int, default=32, help="扫描 k 上界(默认 32)")
    p.add_argument("--anchors", default="", help="附加锚点 k, 逗号分隔(如 64,128,256)")
    p.add_argument("--window", type=int, default=0,
                   help=">0: 滑窗探针模式（DESIGN §7.2）——扫描所有连续 "
                        "window-token 窗口的还原 L1，验证信息前置")
    p.add_argument("--window_chunk", type=int, default=64,
                   help="窗口探针分块大小（控制峰值显存，默认 64）")
    return p.parse_args()


def window_sweep(args, model, loader, num_patches: int, W: int, H: int):
    """滑窗探针: 对每个连续 window-token 窗口喂 decoder, 度量还原 L1。

    目标（DESIGN §7.2 / MATH §8.2）: 前缀课程训练后, 前段窗口（如
    z_s[0:32]）的还原应明显好于后段（信息前置）；训练前实测为任意
    32-token 窗口 ≈0.0033 无差异（信息摊匀）。
    实现: 所有窗口合并进 batch 维一次 decoder 前向, 分块控制显存。
    """
    import numpy as np
    import torch
    S = num_patches - args.window + 1
    print(f"[window] N={num_patches}, window={args.window}, "
          f"{S} 个滑窗, chunk={args.window_chunk}")
    sum_l1 = np.zeros(S)
    sum_l1sq = np.zeros(S)
    n = 0
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x = batch["pixel_values"].cuda()
            B = x.shape[0]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z_cls, z_s, patch = model._encode_z(x)   # (B,1,D)(B,N,D)(B,N,D)
                for s0 in range(0, S, args.window_chunk):
                    s1 = min(s0 + args.window_chunk, S)
                    C = s1 - s0
                    wins = torch.stack(
                        [z_s[:, s:s + args.window] for s in range(s0, s1)], 1)
                    zc = z_cls.unsqueeze(1).expand(B, C, 1, -1).reshape(B * C, 1, -1)
                    zw = wins.reshape(B * C, args.window, -1)
                    pp = patch.unsqueeze(1).expand(B, C, -1, -1).reshape(
                        B * C, num_patches, -1)
                    F_hat = model.decoder(zc, zw)        # (B*C,N,D)
                    # fp32 度量（bf16 分辨率不足以量化 ~0.001 的 L1）
                    l1 = (F_hat.float() - pp.float()).abs().mean(dim=(1, 2))
                    l1 = l1.reshape(B, C).mean(dim=0)    # (C,) 每窗口均值
                    sum_l1[s0:s1] += l1.cpu().numpy()
                    sum_l1sq[s0:s1] += (l1 ** 2).cpu().numpy()
            n += B
            if (bi + 1) % 10 == 0 or (bi + 1) == len(loader):
                print(f"  ... {n}/{len(loader.dataset)} ({time.time() - t0:.0f}s)",
                      flush=True)

    mean = sum_l1 / n
    std = np.sqrt(np.maximum(sum_l1sq / n - mean ** 2, 0.0))
    best = int(mean.argmin())
    # 头部/中部/尾部对比（"信息前置"的判据: 前段明显好于后段）
    head = mean[:min(16, S)]
    tail = mean[max(0, S - 16):]
    print(f"\n[window] 前段 {head.mean():.5f} | 后段 {tail.mean():.5f} | "
          f"最佳窗口起点 s={best} L1={mean[best]:.5f}")
    print(f"[window] 头部窗口: " + " ".join(f"{mean[s]:.5f}" for s in range(min(8, S))))
    print(f"[window] 尾部窗口: " + " ".join(
        f"{mean[s]:.5f}" for s in range(max(0, S - 8), S)))
    print(f"[time] {(time.time() - t0):.0f}s | {n} 图 | N={num_patches}, "
          f"window={args.window}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"n": n, "num_patches": num_patches, "input": [W, H],
                   "window": args.window, "mean_l1": mean.tolist(),
                   "std_l1": std.tolist(), "head_mean": float(head.mean()),
                   "tail_mean": float(tail.mean()), "best_start": best,
                   "time_s": round(time.time() - t0, 1)},
                  f, indent=2, ensure_ascii=False)
    print(f"[save] {args.output}")


def main():
    args = parse_args()
    W, H = (int(v) for v in args.model_input.lower().split("x"))
    num_patches = (W // 14) * (H // 14)

    # ── 模型: 训练好的重建权重（无 TextDecoder）──
    dino = Dinov2Model.from_pretrained(args.dino_dir)
    if getattr(dino.config, "use_mask_token", False):
        dino.config.use_mask_token = False
        del dino.embeddings.mask_token
    model = SRPhase1V2(dinov2=dino, num_patches=num_patches,
                       dim=dino.config.hidden_size)
    sd = torch.load(args.final_model, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected
    model.eval().cuda()

    # ── 数据: test 分片, 与训练同预处理（1600:900 画布 → 448x252）──
    test_files = sorted(glob.glob(os.path.join(args.data_dir, "test-*.parquet")))
    assert test_files, f"无 test-*.parquet in {args.data_dir}"
    coll = V2Collator(model_size=(W, H))
    ds = ParquetImageDataset(test_files, limit=args.limit)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=coll)
    n_total = len(ds)
    print(f"[data] {n_total} 条 test")

    # ── 滑窗探针模式（--window > 0）: 独立验证"信息前置"，不跑 k 扫描 ──
    if args.window > 0:
        assert 1 <= args.window <= num_patches, \
            f"window={args.window} 超出 [1,{num_patches}]"
        window_sweep(args, model, loader, num_patches, W, H)
        return

    # ── k 扫描: 0..kmax + 锚点 + 全量 N ──
    anchors = [int(v) for v in args.anchors.split(",") if v.strip()]
    ks = sorted(set([0] + list(range(1, args.kmax + 1)) + anchors
                    + [num_patches]))
    sum_l1 = {k: 0.0 for k in ks}
    sum_l1sq = {k: 0.0 for k in ks}
    n = 0
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x = batch["pixel_values"].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z_cls, z_s, patch = model._encode_z(x)
                for k in ks:
                    F_hat = model.decoder(z_cls, z_s[:, :k])
                    # fp32 度量（bf16 分辨率不足以量化 ~0.001 的 L1）
                    l1 = (F_hat.float() - patch.float()).abs().mean(dim=(1, 2))
                    sum_l1[k] += l1.sum().item()
                    sum_l1sq[k] += (l1 ** 2).sum().item()
            n += x.shape[0]
            if (bi + 1) % 10 == 0 or (bi + 1) == len(loader):
                print(f"  ... {n}/{n_total} ({time.time() - t0:.0f}s)",
                      flush=True)

    mean = {k: sum_l1[k] / n for k in ks}
    std = {k: ((sum_l1sq[k] / n) - mean[k] ** 2) ** 0.5 for k in ks}
    ref = mean[num_patches]                      # 全量 k=N 作基准
    print(f"\n{'k':>5} {'L1':>10} {'L1/L1(N)':>10} {'std':>8}  "
          f"{'注':>8}")
    for k in ks:
        note = "参考锚点(全量)" if k == num_patches else (
            "仅 z_cls" if k == 0 else "")
        print(f"{k:>5} {mean[k]:>10.5f} {mean[k] / ref:>10.3f} "
              f"{std[k]:>8.5f}  {note}")
    print(f"\n[time] {(time.time() - t0):.0f}s | {n} 图 | "
          f"模型输入 {W}x{H}, N={num_patches}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"n": n, "num_patches": num_patches, "input": [W, H],
                   "kmax": args.kmax, "mean_l1": mean, "std_l1": std,
                   "rel_to_full": {str(k): round(mean[k] / ref, 4) for k in ks},
                   "full_l1": ref, "time_s": round(time.time() - t0, 1)},
                  f, indent=2, ensure_ascii=False)
    print(f"[save] {args.output}")


if __name__ == "__main__":
    main()
