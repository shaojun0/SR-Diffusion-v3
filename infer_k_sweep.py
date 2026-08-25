"""
SR-Diffusion Phase 1 v2 — 还原精度 vs z_s 前缀长度 k 扫描（推理）
=================================================
只喂前 k 个 z_s（+z_cls）给 FeatureDecoder, 重建全量 N 个 DINO patch 特征,
度量 = L1(F_hat, patch)（与训练同一口径）。k=1..kmax 扫描, 附 k=0 与
全量 k=N 作参考锚点。

用法:
    python infer_k_sweep.py \
        --data_dir /root/autodl-tmp/construction_site \
        --dino_dir /root/autodl-tmp/models/dinov2-large \
        --final_model output/phase1_v2/final_model.pt \
        --output output/phase1_v2/k_sweep.json
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
    return p.parse_args()


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
