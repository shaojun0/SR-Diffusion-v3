"""
SR-Diffusion Phase 1 v4 — 推理测试（register K 压缩, 2026-08-28）
=================================================
架构: K 个 special token 进 DINO 序列(24 层直接算 z_s, 默认不冻结) →
    z_s (B,K,D) → v3 式无时序解码 → 像素 patch。无渐进曲线。
测试项（与 v3 同口径, 便于对照）:
    1) 全量重建像素 L1（归一化 + 0-255）—— 参照: v3(BLIP-2)=17.77 /
       v2=23.41 / 平均色≈61 / DINO 线性上限≈9.8;
    2) 双线性探针: z_s 均值 → 线性解码 L1（压缩表示均值的信息量）+
       解码器输出 h → 逐 patch 线性解码 L1（解码链路信息量, 定位瓶颈）。

用法:
    python infer_v4.py \
        --data_dir /root/autodl-tmp/construction_site \
        --dino_dir /root/autodl-tmp/models/dinov2-large \
        --final_model output/phase1_v4_registerK/final_model.pt \
        --output output/phase1_v4_registerK/infer_test.json
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
from model_v4 import SRPhase1V4


def parse_args():
    p = argparse.ArgumentParser(description="infer test for v4 (register K 压缩)")
    p.add_argument("--data_dir", required=True, help="parquet 目录(test-*.parquet)")
    p.add_argument("--dino_dir", default="models/dinov2-large")
    p.add_argument("--final_model", required=True, help="训练好的权重(final_model.pt)")
    p.add_argument("--output", default="output/phase1_v4_registerK/infer_test.json")
    p.add_argument("--model_input", default="448x252")
    p.add_argument("--canvas", default="1600x900")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="只用前 N 条 test(0=全量)")
    p.add_argument("--num_specials", type=int, default=64,
                   help="必须与训练一致(K)")
    p.add_argument("--decoder_depth", type=int, default=2)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--head_hidden", type=int, default=2048)
    p.add_argument("--freeze_dino", action="store_true",
                   help="训练时若 --freeze_dino, 推理也要带")
    p.add_argument("--probe_limit", type=int, default=128,
                   help="双线性探针用前 N 条(0=关闭)")
    return p.parse_args()


def _patch_to_img(pix_patches, H, W):
    """(B,N,588) 归一化像素 patch → (B,H,W,3) 反归一化 0-255 图像。
    布局与 DINO 一致: row-major (先 y 后 x)。支持 torch 张量或 numpy 数组。"""
    is_np = isinstance(pix_patches, np.ndarray)
    if is_np:
        pix_patches = torch.from_numpy(pix_patches)
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

    # ── 模型 ──
    dino = Dinov2Model.from_pretrained(args.dino_dir)
    if getattr(dino.config, "use_mask_token", False):
        dino.config.use_mask_token = False
        del dino.embeddings.mask_token
    model = SRPhase1V4(dinov2=dino, num_patches=num_patches,
                       dim=dino.config.hidden_size,
                       num_specials=args.num_specials,
                       decoder_depth=args.decoder_depth,
                       heads=args.heads, mlp_ratio=args.mlp_ratio,
                       freeze_dino=args.freeze_dino,
                       head_hidden=args.head_hidden)
    sd = torch.load(args.final_model, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval().cuda()
    print(f"[model] loaded {args.final_model}: N={num_patches}, "
          f"K={model.num_specials}, freeze_dino={model.freeze_dino}")

    # ── 数据 ──
    test_files = sorted(glob.glob(os.path.join(args.data_dir, "test-*.parquet")))
    assert test_files, f"无 test-*.parquet in {args.data_dir}"
    coll = V2Collator(model_size=(W, H), canvas=(1600, 900))
    ds = ParquetImageDataset(test_files, limit=args.limit)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=coll)
    n_total = len(ds)
    print(f"[data] {n_total} 条 test")

    # ── 前向: 全量像素 L1 ──
    norm_sum, norm_sq, pix_sum, pix_sq, n = 0.0, 0.0, 0.0, 0.0, 0
    Z_all, H_all, pix_all = [], [], []   # 双探针收集（前 probe_limit 条）
    probe_n = 0
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x = batch["pixel_values"].cuda()            # (B,3,H,W) 归一化
            B, C, Hh, Ww = x.shape
            out = model(x)
            pixels = out["pixels"]                      # (B,N,588)
            target = out["target"]                      # (B,N,588)
            l1_norm = (pixels - target).abs().mean(dim=(1, 2))          # (B,)
            norm_sum += l1_norm.sum().item()
            norm_sq += (l1_norm ** 2).sum().item()
            gt_255 = _patch_to_img(target, Hh, Ww)      # (B,H,W,3)
            recon_255 = _patch_to_img(pixels, Hh, Ww)
            l1_pix = np.abs(recon_255 - gt_255).mean(axis=(1, 2, 3))   # (B,)
            pix_sum += l1_pix.sum().item()
            pix_sq += (l1_pix ** 2).sum().item()
            # 双探针: 只收前 probe_limit 条
            if args.probe_limit > 0 and probe_n < args.probe_limit:
                take = min(B, args.probe_limit - probe_n)
                Z_all.append(out["z_s"][:take].float().cpu().numpy())
                # 解码器输出 h: 复刻 model.forward 内部 (无像素头)
                zs = out["z_s"][:take]
                q = model.decoder.query_base.expand(zs.shape[0], -1, -1)
                for layer in model.decoder.layers:
                    q = q + layer.cross_attn(layer.norm1(q), zs, zs,
                                             need_weights=False)[0]
                    q = q + layer.ffn(layer.norm2(q))
                H_all.append(q.float().cpu().numpy())   # (take,N,D) 解码器输出特征
                pix_all.append(target[:take].float().cpu().numpy())
                probe_n += take
            n += B
            if (bi + 1) % 10 == 0 or (bi + 1) == len(loader):
                print(f"  ... {n}/{n_total} ({time.time() - t0:.0f}s)", flush=True)

    norm_mean = norm_sum / n
    pix_mean = pix_sum / n
    pix_std = np.sqrt(max(pix_sq / n - pix_mean ** 2, 0.0))

    print(f"\n[full] 全量重建像素 L1 (归一化空间) = {norm_mean:.6f}")
    print(f"[full] 全量重建像素 L1 (0-255 空间) = {pix_mean:.2f} ± {pix_std:.2f}")
    print(f"       参照: v3(BLIP-2,K=64,冻结)=17.77 | v2=23.41 | 平均色≈61 | "
          f"DINO 线性上限≈9.8")

    # ── 双探针（口径与 v3 一致, 见 infer_v3.py 注释）──
    probe = None
    if args.probe_limit > 0 and probe_n > 0:
        Z = np.concatenate(Z_all)                        # (M,K,D)
        Hh = np.concatenate(H_all)                       # (M,N,D)
        pix = np.concatenate(pix_all)                    # (M,N,588)
        z_mean = Z.mean(axis=1, keepdims=True)           # (M,1,D)
        X = np.repeat(z_mean, num_patches, axis=1).reshape(-1, Z.shape[-1])
        Wm, *_ = np.linalg.lstsq(X, pix.reshape(-1, PATCH_PX), rcond=1e-6)
        recon = (X @ Wm).reshape(probe_n, num_patches, PATCH_PX)
        rimg = _patch_to_img(recon, H, W)
        gimg = _patch_to_img(pix, H, W)
        probe_l1 = float(np.abs(rimg - gimg).mean())
        within_std = float(Z.std(axis=1).mean())         # z_s 跨位置变异性
        Xh = Hh.reshape(-1, Hh.shape[-1])                # (M*N,D) 逐 patch
        Wh, *_ = np.linalg.lstsq(Xh, pix.reshape(-1, PATCH_PX), rcond=1e-6)
        recon_h = (Xh @ Wh).reshape(probe_n, num_patches, PATCH_PX)
        rimg_h = _patch_to_img(recon_h, H, W)
        h_l1 = float(np.abs(rimg_h - gimg).mean())
        h_within_std = float(Hh.std(axis=1).mean())      # h 跨位置变异性
        probe = {"n": probe_n, "z_s_mean_linear_l1_255": probe_l1,
                 "z_s_within_std": within_std,
                 "decoder_h_linear_l1_255": h_l1,
                 "decoder_h_within_std": h_within_std}
        print(f"[probe] K={model.num_specials} 双探针 ({probe_n} 条):")
        print(f"        z_s 均值 → 线性解码 L1 = {probe_l1:.2f} "
              f"(压缩表示均值的信息量)")
        print(f"        z_s within-std = {within_std:.4f} (K 个 token 间变异性)")
        print(f"        解码器 h → 线性解码 L1 = {h_l1:.2f} "
              f"(对照: ≈全量 ⇒ 解码链路 OK, 瓶颈在压缩表示本身)")
        print(f"        h within-std = {h_within_std:.4f} (N 个 patch 输出变异性)")

    print(f"[time] {(time.time() - t0):.0f}s | {n} 图")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "n": n, "num_patches": num_patches, "num_specials": model.num_specials,
            "input": [W, H], "freeze_dino": model.freeze_dino,
            "full_norm_l1": float(norm_mean),
            "full_pixel_l1_255": float(pix_mean), "full_pixel_std_255": float(pix_std),
            "probe": probe,
            "time_s": round(time.time() - t0, 1),
        }, f, indent=2, ensure_ascii=False)
    print(f"[save] {args.output}")


if __name__ == "__main__":
    main()
