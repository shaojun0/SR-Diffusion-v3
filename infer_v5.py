"""
SR-Diffusion Phase 1 v5 — 推理测试（扩散式渐进细化, 2026-08-31）
=================================================
测试项:
    1) 全量 DDIM 重建像素 L1（归一化 + 0-255）—— 注意: 这是"条件生成"
       探针（从噪声出发, token 驱动还原）, 与 v3/v4 的确定性重建口径
       不同, 数值不可直接对比（参照: v3=17.77 / v4=8.26 / 平均色≈61）。
    2) 渐进阶梯曲线: DDIM 每反向步的 (t, m, L1) —— 若思想成立（token 渐进
       解锁补细节）, L1 应随 m 增大呈阶梯下降; 若平台 ⇒ token 无增量性。
    3) m 扫描探针（增量性直接判据）: 固定噪声水平 t, 同一噪声 ε 下逐量
       解锁 token（m=0..K）, 看 L1 是否随 m 单调下降 —— "每个 token 是否
       补了新信息"。
    4) token 消融: 全程无条件（cond=False, m=0）vs 有条件 DDIM —— 差距
       = token 的价值（活信息保真）。

用法:
    python infer_v5.py \
        --data_dir /root/autodl-tmp/construction_site \
        --dino_dir /root/autodl-tmp/models/dinov2-large \
        --final_model output/phase1_v5_diffusion/final_model.pt \
        --output output/phase1_v5_diffusion/infer_test.json
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
from model_v5 import SRPhase1V5


def parse_args():
    p = argparse.ArgumentParser(description="infer test for v5 (扩散式渐进细化)")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--dino_dir", default="models/dinov2-large")
    p.add_argument("--final_model", required=True)
    p.add_argument("--output", default="output/phase1_v5_diffusion/infer_test.json")
    p.add_argument("--model_input", default="448x252")
    p.add_argument("--batch_size", type=int, default=8,
                   help="DDIM 反向 100 步 + 阶梯记录, batch 小一点稳")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="只用前 N 条 test(0=全量)")
    # ── 必须与训练一致 ──
    p.add_argument("--num_specials", type=int, default=128)
    p.add_argument("--diffusion_steps", type=int, default=1000)
    p.add_argument("--decoder_depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=2.0)
    p.add_argument("--freeze_dino", action="store_true")
    p.add_argument("--unlock", default="linear")
    # ── 推理 ──
    p.add_argument("--ddim_steps", type=int, default=100,
                   help="DDIM 反向步数（阶梯曲线点数）")
    p.add_argument("--cfg_scale", type=float, default=1.0,
                   help="CFG 引导强度(>1 增强 token 作用; 默认 1.0 确定性)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--probe_limit", type=int, default=128,
                   help="m 扫描探针用前 N 条(0=关闭)")
    p.add_argument("--sweep_m", default="0,16,32,48,64,80,96,112,128",
                   help="m 扫描的解锁数(逗号分隔, 0..K)")
    p.add_argument("--sweep_ts", default="250,500,750",
                   help="m 扫描的固定噪声步(逗号分隔, 1..T)")
    return p.parse_args()


def _patch_to_img(pix_patches, H, W):
    """(B,N,588) 归一化像素 patch → (B,H,W,3) 反归一化 0-255 图像。"""
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
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    W, H = (int(v) for v in args.model_input.lower().split("x"))
    num_patches = (W // 14) * (H // 14)
    PATCH_PX = 14 * 14 * 3
    sweep_m = [int(v) for v in args.sweep_m.split(",") if v.strip()]
    sweep_ts = [int(v) for v in args.sweep_ts.split(",") if v.strip()]
    assert all(0 <= m <= args.num_specials for m in sweep_m), "m 越界"
    assert all(1 <= t <= args.diffusion_steps for t in sweep_ts), "t 越界"

    # ── 模型 ──
    dino = Dinov2Model.from_pretrained(args.dino_dir)
    if getattr(dino.config, "use_mask_token", False):
        dino.config.use_mask_token = False
        del dino.embeddings.mask_token
    model = SRPhase1V5(
        dinov2=dino, num_patches=num_patches, dim=dino.config.hidden_size,
        num_specials=args.num_specials,
        diffusion_steps=args.diffusion_steps,
        decoder_depth=args.decoder_depth, heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        freeze_dino=args.freeze_dino, unlock=args.unlock)
    sd = torch.load(args.final_model, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval().cuda()
    print(f"[model] loaded {args.final_model}: N={num_patches}, "
          f"K={model.num_specials}, "
          f"T={model.T}, unlock={model.unlock}, "
          f"freeze_dino={model.freeze_dino}")

    # ── 数据 ──
    test_files = sorted(glob.glob(os.path.join(args.data_dir, "test-*.parquet")))
    assert test_files, f"无 test-*.parquet in {args.data_dir}"
    coll = V2Collator(model_size=(W, H), canvas=(1600, 900))
    ds = ParquetImageDataset(test_files, limit=args.limit)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=coll)
    n_total = len(ds)
    print(f"[data] {n_total} 条 test | DDIM {args.ddim_steps} 步 | "
          f"cfg_scale={args.cfg_scale}")

    # ── 1) 全量 DDIM 重建 + 2) 渐进阶梯曲线 + 4) 消融 ──
    norm_sum, pix_sum, pix_sq, n = 0.0, 0.0, 0.0, 0
    norm_sum_u, pix_sum_u, n_u = 0.0, 0.0, 0      # 无条件消融（前 probe_limit 条）
    curve_acc = {}                                 # (t,m) → [l1,...] 累积
    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x = batch["pixel_values"].cuda()
            B, C, Hh, Ww = x.shape
            res = model.sample(x, steps=args.ddim_steps,
                               cfg_scale=args.cfg_scale,
                               record_curve=True)
            pixels = res["pixels"]                 # (B,N,588)
            target = res["target"]
            # 阶梯曲线: 每图独立记录, 按 (t,m) 聚合
            for ci, (t, m, l1) in enumerate(res["curve"]):
                curve_acc.setdefault((t, m), []).append(l1)
            l1_norm = (pixels - target).abs().mean(dim=(1, 2))      # (B,)
            norm_sum += l1_norm.sum().item()
            gt_255 = _patch_to_img(target, Hh, Ww)
            recon_255 = _patch_to_img(pixels, Hh, Ww)
            l1_pix = np.abs(recon_255 - gt_255).mean(axis=(1, 2, 3))  # (B,)
            pix_sum += l1_pix.sum().item()
            pix_sq += (l1_pix ** 2).sum().item()
            n += B
            # 无条件消融（token 价值）: 前 probe_limit 条
            if args.probe_limit > 0 and n_u < args.probe_limit:
                take = min(B, args.probe_limit - n_u)
                res_u = model.sample(x[:take], steps=args.ddim_steps,
                                     cfg_scale=1.0, cond=False,
                                     record_curve=False, seed=args.seed)
                l1_u = (res_u["pixels"] - res_u["target"]).abs() \
                    .mean(dim=(1, 2)).sum().item()
                norm_sum_u += l1_u
                n_u += take
            if (bi + 1) % 5 == 0 or (bi + 1) == len(loader):
                print(f"  ... {n}/{n_total} ({time.time() - t0:.0f}s)",
                      flush=True)

    norm_mean = norm_sum / n
    pix_mean = pix_sum / n
    pix_std = np.sqrt(max(pix_sq / n - pix_mean ** 2, 0.0))
    uncond_mean = (norm_sum_u / n_u) if n_u else None

    print(f"\n[full] 全量 DDIM 重建像素 L1 (归一化) = {norm_mean:.6f}")
    print(f"[full] 全量 DDIM 重建像素 L1 (0-255) = {pix_mean:.2f} ± {pix_std:.2f}")
    print(f"       口径=条件生成(从噪声出发), 与 v3=17.77/v4=8.26 不可直接对比; "
          f"平均色≈61")
    if uncond_mean is not None:
        gap = norm_mean - uncond_mean
        print(f"[ablat] token 消融: 无条件 DDIM L1 = {uncond_mean:.6f} "
              f"(条件-无条件差 = {gap:.4f} → token 的价值)")

    # ── 2) 渐进阶梯曲线: 按 m 排序输出 ──
    curve = sorted([{"t": t, "m": m,
                     "l1_norm": float(np.mean(v)),
                     "n": len(v)}
                    for (t, m), v in curve_acc.items()],
                   key=lambda c: (c["m"], -c["t"]))
    print(f"\n[stair] 渐进阶梯曲线 (共 {len(curve)} 个采样点):")
    for c in curve:
        print(f"        t={c['t']:4d}  m={c['m']:3d}  L1={c['l1_norm']:.4f} "
              f"(n={c['n']})")
    # 阶梯判定: 按 m 分组的末段均值（m 越大 L1 应越低）
    by_m = {}
    for c in curve:
        by_m.setdefault(c["m"], []).append(c["l1_norm"])
    stair_check = {m: float(np.mean(v)) for m, v in sorted(by_m.items())}
    print(f"[stair] 按 m 聚合平均 L1: {stair_check}")
    if len(stair_check) >= 2:
        m_keys = sorted(stair_check)
        drops = [stair_check[m_keys[i + 1]] - stair_check[m_keys[i]]
                 for i in range(len(m_keys) - 1)]
        print(f"[stair] 相邻 m 的 L1 变化: {['%.4f' % d for d in drops]} "
              f"({'阶梯下降' if all(d <= 1e-9 for d in drops) else '非单调, 平台风险'})")

    # ── 3) m 扫描探针: 固定噪声 + 同一 ε, 逐量解锁 token ──
    sweep = None
    if args.probe_limit > 0:
        sweep = {"ts": sweep_ts, "m": sweep_m,
                 "l1_by_m": {f"t={t}": [] for t in sweep_ts}}
        probe_n = 0
        with torch.no_grad():
            for batch in loader:
                x = batch["pixel_values"].cuda()
                B = x.shape[0]
                take = min(B, args.probe_limit - probe_n)
                if take <= 0:
                    break
                xb = x[:take]
                x0 = model._patches(xb)
                eps = torch.randn_like(x0)          # 每图固定 ε, 跨 m 复用
                for t in sweep_ts:
                    l1s = []
                    for m in sweep_m:
                        xh, _, _ = model.predict_at(xb, t=t, m=m, eps=eps)
                        l1s.append(float(F_l1(xh, x0)))
                    sweep["l1_by_m"][f"t={t}"].append(l1s)
                probe_n += take
                if probe_n >= args.probe_limit:
                    break
        for t in sweep_ts:
            arr = np.mean(sweep["l1_by_m"][f"t={t}"], axis=0)   # (len(m),)
            sweep["l1_by_m"][f"t={t}"] = [float(v) for v in arr]
        print(f"\n[sweep] m 扫描 (固定噪声, 同一 ε, {probe_n} 图平均; "
              f"L1 随 m 下降 = token 增量性):")
        hdr = "        m: " + "  ".join(f"{m:>7d}" for m in sweep_m)
        print(hdr)
        for t in sweep_ts:
            row = "  ".join(f"{v:>7.4f}" for v in sweep["l1_by_m"][f"t={t}"])
            print(f"        t={t:4d}: {row}")

    print(f"[time] {(time.time() - t0):.0f}s | {n} 图")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "n": n, "num_patches": num_patches,
            "num_specials": model.num_specials, "T": model.T,
            "input": [W, H], "freeze_dino": model.freeze_dino,
            "unlock": model.unlock, "ddim_steps": args.ddim_steps,
            "cfg_scale": args.cfg_scale, "seed": args.seed,
            "full_norm_l1": float(norm_mean),
            "full_pixel_l1_255": float(pix_mean),
            "full_pixel_std_255": float(pix_std),
            "uncond_norm_l1": uncond_mean,
            "probe": sweep,
            "staircase": curve,
            "stair_by_m": stair_check,
            "time_s": round(time.time() - t0, 1),
        }, f, indent=2, ensure_ascii=False)
    print(f"[save] {args.output}")


def F_l1(a, b):
    return float((a - b).abs().mean(dim=(1, 2)).mean().detach())


if __name__ == "__main__":
    main()
