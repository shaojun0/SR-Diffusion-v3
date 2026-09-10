"""
probe_step_collapse.py — 测 step1~5 是否还坍缩（blockdiag vs causal）

判据口径与 doc/2026-09-07/ab_compare.py **完全一致**，便于直接对比历史数字：
  step_px_scale[t] = mean |pixel_head(Y_t)|      Y_t = 未累加的原始增量（坍缩主判据）
  prog_curve[t]    = 0-255 空间第 t 步累积结果的整图 L1（渐进曲线）
  E_px[t][r]       = 第 t 步累积结果在第 r 个目标区域(rows)上的 0-255 L1
  E_nrm[t][r]      = 同上，归一化空间

历史参照（causal, slice[0:5], K=35, 2026-09-04）:
  step_px_scale = [1.007, 0.063, 0.058, 0.057, 0.056]   ← 后 4 步只有 Y1 的 5.6–6.3%
  prog_curve    = [20.5587, 20.5489, 20.5456, 20.5476, 20.5544]

用法（cwd = 含 model_v2/data_v2 的代码目录）:
  python probe_step_collapse.py --ckpt <path/final_model.pt> --tag blockdiag \
      --query_mask_mode blockdiag --limit 512 --out probe_blockdiag.json
"""
import argparse, glob, json, os, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Dinov2Model

from data_v2 import ParquetImageDataset, V2Collator, DINO_MEAN, DINO_STD
import model_v2 as M
from model_v2 import SRPhase1V2

W, H, N, PX = 448, 252, 576, 588
STEPS = [1, 4, 9, 16, 25]
K = 35

# register 分块边界: 每步读自己那块 z_s
# step 1→3, 4→5, 9→7, 16→9, 25→11  ⇒ 累计 [0:3],[3:8],[8:15],[15:24],[24:35]
BLOCKS = [(0, 3), (3, 8), (8, 15), (15, 24), (24, 35)]
# 目标区域: 576 patch 均分 5 区。与 ab_compare.py 里 M.region_slices(576, 5)
# 的输出逐项相同 —— 该函数随 region_loss 一起被还原, 故此处内联,
# 保证与历史 E_px 矩阵同口径。
REGIONS = [(0, 115), (115, 230), (230, 345), (345, 460), (460, 576)]


def denorm_clip(patch_tensor):
    """(B,N,588) 归一化 patch → 0-255 clip。patch 布局 (C=3,14,14) 展平,
    故每通道统计量 repeat(196) 成 588 向量。"""
    dev = patch_tensor.device
    std = torch.as_tensor(DINO_STD, device=dev).repeat(196).view(1, 1, -1)
    mean = torch.as_tensor(DINO_MEAN, device=dev).repeat(196).view(1, 1, -1)
    return torch.clamp(patch_tensor * std + mean, 0, 255)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--query_mask_mode", default=None,
                    help="不传 = 按 model_info.json 解析（无字段则 causal）")
    ap.add_argument("--data_dir", default="/root/autodl-tmp/construction_site")
    ap.add_argument("--dino_dir", default="/root/autodl-tmp/models/dinov2-large")
    ap.add_argument("--limit", type=int, default=512)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # ── query_mask_mode 解析: model_info.json 优先 → CLI → causal ──
    info_path = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)),
                             "model_info.json")
    info = None
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
    if info is not None and "query_mask_mode" in info:
        qmm = str(info["query_mask_mode"])
        if args.query_mask_mode and args.query_mask_mode != qmm:
            print(f"[warn] model_info.json={qmm} 与 --query_mask_mode="
                  f"{args.query_mask_mode} 不一致: 以 model_info 为准")
    elif args.query_mask_mode:
        qmm = args.query_mask_mode
    else:
        qmm = "causal"
        print("[info] 无 model_info.json 记录 → fallback causal（旧产物口径）")
    print(f"[cfg] tag={args.tag} query_mask_mode={qmm} ckpt={args.ckpt}")

    T, R = len(STEPS), len(BLOCKS)
    assert len(REGIONS) == T and REGIONS[-1][1] == N, (REGIONS, N)

    dino = Dinov2Model.from_pretrained(args.dino_dir)
    if getattr(dino.config, "use_mask_token", False):
        dino.config.use_mask_token = False
        del dino.embeddings.mask_token
    model = SRPhase1V2(dinov2=dino, num_patches=N, dim=dino.config.hidden_size,
                       heads=8, mlp_ratio=4.0, decoder_steps=STEPS,
                       decoder_depth=2, skip_steps=0, max_steps=5,
                       num_specials=K, query_mask_mode=qmm)
    sd = torch.load(args.ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval().cuda()
    pixel_head = model.pixel_head
    print(f"[model] K={model.num_specials} steps={model.decoder.steps} "
          f"mode={model.query_mask_mode}")

    files = sorted(glob.glob(os.path.join(args.data_dir, "test-*.parquet")))
    ds = ParquetImageDataset(files, limit=args.limit)
    coll = V2Collator(model_size=(W, H), canvas=(1600, 900))
    loader = DataLoader(ds, batch_size=args.bs, shuffle=False,
                        num_workers=args.num_workers, collate_fn=coll)
    print(f"[data] {len(ds)} 张 test (limit={args.limit})", flush=True)

    scale_sum = np.zeros(T)
    nrm_sum = np.zeros((T, R))
    px_sum = np.zeros((T, R))
    n_img = 0
    zstd_sum = 0.0
    blk_cos = np.zeros((R,)); blk_cnt = 0
    t0 = time.time()

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x = batch["pixel_values"].cuda()
            Bc, C, Hh, Ww = x.shape
            z_cls, z_s = model.encode(x)

            # z_s 结构（次要判据, README §4.1 已声明 cos 不作判据, 仅记录）
            zstd_sum += float(z_s.std(dim=1).mean().item()) * Bc
            zn = z_s / z_s.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            for r, (lo, hi) in enumerate(BLOCKS):
                sub = zn[:, lo:hi]
                m = hi - lo
                cs = sub @ sub.transpose(1, 2)
                off = cs.sum(dim=(1, 2)) - cs.diagonal(dim1=1, dim2=2).sum(dim=1)
                blk_cos[r] += float((off / (m * (m - 1))).sum().item()) if m > 1 else 0.0
            blk_cnt += Bc

            # 解码: 拿未累加增量口径
            Y = model.decoder(z_cls, z_s)               # (B,T,N,D)
            Y_pix_raw = pixel_head(Y)                   # 原始增量(不过累加)
            Y_cum = torch.cat([torch.zeros_like(Y[:, :1]),
                               Y.cumsum(dim=1)[:, :-1]], dim=1).detach() + Y
            Y_pix = pixel_head(Y_cum)                   # 累加像素

            target = x.reshape(Bc, C, Hh // 14, 14, Ww // 14, 14) \
                      .permute(0, 2, 4, 1, 3, 5).reshape(Bc, N, PX)
            Y255 = denorm_clip(Y_pix)
            T255 = denorm_clip(target)

            for i in range(T):
                scale_sum[i] += float(Y_pix_raw[:, i].abs().mean().item()) * Bc
                for r, (lo, hi) in enumerate(REGIONS):
                    nrm_sum[i, r] += float((Y_pix[:, i, lo:hi] - target[:, lo:hi])
                                           .abs().sum().item())
                    px_sum[i, r] += float((Y255[:, i, lo:hi] - T255[:, lo:hi])
                                          .abs().sum().item())
            n_img += Bc
            if (bi + 1) % 8 == 0:
                print(f"  batch {bi+1} ({time.time()-t0:.0f}s)", flush=True)
            del Y, Y_cum, Y_pix, Y_pix_raw, z_cls, z_s
            torch.cuda.empty_cache()

    rows_r = [hi - lo for lo, hi in REGIONS]
    E_px = [[px_sum[i, r] / (n_img * rows_r[r] * PX) for r in range(R)]
            for i in range(T)]
    E_nrm = [[nrm_sum[i, r] / (n_img * rows_r[r] * PX) for r in range(R)]
             for i in range(T)]
    res = {
        "tag": args.tag, "ckpt": args.ckpt,
        "query_mask_mode": qmm, "n_img": n_img,
        "steps": STEPS, "num_specials": K,
        "step_px_scale": [float(scale_sum[i] / n_img) for i in range(T)],
        "prog_curve_255": [float(sum(E_px[i][r] * rows_r[r] for r in range(R))
                                 / N) for i in range(T)],
        "E_px": E_px, "E_nrm": E_nrm,
        "regions": REGIONS, "blocks": BLOCKS,
        "z_s_within_std": float(zstd_sum / n_img),
        "z_s_block_cos": [float(v / blk_cnt) for v in blk_cos],
    }
    print(f"\n[{args.tag}] step_px_scale = "
          f"{[round(v, 4) for v in res['step_px_scale']]}")
    print(f"[{args.tag}] prog_curve(0-255) = "
          f"{[round(v, 4) for v in res['prog_curve_255']]}")
    print(f"[{args.tag}] z_s within_std = {res['z_s_within_std']:.4f} | "
          f"block cos = {[round(v,4) for v in res['z_s_block_cos']]}")
    print(f"[{args.tag}] E_px (行=step, 列=region):")
    for i in range(T):
        print(f"    step{i+1:<2} " + " ".join(f"{E_px[i][r]:7.2f}" for r in range(R)))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
