"""
ab_compare.py — region_loss A/B 终版对比 (零训练, 全量 3004 test, 单卡)
A: phase1_v2_slice27_v2_regionloss  (region_loss=True,  新分区损失)
B: phase1_v2_slice27_v2_fullloss    (region_loss=False, 旧全图损失=slice27_v2 复现锚点)
每模型输出(512 口径? 不, 全量 3004):
  - 完整模型解码: full_norm_l1 / full_pixel_l1_255(0-255) / 5 步渐进曲线 / step_px_scale(每步自身输出量级)
  - 分区矩阵 E_px[i][r]: 第 i 步累加结果在第 r 个目标区域(rows)上的 0-255 L1
    → 读法: 对角趋势=每步覆盖自己区域(阶梯); i>r 行上升=后步污染前区(F1 风险); 行 i 覆盖 r>i 若很低=前步已覆盖后区(异常)
  - z_s 结构: within_std + head(1..15)/tail(16..63) 平均非对角 cos
用法(cwd = 含 model_v2/data_v2 的代码目录):
  CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/python ab_compare.py \
      --a output/phase1_v2_slice27_v2_regionloss/final_model.pt \
      --b output/phase1_v2_slice27_v2_fullloss/final_model.pt \
      --out compare_ab.json
"""
import argparse, json, time, numpy as np, torch
from torch.utils.data import DataLoader
from transformers import Dinov2Model
from data_v2 import ParquetImageDataset, V2Collator, DINO_MEAN, DINO_STD
import model_v2 as M
from model_v2 import SRPhase1V2

W, H, N, PX = 448, 252, 576, 588
STEPS = [9, 16, 25, 36, 49]
K = 63
REGIONS = M.region_slices(N, len(STEPS))          # [(0,115),(115,230),(230,345),(345,460),(460,576)]
# z_s 列 j ↔ register 位置 j+1 (K=63 → 列 0..62)
HEAD = list(range(0, 15))                          # register 位置 1..15 (step-1 读区)
TAIL = list(range(15, 63))                         # register 位置 16..63


def denorm_clip(patch_tensor):
    """(B,N,588) 归一化 patch → 0-255 clip。patch 向量布局 = (C=3,14,14) 展平,
    故每通道统计量须 repeat(196) 成 588 向量。"""
    dev = patch_tensor.device
    std = torch.as_tensor(DINO_STD, device=dev).repeat(196).view(1, 1, -1)
    mean = torch.as_tensor(DINO_MEAN, device=dev).repeat(196).view(1, 1, -1)
    return torch.clamp(patch_tensor * std + mean, 0, 255)


def analyze(model_path, tag, dino_dir, data_dir, out_all, limit=0, bs=16):
    print(f"\n===== {tag}: {model_path} =====", flush=True)
    dino = Dinov2Model.from_pretrained(dino_dir)
    if getattr(dino.config, "use_mask_token", False):
        dino.config.use_mask_token = False
        del dino.embeddings.mask_token
    model = SRPhase1V2(dinov2=dino, num_patches=N, dim=dino.config.hidden_size,
                       heads=8, mlp_ratio=4.0, decoder_steps=STEPS,
                       decoder_depth=2, num_specials=K)
    sd = torch.load(model_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval().cuda()
    pixel_head = model.pixel_head

    files = sorted(glob_import(data_dir, "test-*.parquet"))
    coll = V2Collator(model_size=(W, H), canvas=(1600, 900))
    ds = ParquetImageDataset(files, limit=limit)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=8, collate_fn=coll)
    print(f"[data] {len(ds)} 张 test", flush=True)

    T = len(STEPS); R = len(REGIONS)
    rows_r = [hi - lo for lo, hi in REGIONS]
    # 累加器: 0-255-sum 与 norm-sum, 均 (步,区域); 计数恒定, 最后用
    # n_img * rows_r * 588 一次除（避免 per-batch 覆盖 bug）
    nrm_sum = np.zeros((T, R))
    px_sum = np.zeros((T, R))
    scale_sum = np.zeros(T); scale_cnt = 0.0
    zstd_sum = 0.0; zstd_cnt = 0
    cos_head_sum = cos_tail_sum = cos_ht_sum = 0.0; cos_cnt = 0

    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x = batch["pixel_values"].cuda()
            Bc, C, Hh, Ww = x.shape
            z_cls, z_s = model.encode(x)
            # z_s 统计(逐图)
            zn = z_s / z_s.norm(dim=-1, keepdim=True).clamp_min(1e-8)   # (B,K,D)
            zstd_sum += float(z_s.std(dim=1).mean(dim=(0, 1)).sum().item()) * Bc
            zstd_cnt += Bc
            cs_h = zn[:, HEAD] @ zn[:, HEAD].transpose(1, 2)            # (B,m,m)
            m = len(HEAD)
            off_h = cs_h.sum(dim=(1, 2)) - cs_h.diagonal(dim1=1, dim2=2).sum(dim=1)
            cos_head_sum += float((off_h / (m * (m - 1))).sum().item())
            cs_t = zn[:, TAIL] @ zn[:, TAIL].transpose(1, 2)
            mt = len(TAIL)
            off_t = cs_t.sum(dim=(1, 2)) - cs_t.diagonal(dim1=1, dim2=2).sum(dim=1)
            cos_tail_sum += float((off_t / (mt * (mt - 1))).sum().item())
            cs_ht = zn[:, HEAD] @ zn[:, TAIL].transpose(1, 2)
            cos_ht_sum += float(cs_ht.mean(dim=(1, 2)).sum().item())
            cos_cnt += Bc
            # 解码
            Y = model.decoder(z_cls, z_s)
            Y_cum = torch.cat([torch.zeros_like(Y[:, :1]),
                               Y.cumsum(dim=1)[:, :-1]], dim=1).detach() + Y
            Y_pix = pixel_head(Y_cum)                    # (B,T,N,588) norm
            Y_pix_raw = pixel_head(Y)                    # 原始增量(不过累加)
            target = x.reshape(Bc, C, Hh // 14, 14, Ww // 14, 14) \
                       .permute(0, 2, 4, 1, 3, 5).reshape(Bc, N, PX)
            # 0-255 空间(带 clip)与归一化空间的 (步×区域) L1
            Y255 = denorm_clip(Y_pix)
            T255 = denorm_clip(target)
            for i in range(T):
                scale_sum[i] += float(Y_pix_raw[:, i].abs().mean().item()) * Bc
                for r, (lo, hi) in enumerate(REGIONS):
                    nrm_sum[i, r] += float((Y_pix[:, i, lo:hi] - target[:, lo:hi])
                                           .abs().sum().item())
                    px_sum[i, r] += float((Y255[:, i, lo:hi] - T255[:, lo:hi])
                                          .abs().sum().item())
            scale_cnt += Bc
            if (bi + 1) % 16 == 0:
                print(f"  {tag} batch {bi+1} ({time.time()-t0:.0f}s)", flush=True)
            del Y, Y_cum, Y_pix, Y_pix_raw, z_cls, z_s
            torch.cuda.empty_cache()

    n_img = scale_cnt
    # 每(步,区域)均值 = 总和 / (n_img * rows_r * 588)
    pxm = px_sum / (n_img * np.array(rows_r, dtype=np.float64)[None, :] * PX)
    nrmm = nrm_sum / (n_img * np.array(rows_r, dtype=np.float64)[None, :] * PX)
    res = {"tag": tag, "n": n_img,
           "full_norm_l1": float(nrmm[-1].sum()),
           "full_pixel_l1_255": float(pxm[-1].sum()),
           "step_norm_l1": [float(nrmm[i].sum()) for i in range(T)],
           "step_pixel_l1_255": [float(pxm[i].sum()) for i in range(T)],
           "step_px_scale": [float(scale_sum[i] / scale_cnt) for i in range(T)],
           "region_pixel_l1_255": [[float(pxm[i, r]) for r in range(R)]
                                   for i in range(T)],
           "region_norm_l1": [[float(nrmm[i, r]) for r in range(R)]
                              for i in range(T)],
           "regions": REGIONS,
           "z_within_std": float(zstd_sum / zstd_cnt),
           "z_mean_cos_head1_15": float(cos_head_sum / cos_cnt),
           "z_mean_cos_tail16_63": float(cos_tail_sum / cos_cnt),
           "z_mean_cos_head_tail": float(cos_ht_sum / cos_cnt),
           "time_s": round(time.time() - t0, 1)}
    print(f"[full] L1_px={res['full_pixel_l1_255']:.3f} norm={res['full_norm_l1']:.4f}", flush=True)
    print(f"[curve px] {[round(v,3) for v in res['step_pixel_l1_255']]}", flush=True)
    print(f"[step px_scale] {[round(v,4) for v in res['step_px_scale']]}", flush=True)
    print(f"[z] within_std={res['z_within_std']:.4f} cos_head={res['z_mean_cos_head1_15']:.4f} "
          f"cos_tail={res['z_mean_cos_tail16_63']:.4f} cos_ht={res['z_mean_cos_head_tail']:.4f}", flush=True)
    print(f"[region px L1 步x区域] rows=区域 idx; diag 应对应阶梯/覆盖", flush=True)
    for i in range(T):
        print(f"   步{i+1}: {[round(v,2) for v in res['region_pixel_l1_255'][i]]}", flush=True)
    out_all[tag] = res
    return res


def glob_import(data_dir, pat):
    import glob
    return sorted(glob.glob(data_dir + "/" + pat))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", default="compare_ab.json")
    ap.add_argument("--data_dir", default="/root/autodl-tmp/construction_site")
    ap.add_argument("--dino_dir", default="/root/autodl-tmp/models/dinov2-large")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out_all = {}
    analyze(args.a, "A_region", args.dino_dir, args.data_dir, out_all, args.limit)
    analyze(args.b, "B_full", args.dino_dir, args.data_dir, out_all, args.limit)
    with open(args.out, "w") as f:
        json.dump(out_all, f, indent=2, ensure_ascii=False)
    print(f"\n[save] {args.out} DONE", flush=True)


if __name__ == "__main__":
    main()
