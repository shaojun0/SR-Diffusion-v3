"""
probe_e1.py — E1 线性探针 (零训练, ~30 min): 量化 "z_s 还剩多少可线性读出的像素信息"

设计来源: doc/2026-09-04/ANALYSIS_v2_info_left_and_decoder.md §5 (E1)
参照: doc/2026-08-27/trace_info_pixel.py (dec_l1 口径) + probe_slice05_exp1.py
      (当前 register 式 API / 掩码 / 像素 L1 0-255 口径, 512 子集一致性)

对每个 final_model 测量 (同一 512 子集):
  P   = DINO 24 层 + 块内 register 同序列的 patch 段 seq[:,1+K:] (与训练分布一致)
  z_s = encode() 输出的 register 段 seq[:,1:1+K]
  h   = 解码器 Y.cumsum(dim=1)[:,-1] (最后步累加特征, PixelHead 之前)
  参照 = 完整模型解码 (PixelHead) 的像素 L1(0-255) + 平均色基线
线性解码 L1 越小 → 该层携带的可线性利用像素信息越多。

判读 (对照 ANALYSIS §5): z_s→像素 ≈9-14 → 信息在 z_s, 浪费在解码结构;
≈19-20(与全模型相当) → 解码器没浪费多少; ≈40+ → 编码路由就丢了大半。

用法(在 /root/autodl-tmp/sr-diffusion-v2-k99 下):
  CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/python probe_e1.py \
      --limit 512 --batch_size 16
输出: output/probe_e1_slice27_v2.json + output/probe_e1_slice05.json
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
import model_v2 as M
from model_v2 import SRPhase1V2

assert not M.SRV2_MEMORY_OPEN, "E1 探针必须 memory_open=False (不要 export SRV2_MEMORY_OPEN)"

ORIG_BM = M.build_block_mask
ORIG_TM = M.build_causal_query_mask

# (输出目录, K, steps, 名字)
RUNS = [
    ("phase1_v2_block_slice27_v2", 63, [9, 16, 25, 36, 49], "slice27_v2"),
    ("phase1_v2_block_slice05", 35, [1, 4, 9, 16, 25], "slice05"),
]

W, H = 448, 252
NUM_PATCHES = (W // 14) * (H // 14)   # 576
PATCH_PX = 14 * 14 * 3                # 588


def encode_with_patch(model, x):
    """与 model._encode_register 同一 forward, 额外返回 patch 段特征 P。
    seq = [cls(1); specials(K); patches(N)] 过 DINO 24 层 + layernorm。"""
    emb = model.dinov2.embeddings(x)
    specials = model.special_bank(x.shape[0], x.device)
    seq = torch.cat([emb[:, :1], specials, emb[:, 1:]], dim=1)
    for layer in model.dinov2.encoder.layer:
        out = layer(seq)
        seq = out[0] if isinstance(out, (tuple, list)) else out
    seq = model.dinov2.layernorm(seq)          # (B,1+K+N,D)
    K = model.num_specials
    return seq[:, :1], seq[:, 1:1 + K], seq[:, 1 + K:]   # z_cls, z_s, P


def within_std(T):
    return float(T.std(axis=1).mean(axis=(0, 1)))


def patch_seq_to_img(seq):
    """(M,N,14,14,3) -> (M,H,W,3)"""
    Mm = seq.shape[0]
    img = np.zeros((Mm, H, W, 3), np.float32)
    for py in range(H // 14):
        for px in range(W // 14):
            k = py * (W // 14) + px
            img[:, py * 14:(py + 1) * 14, px * 14:(px + 1) * 14] = seq[:, k]
    return img


def dec_l1(Tfeat, pix, gt, name):
    """线性解码 L1 (0-255 尺度, 越小越好)。
    Tfeat:(M,N,D) pix:(M,N,588) gt:(M,H,W,3) 0-255"""
    t0 = time.time()
    X = Tfeat.reshape(-1, Tfeat.shape[-1]).astype(np.float64)
    Y = pix.reshape(-1, PATCH_PX).astype(np.float64)
    Wm, *_ = np.linalg.lstsq(X, Y, rcond=1e-6)
    recon = (X @ Wm).astype(np.float32)
    recon = recon.reshape(pix.shape[0], NUM_PATCHES, 14, 14, 3)
    rimg = patch_seq_to_img(recon)
    l1 = float(np.abs(rimg - gt).mean())
    print(f"  {name} -> 像素 L1 = {l1:.2f}  (lstsq {time.time()-t0:.0f}s)",
          flush=True)
    return l1


def l1_255_batch(Ypix, tg_g, Bt):
    """(Bt,T,N,588)->每步 0-255 L1(T,), 与 probe_slice05_exp1.l1_255 同口径"""
    tg_np = tg_g.cpu().numpy().reshape(Bt, H // 14, W // 14, 14, 14, 3) \
        .transpose(0, 1, 3, 2, 4, 5).reshape(Bt, H, W, 3)
    gt = np.clip(tg_np * DINO_STD + DINO_MEAN, 0, 255)
    arr = Ypix.cpu().numpy()
    out = np.zeros(arr.shape[1])
    for i in range(arr.shape[1]):
        si = arr[:, i].reshape(Bt, H // 14, W // 14, 14, 14, 3) \
            .transpose(0, 1, 3, 2, 4, 5).reshape(Bt, H, W, 3)
        ri = np.clip(si * DINO_STD + DINO_MEAN, 0, 255)
        out[i] += float(np.abs(ri - gt).mean(axis=(1, 2, 3)).sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/root/autodl-tmp/construction_site")
    ap.add_argument("--dino_dir", default="/root/autodl-tmp/models/dinov2-large")
    ap.add_argument("--out_root", default="output")
    ap.add_argument("--limit", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    dino = Dinov2Model.from_pretrained(args.dino_dir)
    if getattr(dino.config, "use_mask_token", False):
        dino.config.use_mask_token = False
        del dino.embeddings.mask_token

    test_files = sorted(glob.glob(os.path.join(args.data_dir, "test-*.parquet")))
    coll = V2Collator(model_size=(W, H), canvas=(1600, 900))
    ds = ParquetImageDataset(test_files, limit=args.limit)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=coll)
    print(f"[data] {len(ds)} 条 test (limit={args.limit})", flush=True)

    for out_dir, K, steps, tag in RUNS:
        out = os.path.join(args.out_root, out_dir, "final_model.pt")
        print(f"\n===== E1: {tag}  (K={K}, steps={steps}) =====", flush=True)
        model = SRPhase1V2(dinov2=dino, num_patches=NUM_PATCHES,
                           dim=dino.config.hidden_size, heads=8, mlp_ratio=4.0,
                           decoder_steps=steps, decoder_depth=2, num_specials=K)
        sd = torch.load(out, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=True)
        assert not missing and not unexpected, (missing, unexpected)
        model.eval().cuda()
        decoder = model.decoder
        pixel_head = model.pixel_head
        print(f"[model] N={NUM_PATCHES} K={model.num_specials} "
              f"steps={decoder.steps}", flush=True)

        # ── Pass 1: 编码缓存 (CPU) ──
        cache = []
        t0 = time.time()
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                x = batch["pixel_values"].cuda()
                Bc, C, Hh, Ww = x.shape
                z_cls, z_s, P = encode_with_patch(model, x)
                target = x.reshape(Bc, C, Hh // 14, 14, Ww // 14, 14) \
                           .permute(0, 2, 4, 1, 3, 5).reshape(Bc, NUM_PATCHES,
                                                              PATCH_PX)
                cache.append((z_cls.cpu(), z_s.cpu(), P.cpu(), target.cpu()))
                if (bi + 1) % 8 == 0:
                    print(f"  encode ... {len(cache)*Bc}/{len(ds)} "
                          f"({time.time()-t0:.0f}s)", flush=True)
        print(f"[cache] done ({time.time()-t0:.0f}s)", flush=True)

        # ── Pass 2: 全模型解码参照 (trained 口径, 0-255) + h 特征 ──
        P_all, Z_all, H_all, pix_all = [], [], [], []
        acc_full_nrm, acc_full_255 = 0.0, 0.0
        step_nrm = np.zeros(len(steps)); step_255 = np.zeros(len(steps))
        n_acc = 0
        t1 = time.time()
        with torch.no_grad():
            for (zc, zs, Pc, tg) in cache:
                Bc = tg.shape[0]
                zc_g, zs_g, tg_g = zc.cuda(), zs.cuda(), tg.cuda()
                Y = decoder(zc_g, zs_g)                      # (B,T,N,D)
                Y_cum = torch.cat([torch.zeros_like(Y[:, :1]),
                                   Y.cumsum(dim=1)[:, :-1]], dim=1).detach() + Y
                Y_pix_cum = pixel_head(Y_cum)                # (B,T,N,588)
                h = Y.cumsum(dim=1)[:, -1]                   # (B,N,D) pre-head
                nrm = (Y_pix_cum - tg_g.unsqueeze(1)).abs().mean(dim=(0, 2, 3))
                s255 = l1_255_batch(Y_pix_cum, tg_g, Bc)
                acc_full_nrm += float(nrm[-1]) * Bc
                acc_full_255 += s255[-1]
                step_nrm += nrm.cpu().numpy() * Bc
                step_255 += s255
                n_acc += Bc
                P_all.append(Pc.numpy()); Z_all.append(zs.numpy())
                H_all.append(h.cpu().numpy())
                # 像素目标 (0-255) 由缓存 target 反归一化
                del Y, Y_cum, Y_pix_cum, h
                torch.cuda.empty_cache()
        print(f"[decode-ref] done ({time.time()-t1:.0f}s)", flush=True)

        P = np.concatenate(P_all); Z = np.concatenate(Z_all)
        Hf = np.concatenate(H_all)
        # (M,N,588) 归一化 patch -> 0-255
        tg_all = torch.cat([t for *_, t in cache], dim=0)
        tg_np = tg_all.numpy().reshape(n_acc, H // 14, W // 14, 14, 14, 3) \
            .transpose(0, 1, 3, 2, 4, 5).reshape(n_acc, H, W, 3)
        gt_img = np.clip(tg_np * DINO_STD + DINO_MEAN, 0, 255).astype(np.float32)
        pix = np.stack([gt_img[:, py * 14:(py + 1) * 14, px * 14:(px + 1) * 14]
                        .reshape(n_acc, -1)
                        for py in range(H // 14) for px in range(W // 14)],
                       axis=1)                                # (M,N,588)
        Mm, Nn = gt_img.shape[0], NUM_PATCHES
        assert pix.shape == (Mm, Nn, PATCH_PX), pix.shape

        img_mean = np.broadcast_to(gt_img.mean(axis=(1, 2), keepdims=True),
                                   gt_img.shape)
        mean_col_l1 = float(np.abs(img_mean - gt_img).mean())

        res = {"model": tag, "K": K, "steps": steps, "n": n_acc,
               "trained_full_norm_l1": acc_full_nrm / n_acc,
               "trained_full_pixel_l1_255": acc_full_255 / n_acc,
               "trained_step_pixel_l1_255":
                   [round(float(v / n_acc), 4) for v in step_255],
               "mean_color_pixel_l1_255": round(mean_col_l1, 4),
               "within_std": {"P": within_std(P), "z_s": within_std(Z),
                              "h": within_std(Hf)},
               "dec_l1_pixel_255": {}}
        print(f"[ref] trained full L1_255="
              f"{res['trained_full_pixel_l1_255']:.2f}  mean-color="
              f"{mean_col_l1:.2f}", flush=True)
        print(f"[std] P={res['within_std']['P']:.3e} "
              f"z_s={res['within_std']['z_s']:.3e} "
              f"h={res['within_std']['h']:.3e}", flush=True)

        res["dec_l1_pixel_255"]["P_dino_patch"] = dec_l1(P, pix, gt_img,
                                                         "DINO patch P")
        res["dec_l1_pixel_255"]["h_laststep"] = dec_l1(Hf, pix, gt_img,
                                                       "decoder h(Y_cum[-1])")
        # z_s 是 K=63 个全图 register token, 与 576 个 patch 不对齐——
        # 逐 patch 线性解码不适定 (512 样本 < K*D=64512 维), 故不在此做线性
        # 探针; 其可及信息由 h(=z_s 经注意力读出的 patch 对齐特征) 与
        # trained 参照共同刻画。仅报告其空间冗余度 (within_std) 与数量级。
        res["dec_l1_pixel_255"]["z_s"] = None
        res["note"] = ("z_s(register K token) 非 patch 对齐, 线性逐 patch "
                       "探针不适定置 None; h=Y_cum[-1] 即 z_s 经解码读出的 "
                       "patch 对齐特征。")
        # 与解码器同容量的线性参照: z_s + 非线性降维无; 结束

        os.makedirs(args.out_root, exist_ok=True)
        outp = os.path.join(args.out_root, f"probe_e1_{tag}.json")
        with open(outp, "w") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"[save] {outp}", flush=True)
        del cache, P, Z, Hf, pix
        torch.cuda.empty_cache()

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
