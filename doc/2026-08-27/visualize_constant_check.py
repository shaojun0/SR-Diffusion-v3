"""
可视化: 还原效果 vs 质心 baseline —— 判断"是不是又是常数"
=================================================
针对 fp32+uniform 训练(round2) 的还原可视化, 与上一版(visualize_recon.py)区别:
  1. 专门选 test 集中 patch 特征"空间变异性最大"的图 —— 若连这些图模型
     都输出常数, 则常数结论铁证; 若有微弱结构, 也能在最大变异图上看到。
  2. 每行四列: 原图 | 真实 DINO 特征 | 模型 F_hat | 质心 baseline
     (同一 PCA→RGB 变换, 同色标) —— 直观对比 F_hat 是贴近真实还是贴近质心。
  3. 每张图下方给出量化: 真实/模型/质心的跨位置 std、F_hat-vs-真实 L1、
     F_hat-vs-质心 L1。若 F_hat≈质心(常数), 则 F_hat 跨位置 std≈0 且
     F_hat-vs-质心 L1≈0。
  4. 全局 PCA(在所选图上拟合一次) —— 同变换、同色标, 避免每图独立 PCA
     造成的假对比。
"""
import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import Dinov2Model

from data_v2 import ParquetImageDataset, V2Collator, DINO_MEAN, DINO_STD
from model_v2 import SRPhase1V2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--dino_dir", default="models/dinov2-large")
    p.add_argument("--final_model", required=True)
    p.add_argument("--model_input", default="448x252")
    p.add_argument("--scan", type=int, default=256, help="从多少张 test 里选变异最大图")
    p.add_argument("--n_images", type=int, default=4, help="展示几张(行)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="recon_constant_check.png")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    W, H = (int(v) for v in args.model_input.lower().split("x"))
    num_patches = (W // 14) * (H // 14)

    dino = Dinov2Model.from_pretrained(args.dino_dir)
    if getattr(dino.config, "use_mask_token", False):
        dino.config.use_mask_token = False
        del dino.embeddings.mask_token
    model = SRPhase1V2(dinov2=dino, num_patches=num_patches,
                       dim=dino.config.hidden_size, reencoder_depth=4)
    sd = torch.load(args.final_model, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval().cuda()
    print(f"[model] N={num_patches}, {len(model.decoder.steps)} 采样步")

    files = sorted(glob.glob(os.path.join(args.data_dir, "test-*.parquet")))
    ds = ParquetImageDataset(files, limit=args.scan)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4,
                        collate_fn=V2Collator(model_size=(W, H)))

    # ── pass 1: 收集特征, 算每图跨位置 std, 挑变异最大的 n_images 张 ──
    all_patch, all_img, all_F, rowstds, all_x = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["pixel_values"].cuda()
            feats = model.dinov2(x).last_hidden_state
            cls, patch = feats[:, 0:1], feats[:, 1:]
            specials = model.special_bank(x.shape[0], x.device)
            z = model.re_encoder(torch.cat([cls, specials, patch], dim=1))
            F = model.decoder(z[:, 0:1], z[:, 1:1 + num_patches])
            rs = patch.std(dim=1).mean(dim=1)                # (B,)
            rowstds.append(rs.cpu().numpy())
            all_patch.append(patch.float().cpu().numpy())
            all_F.append(F.float().cpu().numpy())
            # 原图
            x_np = x.cpu().numpy().transpose(0, 2, 3, 1) * DINO_STD + DINO_MEAN
            all_img.append(np.clip(x_np, 0, 255).astype(np.uint8))
            all_x.append(x)
    rowstds = np.concatenate(rowstds)
    patch = np.concatenate(all_patch)                        # (M,N,D)
    F = np.concatenate(all_F)                                # (M,N,D)
    imgs = np.concatenate(all_img)                           # (M,H,W,3)
    print(f"[scan] {len(rowstds)} 张, 跨位置 std: "
          f"mean={rowstds.mean():.6f} max={rowstds.max():.6f} "
          f"min={rowstds.min():.6f}")
    # 挑变异最大的 n 张 (有代表性的最"有结构"的图)
    top_idx = np.argsort(rowstds)[-args.n_images:][::-1]
    print(f"[pick] 变异最大 {args.n_images} 张: idx={top_idx.tolist()} "
          f"std={[f'{rowstds[i]:.6f}' for i in top_idx]}")

    # ── 全局 PCA(3) 拟合于所选图的真实特征, 同变换投影所有 ──
    P_sel = patch[top_idx].reshape(-1, patch.shape[-1])      # (n*N, D)
    Pc = P_sel - P_sel.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Pc, full_matrices=False)
    W3 = Vt[:3]
    # 解释率
    expl = (S[:3] ** 2).sum() / (S ** 2).sum()
    print(f"[pca] 前3主成分解释的空间变异比例 = {expl:.4f} "
          f"(低 ⇒ 空间变异极弱, 目标近似常数)")

    def proj(M):
        Mc = M - P_sel.mean(axis=0, keepdims=True)
        return Mc @ W3.T                                      # (...,3)

    # 归一化范围: 真实 + F + 质心 的并集
    real_rgb = proj(patch[top_idx])                          # (n,N,3)
    F_rgb = proj(F[top_idx])
    centroid = patch[top_idx].mean(axis=1, keepdims=True)    # (n,1,D)
    c_rgb = proj(np.broadcast_to(centroid, patch[top_idx].shape))
    lo = min(real_rgb.min(), F_rgb.min(), c_rgb.min())
    hi = max(real_rgb.max(), F_rgb.max(), c_rgb.max())
    span = max(hi - lo, 1e-9)

    def to_img(rgb):
        grid = ((rgb - lo) / span).reshape(args.n_images, H // 14, W // 14, 3)
        return np.clip(grid, 0, 1)

    # ── 量化 ──
    print("\n[量化] (n = 所选图)")
    for b, i in enumerate(top_idx):
        P, Fb = patch[i], F[i]
        ctr = P.mean(axis=0)
        rowstd_P = P.std(axis=0).mean()
        rowstd_F = Fb.std(axis=0).mean()
        l1_PF = np.abs(Fb - P).mean()
        l1_PC = np.abs(P - ctr).mean()
        l1_FC = np.abs(Fb - ctr).mean()
        print(f"  img#{i}: 真实跨位置std={rowstd_P:.2e} | "
              f"模型跨位置std={rowstd_F:.2e} | "
              f"F-vs-真实L1={l1_PF:.2e} | F-vs-质心L1={l1_FC:.2e} | "
              f"真实-vs-质心L1={l1_PC:.2e}")

    # ── 绘图 ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    for cand in ("/root/.fonts/wqy-zenhei.ttc",
                 "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"):
        if os.path.exists(cand):
            fm.fontManager.addfont(cand)
            break
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    ncols = 4
    fig, axes = plt.subplots(args.n_images, ncols,
                             figsize=(4.0 * ncols, 3.6 * args.n_images))
    titles = ["原图", "真实 DINO 特征\n(PCA→RGB)", "模型 F_hat\n(PCA→RGB)",
              "质心 baseline\n(每图均值向量)"]
    datas = [imgs[top_idx], to_img(real_rgb), to_img(F_rgb), to_img(c_rgb)]
    for b in range(args.n_images):
        for c in range(ncols):
            ax = axes[b, c]
            if c == 0:
                ax.imshow(datas[0][b])
            else:
                ax.imshow(datas[c][b], interpolation="nearest")
            if b == 0:
                ax.set_title(titles[c], fontsize=12, fontweight="bold")
            if c == 2 and b == args.n_images - 1:
                ax.set_xlabel("同 PCA 变换 / 同色标", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        P, Fb = patch[top_idx[b]], F[top_idx[b]]
        ax = axes[b, 0]
        ax.set_ylabel(f"std={rowstds[top_idx[b]]:.1e}\n(变异最大图)",
                      fontsize=9, rotation=0, labelpad=40, va="center")
    plt.suptitle("还原效果 vs 质心 baseline — 判断是否输出常数\n"
                 f"(fp32+uniform, N={num_patches}, PCA 前3主成分解释率 {expl:.4f})",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
