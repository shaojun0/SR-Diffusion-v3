"""
像素级逆向还原实验 — 回答"还原图为什么单色, 不是应该重新堆叠逆向还原吗"
=================================================
任务说明: 当前模型还原的是 DINO patch 特征(特征空间), 不是像素。
本脚本把特征**逆向解码回像素**, 检验:
  1. 真实 DINO 特征里到底有没有像素信息(能还原出多少结构)?
  2. 模型 F_hat 相比真实特征, 丢了什么(单色 vs 有结构)?
  3. 全图平均色 / 每 patch 平均色 作为参照 —— 还原能力的下限。

方法: 线性解码(最小二乘, 闭式解, 无需训练) —— 特征 (M*N,1024) → 像素 patch (M*N,588)
      448×252 图 = 32×18 个 14×14 patch × 3 通道 = 每 patch 588 维。
      线性映射是"信息量"的保守探针: 若线性解码都能还原结构, 说明特征里
      有像素信息; 若连真实特征线性解码都只有平均色, 说明 DINO 特征本身
      就不承载空间像素结构(任务目标问题, 非模型问题)。
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
    p.add_argument("--n_images", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="pixel_recon_check.png")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    W, H = (int(v) for v in args.model_input.lower().split("x"))
    num_patches = (W // 14) * (H // 14)
    PATCH_PX = 14 * 14 * 3                       # 每 patch 像素维 588

    dino = Dinov2Model.from_pretrained(args.dino_dir)
    if getattr(dino.config, "use_mask_token", False):
        dino.config.use_mask_token = False
        del dino.embeddings.mask_token
    model = SRPhase1V2(dinov2=dino, num_patches=num_patches,
                       dim=dino.config.hidden_size, reencoder_depth=4)
    sd = torch.load(args.final_model, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval().cuda()

    files = sorted(glob.glob(os.path.join(args.data_dir, "test-*.parquet")))
    ds = ParquetImageDataset(files, limit=64)     # 64 张用于拟合解码器
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4,
                        collate_fn=V2Collator(model_size=(W, H)))

    # ── 收集: 真实 DINO 特征, 模型 F_hat, 像素 patch, 原图 ──
    P_feat, F_feat, P_pix, imgs = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["pixel_values"].cuda()
            feats = model.dinov2(x).last_hidden_state
            cls, patch = feats[:, 0:1], feats[:, 1:]
            specials = model.special_bank(x.shape[0], x.device)
            z = model.re_encoder(torch.cat([cls, specials, patch], dim=1))
            F = model.decoder(z[:, 0:1], z[:, 1:1 + num_patches])
            P_feat.append(patch.float().cpu().numpy())       # (B,N,1024)
            F_feat.append(F.float().cpu().numpy())           # (B,N,1024)
            # 像素 patch: 反归一化 → (B,N,588)
            x_np = x.cpu().numpy().transpose(0, 2, 3, 1) * DINO_STD + DINO_MEAN
            x_np = np.clip(x_np, 0, 255)
            B = x_np.shape[0]
            pix = np.zeros((B, num_patches, PATCH_PX), np.float32)
            for bi in range(B):
                for py in range(H // 14):
                    for px in range(W // 14):
                        blk = x_np[bi, py*14:(py+1)*14, px*14:(px+1)*14]  # (14,14,3)
                        pix[bi, py * (W // 14) + px] = blk.reshape(-1)
            P_pix.append(pix)
            imgs.append(np.clip(x_np, 0, 255).astype(np.uint8))
    P_feat = np.concatenate(P_feat); F_feat = np.concatenate(F_feat)
    P_pix = np.concatenate(P_pix); imgs = np.concatenate(imgs)
    M = P_feat.shape[0]
    print(f"[data] {M} 张 (拟合解码器)")

    # ── 线性解码器: 特征 → 像素 (最小二乘闭式解) ──
    Xp = P_feat.reshape(-1, P_feat.shape[-1])                # (M*N,1024)
    Xf = F_feat.reshape(-1, F_feat.shape[-1])
    Y = P_pix.reshape(-1, PATCH_PX)                          # (M*N,588)
    # 真实特征 → 像素
    Wp, *_ = np.linalg.lstsq(Xp, Y, rcond=1e-6)
    # 模型特征 → 像素
    Wf, *_ = np.linalg.lstsq(Xf, Y, rcond=1e-6)
    # 参照: 全图平均色 / 每 patch 平均色
    img_mean = imgs.mean(axis=(1, 2), keepdims=True)         # (M,1,1,3)
    # 每 patch 平均色 (由像素 patch 求均值)
    patch_mean = P_pix.reshape(M, num_patches, 14, 14, 3).mean(axis=(2, 3))  # (M,N,3)

    # ── 解码并量化 ──
    recon_P = (Xp @ Wp).reshape(M, num_patches, PATCH_PX)    # 真实特征解码
    recon_F = (Xf @ Wf).reshape(M, num_patches, PATCH_PX)    # 模型特征解码

    def patch_to_img(pix):
        out = np.zeros((pix.shape[0], H, W, 3), np.float32)
        for py in range(H // 14):
            for px in range(W // 14):
                k = py * (W // 14) + px
                out[:, py*14:(py+1)*14, px*14:(px+1)*14] = \
                    pix[:, k].reshape(-1, 14, 14, 3)
        return np.clip(out, 0, 255)

    def l1(a, b):
        return np.abs(a - b).mean()

    def block_upsample(arr, nrows, ncols):
        """(M,N,3) patch 均值 → 放大回 (M,H,W,3) 块状图"""
        out = np.zeros((arr.shape[0], H, W, 3), np.float32)
        for py in range(nrows):
            for px in range(ncols):
                k = py * ncols + px
                out[:, py*14:(py+1)*14, px*14:(px+1)*14] = arr[:, k][:, None, None, :]
        return np.clip(out, 0, 255)

    rP = patch_to_img(recon_P); rF = patch_to_img(recon_F)
    g_imgmean = np.broadcast_to(img_mean, (M, H, W, 3)).copy()
    g_patchmean = block_upsample(patch_mean, H // 14, W // 14)

    print("\n[量化] 像素还原 L1 (原图归一化到 0-255):")
    print(f"  真实特征→像素 : {l1(rP, imgs):.2f}")
    print(f"  模型F_hat→像素: {l1(rF, imgs):.2f}")
    print(f"  全图平均色    : {l1(g_imgmean, imgs):.2f}  (下限参照)")
    print(f"  每patch平均色 : {l1(g_patchmean, imgs):.2f}  (结构下限参照)")
    print(f"\n  相对全图平均色的改善:")
    base = l1(g_imgmean, imgs)
    print(f"  真实特征解码改善 {base - l1(rP, imgs):.2f} | "
          f"模型解码改善 {base - l1(rF, imgs):.2f}")

    # ── 可视化: 每行一张图 ──
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

    show = list(range(args.n_images))
    titles = ["原图", "真实特征→像素\n(线性解码)", "模型F_hat→像素\n(线性解码)",
              "每patch平均色", "全图平均色"]
    datas = [imgs[show], rP[show], rF[show], g_patchmean[show], g_imgmean[show]]
    fig, axes = plt.subplots(len(show), 5, figsize=(4.2 * 5, 3.6 * len(show)))
    for b, i in enumerate(show):
        for c in range(5):
            ax = axes[b, c]
            ax.imshow(datas[c][b].astype(np.uint8) if c == 0 else datas[c][b])
            if b == 0:
                ax.set_title(titles[c], fontsize=12, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle("像素级逆向还原 — 特征线性解码回原图 (448×252 → 32×18 patch → 解码)\n"
                 "若真实特征解码也≈平均色 ⇒ DINO 特征不承载空间像素结构, 目标定义问题",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
