"""
可视化: 左边原图, 右边各采样步 t 的重建结果(特征图)。
重建目标 = DINO patch 特征 (B,576,1024) → PCA 3 主成分投影到 RGB,
与真实 patch 特征同变换对比, 直观展示"前缀 t 越长还原越准"。
"""
import argparse
import glob
import io

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import Dinov2Model

from data_v2 import ParquetImageDataset, V2Collator, fit_to_canvas, DINO_MEAN, DINO_STD
from model_v2 import SRPhase1V2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--dino_dir", default="models/dinov2-large")
    p.add_argument("--final_model", required=True)
    p.add_argument("--model_input", default="448x252")
    p.add_argument("--n_images", type=int, default=4, help="展示几张图(行)")
    p.add_argument("--steps", default="0,16,64,256,576", help="展示哪些采样步(列)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="recon_visual.png")
    return p.parse_args()


def main():
    args = parse_args()
    W, H = (int(v) for v in args.model_input.lower().split("x"))
    num_patches = (W // 14) * (H // 14)
    show_steps = [int(s) for s in args.steps.split(",") if s.strip()]

    dino = Dinov2Model.from_pretrained(args.dino_dir)
    if getattr(dino.config, "use_mask_token", False):
        dino.config.use_mask_token = False
        del dino.embeddings.mask_token
    model = SRPhase1V2(dinov2=dino, num_patches=num_patches,
                       dim=dino.config.hidden_size, reencoder_depth=4)
    sd = torch.load(args.final_model, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval().cuda()
    T_steps = model.decoder.steps
    print(f"[model] N={num_patches}, {len(T_steps)} 采样步")

    files = sorted(glob.glob(os.path.join(args.data_dir, "test-*.parquet")))
    ds = ParquetImageDataset(files, limit=args.n_images)
    loader = DataLoader(ds, batch_size=args.n_images, shuffle=False,
                        num_workers=4, collate_fn=V2Collator(model_size=(W, H)))

    # 收集: 原图(画布), 真实 patch 特征, 各步重建特征
    imgs = []
    with torch.no_grad():
        for batch in loader:
            x = batch["pixel_values"].cuda()
            feats = model.dinov2(x).last_hidden_state
            cls, patch = feats[:, 0:1], feats[:, 1:]
            specials = model.special_bank(x.shape[0], x.device)
            z = model.re_encoder(torch.cat([cls, specials, patch], dim=1))
            z_cls, z_s = z[:, 0:1], z[:, 1:1 + num_patches]
            _ = model.decoder(z_cls, z_s)
            Y = model.decoder.last_Y                       # (B,|T|,N,D) fp32
            # 原图(反归一化 → 画布图)
            x_np = x.cpu().numpy().transpose(0, 2, 3, 1)
            x_np = x_np * DINO_STD + DINO_MEAN
            x_np = np.clip(x_np, 0, 255).astype(np.uint8)
            imgs = [Image.fromarray(x_np[i]) for i in range(x_np.shape[0])]
            patch_np = patch.float().cpu().numpy()          # (B,N,D)
            Y_np = Y.float().cpu().numpy()                  # (B,|T|,N,D)
            break

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    for cand in ("/root/.fonts/wqy-zenhei.ttc", "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"):
        if os.path.exists(cand):
            fm.fontManager.addfont(cand)
            break
    plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    B = len(imgs)
    ncols = 1 + 1 + len(show_steps)   # 原图 | 真实特征 | 各 t
    fig, axes = plt.subplots(B, ncols, figsize=(3.2 * ncols, 3.2 * B))

    # 每张图独立 PCA(3) 拟合于真实特征, 同一变换投影重建特征
    for b in range(B):
        P = patch_np[b]                    # (N,D) 真实
        Pc = P - P.mean(axis=0, keepdims=True)
        # PCA 3 主成分
        U, S, Vt = np.linalg.svd(Pc, full_matrices=False)
        W3 = Vt[:3]                        # (3,D) 主方向
        def proj(M):
            Mc = M - P.mean(axis=0, keepdims=True)
            return Mc @ W3.T               # (N,3)
        real_rgb = proj(P)                 # (N,3)
        # 归一化到 [0,1]（用真实+全部重建的并集范围, 保证可比）
        all_proj = [real_rgb] + [proj(Y_np[b][i]) for i in range(len(T_steps))]
        lo = min(a.min() for a in all_proj)
        hi = max(a.max() for a in all_proj)
        span = max(hi - lo, 1e-9)

        def to_img(M):
            rgb = (proj(M) - lo) / span     # (N,3)
            grid = rgb.reshape(H // 14, W // 14, 3)   # (18,32,3)
            return np.clip(grid, 0, 1)

        # 列0: 原图
        ax = axes[b, 0]
        ax.imshow(imgs[b])
        ax.set_title("原图" if b == 0 else "", fontsize=12, fontweight="bold")
        ax.axis("off")
        # 列1: 真实特征
        ax = axes[b, 1]
        ax.imshow(to_img(P), interpolation="nearest")
        ax.set_title("真实 DINO 特征\n(PCA→RGB)" if b == 0 else "", fontsize=12, fontweight="bold")
        ax.axis("off")
        # 各 t 列
        for c, t in enumerate(show_steps):
            ti = T_steps.index(t)
            ax = axes[b, 2 + c]
            ax.imshow(to_img(Y_np[b][ti]), interpolation="nearest")
            l1 = np.abs(Y_np[b][ti] - P).mean()
            ax.set_title(f"t={t}\nL1={l1:.5f}" if b == 0 else f"t={t}\nL1={l1:.5f}",
                         fontsize=11)
            ax.axis("off")

    plt.suptitle(f"重建可视化 — 左:原图 | 中:真实特征 | 右:各采样步 t 重建 (fp32, N={num_patches})",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[save] {args.out}")


if __name__ == "__main__":
    import os
    main()
