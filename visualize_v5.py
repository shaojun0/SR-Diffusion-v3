"""
可视化（扩散式渐进细化 v5, 2026-08-31）: 原图 vs DDIM 渐进快照 vs 最终重建。
列: 原图 | 噪声起点 | 若干中间 x̂0 快照（t 递减 = m 递增 = 粗→细）| 最终重建。
直观检验"渐进思想": 快照列应从模糊背景逐步长出结构/细节。

用法:
    python visualize_v5.py \
        --data_dir /root/autodl-tmp/construction_site \
        --dino_dir /root/autodl-tmp/models/dinov2-large \
        --final_model output/phase1_v5_diffusion/final_model.pt \
        --out output/phase1_v5_diffusion/recon_visual.png
"""
import argparse
import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Dinov2Model

from data_v2 import ParquetImageDataset, V2Collator, DINO_MEAN, DINO_STD
from model_v5 import SRPhase1V5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--dino_dir", default="models/dinov2-large")
    p.add_argument("--final_model", required=True)
    p.add_argument("--model_input", default="448x252")
    p.add_argument("--num_specials", type=int, default=128)
    p.add_argument("--diffusion_steps", type=int, default=1000)
    p.add_argument("--decoder_depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=2.0)
    p.add_argument("--freeze_dino", action="store_true")
    p.add_argument("--unlock", default="linear")
    p.add_argument("--ddim_steps", type=int, default=100)
    p.add_argument("--snapshots", type=int, default=5,
                   help="中间快照数(不含最终步)")
    p.add_argument("--n_images", type=int, default=3, help="展示几张图(行)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="output/phase1_v5_diffusion/recon_visual.png")
    return p.parse_args()


def patch_to_img(pix_patches, H, W):
    """(B,N,588) 归一化像素 patch → (B,H,W,3) 反归一化 0-255 (uint8)。"""
    B = pix_patches.shape[0]
    img = pix_patches.reshape(B, H // 14, W // 14, 14, 14, 3) \
                     .permute(0, 1, 3, 2, 4, 5) \
                     .reshape(B, H, W, 3)
    img = img.cpu().numpy() * DINO_STD + DINO_MEAN
    return np.clip(img, 0, 255).astype(np.uint8)


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
    print(f"[model] N={num_patches}, K={model.num_specials}, "
          f"T={model.T}, unlock={model.unlock}, freeze_dino={model.freeze_dino}")

    test_files = sorted(glob.glob(os.path.join(args.data_dir, "test-*.parquet")))
    ds = ParquetImageDataset(test_files, limit=args.n_images)
    loader = DataLoader(ds, batch_size=args.n_images, shuffle=False,
                        num_workers=4,
                        collate_fn=V2Collator(model_size=(W, H)))

    with torch.no_grad():
        for batch in loader:
            x = batch["pixel_values"].cuda()
            B, C, Hh, Ww = x.shape
            z_cls, z_s = model.encode(x)
            x0 = model._patches(x)
            # 手动跑一遍 DDIM, 在选定的 t 上记录 x̂0 快照
            ts = torch.linspace(model.T, 1, args.ddim_steps).round().long()
            ts = ts.unique_consecutive().tolist()
            x_cur = torch.randn_like(x0)                       # x_T ≈ 纯噪声
            snap_idx = sorted(set(np.linspace(0, len(ts) - 1,
                                              args.snapshots + 1).astype(int)))
            snap_idx = snap_idx[:-1]                           # 去掉最后(单独展示)
            snaps = []                                         # [(t, m, x̂0)]
            for i, t in enumerate(ts):
                m = model.unlock_count(t)
                ctx = torch.cat([z_cls, z_s[:, :m]], dim=1) if m > 0 else z_cls
                emb = model._time_embed_full(B, t, x.device)
                xh = model.decoder(x_cur, emb, ctx)
                if i in snap_idx:
                    snaps.append((t, m, xh))
                if t > 1:
                    t_prev = ts[i + 1] if i + 1 < len(ts) else t - 1
                    a_t = float(model.alphas_cumprod[t])
                    a_p = float(model.alphas_cumprod[max(t_prev, 0)])
                    eps_h = (x_cur - (a_t ** 0.5) * xh) \
                        / max(1.0 - a_t, 1e-5) ** 0.5
                    x_cur = (a_p ** 0.5) * xh \
                        + max(1.0 - a_p, 0.0) ** 0.5 * eps_h
            final = xh
            break

    gt = patch_to_img(x0, Hh, Ww)
    final_img = patch_to_img(final, Hh, Ww)
    l1_final = np.abs(final_img.astype(np.float32)
                      - gt.astype(np.float32)).mean(axis=(1, 2, 3))
    n_cols = 2 + len(snaps) + 1

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

    fig, axes = plt.subplots(args.n_images, n_cols,
                             figsize=(3.2 * n_cols, 3.2 * args.n_images))
    for b in range(args.n_images):
        axes[b, 0].imshow(gt[b])
        axes[b, 0].set_title("原图", fontsize=12, fontweight="bold")
        axes[b, 1].imshow(np.full_like(gt[b], 127))
        axes[b, 1].set_title("噪声起点", fontsize=11)
        for j, (t, m, xh_s) in enumerate(snaps):
            img = patch_to_img(xh_s, Hh, Ww)
            l1s_ = float((xh_s - x0).abs().mean().detach())
            axes[b, 2 + j].imshow(img[b])
            axes[b, 2 + j].set_title(f"t={t}\nm={m}  L1={l1s_:.1f}", fontsize=10)
        axes[b, n_cols - 1].imshow(final_img[b])
        axes[b, n_cols - 1].set_title(f"最终重建\nL1={l1_final[b]:.1f}",
                                      fontsize=12, fontweight="bold")
        for c in range(n_cols):
            axes[b, c].set_xticks([])
            axes[b, c].set_yticks([])
    plt.suptitle(f"v5 扩散式渐进细化 (K={model.num_specials}, "
                 f"unlock={model.unlock}, DDIM {args.ddim_steps} 步)",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[save] {args.out} | 最终 L1={['%.1f' % v for v in l1_final]}")


if __name__ == "__main__":
    main()
