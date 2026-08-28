"""
可视化（register K 压缩版 v4, 2026-08-28）: 原图 vs 重建（单次前向）。
用法:
    python visualize_v4.py \
        --data_dir /root/autodl-tmp/construction_site \
        --dino_dir /root/autodl-tmp/models/dinov2-large \
        --final_model output/phase1_v4_registerK/final_model.pt \
        --out output/phase1_v4_registerK/recon_visual.png
"""
import argparse
import os
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import Dinov2Model

from data_v2 import ParquetImageDataset, V2Collator, DINO_MEAN, DINO_STD
from model_v4 import SRPhase1V4


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--dino_dir", default="models/dinov2-large")
    p.add_argument("--final_model", required=True)
    p.add_argument("--model_input", default="448x252")
    p.add_argument("--num_specials", type=int, default=64)
    p.add_argument("--decoder_depth", type=int, default=2)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--head_hidden", type=int, default=2048)
    p.add_argument("--freeze_dino", action="store_true")
    p.add_argument("--n_images", type=int, default=4, help="展示几张图(行)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="output/phase1_v4_registerK/recon_visual.png")
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
    print(f"[model] N={num_patches}, K={model.num_specials}, "
          f"freeze_dino={model.freeze_dino}")

    # 数据: 前 n_images 张 test
    test_files = sorted(glob.glob(os.path.join(args.data_dir, "test-*.parquet")))
    ds = ParquetImageDataset(test_files, limit=args.n_images)
    loader = DataLoader(ds, batch_size=args.n_images, shuffle=False,
                        num_workers=4, collate_fn=V2Collator(model_size=(W, H)))

    with torch.no_grad():
        for batch in loader:
            x = batch["pixel_values"].cuda()
            B, C, Hh, Ww = x.shape
            out = model(x)
            recon = patch_to_img(out["pixels"], Hh, Ww)     # (B,H,W,3) uint8
            gt = patch_to_img(out["target"], Hh, Ww)
            l1 = np.abs(recon.astype(np.float32) - gt.astype(np.float32)) \
                     .mean(axis=(1, 2, 3))                  # (B,)
            break

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

    n_rows = args.n_images
    fig, axes = plt.subplots(n_rows, 2, figsize=(8.0, 3.0 * n_rows))
    for b in range(n_rows):
        for c, (title, img) in enumerate(
                [("原图", gt[b]), (f"重建 (L1={l1[b]:.1f})", recon[b])]):
            ax = axes[b, c]
            ax.imshow(img)
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle(f"register K 压缩重建 (K={model.num_specials} specials, "
                 f"DINO {'冻结' if model.freeze_dino else '不冻结'})",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
