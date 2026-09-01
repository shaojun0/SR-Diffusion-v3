"""
可视化（像素目标版, 2026-08-27）: 原图 vs 各采样步的像素重建。
重建目标 = 原始像素 patch (B,576,588) → 反归一化回 0-255 → 直接显示。
每行一张 test 图, 列 = 原图 + 各采样步的重建（2026-08-31 起为**累加**
语义: 第 n 步 = 前 n 步预测之和, 累积步数越多重建越完整）。

用法:
    python visualize_recon_pixel.py \
        --data_dir /root/autodl-tmp/construction_site \
        --dino_dir /root/autodl-tmp/models/dinov2-large \
        --final_model output/phase1_v2_block/final_model.pt \
        --register_specials --slice_start 4 --slice_end 9 \
        --out output/phase1_v2_block/recon_visual.png
"""
import argparse
import os
import glob

import numpy as np
import torch
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
    p.add_argument("--register_specials", action="store_true",
                   help="register 式模型(与训练 --register_specials 一致)")
    p.add_argument("--decoder_depth", type=int, default=2,
                   help="OutputQueryDecoder 层数(与训练 --decoder_depth 一致)")
    p.add_argument("--slice_start", type=int, default=None,
                   help="分块切片起点(与训练 --slice_start 一致); 默认 None = 全部分块")
    p.add_argument("--slice_end", type=int, default=None,
                   help="分块切片终点(与训练 --slice_end 一致); 默认 None = 全部分块")
    p.add_argument("--n_images", type=int, default=3, help="展示几张图(行)")
    p.add_argument("--steps", default="", help="展示哪些采样步(逗号分隔); 空=自动选 6 个")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="output/phase1_v2_block/recon_visual.png")
    return p.parse_args()


def patch_to_img(pix_patches, H, W):
    """(B,N,14,14,3) 归一化像素 patch → (B,H,W,3) 反归一化 0-255 (uint8)。
    布局与 DINO 一致: row-major (先 y 后 x), N = (H/14)*(W/14)。
    """
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
    model = SRPhase1V2(dinov2=dino, num_patches=num_patches,
                       dim=dino.config.hidden_size, reencoder_depth=4,
                       register_specials=args.register_specials,
                       decoder_depth=args.decoder_depth,
                       skip_steps=args.slice_start,
                       max_steps=args.slice_end)
    sd = torch.load(args.final_model, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval().cuda()
    T_steps = model.decoder.steps
    print(f"[model] N={num_patches}, register_specials={args.register_specials}, "
          f"decoder_depth={args.decoder_depth}, slice=[{args.slice_start}:{args.slice_end}], "
          f"{len(T_steps)} 采样步 {T_steps}")

    # 自动选展示步: 前/中/后均匀取 (含最后一步 = 全量累加结果)
    if args.steps.strip():
        show_steps = [int(s) for s in args.steps.split(",") if s.strip()]
    else:
        n_show = min(5, len(T_steps))
        idx = np.linspace(0, len(T_steps) - 1, n_show).astype(int)
        show_steps = [T_steps[i] for i in idx]
        if T_steps[-1] not in show_steps:
            show_steps.append(T_steps[-1])
    show_idx = [T_steps.index(t) for t in show_steps]
    print(f"[show] 采样步: {show_steps}")

    # 数据: 取前 n_images 张 test
    test_files = sorted(glob.glob(os.path.join(args.data_dir, "test-*.parquet")))
    ds = ParquetImageDataset(test_files, limit=args.n_images)
    loader = DataLoader(ds, batch_size=args.n_images, shuffle=False,
                        num_workers=4, collate_fn=V2Collator(model_size=(W, H)))

    imgs, recons = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["pixel_values"].cuda()
            B, C, Hh, Ww = x.shape
            out = model(x)                              # 同一 forward（两种模式通用）
            Y_pix = out["Y_pix"]                        # (B,|T|,N,588)
            target = out["target_pix"]                  # (B,N,588)
            # 原图 (反归一化)
            img0 = patch_to_img(target, Hh, Ww)             # (B,H,W,3) uint8
            # 各展示步重建
            rec = [patch_to_img(Y_pix[:, i], Hh, Ww) for i in show_idx]
            imgs.append(img0)
            recons.append(rec)
            break
    imgs = imgs[0]
    recons = recons[0]                                       # 每元素 (B,H,W,3)

    # ── 画图 ──
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
    n_cols = 1 + len(show_idx)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.0 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    titles = ["原图"] + [f"累积 {i + 1} 步 (t={t})" for i, t in enumerate(show_steps)]
    for b in range(n_rows):
        for c in range(n_cols):
            ax = axes[b, c]
            if c == 0:
                ax.imshow(imgs[b])
            else:
                ax.imshow(recons[c - 1][b])
            if b == 0:
                ax.set_title(titles[c], fontsize=12, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle("像素级重建可视化 — 原图 vs 各采样步累积重建 (分块掩码, 累加集成)\n"
                 "目标 = 原始像素 (PixelHead), fp32 平权训练, register_specials",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
