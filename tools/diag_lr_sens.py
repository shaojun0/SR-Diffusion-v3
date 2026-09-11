"""diag_lr_sens.py — 一步 AdamW 的"临界 lr"探针（架构稳定性对比）。

同一 seed、同一 batch, 基线(stack_dim=0) vs 加宽(2048/4/16/0.05) 在**初始化点**
做一步 AdamW(lr), 报告 train/held-out loss 变化。若加宽模型在 1.5e-4 处损失
暴涨而基线下降, 说明加宽后**临界 lr 下降**（宽度/深度缩放问题, 非参数量本身）。
"""
import argparse, copy, glob, json, os
import torch
from torch.utils.data import DataLoader
from transformers import Dinov2Model, set_seed
from data_v2 import ParquetImageDataset, V2Collator
from model_v2 import SRPhase1V2

W, H, N = 448, 252, 576
STEPS = [1, 4, 9, 16, 25]
K, DIM = 35, 1024


def build(dino, s, d, h, p, seed=42):
    torch.manual_seed(seed)
    return SRPhase1V2(dinov2=dino, num_patches=N, dim=DIM, heads=h, mlp_ratio=4.0,
                      decoder_steps=STEPS, decoder_depth=d, skip_steps=0,
                      max_steps=5, num_specials=K, query_mask_mode="blockdiag",
                      stack_dim=s, decoder_dropout=p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dino_dir", default="/root/autodl-tmp/models/dinov2-large")
    ap.add_argument("--data_dir", default="/root/autodl-tmp/construction_site")
    ap.add_argument("--lrs", default="1.5e-4,1.0e-4,7.5e-5,5.0e-5,2.5e-5,1.0e-5")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    set_seed(42)
    tf = sorted(glob.glob(os.path.join(a.data_dir, "train-*.parquet")))
    coll = V2Collator(model_size=(W, H), canvas=(1600, 900), angle_step=0.5)
    ds = ParquetImageDataset(tf, limit=32)
    it = iter(DataLoader(ds, batch_size=16, shuffle=False, num_workers=2,
                         collate_fn=coll))
    xb = next(it)["pixel_values"].cuda()
    xh = next(it)["pixel_values"].cuda()
    dino = Dinov2Model.from_pretrained(a.dino_dir)
    if getattr(dino.config, "use_mask_token", False):
        dino.config.use_mask_token = False
        del dino.embeddings.mask_token

    lrs = [float(v) for v in a.lrs.split(",")]
    res = {}
    for tag, cfg in (("baseline", (0, 2, 8, 0.0)),
                     ("stack2x", (2048, 4, 16, 0.05))):
        m = build(copy.deepcopy(dino), *cfg).cuda()
        m.train()
        l0b, l0h = None, None
        row = {}
        for lr in lrs:
            torch.manual_seed(1234)
            m2 = copy.deepcopy(m)
            opt = torch.optim.AdamW(m2.parameters(), lr=lr, betas=(0.9, 0.999),
                                    eps=1e-8, weight_decay=0.01)
            o = m2(xb)
            l0 = float(o["loss"])
            o["loss"].backward()
            torch.nn.utils.clip_grad_norm_(m2.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                l1 = float(m2(xb)["loss"])
                lh0 = float(m2(xh)["loss"])
            if l0b is None:
                l0b, l0h = l0, lh0
            row[f"lr={lr:g}"] = {"loss_at_init": l0, "loss_after_1step": l1,
                                 "delta_train": l1 - l0, "loss_heldout": lh0}
        res[tag] = row
        print(f"[{tag}] " + json.dumps(row, indent=1), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)
        print("[out]", a.out)


if __name__ == "__main__":
    main()
