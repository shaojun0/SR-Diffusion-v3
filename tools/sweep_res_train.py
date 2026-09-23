"""sweep_res_train.py — 分辨率扫描：小模型（DINOv2-small）+ 逐 step nested RD

用户想法（2026-09-22）：图像大小从 ~32×32 到更大，看**各个 step 的 L1 变化率**，
并跟 JPEG 比。本脚本负责"每个分辨率训一个模型 + 逐 step 评估"这一半。

口径（与仓库对齐，见 tools/analyze_earlystop.py）:
  · 预处理: 原图 → **直接 resize 到 S×S**（不做 fit_to_canvas letterbox，
    这样 px = S² 精确、与 JPEG 的食物完全相同；仓库主线是 448×252 画布，
    两者不可直接比数，见报告"口径声明"）
  · 编码器: DINOv2-small (D=384, 12 层)，不冻结；register specials K=N
  · 解码器: OutputQueryDecoder（顺序循环 / 平方块读窗口 / 直接预测损失）
  · 采样步: square_block_starts(N)（全量）
  · carry: 默认 **BPTT**（carry_detach=False）；--detach 可切历史默认
  · bpp = (t+1)·D·β / S²（β=1，与仓库同一"估计 bpp"口径，非真实熵编码）
  · 指标: 0-255 空间 L1 与 PSNR（PSNR 由聚合 MSE 反推 = PSNR(agg)）

用法:
  python sweep_res_train.py --size 112 --steps 1500 --out_dir /root/autodl-tmp/srres/out
  python sweep_res_train.py --size 28  --steps 200 --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import Dinov2Model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_v2 import SRPhase1V2, patches_to_image, square_block_starts
from data_v2 import DINO_MEAN, DINO_STD

MEAN = [float(v) for v in DINO_MEAN]
STD = [float(v) for v in DINO_STD]
DIM_SMALL = 384          # DINOv2-small 隐层维（bpp 里那个 D）


# ─────────────────────────── 数据：直接 resize 到 S×S ───────────────────────────
_MEAN_T = torch.tensor(MEAN).view(3, 1, 1)
_STD_T = torch.tensor(STD).view(3, 1, 1)


class ResizeDS(Dataset):
    """优先读 sweep_res_prep.py 预处理好的 .npy（mmap）：避免每 epoch 重解码
    2040×1404 PNG（实测那是 CPU 瓶颈，GPU util 只有 6%）。没有 .npy 时回落到
    直接读 PNG。"""

    def __init__(self, files, size, train=False, npy=None):
        self.files, self.size, self.train = files, size, train
        self.arr = np.load(npy, mmap_mode="r") if npy and os.path.exists(npy) else None

    def __len__(self):
        return len(self.arr) if self.arr is not None else len(self.files)

    def __getitem__(self, i):
        if self.arr is not None:
            a = np.asarray(self.arr[i], np.float32)           # (S,S,3)
        else:
            img = Image.open(self.files[i]).convert("RGB").resize(
                (self.size, self.size), Image.BICUBIC)
            a = np.asarray(img, np.float32)
        if self.train and np.random.rand() < 0.5:
            a = np.ascontiguousarray(a[:, ::-1])
        t = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)   # (3,H,W)
        return (t - _MEAN_T) / _STD_T


# ─────────────────────────── 模型构造 ───────────────────────────
def use_patch_tokens(model):
    """把 z_s 从 register 输出改成**patch token 输出**（其余完全不变）。

    动机：register 是"共享向量+逐位置 pos"，编码器（DINOv2 预训练时没见过
    register）要从零学会把内容路由进 register —— 实测在 800 张 DIV2K、
    千级步数内学不动（见报告"为什么用 patch token"）。patch token 本身就
    带内容，且块窗口 step k 读的正是 patch k²..(k+1)²−1，逐块读出 ≈ 逐块
    重建，学得快得多。**解码器 / carry / BPTT / 块窗口与仓库逐位一致**，
    变的只有"z_s 取编码器的哪一段输出"。
    """
    def enc(pixel_values):
        emb = model.dinov2.embeddings(pixel_values)
        B = pixel_values.shape[0]
        seq = torch.cat([emb[:, :1], model.special_bank(B, pixel_values.device),
                         emb[:, 1:]], dim=1)
        for layer in model.dinov2.encoder.layer:
            out = layer(seq)
            seq = out[0] if isinstance(out, (tuple, list)) else out
        seq = model.dinov2.layernorm(seq)
        K = model.num_specials
        return seq[:, :1], seq[:, 1 + K:1 + K + K]
    model._encode_register = enc
    return model


def build_model(model_dir, N, depth, stack_dim=0):
    # transformers>=5 的 Dinov2Embeddings.forward 内部**总会**调
    # self.interpolate_pos_encoding(emb, H, W) ⇒ 任意 14 倍数分辨率开箱可用,
    # 不需要（也不接受）interpolate_pos_encoding 开关。
    dino = Dinov2Model.from_pretrained(model_dir)
    steps = square_block_starts(N)
    m = SRPhase1V2(dinov2=dino, num_patches=N, dim=DIM_SMALL, heads=8,
                   decoder_steps=steps, decoder_depth=depth, patch_px=588,
                   mlp_ratio=4.0, stack_dim=stack_dim)
    return m, steps


# ─────────────────────────── 评估：逐 step 曲线 ───────────────────────────
@torch.no_grad()
def evaluate(model, loader, size, steps, beta, device):
    model.eval()
    N = (size // 14) ** 2
    T = len(steps)
    l1s = np.zeros(T)
    mses = np.zeros(T)
    npx = 0
    for x in loader:
        x = x.to(device, non_blocking=True)
        o = model(x)
        B = x.shape[0]
        Y, tgt = o["Y_pix"], o["target_pix"]                 # (B,T,N,588)/(B,N,588)
        rec = patches_to_image(Y.reshape(B * T, N, 588), size, size, MEAN, STD)
        rec = rec.reshape(B, T, size, size, 3)
        gt = patches_to_image(tgt, size, size, MEAN, STD)     # (B,size,size,3)
        gt = gt.unsqueeze(1)
        d = (rec - gt)
        l1s += d.abs().sum(dim=(0, 2, 3, 4)).cpu().numpy()
        mses += (d ** 2).sum(dim=(0, 2, 3, 4)).cpu().numpy()
        npx += B * size * size * 3
    l1s /= npx
    mses /= npx
    psnr = 10.0 * np.log10(255.0 ** 2 / np.maximum(mses, 1e-12))
    tokens = [t + 1 for t in steps]
    bpp = [tk * DIM_SMALL * beta / (size * size) for tk in tokens]
    model.train()
    return {"l1_255": l1s.tolist(), "mse": mses.tolist(), "psnr": psnr.tolist(),
            "tokens": tokens, "bpp": bpp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True, help="图像边长（14 的倍数）")
    ap.add_argument("--train_dir", default="/root/autodl-tmp/srres/data/DIV2K_train_HR")
    ap.add_argument("--val_dir", default="/root/autodl-tmp/srres/data/DIV2K_valid_HR")
    ap.add_argument("--data_root", default="/root/autodl-tmp/srres/data",
                    help="预处理 .npy 所在目录（sweep_res_prep.py 的 --root）")
    ap.add_argument("--model_dir", default="/root/autodl-tmp/srres/models/dinov2-small")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/srres/out")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1, help="梯度累积（有效 bs=batch*accum）")
    ap.add_argument("--lr", type=float, default=1e-3, help="解码器/头/register 的 lr")
    ap.add_argument("--enc_lr_mult", type=float, default=0.1, help="编码器 lr 倍率")
    ap.add_argument("--clip", type=float, default=1.0, help="梯度裁剪范数")
    ap.add_argument("--warm_steps", type=int, default=0,
                    help="A 相：单步+全读 warm start 步数（0=关）")
    ap.add_argument("--head_zero_init", action="store_true",
                    help="PixelHead 末层置零（起点=预测均值，梯度干净）")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--limit_train", type=int, default=0)
    ap.add_argument("--limit_val", type=int, default=0)
    ap.add_argument("--detach", action="store_true", help="关掉 BPTT（历史默认）")
    ap.add_argument("--z_mode", default="register", choices=["register", "patch"],
                    help="z_s 取 register 输出(仓库原版) 还是 patch token 输出")
    ap.add_argument("--arm", default="", help="tag 后缀，默认 BPTT/detach 自动")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_every", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    assert args.size % 14 == 0, "边长必须是 14 的倍数（DINOv2 patch=14）"
    if args.smoke:
        args.steps, args.limit_train, args.limit_val = 60, 64, 16

    torch.set_float32_matmul_precision("high")     # TF32，提速；见报告口径声明
    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    N = (args.size // 14) ** 2
    arm = args.arm or ("detach" if args.detach else "bptt")
    if args.z_mode != "register":
        arm = f"{arm}-{args.z_mode}"
    tag = f"{args.size}_{arm}"
    out = os.path.join(args.out_dir, tag)
    os.makedirs(out, exist_ok=True)

    tr_files = sorted(os.path.join(args.train_dir, f)
                      for f in os.listdir(args.train_dir) if f.endswith(".png"))
    va_files = sorted(os.path.join(args.val_dir, f)
                      for f in os.listdir(args.val_dir) if f.endswith(".png"))
    if args.limit_train:
        tr_files = tr_files[:args.limit_train]
    if args.limit_val:
        va_files = va_files[:args.limit_val]
    print(f"[data] size={args.size} N={N} train={len(tr_files)} val={len(va_files)}",
          flush=True)

    model, steps = build_model(args.model_dir, N, args.depth)
    if args.z_mode == "patch":
        use_patch_tokens(model)
    model.decoder.carry_detach = bool(args.detach)
    if args.head_zero_init:
        # PixelHead 末层置零 ⇒ 起点 = "预测归一化均值(0)"，loss≈0.8 而不是
        # 随机大输出(实测 loss 1.3~4.3 / gn 10~100)。仓库主线从预训练
        # DINOv2-large 起步，本来就在好区域；小模型从零训需要这个。
        nn.init.zeros_(model.pixel_head.net[-1].weight)
        nn.init.zeros_(model.pixel_head.net[-1].bias)
    model = model.to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[model] steps={steps} (|T|={len(steps)}) params={nparam/1e6:.1f}M "
          f"arm={arm} carry_detach={model.decoder.carry_detach} "
          f"head_zero_init={args.head_zero_init}", flush=True)

    npy_tr = os.path.join(args.data_root, f"S{args.size}_train.npy")
    npy_va = os.path.join(args.data_root, f"S{args.size}_val.npy")
    tl = DataLoader(ResizeDS(tr_files, args.size, True, npy_tr),
                    batch_size=args.batch, shuffle=True, num_workers=4,
                    drop_last=True, pin_memory=True)
    vl = DataLoader(ResizeDS(va_files, args.size, False, npy_va),
                    batch_size=args.batch, shuffle=False, num_workers=3,
                    pin_memory=True)
    print(f"[data] npy_train={os.path.exists(npy_tr)} npy_val={os.path.exists(npy_va)}",
          flush=True)

    # 参数分组: 编码器（预训练）小 lr, 解码器/像素头/register 大 lr
    enc_ids = {id(p) for p in model.dinov2.parameters()}
    enc_p = [p for p in model.parameters() if id(p) in enc_ids]
    oth_p = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW(
        [{"params": enc_p, "lr": args.lr * args.enc_lr_mult},
         {"params": oth_p, "lr": args.lr}],
        lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))
    base_lr = [g["lr"] for g in opt.param_groups]

    def set_lr(mult):
        for g, b in zip(opt.param_groups, base_lr):
            g["lr"] = b * mult

    def sched(it, total, warmup):
        if it < warmup:
            return (it + 1) / warmup
        p = (it - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    data_iter = iter(tl)

    def run_phase(tag2, n, warmup):
        """跑 n 个优化步（含梯度累积）；返回平均 loss。"""
        nonlocal data_iter
        model.train()
        acc = 0.0
        for it in range(n):
            set_lr(sched(it, n, warmup))
            opt.zero_grad(set_to_none=True)
            tot = 0.0
            for _ in range(args.accum):
                try:
                    x = next(data_iter)
                except StopIteration:
                    data_iter = iter(tl)
                    x = next(data_iter)
                x = x.to(device, non_blocking=True)
                l = model(x)["loss"]
                tot += float(l)
                (l / args.accum).backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            acc += tot / args.accum
            if it % 50 == 0:
                print(f"[{tag}] {tag2} it{it} loss={tot/args.accum:.4f} "
                      f"lr={base_lr[-1]*sched(it, n, warmup):.2e} gn={float(gn):.2f} "
                      f"{time.time()-t0:.0f}s", flush=True)
            if args.eval_every and it and it % args.eval_every == 0:
                ev = evaluate(model, vl, args.size, model.decoder.steps, 1.0, device)
                hist.append({"phase": tag2, "it": it, "l1_255": ev["l1_255"],
                             "psnr": ev["psnr"]})
        return acc / max(1, n)

    hist = []
    t0 = time.time()
    # ── A 相: 单步 + 全读 warm start（仓库 cpu_probe_bptt_toy 的同一配方）──
    if args.warm_steps > 0:
        model.decoder.steps = [N]
        model.decoder.carry_detach = True
        run_phase("A", args.warm_steps, args.warmup)
        ev = evaluate(model, vl, args.size, [N], 1.0, device)
        print(f"[{tag}] A 相结束: 全读单步 L1={ev['l1_255'][0]:.2f} "
              f"PSNR={ev['psnr'][0]:.2f}", flush=True)
        model.decoder.steps = steps
        model.decoder.carry_detach = bool(args.detach)

    # ── B 相: 真实多步轨迹 ──
    run_phase("B", args.steps, min(args.warmup, max(1, args.steps // 10)))

    ev = evaluate(model, vl, args.size, steps, 1.0, device)
    res = {"tag": tag, "size": args.size, "num_patches": N, "steps": steps,
           "dim": DIM_SMALL, "depth": args.depth, "arm": arm, "z_mode": args.z_mode,
           "carry_detach": model.decoder.carry_detach,
           "params_M": nparam / 1e6, "train_imgs": len(tr_files),
           "val_imgs": len(va_files), "train_steps": args.steps,
           "warm_steps": args.warm_steps, "head_zero_init": args.head_zero_init,
           "batch": args.batch, "accum": args.accum,
           "eff_batch": args.batch * args.accum, "lr": args.lr,
           "enc_lr_mult": args.enc_lr_mult, "clip": args.clip, "seed": args.seed,
           "tf32": True, "train_secs": round(time.time() - t0, 1),
           "hist": hist, **ev}
    with open(os.path.join(out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    torch.save(model.state_dict(), os.path.join(out, "final_model.pt"))
    with open(os.path.join(out, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print(f"[{tag}] DONE {time.time()-t0:.0f}s  L1: " +
          " ".join(f"{v:.2f}" for v in ev["l1_255"]), flush=True)
    print(f"[{tag}] PSNR: " + " ".join(f"{v:.2f}" for v in ev["psnr"]), flush=True)
    print(f"[{tag}] bpp : " + " ".join(f"{v:.3f}" for v in ev["bpp"]), flush=True)
    print("RESULT_JSON " + os.path.join(out, "result.json"), flush=True)


if __name__ == "__main__":
    main()
