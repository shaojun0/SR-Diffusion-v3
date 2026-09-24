"""12 组实验的统一训练/评估 harness。

统一口径（所有 12 组完全一致，唯一变量 = backbone x 分辨率）：
  * 数据    : DIV2K train HR 800 张 (train) / valid HR 100 张 (val)，短边缩放+中心裁剪到 SxS
  * 目标    : [0,1] RGB
  * 损失    : L1（直接重建，无 deep supervision）
  * 优化器  : AdamW, lr 2e-4, weight_decay 1e-4, cosine schedule, 200 warmup steps
  * epochs  : 40（用户指定，统一）
  * batch   : 自动按显存选（默认 8，448² 下大 backbone 会降）
  * 混合精度: bf16 autocast（4080 SUPER 支持），fp32 参数
  * 指标    : val L1（[0,1]）、PSNR、SSIM
  * 输出    : <out>/config.json, metrics.json, history.json, best.pt, last.pt, train.log
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
warnings.filterwarnings("ignore")
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
try:
    import transformers.utils.logging as _tul
    _tul.set_verbosity_error()
    _tul.disable_progress_bar()
except Exception:
    pass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import BACKBONES, build_model


# ---------------------------------------------------------------------------
class NpyImages(Dataset):
    def __init__(self, path: str, train: bool):
        self.arr = np.load(path, mmap_mode="r")
        self.train = train

    def __len__(self):
        return self.arr.shape[0]

    def __getitem__(self, i):
        img = np.asarray(self.arr[i], dtype=np.float32) / 255.0
        t = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        if self.train and torch.rand(1).item() < 0.5:
            t = torch.flip(t, dims=[2])
        return t


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """简化 SSIM（11x11 高斯窗，逐通道平均）。a/b: (N,3,H,W) in [0,1]。"""
    win = 11
    sigma = 1.5
    coords = torch.arange(win, dtype=a.dtype, device=a.device) - win // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    kernel = (g.t() @ g).expand(a.shape[1], 1, win, win).contiguous()
    pad = win // 2
    mu_a = F.conv2d(a, kernel, padding=pad, groups=a.shape[1])
    mu_b = F.conv2d(b, kernel, padding=pad, groups=a.shape[1])
    va = F.conv2d(a * a, kernel, padding=pad, groups=a.shape[1]) - mu_a ** 2
    vb = F.conv2d(b * b, kernel, padding=pad, groups=a.shape[1]) - mu_b ** 2
    vab = F.conv2d(a * b, kernel, padding=pad, groups=a.shape[1]) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * vab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return float(s.mean())


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    tot_l1, tot_psnr, tot_ssim, n = 0.0, 0.0, 0.0, 0
    for x in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            rec = model(x)
        rec = rec.float().clamp(0, 1)
        l1 = F.l1_loss(rec, x).item()
        mse = F.mse_loss(rec, x).item()
        psnr = 10 * math.log10(1.0 / max(mse, 1e-12))
        tot_l1 += l1 * x.shape[0]
        tot_psnr += psnr * x.shape[0]
        tot_ssim += ssim(rec, x) * x.shape[0]
        n += x.shape[0]
    model.train()
    return {"l1": tot_l1 / n, "psnr": tot_psnr / n, "ssim": tot_ssim / n}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=list(BACKBONES))
    ap.add_argument("--resolution", type=int, required=True, choices=[224, 448])
    ap.add_argument("--data_dir", default="/root/autodl-tmp/rrnet/data")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=0, help="0=自动")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lstm_hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_train", type=int, default=0, help="冒烟测试用")
    ap.add_argument("--max_steps", type=int, default=0, help="冒烟测试用")
    ap.add_argument("--no_pretrained", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "train.log")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else None

    model = build_model(args.backbone, args.resolution, pretrained=not args.no_pretrained,
                        lstm_hidden=args.lstm_hidden).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"=== {args.backbone} @ {args.resolution} ===")
    try:
        log(model.load_report())
    except Exception as e:  # pragma: no cover
        log(f"load_report failed: {e!r}")
    log(f"params total={n_param/1e6:.2f}M trainable={n_train/1e6:.2f}M amp={'bf16' if amp_dtype else 'fp32'}")

    tr_ds = NpyImages(os.path.join(args.data_dir, f"div2k_train_{args.resolution}.npy"), True)
    va_ds = NpyImages(os.path.join(args.data_dir, f"div2k_val_{args.resolution}.npy"), False)
    if args.limit_train:
        tr_ds.arr = tr_ds.arr[: args.limit_train]
        va_ds.arr = va_ds.arr[: max(8, args.limit_train // 4)]

    # 自动 batch：从目标值往下试，OOM 就减半
    bs = args.batch_size or (16 if args.resolution == 224 else 8)
    while True:
        try:
            probe = next(iter(DataLoader(tr_ds, batch_size=bs, shuffle=True))).to(device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = F.l1_loss(model(probe), probe)
            loss.backward()
            model.zero_grad(set_to_none=True)
            del probe, loss
            torch.cuda.empty_cache()
            log(f"auto batch_size = {bs}")
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs = max(1, bs // 2)
            log(f"OOM, retry batch_size = {bs}")

    tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True, num_workers=args.num_workers,
                           drop_last=True, pin_memory=True, persistent_workers=args.num_workers > 0)
    va_loader = DataLoader(va_ds, batch_size=max(1, bs), shuffle=False, num_workers=2, pin_memory=True)

    steps_per_epoch = len(tr_loader)
    total_steps = steps_per_epoch * args.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / args.warmup_steps
        p = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * args.lr * (1 + math.cos(math.pi * min(1.0, p)))

    json.dump(vars(args) | {"batch_size_used": bs, "steps_per_epoch": steps_per_epoch,
                            "total_steps": total_steps, "params_M": n_param / 1e6,
                            "params_trainable_M": n_train / 1e6},
              open(os.path.join(args.out_dir, "config.json"), "w"), indent=2)

    history, best_l1, gstep = [], float("inf"), 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run, nb = 0.0, 0
        for x in tr_loader:
            x = x.to(device, non_blocking=True)
            for g in opt.param_groups:
                g["lr"] = lr_at(gstep)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                rec = model(x)
                loss = F.l1_loss(rec, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            run += loss.item()
            nb += 1
            gstep += 1
            if args.max_steps and gstep >= args.max_steps:
                break
            if nb % 50 == 0:
                el = time.time() - t0
                log(f"  ep{epoch} step{nb}/{steps_per_epoch} g{gstep} loss={run/nb:.5f} "
                    f"lr={lr_at(gstep):.2e} {el/max(1,gstep):.2f}s/it")
        tr_l1 = run / max(1, nb)
        if args.max_steps and gstep >= args.max_steps:
            log("max_steps reached -> stop")
            break
        vm = evaluate(model, va_loader, device, amp_dtype)
        el = time.time() - t0
        log(f"epoch {epoch}/{args.epochs} train_l1={tr_l1:.5f} val_l1={vm['l1']:.5f} "
            f"val_psnr={vm['psnr']:.3f} val_ssim={vm['ssim']:.4f} elapsed={el/60:.1f}m")
        history.append({"epoch": epoch, "train_l1": tr_l1, **vm, "elapsed_s": el})
        torch.save({"model": model.state_dict(), "epoch": epoch, "val": vm,
                    "args": vars(args)}, os.path.join(args.out_dir, "last.pt"))
        if vm["l1"] < best_l1:
            best_l1 = vm["l1"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": vm,
                        "args": vars(args)}, os.path.join(args.out_dir, "best.pt"))
        json.dump(history, open(os.path.join(args.out_dir, "history.json"), "w"), indent=2)

    json.dump({"backbone": args.backbone, "resolution": args.resolution,
               "best_val_l1": best_l1, "epochs_done": len(history),
               "batch_size": bs, "params_M": n_param / 1e6,
               "params_trainable_M": n_train / 1e6,
               "total_time_min": (time.time() - t0) / 60,
               "final": history[-1] if history else None},
              open(os.path.join(args.out_dir, "metrics.json"), "w"), indent=2)
    log(f"DONE best_val_l1={best_l1:.5f} total={(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
