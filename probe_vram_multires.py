"""显存探针：可变 N 的编码器（不设最大尺寸）+ 固定 4 分辨率解码器（K=144, 12 步）。

用法示例:
  python probe_vram_multires.py enc --bs 1 --dtypes fp32,bf16 --ckpt 0,1
  python probe_vram_multires.py dec --bs 1 --dtypes fp32,bf16
  python probe_vram_multires.py all --bs 1
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GIB = 2 ** 30
K = 144


def human(b):
    return f"{b / GIB:.2f} GiB"


def pick_hw(n_patches):
    """选一对 14 的倍数 (h,w) 使 patch 数 ≈ n_patches（尽量接近正方形）。"""
    best = None
    for gh in range(1, 200):
        for gw in (max(1, n_patches // gh), max(1, n_patches // gh) + 1):
            n = gh * gw
            if best is None or abs(n - n_patches) < abs(best[0] - n_patches):
                best = (n, gh, gw)
    n, gh, gw = best
    return n, gh * 14, gw * 14


# ══════════════════════ 编码器探针 ══════════════════════

def probe_encoder(dino_path, counts, dtype, ckpt, train_weights, bs, results):
    from transformers import Dinov2Model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = Dinov2Model.from_pretrained(dino_path, attn_implementation="sdpa")
    model.train()
    if not train_weights:
        for p in model.parameters():
            p.requires_grad_(False)
    if ckpt:
        model.gradient_checkpointing_enable()
    if dtype == "bf16":
        model = model.to(torch.bfloat16)
    model = model.cuda()
    D = model.config.hidden_size
    # 参数量显存（模型权重常驻）
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    for n_target in counts:
        N, H, W = pick_hw(n_target)
        if N < K:
            results.append(dict(probe="enc", N=N, H=H, W=W, bs=bs, dtype=dtype,
                                ckpt=ckpt, train_weights=train_weights,
                                status="skip N<K", peak=None))
            continue
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            x = torch.randn(bs, 3, H, W, device="cuda",
                            dtype=torch.bfloat16 if dtype == "bf16" else torch.float32)
            specials = nn.Parameter(torch.randn(bs, K, D, device="cuda",
                                                dtype=torch.bfloat16 if dtype == "bf16" else torch.float32) * 0.02)
            t0 = time.time()
            emb = model.embeddings(x)                      # (B,1+N,D)
            seq = torch.cat([emb[:, :1], specials, emb[:, 1:]], dim=1)
            out = model.encoder(seq)
            if hasattr(out, "last_hidden_state"):
                seq = out.last_hidden_state
            elif isinstance(out, (tuple, list)):
                seq = out[0]
            else:
                seq = out
            seq = model.layernorm(seq)
            z_cls, z_s = seq[:, :1], seq[:, 1:1 + K]
            loss = (z_cls.float() ** 2).mean() + (z_s.float() ** 2).mean()
            loss.backward()
            torch.cuda.synchronize()
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated()
            results.append(dict(probe="enc", N=N, H=H, W=W, bs=bs, dtype=dtype,
                                ckpt=ckpt, train_weights=train_weights,
                                status="ok", peak=peak, sec=round(dt, 2),
                                param_bytes=param_bytes))
            print(f"[enc] N={N:6d} ({W}x{H}) bs={bs} {dtype} ckpt={int(ckpt)} "
                  f"train_w={int(train_weights)}: peak={human(peak)} "
                  f"({dt:.2f}s) params={human(param_bytes)}", flush=True)
        except torch.cuda.OutOfMemoryError as e:
            results.append(dict(probe="enc", N=N, H=H, W=W, bs=bs, dtype=dtype,
                                ckpt=ckpt, train_weights=train_weights,
                                status="OOM", peak=None, err=str(e)[:120]))
            print(f"[enc] N={N:6d} ({W}x{H}) bs={bs} {dtype} ckpt={int(ckpt)}: OOM", flush=True)
        del x
        torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()


# ══════════════════════ 解码器探针（4 个固定分辨率） ══════════════════════

def probe_decoders(repo, resos, dim, heads, depth, dtype, bs, results,
                   shared_pixel_head=True):
    from model_v2 import OutputQueryDecoder, PixelHead
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    decs = nn.ModuleList([OutputQueryDecoder(dim=dim, num_patches=n,
                                             heads=heads, depth=depth,
                                             num_specials=K)
                          for (w, h, n) in resos]).cuda()
    ph = PixelHead(dim=dim, patch_px=588).cuda()
    if dtype == "bf16":
        decs = decs.to(torch.bfloat16)
        ph = ph.to(torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    try:
        z_cls = torch.randn(bs, 1, dim, device="cuda",
                            dtype=torch.bfloat16 if dtype == "bf16" else torch.float32)
        z_s = torch.randn(bs, K, dim, device="cuda",
                          dtype=torch.bfloat16 if dtype == "bf16" else torch.float32) * 0.02
        t0 = time.time()
        total = 0.0
        for (w, h, n), dec in zip(resos, decs):
            Y = dec(z_cls, z_s)                    # (B,|T|,N,D)
            pix = ph(Y)                            # (B,|T|,N,588)
            total = total + pix.float().abs().mean()
        total.backward()
        torch.cuda.synchronize()
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated()
        results.append(dict(probe="dec", resos=[f"{w}x{h}" for w, h, _ in resos],
                            bs=bs, dtype=dtype, status="ok", peak=peak,
                            sec=round(dt, 2)))
        print(f"[dec] 4x{ [f'{w}x{h}' for w,h,_ in resos] } bs={bs} {dtype}: "
              f"peak={human(peak)} ({dt:.2f}s)", flush=True)
    except torch.cuda.OutOfMemoryError as e:
        results.append(dict(probe="dec", resos=[f"{w}x{h}" for w, h, _ in resos],
                            bs=bs, dtype=dtype, status="OOM", err=str(e)[:120]))
        print(f"[dec] bs={bs} {dtype}: OOM", flush=True)
    del decs, ph
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["enc", "dec", "all"])
    ap.add_argument("--dino", default="/root/autodl-tmp/models/dinov2-large")
    ap.add_argument("--repo", default="/root/autodl-tmp/srdiff-multi")
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--dtypes", default="fp32,bf16")
    ap.add_argument("--ckpt", default="0,1")
    ap.add_argument("--counts", default="256,576,1024,2176,4096,8704,14500")
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--out", default="/root/autodl-tmp/logs/probe_vram.json")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    resos = [(448, 252, 576), (252, 448, 576), (224, 224, 256), (448, 448, 1024)]
    results = []
    print(f"GPU={torch.cuda.get_device_name(0)} total={human(torch.cuda.get_device_properties(0).total_memory)}", flush=True)
    if args.mode in ("enc", "all"):
        counts = [int(c) for c in args.counts.split(",")]
        for dt in args.dtypes.split(","):
            for ck in args.ckpt.split(","):
                for tw in (1, 0):
                    if tw == 0 and ck == "1":
                        continue          # 冻结权重时 checkpointing 无意义
                    probe_encoder(args.dino, counts, dt, ck == "1", bool(tw), args.bs, results)
    if args.mode in ("dec", "all"):
        for dt in args.dtypes.split(","):
            probe_decoders(args.repo, resos, args.dim, args.heads, args.depth,
                           dt, args.bs, results)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("WROTE", args.out, flush=True)


if __name__ == "__main__":
    main()
