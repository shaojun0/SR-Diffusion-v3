"""train_multi — MultiResSR 的自然图像微调训练循环（2026-09-26）

- 纯 PyTorch DDP（torchrun --nproc_per_node=2），不用 HF Trainer：
  需要自定义「按编码器输入形状分桶」的 batch sampler 与逐解码器加权损失。
- bf16 autocast（默认）或纯 fp32；编码器梯度检查点默认开（显存实测见 probe）。
- DINOv2 与新模块分组学习率（历史 v4 最优配方：DINO 1.5e-4 / 新模块 3e-4）。

用法：
  torchrun --nproc_per_node=2 train_multi.py --output_dir output/multires --max_steps 2000 ...
"""
import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from data_multi import (DEFAULT_SOURCES, BucketBatchSampler, MultiResCollator,
                        MultiSourceDataset, build_index)
from model_multi import RESOS, MultiResSR


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dino_dir", default="/root/autodl-tmp/models/dinov2-large")
    p.add_argument("--output_dir", default="/root/autodl-tmp/output/multires")
    p.add_argument("--data_cache", default="/root/autodl-tmp/data/index_cache.json")
    p.add_argument("--data_root", default="/root/autodl-tmp/data")
    p.add_argument("--long_side_cap", type=int, default=1280)
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8, help="每卡 batch 上限（同桶内）")
    p.add_argument("--patch_budget", type=int, default=8192,
                   help="每批编码器 patch 数上限：bs = min(batch_size, budget/N)，"
                        "patch 越大的桶 batch 越小（0=关闭）")
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr_new", type=float, default=3e-4)
    p.add_argument("--lr_dino", type=float, default=1.5e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_specials", type=int, default=144)
    p.add_argument("--decoder_depth", type=int, default=2)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--step_plan", default="square", choices=["square", "fixed"])
    p.add_argument("--block", type=int, default=12)
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--no_grad_ckpt", action="store_true")
    p.add_argument("--grad_ckpt_decoder", action="store_true",
                   help="4 个解码器各整段 12 步做一次检查点重算（省激活, 慢 ~30%%）")
    p.add_argument("--freeze_dino", action="store_true")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true", help="只跑几十步 + 打印显存")
    p.add_argument("--limit", type=int, default=0, help="只用前 N 条样本（调试）")
    p.add_argument("--only_patches", type=int, default=0,
                   help="只保留编码器 patch 数 = N 的桶（显存冒烟用）")
    p.add_argument("--resume", default="")
    p.add_argument("--reset_step_on_resume", action="store_true",
                   help="载入 ckpt 的权重/优化器但把优化步计数归零 —— 开新一轮 epoch "
                        "（新的 warmup+cosine）时必须开，否则 lr_scale(大 step)≈0 且 "
                        "step>=max_steps 会立刻退出")
    return p.parse_args()


def build_sources(root, caps=None):
    caps = caps or {}
    out = []
    for path, exts, cap in DEFAULT_SOURCES:
        rel = os.path.relpath(path, "/root/autodl-tmp/data")
        out.append((os.path.join(root, rel), exts, caps.get(rel, cap)))
    return out


def main():
    args = get_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.manual_seed(args.seed + rank)
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    if world > 1:
        dist.init_process_group("nccl")
    is_main = rank == 0

    def log(*a):
        if is_main:
            print(*a, flush=True)

    log(f"[env] world={world} local_rank={local_rank} "
        f"gpu={torch.cuda.get_device_name(local_rank)} precision={args.precision}")

    # ── 数据索引（rank0 建 + 落盘, 其余等文件）──
    if not os.path.exists(args.data_cache):
        if is_main:
            os.makedirs(os.path.dirname(args.data_cache), exist_ok=True)
            build_index(build_sources(args.data_root), args.data_cache,
                        args.long_side_cap)
        if world > 1:
            dist.barrier()
    index = build_index(sources=[], cache=args.data_cache) if os.path.exists(args.data_cache) \
        else build_index(build_sources(args.data_root), "", args.long_side_cap)
    if args.limit:
        index = index[:args.limit]
    if args.only_patches:
        index = [r for r in index
                 if (r[1] // 14) * (r[2] // 14) == args.only_patches]
    if not index:
        raise SystemExit("[data] 过滤后没有样本（检查 --only_patches / 数据目录）")
    shapes = [(bh, bw) for (_, bw, bh, _) in index]
    log(f"[data] 样本 {len(index)} | 桶 {len(set(shapes))} 种 | "
        f"top5 {sorted(((shapes.count(s), s) for s in set(shapes)), reverse=True)[:5]}")

    ds = MultiSourceDataset(index, RESOS)
    sampler = BucketBatchSampler(shapes, args.batch_size, shuffle=True,
                                 seed=args.seed, drop_last=True,
                                 rank=rank, world_size=world,
                                 patch_budget=args.patch_budget)
    dl = DataLoader(ds, batch_sampler=sampler, num_workers=args.num_workers,
                    collate_fn=MultiResCollator(), pin_memory=True,
                    persistent_workers=args.num_workers > 0,
                    prefetch_factor=4 if args.num_workers > 0 else None)

    # ── 模型 ──
    from transformers import Dinov2Model
    dino = Dinov2Model.from_pretrained(args.dino_dir, attn_implementation="sdpa")
    dim = dino.config.hidden_size
    if not args.no_grad_ckpt:
        dino.gradient_checkpointing_enable()
    model = MultiResSR(dino, num_specials=args.num_specials, dim=dim,
                       heads=args.heads, depth=args.decoder_depth,
                       mlp_ratio=args.mlp_ratio, step_plan=args.step_plan,
                       block=args.block,
                       decoder_ckpt=args.grad_ckpt_decoder)
    if args.freeze_dino:
        for p in model.dinov2.parameters():
            p.requires_grad_(False)
    # HF Dinov2 的 mask_token 在我们的 register 路径里用不到 ⇒ 必须关梯度,
    # 否则 DDP(find_unused_parameters=False) 会因为未使用参数报错
    for n, p in model.dinov2.named_parameters():
        if "mask_token" in n:
            p.requires_grad_(False)
    model = model.to(dev)
    n_all = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[model] 参数 {n_all/1e6:.1f}M, 可训 {n_tr/1e6:.1f}M, K={args.num_specials}, "
        f"steps={model.steps}")

    dino_ids = {id(p) for p in model.dinov2.parameters()}
    groups = [
        {"params": [p for p in model.parameters() if p.requires_grad and id(p) not in dino_ids],
         "lr": args.lr_new, "base_lr": args.lr_new, "name": "new"},
        {"params": [p for p in model.parameters() if p.requires_grad and id(p) in dino_ids],
         "lr": args.lr_dino, "base_lr": args.lr_dino, "name": "dino"},
    ]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    warmup = max(1, int(args.max_steps * args.warmup_ratio))

    def lr_scale(step):
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, args.max_steps - warmup)
        return 0.5 * (1 + torch.cos(torch.tensor(prog * 3.141592653589793))).item()

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = 0 if args.reset_step_on_resume else ck.get("step", 0)
        log(f"[resume] {args.resume} @ ckpt_step={ck.get('step')} "
            f"→ start_step={start_step} (reset={bool(args.reset_step_on_resume)})")

    if world > 1:
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=False, gradient_as_bucket_view=True)

    os.makedirs(args.output_dir, exist_ok=True)
    jsonl = open(os.path.join(args.output_dir, "train_log.jsonl"), "a")
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32

    step = start_step
    t0 = time.time()
    running, running_n = 0.0, 0
    stop = False
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for it, batch in enumerate(dl):
            if step >= args.max_steps:
                stop = True
                break
            px = batch["pixel_values"].to(dev, non_blocking=True)
            tgt = [t.to(dev, non_blocking=True) for t in batch["targets"]]
            best = batch["best_idx"].to(dev, non_blocking=True)
            ctx = (torch.autocast("cuda", dtype=amp_dtype)
                   if args.precision == "bf16" else torch.enable_grad())
            with ctx:
                out = model(px, tgt, best)
                loss = out["loss"] / args.grad_accum
            loss.backward()
            if (it + 1) % args.grad_accum == 0:
                gnorm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.grad_clip)
                sc = lr_scale(step)
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * sc
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                running += float(out["loss"].detach())
                running_n += 1
                if step % args.log_every == 0:
                    peak = torch.cuda.max_memory_allocated(dev) / 2 ** 30
                    rec = {"step": step, "epoch": epoch,
                           "loss": round(running / max(1, running_n), 5),
                           "recon": round(float(out["recon"]), 5),
                           "per_decoder": [round(float(x), 5) for x in out["per_decoder"]],
                           "grad_norm": round(float(gnorm), 4),
                           "lr_new": round(opt.param_groups[0]["lr"], 8),
                           "peak_gib": round(peak, 2),
                           "sec_per_step": round((time.time() - t0) / max(1, step - start_step), 3)}
                    jsonl.write(json.dumps(rec) + "\n")
                    jsonl.flush()
                    log(f"[{step}/{args.max_steps}] loss={rec['loss']} recon={rec['recon']} "
                        f"dec={rec['per_decoder']} gnorm={rec['grad_norm']} "
                        f"peak={rec['peak_gib']}GiB {rec['sec_per_step']}s/step")
                    running, running_n = 0.0, 0
                if step % args.save_every == 0 and is_main:
                    raw = model.module if world > 1 else model
                    torch.save({"model": raw.state_dict(), "opt": opt.state_dict(),
                                "step": step, "args": vars(args)},
                               os.path.join(args.output_dir, f"ckpt_{step}.pt"))
                    log(f"[save] ckpt_{step}.pt")
            if args.smoke and step >= start_step + 6:
                stop = True
                break
        if stop:
            break

    if is_main:
        raw = model.module if world > 1 else model
        torch.save({"model": raw.state_dict(), "opt": opt.state_dict(),
                    "step": step, "args": vars(args)},
                   os.path.join(args.output_dir, "final_model.pt"))
        peak = torch.cuda.max_memory_allocated(dev) / 2 ** 30
        log(f"[done] step={step} peak={peak:.2f}GiB → {args.output_dir}/final_model.pt")
    jsonl.close()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
