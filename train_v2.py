"""
SR-Diffusion Phase 1 v2 — 训练
=================================================
架构: DINOv2-large(参数不冻结) → ReEncoder → FeatureDecoder,
     L1 重建 DINO patch 特征（见 model_v2.py）。

数据: construction_site parquet（中文工地数据）→ 旋转+缩放+填充
     到 1600:900 画布 → 16:9 模型输入（见 data_v2.py）。

解耦 / 不造轮子:
    · 分布式 / 混合精度 / 检查点  → accelerate（save_state/load_state）
    · 调度器                       → transformers get_scheduler (cosine)
    · 优化器                       → torch.optim.AdamW
    · 数据                         → datasets + DataLoader + 自定义 collate

多卡: accelerate launch --multi_gpu --num_processes N
（bf16 在代码内显式 torch.autocast 处理，见下方注释——accelerate 1.14
  的 prepare 对 forward 的 autocast 包装在本环境有 bug: conv 输入 fp32
  而 bias 被自动 cast 成 bf16 报错; 显式 autocast 无此问题）

用法:
    accelerate launch --multi_gpu --num_processes 2 \
        train_v2.py --data_dir /root/autodl-tmp/construction_site \
        --dino_dir /root/autodl-tmp/models/dinov2-large \
        --output_dir output/phase1_v2
"""
import argparse
import glob
import json
import os
import time
from contextlib import nullcontext

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import Dinov2Model, get_scheduler
from accelerate import Accelerator
from accelerate.utils import set_seed

from data_v2 import ParquetImageDataset, V2Collator
from model_v2 import SRPhase1V2


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="train phase1 v2 (DINOv2-large unfrozen)")
    p.add_argument("--data_dir", required=True, help="parquet 目录(train-*.parquet)")
    p.add_argument("--dino_dir", default="models/dinov2-large")
    p.add_argument("--output_dir", default="output/phase1_v2")
    p.add_argument("--model_input", default="448x252", help="16:9 模型输入 WxH (14 的倍数)")
    p.add_argument("--canvas", default="1600x900", help="旋转+缩放+填充目标画布")
    p.add_argument("--angle_step", type=float, default=0.5, help="最优旋转角网格步长(度)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--max_steps", type=int, default=0, help="0 = epochs 决定")
    p.add_argument("--batch_size", type=int, default=8, help="每卡 batch")
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--reencoder_depth", type=int, default=4)
    p.add_argument("--decoder_depth", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="只用前 N 条训练样本(调试)")
    p.add_argument("--eval_limit", type=int, default=0, help="eval 只用前 N 条(冒烟)")
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default=None, help="accelerate checkpoint 目录")
    p.add_argument("--smoke", action="store_true", help="3 步冒烟, 不存盘")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
# Eval — 仅主进程, 全量 test 分片
# ═══════════════════════════════════════════════════════════════

def evaluate(acc: Accelerator, model: torch.nn.Module, loader: DataLoader) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x = batch["pixel_values"].to(acc.device)   # eval loader 未 prepare, 显式搬设备
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
            bs = x.shape[0]
            total += out["loss"].item() * bs
            n += bs
    acc.print(f"[eval] recon L1 = {total / max(n, 1):.6f}  (n={n})")
    return total / max(n, 1)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)

    W, H = (int(v) for v in args.model_input.lower().split("x"))
    CW, CH = (int(v) for v in args.canvas.lower().split("x"))
    assert W % 14 == 0 and H % 14 == 0, "DINOv2-large patch=14, 输入须为 14 的倍数"
    num_patches = (W // 14) * (H // 14)

    # mixed_precision="no" + 显式 torch.autocast(bf16)：本环境（torch 2.13）
    # 在「输入在 CPU、模型在 CUDA」且开 autocast 时 conv 会报 dtype 错，
    # 显式 .to(acc.device) + 显式 autocast 规避；权重保持 fp32（bf16 只算不存）。
    acc = Accelerator(mixed_precision="no", step_scheduler_with_optimizer=False)

    # ── 数据 ──
    train_files = sorted(glob.glob(os.path.join(args.data_dir, "train-*.parquet")))
    test_files = sorted(glob.glob(os.path.join(args.data_dir, "test-*.parquet")))
    assert train_files, f"无 train-*.parquet in {args.data_dir}"
    coll = V2Collator(model_size=(W, H), canvas=(CW, CH), angle_step=args.angle_step)
    train_ds = ParquetImageDataset(train_files, limit=args.limit)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=coll,
                              drop_last=True, pin_memory=True)
    test_loader = None
    if test_files:
        test_ds = ParquetImageDataset(test_files, limit=args.eval_limit)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=coll,
                                 drop_last=False)
    else:
        acc.print("[warn] 无 test-*.parquet, 跳过 eval")

    # ── 模型: DINOv2-large 不冻结 ──
    dino = Dinov2Model.from_pretrained(args.dino_dir)
    # 不调用 requires_grad_(False) —— 参数全部可训练
    model = SRPhase1V2(dinov2=dino, num_patches=num_patches,
                       dim=dino.config.hidden_size,
                       reencoder_depth=args.reencoder_depth,
                       decoder_depth=args.decoder_depth)
    model.init_reencoder_from_dino(args.reencoder_depth)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    acc.print(f"[model] 可训练参数 {n_train / 1e6:.1f}M (含 DINOv2-large, 不冻结); "
              f"输入 {W}x{H}, patches={num_patches}")

    # ── 优化 / 调度 ──
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model, optimizer, train_loader = acc.prepare(model, optimizer, train_loader)

    steps_per_epoch = len(train_loader) // args.grad_accum
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    scheduler = get_scheduler("cosine", optimizer=optimizer,
                              num_warmup_steps=int(total_steps * args.warmup_ratio),
                              num_training_steps=total_steps)
    acc.print(f"[train] {len(train_ds)} 样本 | 每卡 bs={args.batch_size} "
              f"x {acc.num_processes} 卡 | grad_accum={args.grad_accum} | "
              f"~{steps_per_epoch} 步/epoch x {args.epochs} = {total_steps} 步")

    global_step = 0
    if args.resume:
        acc.load_state(args.resume)
        global_step = int(os.path.basename(args.resume).rsplit("-", 1)[-1])
        acc.print(f"[resume] 从 {args.resume} 恢复, 续训于 step {global_step}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # ── 训练循环（手动梯度累积: 显式、无插件魔法）──
    ema_loss, micro, t0 = None, 0, time.time()
    model.train()
    for epoch in range(args.epochs):
        if global_step >= total_steps:
            break
        for batch in train_loader:
            x = batch["pixel_values"].to(acc.device)
            ctx = acc.no_sync(model) if micro % args.grad_accum != 0 else nullcontext()
            with ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(x)["loss"] / args.grad_accum
                acc.backward(loss)

            if micro % args.grad_accum == args.grad_accum - 1:
                acc.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                lv = loss.item() * args.grad_accum
                ema_loss = lv if ema_loss is None else 0.98 * ema_loss + 0.02 * lv

                if global_step % args.log_every == 0:
                    el = time.time() - t0
                    sps = global_step * args.batch_size * acc.num_processes / max(el, 1e-9)
                    acc.print(
                        f"[step {global_step}/{total_steps}] epoch {epoch + 1} | "
                        f"loss {ema_loss:.5f} | lr {scheduler.get_last_lr()[0]:.2e} | "
                        f"{sps:.1f} 样本/s | 已用 {el / 60:.1f}min", flush=True)

                if test_loader is not None and global_step % args.eval_every == 0:
                    if acc.is_main_process:
                        evaluate(acc, model, test_loader)
                    model.train()

                if not args.smoke and global_step % args.save_every == 0:
                    acc.wait_for_everyone()
                    ckpt = os.path.join(args.output_dir, f"ckpt-{global_step}")
                    acc.save_state(ckpt)
                    acc.print(f"[save] {ckpt}", flush=True)

                if global_step >= total_steps:
                    break
            micro += 1

    # ── 收尾: 仅主进程导出推理权重 ──
    acc.wait_for_everyone()
    if acc.is_main_process:
        sd = acc.unwrap_model(model).state_dict()
        sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
              for k, v in sd.items()}
        final = os.path.join(args.output_dir, "final_model.pt")
        torch.save(sd, final)
        with open(os.path.join(args.output_dir, "model_info.json"), "w") as f:
            json.dump({"input_size": [W, H], "canvas": [CW, CH],
                       "num_patches": num_patches, "dim": dino.config.hidden_size,
                       "reencoder_depth": args.reencoder_depth,
                       "decoder_depth": args.decoder_depth,
                       "dino_dir": args.dino_dir}, f, indent=2)
        acc.print(f"[final] {final} 已保存 (bf16, 含 DINO 权重)")


if __name__ == "__main__":
    main()
