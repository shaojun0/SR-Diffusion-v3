"""
SR-Diffusion Phase 1 v4 — 训练（register K 压缩, DINO 不冻结, 2026-08-28）
=================================================
架构（model_v4.py）: K 个 special token 拼进 DINO 输入序列（1+K+N token）,
    DINO 24 层全双向直接算 z_s (B,K,D) → v3 式无时序解码器 → 像素 patch,
    平权 L1。DINO **默认不冻结**（--freeze_dino 可冻结做对照）。

与 v3（BLIP-2, K=64, DINO 冻结）的对照: 编码器形态 + DINO 可训性。
与 v2 register（K=N=576）的对照: K=64 固定压缩 + 无时序解码器。

学习率: 分组 —— DINO 1.5e-4（微调不宜过高, v2 教训）+ 新模块 3e-4
（--lr / --dino_lr 可改）; 全 fp32; 显式优化器只收可训练参数。

用法:
    accelerate launch --multi_gpu --num_processes 2 \
        train_v4.py --data_dir /root/autodl-tmp/construction_site \
        --dino_dir /root/autodl-tmp/models/dinov2-large \
        --output_dir output/phase1_v4_registerK
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from transformers import Dinov2Model, Trainer, TrainingArguments, \
    get_cosine_schedule_with_warmup
from transformers.trainer_utils import set_seed

from data_v2 import ParquetImageDataset, V2Collator
from model_v4 import SRPhase1V4


def parse_args():
    p = argparse.ArgumentParser(description="train phase1 v4 (register K 压缩, DINO 不冻结)")
    p.add_argument("--data_dir", required=True, help="parquet 目录(train-*.parquet)")
    p.add_argument("--dino_dir", default="models/dinov2-large")
    p.add_argument("--output_dir", default="output/phase1_v4")
    p.add_argument("--model_input", default="448x252", help="16:9 模型输入 WxH (14 的倍数)")
    p.add_argument("--canvas", default="1600x900", help="旋转+缩放+填充目标画布")
    p.add_argument("--angle_step", type=float, default=0.5, help="最优旋转角网格步长(度)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--max_steps", type=int, default=0, help="0 = epochs 决定")
    p.add_argument("--batch_size", type=int, default=16, help="每卡 batch")
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4, help="新模块学习率")
    p.add_argument("--dino_lr", type=float, default=1.5e-4,
                   help="DINO 学习率(微调不宜过高, v2 教训); 冻结时忽略")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="只用前 N 条训练样本(调试)")
    p.add_argument("--eval_limit", type=int, default=0, help="eval 只用前 N 条(冒烟)")
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default=None, help="Trainer checkpoint 目录")
    p.add_argument("--smoke", action="store_true",
                   help="冒烟: 不存 checkpoint, 但照常导出 final_model.pt")
    # ── 模型 ──
    p.add_argument("--num_specials", type=int, default=64,
                   help="K: register special token 数(固定压缩维度)")
    p.add_argument("--decoder_depth", type=int, default=2)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--head_hidden", type=int, default=2048, help="像素头隐藏维")
    p.add_argument("--freeze_dino", action="store_true",
                   help="冻结 DINO(默认不冻结; 冻结=对照 v3 的 DINO 变量)")
    return p.parse_args()


def compute_metrics(eval_pred):
    preds = eval_pred.predictions
    if isinstance(preds, dict):
        items = preds
    elif isinstance(preds, (tuple, list)):
        items = dict(zip(("recon", "pixels", "target", "z_s"), preds))
    else:
        items = {}
    metrics = {}
    for k in ("loss", "recon"):
        v = items.get(k)
        if v is not None:
            v = np.asarray(v)
            if v.size > 0:
                metrics[k] = float(v.mean())
    return metrics


class SRPhase1V4Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.can_return_loss = True   # 无 labels 模型也允许 eval 算 loss


def main():
    args = parse_args()
    set_seed(args.seed)

    W, H = (int(v) for v in args.model_input.lower().split("x"))
    CW, CH = (int(v) for v in args.canvas.lower().split("x"))
    assert W % 14 == 0 and H % 14 == 0, "DINOv2-large patch=14, 输入须为 14 的倍数"
    num_patches = (W // 14) * (H // 14)

    # ── 数据 ──
    train_files = sorted(glob.glob(os.path.join(args.data_dir, "train-*.parquet")))
    test_files = sorted(glob.glob(os.path.join(args.data_dir, "test-*.parquet")))
    assert train_files, f"无 train-*.parquet in {args.data_dir}"
    coll = V2Collator(model_size=(W, H), canvas=(CW, CH), angle_step=args.angle_step)
    train_ds = ParquetImageDataset(train_files, limit=args.limit)
    eval_ds = None
    if test_files:
        eval_ds = ParquetImageDataset(test_files, limit=args.eval_limit)
    else:
        print("[warn] 无 test-*.parquet, 跳过 eval")

    # ── 模型: register K 压缩（DINO 默认不冻结）──
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

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[model] 可训练参数 {n_train / 1e6:.1f}M / 总 {n_all / 1e6:.1f}M "
          f"(DINO {'冻结' if args.freeze_dino else '不冻结'}); "
          f"输入 {W}x{H}, patches={num_patches}, K={args.num_specials} specials "
          f"(序列 {1 + args.num_specials + num_patches} token)")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # ── Trainer ──
    n_proc = int(os.environ.get("WORLD_SIZE", "1"))
    steps_per_epoch = max(1, len(train_ds) // (args.batch_size * n_proc))
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        max_grad_norm=args.grad_clip,
        dataloader_num_workers=args.num_workers,
        dataloader_drop_last=True,
        dataloader_pin_memory=True,
        logging_steps=args.log_every,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_every,
        save_strategy="no" if args.smoke else "steps",
        save_steps=args.save_every,
        seed=args.seed,
        report_to=[],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        # 全 fp32: 不设 bf16/fp16, 不套 autocast
    )

    # 显式优化器: 分组学习率 —— DINO(微调低 lr) vs 新模块(高 lr)
    dino_ids = {id(p) for p in dino.parameters()}
    new_params = [p for p in model.parameters()
                  if p.requires_grad and id(p) not in dino_ids]
    dino_params = [p for p in dino.parameters() if p.requires_grad]
    groups = [{"params": new_params, "lr": args.lr,
               "weight_decay": args.weight_decay}]
    if dino_params:
        groups.append({"params": dino_params, "lr": args.dino_lr,
                       "weight_decay": args.weight_decay})
    opt = torch.optim.AdamW(groups)
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    trainer = SRPhase1V4Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=coll,
        compute_metrics=compute_metrics if eval_ds is not None else None,
        optimizers=(opt, sched),
    )

    n_proc = trainer.accelerator.num_processes
    trainer.accelerator.print(
        f"[train] {len(train_ds)} 样本 | 每卡 bs={args.batch_size} "
        f"x {n_proc} 卡 | grad_accum={args.grad_accum} | "
        f"~{steps_per_epoch} 步/epoch x {args.epochs} = {total_steps} 步 "
        f"| warmup {warmup_steps} 步 | lr 新模块={args.lr} DINO={args.dino_lr}")

    if args.resume:
        trainer.accelerator.print(f"[resume] 从 {args.resume} 恢复 (Trainer checkpoint)")
    trainer.train(resume_from_checkpoint=args.resume)

    # ── 收尾: 仅主进程导出推理权重 ──
    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        raw = trainer.accelerator.unwrap_model(trainer.model)
        sd = raw.state_dict()
        final = os.path.join(args.output_dir, "final_model.pt")
        torch.save(sd, final)
        info = {"input_size": [W, H], "canvas": [CW, CH],
                "num_patches": num_patches, "dim": dino.config.hidden_size,
                "num_specials": args.num_specials,
                "decoder_depth": args.decoder_depth,
                "heads": args.heads, "mlp_ratio": args.mlp_ratio,
                "head_hidden": args.head_hidden,
                "freeze_dino": args.freeze_dino,
                "lr": args.lr, "dino_lr": args.dino_lr,
                "target": "pixel_values (归一化空间, PixelHead 解码)",
                "dino_dir": args.dino_dir, "dtype": "fp32"}
        with open(os.path.join(args.output_dir, "model_info.json"), "w") as f:
            json.dump(info, f, indent=2)
        print(f"[final] {final} 已保存 (fp32, DINO {'冻结' if args.freeze_dino else '含权重'})")


if __name__ == "__main__":
    main()
