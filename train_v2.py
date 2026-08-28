"""
SR-Diffusion Phase 1 v2 — 训练（test 分支: 注意力机制改写后, 像素目标）
=================================================
⚠️ 项目目标（权威版 2026-08-28 修订 v2, 见 doc/2026-08-28/GOAL_compression_for_nlp.md）:
    本训练是 Phase 1 的**脚手架**——用像素重建代理任务练编码器的
    "token 压缩 + 联想"能力; 完成后冻结编码器接 Qwen 做 NLP 解码。
    **验收标准是 Phase 2 文字生成质量**; 像素重建 = **信息保持探针**
    （能还原像素 ⇒ z 携带整图信息, 重建质量决定 NLP 天花板）——不是
    "不追", 而是追"K 压缩下活信息保真"（纹理级清晰度属死信息才不追）。
架构（model_v2.py, test 分支）: DINOv2-large(参数不冻结) → ReEncoder(因果
    specials 前缀链) → OutputQueryDecoder（输出查询注意力 + KV 因果 +
    平方采样计划）→ PixelHead → **像素 patch 预测**, 平权全覆盖 L1
    重建原始像素 pixel_values（非 DINO 特征）。无 TextDecoder。

2026-08-27 重大修复（用户发现）:
    监督目标从「DINO patch 特征」改为「原始像素 pixel_values」——
    之前特征目标退化（工地图特征空间近常数, 学质心即低 L1, 假收敛）;
    像素目标有真实空间结构, 强制模型保留空间信息。
    PixelHead: 每 patch 特征 (N,D) → (N, 14*14*3) 像素解码。

2026-08-27 训练口径调整（用户要求, 沿自上轮）:
    · 去掉加权体系: 全部采样步**平权**（loss = mean_t L1, 无 density/
      uniform/capability 权重）—— decoder 的 loss_weight 机制已整块删除;
    · 全 fp32 训练: 训练/评估不用 bf16/fp16 —— 公平性（之前 bf16 算
      fp32 存被质疑）; TrainingArguments 不设 bf16/fp16, 不套 autocast;
    · batch_size 默认提高到 16/卡（97GB 显存充裕）。

HF Trainer 风格（消除造轮子）:
    · 训练循环 / 梯度累积 / 调度器 / checkpoint / 分布式 → 全部交给
      transformers.Trainer + TrainingArguments（lr_scheduler_type="cosine"
      + warmup_ratio, save_strategy="steps", ddp_find_unused_parameters=
      False, report_to=[]）; 优化器 → Trainer 默认 AdamW。不再手写
      acc.no_sync / 手动 grad_accum / get_scheduler / acc.save_state。
    · 数据 → data_v2.py 的 ParquetImageDataset + V2Collator 直接作为
      train_dataset / eval_dataset / data_collator 传给 Trainer。
    · 模型 forward 返回 dict{"loss", ...}: Trainer.compute_loss 原生支持
      dict 输出取 "loss" 键（transformers 4.x / 5.x 均如此:
      loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0],
      dict 缺 "loss" 才报错）——无需子类化 Trainer。

数据（data_v2.py）: 原图 → 旋转(最优角) → 等比缩放 → 居中填充 1600:900
    (16:9) 画布 → 16:9 模型输入 (448×252)。轮廓不变形、内容面积最大化。

多卡: accelerate launch --multi_gpu --num_processes N
（Trainer 在 accelerate launch 下自动接管 DDP, 等价于 Trainer 自管 DDP;
全 fp32 无 autocast, 无需显式 .to(device)）

用法:
    accelerate launch --multi_gpu --num_processes 2 \
        train_v2.py --data_dir /root/autodl-tmp/construction_site \
        --dino_dir /root/autodl-tmp/models/dinov2-large \
        --output_dir output/phase1_v2_pixelfp32
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from transformers import Dinov2Model, Trainer, TrainingArguments
from transformers.trainer_utils import set_seed

from data_v2 import ParquetImageDataset, V2Collator
from model_v2 import SRPhase1V2


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="train phase1 v2 (DINOv2-large unfrozen, OutputQueryDecoder)")
    p.add_argument("--data_dir", required=True, help="parquet 目录(train-*.parquet)")
    p.add_argument("--dino_dir", default="models/dinov2-large")
    p.add_argument("--output_dir", default="output/phase1_v2")
    p.add_argument("--model_input", default="448x252", help="16:9 模型输入 WxH (14 的倍数)")
    p.add_argument("--canvas", default="1600x900", help="旋转+缩放+填充目标画布")
    p.add_argument("--angle_step", type=float, default=0.5, help="最优旋转角网格步长(度)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--max_steps", type=int, default=0, help="0 = epochs 决定")
    p.add_argument("--batch_size", type=int, default=16, help="每卡 batch")
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1.5e-4)
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
    p.add_argument("--resume", default=None, help="Trainer checkpoint 目录(如 output_dir/checkpoint-123)")
    p.add_argument("--smoke", action="store_true",
                   help="冒烟: 不存 checkpoint, 但照常导出 final_model.pt")
    # ── 模型（ReEncoder）──
    p.add_argument("--reencoder_depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--no_causal_specials", action="store_true",
                   help="关闭 ReEncoder 的 causal specials 块掩码(全双向)")
    # ── 模型（OutputQueryDecoder）──
    p.add_argument("--decoder_steps", default=None,
                   help="解码器采样时刻列表(逗号分隔), 默认 square_step_schedule(N)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════
# Eval 指标 — Trainer 的 prediction_step 对无 labels 模型直接 forward,
# 把输出 dict 按插入序转成值的元组 (loss, recon, F_hat)（多 batch 后是
# 拼接数组的元组; 4.x 与 5.x 行为一致）。该路径 loss=None, eval loop 不会
# 自动算 eval_loss, 所以在这里显式从模型输出里取 loss / recon。
# ═══════════════════════════════════════════════════════════════

def compute_metrics(eval_pred):
    preds = eval_pred.predictions
    if isinstance(preds, dict):
        items = preds
    elif isinstance(preds, (tuple, list)):
        # prediction_step 无 labels 路径: logits = 除 loss 外所有输出
        # (recon, F_hat)（dict 插入序）。loss 单独在 eval_loss 里报告。
        items = dict(zip(("recon", "F_hat"), preds))
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


# ═══════════════════════════════════════════════════════════════
# Trainer 子类（官方扩展点, 不算造轮子）: 本模型无 labels、无 config,
# Trainer 默认 can_return_loss=False → eval 时 loss_without_labels=False
# → eval_loss 为 None。置 True 让 eval 也走 compute_loss 路径, 正常报告
# eval_loss（模型 forward 返回 dict 含 "loss", compute_loss 原生支持）。
# ═══════════════════════════════════════════════════════════════

class SRPhase1V2Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.can_return_loss = True   # 无 labels 模型也允许 eval 算 loss


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

    steps = None
    if args.decoder_steps:
        steps = [int(s) for s in args.decoder_steps.split(",") if s.strip()]
        assert steps and all(0 <= s <= num_patches for s in steps), \
            f"decoder_steps 越界: {steps} (N={num_patches})"

    # ── 数据（纯重建模式, tokenizer=None; data_v2.py 已提供）──
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

    # ── 模型: DINOv2-large 不冻结 → ReEncoder → OutputQueryDecoder ──
    dino = Dinov2Model.from_pretrained(args.dino_dir)
    # 权重带 mask_token(use_mask_token=True) 但本任务不传 bool_masked_pos,
    # 该参数从不参与前向 → DDP 报"未用参数"。移除并关掉 flag。
    if getattr(dino.config, "use_mask_token", False):
        dino.config.use_mask_token = False
        del dino.embeddings.mask_token

    model = SRPhase1V2(dinov2=dino, num_patches=num_patches,
                       dim=dino.config.hidden_size,
                       reencoder_depth=args.reencoder_depth,
                       heads=args.heads, mlp_ratio=args.mlp_ratio,
                       causal_specials=not args.no_causal_specials,
                       decoder_steps=steps)
    model.init_reencoder_from_dino(args.reencoder_depth)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] 可训练参数 {n_train / 1e6:.1f}M (含 DINOv2-large, 不冻结); "
          f"输入 {W}x{H}, patches={num_patches}, decoder 采样 "
          f"{len(model.decoder.steps)} 步 {model.decoder.steps[:6]}...")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    # ── Trainer: 训练循环/梯度累积/调度/checkpoint/分布式全部交给它 ──
    n_proc = int(os.environ.get("WORLD_SIZE", "1"))      # accelerate/DDP 进程数
    steps_per_epoch = max(1, len(train_ds) // (args.batch_size * n_proc))
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,   # 0 = epochs 决定
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
        save_strategy="no" if args.smoke else "steps",   # 冒烟不存 checkpoint
        save_steps=args.save_every,
        seed=args.seed,
        report_to=[],                  # 不上报 wandb / tensorboard
        remove_unused_columns=False,   # 原始 dict 样本交给 V2Collator
        ddp_find_unused_parameters=False,
        # 全 fp32: 不设 bf16/fp16, 不套 autocast
    )

    trainer = SRPhase1V2Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=coll,
        compute_metrics=compute_metrics if eval_ds is not None else None,
    )

    n_proc = trainer.accelerator.num_processes
    trainer.accelerator.print(
        f"[train] {len(train_ds)} 样本 | 每卡 bs={args.batch_size} "
        f"x {n_proc} 卡 | grad_accum={args.grad_accum} | "
        f"~{steps_per_epoch} 步/epoch x {args.epochs} = {total_steps} 步 "
        f"| warmup {warmup_steps} 步")

    if args.resume:
        trainer.accelerator.print(f"[resume] 从 {args.resume} 恢复 (Trainer checkpoint)")
    trainer.train(resume_from_checkpoint=args.resume)

    # ── 收尾: 仅主进程导出推理权重（全 fp32 训练, 权重天然 fp32）──
    trainer.accelerator.wait_for_everyone()
    if trainer.accelerator.is_main_process:
        raw = trainer.accelerator.unwrap_model(trainer.model)   # DDP 包装下取回裸模型
        sd = raw.state_dict()
        final = os.path.join(args.output_dir, "final_model.pt")
        torch.save(sd, final)
        info = {"input_size": [W, H], "canvas": [CW, CH],
                "num_patches": num_patches, "dim": dino.config.hidden_size,
                "reencoder_depth": args.reencoder_depth,
                "heads": args.heads, "mlp_ratio": args.mlp_ratio,
                "causal_specials": not args.no_causal_specials,
                "decoder_steps": raw.decoder.steps,
                "target": "pixel_values (归一化空间, PixelHead 解码)",
                "dino_dir": args.dino_dir, "dtype": "fp32"}
        with open(os.path.join(args.output_dir, "model_info.json"), "w") as f:
            json.dump(info, f, indent=2)
        print(f"[final] {final} 已保存 (fp32, 含 DINO 权重)")


if __name__ == "__main__":
    main()
