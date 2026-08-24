"""
SR-Diffusion Phase 1 v3 training — Lagrangian budget-constrained tokenizer.
HF Trainer based (mirrors train_phase1.py structure).

Two-stage schedule (driven by a TrainerCallback):
  Stage 1 (warmup):  τ pinned to -2 (≈ all tokens kept), rate terms OFF.
                     Teaches re-encoder + decoder to reconstruct the full
                     DINO feature map.  Never dead-locks.
  Stage 2 (Lagrange): RateHead enabled → τ becomes the primal variable;
                     LagrangianBudget's dual ascent drives λ so that the
                     constraint R ≤ R_target is SATISFIED (not approximated).
                     R_target_eff anneals 1.0 → R_target over anneal_steps.

Key difference from train_phase1.py (v1): no trust region / TTUR / dead-zone
hinge / force magnification.  The Lagrangian multiplier replaces all of them:
  · primal (optimizer):  θ ← θ − η·∇[L_recon + λ·(R−R_target) + (ρ/2)·ReLU(R−R_target)²]
  · dual (per step):     λ ← clamp(λ + η_λ·(R_ema − R_target_eff), 0, λ_max)

Usage:
  python train_phase1_v3.py \
      --data_dir /root/autodl-tmp/imagenet/val \
      --dino facebook/dinov2-base --cache_dir /root/hf_cache \
      --output_dir output/phase1_v3 --limit 20000 \
      --num_train_epochs 4 --per_device_train_batch_size 16 \
      --stage1_steps 800 --anneal_steps 2000 --rate_target 0.25

Ablation (sweep λ to trace the R-D Pareto frontier, instead of the constraint):
  python train_phase1_v3.py ... --use_lagrangian 0 --lambda_fixed 0.1
  (sweep --lambda_fixed ∈ {0.01, 0.03, 0.1, 0.3, 1.0} → (D, R) points)
"""
import argparse, os, glob, random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    Dinov2Model, Trainer, TrainingArguments, TrainerCallback,
)

from model_phase1_v3 import SRPhase1V3, LagrangianBudget, RAW_MAX

# ═══════════════════════════════════════════════════════════════
# Dataset / collate — same as train_phase1.py
# ═══════════════════════════════════════════════════════════════

class ImageDirDataset(Dataset):
    """__getitem__ → {"image_path": str}. Preprocessing deferred to collate."""

    def __init__(self, root: str, limit: int = 0):
        exts = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG",
                "*.webp", "*.WEBP", "*.bmp", "*.BMP")
        files = []
        for ext in exts:
            files += glob.glob(os.path.join(root, "**", ext), recursive=True)
        files = sorted(files)
        if limit > 0:
            files = files[:limit]
        self.files = files
        print(f"[data] {len(files)} images from {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return {"image_path": self.files[idx]}


DINO_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
DINO_STD = np.array([0.229, 0.224, 0.225], np.float32)


class ImageCollector:
    def __call__(self, batch: list) -> dict:
        xs = []
        for item in batch:
            img = Image.open(item["image_path"]).convert("RGB")
            w, h = img.size
            s = min(w, h)
            if s > 224:
                x0 = random.randint(0, w - 224)
                y0 = random.randint(0, h - 224)
                img = img.crop((x0, y0, x0 + 224, y0 + 224))
            else:
                img = img.resize((224, 224), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = (arr - DINO_MEAN) / DINO_STD
            xs.append(torch.from_numpy(arr).permute(2, 0, 1))
        return {"pixel_values": torch.stack(xs)}


# ═══════════════════════════════════════════════════════════════
# Callback — stage switch + gate temperature anneal
# ═══════════════════════════════════════════════════════════════
# 对偶步由 LagrangianBudget 在 forward 内自驱动（train 模式），回调只负责
# 阶段切换与 T 退火。

class V3ScheduleCallback(TrainerCallback):
    def __init__(self, stage1_steps: int, anneal_steps: int,
                 T: float, T_min: float):
        self.stage1_steps = stage1_steps
        self.anneal_steps = anneal_steps
        self.T = T
        self.T_min = T_min

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        step = state.global_step
        if step < self.stage1_steps:
            model.set_stage(1)
        else:
            model.set_stage(2)          # 幂等（_stage 守卫），只在首次切换时 reset 乘子
            prog = min(1.0, (step - self.stage1_steps) / max(1, self.anneal_steps))
            model.gate.T = max(self.T_min, self.T * (1 - 0.7 * prog))

    def on_log(self, args, state, control, model=None, **kwargs):
        if model is not None:
            control.should_log = True


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--dino", default="facebook/dinov2-base")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--output_dir", default="output/phase1_v3")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--init_reencoder", type=int, default=1,
                    help="warm-start ReEncoder from DINO encoder layers")
    ap.add_argument("--hard_mix_prob", type=float, default=0.0,
                    help="stage-2 training: prob of feeding the decoder the "
                         "hard-pruned selected z_s instead of zero-padded full "
                         "length (aligns train/inference decoder memory; "
                         "recommended 0.2-0.3 if eval recon >> train)")
    ap.add_argument("--resume_from", default=None,
                    help="dir with model.safetensors/pytorch_model.bin to load "
                         "weights from before training")

    # stage schedule
    ap.add_argument("--stage1_steps", type=int, default=800)
    ap.add_argument("--anneal_steps", type=int, default=2000,
                    help="stage-2 中 R_target_eff 从 1.0 退火到 rate_target 的步数")
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--T_min", type=float, default=0.5)

    # architecture
    ap.add_argument("--select_on", default="zs", choices=["zs", "cls"],
                    help="zs: ScoreHead(z_s) 逐位置内容感知; cls: 全局描述子")

    # ── 拉格朗日（v3 核心）──
    ap.add_argument("--use_lagrangian", type=int, default=1,
                    help="1: 对偶上升满足 R ≤ rate_target; 0: 固定 λ_rate 惩罚（ablation）")
    ap.add_argument("--rate_target", type=float, default=0.25,
                    help="预算: 平均保留 token 比例 ≤ R_target（约束，对偶上升去满足它）")
    ap.add_argument("--lambda_init", type=float, default=0.1,
                    help="λ 起点（>0 起步就有预算压力）")
    ap.add_argument("--eta_lambda", type=float, default=0.05,
                    help="对偶步长（rate 单位；太大 λ 振荡，太小收敛慢）")
    ap.add_argument("--lambda_max", type=float, default=100.0)
    ap.add_argument("--rho", type=float, default=1.0,
                    help="增广项权重（method of multipliers；0 = 纯对偶上升）")
    ap.add_argument("--lambda_ent", type=float, default=0.01,
                    help="反塌缩熵权重")
    ap.add_argument("--lambda_fixed", type=float, default=0.0,
                    help="use_lagrangian=0 时的固定惩罚权重（描 R-D 帕累托前沿）")
    ap.add_argument("--k_min", type=int, default=0, help="k 硬守卫下限（0=关）")
    ap.add_argument("--k_max", type=int, default=0, help="k 硬守卫上限（0=关）")

    # HF Trainer args (mirror train_phase1.py style)
    ap.add_argument("--num_train_epochs", type=int, default=4)
    ap.add_argument("--per_device_train_batch_size", type=int, default=16)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--learning_rate", type=float, default=3e-4)
    ap.add_argument("--lr_rate_head", type=float, default=None,
                    help="RateHead(τ) 学习率。默认 learning_rate/10 —— 双时间尺度"
                         "（TTSA: dk/dτ≈−100，τ 必须走得比主网络慢，否则 k 振荡塌缩）")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--max_grad_norm", type=float, default=5.0)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--save_total_limit", type=int, default=5)
    ap.add_argument("--bf16", type=int, default=1)
    ap.add_argument("--dataloader_num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deepspeed", default=None, help="path to ds_config.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── frozen DINOv2 + model ──
    print(f"[model] loading frozen {args.dino} ...")
    dino = Dinov2Model.from_pretrained(args.dino, cache_dir=args.cache_dir).to(device).eval()
    for p in dino.parameters():
        p.requires_grad_(False)
    dim = dino.config.hidden_size
    num_patches = (224 // dino.config.patch_size) ** 2

    model = SRPhase1V3.build_model(
        dino, num_patches=num_patches, dim=dim,
        T=args.T, select_on=args.select_on, lambda_ent=args.lambda_ent,
        rate_target=args.rate_target, lambda_init=args.lambda_init,
        eta_lambda=args.eta_lambda, lambda_max=args.lambda_max,
        rho=args.rho, k_min=args.k_min,
        k_max=(args.k_max or None),
        init_reencoder=bool(args.init_reencoder),
        use_lagrangian=bool(args.use_lagrangian),
    ).to(device)
    if not args.use_lagrangian:
        model.set_lambda_rate(args.lambda_fixed)
    model.set_stage(1)
    model.hard_input_prob = args.hard_mix_prob
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params: {n_params/1e6:.2f}M | "
          f"select_on={args.select_on} | "
          f"{'Lagrangian R_target=' + str(args.rate_target) if args.use_lagrangian else 'fixed λ_rate=' + str(args.lambda_fixed)}")

    # ── optional: load weights from an existing checkpoint ──
    if args.resume_from:
        p_bin = os.path.join(args.resume_from, "pytorch_model.bin")
        if os.path.exists(p_bin):
            sd = torch.load(p_bin, map_location=device)
        else:
            from safetensors.torch import load_file
            sd = load_file(os.path.join(args.resume_from, "model.safetensors"))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[resume] loaded {args.resume_from}: "
              f"missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing):
            print(f"  missing keys (sample): {missing[:5]}")

    # ── dataset + collator ──
    dataset = ImageDirDataset(args.data_dir, limit=args.limit)
    collator = ImageCollector()

    # ── training args ──
    total_steps = (len(dataset) // args.per_device_train_batch_size) * args.num_train_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy="steps",
        eval_strategy="no",
        bf16=bool(args.bf16),
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=[],
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        deepspeed=args.deepspeed,
        seed=args.seed,
    )

    schedule = V3ScheduleCallback(args.stage1_steps, args.anneal_steps,
                                  args.T, args.T_min)

    # 双时间尺度优化器（TTSA）: 主网络 lr，RateHead(τ) 用更小的 lr/10。
    # τ 是预算的原变量，但 dk/dτ≈−100（v1 实测）——τ 走太快 k 就振荡塌缩；
    # 对偶变量 λ 已经承担了慢外环，τ 必须比主网络更稳。
    lr_rh = args.lr_rate_head if args.lr_rate_head is not None else args.learning_rate / 10.0
    main_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "rate_head" not in n]
    pol_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and "rate_head" in n]
    opt = torch.optim.AdamW([
        {"params": main_params, "lr": args.learning_rate},
        {"params": pol_params, "lr": lr_rh},
    ], weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    optimizers = (opt, sched)
    print(f"[TTSA] main lr={args.learning_rate}, RateHead(τ) lr={lr_rh}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[schedule],
        optimizers=optimizers,
    )

    print(f"\n{'='*60}")
    print(f"Phase-1 v3 training (HF Trainer) — 拉格朗日预算约束")
    print(f"  stage1={args.stage1_steps} anneal={args.anneal_steps} "
          f"{'R_target=' + str(args.rate_target) if args.use_lagrangian else 'λ_rate=' + str(args.lambda_fixed)}")
    print(f"  init_reencoder={bool(args.init_reencoder)} select_on={args.select_on}")
    print(f"{'='*60}\n")

    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    print(f"\nFinal model saved to {final_dir}")


if __name__ == "__main__":
    main()
