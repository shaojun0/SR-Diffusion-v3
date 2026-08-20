"""
SR-Diffusion Phase 1 training — adaptive token budget + feature reconstruction.
HF Trainer based (mirrors train.py structure: Trainer + TrainingArguments +
data collator does preprocessing).

Two-stage schedule (driven by a TrainerCallback):
  Stage 1 (warmup):  τ pinned to -2 (all tokens kept), λ_rate = 0.
                     Teaches re-encoder + decoder to reconstruct the full
                     DINO feature map.  Never dead-locks.
  Stage 2 (anneal):  RateHead enabled, λ_rate annealed 0 → target.
                     Mask sparsifies; budget becomes content-adaptive.

Usage:
  python train_phase1.py \
      --data_dir /root/autodl-tmp/imagenet/val \
      --dino facebook/dinov2-base --cache_dir /root/hf_cache \
      --output_dir output/phase1 --limit 20000 \
      --num_train_epochs 4 --per_device_train_batch_size 16 \
      --stage1_steps 800 --anneal_steps 3000 --lambda_rate_target 0.04
"""
import argparse, os, glob, random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    Dinov2Model, Trainer, TrainingArguments, TrainerCallback,
)

from model_phase1 import SRPhase1, BudgetTrustRegion

# ═══════════════════════════════════════════════════════════════
# Dataset — flat JPEG dir or ImageFolder; returns raw path (like
# RawParquetDataset returns raw image_bytes; preprocessing in collate)
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


# ═══════════════════════════════════════════════════════════════
# Collate — image load + 224² random crop + DINO normalize (mirrors
# SVDCollector: heavy lifting in collate_fn, CPU-friendly)
# ═══════════════════════════════════════════════════════════════

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
# Callback — two-stage schedule + PPO-style trust-region updates
# ═══════════════════════════════════════════════════════════════

class StageScheduleCallback(TrainerCallback):
    def __init__(self, stage1_steps: int, anneal_steps: int,
                 lambda_rate_target: float, T: float, T_min: float,
                 trust_region: bool = False, lr_pol: float = 1e-4):
        self.stage1_steps = stage1_steps
        self.anneal_steps = anneal_steps
        self.lambda_rate_target = lambda_rate_target
        self.T = T
        self.T_min = T_min
        self.trust_region = trust_region
        self.lr_pol = lr_pol

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        step = state.global_step
        if step < self.stage1_steps:
            model.set_stage(1)
            model.set_lambda_rate(0.0)
        else:
            if step == self.stage1_steps and self.trust_region and model.trust_region is not None:
                # 进入 stage-2：参考策略与 RateHead 硬同步（消除 EMA 残留偏差，
                # 防 τ_ref 与 τ_net 起点不一致；stage-1 期间 rate_head 未训练，
                # target 是 enable_trust_region 时的旧拷贝，此处强制对齐）
                btr = model.trust_region
                with torch.no_grad():
                    for p, pt in zip(model.rate_head.parameters(),
                                     model.rate_head_target.parameters()):
                        pt.data.copy_(p.data)
                btr.step = 0                 # 退火窗口从 stage-2 起算
                btr.rate_ema = None          # 日志 EMA 重置
            model.set_stage(2)
            if self.trust_region and model.trust_region is not None:
                # trust-region controller handles its own scheduling
                pass
            else:
                prog = min(1.0, (step - self.stage1_steps) / max(1, self.anneal_steps))
                model.set_lambda_rate(self.lambda_rate_target * prog)
                model.gate.T = max(self.T_min, self.T * (1 - 0.7 * prog))

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or not self.trust_region:
            return
        btr = model.trust_region
        if btr is None or model.fixed_tau is not None:
            return
        # EMA reference policy + adaptive β (PPO-KL dual gradient)
        btr.update_ref(model)
        kl_step = getattr(btr, "_last_kl", 0.0)
        btr.update_beta(kl_step)

    def on_log(self, args, state, control, model=None, **kwargs):
        if model is not None:
            control.should_log = True  # keep default logging


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--dino", default="facebook/dinov2-base")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--output_dir", default="output/phase1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--init_reencoder", type=int, default=1,
                    help="warm-start ReEncoder from DINO encoder layers")

    # stage schedule
    ap.add_argument("--stage1_steps", type=int, default=800)
    ap.add_argument("--anneal_steps", type=int, default=3000)
    ap.add_argument("--lambda_rate_target", type=float, default=0.04,
                    help="plain-path 线性 λr（anneal 目标）；v3 推荐 trust_region=1")
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--T_min", type=float, default=0.7)

    # PPO-style trust region (budget stabilization) —— v3 默认开启
    ap.add_argument("--trust_region", type=int, default=1,
                    help="enable PPO-KL budget stabilization (stage-2) [v3: 必须开]")
    ap.add_argument("--lr_pol", type=float, default=1e-3,
                    help="RateHead lr (TTUR: slow policy head; v3: 1e-3，配 trust region)")
    ap.add_argument("--delta_tau", type=float, default=0.05)
    ap.add_argument("--kl_target", type=float, default=0.005)
    ap.add_argument("--beta0", type=float, default=1.0)
    ap.add_argument("--k_min", type=int, default=8,
                    help="k 守卫下限（原 2 允许 k=4 塌缩）")
    ap.add_argument("--k_max", type=int, default=250)
    ap.add_argument("--rate_min", type=float, default=0.03)
    ap.add_argument("--rate_max", type=float, default=0.25)
    ap.add_argument("--tr_T_min", type=float, default=0.3)
    ap.add_argument("--tr_gumbel_steps", type=int, default=800)
    # v3 新增：力平衡 + 触界保护
    ap.add_argument("--lambda_rate_hinge", type=float, default=5.0,
                    help="dead-zone hinge 刚度（独立于 lambda_rate_target；勿混用）")
    ap.add_argument("--recon_tau_scale", type=float, default=50.0,
                    help="recon 的 τ 路径梯度放大倍数（力平衡标定，主旋钮）")
    ap.add_argument("--raw_max", type=float, default=1.472,
                    help="atanh(1.8/2)：raw clamp，τ≤1.8 且 dτ/draw≥0.38")
    ap.add_argument("--tau_soft", type=float, default=1.5)
    ap.add_argument("--lambda_tau", type=float, default=1.0)

    # HF Trainer args (mirror train.py style)
    ap.add_argument("--num_train_epochs", type=int, default=4)
    ap.add_argument("--per_device_train_batch_size", type=int, default=16)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--learning_rate", type=float, default=3e-4)
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

    # ── frozen DINOv2 + model (build_model mirrors SRQwenVLv10.build_model) ──
    print(f"[model] loading frozen {args.dino} ...")
    dino = Dinov2Model.from_pretrained(args.dino, cache_dir=args.cache_dir).to(device).eval()
    for p in dino.parameters():
        p.requires_grad_(False)
    dim = dino.config.hidden_size
    num_patches = (224 // dino.config.patch_size) ** 2

    model = SRPhase1.build_model(
        dino, num_patches=num_patches, dim=dim,
        T=args.T, lambda_rate=0.0,
        init_reencoder=bool(args.init_reencoder),
    ).to(device)
    model.set_stage(1)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params: {n_params/1e6:.2f}M")

    # ── dataset + collator (Trainer-managed) ──
    dataset = ImageDirDataset(args.data_dir, limit=args.limit)
    collator = ImageCollector()

    # ── training args (mirror train.py: bf16, cosine, warmup, deepspeed) ──
    # transformers 5.x: warmup_ratio removed → convert to warmup_steps
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

    schedule = StageScheduleCallback(
        args.stage1_steps, args.anneal_steps,
        args.lambda_rate_target, args.T, args.T_min,
        trust_region=bool(args.trust_region), lr_pol=args.lr_pol,
    )

    # ── PPO-style trust region: controller + RateHead slow lr (TTUR) ──
    optimizers = None
    if args.trust_region:
        btr = BudgetTrustRegion(
            n=num_patches,
            delta_tau=args.delta_tau,
            kl_target=args.kl_target,
            beta=args.beta0,
            k_min=args.k_min,
            k_max=args.k_max,
            T=args.T,
            T_min=args.tr_T_min,
            gumbel_steps=args.tr_gumbel_steps,
            rate_min=args.rate_min,
            rate_max=args.rate_max,
            lambda_rate=args.lambda_rate_hinge,   # ← 独立 hinge 刚度（勿用 lambda_rate_target）
            recon_tau_scale=args.recon_tau_scale,
            raw_max=args.raw_max,
            tau_soft=args.tau_soft,
            lambda_tau=args.lambda_tau,
        )
        model.enable_trust_region(btr)
        # two timescales: main params 3e-4, RateHead 1e-4 (TTUR)
        main_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and "rate_head" not in n]
        pol_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and "rate_head" in n]
        opt = torch.optim.AdamW([
            {"params": main_params, "lr": args.learning_rate},
            {"params": pol_params, "lr": args.lr_pol},
        ], weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
        optimizers = (opt, sched)
        print(f"[trust-region] RateHead lr={args.lr_pol} (TTUR), "
              f"k∈[{args.k_min},{args.k_max}], rate∈[{args.rate_min},{args.rate_max}]")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[schedule],
        optimizers=optimizers,
    )

    print(f"\n{'='*60}")
    print(f"Phase-1 training (HF Trainer)")
    print(f"  stage1={args.stage1_steps} anneal={args.anneal_steps} "
          f"λ_target={args.lambda_rate_target}")
    print(f"  init_reencoder={bool(args.init_reencoder)}")
    print(f"{'='*60}\n")

    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    print(f"\nFinal model saved to {final_dir}")


if __name__ == "__main__":
    main()
