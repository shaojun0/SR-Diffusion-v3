#!/usr/bin/env bash
# run_896_d4_bptt_slice012.sh — 896×504（=(448×2)×(252×2)）/ depth=4 / BPTT / slice[0:12]
#
# 用户口径（2026-09-22 晚）:「基于 BPTT 跑一个实验: layer=4, slice=0~12,
# (448×2)×(252×2)」+「这次试试 fp16, 优化器用 8 位 AdamW」
# ⇒ 输入分辨率 **896×504 = (448×2)×(252×2)**（16:9, 14 的倍数）, decoder_depth=4,
#   采样步切片 [0:12], **fp16 AMP + bitsandbytes AdamW8bit**。
#
# 口径:
#   · 数据 / 画布: construction_site 7,009 train / 3,004 test; fit_to_canvas(1600×900)
#     → BICUBIC resize 到模型输入（同一预处理; 重建目标 = 输入本身）
#   · 896×504 ⇒ patches 64×36 = **2304**; slice[0:12] ⇒ 12 步 [1,4,…,144],
#     K = derive_num_specials = min((12+1)²−1, N) = **168**（序列 1+168+2304 = 2473）
#   · 40 epoch = 8,760 优化步（全局 batch 32）, lr 1.5e-4 cosine + warmup 262 步（3%）, seed 42
#   · carry: **BPTT**（carry_detach=False, 仓库当前默认）; 平方块读窗口; layer_tap 关
#   · 损失: 直接预测 mean_t L1(PixelHead(Y_t), target)
#   · 精度: **fp16 混合精度**（autocast + GradScaler; 权重/final_model.pt 仍 fp32）
#   · 优化器: **bitsandbytes AdamW8bit**（`--optim adamw_bnb_8bit`）
#
# ⚠️ 显存/步速实测（2026-09-22, RTX PRO 6000 Blackwell 97 GiB, 单卡 CUDA_VISIBLE_DEVICES=0）:
#   | 配置 | peak | 单步(含优化器) | ms/img |
#   |---|---|---|---|
#   | fp32  bs=4 | 58.3 GiB | 1.79 s | 448  |
#   | fp16+8bit bs=4 | **34.3 GiB** | **0.41 s** | **102** |
#   | fp16+8bit bs=8 | 66.3 GiB | 0.91 s | 114 |
#   | fp16+8bit bs≥12 | OOM | — | — |
#   ⇒ bs=4 既最省显存又最快（bs=8 单图反而更慢）→ **--batch_size 4 --grad_accum 4**
#     凑回全局 batch 32（= 基线 16×2×1）; 无 BatchNorm ⇒ 累积与单批 bs32 数学等价。
#   ⇒ 预估墙钟 ≈ 8,760 步 × (4×0.41 s) ≈ 4.0 h（+ 每 2,000 步一次 fp32 eval）。
#   （作为对照: 纯 fp32 同参数要 ≈17–18 h; fp16 让 encoder/decoder 走 tensor core。）
#
# ⚠️ train_v2.py 的 steps_per_epoch 公式不计 grad_accum ⇒ 日志打印的 total_steps
#   是 35,040（4× 真实值）。真实优化步数由 num_train_epochs 决定 = 8,760 ✓,
#   但 warmup_steps 会按 35,040 算 ⇒ 必须显式给 --warmup_ratio 0.0075
#   (= 262/35,040) 才等于基线的 262 步。
#
# 用法（服务器）:
#   cd /root/autodl-tmp/sr-diffusion-v3-tap
#   setsid nohup bash tools/run_896_d4_bptt_slice012.sh > /dev/null 2>&1 < /dev/null &
# 冒烟（单卡, 2 步）:
#   NUM_GPUS=1 SMOKE=1 bash tools/run_896_d4_bptt_slice012.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/root/miniconda3/bin:$PATH
export HF_HUB_OFFLINE=1
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=${OUT:-/root/autodl-tmp/construction_site/out_896_d4_bptt_slice012}
LOG=${LOG:-/root/train_logs/896_d4_bptt_slice012.log}
SMOKE=${SMOKE:-0}
EPOCHS=${EPOCHS:-40}
BATCH=${BATCH:-4}
ACCUM=${ACCUM:-4}
WARMUP_RATIO=${WARMUP_RATIO:-0.0075}
OPTIM=${OPTIM:-adamw_bnb_8bit}
FP16=${FP16:-1}
mkdir -p "$(dirname "$LOG")" "$OUT"

SMOKE_ARGS=()
if [ "$SMOKE" = "1" ]; then
  # limit 128 = 64/进程 = 16 micro-batch = 4 优化步 ≥ max_steps 2
  SMOKE_ARGS=(--smoke --limit 128 --eval_limit 16 --max_steps 2 --eval_every 1)
fi

FP16_ARGS=()
[ "$FP16" = "1" ] && FP16_ARGS=(--fp16)

NUM_GPUS=${NUM_GPUS:-2} bash run_v2_train.sh \
  --data_dir /root/autodl-tmp/construction_site \
  --dino_dir /root/autodl-tmp/models/dinov2-large \
  --output_dir "$OUT" \
  --model_input 896x504 \
  --decoder_depth 4 \
  --slice_start 0 --slice_end 12 \
  --epochs "$EPOCHS" --batch_size "$BATCH" --grad_accum "$ACCUM" \
  --warmup_ratio "$WARMUP_RATIO" \
  --optim "$OPTIM" \
  "${FP16_ARGS[@]}" \
  --eval_every 2000 --save_every 2000 --log_every 50 \
  "${SMOKE_ARGS[@]}" \
  > "$LOG" 2>&1
