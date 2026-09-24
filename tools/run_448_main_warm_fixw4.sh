#!/usr/bin/env bash
# run_448_main_warm_fixw4.sh — 主线 448×252 / construction_site / dinov2-large
#   = 两相配方（A 相 = --warm_steps 单步全读热启动）
#   + 「固定4」采样计划（--step_plan fixed --block 4: 步值自然数 1..⌈N/4⌉、
#     每步固定读 4 个 z_s、K=N=576、|T|=144）
#
# 用户口径（2026-09-24）：「448*252 的 warm_step + 固定4 time step 实验，在工地数据集上」
#   预算 = A 相 16 epoch + B 相 40 epoch（总 56 epoch）。
#
# 与历史主线臂（output/phase1_v2_bptt: square 计划 24 步 / 40 epoch / 全局 bs32 / fp32）的
# **唯一变量** = 采样计划（square 24 步 → fixed 144 步）+ 两相热启动（前 16 epoch 单步全读）。
#   · 全局 batch 保持 32 = 每卡 bs4 × 2 卡 × grad_accum4。
#     fixed-4 的 BPTT 显存实测（448×252 / dinov2-large / d2 / fp32）：
#     bs1 18.6GB、bs2 38.3GB、bs4 75.2GB（单卡 fwd+bwd）→ DDP 训练实测峰值 87.3GB、**bs8 OOM**
#     ⇒ bs4 是 95GB 卡上的上限（90% 占用；PYTORCH_CUDA_ALLOC_CONF=expandable_segments 已开）。
#   · ⇒ 微批/epoch = 7009//8 = 876，优化步/epoch = 876//4 = **219**（与历史逐位一致）。
#   · 精度 = 纯 fp32（历史口径；fp16+8bit 已实测把 BPTT 精修整块抹掉，见
#     doc/2026-09-22/REPORT_res896_448_fp16_8bit.md）。
#   · LR 1.5e-4 / wd 0.01 / cosine + warmup 3% / grad_clip 1.0 / seed 42 / decoder_depth 2。
#   · 步数核算（train_v2.py 2026-09-24 修正）: warmup/LR 调度按**优化步**计，
#     总优化步 = 56×219 = 12264 ⇒ warmup = 367 步；A 相切换点 = 16×219 = 3504 优化步。
#
# 输出: $OUT/{final_model.pt,args.json,model_info.json,checkpoint-*}
# 日志: $LOG（训练 stdout/stderr）；driver 日志见 setsid 重定向
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/root/miniconda3/bin:$PATH
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA=${DATA:-/root/autodl-tmp/construction_site}
DINO=${DINO:-/root/autodl-tmp/models/dinov2-large}
OUT=${OUT:-/root/autodl-tmp/construction_site/out_448_main_warm_fixw4}
LOG=${LOG:-/root/train_logs/448_main_warm_fixw4.log}
NUM_GPUS=${NUM_GPUS:-2}
BATCH=${BATCH:-4}
ACCUM=${ACCUM:-4}
BLOCK=${BLOCK:-4}
A_EPOCHS=${A_EPOCHS:-16}
B_EPOCHS=${B_EPOCHS:-40}
EVAL_EVERY=${EVAL_EVERY:-1752}    # 8 epoch（1752 = 8×219），含 A→B 切换点 3504
SAVE_EVERY=${SAVE_EVERY:-1752}

TRAIN_N=$(/root/miniconda3/bin/python -c \
  "import glob, pyarrow.parquet as pq; print(sum(pq.ParquetFile(f).metadata.num_rows for f in glob.glob('${DATA}/train-*.parquet')))")
MICRO_PER_EPOCH=$(( TRAIN_N / (BATCH * NUM_GPUS) ))
OPT_PER_EPOCH=$(( MICRO_PER_EPOCH / ACCUM ))
EPOCHS=$(( A_EPOCHS + B_EPOCHS ))
WARM_STEPS=$(( A_EPOCHS * OPT_PER_EPOCH ))

mkdir -p "$(dirname "$LOG")" "$OUT"
echo "===== 448 main warm+fixw${BLOCK} | $(date '+%F %T')"
echo "  data=${DATA}  train=${TRAIN_N}  global_bs=$((BATCH*NUM_GPUS*ACCUM)) (bs${BATCH}×${NUM_GPUS}gpu×accum${ACCUM})"
echo "  opt/epoch=${OPT_PER_EPOCH}  A=${A_EPOCHS}ep (warm_steps=${WARM_STEPS})  B=${B_EPOCHS}ep"
echo "  total=${EPOCHS} epoch / $((EPOCHS*OPT_PER_EPOCH)) 优化步  warmup=$((EPOCHS*OPT_PER_EPOCH*3/100)) 步  fp32"
echo "  out=${OUT}  log=${LOG}"

NUM_GPUS="$NUM_GPUS" bash run_v2_train.sh \
  --data_dir "$DATA" \
  --dino_dir "$DINO" \
  --output_dir "$OUT" \
  --model_input 448x252 \
  --decoder_depth 2 \
  --step_plan fixed --block "$BLOCK" \
  --epochs "$EPOCHS" --batch_size "$BATCH" --grad_accum "$ACCUM" \
  --warm_steps "$WARM_STEPS" \
  --eval_every "$EVAL_EVERY" --save_every "$SAVE_EVERY" --log_every 20 \
  --num_workers 12 \
  > "$LOG" 2>&1

echo "===== EXIT rc=$? $(date '+%F %T')"
