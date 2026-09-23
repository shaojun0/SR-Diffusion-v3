#!/usr/bin/env bash
# run_448_d2_bptt_fp16_8bit.sh — 448×252 / **原始配方** / BPTT + fp16 + 8-bit AdamW（对照臂）
#
# 用户口径（2026-09-22 晚）:「跑完这个（896）再跑一下**原始的 BPTT + fp16 和 8 位的 adamw**
# 作为对照（**448×252**）」。用户确认取 **B 读法 = 原始配方不动, 只把精度/优化器换成
# fp16 + adamw_bnb_8bit**（而不是"与 896 臂逐项对齐"的 A 读法）。
#
# 配方（= 历史 448×252/BPTT 臂的原始口径, 一个字不改）:
#   · --model_input 448×252（= train_v2.py 的**默认值**, 即"原始分辨率"）
#   · **不传 --slice_start/--slice_end** ⇒ N=576 的**全量 24 步** [1,4,…,576];
#     K = min(max_t((⌊√t⌋+1)²−1), N) = min(624, 576) = **576 = N**（无花瓶 register）
#   · --decoder_depth **2**（= train_v2.py 默认, 即历史 448 臂的深度; 显式写出以便 args.json 留档）
#   · 40 epoch = **8,760 优化步**; 每卡 bs **16** × 2 卡 × grad_accum **1** = 全局 32
#     ⇒ `steps_per_epoch = 7009//32 = 219`、`warmup_steps = 262`（3% 默认）
#     ⇒ **不需要 run_896 那个 --warmup_ratio 0.0075 的补丁**（grad_accum=1 时公式自洽）
#   · carry = **BPTT**; 平方块读窗口（layer_tap 已于 2026-09-23 移除）; 直接预测损失 mean_t L1
#   · lr 1.5e-4 cosine / wd 0.01 / clip 1.0 / seed 42 / 数据与切分同 896 臂
#   ⇒ 与文档里历史 448×252/BPTT 数字的**唯一差别 = fp16 + adamw_bnb_8bit**
#
# 用法（服务器）:
#   cd /root/autodl-tmp/sr-diffusion-v3-tap
#   setsid nohup bash tools/run_448_d2_bptt_fp16_8bit.sh >/dev/null 2>&1 </dev/null &
# 冒烟（单卡 2 步, 记得换 OUT/LOG）:
#   OUT=/root/autodl-tmp/construction_site/out_smoke_448 LOG=/root/train_logs/smoke_448.log \
#     NUM_GPUS=1 SMOKE=1 bash tools/run_448_d2_bptt_fp16_8bit.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/root/miniconda3/bin:$PATH
export HF_HUB_OFFLINE=1
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=${OUT:-/root/autodl-tmp/construction_site/out_448_d2_bptt_fp16_8bit}
LOG=${LOG:-/root/train_logs/448_d2_bptt_fp16_8bit.log}
SMOKE=${SMOKE:-0}
EPOCHS=${EPOCHS:-40}
BATCH=${BATCH:-16}
ACCUM=${ACCUM:-1}
OPTIM=${OPTIM:-adamw_bnb_8bit}
FP16=${FP16:-1}
mkdir -p "$(dirname "$LOG")" "$OUT"

SMOKE_ARGS=()
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS=(--smoke --limit 64 --eval_limit 16 --max_steps 2 --eval_every 1)
fi

FP16_ARGS=()
[ "$FP16" = "1" ] && FP16_ARGS=(--fp16)

NUM_GPUS=${NUM_GPUS:-2} bash run_v2_train.sh \
  --data_dir /root/autodl-tmp/construction_site \
  --dino_dir /root/autodl-tmp/models/dinov2-large \
  --output_dir "$OUT" \
  --model_input 448x252 \
  --decoder_depth 2 \
  --epochs "$EPOCHS" --batch_size "$BATCH" --grad_accum "$ACCUM" \
  --optim "$OPTIM" \
  "${FP16_ARGS[@]}" \
  --eval_every 2000 --save_every 2000 --log_every 50 \
  "${SMOKE_ARGS[@]}" \
  > "$LOG" 2>&1
