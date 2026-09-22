#!/usr/bin/env bash
# run_224_d4_bptt_slice1.sh — 224×126 / depth=4 / BPTT / **切片 [1:12]（11 步）**
#
# 与基线 out_224_d4_bptt 的**唯一差别** = 采样步集切片:
#   基线（[0:12], 12 步）: t = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144]
#                          窗口宽 = [4, 5, 7, 9, 11, …, 23, 1]（首步 A[0:3] = z_cls + z_s[0..2]）
#   本臂（[1:12], 11 步）: t = [4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144]
#                          窗口宽 = [9, 7, 9, 11, …, 23, 1]（首步 A[0:8] = z_cls + z_s[0..7]）
#   ⇒ 等于**去掉 t=1 那一步**，把原来的 t=1 与 t=4 两个窗口合并成首步的 [0,8]。
#   ⇒ K 仍是 144、A[0..144] 仍被完整覆盖（无花瓶 register，已本地断言）。
#
# 其余**完全沿用基线配方**（可比对照）:
#   construction_site 7,009 训练 / test 3,004 + 训练集随机 1,000 抽样评测;
#   224×126 ⇒ patches 16×9=144; 40 epoch = 8,760 步; 2 卡 × bs16（全局 32）;
#   lr 1.5e-4 cosine + warmup 3%; BPTT（carry_detach=False, 仓库默认）; decoder depth=4;
#   直接预测损失; **不用 --layer_tap**（平方块读窗口, z_s 全部来自 DINOv2 末层）。
#
# 用法（服务器）:
#   cd /root/autodl-tmp/sr-diffusion-v3-tap
#   setsid nohup bash tools/run_224_d4_bptt_slice1.sh > /dev/null 2>&1 < /dev/null &
# 冒烟（单卡, 4 步）:
#   NUM_GPUS=1 SMOKE=1 OUT=/root/autodl-tmp/construction_site/out_smoke_slice1 \
#     LOG=/root/train_logs/slice1_smoke.log bash tools/run_224_d4_bptt_slice1.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/root/miniconda3/bin:$PATH
export HF_HUB_OFFLINE=1
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=${OUT:-/root/autodl-tmp/construction_site/out_224_d4_bptt_slice1}
LOG=${LOG:-/root/train_logs/slice1_224_d4_bptt.log}
SMOKE=${SMOKE:-0}
mkdir -p "$(dirname "$LOG")" "$OUT"

SMOKE_ARGS=()
if [ "$SMOKE" = "1" ]; then
  SMOKE_ARGS=(--smoke --limit 64 --eval_limit 32 --max_steps 4 --eval_every 2)
fi

NUM_GPUS=${NUM_GPUS:-2} bash run_v2_train.sh \
  --data_dir /root/autodl-tmp/construction_site \
  --dino_dir /root/autodl-tmp/models/dinov2-large \
  --output_dir "$OUT" \
  --model_input 224x126 \
  --decoder_depth 4 \
  --slice_start 1 --slice_end 12 \
  --epochs 40 --batch_size 16 \
  --eval_every 2000 --save_every 2000 --log_every 50 \
  "${SMOKE_ARGS[@]}" \
  > "$LOG" 2>&1
