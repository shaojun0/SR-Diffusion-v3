#!/usr/bin/env bash
# run_224_d4_bptt_tap.sh — 224×126 / depth=4 / BPTT / **DINOv2 逐层 tap（金字塔读出）**
#
# 与基线 out_224_d4_bptt（平方块读窗口, z_s 全部来自末层）**唯一差别** = --layer_tap:
#   · 24 层 DINOv2 按每 2 层一组 = 12 组, 与 12 个采样步**从顶到底**一一对应;
#   · 从顶往下第 g 组（g=1..12）在该组两层之后读出 2g−1 个 register
#     （g=1 → 1 个、g=2 → 3 个、… g=12 → 23 个, 合计 1+3+…+23 = 144 = K）,
#     **读出即从序列里删除**（"用掉的向量不进入下一层"）;
#   · 解码器第 i 步只读第 i 组那 2i−1 个向量（不读 z_cls）⇒ 顶部少量向量走满
#     24 层（语义构建）/ 底部大量向量只走 2 层（细节描绘）, 即 YOLO/FPN 式深浅分工。
#
# 其余**完全沿用基线配方**（可比对照）:
#   construction_site 7,009 训练 / test 3,004; 224×126 ⇒ patches 16×9=144;
#   40 epoch = 8,760 步; 2 卡 × bs16（全局 32）; lr 1.5e-4 cosine + warmup 3%;
#   BPTT（carry_detach=False, 仓库当前默认）; decoder depth=4; 直接预测损失。
#
# 用法（服务器）:
#   cd /root/autodl-tmp/sr-diffusion-v3-tap
#   setsid nohup bash tools/run_224_d4_bptt_tap.sh > /dev/null 2>&1 &
# 冒烟（单卡, 4 步）:
#   NUM_GPUS=1 SMOKE=1 bash tools/run_224_d4_bptt_tap.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/root/miniconda3/bin:$PATH
export HF_HUB_OFFLINE=1
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=${OUT:-/root/autodl-tmp/construction_site/out_224_d4_bptt_tap}
LOG=${LOG:-/root/train_logs/tap_224_d4_bptt.log}
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
  --layer_tap \
  --epochs 40 --batch_size 16 \
  --eval_every 2000 --save_every 2000 --log_every 50 \
  "${SMOKE_ARGS[@]}" \
  > "$LOG" 2>&1
