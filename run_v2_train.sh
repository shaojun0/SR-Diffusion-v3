#!/usr/bin/env bash
# run_v2_train.sh — phase1 v2 训练入口（多卡 DDP + bf16）
#
# 用法:
#   NUM_GPUS=2 ./run_v2_train.sh \
#       --data_dir /root/autodl-tmp/construction_site \
#       --dino_dir /root/autodl-tmp/models/dinov2-large \
#       --output_dir output/phase1_v2
# 单卡冒烟:
#   NUM_GPUS=1 ./run_v2_train.sh --smoke --limit 32 --max_steps 3 --eval_every 2 ...
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/root/miniconda3/bin:$PATH
export HF_HUB_OFFLINE=1

NUM_GPUS="${NUM_GPUS:-2}"
LOG="${LOG:-logs/train_v2.log}"
mkdir -p logs output

if [ "${NUM_GPUS}" -eq 1 ]; then
  exec python train_v2.py "$@"
fi

exec accelerate launch --multi_gpu --num_processes "${NUM_GPUS}" \
  --num_machines 1 --same_network \
  train_v2.py "$@"
