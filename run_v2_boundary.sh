#!/usr/bin/env bash
# run_v2_boundary.sh — 边界实验: 最大特殊 token 数 K=64 + 只监督"前 32/前 64"
# 两个前缀序列（decoder_steps=[32,64], test 分支 OutputQueryDecoder）
#
# 实验目的（压缩边界探索）:
#   · K=64: 压缩键上限压到 64（对比默认 K=128 的 2.6% 键压缩, 这里是
#     64/576 ≈ 11% 键还原全部 patch——边界条件探测压缩率上界）;
#   · decoder_steps=[32,64]: 训练/推理只走两个采样时刻——KV 前缀长度 32
#     与 64（全量前缀）。32-token 前缀 vs 64-token 全量前缀的重建能力
#     对比, 检验"半压缩"下的信息保持。
#   · register_specials: specials 进 DINO 序列（1+64+576 token, 全双向）,
#     由 24 层算出 z_s —— 与 K=64 配套的显存/速度边界验证。
#
# 用法:
#   NUM_GPUS=2 ./run_v2_boundary.sh            # 完整训练 (40 epochs)
#   NUM_GPUS=1 ./run_v2_boundary.sh --smoke --limit 32 --max_steps 3
#                                             # 单卡冒烟（--smoke 不存
#                                             # checkpoint, 仍导出 final）
# 日志: LOG=logs/boundary_k64.log NUM_GPUS=1 ./run_v2_boundary.sh --smoke ...
# （"$@" 追加参数透传 train_v2.py, 如 --eval_every 500 --save_every 1000）
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/root/miniconda3/bin:$PATH
export HF_HUB_OFFLINE=1

NUM_GPUS="${NUM_GPUS:-2}"
LOG="${LOG:-logs/train_v2_boundary_k64.log}"
mkdir -p logs output

# 边界实验固定配置（用户指定）; 追加 "$@" 可覆盖/增补（如冒烟参数）
ARGS=(
  --data_dir /root/autodl-tmp/construction_site
  --dino_dir /root/autodl-tmp/models/dinov2-large
  --output_dir output/phase1_v2_boundary_k64
  --epochs 40
  --batch_size 16
  --num_workers 8
  --num_specials 64
  --decoder_steps "32,64"
  --loss_min_t 5
  --register_specials
)

if [ "${NUM_GPUS}" -eq 1 ]; then
  # 注意: exec + 管道 不会替换当前 shell（管道需 fork），会导致脚本
  # 继续执行下面的 accelerate 分支 → 用普通执行 + 显式 exit 传播退出码。
  python train_v2.py "${ARGS[@]}" "$@" 2>&1 | tee -a "${LOG}"
  exit "${PIPESTATUS[0]}"
fi

exec accelerate launch --multi_gpu --num_processes "${NUM_GPUS}" \
  --num_machines 1 --same_network \
  train_v2.py "${ARGS[@]}" "$@" 2>&1 | tee -a "${LOG}"
