#!/usr/bin/env bash
# run_v2_boundary.sh — 边界实验: 最大特殊 token 数 K=64 + 只采样两个时刻
# decoder_steps=[32,64]（register 式, num_specials 与 N 解耦后唯一路径）
#
# 实验目的（压缩边界探索, 2026-09-02 语义更新——register 式恒开, 无
# ReEncoder/causal_specials/register_specials/loss_min_t 等旧参数）:
#   · --num_specials 64（显式 K）: 压缩键上限压到 64（64/576 ≈ 11% 键还原
#     全部 patch——边界条件探测压缩率上界）; 显式 K 断言: max(采样步) ≤ K,
#     这里 max(decoder_steps)=64 ≤ 64 ✓;
#   · decoder_steps=[32,64]: 训练/推理只走两个采样时刻——块起点 32（块
#     5=[25..35], 首步前缀规则可见 0..35）与 64（块 8=[64..80] 被 K=64
#     截到 [64..64], 只见自身; 位置 36..63 不被覆盖——显式非默认 steps
#     不保证全覆盖, 见 DESIGN_v2_num_specials_from_max_steps.md）;
#   · 序列 = [cls; specials(64); patches(576)] = 1+64+576 token, DINO 24 层
#     全双向算出 z_s（与 K=64 配套的显存/速度边界验证）。
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
