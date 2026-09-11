#!/bin/bash
# self.stack 4x-depth 实验 (d_model 2048 = 2x / heads 16 = 2x / depth 8 = 4x base2 /
# dropout 0.05), slice[0:5], blockdiag, K=35, 8760 步 (40 epoch), 2x RTX PRO 6000 DDP.
#
# lr 依据 Step Law (arXiv:2503.04715v7, η(N,D)=1.79·N^-0.713·D^0.307):
#   · 绝对值口径 (论文 Eq.1 直接代入): N=847.671M, D=1.7158e8 -> η_opt = 2.60e-4
#     但 D 比论文拟合域下界 (2e9) 低 11.7x, 该项目已实测绝对外推高估 8~14x -> 不采用。
#   · 比值口径 (论文不变量: η ∝ N^-0.713), 锚在**已实测健康**的 stack2x@lr1.0e-4:
#       lr = 1.0e-4 * (847.671e6 / 579.080e6)^-0.713 = 1.0e-4 * 0.7621 = 7.62e-5
#     (对照: 从基线锚 1.5e-4 -> 7.82e-5; 两者差 3%)
#   -> 取 7.6e-5
export PATH=/root/miniconda3/bin:$PATH
export HF_HUB_OFFLINE=1
export PYTHONPATH=/root/autodl-tmp/sr-diffusion-v3-stack2x
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/sr-diffusion-v3-stack2x || exit 99
LOG=/root/train_logs/stack8x_slice05.log
echo "==== [wrapper] $(date "+%F %T") start ====" | tee -a "$LOG"
# 显存: depth=8 在 bs16/卡 会 OOM (实测 92.3/95 GiB 已分配, 反向第 1 步就炸);
# 因此每卡 bs 16->8 且 grad_accum 1->2 —— **全局 batch 仍 = 8x2卡x2 = 32**,
# 与基线/stack2x 完全一致 (梯度按 32 样本平均), 全 fp32 不变。
accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --same_network train_v2.py \
  --data_dir /root/autodl-tmp/construction_site --dino_dir /root/autodl-tmp/models/dinov2-large \
  --output_dir output/phase1_v2_stack8x_slice05 --model_input 448x252 --canvas 1600x900 --angle_step 0.5 \
  --epochs 40 --max_steps 8760 --batch_size 8 --grad_accum 2 \
  --lr 7.6e-5 --weight_decay 0.01 --warmup_ratio 0.03 --grad_clip 1.0 \
  --num_workers 8 --limit 0 --eval_limit 0 --eval_every 2000 --save_every 2000 --log_every 20 \
  --seed 42 --heads 16 --mlp_ratio 4.0 --decoder_depth 8 --stack_dim 2048 --decoder_dropout 0.05 \
  --slice_start 0 --slice_end 5 >> "$LOG" 2>&1
EXIT=$?
echo "TRAIN_EXIT=$EXIT" >> "$LOG"
echo "==== [wrapper] $(date "+%F %T") finished exit=$EXIT ====" >> "$LOG"
