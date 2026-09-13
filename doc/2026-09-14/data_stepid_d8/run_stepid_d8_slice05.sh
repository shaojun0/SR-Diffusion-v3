#!/bin/bash
# 臂① @ depth-8 (self.stack 8x): d_model 2048 / heads 16 / depth 8 / dropout 0.05,
# slice[0:5], blockdiag, K=35, 8760 步 (40 epoch), 2x RTX PRO 6000 DDP.
#
# 单变量对照: 与 stack8x (无注入, eval_recon 0.4310) 逐项相同, 仅多臂① 零初始化 step_embed
# (5x2048=10240 参数, 零初始化 => 训练起点逐位等于 stack8x)。
# lr 沿用 depth-8 处方 7.6e-5 (Step Law 比值口径, 见 REPORT_v2_stack8x_slice05.md);
# step_embed 参数量可忽略, 不触发重新定标。
# 显存: depth-8 在 bs16/卡 会 OOM (stack8x 实测), 故 bs 8 + grad_accum 2 => 全局 batch 仍 32。
export PATH=/root/miniconda3/bin:$PATH
export HF_HUB_OFFLINE=1
export PYTHONPATH=/root/autodl-tmp/sr-diffusion-v3-stepid-d8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/sr-diffusion-v3-stepid-d8 || exit 99
LOG=/root/train_logs/stepid_d8_slice05.log
echo "==== [wrapper] $(date "+%F %T") start ====" | tee -a "$LOG"
accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --same_network train_v2.py \
  --data_dir /root/autodl-tmp/construction_site --dino_dir /root/autodl-tmp/models/dinov2-large \
  --output_dir output/phase1_v2_stepid_d8_slice05 --model_input 448x252 --canvas 1600x900 --angle_step 0.5 \
  --epochs 40 --max_steps 8760 --batch_size 8 --grad_accum 2 \
  --lr 7.6e-5 --weight_decay 0.01 --warmup_ratio 0.03 --grad_clip 1.0 \
  --num_workers 8 --limit 0 --eval_limit 0 --eval_every 2000 --save_every 2000 --log_every 20 \
  --seed 42 --heads 16 --mlp_ratio 4.0 --decoder_depth 8 --stack_dim 2048 --decoder_dropout 0.05 \
  --slice_start 0 --slice_end 5 >> "$LOG" 2>&1
EXIT=$?
echo "TRAIN_EXIT=$EXIT" >> "$LOG"
echo "==== [wrapper] $(date "+%F %T") finished exit=$EXIT ====" >> "$LOG"
