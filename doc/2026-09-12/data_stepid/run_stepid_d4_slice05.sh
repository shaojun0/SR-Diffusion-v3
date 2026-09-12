#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
export HF_HUB_OFFLINE=1
export PYTHONPATH=/root/autodl-tmp/sr-diffusion-v3-stepid
cd /root/autodl-tmp/sr-diffusion-v3-stepid || exit 99
LOG=/root/train_logs/stepid_d4_slice05.log
echo "==== [wrapper] $(date "+%F %T") start ====" | tee -a "$LOG"
accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --same_network train_v2.py \
  --data_dir /root/autodl-tmp/construction_site --dino_dir /root/autodl-tmp/models/dinov2-large \
  --output_dir output/phase1_v2_stepid_d4_slice05 --model_input 448x252 --canvas 1600x900 --angle_step 0.5 \
  --epochs 40 --max_steps 8760 --batch_size 16 --grad_accum 1 \
  --lr 1.0e-4 --weight_decay 0.01 --warmup_ratio 0.03 --grad_clip 1.0 \
  --num_workers 8 --limit 0 --eval_limit 0 --eval_every 2000 --save_every 2000 --log_every 20 \
  --seed 42 --heads 16 --mlp_ratio 4.0 --decoder_depth 4 --stack_dim 2048 --decoder_dropout 0.05 \
  --slice_start 0 --slice_end 5 >> "$LOG" 2>&1
EXIT=$?
echo "TRAIN_EXIT=$EXIT" >> "$LOG"
echo "==== [wrapper] $(date "+%F %T") finished exit=$EXIT ====" >> "$LOG"
