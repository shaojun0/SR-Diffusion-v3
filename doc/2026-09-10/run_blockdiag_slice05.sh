#!/bin/bash
# blockdiag slice[0:5] 单卡训练
# 代码 = SR-Diffusion-v3 main HEAD d72e5eb (2026-09-10)
# 与历史 run_train_slice05.sh 的唯一差别:
#   num_processes 2 -> 1 (降配置只剩一张卡)
#   batch_size   16 -> 32 (保持全局 batch = 32, 控制变量)
# 不显式传 --query_mask_mode, 走新默认 blockdiag (train_v2.py 会打印并在 model_info.json 记录)
export PATH=/root/miniconda3/bin:$PATH
export HF_HUB_OFFLINE=1
export PYTHONPATH=/root/autodl-tmp/sr-diffusion-v3-qmask
cd /root/autodl-tmp/sr-diffusion-v3-qmask || exit 99

echo "==== [wrapper] $(date '+%F %T') start ===="
python train_v2.py \
  --data_dir /root/autodl-tmp/construction_site --dino_dir /root/autodl-tmp/models/dinov2-large \
  --output_dir output/phase1_v2_block_slice05_blockdiag --model_input 448x252 --canvas 1600x900 --angle_step 0.5 \
  --epochs 40 --max_steps 8760 --batch_size 32 --grad_accum 1 \
  --lr 1.5e-4 --weight_decay 0.01 --warmup_ratio 0.03 --grad_clip 1.0 \
  --num_workers 8 --limit 0 --eval_limit 0 --eval_every 2000 --save_every 2000 --log_every 20 \
  --seed 42 --heads 8 --mlp_ratio 4.0 --decoder_depth 2 --slice_start 0 --slice_end 5 \
  > /root/train_logs/blockdiag_slice05.log 2>&1
EXIT=$?
echo "TRAIN_EXIT=$EXIT" >> /root/train_logs/blockdiag_slice05.log
echo "==== [wrapper] $(date '+%F %T') finished exit=$EXIT ===="
