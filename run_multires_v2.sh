#!/bin/bash
# 六源全量微调（ImageNet/COCO/DIV2K/Flickr2K/OpenImages/Vimeo）
# 从第一轮 ckpt_2000 续训（权重+优化器），优化步归零重跑 warmup+cosine
cd /root/autodl-tmp/srdiff-multi
export PATH=/root/miniconda3/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CKPT=/root/autodl-tmp/output/multires_natural/ckpt_2000.pt
STEPS=$(python epoch_steps.py /root/autodl-tmp/data/index_cache_v2.json 6 6144 2 | awk "{print \$4}")
echo "[v2] steps_per_rank=$STEPS resume=$CKPT $(date)"
torchrun --nproc_per_node=2 --master_port=29550 train_multi.py \
  --output_dir /root/autodl-tmp/output/multires_natural_v2 \
  --data_cache /root/autodl-tmp/data/index_cache_v2.json \
  --resume $CKPT --reset_step_on_resume \
  --batch_size 6 --patch_budget 6144 --grad_ckpt_decoder --precision bf16 \
  --lr_new 3e-4 --lr_dino 1.5e-4 --weight_decay 0.01 \
  --warmup_ratio 0.03 --grad_clip 1.0 \
  --num_specials 144 --decoder_depth 2 --heads 8 --step_plan square \
  --num_workers 10 --log_every 20 --save_every 2000 \
  --epochs 1 --max_steps $STEPS
echo RUN_V2_DONE
