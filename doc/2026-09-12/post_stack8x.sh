#!/bin/bash
# 训练后处理 (stack8x): 新模型全量推理 + 基线同口径推理(若无) + 终版探针 +
# 各 checkpoint 探针 + per-step 增益分析 + A/B 对比图表。
export PATH=/root/miniconda3/bin:$PATH
export HF_HUB_OFFLINE=1
export PYTHONPATH=/root/autodl-tmp/sr-diffusion-v3-stack2x
cd /root/autodl-tmp/sr-diffusion-v3-stack2x || exit 99
TLOG=/root/train_logs/stack8x_slice05.log
POST=/root/train_logs/stack8x_POSTDONE
rm -f "$POST"
while ! grep -q "^TRAIN_EXIT=" "$TLOG" 2>/dev/null; do sleep 60; done
EX=$(grep "^TRAIN_EXIT=" "$TLOG" | tail -1 | cut -d= -f2)
if [ "$EX" != "0" ]; then echo "TRAIN_FAILED exit=$EX $(date "+%F %T")" > "$POST"; exit 1; fi
echo "TRAIN_OK $(date "+%F %T")" >> "$POST"
OUT=output/phase1_v2_stack8x_slice05
BASE=/root/autodl-tmp/sr-diffusion-v3-qmask/output/phase1_v2_block_slice05_blockdiag/final_model.pt
mkdir -p output/probe
D="--data_dir /root/autodl-tmp/construction_site --dino_dir /root/autodl-tmp/models/dinov2-large"

# 1) 新模型全量推理 (test 3004)
CUDA_VISIBLE_DEVICES=0 python infer_v2_test.py $D --final_model $OUT/final_model.pt \
  --output $OUT/infer_test.json --batch_size 32 --num_workers 8 > /root/train_logs/stack8x_infer.log 2>&1
echo "INFER_NEW_EXIT=$?" >> "$POST"

# 2) 基线全量推理 (已存在则跳过, 保持同口径复用)
if [ ! -f output/baseline_blockdiag_slice05_infer_test.json ]; then
  CUDA_VISIBLE_DEVICES=0 python infer_v2_test.py $D --final_model $BASE \
    --output output/baseline_blockdiag_slice05_infer_test.json --batch_size 32 --num_workers 8 > /root/train_logs/baseline_infer.log 2>&1
  echo "INFER_BASE_EXIT=$?" >> "$POST"
else
  echo "INFER_BASE_EXIT=0 (reused)" >> "$POST"
fi

# 3) 终版探针 (512 张)
CUDA_VISIBLE_DEVICES=0 python probe_step_collapse.py $D --ckpt $OUT/final_model.pt \
  --tag stack8x_slice05 --limit 512 --bs 16 --num_workers 8 \
  --out output/probe/probe_stack8x_slice05.json > /root/train_logs/probe_stack8x.log 2>&1
echo "PROBE_NEW_EXIT=$?" >> "$POST"

# 4) 基线探针 (已存在则跳过)
if [ ! -f output/probe/probe_baseline_blockdiag_slice05.json ]; then
  CUDA_VISIBLE_DEVICES=0 python probe_step_collapse.py $D --ckpt $BASE \
    --tag baseline_blockdiag_slice05 --limit 512 --bs 16 --num_workers 8 \
    --out output/probe/probe_baseline_blockdiag_slice05.json > /root/train_logs/probe_baseline.log 2>&1
  echo "PROBE_BASE_EXIT=$?" >> "$POST"
else
  echo "PROBE_BASE_EXIT=0 (reused)" >> "$POST"
fi

# 5) 各 checkpoint 探针 (128 张, 看训练过程中 step 分工的演化)
MI=$OUT/model_info.json
for ck in $OUT/checkpoint-*; do
  [ -d "$ck" ] || continue
  step=${ck##*-}
  o=output/probe/probe_stack8x_slice05_step${step}.json
  [ -f "$o" ] && continue
  cp -f "$MI" "$ck/model_info.json"
  WP="$ck/final_model.pt"
  if [ -f "$ck/model.safetensors" ]; then
    python - "$ck/model.safetensors" "$WP" <<"PY"
import sys, torch
from safetensors.torch import load_file
torch.save(load_file(sys.argv[1]), sys.argv[2])
PY
  elif [ -f "$ck/pytorch_model.bin" ]; then cp -f "$ck/pytorch_model.bin" "$WP"
  else echo "NO_WEIGHTS $ck" >> "$POST"; continue; fi
  CUDA_VISIBLE_DEVICES=0 python probe_step_collapse.py $D --ckpt "$WP" \
    --tag "stack8x_slice05_step${step}" --limit 128 --bs 16 --num_workers 8 \
    --out "$o" >> /root/train_logs/probe_stack8x_ckpt.log 2>&1
  echo "CKPT $step EXIT=$?" >> "$POST"
done

# 6) A/B 全量对比 (markdown + png)
CUDA_VISIBLE_DEVICES=0 python ab_compare_stack2x.py \
  --new $OUT/infer_test.json --base output/baseline_blockdiag_slice05_infer_test.json \
  --new_log $TLOG --base_log /root/train_logs/blockdiag_slice05.log \
  --new_name "self.stack 8x (2048/16/8/dropout0.05)" --new_spe 219 --base_spe 219 \
  --md $OUT/AB_stack8x.md --plot $OUT/AB_stack8x.png >> /root/train_logs/stack8x_ab.log 2>&1
echo "AB_EXIT=$?" >> "$POST"

# 7) 各采样 step 增益分析
python step_gain_analysis.py --md $OUT/STEP_GAIN.md --json $OUT/STEP_GAIN.json \
  --probe baseline=output/probe/probe_baseline_blockdiag_slice05.json \
  --probe stack2x_lr1e4=output/probe/probe_stack2x_lr1e4_slice05.json \
  --probe stack8x=output/probe/probe_stack8x_slice05.json \
  --glob "stack8x_step=output/probe/probe_stack8x_slice05_step*.json" \
  --infer baseline=output/baseline_blockdiag_slice05_infer_test.json \
  --infer stack2x_lr1e4=output/phase1_v2_stack2x_lr1e4_slice05/infer_test.json \
  --infer stack8x=$OUT/infer_test.json >> /root/train_logs/stack8x_gain.log 2>&1
echo "GAIN_EXIT=$?" >> "$POST"

echo "POST_DONE $(date "+%F %T")" >> "$POST"
