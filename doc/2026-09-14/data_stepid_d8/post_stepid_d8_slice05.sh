#!/bin/bash
# 训练后处理 (臂① @ depth-8): 全量 test 推理 + 终版探针(512) + 各 ckpt 探针(128)
# + 四臂 step-gain 分析(含 stack8x depth-8 对照) + A/B 对比。
export PATH=/root/miniconda3/bin:$PATH
export HF_HUB_OFFLINE=1
export PYTHONPATH=/root/autodl-tmp/sr-diffusion-v3-stepid-d8
cd /root/autodl-tmp/sr-diffusion-v3-stepid-d8 || exit 99
TLOG=/root/train_logs/stepid_d8_slice05.log
POST=/root/train_logs/stepid_d8_POSTDONE
rm -f "$POST"
while ! grep -q "^TRAIN_EXIT=" "$TLOG" 2>/dev/null; do sleep 60; done
EX=$(grep "^TRAIN_EXIT=" "$TLOG" | tail -1 | cut -d= -f2)
if [ "$EX" != "0" ]; then echo "TRAIN_FAILED exit=$EX $(date "+%F %T")" > "$POST"; exit 1; fi
echo "TRAIN_OK $(date "+%F %T")" >> "$POST"

OUT=output/phase1_v2_stepid_d8_slice05
S2X=/root/autodl-tmp/sr-diffusion-v3-stack2x
mkdir -p output/probe
D="--data_dir /root/autodl-tmp/construction_site --dino_dir /root/autodl-tmp/models/dinov2-large"

# 1) 新模型全量推理 (test 3004)
CUDA_VISIBLE_DEVICES=0 python infer_v2_test.py $D --final_model $OUT/final_model.pt \
  --output $OUT/infer_test.json --batch_size 32 --num_workers 8 > /root/train_logs/stepid_d8_infer.log 2>&1
echo "INFER_NEW_EXIT=$?" >> "$POST"

# 2) 终版探针 (512 张)
CUDA_VISIBLE_DEVICES=0 python probe_step_collapse.py $D --ckpt $OUT/final_model.pt \
  --tag stepid_d8_slice05 --limit 512 --bs 16 --num_workers 8 \
  --out output/probe/probe_stepid_d8_slice05.json > /root/train_logs/stepid_d8_probe.log 2>&1
echo "PROBE_NEW_EXIT=$?" >> "$POST"

# 3) 各 checkpoint 探针 (128 张, 看训练过程中 step 分工的演化)
MI=$OUT/model_info.json
for ck in $OUT/checkpoint-*; do
  [ -d "$ck" ] || continue
  step=${ck##*-}
  o=output/probe/probe_stepid_d8_slice05_step${step}.json
  [ -f "$o" ] && continue
  cp -f "$MI" "$ck/model_info.json" 2>/dev/null
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
    --tag "stepid_d8_slice05_step${step}" --limit 128 --bs 16 --num_workers 8 \
    --out "$o" >> /root/train_logs/stepid_d8_ckpt.log 2>&1
  echo "CKPT $step EXIT=$?" >> "$POST"
done

# 4) A/B 全量对比 vs depth-8 基线 stack8x (单变量: 仅 step_embed)
if [ -f "$S2X/output/phase1_v2_stack8x_slice05/infer_test.json" ]; then
  CUDA_VISIBLE_DEVICES=0 python ab_compare_stack2x.py \
    --new $OUT/infer_test.json --base $S2X/output/phase1_v2_stack8x_slice05/infer_test.json \
    --new_log $TLOG --base_log /root/train_logs/stack8x_slice05.log \
    --new_name "arm1 stepid deep8 (2048/16/8 + step_embed)" \
    --new_spe 219 --base_spe 219 \
    --md $OUT/AB_stepid_d8_vs_stack8x.md --plot $OUT/AB_stepid_d8_vs_stack8x.png \
    >> /root/train_logs/stepid_d8_ab.log 2>&1
  echo "AB_STACK8X_EXIT=$?" >> "$POST"
else
  echo "AB_STACK8X_EXIT=SKIP_NO_BASE" >> "$POST"
fi

# 5) 各采样 step 增益分析 (baseline / stack2x / stack8x / 臂①@depth8 四路)
python step_gain_analysis.py --md $OUT/STEP_GAIN.md --json $OUT/STEP_GAIN.json \
  --probe baseline=$S2X/output/probe/probe_baseline_blockdiag_slice05.json \
  --probe stack2x_lr1e4=$S2X/output/probe/probe_stack2x_lr1e4_slice05.json \
  --probe stack8x=$S2X/output/probe/probe_stack8x_slice05.json \
  --probe stepid_d8=output/probe/probe_stepid_d8_slice05.json \
  --glob "stepid_d8_step=output/probe/probe_stepid_d8_slice05_step*.json" \
  --infer baseline=$S2X/output/baseline_blockdiag_slice05_infer_test.json \
  --infer stack2x_lr1e4=$S2X/output/phase1_v2_stack2x_lr1e4_slice05/infer_test.json \
  --infer stack8x=$S2X/output/phase1_v2_stack8x_slice05/infer_test.json \
  --infer stepid_d8=$OUT/infer_test.json > /root/train_logs/stepid_d8_gain.log 2>&1
echo "GAIN_EXIT=$?" >> "$POST"

echo "POST_DONE $(date "+%F %T")" >> "$POST"
