#!/bin/bash
export PATH=/root/miniconda3/bin:$PATH
export HF_HUB_OFFLINE=1
cd /root/autodl-tmp/sr-diffusion-v3-stepid || exit 99
TLOG=/root/train_logs/stepid_d4_slice05.log
POSTF=/root/train_logs/stepid_d4_POSTDONE
rm -f "$POSTF"
while ! grep -q "^TRAIN_EXIT=" "$TLOG" 2>/dev/null; do sleep 60; done
EX=$(grep "^TRAIN_EXIT=" "$TLOG" | tail -1 | cut -d= -f2)
if [ "$EX" != "0" ]; then echo "TRAIN_FAILED exit=$EX $(date "+%F %T")" > "$POSTF"; exit 1; fi
echo "TRAIN_OK $(date "+%F %T")" >> "$POSTF"
OUT=output/phase1_v2_stepid_d4_slice05
S2X=/root/autodl-tmp/sr-diffusion-v3-stack2x
mkdir -p output/probe
D="--data_dir /root/autodl-tmp/construction_site --dino_dir /root/autodl-tmp/models/dinov2-large"
CUDA_VISIBLE_DEVICES=0 python infer_v2_test.py $D --final_model $OUT/final_model.pt \
  --output $OUT/infer_test.json --batch_size 32 --num_workers 8 > /root/train_logs/stepid_d4_infer.log 2>&1
echo "INFER_NEW_EXIT=$?" >> "$POSTF"
CUDA_VISIBLE_DEVICES=0 python probe_step_collapse.py $D --ckpt $OUT/final_model.pt \
  --tag stepid_d4_slice05 --limit 512 --bs 16 --num_workers 8 \
  --out output/probe/probe_stepid_d4_slice05.json > /root/train_logs/stepid_d4_probe.log 2>&1
echo "PROBE_NEW_EXIT=$?" >> "$POSTF"
MI=$OUT/model_info.json
for ck in $OUT/checkpoint-*; do
  [ -d "$ck" ] || continue
  step=${ck##*-}
  o=output/probe/probe_stepid_d4_slice05_step${step}.json
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
  else echo "NO_WEIGHTS $ck" >> "$POSTF"; continue; fi
  CUDA_VISIBLE_DEVICES=0 python probe_step_collapse.py $D --ckpt "$WP" \
    --tag "stepid_d4_slice05_step${step}" --limit 128 --bs 16 --num_workers 8 \
    --out "$o" >> /root/train_logs/stepid_d4_ckpt.log 2>&1
  echo "CKPT $step EXIT=$?" >> "$POSTF"
done
python step_gain_analysis.py --md STEP_GAIN_stepid.md --json STEP_GAIN_stepid.json \
  --probe baseline=$S2X/output/probe/probe_baseline_blockdiag_slice05.json \
  --probe stack2x_lr1e4=$S2X/output/probe/probe_stack2x_lr1e4_slice05.json \
  --probe stepid_d4=output/probe/probe_stepid_d4_slice05.json \
  --glob "stepid_d4_step=output/probe/probe_stepid_d4_slice05_step*.json" \
  --infer baseline=$S2X/output/baseline_blockdiag_slice05_infer_test.json \
  --infer stack2x_lr1e4=$S2X/output/phase1_v2_stack2x_lr1e4_slice05/infer_test.json \
  --infer stepid_d4=$OUT/infer_test.json > /root/train_logs/stepid_d4_gain.log 2>&1
echo "GAIN_EXIT=$?" >> "$POSTF"
echo "POST_DONE $(date "+%F %T")" >> "$POSTF"
