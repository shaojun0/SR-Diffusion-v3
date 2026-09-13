#!/bin/bash
# Controlled 700-step screening A/B for the shared-[cls] collapse.
#
# Both variants run with the EXACT production config and schedule (max_steps 8760 ⇒ warmup
# 263 steps, identical cosine shape) and are killed after STEP_LIMIT steps. The first 700
# steps are therefore directly comparable with the collapsed arm③ v1 run (same seed/batch).
#
#   D1 ctl_sharedcls    : shared [cls] only, no step_embed
#   D2 allrows_sharedcls: shared [cls] + step_embed on ALL query rows  (= arm①'s scope)
export PATH=/root/miniconda3/bin:$PATH
export HF_HUB_OFFLINE=1
STEP_LIMIT=700

run_one () {
  local tag=$1 dir=$2
  local log=/root/train_logs/${tag}.log
  cd "$dir" || { echo "CD_FAIL $dir"; return 99; }
  rm -f "$log"
  setsid accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --same_network train_v2.py \
    --data_dir /root/autodl-tmp/construction_site --dino_dir /root/autodl-tmp/models/dinov2-large \
    --output_dir output/phase1_v2_${tag} --model_input 448x252 --canvas 1600x900 --angle_step 0.5 \
    --epochs 40 --max_steps 8760 --batch_size 16 --grad_accum 1 \
    --lr 1.0e-4 --weight_decay 0.01 --warmup_ratio 0.03 --grad_clip 1.0 \
    --num_workers 8 --limit 0 --eval_limit 0 --eval_every 2000 --save_every 2000 --log_every 20 \
    --seed 42 --heads 16 --mlp_ratio 4.0 --decoder_depth 4 --stack_dim 2048 --decoder_dropout 0.05 \
    --slice_start 0 --slice_end 5 > "$log" 2>&1 &
  local pid=$!
  local cur=0
  while kill -0 "$pid" 2>/dev/null; do
    cur=$(tail -c 6000 "$log" | tr '\r' '\n' | grep -oE '[0-9]+/8760' | tail -1 | cut -d/ -f1)
    [ -n "$cur" ] && [ "$cur" -ge "$STEP_LIMIT" ] && break
    sleep 15
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM -"$pid" 2>/dev/null; sleep 10; kill -KILL -"$pid" 2>/dev/null
  fi
  echo "SCREEN_KILLED_AT=${cur:-?} $tag $(date '+%F %T')" >> "$log"
  echo "=== $tag killed at step ${cur:-?} ==="
}

run_one ctl_sharedcls_d4 /root/autodl-tmp/sr-diffusion-v3-stepid-shcls-ctl
run_one allrows_sharedcls_d4 /root/autodl-tmp/sr-diffusion-v3-stepid-shcls-allrows
echo "AB_DIAG_DONE $(date '+%F %T')"
