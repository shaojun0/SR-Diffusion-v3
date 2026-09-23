#!/bin/bash
# run_rerun_fixw4_10k.sh — 「自然数步值 + 每步固定读 4 个 token」全分辨率重跑
#
# 用户口径（2026-09-23 第二轮）：
#   「现在 t 不再是按 1 4 9 16 进行增长，而是按自然数的形式增长」
#   → 步值 = 自然数 1..⌈N/block⌉，每步固定读 block 个 z_s（`--step_plan fixed --block 4`）。
#   预算沿用上一轮：A 相 3000（单步全读 warm start）+ B 相 7000 = 总 1 万优化步。
#
# 与 `run_rerun_A3000_10k.sh`（平方块计划）的**唯一变量 = 采样计划**：
#   square: 步值 k²、|T|=√N（28/56/112/224/336 → 2/4/8/16/24 步，窗口宽 2k+1）
#   fixed : 步值 k 、|T|=N/4（→ 1/4/16/64/144 步，窗口宽恒 4；首步 5 含 z_cls）
# K=N、z_mode=patch、BPTT、lr、head_zero_init、warmup、seed、batch 全部一致。
#
# ⚠️ 成本：|T| 变为 N/4 ⇒ B 相单步成本约 ×(√N/4)，336² 约 6×（预计 ~5.5–6 h），
#    五臂合计预计 **6–8 h**。跑完自动顺序推进，用 screen/setsid 挂住。
#
# 输出：/root/autodl-tmp/srres/out_fixw4_10k/{size}_A3000-fixw4-patch/{result.json,args.json,final_model.pt}
# 日志：/root/logs/fixw4_{size}_A3000_10k.log
set -u
P=/root/miniconda3/bin/python
S=/root/sr-diffusion-v3/tools/sweep_res_train.py
DATA=/root/autodl-tmp/srres/data
OUT=/root/autodl-tmp/srres/out_fixw4_10k
LOG=/root/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT" "$LOG"

WARM=3000
BUDGET=7000
BLOCK=4
SIZES="${SIZES:-336 224 112 56 28}"

run_one () {   # $1=size $2=batch $3=accum
  local s=$1 b=$2 a=$3
  local log="$LOG/fixw4_${s}_A3000_10k.log"
  echo "===== START size=$s batch=$b accum=$a warm=$WARM B=$BUDGET block=$BLOCK $(date '+%F %T')"
  $P -u "$S" --size "$s" --z_mode patch --arm A3000 \
     --step_plan fixed --block "$BLOCK" \
     --batch "$b" --accum "$a" \
     --warm_steps "$WARM" --steps "$BUDGET" --warmup 100 \
     --lr 1.5e-4 --enc_lr_mult 1.0 --head_zero_init --eval_every 500 \
     --data_root "$DATA" \
     --train_dir "$DATA/DIV2K_train_HR" --val_dir "$DATA/DIV2K_valid_HR" \
     --model_dir /root/autodl-tmp/models/dinov2-small \
     --out_dir "$OUT" > "$log" 2>&1
  local rc=$?
  echo "===== END   size=$s rc=$rc $(date '+%F %T')"
  if [ "$rc" -eq 0 ]; then
    grep -E "^\[.*\] (A 相结束|DONE)" "$log" | tail -2
  else
    echo "----- FAILED size=$s, last log lines -----"
    tail -25 "$log"
  fi
  return $rc
}

for s in $SIZES; do
  case "$s" in
    336) run_one 336 8 2 ;;
    *)   run_one "$s" 16 1 ;;
  esac
done

echo "FIXW4_ALL_DONE $(date '+%F %T')"
