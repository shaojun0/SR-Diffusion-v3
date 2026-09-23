#!/bin/bash
# run_rerun_A3000_10k.sh — 分辨率扫描「修正配方」重跑：A 相 3000 + B 相 7000 = 总 1 万步
#
# 动机（HANDOFF_20260923_main_cleanup.md §2 / REPORT_224_A3000_verdict.md）：
#   原扫描的 A 相（单步全读 warm start）在所有分辨率恒 1000 步，而 224²/336²
#   实测要 1500~2000 步才逃出平凡解 ⇒ 高分辨率臂是「从没逃逸的状态」进 B 相的。
#   224² 判决实验（A=3000 / B=5000）已把平线变成单调降（14.26→18.69 dB）。
#   本脚本把该修正配方推广到全分辨率，**从 336² 开始依次往下补**
#   （顺序由用户指定：336 → 224 → 112 → 56 → 28）。
#
# 与旧臂的唯一变量：warm_steps 1000→3000、B 相预算 → 7000（总 10000 步）。
# batch/accum 沿用各分辨率历史口径（336: 8×2；其余 16×1），有效 batch 恒 16。
# 其余（z_mode=patch / BPTT / lr 1.5e-4 / enc_lr_mult 1.0 / head_zero_init /
# warmup 100 / seed 42 / TF32）与旧扫描逐项一致。
#
# 输出：/root/autodl-tmp/srres/out_A3000_10k/{size}_A3000-patch/{result.json,final_model.pt,args.json}
# 日志：/root/logs/rerun_{size}_A3000_10k.log
# ⚠️ 故意不写进 out_224_A3000/，以免覆盖 224 判决实验的产物。
set -u
P=/root/miniconda3/bin/python
S=/root/sr-diffusion-v3/tools/sweep_res_train.py
DATA=/root/autodl-tmp/srres/data
OUT=/root/autodl-tmp/srres/out_A3000_10k
LOG=/root/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT" "$LOG"

WARM=3000
BUDGET=7000
SIZES="${SIZES:-336 224 112 56 28}"

run_one () {   # $1=size $2=batch $3=accum
  local s=$1 b=$2 a=$3
  local log="$LOG/rerun_${s}_A3000_10k.log"
  echo "===== START size=$s batch=$b accum=$a warm=$WARM B=$BUDGET $(date '+%F %T')"
  $P -u "$S" --size "$s" --z_mode patch --arm A3000 --batch "$b" --accum "$a" \
     --warm_steps "$WARM" --steps "$BUDGET" --warmup 100 \
     --lr 1.5e-4 --enc_lr_mult 1.0 --head_zero_init --eval_every 500 \
     --data_root "$DATA" \
     --train_dir "$DATA/DIV2K_train_HR" --val_dir "$DATA/DIV2K_valid_HR" \
     --model_dir /root/autodl-tmp/models/dinov2-small \
     --out_dir "$OUT" > "$log" 2>&1
  local rc=$?
  echo "===== END   size=$s rc=$rc $(date '+%F %T')"
  if [ "$rc" -eq 0 ]; then
    grep -E "^\[.*\] (A 相结束|DONE|PSNR)" "$log" | tail -4
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

echo "RERUN_ALL_DONE $(date '+%F %T')"
