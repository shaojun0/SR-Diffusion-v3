#!/bin/bash
# 塌缩守卫 (臂① @ depth-8): 连续 8 条日志 grad_norm<1e-2 即判定塌缩并终止
# (健康臂 grad ~0.2-3; stack8x 实测中位数 2.32)。
export PATH=/root/miniconda3/bin:$PATH
TLOG=/root/train_logs/stepid_d8_slice05.log
G=/root/train_logs/stepid_d8_GUARD
rm -f "$G"
while true; do
  grep -q "^TRAIN_EXIT=" "$TLOG" 2>/dev/null && exit 0
  bad=$(tr "\r" "\n" < "$TLOG" 2>/dev/null | grep -oE "\x27grad_norm\x27: \x27[0-9.eE+-]+\x27" | tail -8 | sed "s/.*: .\([0-9.eE+-]*\)./\1/" | awk "\$1<1e-2" | wc -l)
  if [ "${bad:-0}" -ge 8 ]; then
    echo "COLLAPSE_DETECTED $(date "+%F %T"): last 8 grad_norm all <1e-2 -> kill" >> "$G"
    pkill -9 -f "output/phase1_v2_stepid_d8_slice05"
    echo "KILLED $(date "+%F %T")" >> "$G"
    exit 0
  fi
  sleep 60
done
