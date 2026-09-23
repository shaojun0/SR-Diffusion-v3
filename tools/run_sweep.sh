#!/bin/bash
# run_sweep.sh — 分辨率扫描：逐个分辨率训小模型 → JPEG/WebP 基线 → 汇总报告
#
# 口径见 sweep_res_train.py 头部。要点：
#   · z_mode=patch（z_s = patch token；register 版在 800 张上 3500 步都学不动）
#   · BPTT（唯一路径）+ 直接预测损失 + 平方块读窗口（仓库原版解码器）
#   · head_zero_init；lr=1.5e-4（仓库配方）；有效 batch 统一 16
#   · 分辨率 28/56/112/224/336（14 的倍数，DINOv2 patch=14）
#   · 预算：warm(单步全读) 1000 步 + nested 1500 步 = 2500 步/分辨率
set -u
cd /root/autodl-tmp/srres/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/root/miniconda3/bin/python
S=/root/autodl-tmp/srres/code/tools/sweep_res_train.py
J=/root/autodl-tmp/srres/code/tools/sweep_res_jpeg.py
R=/root/autodl-tmp/srres/code/tools/sweep_res_report.py
OUT=/root/autodl-tmp/srres/out
LOG=/root/autodl-tmp/srres/logs
mkdir -p "$OUT" "$LOG"

WARM=${WARM:-1000}
BUDGET=${BUDGET:-1500}

run_one () {   # $1=size  $2=batch  $3=accum  $4=extra flags
  local s=$1 b=$2 a=$3 extra=$4
  echo "===== train size=$s batch=$b accum=$a $extra  $(date '+%F %T')"
  $P -u $S --size "$s" --z_mode patch --batch "$b" --accum "$a" \
     --warm_steps "$WARM" --steps "$BUDGET" --warmup 100 \
     --lr 1.5e-4 --enc_lr_mult 1.0 --head_zero_init \
     $extra --out_dir "$OUT" 2>&1 | tail -6
  echo "===== done size=$s $extra  $(date '+%F %T')"
}

# ── 主臂：BPTT ──
run_one 28  16 1 ""
run_one 56  16 1 ""
run_one 112 16 1 ""
run_one 224 16 1 ""
run_one 336  8 2 ""

# ── 对照臂（2026-09-23 删除）──
# 原 detach 对照臂（--detach, 56/224）随 model_v2 的 carry 开关一起移除:
# detach 与 BPTT 共用同一套权重形状、只能靠开关区分, 会静默算错; 历史结果见
# doc/2026-09-22/res_sweep/runs/*_detach-patch/（用当时 commit 的代码复现）。

# ── JPEG/WebP 基线（同图同分辨率）──
for s in 28 56 112 224 336; do
  echo "===== jpeg size=$s  $(date '+%F %T')"
  $P -u $J --size "$s" --out "$OUT/${s}_jpeg.json" 2>&1 | tail -3
done

# ── 汇总报告 + 图 ──
$P -u $R --out_dir "$OUT" --report "$OUT/sweep_res_report.md" \
   --arms bptt-patch 2>&1 | tail -20

echo "SWEEP_ALL_DONE $(date '+%F %T')"
