#!/bin/bash
# run_sweep2.sh — 第二遍：**预算随分辨率放大**（第一遍等预算的对照）
#
# 第一遍（等预算 2500 步）的结论是"越大越训不动"（224/336 停在 13.6 dB，
# 连 per-patch-mean 17.8 dB 都打不过）⇒ 那是**预算混入**，不是分辨率效应。
# 本遍把 nested 预算按 N 放大，并用 sweep_res_prep.py 预处理好的 .npy
# （mmap 直读，去掉每 epoch 重解码 2040×1404 PNG 的 CPU 瓶颈）。
set -u
cd /root/autodl-tmp/srres/code
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=/root/miniconda3/bin/python
S=/root/autodl-tmp/srres/code/tools/sweep_res_train.py
OUT=/root/autodl-tmp/srres/out

one () {  # $1=size $2=warm $3=nested $4=batch $5=accum
  echo "===== long size=$1 warm=$2 nested=$3 bs=$4x$5  $(date '+%F %T')"
  $P -u $S --size "$1" --z_mode patch --arm long --batch "$4" --accum "$5" \
     --warm_steps "$2" --steps "$3" --warmup 100 \
     --lr 1.5e-4 --enc_lr_mult 1.0 --head_zero_init \
     --out_dir "$OUT" 2>&1 | tail -5
  echo "===== long done size=$1  $(date '+%F %T')"
}

one 28  600 900  16 1
one 56  800 1500 16 1
one 112 1000 2800 16 1
one 224 1000 5000 16 1
one 336 1000 6000  8 2

echo "SWEEP2_ALL_DONE $(date '+%F %T')"
