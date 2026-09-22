#!/usr/bin/env bash
# eval_448_d2_bptt_fp16_8bit.sh — 等「448×252 / d2 / BPTT / fp16+8bit」训练结束 → 自动评测
#
# 模型: /root/autodl-tmp/construction_site/out_448_d2_bptt_fp16_8bit/final_model.pt
#   输入 448×252 ⇒ patches 32×18 = 576, specials = **576**（K=N, 全量 24 步 [1,4,…,576]）
#   训练集 construction_site train 7,009 张; 测试集 test 3,004 张
#
# 评测三件事（与 tools/eval_224_d4_bptt.sh 同口径, 便于逐行对比）:
#   1) test  3,004 张（schema v2: PSNR/MS-SSIM/内容区/平凡基线/逐图逐step）
#   2) train 随机 1,000 张（seed=42 ⇒ 看 train/test 差距 = 过拟合程度）
#   3) 经典 codec 基线在同一 448×252 画布上重跑（JPEG/WebP）⇒ 同 bpp 轴可比
set -uo pipefail
cd "$(dirname "$0")/.."

export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH="$PWD"
export HF_HUB_OFFLINE=1
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets

OUT=${OUT:-/root/autodl-tmp/construction_site/out_448_d2_bptt_fp16_8bit}
MODEL=$OUT/final_model.pt
EVAL=${EVAL:-/root/autodl-tmp/cot_l1/eval_448_d2}
CS=/root/autodl-tmp/construction_site
TRAIN1K=/root/autodl-tmp/construction_site_train1k
DINO=/root/autodl-tmp/models/dinov2-large
mkdir -p "$EVAL"

echo "[eval] 等训练进程出现（防竞态：先起看护也能等到）…"
for _ in $(seq 1 90); do pgrep -f "[t]rain_v2.py" >/dev/null 2>&1 && break; sleep 10; done
echo "[eval] 等训练进程退出…"
while pgrep -f "[t]rain_v2.py" >/dev/null 2>&1; do sleep 60; done
echo "[eval] 训练已退出，等 final_model.pt…"
for _ in $(seq 1 60); do [ -f "$MODEL" ] && break; sleep 30; done
if [ ! -f "$MODEL" ]; then echo "[eval] 没有 $MODEL，退出"; exit 1; fi
echo "[eval] model md5 = $(md5sum "$MODEL" | cut -d' ' -f1)"

infer() {  # $1=data_dir $2=out.json $3=limit
  python -u infer_v2_test.py --data_dir "$1" --dino_dir "$DINO" \
    --final_model "$MODEL" --output "$2" --model_input 448x252 \
    --batch_size 16 --num_workers 8 --limit "$3"
}

echo "[eval] (1/3) test 3,004 张"
infer "$CS" "$EVAL/test_448_d2_bptt_fp16_8bit.json" 0 2>&1 | tee "$EVAL/test.log"

echo "[eval] (2/3) train 随机 1,000 张"
python tools/make_train_sample.py --data_dir "$CS" --out_dir "$TRAIN1K" --n 1000 --seed 42
infer "$TRAIN1K" "$EVAL/train1k_448_d2_bptt_fp16_8bit.json" 0 2>&1 | tee "$EVAL/train1k.log"

echo "[eval] (3/3) 经典 codec 基线 @448x252"
python -u tools/baseline_jpeg_sweep.py --data_dir "$CS" --w 448 --h 252 \
  --out "$EVAL/baseline_classic_448.json" 2>&1 | tee "$EVAL/baseline.log"

echo "EVAL_ALL_DONE"
