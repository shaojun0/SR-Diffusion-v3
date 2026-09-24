#!/usr/bin/env bash
# 12 组实验串行编排：6 backbone x {224,448}，统一 40 epoch。
# 可重复执行：已完成（有 metrics.json）的配置自动跳过 -> 断点续跑。
set -u
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

CODE=/root/autodl-tmp/rrnet/code
CKPT=/root/autodl-tmp/rrnet/ckpt
LOGS=/root/autodl-tmp/rrnet/logs
PY=/root/miniconda3/bin/python
EPOCHS=${EPOCHS:-40}
BACKBONES=${BACKBONES:-"resnet-10 resnet-18 resnet-34 resnet-50 resnet-101 resnet-152"}
RESOLUTIONS=${RESOLUTIONS:-"224 448"}

mkdir -p "$CKPT" "$LOGS"
cd "$CODE"

STATUS="$LOGS/run_all.status"
echo "START $(date -Is) epochs=$EPOCHS backbones='$BACKBONES' res='$RESOLUTIONS'" >> "$STATUS"

for res in $RESOLUTIONS; do
  for bb in $BACKBONES; do
    tag="${bb}_${res}"
    out="$CKPT/$tag"
    if [ -f "$out/metrics.json" ]; then
      echo "SKIP  $tag (metrics.json exists)" | tee -a "$STATUS"
      continue
    fi
    echo "BEGIN $tag $(date -Is)" | tee -a "$STATUS"
    mkdir -p "$out"
    "$PY" -u train.py \
        --backbone "$bb" --resolution "$res" \
        --out_dir "$out" --epochs "$EPOCHS" \
        --num_workers 6 \
        > "$LOGS/train_$tag.log" 2>&1
    rc=$?
    echo "END   $tag rc=$rc $(date -Is)" | tee -a "$STATUS"
    if [ $rc -ne 0 ]; then
      echo "  !! $tag failed, tail:" | tee -a "$STATUS"
      tail -n 5 "$LOGS/train_$tag.log" | tee -a "$STATUS"
    fi
    nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee -a "$STATUS"
  done
done
echo "ALLDONE $(date -Is)" | tee -a "$STATUS"
"$PY" -u aggregate.py 2>&1 | tee -a "$STATUS"
