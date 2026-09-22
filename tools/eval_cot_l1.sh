#!/usr/bin/env bash
# eval_cot_l1.sh — 等 Phase-1 训练结束 → 双模型 / 多测试集评估（像素 L1 判据）
#
# 两个模型（同一份当前代码，carry detach）：
#   BIG  = 本次 CoT 全量图片（75,717 / 7 数据集）训练产物（训练中，40 epoch / 85160 步）
#   BASE = 原始 construction_site 基线（run1_detach，8760 步，文档口径 23.72 / 3004 张）
# 测试集：
#   1) CoT 全量 test（7,570 张，随机 90/10 留出）
#   2) CoT 逐数据集 test（test-0000i 与 7 个数据集按构建顺序一一对应）
#   3) construction_site test（3,004 张，原始基线同测试集；仅对 BASE 无泄漏）
# 结果落 /root/autodl-tmp/cot_l1/eval/，最后打印 EVAL_ALL_DONE。
set -uo pipefail
cd "$(dirname "$0")/.."

export PATH=/root/miniconda3/bin:$PATH
export HF_HUB_OFFLINE=1
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_DATASETS_CACHE=/root/autodl-tmp/hf_cache/datasets

OUT=${OUT:-/root/autodl-tmp/cot_l1/output_phase1_v2}
BIG=$OUT/final_model.pt
BASE=${BASE:-/root/autodl-tmp/SR-Diffusion-v3/output/phase1_v2_run1_detach/final_model.pt}
EVAL=/root/autodl-tmp/cot_l1/eval
PARQ=/root/autodl-tmp/cot_l1/parquet
DINO=/root/autodl-tmp/models/dinov2-large
CS=/root/autodl-tmp/construction_site
mkdir -p "$EVAL/by_dataset"

DATASETS=(
  aswin00000__ConstructionSiteCleanedDataSet
  baizhanquan__FireDetectionDataset
  chandrabhuma__multi_building_defect_vqa
  hayden-yuma__roadwork
  hf-vision__hardhat
  iluvvatar__wood_surface_defects
  kevincluo__structure_wildfire_damage_classification
)

infer() {  # $1=model.pt $2=data_dir $3=out.json
  python infer_v2_test.py --data_dir "$2" --dino_dir "$DINO" \
    --final_model "$1" --output "$3" --batch_size 32 --num_workers 8
}

echo "[eval] 等训练进程退出…"
while pgrep -f "[t]rain_v2.py" >/dev/null 2>&1; do sleep 60; done
echo "[eval] 训练已退出，等 final_model.pt…"
for _ in $(seq 1 40); do [ -f "$BIG" ] && break; sleep 30; done
if [ ! -f "$BIG" ]; then
  echo "[eval] 没有 $BIG，退出（训练可能失败）"
  exit 1
fi
echo "[eval] BIG  md5 = $(md5sum "$BIG" | cut -d' ' -f1)"
echo "[eval] BASE md5 = $(md5sum "$BASE" | cut -d' ' -f1)"

for M in BIG BASE; do
  M2=$( [ "$M" = BIG ] && echo "$BIG" || echo "$BASE" )
  MD=$( [ "$M" = BIG ] && echo "bigdata" || echo "baseline" )
  mkdir -p "$EVAL/$MD/by_dataset"

  # 1) CoT 全量 test
  infer "$M2" "$PARQ" "$EVAL/$MD/cot_l1_test_all.json" \
    2>&1 | tee "$EVAL/$MD/cot_l1_test_all.log"

  # 2) 逐数据集
  i=0
  for ds in "${DATASETS[@]}"; do
    d="$EVAL/$MD/by_dataset/$ds"; mkdir -p "$d"
    ln -sf "$PARQ/test-0000${i}-of-00007.parquet" "$d/test-00000-of-00001.parquet"
    echo "[eval] $MD / $ds  ← test-0000${i}-of-00007.parquet"
    infer "$M2" "$d" "$EVAL/$MD/by_dataset/$ds.json" \
      2>&1 | tee "$EVAL/$MD/by_dataset/$ds.log"
    i=$((i + 1))
  done

  # 3) construction_site test（原始基线同测试集）
  infer "$M2" "$CS" "$EVAL/$MD/construction_site_test.json" \
    2>&1 | tee "$EVAL/$MD/construction_site_test.log"
done

echo "EVAL_ALL_DONE"
