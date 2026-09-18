#!/bin/bash
# run_cot_full.sh — 全量 CoT 生成驱动（两车道，可续跑）。
# 1) 等所有数据集构建完成；2) 按行数均衡分道；3) 逐数据集生成到 $OUT。
set -u
export PATH=/root/miniconda3/bin:$PATH
ROOT=${COT_ROOT:-/root/autodl-tmp/cot_full}
OUT=${COT_OUTDIR:-/root/autodl-tmp/cot_out}
LOG=/root/translate_logs/cot_full
mkdir -p "$OUT" "$LOG"

ALL="aswin00000__ConstructionSiteCleanedDataSet iluvvatar__wood_surface_defects hf-vision__hardhat baizhanquan__FireDetectionDataset hayden-yuma__roadwork chandrabhuma__multi_building_defect_vqa kevincluo__structure_wildfire_damage_classification"

echo "[$(date +%H:%M:%S)] 等待构建完成 ..." >> "$LOG/driver.log"
for i in $(seq 1 180); do
  nb=$(pgrep -fc cot_build_full.py 2>/dev/null || echo 0)
  miss=""
  for ds in $ALL; do
    grep -l "\"dataset\": \"$ds\"" "$ROOT"/manifest*.jsonl >/dev/null 2>&1 || miss="$miss $ds"
  done
  if [ "${nb:-0}" = "0" ] && [ -z "$miss" ]; then
    echo "[$(date +%H:%M:%S)] 构建全部完成" >> "$LOG/driver.log"; break
  fi
  sleep 60
done

# ---- 按行数均衡分道 ----
python3 - "$ROOT" "$LOG" "$ALL" <<'PY'
import glob, json, os, sys
root, log, all_ds = sys.argv[1], sys.argv[2], sys.argv[3].split()
cnt = {}
for mf in glob.glob(os.path.join(root, "manifest*.jsonl")):
    for line in open(mf, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["dataset"] in all_ds:
            cnt[r["dataset"]] = cnt.get(r["dataset"], 0) + 1
lanes = [[], []]
load = [0, 0]
for ds, n in sorted(cnt.items(), key=lambda x: -x[1]):
    k = 0 if load[0] <= load[1] else 1
    lanes[k].append(ds); load[k] += n
for i, name in enumerate(("A", "B")):
    with open(os.path.join(log, "lane%s.txt" % name), "w") as f:
        for ds in lanes[i]:
            f.write("%s\n" % ds)
print("laneA=%s load=%d | laneB=%s load=%d" % (lanes[0], load[0], lanes[1], load[1]))
with open(os.path.join(log, "plan.txt"), "w") as f:
    f.write("laneA=%s load=%d\nlaneB=%s load=%d\ntotal=%d\n" % (lanes[0], load[0], lanes[1], load[1], sum(load)))
PY

run_lane() {
  local lane="$1" port="$2"
  local logf="$LOG/lane${lane}.log"
  : > "$logf"
  while read -r ds; do
    [ -z "$ds" ] && continue
    echo "[$(date +%H:%M:%S)] lane${lane} START $ds" >> "$logf"
    COT_ROOT="$ROOT" python3 /root/translate/cot_generate.py --datasets "$ds" \
      --base-url "http://127.0.0.1:${port}/v1" --batch-size 8 --out-dir "$OUT" >> "$logf" 2>&1
    echo "[$(date +%H:%M:%S)] lane${lane} rc=$? $ds" >> "$logf"
  done < "$LOG/lane${lane}.txt"
  echo "[$(date +%H:%M:%S)] LANE${lane} ALL DONE" >> "$logf"
}

run_lane A 8100 &
PA=$!
run_lane B 8101 &
PB=$!
wait $PA; wait $PB
rm -f /root/autodl-tmp/cot_out/ALL_DONE
for f in "$OUT"/*.jsonl; do [ -e "$f" ] && wc -l < "$f"; done | paste -sd+ | bc > "$OUT/TOTAL_LINES.txt" 2>/dev/null || true
echo "ALL DONE $(date +%H:%M:%S)" > "$OUT/ALL_DONE"
echo "[$(date +%H:%M:%S)] ALL DONE" >> "$LOG/driver.log"
