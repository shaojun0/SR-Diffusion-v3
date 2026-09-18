#!/bin/bash
# mm-datasets-add2.sh -- true-resume supplement downloader (replaces mm-datasets-add.sh)
#
# Fixes the rc=18 death loop: the old fetch_one() truncated a .part and restarted
# from byte 0 after every mid-transfer socket close.  This wrapper drives
# /root/mm_add2.py, which always continues with `curl -C -`, promotes exact-size
# partials, retries with stall detection, and verifies byte counts against the
# mirror tree listing.
#
# Used by the running (setsid-detached) job:
#   setsid nohup /root/mm-datasets-add2.sh >>/root/mm-datasets-add2.console 2>&1 &
#   (structured lines go to /root/mm-datasets-add2.log; console holds the dupes)
#
# Re-runnable & idempotent: complete repos (status OK) are skipped, partial
# files continue.  Env: WORKERS (default 12), PASSES (default 3), FORCE=1 to redo.

set -u
export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_XET=1

BASE="/root/autodl-tmp/mm-datasets-add"
LOG="/root/mm-datasets-add2.log"
STATUS_DIR="/root/mm-datasets-add-status"
LOCKFILE="/root/mm-datasets-add2.lock"
PY="/root/mm_add2.py"
WORKERS="${WORKERS:-12}"
PASSES="${PASSES:-3}"
FORCE="${FORCE:-0}"

mkdir -p "$BASE" "$STATUS_DIR"

exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "[$(date '+%F %T')] mm-datasets-add2 already running; exiting." >>"$LOG"
  exit 0
fi

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

# priority order: the three known-broken first, then the never-started repos
# small -> large so the directory total climbs fast and risk is back-loaded.
# MVTec-4K + satellite-building are last: already complete, listed only so a
# re-run verifies all 13 repos (status OK -> immediate SKIP).
REPOS=(
  "baizhanquan__FireDetectionDataset-flame-forest-flameye-wildfire|baizhanquan/FireDetectionDataset-flame-forest-flameye-wildfire"
  "hiennguyen9874__fire-smoke-detection|hiennguyen9874/fire-smoke-detection"
  "hf-vision__hardhat|hf-vision/hardhat"
  "keremberke__hard-hat-detection|keremberke/hard-hat-detection"
  "Voxel51__hard-hat-detection|Voxel51/hard-hat-detection"
  "iluvvatar__wood_surface_defects|iluvvatar/wood_surface_defects"
  "SRuibo__Sewer-pipe-defects|SRuibo/Sewer-pipe-defects"
  "kevincluo__structure_wildfire_damage_classification|kevincluo/structure_wildfire_damage_classification"
  "adarshchandrashekar__flame2-rgb-ir|adarshchandrashekar/flame2-rgb-ir"
  "hayden-yuma__roadwork|hayden-yuma/roadwork"
  "fireviewer__fire-smoke-detection-corpus-v1|fireviewer/fire-smoke-detection-corpus-v1"
  "XimiaoZhang__MVTec-4K|XimiaoZhang/MVTec-4K"
  "keremberke__satellite-building-segmentation|keremberke/satellite-building-segmentation"
)

log "################ mm-datasets-add2.sh START pid=$$ workers=$WORKERS passes=$PASSES ################"
log "disk before: $(df -h /root/autodl-tmp | tail -1)"

for pass in $(seq 1 "$PASSES"); do
  log "================ PASS $pass/$PASSES ================"
  remaining=0
  for entry in "${REPOS[@]}"; do
    dir="${entry%%|*}"; repo="${entry#*|}"
    status="$STATUS_DIR/$dir.status"
    if [ "$FORCE" != "1" ] && [ -f "$status" ] && grep -q '^OK' "$status"; then
      log "SKIP $repo (status OK)"
      continue
    fi
    log "=== START $repo -> $BASE/$dir"
    python3 "$PY" --repo "$repo" --dir "$BASE/$dir" --workers "$WORKERS" \
        --status "$status" \
        --manifest "$STATUS_DIR/$dir.tree.json" \
        --verify-json "$STATUS_DIR/$dir.verify.json" \
        --log-file "$LOG"
    rc=$?
    if [ "$rc" != "0" ]; then
      remaining=$((remaining+1))
      log "=== REPO-INCOMPLETE rc=$rc $repo size=$(du -sh "$BASE/$dir" 2>/dev/null | cut -f1)"
    else
      log "=== OK $repo size=$(du -sh "$BASE/$dir" 2>/dev/null | cut -f1)"
    fi
  done
  log "disk after pass $pass: $(df -h /root/autodl-tmp | tail -1)"
  [ "$remaining" = "0" ] && break
  log "PASS $pass finished with $remaining incomplete repo(s); sleeping 60s before retry"
  sleep 60
done

log "================ TOTAL ================"
log "add-dir bytes: $(find "$BASE" -type f -printf '%s\n' | awk '{s+=$1} END{print s}')"
log "add-dir sizes:"; du -sh "$BASE"/* | tee -a "$LOG"
log "################ mm-datasets-add2.sh DONE pid=$$ ################"
