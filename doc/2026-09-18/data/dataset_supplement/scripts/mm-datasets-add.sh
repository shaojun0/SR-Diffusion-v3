#!/bin/bash
# mm-datasets-add.sh
# Resumable download of the 13 SUPPLEMENT datasets from hf-mirror.com into
#   /root/autodl-tmp/mm-datasets-add/<org>__<name>
# ONLY ADDS data: never reads/writes /root/autodl-tmp/mm-datasets, construction_site,
# SR-Diffusion-v3 or sr-diffusion-v3-bptt-ksweep.
#
# Method = same verified transport as /root/mm-datasets-dl.sh:
#   huggingface.co is unreachable from this server; `hf download` stalls (Xet) and is
#   slow without it. Direct `curl -L -C -` against
#   https://hf-mirror.com/datasets/<repo>/resolve/main/<path> gives HTTP 206 and real
#   resume. The file plan comes from /root/mm_tree_jobs.py, which does its own
#   Link-header pagination against the mirror (huggingface_hub pagination is broken here
#   for repos with >1000 files).
#
# Resumable: complete files (byte size matches) are skipped, partials continue with
# `curl -C -`. Safe to re-run at any time. Env: JOBS (default 16), ATTEMPTS (3), FORCE=1.

set -u
export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_DISABLE_XET=1

ENDPOINT="$HF_ENDPOINT"
BASE="/root/autodl-tmp/mm-datasets-add"
LOG="/root/mm-datasets-add.log"
STATUS_DIR="/root/mm-datasets-add-status"
JOBS="${JOBS:-16}"
ATTEMPTS="${ATTEMPTS:-3}"
FORCE="${FORCE:-0}"
TREE_HELPER="/root/mm_tree_jobs.py"

mkdir -p "$BASE" "$STATUS_DIR"

# ---- sharding: several workers split the repo list, each with its own lock ----
SHARD="${SHARD:-0/1}"
SHARD_ID="${SHARD%%/*}"
SHARD_N="${SHARD##*/}"
LOCKFILE="/root/mm-datasets-add.${SHARD_ID}.lock"
ONLY="${ONLY:-}"          # optional space-separated dir names to restrict this shard
in_only() { [ -z "$ONLY" ] && return 0; local x; for x in $ONLY; do [ "$x" = "$1" ] && return 0; done; return 1; }

# ---- single instance guard (per shard) -------------------------------------
exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "[$(date '+%F %T')] shard $SHARD already running; exiting." >>"$LOG"
  exit 0
fi

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" >>"$LOG"; }

# ---- one file fetcher (exported for xargs) ---------------------------------
fetch_one() {
  local dest="$1" url="$2" size="${3:-}"
  local part="${dest}.part" cur=0 rc=0
  mkdir -p "$(dirname "$dest")"
  if [ -f "$dest" ]; then
    cur=$(stat -c %s "$dest" 2>/dev/null || echo 0)
    if [ -n "$size" ] && [ "$cur" = "$size" ]; then return 0; fi
    if [ -z "$size" ] && [ "$cur" -gt 0 ]; then return 0; fi
  fi
  if [ -f "$part" ] && [ -n "$size" ] && [ "$(stat -c %s "$part" 2>/dev/null || echo 0)" = "$size" ]; then
    mv -f "$part" "$dest"; return 0
  fi
  log "GET $url (want=${size:-unknown})"
  curl -L --fail --silent --show-error --retry 4 --retry-delay 5 --retry-connrefused \
       --connect-timeout 30 --speed-limit 2048 --speed-time 300 \
       -C - -o "$part" "$url" >>"$LOG" 2>&1 || rc=$?
  if [ "$rc" -ne 0 ]; then
    log "RESUME-FAIL rc=$rc, restarting from 0: $url"
    curl -L --fail --silent --show-error --retry 4 --retry-delay 5 --retry-connrefused \
         --connect-timeout 30 --speed-limit 2048 --speed-time 300 \
         -o "$part" "$url" >>"$LOG" 2>&1 || rc=$?
  fi
  if [ "$rc" -ne 0 ]; then
    log "FAIL rc=$rc $url"
    printf '%s\n' "$url" >>"$FAILFILE"
    return 1
  fi
  if [ -n "$size" ]; then
    local got; got=$(stat -c %s "$part" 2>/dev/null || echo 0)
    if [ "$got" != "$size" ]; then
      log "SIZE-MISMATCH $url got=$got want=$size"
      printf '%s\n' "$url" >>"$FAILFILE"
      return 1
    fi
  fi
  mv -f "$part" "$dest"
  return 0
}
export -f fetch_one log
export LOG FAILFILE

# ---- one repo --------------------------------------------------------------
download_repo() {
  local dir="$1" repo="$2" dest="$BASE/$dir" jobs err
  jobs="$(mktemp /tmp/mmaddjobs.XXXXXX)"; err="$(mktemp /tmp/mmadderr.XXXXXX)"
  local prc=0
  python3 "$TREE_HELPER" "$repo" "$dest" >"$jobs" 2>"$err" || prc=$?
  if [ "$prc" -ne 0 ]; then
    ERRMSG="$(tr '\n' ' ' <"$err")"
    LIST_RC="$prc"
    log "LISTING-FAIL($prc) $repo: $ERRMSG"
    rm -f "$jobs" "$err"
    return 1
  fi
  local files bytes
  files=$(grep -o 'FILES=[0-9]*' "$err" | cut -d= -f2)
  bytes=$(grep -o 'BYTES=[0-9]*' "$err" | cut -d= -f2)
  log "PLAN $repo files=${files:-?} bytes=${bytes:-?}"
  export FAILFILE="$STATUS_DIR/$dir.fail"
  rm -f "$FAILFILE"
  xargs -0 -P "$JOBS" -n 3 bash -c 'fetch_one "$@"' _ <"$jobs"
  rm -f "$jobs" "$err"
  if [ -s "$FAILFILE" ]; then
    local nfail probe code
    nfail=$(wc -l <"$FAILFILE")
    probe=$(head -1 "$FAILFILE")
    code=$(curl -sL -o /dev/null -w '%{http_code}' --max-time 30 "$probe" 2>/dev/null || echo 000)
    log "REPO-INCOMPLETE $repo failures=$nfail probe_http=$code"
    if [ "$code" = "401" ] || [ "$code" = "403" ]; then
      LIST_RC=3
      ERRMSG="HTTP $code on file download (gated repo / HF token required): $probe"
      return 1
    fi
    return 1
  fi
  log "REPO-DONE $repo files=${files:-?} bytes=${bytes:-?}"
  return 0
}

# ---- selected repos: small -> large ---------------------------------------
REPOS=(
  "hf-vision__hardhat|hf-vision/hardhat"
  "keremberke__satellite-building-segmentation|keremberke/satellite-building-segmentation"
  "keremberke__hard-hat-detection|keremberke/hard-hat-detection"
  "Voxel51__hard-hat-detection|Voxel51/hard-hat-detection"
  "iluvvatar__wood_surface_defects|iluvvatar/wood_surface_defects"
  "SRuibo__Sewer-pipe-defects|SRuibo/Sewer-pipe-defects"
  "kevincluo__structure_wildfire_damage_classification|kevincluo/structure_wildfire_damage_classification"
  "baizhanquan__FireDetectionDataset-flame-forest-flameye-wildfire|baizhanquan/FireDetectionDataset-flame-forest-flameye-wildfire"
  "adarshchandrashekar__flame2-rgb-ir|adarshchandrashekar/flame2-rgb-ir"
  "hayden-yuma__roadwork|hayden-yuma/roadwork"
  "hiennguyen9874__fire-smoke-detection|hiennguyen9874/fire-smoke-detection"
  "fireviewer__fire-smoke-detection-corpus-v1|fireviewer/fire-smoke-detection-corpus-v1"
  "XimiaoZhang__MVTec-4K|XimiaoZhang/MVTec-4K"
)

log "################ mm-datasets-add.sh START pid=$$ shard=$SHARD jobs=$JOBS attempts=$ATTEMPTS ################"
log "disk before: $(df -h /root/autodl-tmp | tail -1)"

IDX=-1
for entry in "${REPOS[@]}"; do
  IDX=$((IDX+1))
  if [ "$SHARD_N" -gt 1 ] && [ $((IDX % SHARD_N)) -ne "$SHARD_ID" ]; then continue; fi
  in_only "${entry%%|*}" || continue
  dir="${entry%%|*}"
  repo="${entry#*|}"
  dest="$BASE/$dir"
  status="$STATUS_DIR/$dir.status"

  if [ "$FORCE" != "1" ] && [ -f "$status" ] && grep -q '^OK' "$status"; then
    log "SKIP $repo (status OK)"
    continue
  fi

  log "=== START $repo -> $dest"
  printf 'START %s\n' "$(date '+%F %T')" >"$status"

  ok=0
  for attempt in $(seq 1 "$ATTEMPTS"); do
    log "ATTEMPT $attempt/$ATTEMPTS $repo"
    LIST_RC=""
    if download_repo "$dir" "$repo"; then ok=1; else rc=$?; fi
    [ "$ok" = "1" ] && break

    if [ -n "${LIST_RC:-}" ]; then
      case "$LIST_RC" in
        3) printf 'BLOCKED gated/unauthorized (%s) %s\n' "$ERRMSG" "$(date '+%F %T')" >"$status"
           log "BLOCKED $repo (gated/unauthorized): $ERRMSG"; ok=2; break ;;
        4) printf 'BLOCKED not-found on mirror (%s) %s\n' "$ERRMSG" "$(date '+%F %T')" >"$status"
           log "BLOCKED $repo (not found on mirror): $ERRMSG"; ok=2; break ;;
      esac
    fi
    sleep 15
  done

  if [ "$ok" = "1" ]; then
    printf 'OK %s\n' "$(date '+%F %T')" >"$status"
    log "=== OK $repo size=$(du -sh "$dest" 2>/dev/null | cut -f1)"
  elif [ "$ok" = "2" ]; then
    log "=== BLOCKED $repo"
  else
    printf 'FAILED rc=%s %s\n' "${rc:-1}" "$(date '+%F %T')" >"$status"
    log "=== FAILED rc=${rc:-1} $repo"
  fi
done

log "disk after: $(df -h /root/autodl-tmp | tail -1)"
log "################ mm-datasets-add.sh DONE pid=$$ ################"
