# SUPPLEMENT_DOWNLOAD — adding ≈131 GB of in-domain multimodal data

Written 2026-09-17 14:32 (server time) · server `connect.westb.seetacloud.com:38024`
(autodl-container-jnb93wme4w-bf7ae4cd) · status: **RUNNING**

## 1. What was selected (13 repos, 130.74 GB measured)

Full detail, candidate table, per-criterion checks and the exclusion list:
`datasets/supplement_plan.md`. Machine-readable:
`/root/autodl-tmp/mm-datasets-add/manifest_selected.json` (server) and
`datasets/supplement/manifest_selected.json` (local, identical).

| repo | GB | task |
|---|---:|---|
| XimiaoZhang/MVTec-4K | 45.69 | industrial anomaly / defect segmentation (4K, masks) |
| fireviewer/fire-smoke-detection-corpus-v1 | 32.22 | fire/smoke object detection (curated multi-source) |
| hiennguyen9874/fire-smoke-detection | 11.57 | fire/smoke object detection (72 159 + 18 112) |
| hayden-yuma/roadwork | 10.52 | roadwork/work-zone scenes, English captions + tags |
| adarshchandrashekar/flame2-rgb-ir | 8.97 | flame classification, RGB + IR (53 451) |
| baizhanquan/FireDetectionDataset-flame-forest-flameye-wildfire | 8.90 | wildfire fire/smoke detection (15 615) |
| kevincluo/structure_wildfire_damage_classification | 5.08 | structural damage classification (6 classes) |
| SRuibo/Sewer-pipe-defects | 2.37 | sewer pipe defect detection (1 953 YOLO labels) |
| iluvvatar/wood_surface_defects | 2.20 | wood surface defect detection (20 276) |
| Voxel51/hard-hat-detection | 1.33 | PPE/hard-hat detection (5 000 VOC) |
| keremberke/hard-hat-detection | 1.12 | hard-hat detection (Roboflow) |
| keremberke/satellite-building-segmentation | 0.49 | building segmentation |
| hf-vision/hardhat | 0.27 | hard-hat detection (7 063) |
| **total** | **130.74** | kept corpus becomes 8.109 + 130.74 = **138.85 GB** |

All 13: mirror `gated=false`, tree listing 200, byte-range probe 206, images + real supervision
(boxes / masks / labels / captions), no conversation field, English or no text.

## 2. Target dir, methods, and deliverables on the server

* Destination (new, independent of the DROPped 202 GB mirror):
  **`/root/autodl-tmp/mm-datasets-add/<org>__<name>/`**
* Transport: direct `curl -L -C -` against
  `https://hf-mirror.com/datasets/<repo>/resolve/main/<path>` (HTTP 206, redirect stays on the mirror).
  `hf download` is *not* used — it stalls on Xet (needs huggingface.co) and is ~0.17 MB/s without it.
* File plan: `/root/mm_tree_jobs.py` (mirror-native `Link:` pagination; `huggingface_hub` pagination
  breaks for >1000-file repos because the mirror proxies the `huggingface.co` next-URL verbatim).
* Downloader: **`/root/mm-datasets-add.sh`** (local copy `datasets/supplement/mm-datasets-add.sh`)
  * per-file resume (`curl -C -`), complete files skipped by byte-size match, size verification
  * 3 attempts per repo, one failing repo never aborts the others
  * per-repo status `/root/mm-datasets-add-status/<dir>.status` (`START` / `OK` / `FAILED rc=…` / `BLOCKED …`)
  * log `/root/mm-datasets-add.log`; per-shard stdout `/root/mm-add-shard{0..3}.out`
  * `setsid nohup … < /dev/null &` → detached from ssh; `flock` single-instance per shard
  * added `SHARD=i/n` + `ONLY="dir …"` so several workers split the repo list (a single repo with few
    files cannot fill 16 streams; 4 shards keep ~27 curl streams busy)
* `.part` files are left in place on restart and resumed.

### Exact launch command (reproducible)
```bash
ssh -p 38024 root@connect.westb.seetacloud.com 'cd /root
SHARD=0/4 JOBS=10 ONLY="XimiaoZhang__MVTec-4K" setsid nohup /root/mm-datasets-add.sh > /root/mm-add-shard0.out 2>&1 < /dev/null &
SHARD=1/4 JOBS=6  ONLY="fireviewer__fire-smoke-detection-corpus-v1 keremberke__satellite-building-segmentation keremberke__hard-hat-detection Voxel51__hard-hat-detection hf-vision__hardhat" setsid nohup /root/mm-datasets-add.sh > /root/mm-add-shard1.out 2>&1 < /dev/null &
SHARD=2/4 JOBS=6  ONLY="hiennguyen9874__fire-smoke-detection hayden-yuma__roadwork adarshchandrashekar__flame2-rgb-ir SRuibo__Sewer-pipe-defects iluvvatar__wood_surface_defects" setsid nohup /root/mm-datasets-add.sh > /root/mm-add-shard2.out 2>&1 < /dev/null &
SHARD=3/4 JOBS=6  ONLY="baizhanquan__FireDetectionDataset-flame-forest-flameye-wildfire kevincluo__structure_wildfire_damage_classification" setsid nohup /root/mm-datasets-add.sh > /root/mm-add-shard3.out 2>&1 < /dev/null &'
```
Re-run at any time to resume/finish (it skips `OK` repos); add new repos to `REPOS=()` in the script.

## 3. Live progress (measured)

| when (server) | bytes on disk | note |
|---|---:|---|
| 14:14:42 | 0 | first single-shard launch |
| 14:20:47 | ≈0.19 GB | restarted as 4 shards (single repo could not saturate the link) |
| 14:28:33 → 14:29:33 | 1.720 → 1.960 GB | **4.0 MB/s instantaneous** (60 s sample) |
| 14:29:38 | **2.3 GB** | 5 repo dirs in progress, 27 curl streams |

* Worker PIDs (detached, 4 shards): **477854, 477855, 477856, 477857** (parent `bash /root/mm-datasets-add.sh`)
* Status files: all 5 started repos `START`, **0 FAILED, 0 BLOCKED, 0 OK yet**
* Per repo: MVTec-4K 821 M · hiennguyen9874 639 M · baizhanquan 448 M · satellite-building 224 M · hf-vision 192 M
* Disk: `/dev/md0` 1.1 T, **482 G free** (57 % used) — 130.7 GB fits with ~350 GB headroom
* Existing mirror untouched: `/root/autodl-tmp/mm-datasets` = **202 G** (unchanged)

**ETA:** 128.4 GB remaining at the measured 4.0 MB/s ≈ **8.9 h** → completion ≈
**2026-09-17 23:30 server time** (realistically 9–12 h, mirror speed is variable → 23:30 ± 1.5 h).

### Verify command
```bash
ssh -p 38024 root@connect.westb.seetacloud.com 'du -sh /root/autodl-tmp/mm-datasets-add; \
  du -sh /root/autodl-tmp/mm-datasets-add/*/; \
  cat /root/mm-datasets-add-status/*.status; tail -20 /root/mm-datasets-add.log; \
  pgrep -af "bash /root/mm-datasets-add.sh"'
```

## 4. Failed / blocked / excluded

* **FAILED: none. BLOCKED: none.** Every selected repo listed and returned HTTP 206 on the range probe.
* Excluded despite looking promising (details in `supplement_plan.md`):
  * `Piyush152/ppe-detection` + its duplicate `qxnam/ppe-detection` — 4 978 PNG with **0 label files**
    and a **403** range probe (fails criteria 2 and 5).
  * `toth235a/concrete_crack_combined` — 14 112 images in `Public_all.zip`, **no annotation entries**.
  * `mountainmoon2000/wildfire_mm` (33.5 GB) — labels verified inside the zips, but its FASDD content is
    already inside the selected `fireviewer` corpus → dropped as duplicate.
  * `htfhgf/flame_test`, `MahedixHasan/pose-guided-fall-detection-icta2026`, `romainpuech/wildfire-drone-routing-data`,
    `CypressLI/3Dlarge_construction`, `fucthin/fire_smoke` — unverifiable labels / video / rasters / 100 GB+ single point.
  * MVTec-AD duplicates (`Voxel51/`, `TheoM55/`, `foersben/`, …) and `Hajorda/flameye-wildfire-detection`
    (byte-identical to the selected `baizhanquan/…`) — excluded to avoid double-counting.
* No HF token exists on the server; nothing selected needs one.

## 5. Constraints honoured

* Only additions: nothing under `/root/autodl-tmp/mm-datasets/`, `construction_site`,
  `construction_site_original_en`, `SR-Diffusion-v3`, `sr-diffusion-v3-bptt-ksweep` was read, moved or deleted.
* A live training smoke run was present during the whole operation and was **not** touched:
  `train_v2.py … --output_dir /root/autodl-tmp/gnnA_off_K35_2000` (and the earlier `gnnA_smoke_sum`, 30 min
  `timeout`) on GPU 1. The downloader is network/disk-only, no GPU use.
* No packages installed on the server (only the pre-existing `requests`/`curl`).
* All sizes are measured (mirror tree sums of the current `main` revision + HTTP range probes), not estimated.
  Note `usedStorage` from `/api/datasets/<id>` over-reports (all revisions counted): e.g. `fireviewer`
  78.10 GB reported vs **32.22 GB** actually downloaded.

---

# ADDENDUM 2026-09-18 06:56 — repair complete, status: **DONE (13/13 OK, 130.74 GB)**

The status block above (§3, 2026-09-17) is left as written. This addendum records what
happened afterwards; full detail in **`SUPPLEMENT_REPAIR.md`**.

**Outcome: the supplement set is complete and byte-verified.** 130 741 857 441 B
(130.74186 GB) in `/root/autodl-tmp/mm-datasets-add/`, 15 717 files, 13/13 repos `OK`,
0 INCOMPLETE / 0 FAILED / 0 BLOCKED, 0 `.part` leftovers. Disk 782 G used / 269 G free.
The 2026-09-17 §3 ETA of ≈23:30 ± 1.5 h was **not** met — the job had in fact stalled.

### Why §3's status never advanced past 62 GB

Two independent faults (measured, both in `/root/mm-datasets-add.sh`):

1. **`RESUME-FAIL rc=18, restarting from 0` (script lines 66–74).** `rc=18` is
   `CURLE_PARTIAL_FILE` — the Xet CAS bridge (`cas-bridge.xethub.hf.co`) closes the
   socket mid-body. 124 × `curl: (18)` (mean 311 MB unread, up to 699 MB), 73 ×
   `RESUME-FAIL`, 73 × terminal `FAIL`. The fallback curl **omitted `-C -`**, so every
   mid-transfer close truncated the `.part` and re-downloaded the whole 300–900 MB file
   from byte 0 — 7.5 h with no net growth. Four of the five `.part` files were in fact
   byte-exact complete and only needed renaming.
2. **8 of 13 repos were assigned to no shard.** In the §2 launch command, a repo is
   processed only if it appears in the `ONLY=` list of its own `SHARD=i/4` arithmetic
   shard (`IDX % 4`). The hand-written lists did not line up: fireviewer, keremberke
   hard-hat, Voxel51, iluvvatar, SRuibo, kevincluo, adarshchandrashekar and hayden-yuma
   matched no shard and have **zero log lines** — they were never attempted.

### Fix and relaunch (2026-09-18 05:22:19)

* New true-resume downloader `/root/mm_add2.py` (local `datasets/repair_tools/mm_add2.py`)
  driven by `/root/mm-datasets-add2.sh` (local `datasets/repair_tools/mm-datasets-add2.sh`).
  `aria2c` is not installed and nothing may be installed, so the transfer engine is curl:
  **every** attempt is `curl -L --fail -C - … -o <file>.part <url>`; a failed attempt keeps
  the partial and the next continues from the new offset (the 302 to the CAS bridge is
  re-followed each time). Exact-size partials are promoted without re-downloading. Retries
  are bounded by stall detection (12 attempts with 0 growth), not by a 3-attempt cap; the
  only from-zero path left is a `.part` *larger* than the tree size (`CORRUPT-OVERSIZE`).
  HTTP 429 from the mirror triggers a global exponential backoff across all workers.
  Every repo is verified against the mirror tree listing (file count + exact byte sum) and
  writes `<dir>.status` / `<dir>.tree.json` / `<dir>.verify.json`; the job is single-pass
  corrected by a second pass and is fully idempotent.
* Launch: `setsid nohup /root/mm-datasets-add2.sh >>/root/mm-datasets-add2.console 2>&1 </dev/null &`
  — wrapper PID **646884**, log `/root/mm-datasets-add2.log` (2 123 lines), detached and
  re-runnable; it now lists all 13 repos.
* Speed: 64.20 GB added in **92 m 44 s** (mean 11.5 MB/s, peak 17.7 MB/s). One 429-limited
  window (SRuibo, 3 907 small files) cost ≈20 min and left a 6 046 B gap, closed by pass 2.
* Final verification pass (`--verify-only`, 06:56:33–06:56:43) on all 13 repos: tree bytes
  == local bytes exactly, 0 wrong / 0 extra / 0 missing. Total **130 741 857 441 B**, i.e.
  **+34 779 B vs. the frozen plan** — the whole delta is `fireviewer`, which grew 50 → 59
  files on the mirror between the two dates. **Gap to target: 0.00 GB.**
* Constraints unchanged and honoured: only `/root/autodl-tmp/mm-datasets-add/` written;
  `/root/autodl-tmp/mm-datasets` still 202 G; the live GPU job (PID 643233, 62 162 MiB)
  was never touched; no packages installed; no commit/push.
