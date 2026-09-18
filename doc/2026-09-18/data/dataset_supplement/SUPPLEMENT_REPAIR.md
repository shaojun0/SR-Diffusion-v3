# SUPPLEMENT_REPAIR — fixing the stalled 130.74 GB supplement download

Written 2026-09-18 (server time) · server `connect.westb.seetacloud.com:38024`
(`autodl-container-jnb93wme4w-bf7ae4cd`) · companion doc: `SUPPLEMENT_DOWNLOAD.md`
(status section there is **appended**, not overwritten) · plan: `supplement_plan.md`,
`manifest_selected.json` (local `datasets/supplement/manifest_selected.json`, server
`/root/autodl-tmp/mm-datasets-add/manifest_selected.json`).

Everything below is **measured**, not estimated. New artefacts:
`repair_tools/mm_add2.py`, `repair_tools/mm-datasets-add2.sh` (copies of
`/root/mm_add2.py`, `/root/mm-datasets-add2.sh` on the server).

---

## 1. Diagnosis of the stall (7.5 h, 62 GB / 130.74 GB, zero net progress)

### 1.1 Two independent faults

| # | fault | evidence |
|---|---|---|
| **A** | The failure path of the old fetcher **truncated the partial and re-downloaded from byte 0**. | `/root/mm-datasets-add.sh:66-74` — the `curl … -C -` resume is followed by a fallback `curl … -o "$part"` **without** `-C -`, whose `log` line reads `RESUME-FAIL rc=18, restarting from 0`. 73 such events in the log. |
| **B** | The 4-shard launch command assigned **8 of the 13 repos to no shard at all**, so they were never even attempted. | zero log lines for those 8 repos; `ONLY=` list vs. `IDX % 4` arithmetic below. |

### 1.2 Fault A — rc=18 and the from-zero retry

`rc=18` is `CURLE_PARTIAL_FILE`: the mirror hands `resolve/main/<path>` to
`cas-bridge.xethub.hf.co` (Xet CAS/S3, HTTP 206 + `Accept-Ranges: bytes`) and that
socket is **closed mid-body**. 124 occurrences in `/root/mm-datasets-add.log`:

```
curl: (18) transfer closed with 325026654 bytes remaining to read
[2026-09-17 21:42:01] RESUME-FAIL rc=18, restarting from 0: …/train-00005-of-00019.parquet
[2026-09-17 21:45:10] FAIL rc=18 …/train-00005-of-00019.parquet
```

* `curl: (18)` count **124**, mean unread remainder **311 MB** (range 14 MB – 699 MB)
* `RESUME-FAIL rc=18/35/143, restarting from 0` **73** times; terminal `FAIL rc=` **73**
* `rc=35` (OpenSSL connect reset to `cas-bridge.xethub.hf.co:443`) 10 ×; `rc=143`
  (SIGTERM from the `timeout` wrapper) 2 ×; **`SIZE-MISMATCH` 0** (sizes were never the issue)

The exact code path (`/root/mm-datasets-add.sh`, `fetch_one()`):

```bash
66:  curl -L … -C - -o "$part" "$url" … || rc=$?          # real resume
69:  if [ "$rc" -ne 0 ]; then
70:    log "RESUME-FAIL rc=$rc, restarting from 0: $url"
71:    curl -L … -o "$part" "$url" … || rc=$?               # ← NO -C -: .part truncated to 0
74:  fi
```

So a transfer that died at 700 MB of a 727 MB file was replaced by a fresh 0 → 727 MB
attempt over a ~16 MB/s link. That second attempt usually died too (same Xet socket
close), producing `REPO-INCOMPLETE` and `FAILED rc=1`. Because the full-file retry is
~45 s+ per attempt and each repo was capped at `ATTEMPTS=3`, the directory total sat
at ~62 GB for the last 7.5 h.

**Side effect worth recording:** four `.part` files were in fact **already complete**
(byte-exact vs. the mirror tree) but never renamed/verified:

| `.part` | on-disk bytes | mirror tree bytes |
|---|---:|---:|
| `hiennguyen9874…/data/train-00005-of-00019.parquet.part` | 898 823 812 | 898 823 812 ✔ |
| `hiennguyen9874…/data/train-00011-of-00019.parquet.part` | 713 861 936 | 713 861 936 ✔ |
| `baizhanquan…/data/train-00012-of-00013.parquet.part` | 727 927 054 | 727 927 054 ✔ |
| `hf-vision__hardhat/data/train-…3e0d95da2811afee.parquet.part` | 199 956 568 | 199 956 568 ✔ |

Only `hiennguyen9874…/data/train-00002-of-00019.parquet.part` was genuinely short
(50 588 025 / 607 526 997). So the "damage" was mainly the wasted re-downloads, not
lost data.

### 1.3 Fault B — 8 repos were never assigned to a shard

The launch command (see `SUPPLEMENT_DOWNLOAD.md` §2) ran 4 shards with
`SHARD=i/4` **and** a hand-written `ONLY="…"` list per shard. The script skips a repo
when `IDX % 4 != SHARD_ID`, so a repo is only processed if it appears in the `ONLY=`
list **of its own arithmetic shard**. The lists did not match:

```
IDX= 0 mod4=0 shard0_ONLY NOT-IN-ONLY   hf-vision__hardhat            (started 14:14 by the pre-shard run)
IDX= 1 mod4=1 ASSIGNED                  keremberke__satellite-…       OK
IDX= 2 mod4=2 NOT-IN-ONLY               keremberke__hard-hat-detection      ← never downloaded
IDX= 3 mod4=3 NOT-IN-ONLY               Voxel51__hard-hat-detection         ← never downloaded
IDX= 4 mod4=0 NOT-IN-ONLY               iluvvatar__wood_surface_defects     ← never downloaded
IDX= 5 mod4=1 NOT-IN-ONLY               SRuibo__Sewer-pipe-defects          ← never downloaded
IDX= 6 mod4=2 NOT-IN-ONLY               kevincluo__structure_wildfire_…     ← never downloaded
IDX= 7 mod4=3 ASSIGNED                  baizhanquan__FireDetection…         OK
IDX= 8 mod4=0 NOT-IN-ONLY               adarshchandrashekar__flame2-rgb-ir  ← never downloaded
IDX= 9 mod4=1 NOT-IN-ONLY               hayden-yuma__roadwork               ← never downloaded
IDX=10 mod4=2 ASSIGNED                  hiennguyen9874__fire-smoke-detection OK
IDX=11 mod4=3 NOT-IN-ONLY               fireviewer__fire-smoke-detection-…  ← never downloaded
IDX=12 mod4=0 ASSIGNED                  XimiaoZhang__MVTec-4K               OK
```

The 8 "NOT-IN-ONLY" rows are exactly the 8 repos with zero log lines and no directory.
Shard 1 (`ONLY` = fireviewer/satellite/hard-hat/Voxel51/hf-vision) actually matched only
`satellite` (IDX 1) and exited `DONE` at 15:46:27 after one repo.

### 1.4 Per-repo state at 2026-09-18 05:20 (before the repair)

Byte sums below are `sum(size of regular files)`, i.e. directory entries excluded
(`du -sb` over-reports by 4 KiB per directory, which is why the earlier
"+4 194 bytes" deltas were not real).

| repo | files on disk | `.part` | bytes on disk | mirror tree files | mirror tree bytes | state |
|---|---:|---:|---:|---:|---:|---|
| XimiaoZhang/MVTec-4K | 6 600 | 0 | 45 690 454 283 | 6 600 | 45 690 454 283 | complete |
| keremberke/satellite-building-segmentation | 11 | 0 | 494 184 417 | 11 | 494 184 417 | complete |
| hiennguyen9874/fire-smoke-detection | 23 | 3 | 11 011 812 251 | 26 | 11 568 747 054 | **−557 MB missing** |
| baizhanquan/FireDetectionDataset-… | 21 | 1 | 8 903 859 559 | 22 | 8 903 855 354 | 1 complete `.part` not promoted |
| hf-vision/hardhat | 3 | 1 | 266 889 737 | 4 | 266 889 536 | 1 complete `.part` not promoted |
| other 8 repos | 0 | 0 | 0 | 1 021 | 63 818 000 000 | **never started** |
| **total** | | | **66 537 731 903 B (66.54 GB)** | | **130 741 822 662 B (130.74 GB)** | |

---

## 2. The improved downloader (true resume, no restart-from-0)

`aria2c` is **not installed** on this host (`which aria2c` empty, `find / -name aria2c`
empty, no conda env has it) and packages must not be installed, so the transfer
engine is `curl`, driven by a Python orchestrator that implements the resume policy
aria2 would have given us. Single-stream throughput was measured at **16–17 MB/s**
and 8 parallel byte-ranges gave the same aggregate (16.3 MB/s) — the link, not the
connection count, is the limit, so parallelism across *files* (12 workers) is used
for stall tolerance rather than multi-connection per file.

`/root/mm_add2.py` (local `repair_tools/mm_add2.py`) — per repo:

1. **Plan** from the mirror's own tree API with `Link:`-header pagination
   (`/api/datasets/<repo>/tree/main`, `limit=1000` + cursor loop), the same
   mirror-native method as `/root/mm_tree_jobs.py`; the snapshot is written to
   `/root/mm-datasets-add-status/<dir>.tree.json`.
2. **Skip/verify** every file whose local size already equals the tree size;
   promote a `.part` whose size equals the tree size **without re-downloading a byte**.
3. **Fetch** the rest with a per-file resume loop:
   `curl -L --fail -C - --connect-timeout 30 --speed-limit 8192 --speed-time 120 -o <file>.part <url>`.
   - every attempt is a fresh `-C -` against the mirror URL (the 302 to the Xet CAS
     bridge is re-followed each time, so the signed URL never expires on us);
   - a failed attempt **keeps** the `.part` and the next attempt continues from the
     new offset — `restarting from 0` no longer exists in any failure path;
   - retries are bounded by **stall detection** (12 consecutive attempts with 0 byte
     growth → FILE-FAILED), not by a 3-attempt cap;
   - the only from-zero path left is a **provably corrupt** `.part` whose size
     *exceeds* the expected size; it is logged as `CORRUPT-OVERSIZE` and applies to
     that single file only.
4. **Verify** the whole repo against the tree listing: exact file count + exact byte
   sum, listing `present_exact / present_wrong / missing / local_bytes vs expected_bytes`;
   redundant `.part` leftovers are deleted only when the final file is exact.
5. **Status/log** per repo (`/root/mm-datasets-add-status/<dir>.status` =
   `RUNNING`/`OK`/`INCOMPLETE` + counts, `<dir>.verify.json`, `<dir>.tree.json`),
   idempotent and re-runnable.

`/root/mm-datasets-add2.sh` (local `repair_tools/mm-datasets-add2.sh`) drives the repos
**single-shard, no `ONLY=`** (the shard-arithmetic bug from §1.3 cannot recur), three
passes, `flock` single-instance. It now lists **all 13** repos: 11 needed work on
2026-09-18, and MVTec-4K + satellite-building were added last so a re-run verifies the
whole set (they were already `OK` and only `SKIP`ped).

### Launch command actually used
```bash
ssh -p 38024 root@connect.westb.seetacloud.com 'cd /root
  setsid nohup /root/mm-datasets-add2.sh >>/root/mm-datasets-add2.console 2>&1 </dev/null &'
```

* wrapper PID **646884** (`/bin/bash /root/mm-datasets-add2.sh`) — ran to completion
  and exited `2026-09-18 06:55:03`; child `python3 /root/mm_add2.py` PIDs were
  646916 (keremberke/hard-hat) … 672341 (SRuibo pass 1) … last one exited 06:55:03.
  **No downloader process is running now** (`pgrep -af 'mm_add2.py|mm-datasets-add2.sh'` empty).
  Re-running the same command resumes/repairs idempotently (status `OK` → SKIP).
* log `/root/mm-datasets-add2.log` (2 123 lines), console duplicate
  `/root/mm-datasets-add2.console`
* started **2026-09-18 05:22:19** server time; detached via `setsid nohup` (survives ssh exit)
* single-instance via `flock /root/mm-datasets-add2.lock`
* the wrapper now lists **all 13 repos** (MVTec-4K and satellite-building added last,
  they are `OK` and skip instantly), so one command re-verifies/re-repairs the whole set
* `/root/mm_add2.py` was patched mid-run to add **global 429 backoff** (all workers pause
  on a rate-limit reply) and to cap workers at 6 for repos with >1 000 files; the patch
  took effect from the kevincluo process onward and again in pass 2 (SRuibo)

---

## 3. Progress after the repair (measured)

Byte sum = `find … -type f ! -name '*.part' -printf '%s\n' | awk` (regular files only).
The run added **64.20 GB in 92 m 44 s** (66.538 → 130.742 GB), mean **11.5 MB/s**,
peak sustained **17.7 MB/s** (single stream; 8 parallel byte-ranges measured the same
16.3 MB/s, so the link — not the connection count — is the ceiling).

| when (server) | bytes on disk | event |
|---|---:|---|
| 05:20 (pre-repair) | 66.538 GB | 5 dirs, 2 FAILED, 1 unfinished, 8 never started |
| 05:22:19 | 66.538 GB | repaired downloader launched (wrapper PID 646884) |
| 05:23:24 | 67.657 GB | keremberke/hard-hat OK — 1.119 GB in 65 s (17.2 MB/s) |
| 05:33:15 | 68.983 GB | Voxel51 OK — 1.326 GB / 5 006 files in 590 s (small-file overhead) |
| 05:35:21 | 71.186 GB | iluvvatar OK — 2.203 GB in 126 s (17.5 MB/s) |
| 05:40 → 06:00 | 73.946 GB | **429-limited**: SRuibo's 3 907 tiny files, only +90 KB in 20 min; 1 560 × `HTTP 429`; finished pass 1 with 6 046 B missing |
| 06:05:10 | 79.027 GB | kevincluo OK — 5.080 GB in 290 s (17.5 MB/s) |
| 06:13:37 | 88.002 GB | adarshchandrashekar OK — 8.975 GB in 507 s (17.7 MB/s) |
| 06:23:32 | 98.525 GB | hayden-yuma OK — 10.524 GB in 595 s (17.7 MB/s) |
| 06:53:47 | 130.742 GB | fireviewer OK — 32.217 GB in 1 815 s |
| 06:55:02 | **130.742 GB** | pass 2: SRuibo OK in 15 s (the 6 KB gap, with 429 backoff) |
| 06:55:03 | | wrapper `DONE`; no `mm_add2.py` process left |
| 06:56:33–06:56:43 | **130.742 GB** | independent `--verify-only` pass over all 13 repos: every one `OK` |

Run counters from `/root/mm-datasets-add2.log` (2 123 lines): `RESUME-STALL` 1 536,
`RATE-LIMITED 429` 1 560, `FILE-FAILED` 20 — all 20 recovered on pass 2.
The 429 window cost ≈ 20 min and is the only reason the run is not ~70 min; the
new downloader handled it by pausing all workers and retrying, **not** by discarding
bytes (note `part_bytes` in the 5-min samples: partials kept growing across samples,
e.g. 06:30 → 06:50 6.6 → 9.3 GB of in-flight `.part` data, and every one of them
landed).

ETA check: projected ~07:00 at launch, actual **06:55:03** (within 5 min).

---

## 4. Per-repo landing verification (all 13, 2026-09-18 06:56 server time)

Method: mirror tree API (native `Link:` cursor pagination, `limit=1000`) as the
authority, then an exact per-file size comparison of every local regular file,
`--verify-only` (no bytes transferred). Machine-readable:
`/root/mm-datasets-add-status/<dir>.verify.json` and `<dir>.tree.json`.

| # | repo | tree files | present exact | wrong | extra | missing | tree bytes | local bytes | `.part` | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | XimiaoZhang/MVTec-4K | 6 600 | 6 600 | 0 | 0 | 0 | 45 690 454 283 | 45 690 454 283 | 0 | **OK** |
| 2 | fireviewer/fire-smoke-detection-corpus-v1 | 59 | 59 | 0 | 0 | 0 | 32 216 812 271 | 32 216 812 271 | 0 | **OK** |
| 3 | hiennguyen9874/fire-smoke-detection | 26 | 26 | 0 | 0 | 0 | 11 568 747 054 | 11 568 747 054 | 0 | **OK** |
| 4 | hayden-yuma/roadwork | 33 | 33 | 0 | 0 | 0 | 10 524 080 260 | 10 524 080 260 | 0 | **OK** |
| 5 | adarshchandrashekar/flame2-rgb-ir | 20 | 20 | 0 | 0 | 0 | 8 974 944 159 | 8 974 944 159 | 0 | **OK** |
| 6 | baizhanquan/FireDetectionDataset-… | 22 | 22 | 0 | 0 | 0 | 8 903 855 354 | 8 903 855 354 | 0 | **OK** |
| 7 | kevincluo/structure_wildfire_damage_classification | 11 | 11 | 0 | 0 | 0 | 5 079 796 793 | 5 079 796 793 | 0 | **OK** |
| 8 | SRuibo/Sewer-pipe-defects | 3 907 | 3 907 | 0 | 0 | 0 | 2 374 137 943 | 2 374 137 943 | 0 | **OK** |
| 9 | iluvvatar/wood_surface_defects | 7 | 7 | 0 | 0 | 0 | 2 202 921 848 | 2 202 921 848 | 0 | **OK** |
| 10 | Voxel51/hard-hat-detection | 5 006 | 5 006 | 0 | 0 | 0 | 1 326 490 167 | 1 326 490 167 | 0 | **OK** |
| 11 | keremberke/hard-hat-detection | 11 | 11 | 0 | 0 | 0 | 1 118 543 356 | 1 118 543 356 | 0 | **OK** |
| 12 | keremberke/satellite-building-segmentation | 11 | 11 | 0 | 0 | 0 | 494 184 417 | 494 184 417 | 0 | **OK** |
| 13 | hf-vision/hardhat | 4 | 4 | 0 | 0 | 0 | 266 889 536 | 266 889 536 | 0 | **OK** |
| | **total** | **15 717** | **15 717** | **0** | **0** | **0** | **130 741 857 441** | **130 741 857 441** | **0** | **13/13 OK** |

Independent cross-check (no JSON involved):
`find /root/autodl-tmp/mm-datasets-add -mindepth 1 -maxdepth 1 -type d -exec find {} -type f -printf '%s\n' \; | awk '{s+=$1} END{print s}'`
→ **130 741 857 441 B = 130.74186 GB**. Grand total including the pre-existing
`manifest_selected.json` (8 686 B) = 130 741 866 127 B. Files: **15 717 + 1**.
`.part` leftovers: **0**.

### Gap vs. the plan

* Frozen plan (`manifest_selected.json`, 2026-09-17): **130 741 822 662 B (130.74 GB)**.
* Live `main` trees measured 2026-09-18: **130 741 857 441 B** → **+34 779 B (+0.00003 %)**.
  The entire delta is `fireviewer/fire-smoke-detection-corpus-v1`, which grew from
  **50 → 59 files** between the two dates (all 9 additions are tiny metadata files);
  the other 12 repos match the plan byte-for-byte.
* **Gap: 0.00 GB.** The 2026-09-17 target of 130.74 GB is met exactly against the
  current mirror revision, and the 120–145 GB acceptance band from the plan is
  satisfied on the nose. No repo is INCOMPLETE, FAILED or BLOCKED.
* Disk after: `/dev/md0` 1.1 T, **782 G used / 269 G free (75 %)**.
* Existing mirror `/root/autodl-tmp/mm-datasets` still **202 G**, untouched.

---

## 5. Constraints honoured

* Only `/root/autodl-tmp/mm-datasets-add/` was written. `/root/autodl-tmp/mm-datasets/`,
  training dirs, translation dirs and `/root/autodl-tmp/models/Qwen3.8-27B` were **not**
  read, modified or deleted.
* **No GPU used**: the downloader is pure network/disk; `nvidia-smi` was only queried,
  never a compute allocation. The other agent's GPU job was live throughout
  (`nvidia-smi --query-compute-apps` → PID **643233**, `python3`, **62 162 MiB**) and was
  never signalled, paused or killed. No other process on the host was touched.
* No packages installed (`aria2c` absent → curl path; `requests` was already present).
* `.part` files deleted only when they were our own and provably redundant
  (final file present at exact mirror size); nothing destructive was forced.
* No `git commit` / `git push`.
