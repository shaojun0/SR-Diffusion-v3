#!/usr/bin/env python3
"""True-resume downloader for hf-mirror.com dataset repos (supplement set).

Replaces fetch_one() in /root/mm-datasets-add.sh, whose network-failure path was

    RESUME-FAIL rc=18, restarting from 0: <url>
    curl ... -o "$part" "$url"          # <-- no -C -, truncates the partial

i.e. every rc=18 (CURLE_PARTIAL_FILE, the Xet CAS bridge closing the socket
mid-transfer) threw away all bytes already received and re-downloaded from byte 0.
With ~700MB parquet files and a ~16 MB/s capped pipe that never converged.

Guarantees implemented here
  * a partial is NEVER truncated on network failure: every attempt is `curl -C -`
    against the freshly-renamed mirror URL, so the byte offset keeps growing
  * a `.part` whose size equals the mirror tree size is promoted to the final
    name without re-downloading a single byte
  * per-file retry loop with stall detection (no fixed "3 attempts" cap)
  * exact-size verification of every file against the mirror tree listing
  * per-repo manifest snapshot + status/verify JSON; idempotent, re-runnable
  * the only restart-from-zero path is a provably corrupt `.part` (size > expected)

Usage:
  mm_add2.py --repo <org/name> --dir <dest> [--workers N]
             [--status FILE] [--manifest FILE] [--verify-json FILE]
             [--verify-only] [--jobs-out FILE]
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

ENDPOINT = "https://hf-mirror.com"
PAGE_LIMIT = 1000
MAX_PAGES = 200000
TIMEOUT = 90
TREE_RETRIES = 5
CURL = "/usr/bin/curl"
MAX_ATTEMPTS = 200          # outer loop; each attempt is a fresh `curl -C -`
MAX_STALLS = 12             # consecutive attempts with zero byte growth -> give up
STALL_SLEEP = 8

_LOCK = threading.Lock()
LOG_FH = None
_COOLDOWN = {"until": 0.0}


def log(msg):
    line = "[%s] %s" % (time.strftime("%F %T"), msg)
    with _LOCK:
        print(line, flush=True)
        if LOG_FH:
            LOG_FH.write(line + "\n")
            LOG_FH.flush()


def global_pause(seconds):
    """hf-mirror returns 429 when many small files are requested at once; when any
    worker sees one, every worker backs off together."""
    with _LOCK:
        _COOLDOWN["until"] = max(_COOLDOWN["until"], time.time() + seconds)


def wait_cooldown():
    while True:
        with _LOCK:
            rem = _COOLDOWN["until"] - time.time()
        if rem <= 0:
            return
        time.sleep(min(rem, 5))


# --------------------------------------------------------------------------- #
# mirror-native tree listing (Link-header pagination; huggingface.co unreachable)
# --------------------------------------------------------------------------- #
def tree_files(repo):
    """Return (files, total_bytes); files = [{path,size,url,oid}]."""
    url = f"{ENDPOINT}/api/datasets/{repo}/tree/main"
    params = {"expand": "false", "recursive": "true", "limit": str(PAGE_LIMIT)}
    seen = set()
    files = []
    total = 0
    for _ in range(MAX_PAGES):
        resp = None
        last_err = None
        for attempt in range(TREE_RETRIES):
            try:
                resp = requests.get(url, params=params, timeout=TIMEOUT)
                break
            except requests.RequestException as e:
                last_err = e
                time.sleep(min(2 ** attempt, 20))
        if resp is None:
            raise RuntimeError(f"tree network error after {TREE_RETRIES} tries: {last_err}")
        if resp.status_code in (401, 403):
            raise PermissionError(f"HTTP {resp.status_code} listing tree (gated/unauthorized)")
        if resp.status_code == 404:
            raise FileNotFoundError("HTTP 404 listing tree (repo not found on mirror)")
        if resp.status_code == 429:
            time.sleep(10)
            continue
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            raise RuntimeError(f"unexpected tree payload: {str(data)[:200]}")
        for it in data:
            if it.get("type") != "file":
                continue
            path = it.get("path")
            if not path:
                continue
            size = it.get("size")
            lfs = it.get("lfs") or {}
            if lfs.get("size"):
                size = lfs["size"]
            try:
                size = int(size)
            except (TypeError, ValueError):
                size = None
            files.append({
                "path": path,
                "size": size,
                "url": f"{ENDPOINT}/datasets/{repo}/resolve/main/" + urllib.parse.quote(path),
                "oid": lfs.get("oid") or it.get("oid"),
            })
            total += size or 0
        link = resp.headers.get("Link", "")
        nxt = None
        for part in link.split(","):
            if 'rel="next"' in part:
                nxt = part.split(";")[0].strip().strip("<>").strip()
                break
        if not nxt:
            return files, total
        pu = urllib.parse.urlparse(nxt)
        q = dict(urllib.parse.parse_qsl(pu.query))
        cur = q.get("cursor", "")
        if not cur or cur in seen:
            return files, total
        seen.add(cur)
        url = ENDPOINT + pu.path
        params = q
    return files, total


# --------------------------------------------------------------------------- #
# transfer primitive: always resume, never restart
# --------------------------------------------------------------------------- #
def curl_resume(url, part):
    """One resumable fetch attempt. Returns (rc, stderr_tail)."""
    cmd = [CURL, "-L", "--fail", "--silent", "--show-error",
           "--connect-timeout", "30",
           "--speed-limit", "8192", "--speed-time", "120",
           "-C", "-", "-o", part, url]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    except subprocess.TimeoutExpired:
        return 28, "local timeout 7200s"
    return p.returncode, (p.stderr or "").strip().splitlines()[-1][:300] if (p.stderr or "").strip() else ""


def fetch_file(spec):
    dest, url, size = spec["dest"], spec["url"], spec["size"]
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if os.path.exists(dest):
        got = os.path.getsize(dest)
        if size is None or got == size:
            return {"path": dest, "status": "already", "bytes": got, "want": size}
        log(f"CORRUPT-FINAL size={got} want={size} removing {dest}")
        os.remove(dest)

    if not os.path.exists(part):
        open(part, "wb").close()

    cur = os.path.getsize(part)
    if size is not None and cur == size:
        os.replace(part, dest)
        return {"path": dest, "status": "promoted", "bytes": cur, "want": size}

    attempts = 0
    stalls = 0
    rate_hits = 0
    last_err = ""
    while attempts < MAX_ATTEMPTS and stalls < MAX_STALLS:
        wait_cooldown()
        attempts += 1
        cur = os.path.getsize(part)
        if size is not None and cur == size:
            break
        if size is not None and cur > size:
            log(f"CORRUPT-OVERSIZE {dest} got={cur} want={size} -> restart-from-0 (this file only)")
            os.remove(part)
            open(part, "wb").close()
            cur = 0
        rc, err = curl_resume(url, part)
        new = os.path.getsize(part)
        if rc == 0 and (size is None or new == size):
            last_err = ""
            break
        last_err = f"rc={rc} {err} got={new}/{size}"
        if rc == 22 and ("429" in err or "Too Many Requests" in err):
            rate_hits += 1
            wait = min(20 * (2 ** (rate_hits - 1)), 300)
            log(f"RATE-LIMITED 429 (hit {rate_hits}) sleeping {wait}s "
                f"{os.path.basename(dest)}")
            global_pause(wait)
            if rate_hits >= 25:
                break
            continue
        if new > cur:
            stalls = 0
            log(f"RESUME-OK +{new - cur}B ({new}/{size}) [{attempts}] {os.path.basename(dest)}")
        else:
            stalls += 1
            log(f"RESUME-STALL ({stalls}/{MAX_STALLS}) rc={rc} {err} at {new}/{size} "
                f"{os.path.basename(dest)}")
            time.sleep(STALL_SLEEP)

    cur = os.path.getsize(part)
    if size is None or cur == size:
        os.replace(part, dest)
        return {"path": dest, "status": "ok", "bytes": cur, "want": size,
                "attempts": attempts}
    return {"path": dest, "status": "failed", "bytes": cur, "want": size,
            "attempts": attempts, "error": last_err}


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def verify(dest_root, files, cleanup_partials=True):
    expected = {f["path"]: f["size"] for f in files}
    seen = set()
    present_ok = present_wrong = present_extra = 0
    local_bytes = 0
    wrong = []
    partials = []
    for dirpath, _dirnames, filenames in os.walk(dest_root):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, dest_root)
            if fn.endswith(".part"):
                partials.append({"path": rel, "bytes": os.path.getsize(full)})
                continue
            seen.add(rel)
            sz = os.path.getsize(full)
            local_bytes += sz
            exp = expected.get(rel)
            if exp is None:
                present_extra += 1
                wrong.append({"path": rel, "got": sz, "want": None, "why": "extra"})
            elif sz == exp:
                present_ok += 1
            else:
                present_wrong += 1
                wrong.append({"path": rel, "got": sz, "want": exp, "why": "size"})

    # a leftover .part is pure waste once its final file is present and exact
    kept_partials = []
    for p in partials:
        final = os.path.join(dest_root, p["path"][:-len(".part")])
        if (cleanup_partials and os.path.exists(final)
                and os.path.getsize(final) == expected.get(p["path"][:-len(".part")])):
            os.remove(os.path.join(dest_root, p["path"]))
            log(f"CLEANUP removed redundant {p['path']} (final file complete)")
        else:
            kept_partials.append(p)

    missing = [{"path": p, "size": s} for p, s in expected.items() if p not in seen]
    return {
        "expected_files": len(expected),
        "expected_bytes": sum(v for v in expected.values() if v),
        "present_exact": present_ok,
        "present_wrong": present_wrong,
        "present_extra": present_extra,
        "missing_files": len(missing),
        "missing_bytes": sum(m["size"] or 0 for m in missing),
        "local_bytes": local_bytes,
        "partials": kept_partials,
        "missing": missing[:50],
        "wrong": wrong[:50],
    }


# --------------------------------------------------------------------------- #
def main():
    global LOG_FH
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--status")
    ap.add_argument("--manifest")
    ap.add_argument("--verify-json")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--log-file")
    a = ap.parse_args()

    if a.log_file:
        LOG_FH = open(a.log_file, "a")

    dest_root = a.dir
    os.makedirs(dest_root, exist_ok=True)

    if a.status:
        with open(a.status, "w") as fh:
            fh.write("RUNNING %s\n" % time.strftime("%F %T"))

    t0 = time.time()
    log(f"=== LIST {a.repo} (mirror pagination)")
    files, total = tree_files(a.repo)
    log(f"=== PLAN {a.repo} files={len(files)} bytes={total}")

    # hf-mirror starts answering 429 above roughly 8 requests/s; repos made of
    # thousands of small files need fewer parallel connection setups.
    if len(files) > 1000 and a.workers > 6:
        log(f"=== RATE-CAP workers {a.workers} -> 6 ({len(files)} files, mirror 429-limited)")
        a.workers = 6

    if a.manifest:
        with open(a.manifest, "w") as fh:
            json.dump({"repo": a.repo, "generated": time.strftime("%F %T"),
                       "files": len(files), "bytes": total, "tree": files}, fh)

    if not a.verify_only:
        todo = []
        for f in files:
            dest = os.path.join(dest_root, f["path"])
            if os.path.exists(dest) and f["size"] is not None and os.path.getsize(dest) == f["size"]:
                continue
            todo.append({"dest": dest, "url": f["url"], "size": f["size"]})
        done_bytes = sum(f["size"] or 0 for f in files) - sum(t["size"] or 0 for t in todo)
        log(f"=== FETCH {a.repo} to_download_files={len(todo)} "
            f"to_download_bytes={sum(t['size'] or 0 for t in todo)} already_bytes={done_bytes} "
            f"workers={a.workers}")

        results = []
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(fetch_file, t): t for t in todo}
            n = 0
            for fu in as_completed(futs):
                r = fu.result()
                results.append(r)
                n += 1
                if n % 25 == 0 or r["status"] == "failed" or n == len(todo):
                    okb = sum(x["bytes"] for x in results if x["status"] in ("ok", "promoted", "already"))
                    log(f"PROGRESS {a.repo} {n}/{len(todo)} files "
                        f"failed={sum(1 for x in results if x['status'] == 'failed')} "
                        f"confirmed_bytes={okb}")
        failed = [r for r in results if r["status"] == "failed"]
        for r in failed[:20]:
            log(f"FILE-FAILED {r['path']} got={r['bytes']} want={r['want']} {r.get('error','')}")
    else:
        failed = []

    v = verify(dest_root, files)
    v["repo"] = a.repo
    v["dir"] = dest_root
    if a.verify_json:
        with open(a.verify_json, "w") as fh:
            json.dump(v, fh, indent=1)
    log(f"=== VERIFY {a.repo} expected_files={v['expected_files']} present_exact={v['present_exact']} "
        f"wrong={v['present_wrong']} extra={v['present_extra']} missing={v['missing_files']} "
        f"expected_bytes={v['expected_bytes']} local_bytes={v['local_bytes']} "
        f"partials={len(v['partials'])}")

    complete = (v["missing_files"] == 0 and v["present_wrong"] == 0
                and v["local_bytes"] >= v["expected_bytes"] and not failed)
    elapsed = time.time() - t0
    if a.status:
        with open(a.status, "w") as fh:
            if complete:
                fh.write("OK %s\n" % time.strftime("%F %T"))
                fh.write("files=%d bytes=%d\n" % (v["present_exact"], v["local_bytes"]))
            else:
                fh.write("INCOMPLETE %s\n" % time.strftime("%F %T"))
                fh.write("present_exact=%d wrong=%d missing=%d local_bytes=%d expected_bytes=%d\n"
                         % (v["present_exact"], v["present_wrong"], v["missing_files"],
                            v["local_bytes"], v["expected_bytes"]))
    log(f"=== {'OK' if complete else 'INCOMPLETE'} {a.repo} elapsed={elapsed:.0f}s "
        f"local_bytes={v['local_bytes']}/{v['expected_bytes']}")
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
