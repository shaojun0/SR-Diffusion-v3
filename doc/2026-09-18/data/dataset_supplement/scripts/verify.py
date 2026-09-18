#!/usr/bin/env python3
"""Verify candidate repos on hf-mirror: gated, size, tree, label files, README, range GET."""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter

EP = "https://hf-mirror.com"
OUT = "/home/linaro/dsh/srdiff_overnight/datasets/supplement/verify_report.jsonl"
PAGE = 1000
MAXPAGES = 400

LABEL_HINT = re.compile(
    r'(^|/)(labels?|ground_truth|gt|masks?|annotations?|segmentation|bbox|captions?|metadata)(/|$)'
    r'|\.(txt|json|jsonl|xml|csv|yaml|yml)$', re.I)
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff', '.gif', '.JPG', '.PNG'}
TXT_EXT = {'.txt', '.json', '.jsonl', '.xml', '.csv', '.yaml', '.yml', '.parquet', '.arrow'}


def fetch(url, timeout=60, tries=5, headers=None, raw=True):
    last = None
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers or {"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(5 * (a + 1))
                last = e
                continue
            return e.code, b"", dict(e.headers or {})
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(3 * (a + 1), 20))
    raise RuntimeError(f"{url}: {last}")


def tree(repo):
    url = f"{EP}/api/datasets/{repo}/tree/main"
    params = {"expand": "false", "recursive": "true", "limit": str(PAGE)}
    files, seen, pages = [], set(), 0
    while True:
        st, body, hd = fetch(url + "?" + urllib.parse.urlencode(params))
        if st in (401, 403):
            raise PermissionError(f"HTTP {st} listing tree")
        if st == 404:
            raise FileNotFoundError("HTTP 404")
        if st != 200:
            raise RuntimeError(f"HTTP {st} listing tree")
        data = json.loads(body)
        for it in data:
            if it.get("type") == "file":
                sz = it.get("size") or (it.get("lfs") or {}).get("size") or 0
                files.append((it["path"], sz))
        pages += 1
        link = hd.get("Link") or hd.get("link") or ""
        nxt = None
        for part in link.split(","):
            if 'rel="next"' in part:
                nxt = part.split(";")[0].strip().strip("<>")
                break
        if not nxt or pages >= MAXPAGES:
            break
        pu = urllib.parse.urlparse(nxt)
        q = dict(urllib.parse.parse_qsl(pu.query))
        if not q.get("cursor") or q["cursor"] in seen:
            break
        seen.add(q["cursor"])
        url, params = EP + pu.path, q
    return files, pages


def verify(repo):
    rec = {"repo": repo, "ts": time.strftime("%F %T")}
    try:
        st, body, _ = fetch(f"{EP}/api/datasets/{urllib.parse.quote(repo)}")
        if st != 200:
            rec.update(ok=False, err=f"info HTTP {st}")
            return rec
        d = json.loads(body)
        rec.update(usedStorage=d.get("usedStorage"), gated=d.get("gated"),
                   private=d.get("private"), disabled=d.get("disabled"),
                   sha=d.get("sha"), lastModified=d.get("lastModified"),
                   siblings=len(d.get("siblings") or []),
                   cardData=d.get("cardData") or {},
                   downloads=d.get("downloads"), likes=d.get("likes"))
    except Exception as e:  # noqa: BLE001
        rec.update(ok=False, err=f"info {type(e).__name__}: {e}"[:200])
        return rec
    try:
        files, pages = tree(repo)
    except Exception as e:  # noqa: BLE001
        rec.update(ok=False, err=f"tree {type(e).__name__}: {e}"[:200])
        return rec
    total = sum(s for _, s in files)
    ext = Counter(os.path.splitext(p)[1].lower() for p, _ in files)
    label_files = [p for p, _ in files if LABEL_HINT.search(p)]
    img_files = [p for p, _ in files if os.path.splitext(p)[1].lower() in IMG_EXT]
    big = sorted(files, key=lambda x: -x[1])[:6]
    rec.update(ok=True, tree_files=len(files), tree_bytes=total, pages=pages,
               ext_top=ext.most_common(12), n_label=len(label_files),
               label_samples=label_files[:8], n_img=len(img_files),
               biggest=[{"p": p, "b": s} for p, s in big])
    # README
    try:
        st, body, _ = fetch(f"{EP}/datasets/{repo}/resolve/main/README.md", tries=2)
        rec["readme_http"] = st
        rec["readme_head"] = body.decode("utf-8", "replace")[:3500] if st == 200 else ""
    except Exception as e:  # noqa: BLE001
        rec["readme_http"] = f"err {e}"[:80]
    # range GET on first data file (skip .gitattributes / README)
    probe = None
    for p, s in files:
        if p.lower() in (".gitattributes", "readme.md"):
            continue
        if s and s > 4096:
            probe = p
            break
    if probe:
        try:
            st, body, hd = fetch(f"{EP}/datasets/{repo}/resolve/main/" + urllib.parse.quote(probe),
                                 tries=2, headers={"User-Agent": "curl/8", "Range": "bytes=0-1023"})
            rec["range_probe"] = {"path": probe, "http": st,
                                  "content_range": hd.get("Content-Range") or hd.get("content-range"),
                                  "got": len(body)}
        except Exception as e:  # noqa: BLE001
            rec["range_probe"] = {"path": probe, "http": f"err {e}"[:80]}
    return rec


def main():
    repos = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
    done = set()
    if os.path.exists(OUT) and "--resume" in sys.argv:
        for l in open(OUT):
            try:
                done.add(json.loads(l)["repo"])
            except Exception:  # noqa: BLE001
                pass
    with open(OUT, "a") as f:
        for r in repos:
            if r in done:
                continue
            rec = verify(r)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[{rec.get('ok')}] {r} bytes={rec.get('tree_bytes')} files={rec.get('tree_files')} "
                  f"nlabel={rec.get('n_label')} nimg={rec.get('n_img')} probe={rec.get('range_probe', {}).get('http')}"
                  f" {rec.get('err','')}", flush=True)
            time.sleep(0.5)


if __name__ == "__main__":
    main()
