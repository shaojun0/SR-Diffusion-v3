#!/usr/bin/env python3
"""Server-side (hf-mirror) verification of candidate HF dataset repos.

Mirror-native tree listing (own Link pagination, like mm_tree_jobs.py).
Writes JSONL incrementally: /root/mm-verify.jsonl
Usage: mm_verify.py <repos_file>
"""
import json
import os
import re
import sys
import time
import urllib.parse
from collections import Counter

import requests

EP = "https://hf-mirror.com"
OUT = "/root/mm-verify.jsonl"
PAGE = 1000
MAXPAGES = 400

LABEL_HINT = re.compile(
    r'(^|/)(labels?|ground_truth|gt|masks?|annotations?|segmentation|bbox|captions?|metadata|Annotations)(/|$)'
    r'|\.(txt|json|jsonl|xml|csv|yaml|yml)$', re.I)
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff', '.gif'}

S = requests.Session()
S.headers.update({"User-Agent": "curl/8"})


def get(url, params=None, timeout=90, tries=5, headers=None):
    last = None
    for a in range(tries):
        try:
            r = S.get(url, params=params, timeout=timeout, headers=headers)
            if r.status_code == 429:
                time.sleep(6 * (a + 1))
                last = "HTTP 429"
                continue
            return r
        except requests.RequestException as e:
            last = e
            time.sleep(min(3 * (a + 1), 20))
    raise RuntimeError(f"{url}: {last}")


def tree(repo):
    url = f"{EP}/api/datasets/{repo}/tree/main"
    params = {"expand": "false", "recursive": "true", "limit": str(PAGE)}
    files, seen, pages = [], set(), 0
    while True:
        r = get(url, params=params)
        if r.status_code in (401, 403):
            raise PermissionError(f"HTTP {r.status_code} listing tree")
        if r.status_code == 404:
            raise FileNotFoundError("HTTP 404")
        r.raise_for_status()
        data = r.json()
        for it in data:
            if it.get("type") == "file":
                sz = it.get("size") or (it.get("lfs") or {}).get("size") or 0
                files.append((it["path"], sz))
        pages += 1
        link = r.headers.get("Link", "")
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
        r = get(f"{EP}/api/datasets/{urllib.parse.quote(repo)}")
        if r.status_code != 200:
            rec.update(ok=False, err=f"info HTTP {r.status_code}")
            return rec
        d = r.json()
        rec.update(usedStorage=d.get("usedStorage"), gated=d.get("gated"),
                   private=d.get("private"), disabled=d.get("disabled"), sha=d.get("sha"),
                   lastModified=d.get("lastModified"), siblings=len(d.get("siblings") or []),
                   cardData=d.get("cardData") or {}, downloads=d.get("downloads"),
                   likes=d.get("likes"))
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
    try:
        r = get(f"{EP}/datasets/{repo}/resolve/main/README.md", timeout=45, tries=2)
        rec["readme_http"] = r.status_code
        rec["readme_head"] = r.text[:3500] if r.status_code == 200 else ""
    except Exception as e:  # noqa: BLE001
        rec["readme_http"] = f"err {e}"[:80]
    probe = None
    for p, s in files:
        if p.lower() in (".gitattributes", "readme.md"):
            continue
        if s and s > 4096:
            probe = p
            break
    if probe:
        try:
            r = get(f"{EP}/datasets/{repo}/resolve/main/" + urllib.parse.quote(probe),
                    timeout=60, tries=2, headers={"Range": "bytes=0-1023"})
            rec["range_probe"] = {"path": probe, "http": r.status_code,
                                  "content_range": r.headers.get("Content-Range"),
                                  "got": len(r.content)}
        except Exception as e:  # noqa: BLE001
            rec["range_probe"] = {"path": probe, "http": f"err {e}"[:80]}
    return rec


def main():
    repos = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
    done = set()
    if os.path.exists(OUT):
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
                  f"nlabel={rec.get('n_label')} nimg={rec.get('n_img')} "
                  f"probe={rec.get('range_probe', {}).get('http')} {rec.get('err', '')}", flush=True)
            time.sleep(0.3)


if __name__ == "__main__":
    main()
