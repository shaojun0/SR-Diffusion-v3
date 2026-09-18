#!/usr/bin/env python3
"""Keyword sweep of hf-mirror dataset search -> candidate table (JSONL).

Mirror-native: all requests to https://hf-mirror.com.
"""
import json
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

EP = "https://hf-mirror.com"
OUT = "/home/linaro/dsh/srdiff_overnight/datasets/supplement/candidates_raw.json"

DOMAIN_KW = [
    "construction site", "construction safety", "construction", "PPE detection",
    "personal protective equipment", "hard hat", "safety helmet", "safety vest",
    "high visibility vest", "safety harness", "scaffolding", "crane", "excavator",
    "heavy equipment", "industrial safety", "workplace safety", "work zone",
    "hazard detection", "hazard", "fire detection", "fire smoke", "smoke detection",
    "flame", "wildfire", "crack detection", "concrete crack", "pavement crack",
    "road damage", "pothole", "bridge defect", "building defect", "facade defect",
    "infrastructure inspection", "corrosion detection", "rust detection",
    "sewer defect", "pipeline defect", "weld defect", "surface defect",
    "industrial defect", "anomaly detection industrial", "mining safety",
    "electrical safety", "forklift", "warehouse safety", "worker detection",
    "fall detection", "power line inspection", "transmission tower",
    "solar panel defect", "wind turbine", "tunnel crack", "railway inspection",
    "defect segmentation", "chemical safety", "gas leak", "oil spill",
    "mvtec", "industrial inspection", "logistics safety", "traffic safety",
    "excavation", "demolition", "formwork", "rebar", "bricklaying",
    "occupational safety", "confined space", "ladder safety", "guardrail",
    "traffic cone", "roadwork", "heavy machinery", "steel structure defect",
    "pipe inspection", "thermal defect", "weapon detection", "firearm",
]

GENERAL_KW = [
    "visual question answering", "vqa", "image captioning", "object detection",
    "coco", "open images", "flickr30k", "textcaps", "docvqa", "chartqa",
    "visual genome", "referring expression", "image segmentation", "grounding",
    "ocr vqa", "scene text", "video anomaly", "counting", "depth estimation",
    "remote sensing", "satellite", "aerial imagery", "medical imaging",
    "chest xray", "traffic sign", "autonomous driving", "pedestrian",
    "plant disease", "waste detection", "recycling", "agriculture",
]

KWS = DOMAIN_KW + GENERAL_KW


def get(url, timeout=60, tries=4):
    last = None
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(2 ** a, 15))
    raise RuntimeError(f"{url}: {last}")


def search(kw):
    url = f"{EP}/api/datasets?" + urllib.parse.urlencode(
        {"search": kw, "limit": "100", "full": "true"})
    try:
        data = json.loads(get(url))
    except Exception as e:  # noqa: BLE001
        return kw, [], str(e)
    return kw, data if isinstance(data, list) else [], None


def main():
    kws = KWS
    if len(sys.argv) > 1 and sys.argv[1] == "domain":
        kws = DOMAIN_KW
    pool = {}
    lock = threading.Lock()

    def work(kw):
        k, rows, err = search(kw)
        with lock:
            if err:
                print(f"[ERR] {k}: {err}", file=sys.stderr)
            for r in rows:
                rid = r.get("id")
                if not rid:
                    continue
                rec = pool.setdefault(rid, {"id": rid, "kws": [], "gated": r.get("gated"),
                                            "private": r.get("private"), "downloads": r.get("downloads"),
                                            "likes": r.get("likes"), "tags": r.get("tags") or [],
                                            "lastModified": r.get("lastModified"),
                                            "cardData": r.get("cardData") or {}})
                rec["kws"].append(k)
                rec["downloads"] = max(rec.get("downloads") or 0, r.get("downloads") or 0)
                rec["likes"] = max(rec.get("likes") or 0, r.get("likes") or 0)
            print(f"[ok] {k}: {len(rows)} rows, pool={len(pool)}", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(work, kws))

    with open(OUT, "w") as f:
        for rec in sorted(pool.values(), key=lambda x: -(x.get("downloads") or 0)):
            f.write(json.dumps(rec) + "\n")
    print(f"WROTE {OUT} repos={len(pool)} kws={len(kws)}")


if __name__ == "__main__":
    main()
