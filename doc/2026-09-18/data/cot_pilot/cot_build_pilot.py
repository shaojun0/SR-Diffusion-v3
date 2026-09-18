#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cot_build_pilot.py — CoT 试点数据构建：每数据集抽 100 张，统一到 12:9(1200x900)，
横图 letterbox 填充、竖图旋转 90° 后填充；bbox 换算为画布归一化 0-1000。

产出：/root/autodl-tmp/cot_pilot/<dataset>/images/*.jpg + manifest.jsonl
只读原始数据集，不写回。
"""
import glob
import json
import io
import os
import random
import sys

import pyarrow.parquet as pq
from PIL import Image

OUT_ROOT = "/root/autodl-tmp/cot_pilot"
CANVAS = (1200, 900)          # 12:9
N = int(os.environ.get("PILOT_N", "100"))
SEED = 20260918
BASE_MM = "/root/autodl-tmp/mm-datasets"
BASE_ADD = "/root/autodl-tmp/mm-datasets-add"


def hf_names(f, col="label"):
    try:
        meta = pq.ParquetFile(f).schema_arrow.metadata or {}
        j = json.loads(meta[b"huggingface"].decode("utf-8"))
        return j["info"]["features"][col]["names"]
    except Exception:
        return None


def norm_xyxy_to_canvas(x1, y1, x2, y2, w, h, rotated, canvas=CANVAS):
    """归一化 xyxy（原图）-> 旋转/letterbox 后画布归一化 0-1000。"""
    if rotated:                     # 竖图顺时针转 90°: (x,y) -> (1-y, x)
        x1, y1, x2, y2 = 1 - y2, x1, 1 - y1, x2
        w, h = h, w
    cw, ch = canvas
    s = min(cw / w, ch / h)
    nw, nh = w * s, h * s
    ox, oy = (cw - nw) / 2.0, (ch - nh) / 2.0
    X1 = (x1 * w * s + ox) / cw * 1000.0
    Y1 = (y1 * h * s + oy) / ch * 1000.0
    X2 = (x2 * w * s + ox) / cw * 1000.0
    Y2 = (y2 * h * s + oy) / ch * 1000.0
    clip = lambda v: max(0.0, min(1000.0, round(v, 1)))
    return [clip(X1), clip(Y1), clip(X2), clip(Y2)]


def preprocess(img_bytes, canvas=CANVAS):
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = im.size
    rotated = h > w
    if rotated:
        im = im.transpose(Image.ROTATE_270)   # 顺时针 90°
        w, h = im.size
    cw, ch = canvas
    s = min(cw / w, ch / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    im2 = im.resize((nw, nh), Image.LANCZOS)
    canvas_im = Image.new("RGB", canvas, (127, 127, 127))
    canvas_im.paste(im2, ((cw - nw) // 2, (ch - nh) // 2))
    buf = io.BytesIO()
    canvas_im.save(buf, "JPEG", quality=92)
    return buf.getvalue(), {"orig_w": w if not rotated else h, "orig_h": h if not rotated else w,
                            "rotated": rotated, "scale": round(s, 6)}


DATASETS = {
    "aswin00000__ConstructionSiteCleanedDataSet": {
        "glob": BASE_MM + "/aswin00000__ConstructionSiteCleanedDataSet/data/*.parquet",
        "task": "detect", "require_boxes": True, "img_col": "image",
    },
    "iluvvatar__wood_surface_defects": {
        "glob": BASE_ADD + "/iluvvatar__wood_surface_defects/**/*.parquet",
        "task": "detect", "require_boxes": True, "img_col": "image",
    },
    "hf-vision__hardhat": {
        "glob": BASE_ADD + "/hf-vision__hardhat/**/*.parquet",
        "task": "detect", "require_boxes": True, "img_col": "image",
    },
    "baizhanquan__FireDetectionDataset": {
        "glob": BASE_ADD + "/baizhanquan__FireDetectionDataset-flame-forest-flameye-wildfire/**/*.parquet",
        "task": "detect", "require_boxes": True, "img_col": "image",
    },
    "hayden-yuma__roadwork": {
        "glob": BASE_ADD + "/hayden-yuma__roadwork/**/*.parquet",
        "task": "caption", "img_col": "image",
    },
    "chandrabhuma__multi_building_defect_vqa": {
        "glob": BASE_MM + "/chandrabhuma__multi_building_defect_vqa/**/*.parquet",
        "task": "vqa", "img_col": "image",
    },
    "kevincluo__structure_wildfire_damage_classification": {
        "glob": BASE_ADD + "/kevincluo__structure_wildfire_damage_classification/**/*.parquet",
        "task": "classify", "img_col": "image",
    },
}


def extract_boxes(name, r, w, h):
    """返回 [(label, [x1,y1,x2,y2] 归一化)]"""
    out = []
    if name.startswith("aswin"):
        for k in (1, 2, 3, 4):
            v = r.get("rule_%d_violation" % k)
            if not v:
                continue
            reason = (v.get("reason") or "").strip()
            for bb in (v.get("bounding_box") or []):
                if len(bb) == 4:
                    out.append(("rule%d: %s" % (k, reason[:60]), list(bb)))
    elif name.startswith("iluvvatar"):
        # README: "Bounding boxes converted to YOLO format" => [x_center, y_center, w, h] 归一化
        for o in (r.get("objects") or []):
            bb = o.get("bb") or []
            if len(bb) == 4:
                xc, yc, bw, bh = bb
                out.append((o.get("label", "?"),
                            [xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2]))
    elif name.startswith("hf-vision"):
        objs = r.get("objects") or {}
        for bb, cat in zip(objs.get("bbox") or [], objs.get("category") or []):
            if len(bb) == 4 and w and h:
                x, y, bw, bh = bb
                out.append((cat, [x / w, y / h, (x + bw) / w, (y + bh) / h]))
    elif name.startswith("baizhanquan"):
        for a in (r.get("annotations") or []):
            xc, yc, bw, bh = a.get("x_center"), a.get("y_center"), a.get("width"), a.get("height")
            if None not in (xc, yc, bw, bh):
                out.append((a.get("class_name", "?"), [xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2]))
    return out


def main():
    random.seed(SEED)
    os.makedirs(OUT_ROOT, exist_ok=True)
    manifest = []
    stats = {}
    for name, cfg in DATASETS.items():
        files = sorted(glob.glob(cfg["glob"]))
        if not files:
            print("[skip] %s 无文件" % name)
            continue
        d = os.path.join(OUT_ROOT, name)
        os.makedirs(os.path.join(d, "images"), exist_ok=True)
        cols = [cfg["img_col"]]
        got = 0
        tried = 0
        random.shuffle(files)
        names = hf_names(files[0]) if cfg["task"] == "classify" else None
        for f in files:
            if got >= N:
                break
            pf = pq.ParquetFile(f)
            sch = pf.schema_arrow.names
            want = [c for c in ["image_caption", "rule_1_violation", "rule_2_violation",
                                "rule_3_violation", "rule_4_violation", "objects",
                                "annotations", "width", "height", "scene_description",
                                "question", "answer", "label"] if c in sch]
            for batch in pf.iter_batches(batch_size=16, columns=[cfg["img_col"]] + want):
                if got >= N:
                    break
                rows = batch.to_pylist()
                random.shuffle(rows)
                for r in rows:
                    if got >= N:
                        break
                    tried += 1
                    img = r.get(cfg["img_col"]) or {}
                    b = img.get("bytes") if isinstance(img, dict) else None
                    if not b:
                        continue
                    try:
                        im = Image.open(io.BytesIO(b)); w, h = im.size
                    except Exception:
                        continue
                    boxes = extract_boxes(name, r, w, h)
                    if cfg.get("require_boxes") and not boxes:
                        continue
                    try:
                        jpg, meta = preprocess(b)
                    except Exception as e:
                        print("  preprocess fail:", str(e)[:80]); continue
                    rid = "%s#%d" % (os.path.basename(f), got)
                    img_rel = os.path.join(name, "images", "%04d.jpg" % got)
                    open(os.path.join(OUT_ROOT, img_rel), "wb").write(jpg)
                    rec = {
                        "id": "%s|%s" % (name, rid), "dataset": name, "row_id": rid,
                        "src_shard": os.path.basename(f), "task": cfg["task"],
                        "image": img_rel, "canvas": list(CANVAS), **meta,
                        "boxes": [{"label": lb,
                                   "bbox1000": norm_xyxy_to_canvas(*bb, meta["orig_w"], meta["orig_h"],
                                                                   meta["rotated"])}
                                  for lb, bb in boxes],
                        "gt": {},
                    }
                    if cfg["task"] == "detect":
                        rec["gt"]["caption"] = (r.get("image_caption") or "").strip()
                    elif cfg["task"] == "caption":
                        rec["gt"]["text"] = (r.get("scene_description") or "").strip()
                    elif cfg["task"] == "vqa":
                        rec["gt"]["question"] = (r.get("question") or "").strip()
                        rec["gt"]["answer"] = (r.get("answer") or "").strip()
                    elif cfg["task"] == "classify":
                        lab = r.get("label")
                        try:
                            lab = names[int(lab)] if names else str(lab)
                        except Exception:
                            lab = str(lab)
                        rec["gt"]["label"] = lab
                    manifest.append(rec)
                    got += 1
        stats[name] = {"sampled": got, "scanned": tried,
                       "with_boxes": sum(1 for m in manifest if m["dataset"] == name and m["boxes"]),
                       "rotated": sum(1 for m in manifest if m["dataset"] == name and m["rotated"])}
        print("[ok] %-58s sampled=%d rotated=%d with_boxes=%d" %
              (name, got, stats[name]["rotated"], stats[name]["with_boxes"]), flush=True)

    with open(os.path.join(OUT_ROOT, "manifest.jsonl"), "w", encoding="utf-8") as fo:
        for m in manifest:
            fo.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT_ROOT, "build_stats.json"), "w", encoding="utf-8") as fo:
        json.dump(stats, fo, ensure_ascii=False, indent=1)
    print("manifest rows:", len(manifest))


if __name__ == "__main__":
    main()
