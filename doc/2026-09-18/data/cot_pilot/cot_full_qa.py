#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cot_full_qa.py — 全量 CoT 产物的机械质检（不调模型，纯规则）。

检查项：
  1) JSON 解析失败 / 记录损坏
  2) detect 类目标数是否与真值框数一致（aligned）
  3) CoT 为空或过短
  4) 退化重复（同一字符连续 >=8，或同一 20 字子串重复 >=5 次）
  5) CoT 是否覆盖了真值类别标签
  6) 画布一致性：manifest 声明的 canvas 必须是 [1200,900]，
     且产物所指**实际 JPEG 文件头尺寸**必须是 1200x900
     （默认全量核查；QA_IMG_SAMPLE=N>0 时每 N 行抽样核查一张）
  7) detect 行真值框坐标必须落在 0–1000 且 x1<x2、y1<y2

输出：stdout 汇总 + cot_out/QA.json

修复记录（2026-09-18 接手会话）：
  旧版第 6 项写作 `if r.get("canvas") != [1200, 900]`，但生成产物
  （cot_generate.py 的 out 字典）**不含 canvas 字段**——canvas 只存在于
  manifest。因此该条件对每一行都成立，bad_canvas 恒等于全部行数（100% 误报），
  而真正的"图是否为 1200x900"从未被验证。现改为：
    a) 按 id 连接 manifest，核对**声明画布**；
    b) 用 PIL 读实际图片头，核对**真实尺寸**。
"""
import glob
import json
import os
import re
import statistics
from collections import Counter

OUT = os.environ.get("COT_OUTDIR", "/root/autodl-tmp/cot_out")
CTX = os.environ.get("COT_ROOT", "/root/autodl-tmp/cot_full")
CANVAS = [1200, 900]
IMG_SAMPLE = int(os.environ.get("QA_IMG_SAMPLE", "0"))  # 0 = 全量核查实际图片尺寸

rep_char = re.compile(r"(.)\1{7,}")
rep_sub = re.compile(r"(.{20,40}?)\1{4,}", re.S)

try:
    from PIL import Image
except Exception:                                    # pragma: no cover
    Image = None

# ---- manifest: id -> 声明的 canvas ----
declared = {}
n_manifest = 0
for mf in sorted(glob.glob(os.path.join(CTX, "manifest*.jsonl"))):
    for line in open(mf, encoding="utf-8"):
        if not line.strip():
            continue
        try:
            m = json.loads(line)
        except Exception:
            continue
        declared[m.get("id")] = m.get("canvas")
        n_manifest += 1
if not declared:
    print("[qa][WARN] 未找到 %s/manifest*.jsonl，声明画布无法核对（bad_canvas 将不可信）" % CTX)


def actual_size(rel):
    if Image is None or not rel:
        return None
    try:
        with Image.open(os.path.join(CTX, rel)) as im:
            return list(im.size)
    except Exception:
        return None


res = {}
tot = Counter()
len_all = []
for f in sorted(glob.glob(os.path.join(OUT, "*.jsonl"))):
    ds = os.path.basename(f)[:-len(".jsonl")]
    c = Counter()
    lens = []
    miss_label = 0
    det = 0
    seen = 0
    for line in open(f, encoding="utf-8"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            c["bad_json"] += 1
            continue
        c["n"] += 1
        if not r.get("parse_ok"):
            c["parse_fail"] += 1

        # ---- 6) 画布：声明值 + 实际图片尺寸 ----
        if declared.get(r.get("id")) != CANVAS:
            c["bad_canvas"] += 1
        seen += 1
        if IMG_SAMPLE <= 0 or seen % IMG_SAMPLE == 1:
            sz = actual_size(r.get("image"))
            if sz is None:
                c["missing_image"] += 1
            elif sz != CANVAS:
                c["bad_image_size"] += 1

        if r.get("rotated"):
            c["rotated"] += 1
        cot = r.get("cot") or ""
        lens.append(len(cot))
        len_all.append(len(cot))
        if len(cot) < 20:
            c["too_short"] += 1
        if rep_char.search(cot) or rep_sub.search(cot):
            c["degenerate_repeat"] += 1
        if r.get("task") == "detect":
            det += 1
            if not r.get("aligned"):
                c["misaligned"] += 1
            labels = {b["label"] for b in (r.get("gt_boxes") or [])}
            if labels and not any(str(lb)[:12] in cot for lb in labels):
                miss_label += 1
            # ---- 7) 真值框坐标合法性 ----
            for b in (r.get("gt_boxes") or []):
                bb = b.get("bbox1000") or []
                ok = (len(bb) == 4 and all(0 <= v <= 1000 for v in bb)
                      and bb[0] < bb[2] and bb[1] < bb[3])
                if not ok:
                    c["bad_bbox"] += 1
                    break
        if not cot:
            c["empty_cot"] += 1
    c["detect_rows"] = det
    c["label_missing"] = miss_label
    if lens:
        c["cot_len_median"] = int(statistics.median(lens))
        c["cot_len_min"] = min(lens)
        c["cot_len_max"] = max(lens)
    res[ds] = dict(c)
    tot.update(c)

res["_total"] = dict(tot)
res["_total"]["manifest_rows"] = n_manifest
if len_all:
    res["_total"]["cot_len_median"] = int(statistics.median(len_all))
    res["_total"]["cot_len_min"] = min(len_all)
    res["_total"]["cot_len_max"] = max(len_all)
json.dump(res, open(os.path.join(OUT, "QA.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

hdr = ("%-50s %7s %6s %6s %8s %8s %8s %9s %10s %8s" %
       ("dataset", "rows", "parse✗", "align✗", "空/过短", "退化重复", "缺标签", "画布✗", "尺寸✗", "框✗"))
print(hdr)
for ds in sorted(k for k in res if k != "_total"):
    c = res[ds]
    print("%-50s %7d %6d %6d %8d %8d %8d %9d %9d %8d" % (
        ds[:50], c.get("n", 0), c.get("parse_fail", 0) + c.get("bad_json", 0),
        c.get("misaligned", 0), c.get("too_short", 0) + c.get("empty_cot", 0),
        c.get("degenerate_repeat", 0), c.get("label_missing", 0),
        c.get("bad_canvas", 0), c.get("bad_image_size", 0) + c.get("missing_image", 0),
        c.get("bad_bbox", 0)))
t = res["_total"]
print("%-50s %7d %6d %6d %8d %8d %8d %9d %9d %8d" % (
    "TOTAL", t.get("n", 0), t.get("parse_fail", 0) + t.get("bad_json", 0),
    t.get("misaligned", 0), t.get("too_short", 0) + t.get("empty_cot", 0),
    t.get("degenerate_repeat", 0), t.get("label_missing", 0),
    t.get("bad_canvas", 0), t.get("bad_image_size", 0) + t.get("missing_image", 0),
    t.get("bad_bbox", 0)))
print("CoT 长度: 中位 %s / 最短 %s / 最长 %s  |  manifest 行数 %d  |  实际图片尺寸核查 %s" %
      (t.get("cot_len_median"), t.get("cot_len_min"), t.get("cot_len_max"), n_manifest,
       "全量" if IMG_SAMPLE <= 0 else "每 %d 行 1 张" % IMG_SAMPLE))
