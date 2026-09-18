#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cot_pilot_report.py — 汇总 CoT 试点：生成质量 + DS 视觉复核结果。"""
import json
import os
import sys
from collections import Counter, defaultdict

GEN_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/linaro/dsh/judge_work/cot/gen"
VERIFY = sys.argv[2] if len(sys.argv) > 2 else "/home/linaro/dsh/judge_work/cot/verify.jsonl"
OUT = sys.argv[3] if len(sys.argv) > 3 else "COT_PILOT_REPORT.md"

gen = []
for fn in sorted(os.listdir(GEN_DIR)):
    if fn.endswith(".jsonl"):
        for line in open(os.path.join(GEN_DIR, fn), encoding="utf-8"):
            if line.strip():
                gen.append(json.loads(line))

ver = {}
if os.path.exists(VERIFY):
    for line in open(VERIFY, encoding="utf-8"):
        if line.strip():
            o = json.loads(line)
            ver[o["id"]] = o

by_ds = defaultdict(list)
for r in gen:
    by_ds[r["dataset"]].append(r)

L = []
A = L.append
A("# CoT 试点报告（Qwen3.8-27B 生成 + DeepSeek 视觉复核）\n")
A("口径：图片统一到 12:9（1200×900），横图 letterbox 填充、竖图旋转 90° 后填充；")
A("bbox 坐标换算到画布并归一化 0–1000；每图一条思维链，覆盖该图所有框；")
A("detect 类先输出空间解构（目标在什么结构的什么方位）再拼接真值坐标；caption/vqa/classify 输出推理 + 答案。\n")

A("## 1. 生成覆盖\n")
A("| dataset | 任务 | 抽样 | 生成成功 | 解析失败 | 目标数对齐 | 竖图旋转 | bbox 总数 |")
A("|---|---|---:|---:|---:|---:|---:|---:|")
tot_ok = tot = tot_align = 0
for ds, items in sorted(by_ds.items()):
    ok = sum(1 for r in items if r["parse_ok"])
    al = sum(1 for r in items if r.get("aligned"))
    rot = sum(1 for r in items if r.get("rotated"))
    nb = sum(len(r.get("gt_boxes") or []) for r in items)
    tot_ok += ok; tot += len(items); tot_align += al
    A("| %s | %s | %d | %d | %d | %d | %d | %d |" %
      (ds, items[0]["task"], len(items), ok, len(items) - ok, al, rot, nb))
A("| **合计** | | **%d** | **%d** | **%d** | **%d** | | |" % (tot, tot_ok, tot - tot_ok, tot_align))
A("")

if ver:
    A("## 2. DeepSeek 视觉复核结果\n")
    A("| dataset | 已复核 | pass | minor | fail | 解析/接口失败 | 通过率 |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    tot_c = Counter()
    for ds, items in sorted(by_ds.items()):
        ids = [r["id"] for r in items]
        sub = [ver[i] for i in ids if i in ver]
        c = Counter(o.get("v") for o in sub)
        good = c.get("pass", 0) + c.get("minor", 0)
        for k, v in c.items():
            tot_c[k] += v
        A("| %s | %d | %d | %d | %d | %d | %.1f%% |" %
          (ds, len(sub), c.get("pass", 0), c.get("minor", 0), c.get("fail", 0),
           c.get("parse_error", 0) + c.get("api_error", 0),
           100.0 * good / max(len(sub), 1)))
    n_all = sum(tot_c.values())
    good_all = tot_c.get("pass", 0) + tot_c.get("minor", 0)
    A("| **合计** | **%d** | **%d** | **%d** | **%d** | **%d** | **%.1f%%** |" %
      (n_all, tot_c.get("pass", 0), tot_c.get("minor", 0), tot_c.get("fail", 0),
       tot_c.get("parse_error", 0) + tot_c.get("api_error", 0), 100.0 * good_all / max(n_all, 1)))
    A("")

    A("## 3. 复核发现的问题类型分布\n")
    tcnt = Counter()
    for o in ver.values():
        for it in (o.get("issues") or []):
            tcnt[str(it.get("type", "?"))] += 1
    A("| 类型 | 次数 |")
    A("|---|---:|")
    for t, c in tcnt.most_common():
        A("| %s | %d |" % (t, c))
    A("")
    # 各判据不通过率
    A("### 3.1 分判据不通过率\n")
    A("| 判据 | 不通过 | 占比 |")
    A("|---|---:|---:|")
    for key, label in (("spatial_ok", "spatial 空间结构不符"), ("align_ok", "目标方位/结构不符"),
                       ("invented", "描述了不存在的目标"), ("coord_ok", "方位与坐标矛盾")):
        bad = 0; n = 0
        for o in ver.values():
            if key in o:
                n += 1
                bad += (o.get(key) is True) if key == "invented" else (o.get(key) is False)
        A("| %s | %d | %.1f%% |" % (label, bad, 100.0 * bad / max(n, 1)))
    A("")

    A("## 4. fail 样例（最多 12 条）\n")
    shown = 0
    for r in gen:
        o = ver.get(r["id"])
        if not o or o.get("v") != "fail":
            continue
        A("### %s · %s" % (r["dataset"], r["row_id"]))
        A("- 图: `%s`（旋转=%s，目标 %d 个）" % (r["image"], r.get("rotated"), len(r.get("gt_boxes") or [])))
        A("- 真值: %s" % "; ".join("%s@[%s]" % (b["label"], ",".join(str(v) for v in b["bbox1000"]))
                                   for b in (r.get("gt_boxes") or []))[:200])
        A("- CoT: %s" % (r.get("cot") or "")[:400].replace("\n", " "))
        A("- 复核: %s" % "; ".join("%s/%s: %s" % (i.get("type"), i.get("target"), i.get("problem"))
                                   for i in (o.get("issues") or [])))
        shown += 1
        if shown >= 12:
            break
    A("")
    A("## 5. pass 样例（最多 3 条，供人工抽检）\n")
    shown = 0
    for r in gen:
        o = ver.get(r["id"])
        if not o or o.get("v") != "pass" or r["task"] == "detect" and len(r.get("gt_boxes") or []) == 0:
            continue
        A("### %s · %s" % (r["dataset"], r["row_id"]))
        A("- 图: `%s`（旋转=%s）" % (r["image"], r.get("rotated")))
        A("- 真值: %s" % "; ".join("%s@[%s]" % (b["label"], ",".join(str(v) for v in b["bbox1000"]))
                                   for b in (r.get("gt_boxes") or []))[:160])
        A("- CoT:\n\n```\n%s\n```\n" % (r.get("cot") or "")[:700])
        shown += 1
        if shown >= 3:
            break

open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote", OUT)
print("gen=%d verified=%d" % (len(gen), len(ver)))
if ver:
    c = Counter(o.get("v") for o in ver.values())
    print("verify counts:", dict(c))
