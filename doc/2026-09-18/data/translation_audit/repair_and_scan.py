#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""repair_and_scan.py — (1) 修复 parse_error（实为 garble 判决被截断）
                        (2) 规则扫描全文重复字/乱码，与模型判决交叉验证"""
import json
import re
import sys
from collections import Counter

VERD = "verdicts_all.jsonl"
UNITS = "judge_units_all.jsonl"

# ---- 1. 修复 parse_error ----
repaired = []
seen = set()
kept = []
for line in open(VERD, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if r["v"] not in ("ok", "minor", "major"):
        raw = r.get("raw") or ""
        if '"v":"major"' in raw or '"v": "major"' in raw:
            t = "garble" if "garble" in raw else "other"
            r = dict(r)
            r["v"] = "major"
            r["e"] = [t]
            r["err"] = [{"type": t, "en": (r.get("text") or "")[:40], "zh": (r.get("zh") or "")[:40]}]
            r["repaired_from"] = "parse_error_truncated"
            repaired.append(r)
    kept.append(r)

if repaired:
    with open(VERD, "a", encoding="utf-8") as fo:
        for r in repaired:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---- 2. 规则扫描 ----
rep = re.compile(r"(.)\1{7,}")          # 同一字符连续 >=8 次
rows = []
for line in open(UNITS, encoding="utf-8"):
    line = line.strip()
    if line:
        rows.append(json.loads(line))

judge = {}
for line in open(VERD, encoding="utf-8"):
    line = line.strip()
    if line:
        r = json.loads(line)
        judge[r["id"]] = r

n_rep = 0
n_nocjk = 0
n_long = 0
samples = []
for r in rows:
    zh = r.get("zh") or ""
    m = rep.search(zh)
    if m:
        n_rep += 1
        if len(samples) < 5:
            samples.append((r["id"], m.group(0)[:24], len(zh)))
    if not re.search(r"[\u4e00-\u9fff]", zh):
        n_nocjk += 1
    if len(zh) > 400:
        n_long += 1

print("规则扫描（全部 %d 单元）:" % len(rows))
print("  同一字符连续>=8 次（乱码）: %d (%.2f%%)" % (n_rep, 100.0 * n_rep / len(rows)))
print("  译文不含任何汉字: %d (%.2f%%)" % (n_nocjk, 100.0 * n_nocjk / len(rows)))
print("  译文超长(>400字): %d (%.2f%%)" % (n_long, 100.0 * n_long / len(rows)))
for s in samples:
    print("   样例:", s)
print("修复 parse_error -> major:", len(repaired))
