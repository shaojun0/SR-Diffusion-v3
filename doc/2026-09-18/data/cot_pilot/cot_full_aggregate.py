#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cot_full_aggregate.py — 全量 CoT 产物汇总（在服务器上跑）。

输出：
  /root/autodl-tmp/cot_out/SUMMARY.json   总体与逐数据集统计
  /root/autodl-tmp/cot_out/SUMMARY.md     人读报告
并打印一行结论。
"""
import glob
import json
import os
from collections import Counter, defaultdict

OUT = os.environ.get("COT_OUTDIR", "/root/autodl-tmp/cot_out")
MAN = os.environ.get("COT_ROOT", "/root/autodl-tmp/cot_full")

# 计划条数（来自 build_stats*.json）
planned = defaultdict(int)
for f in glob.glob(os.path.join(MAN, "build_stats*.json")):
    try:
        d = json.load(open(f, encoding="utf-8"))
        for k, v in d.get("datasets", {}).items():
            planned[k] += v.get("sampled", 0)
    except Exception:
        pass

stats = {}
tot = Counter()
for f in sorted(glob.glob(os.path.join(OUT, "*.jsonl"))):
    ds = os.path.basename(f)[:-len(".jsonl")]
    c = Counter(); n = 0
    cot_len = 0
    for line in open(f, encoding="utf-8"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            c["bad_json"] += 1
            continue
        n += 1
        c["parse_ok" if r.get("parse_ok") else "parse_fail"] += 1
        if r.get("task") == "detect":
            c["aligned" if r.get("aligned") else "misaligned"] += 1
        c["task:" + str(r.get("task"))] += 1
        if r.get("rotated"):
            c["rotated"] += 1
        cot_len += len(r.get("cot") or "")
    stats[ds] = {"n": n, "planned": planned.get(ds, 0), **c}
    tot.update(c); tot["n"] += n

ok = tot.get("parse_ok", 0)
al = tot.get("aligned", 0)
det = tot.get("task:detect", 0)
res = {
    "total": tot["n"],
    "planned_total": sum(planned.values()),
    "parse_ok": ok,
    "parse_fail": tot.get("parse_fail", 0),
    "bad_json": tot.get("bad_json", 0),
    "detect_rows": det,
    "aligned": al,
    "rotated": tot.get("rotated", 0),
    "by_dataset": stats,
}
json.dump(res, open(os.path.join(OUT, "SUMMARY.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

L = ["# 全量 CoT 生成汇总\n",
     "| 数据集 | 已生成 | 计划 | 解析成功 | 解析失败 | detect 对齐 | 旋转 |",
     "|---|---:|---:|---:|---:|---:|---:|"]
for ds in sorted(stats):
    s = stats[ds]
    aln = "%d/%d" % (s.get("aligned", 0), s.get("task:detect", 0)) if s.get("task:detect") else "—"
    L.append("| %s | %d | %d | %d | %d | %s | %d |" %
             (ds, s["n"], s["planned"], s.get("parse_ok", 0), s.get("parse_fail", 0) + s.get("bad_json", 0),
              aln, s.get("rotated", 0)))
L.append("| **合计** | **%d** | **%d** | **%d** | **%d** | **%d/%d** | **%d** |" %
         (tot["n"], sum(planned.values()), ok, tot.get("parse_fail", 0) + tot.get("bad_json", 0), al, det, tot.get("rotated", 0)))
open(os.path.join(OUT, "SUMMARY.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("total=%d planned=%d parse_ok=%d parse_fail=%d detect_aligned=%d/%d rotated=%d" %
      (tot["n"], sum(planned.values()), ok, tot.get("parse_fail", 0) + tot.get("bad_json", 0), al, det, tot.get("rotated", 0)))
