#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cot_full_repair.py — 事后回收解析失败：用加固后的解析器重解析 raw，就地把 parse_fail 修复。

用法（**务必等全部生成结束后再跑**，因为它会重写产物文件）：
    python3 cot_full_repair.py            # 就地修复，先生成 .pre_repair.bak
    python3 cot_full_repair.py --dry-run  # 只报告
"""
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, "/root/translate")
import cot_generate as g   # noqa: E402  复用加固后的 parse_json

OUT = os.environ.get("COT_OUTDIR", "/root/autodl-tmp/cot_out")
DRY = "--dry-run" in sys.argv

total_fail = total_fixed = total_left = 0
for f in sorted(glob.glob(os.path.join(OUT, "*.jsonl"))):
    recs = []
    n_fail = n_fix = 0
    for line in open(f, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("parse_ok"):
            n_fail += 1
            raw = r.get("raw") or ""
            obj = g.parse_json(raw) if raw else None
            if obj:
                r["gen"] = obj
                r["parse_ok"] = True
                r["repaired"] = True
                # 重新拼接 cot（与生成时同一逻辑）
                if r.get("task") == "detect":
                    positions = obj.get("objects") or obj.get("object") or []
                    gtb = r.get("gt_boxes") or []
                    if isinstance(positions, list) and len(positions) == len(gtb):
                        parts = [str(obj.get("spatial", "")).strip()]
                        for b, pos in zip(gtb, positions):
                            if isinstance(pos, dict):
                                pos = pos.get("position") or pos.get("desc") or json.dumps(pos, ensure_ascii=False)
                            x1, y1, x2, y2 = b["bbox1000"]
                            parts.append("%s：%s → 坐标 [%s, %s, %s, %s]" % (b["label"], str(pos).strip(), x1, y1, x2, y2))
                        r["cot"] = "\n".join(p for p in parts if p)
                        r["aligned"] = True
                else:
                    r["cot"] = str(obj.get("cot") or "").strip() or None
                n_fix += 1
        recs.append(r)
    total_fail += n_fail; total_fixed += n_fix; total_left += (n_fail - n_fix)
    print("%-52s parse_fail=%-4d 修复=%-4d 剩余=%d" % (os.path.basename(f)[:52], n_fail, n_fix, n_fail - n_fix))
    if n_fix and not DRY:
        shutil.copy2(f, f + ".pre_repair.bak")
        tmp = f + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fo:
            for r in recs:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, f)

print("合计: parse_fail=%d 修复=%d 剩余=%d %s" % (total_fail, total_fixed, total_left, "(dry-run)" if DRY else ""))
