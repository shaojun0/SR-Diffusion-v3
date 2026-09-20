#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cot_full_repair.py — 事后回收解析失败：用加固后的解析器重解析 raw，就地把 parse_fail 修复。

用法（**务必等全部生成结束后再跑**，因为它会重写产物文件）：
    python3 cot_full_repair.py            # 就地修复，先生成 .pre_repair.bak
    python3 cot_full_repair.py --dry-run  # 只报告

⚠️ 安全约定（2026-09-18 事故后加固）：
  本脚本用 shutil.copy2 + os.replace 重写产物；而生成器 cot_generate.py 是
  `fo=open(out_path,"a")` 一次打开、**全程持有 fd**，运行中重写会 unlink 旧 inode ⇒
  writer 继续往已删除的 inode 追加（可见文件冻结）。事故记录见
  translate_logs/cot_full/INCIDENT_20260918_repair_orphan.md。
  因此：① 只在生成全部停止后运行；② 逻辑已包进 main()，**import 本模块不再触发写盘**。
"""
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, "/root/translate")
import cot_generate as g   # noqa: E402  复用加固后的 parse_json


def main():
    out = os.environ.get("COT_OUTDIR", "/root/autodl-tmp/cot_out")
    dry = "--dry-run" in sys.argv

    total_fail = total_fixed = total_left = 0
    for f in sorted(glob.glob(os.path.join(out, "*.jsonl"))):
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
        if n_fix and not dry:
            shutil.copy2(f, f + ".pre_repair.bak")
            tmp = f + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fo:
                for r in recs:
                    fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            os.replace(tmp, f)

    print("合计: parse_fail=%d 修复=%d 剩余=%d %s" % (total_fail, total_fixed, total_left, "(dry-run)" if dry else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
