#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cot_full_repair2.py — 二次回收：对 cot_full_repair.py 修不掉的 parse_fail 再试一轮
「括号配平」修复（只补/换结尾括号，不改动任何已有字符，且必须能 json.loads 通过并含期望键）。

背景（2026-09-20 收尾实测）：8 条 parse_fail 分两类
  A. raw 被 max_tokens 截断（长度正好 1500，模型在 self-doubt 循环里）——**不可修**；
  B. raw 只是结尾括号写错（`]]` 应为 `]}`，或漏了 `}`）——**一条字符级修复即可**。
本脚本只处理 B 类；A 类保持 parse_ok=False。

用法：
    python3 cot_full_repair2.py --dry-run
    python3 cot_full_repair2.py            # 就地修复，写 .pre_repair2.bak
"""
import glob
import json
import os
import shutil
import sys

OUT = os.environ.get("COT_OUTDIR", "/root/autodl-tmp/cot_out")
KEYS = ("spatial", "objects", "object", "cot", "evidence")


def balance(s):
    """补/换结尾括号：不改动已有字符，仅在必要时于末尾补引号与闭合括号。"""
    out, stack, instr, esc = [], [], False, False
    for ch in s:
        out.append(ch)
        if instr:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
        else:
            if ch == '"':
                instr = True
            elif ch == "{":
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            elif ch in "}]":
                if stack and stack[-1] == ch:
                    stack.pop()
    r = "".join(out)
    if instr:
        r += '"'
    r = r.rstrip()
    while r.endswith(","):
        r = r[:-1]
    while stack:
        r += stack.pop()
    return r


def in_string(s):
    """结尾是否仍处于未闭合的字符串里（= 被 max_tokens 截断，不可用）。"""
    instr = esc = False
    for ch in s:
        if instr:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
        else:
            if ch == '"':
                instr = True
    return instr


def candidates(raw):
    raw = raw.strip()
    seen = set()
    for c in [raw, balance(raw)] + [balance(raw[:-k]) for k in (1, 2, 3)]:
        if c and c not in seen:
            seen.add(c)
            yield c


def deep_parse(raw):
    raw = (raw or "").strip()
    # 截断守卫：结尾仍在字符串里、或结尾不是结构符号 ⇒ 判定为被 max_tokens 截断，不修
    if in_string(raw) or (raw and raw[-1] not in '"]}'):
        return None
    for c in candidates(raw):
        try:
            o = json.loads(c)
        except Exception:
            continue
        if isinstance(o, dict) and any(k in o for k in KEYS):
            return o
    return None


def main():
    dry = "--dry-run" in sys.argv
    total_fail = total_fix = 0
    for f in sorted(glob.glob(os.path.join(OUT, "*.jsonl"))):
        recs, n_fail, n_fix = [], 0, 0
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("parse_ok"):
                n_fail += 1
                obj = deep_parse(r.get("raw") or "")
                if obj:
                    r["gen"] = obj
                    r["parse_ok"] = True
                    r["repaired"] = "repair2"
                    if r.get("task") == "detect":
                        pos = obj.get("objects") or obj.get("object") or []
                        gtb = r.get("gt_boxes") or []
                        if isinstance(pos, list) and len(pos) == len(gtb):
                            parts = [str(obj.get("spatial", "")).strip()]
                            for b, p in zip(gtb, pos):
                                if isinstance(p, dict):
                                    p = p.get("position") or p.get("desc") or json.dumps(p, ensure_ascii=False)
                                x1, y1, x2, y2 = b["bbox1000"]
                                parts.append("%s：%s → 坐标 [%s, %s, %s, %s]" % (b["label"], str(p).strip(), x1, y1, x2, y2))
                            r["cot"] = "\n".join(x for x in parts if x)
                            r["aligned"] = True
                    else:
                        r["cot"] = str(obj.get("cot") or "").strip() or None
                    n_fix += 1
            recs.append(r)
        total_fail += n_fail; total_fix += n_fix
        print("%-52s parse_fail=%-3d 二次修复=%-3d 剩余=%d" % (os.path.basename(f)[:52], n_fail, n_fix, n_fail - n_fix))
        if n_fix and not dry:
            shutil.copy2(f, f + ".pre_repair2.bak")
            tmp = f + ".tmp2"
            with open(tmp, "w", encoding="utf-8") as fo:
                for r in recs:
                    fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            os.replace(tmp, f)
    print("合计: parse_fail=%d 二次修复=%d 剩余=%d %s" % (total_fail, total_fix, total_fail - total_fix, "(dry-run)" if dry else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
