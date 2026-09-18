#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zh_fix.py — aswin 工地隐患数据「中文译文」的确定性术语/缩写修正。

依据用户 2026-09-18 的三条口径（对施工安全语料的中文质检规则）：

  R1 语义：`not using PPE` 一类否定句不得译成「不得使用…」。
          例：`Person on the left not using PPE.` 原译「左侧人员不得使用PPE。」
          → 语义反转，应为「未使用个人防护装备」。
  R2 术语：`hard hat` 统一译「安全帽」，不得出现「硬帽」。
  R3 缩写：任何缩写首次出现须在后紧跟的括号内给出全称；中文语境下全称可为英文。
          例：`PPE` → `个人防护装备（PPE，Personal Protective Equipment）`。

同一批数据里与 R1/R2 同源的少数残留英文/错译（formwork、hard hats）一并修正，
见 SPECIAL 表；其余语义错误在报告中登记、不在此脚本内静默改动。

用法：
  python3 zh_fix.py data/aswin_hazard_manifest_zh.jsonl [--in-place] [--report]

作为库：
  from zh_fix import fix_text, fix_row
"""
import argparse
import json
import os
import re
import sys

# --------------------------------------------------------------------------
# 1) 整句精确修正（源句冻结，可精确匹配；必须最先执行）
#    左侧「EN 原文」为对照，便于人工核对。
# --------------------------------------------------------------------------
SPECIAL = {
    # R1：语义反转（EN: Person on the left not using PPE.）
    "左侧人员不得使用PPE。":
        "左侧人员未使用个人防护装备（PPE，Personal Protective Equipment）。",
    # 与 R2 同源的残留英文（EN: The workers in the formwork are not wearing hard hats.）
    "formwork 的工人未佩戴安全帽。":
        "模板（formwork）上的工人未佩戴安全帽。",
    # EN: The two workers on the formworks and another worker on the earthen road
    #     in the background are not wearing hard hats.
    "两名工人在Form沃克和另一名工人在厄尔顿路上，背景中均未佩戴安全帽。":
        "两名工人在模板（formwork）上，另一名工人在背景中的土路上，均未佩戴安全帽。",
    # EN: One worker on the left and two workers sitting on the right
    #     are not wearing hard hats.  （原译把 sitting 误作「为…的」）
    "左侧的一名工人和右侧的两名为“hard hats”的工人均未佩戴安全帽。":
        "左侧的一名工人和右侧坐着的两名工人均未佩戴安全帽。",
}

# --------------------------------------------------------------------------
# 2) 逐词修正（幂等）
# --------------------------------------------------------------------------
SIMPLE = [
    # R2：hard hat → 安全帽。先处理「硬帽子」，避免出现「安全帽子」。
    ("硬帽子", "安全帽"),
    ("硬帽", "安全帽"),
    # 同源拼写/断词错误（品牌名保留拉丁写法，仅纠正明显错误）
    ("KOBEL CO", "Kobelco"),
    ("DOOSan", "Doosan"),
    # 残留英文（若 SPECIAL 未覆盖的变体）
    ("formwork", "模板（formwork）"),
]

# --------------------------------------------------------------------------
# 3) R3 缩写表：缩写 → 首次展开式（括号内为全称，全称可为英文）
#    若文本中已存在「（缩写，」形式则跳过，保证幂等。
# --------------------------------------------------------------------------
ABBREV = {
    "PPE": "个人防护装备（PPE，Personal Protective Equipment）",
    "SUV": "运动型多用途汽车（SUV，Sport Utility Vehicle）",
    "XCMG": "徐工集团（XCMG，Xuzhou Construction Machinery Group）",
}

_ABBREV_RE = {
    k: re.compile(r"(?<![A-Za-z0-9])" + re.escape(k) + r"(?![A-Za-z0-9])")
    for k in ABBREV
}


def fix_text(s):
    """对单个中文串做确定性修正；None/空串原样返回。"""
    if not s:
        return s
    if s in SPECIAL:
        return SPECIAL[s]
    for a, b in SIMPLE:
        if a in s:
            # 已展开形式（如「模板（formwork）」）不重复包裹
            if a == "formwork" and "（formwork）" in s:
                continue
            s = s.replace(a, b)
    for tok, full in ABBREV.items():
        if ("（%s，" % tok) in s or ("(%s," % tok) in s:
            continue
        s = _ABBREV_RE[tok].sub(full, s)
    return s


def fix_row(r):
    """就地修正一行的 caption_zh 与 boxes[].reason_zh。"""
    if r.get("caption_zh"):
        r["caption_zh"] = fix_text(r["caption_zh"])
    for b in r.get("boxes") or []:
        if b.get("reason_zh"):
            b["reason_zh"] = fix_text(b["reason_zh"])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", default=None, help="输出路径（默认覆写 src）")
    ap.add_argument("--in-place", action="store_true", help="等价于 --out src")
    ap.add_argument("--report", action="store_true", help="打印所有改动对，不写文件")
    a = ap.parse_args()

    dst = a.src if (a.in_place or not a.out) else a.out

    n = n_cap = n_reason = 0
    changes = []
    out_lines = []
    with open(a.src, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            before = (r.get("caption_zh"), [b.get("reason_zh") for b in r.get("boxes") or []])
            fix_row(r)
            after = (r.get("caption_zh"), [b.get("reason_zh") for b in r.get("boxes") or []])
            if before[0] != after[0]:
                n_cap += 1
                changes.append(("caption", before[0], after[0]))
            for i, (x, y) in enumerate(zip(before[1], after[1])):
                if x != y:
                    n_reason += 1
                    changes.append(("reason[%d]" % i, x, y))
            out_lines.append(json.dumps(r, ensure_ascii=False))
            n += 1

    print("rows=%d | caption_zh 改动=%d | reason_zh 改动=%d | 合计改动=%d"
          % (n, n_cap, n_reason, len(changes)))
    if a.report:
        for k, x, y in changes:
            print("--- %s" % k)
            print("  旧: %s" % x)
            print("  新: %s" % y)
        return
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")
    print("written:", os.path.abspath(dst))


if __name__ == "__main__":
    main()
