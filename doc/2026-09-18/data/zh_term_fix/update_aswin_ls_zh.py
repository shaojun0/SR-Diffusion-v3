#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""update_aswin_ls_zh.py — 把修正后的中文 meta_html 就地写回 Label Studio 已有任务。

背景：`seed_aswin_ls.py` 在导入时把 `reason_zh` / `caption_zh` 渲染进每行的
`data.meta_html`。译文修正后如果 `--recreate` 重灌，会**连同人工标注一起删掉**；
本脚本改为按 `row_id` 匹配、只 `PATCH` 任务的 `data` 字段，
**不动 predictions / annotations**（实测标注逐条不变）。

用法：
  LS_PASSWORD='...' python3 update_aswin_ls_zh.py --dry-run   # 只看会改多少条
  LS_PASSWORD='...' python3 update_aswin_ls_zh.py             # 全量就地更新
  LS_PASSWORD='...' python3 update_aswin_ls_zh.py --limit 5   # 先小批验证
"""
import argparse
import json
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import seed_aswin_ls as s  # noqa: E402  复用 login/api/load_clauses/build_tasks


def fetch_all_tasks(pid):
    out, page = [], 1
    while True:
        st, t = s.api("/tasks/?project=%d&page=%d&page_size=100" % (pid, page))
        if not isinstance(t, dict):
            break
        batch = t.get("tasks") or []
        if not batch:
            break
        out.extend(batch)
        page += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 条（0=全部）")
    a = ap.parse_args()
    if not s.PASSWORD:
        sys.exit("need LS_PASSWORD")

    if not s.login():
        sys.exit("登录失败")
    st, projects = s.api("/projects/")
    items = projects.get("results", []) if isinstance(projects, dict) else projects
    pid = next((p["id"] for p in items if p.get("title") == s.PROJECT_TITLE), None)
    if not pid:
        sys.exit("找不到项目: %s" % s.PROJECT_TITLE)
    print("project id=%d" % pid)

    clauses, aswin = s.load_clauses()
    want = {t["data"]["row_id"]: t["data"] for t in s.build_tasks(clauses, aswin)}
    existing = fetch_all_tasks(pid)
    print("LS 任务=%d | manifest 行=%d" % (len(existing), len(want)))

    n_changed = n_same = n_missing = n_ok = n_fail = 0
    ann_before_total = 0
    for task in existing:
        rid = task["data"].get("row_id")
        ann_before_total += task.get("total_annotations") or 0
        new_data = want.get(rid)
        if not new_data:
            n_missing += 1
            continue
        if task["data"].get("meta_html") == new_data.get("meta_html"):
            n_same += 1
            continue
        n_changed += 1
        if a.dry_run:
            continue
        if a.limit and n_ok >= a.limit:
            continue
        st, resp = s.api("/tasks/%d/" % task["id"], "PATCH", {"data": new_data})
        if st == 200:
            n_ok += 1
        else:
            n_fail += 1
            print("  PATCH task %s failed: %s %s" % (task["id"], st, str(resp)[:200]))
        if (n_ok + n_fail) % 200 == 0:
            print("  ...patched=%d failed=%d" % (n_ok, n_fail))

    print("需更新=%d | 已一致=%d | 无匹配=%d" % (n_changed, n_same, n_missing))
    if not a.dry_run:
        print("PATCH 成功=%d 失败=%d" % (n_ok, n_fail))
    # 复核：标注总数不应变化
    st, p = s.api("/projects/%d/" % pid)
    if isinstance(p, dict):
        print("项目标注数 before=%d after=%d（应相等，本脚本不改标注）"
              % (ann_before_total, p.get("total_annotations_number") or 0))


if __name__ == "__main__":
    main()
