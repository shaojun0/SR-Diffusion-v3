#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""summarize_cot_l1.py — 汇总 eval_cot_l1.sh 的评估结果，出对比表。

用法:
    python summarize_cot_l1.py [--eval /root/autodl-tmp/cot_l1/eval]

读取:
    <eval>/bigdata/cot_l1_test_all.json
    <eval>/bigdata/construction_site_test.json
    <eval>/bigdata/by_dataset/<ds>.json
    <eval>/baseline/  （同样结构）

输出: 全量/逐数据集/construction_site 三张对比表（像素 L1 0-255，越低越好）。
"""
from __future__ import annotations

import argparse
import json
import os

DATASETS = [
    "aswin00000__ConstructionSiteCleanedDataSet",
    "baizhanquan__FireDetectionDataset",
    "chandrabhuma__multi_building_defect_vqa",
    "hayden-yuma__roadwork",
    "hf-vision__hardhat",
    "iluvvatar__wood_surface_defects",
    "kevincluo__structure_wildfire_damage_classification",
]
MODELS = [("baseline", "construction_site 基线(7009 训练)"), ("bigdata", "CoT 全量(68147 训练)")]


def load(path: str):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def px(d):
    return None if d is None else d.get("full_pixel_l1_255")


def line(name: str, a, b):
    sa = "—" if a is None else f"{a:.2f}"
    sb = "—" if b is None else f"{b:.2f}"
    delta = "—" if (a is None or b is None) else f"{b - a:+.2f}"
    print(f"  {name:<46} {sa:>8} {sb:>8} {delta:>9}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="/root/autodl-tmp/cot_l1/eval")
    args = ap.parse_args()

    has_any = False
    print("\n== 全量 test（像素 L1 0-255，越低越好）==")
    print(f"  {'测试集':<46} {'基线':>8} {'CoT全量':>8} {'Δ(新-旧)':>9}")
    for tag, path in [
        ("CoT 全量 test (7,570 张)", os.path.join(args.eval, "%s/cot_l1_test_all.json")),
        ("construction_site test (3,004 张)", os.path.join(args.eval, "%s/construction_site_test.json")),
    ]:
        a = px(load(path % "baseline"))
        b = px(load(path % "bigdata"))
        if a is not None or b is not None:
            has_any = True
        line(tag, a, b)

    print("\n== 逐数据集 test（像素 L1 0-255）==")
    print(f"  {'数据集（CoT test 分片）':<46} {'基线':>8} {'CoT全量':>8} {'Δ(新-旧)':>9}")
    for ds in DATASETS:
        a = px(load(os.path.join(args.eval, "baseline/by_dataset", ds + ".json")))
        b = px(load(os.path.join(args.eval, "bigdata/by_dataset", ds + ".json")))
        if a is not None or b is not None:
            has_any = True
        line(ds[:44], a, b)

    # 非 aswin 小计（对两个模型都无泄漏的子集）
    print("\n== 非 aswin 子集（基线从未见过这 6 个数据集）==")
    for key, name in [("baseline", "基线"), ("bigdata", "CoT全量")]:
        vals, ns = [], []
        for ds in DATASETS[1:]:
            d = load(os.path.join(args.eval, key, "by_dataset", ds + ".json"))
            if d is not None:
                vals.append(d["full_pixel_l1_255"] * d["n"])
                ns.append(d["n"])
        if vals:
            print(f"  {name:<10} 加权像素 L1 = {sum(vals) / sum(ns):.2f}  (n={sum(ns)})")

    if not has_any:
        print("\n[warn] 还没有任何评估结果（训练可能仍在跑）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
