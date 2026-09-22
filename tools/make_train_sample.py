#!/usr/bin/env python3
"""make_train_sample.py — 从 train-*.parquet 随机抽 N 张, 写成 test-*.parquet

动机（2026-09-22 用户口径）：测试时"顺便在训练集里抽部分数据评测" ⇒ 需要看
train/test 差距（过拟合程度）。必须**可复现**（固定 seed），而不是"取前 N 张"。

用法:
    python tools/make_train_sample.py --data_dir /root/autodl-tmp/construction_site \
        --out_dir /root/autodl-tmp/construction_site_train1k --n 1000 --seed 42
"""
from __future__ import annotations

import argparse
import glob
import os

from datasets import load_dataset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.data_dir, "train-*.parquet")))
    assert files, f"无 train-*.parquet in {a.data_dir}"
    ds = load_dataset("parquet", data_files=files, split="train")
    print(f"[data] train 全量 {len(ds)} 张 from {len(files)} shard")
    n = min(a.n, len(ds))
    sub = ds.shuffle(seed=a.seed).select(range(n))
    os.makedirs(a.out_dir, exist_ok=True)
    # 命名成 test-*.parquet ⇒ infer_v2_test.py 的 glob 直接可用
    sub.to_parquet(os.path.join(a.out_dir, "test-00000-of-00001.parquet"))
    print(f"[ok] 抽 {n} 张(seed={a.seed}) → {a.out_dir}/test-00000-of-00001.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
