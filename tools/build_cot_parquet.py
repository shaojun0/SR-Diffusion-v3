#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_cot_parquet.py — 把 CoT 全量图片目录构建成 Phase-1 训练用的 parquet。

背景
----
原始 Phase-1 实验用 `construction_site` parquet（= aswin 10,013 行，7009 train /
3004 test，列 image struct{bytes,path} + image_caption + violations）。本脚本把
CoT 数据线的**图片部分**（服务器 `/root/autodl-tmp/cot_full/<dataset>/images/*.jpg`，
7 数据集 75,717 张）打成**同 schema** 的 parquet，使 `train_v2.py` / `data_v2.py`
零改动即可训练（"更大的数据集"）。

划分（关键：与原始基线做无泄漏对比）
------------------------------------
`cot_full/aswin`（10,013 张）与原始 `construction_site`（7009 train / 3004 test）
是**同一批图、不同文件名**。若纯随机切分，会把原始 test 图混进新 train，导致
"用更大数据集训练"的模型在原始 test 上虚高。因此本脚本支持 `--anchor-test`：
把 anchor parquet（construction_site test）里出现过的图**一律强制放进 test**，
其余图按数据集分层随机切分（默认 test 10%，seed 固定）。

效果：
  · CoT test = 原始 construction_site test 的 3,004 张（内容一致）+ 其余 6 数据集的 10%
  · CoT train 与 construction_site test 交集 = 0 ⇒ 与原始基线同测试集可比。

输出
----
<dst>/train-00000-of-000NN.parquet
<dst>/test-00000-of-000NN.parquet
列：image struct<bytes: binary, path: string>, image_caption: large_string,
    violations: large_string（后两列为空串，纯重建模式不使用）。

用法
----
python build_cot_parquet.py \
    --src /root/autodl-tmp/cot_full \
    --dst /root/autodl-tmp/cot_l1/parquet \
    --anchor-test '/root/autodl-tmp/construction_site/test-*.parquet' \
    --test-frac 0.1 --seed 42
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import random
import shutil
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

EXTS = (".jpg", ".jpeg", ".png", ".webp")

SCHEMA = pa.schema(
    [
        ("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ("image_caption", pa.large_string()),
        ("violations", pa.large_string()),
    ]
)


def list_images(root: Path) -> dict[str, list[Path]]:
    """返回 {dataset: [image...]}，按数据集分层。"""
    out: dict[str, list[Path]] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = [
            p
            for p in sorted(d.rglob("*"))
            if p.is_file() and p.suffix.lower() in EXTS
        ]
        if imgs:
            out[d.name] = imgs
    return out


def md5_set(pattern: str, batch: int = 128) -> set[str]:
    """anchor parquet 中 image.bytes 的 md5 集合（流式，避免整表进内存）。"""
    s: set[str] = set()
    files = sorted(glob.glob(pattern))
    for f in files:
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=batch, columns=["image"]):
            for p in b.column("image").to_pylist():
                data = p["bytes"] if isinstance(p, dict) else p
                s.add(hashlib.md5(data).hexdigest())
    print(f"[anchor] {len(files)} 个 parquet → {len(s)} 个 md5", flush=True)
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/root/autodl-tmp/cot_full")
    ap.add_argument("--dst", default="/root/autodl-tmp/cot_l1/parquet")
    ap.add_argument(
        "--anchor-test",
        default="",
        help="glob: 其中的图片（按内容 md5 匹配）强制进 test；空串=纯随机切分",
    )
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard-size", type=int, default=5000)
    ap.add_argument(
        "--max-shard-bytes",
        type=float,
        default=1.5e9,
        help="单 shard 的 image bytes 上限（bytes）。超过会触发 pyarrow binary "
        "int32 offset 的 2GB 分段，struct 列变成 chunked array 后 HF datasets "
        "parquet reader 直接报 'Nested data conversions not implemented'。",
    )
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    assert src.is_dir(), f"src 不存在: {src}"
    assert 0.0 < args.test_frac < 0.5, "test-frac 取 (0,0.5)"

    anchor = md5_set(args.anchor_test) if args.anchor_test else set()

    per_ds = list_images(src)
    total = sum(len(v) for v in per_ds.values())
    print(f"[scan] {len(per_ds)} 数据集 / {total} 张图片", flush=True)
    for k, v in per_ds.items():
        print(f"        {k}: {len(v)}")

    tmp = dst.parent / (dst.name + ".building")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "test": 0}
    anchor_hit = {"train": 0, "test": 0}
    ds_counts = {k: {"train": 0, "test": 0} for k in per_ds}
    shard_sizes = {"train": 0, "test": 0}
    buf: dict[str, list[dict]] = {"train": [], "test": []}
    buf_bytes = {"train": 0, "test": 0}
    t0 = time.time()

    def flush(split: str) -> None:
        if not buf[split]:
            return
        idx = shard_sizes[split]
        path = tmp / f"{split}-{idx:05d}-of-PENDING.parquet"
        table = pa.Table.from_pylist(buf[split], schema=SCHEMA)
        # JPEG 已压缩，zstd 只烧 CPU 不提收益 ⇒ 用快的 snappy。
        pq.write_table(table, path, compression="snappy")
        print(
            f"[write] {path.name}  {len(buf[split])} rows  "
            f"{buf_bytes[split] / 1e9:.2f} GB",
            flush=True,
        )
        buf[split] = []
        buf_bytes[split] = 0
        shard_sizes[split] += 1

    # 先哈希一遍（只留 md5，不留 bytes），判定哪些数据集被 anchor 覆盖。
    # 被 anchor 覆盖的数据集：test = anchor 命中，其余全进 train（复用原始切分）；
    # 未被覆盖的数据集：按 test_frac 随机切分。
    print("[hash] 计算图片 content-md5（判定 anchor 覆盖）…", flush=True)
    hashes: dict[str, dict[Path, str]] = {}
    for ds, imgs in per_ds.items():
        hashes[ds] = {p: hashlib.md5(p.read_bytes()).hexdigest() for p in imgs}
    anchored = {
        ds: any(h in anchor for h in hashes[ds].values()) for ds in per_ds
    }
    print(f"[hash] anchor 覆盖的数据集: {[d for d, v in anchored.items() if v]}")

    for ds, imgs in per_ds.items():
        rng = random.Random(args.seed)
        order = list(imgs)
        rng.shuffle(order)
        n_test = max(1, int(round(len(order) * args.test_frac)))
        random_test = set(order[:n_test])
        for p in order:
            data = p.read_bytes()
            if anchored[ds]:
                split = "test" if hashes[ds][p] in anchor else "train"
                if split == "test":
                    anchor_hit["test"] += 1
            else:
                split = "test" if p in random_test else "train"
            buf[split].append(
                {
                    "image": {"bytes": data, "path": str(p)},
                    "image_caption": "",
                    "violations": "",
                }
            )
            buf_bytes[split] += len(data)
            counts[split] += 1
            ds_counts[ds][split] += 1
            if (
                len(buf[split]) >= args.shard_size
                or buf_bytes[split] >= args.max_shard_bytes
            ):
                flush(split)
        flush("train")
        flush("test")
        print(
            f"[split] {ds}: train={ds_counts[ds]['train']} "
            f"test={ds_counts[ds]['test']}",
            flush=True,
        )

    for split in ("train", "test"):
        n = shard_sizes[split]
        for i in range(n):
            old = tmp / f"{split}-{i:05d}-of-PENDING.parquet"
            new = tmp / f"{split}-{i:05d}-of-{n:05d}.parquet"
            old.rename(new)
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)

    print("\n[summary]")
    for ds in per_ds:
        print(f"  {ds}: train={ds_counts[ds]['train']} test={ds_counts[ds]['test']}")
    print(f"  TOTAL: train={counts['train']} test={counts['test']}")
    if anchor:
        print(
            f"  anchor(test) 命中: {anchor_hit['test']} / {len(anchor)}"
            f"  （未命中=该图不在 cot_full 或已在其它 split）"
        )
    print(f"  shards: train={shard_sizes['train']} test={shard_sizes['test']}")
    print(f"  dst = {dst}   ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
