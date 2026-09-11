#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 parse_metrics.py 产出的 metrics.json 写成 TensorBoard scalar 事件文件。

用途：6006 端口的 TensorBoard 看板。

设计
----
* 每次运行**幂等重写**：先删除 ``--logdir`` 下已有的 ``events.out.tfevents.*``，
  再按当前 metrics.json 全量写入。这样日志增长时不会产生重复点。
* 写到**新的** logdir（默认 ``/root/tf-logs-metrics``），绝不触碰既有的
  ``/root/tf-logs``（6007 端口的 TensorBoard 在用）。
* 需要 torch（``torch.utils.tensorboard.SummaryWriter``），因此用 conda base 的
  Python 运行 —— 那里已有 torch 2.x，无需安装任何东西。

写入的 tag
----------
* ``<run>/train/loss``、``<run>/train/grad_norm``、``<run>/train/learning_rate``
* ``<run>/eval/eval_loss``、``<run>/eval/eval_recon``
* ``<run>/probe/step_px_scale/step{k}``、``<run>/probe/E_px/step{k}_region{r}``
* ``<run>/probe/z_s_within_std``、``<run>/probe/block_cos/step{k}``
* ``<run>/probe/prog_curve_255/step{k}``

用法::

    /root/miniconda3/bin/python emit_tensorboard.py \\
        --metrics data/metrics.json --logdir /root/tf-logs-metrics
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def _purge_events(logdir: str):
    """只删事件文件，保留目录结构与其它内容。"""
    for pat in ("events.out.tfevents.*", "*.tmp"):
        for p in glob.glob(os.path.join(logdir, "**", pat), recursive=True):
            try:
                os.remove(p)
            except OSError:
                pass


def _add(writer, tag, value, step):
    if value is None:
        return
    try:
        writer.add_scalar(tag, float(value), int(step or 0))
    except (TypeError, ValueError):
        pass


def emit(metrics_path: str, logdir: str, verbose: bool = True) -> int:
    with open(metrics_path, "r", encoding="utf-8") as fh:
        bundle = json.load(fh)

    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # pragma: no cover
        print(f"[emit_tensorboard] 无法导入 SummaryWriter: {exc}", file=sys.stderr)
        return 2

    os.makedirs(logdir, exist_ok=True)
    _purge_events(logdir)

    n_scalars = 0
    series = bundle.get("series", {})

    for run, s in series.items():
        run_dir = os.path.join(logdir, run)
        os.makedirs(run_dir, exist_ok=True)
        w = SummaryWriter(log_dir=run_dir)

        for p in s.get("train", []):
            step = p.get("step")
            _add(w, "train/loss", p.get("loss"), step)
            _add(w, "train/grad_norm", p.get("grad_norm"), step)
            _add(w, "train/learning_rate", p.get("learning_rate"), step)
            _add(w, "train/epoch", p.get("epoch"), step)
            n_scalars += 3
        for p in s.get("eval", []):
            step = p.get("step")
            _add(w, "eval/eval_loss", p.get("eval_loss"), step)
            _add(w, "eval/eval_recon", p.get("eval_recon"), step)
            n_scalars += 2
        w.flush()

    # --- 探针：每个 checkpoint 一条曲线 + 用 step 作为 global_step ---
    for run, hist in bundle.get("probe_history", {}).items():
        if not hist:
            continue
        run_dir = os.path.join(logdir, run)
        os.makedirs(run_dir, exist_ok=True)
        w = SummaryWriter(log_dir=run_dir)

        for e in hist:
            step = e.get("step") or 0
            for i, v in enumerate(e.get("step_px_scale") or [], start=1):
                _add(w, f"probe/step_px_scale/step{i}", v, step)
                n_scalars += 1
            for i, v in enumerate(e.get("prog_curve_255") or [], start=1):
                _add(w, f"probe/prog_curve_255/step{i}", v, step)
                n_scalars += 1
            for i, v in enumerate(e.get("z_s_block_cos") or [], start=1):
                _add(w, f"probe/block_cos/step{i}", v, step)
                n_scalars += 1
            _add(w, "probe/z_s_within_std", e.get("z_s_within_std"), step)
            for si, row in enumerate(e.get("E_px") or [], start=1):
                for ri, v in enumerate(row, start=1):
                    _add(w, f"probe/E_px/step{si}_region{ri}", v, step)
                    n_scalars += 1
        w.flush()

    # --- 推理结果：只有单个标量，用 step=total_steps 记录 ---
    total_steps = bundle.get("config", {}).get("total_steps", 0)
    for run, inf in bundle.get("infer", {}).items():
        if not inf:
            continue
        run_dir = os.path.join(logdir, run)
        os.makedirs(run_dir, exist_ok=True)
        w = SummaryWriter(log_dir=run_dir)
        d = inf.get("data", {})
        _add(w, "infer/full_norm_l1", d.get("full_norm_l1"), total_steps)
        _add(w, "infer/full_pixel_l1_255", d.get("full_pixel_l1_255"), total_steps)
        for i, v in enumerate(d.get("step_pixel_l1_255") or [], start=1):
            _add(w, f"infer/step_pixel_l1_255/step{i}", v, total_steps)
            n_scalars += 1
        w.flush()

    if verbose:
        print(f"[emit_tensorboard] logdir={logdir} runs={list(series)} "
              f"scalars≈{n_scalars}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="写 TensorBoard 事件文件")
    ap.add_argument("--metrics", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "metrics.json"))
    ap.add_argument("--logdir", default=os.environ.get("TF_LOGDIR", "/root/tf-logs-metrics"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.metrics):
        print(f"[emit_tensorboard] metrics 不存在: {args.metrics}", file=sys.stderr)
        return 1
    return emit(args.metrics, args.logdir, verbose=not args.quiet)


if __name__ == "__main__":
    sys.exit(main())
