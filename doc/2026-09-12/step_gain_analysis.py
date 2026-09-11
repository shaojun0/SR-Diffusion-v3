#!/usr/bin/env python3
"""step_gain_analysis.py — 各采样 step 增益分析 (baseline vs 加宽模型 vs 各 checkpoint)。

输入 = `probe_step_collapse.py` 的 JSON（step 1/4/9/16/25 的累积重建误差与增量幅度），
可选 `infer_v2_test.py` 的 JSON（test 全量 3004 张的渐进曲线）。

对每个模型给出:
  cum_px[t]   第 t 步「累积输出」的整图 0-255 像素 L1（越小越好）
  marg[t]     第 t 步的**边际增益** = cum_px[t-1] - cum_px[t]（>0 = 加这一步后误差下降）
  scale[t]    第 t 步**原始增量**幅度 mean|pixel_head(Y_t)|（该步"发声"强度）
  rel[t]      scale[t] / scale[1]

用法:
  python step_gain_analysis.py --md out.md --json out.json \
    --probe baseline=probe_base.json --probe stack8x=probe_s8.json \
    --glob  "stack8x@ckpt=probe_stack8x_slice05_step*.json" \
    --infer stack8x=infer_test.json
"""
import argparse
import glob
import json
import os
import re


def load(p):
    with open(p) as f:
        return json.load(f)


def margins(cum):
    """边际增益: 第 i 步 (i>=1) = cum[i-1]-cum[i]; 第 0 步无前序 -> None。"""
    return [None] + [cum[i - 1] - cum[i] for i in range(1, len(cum))]


def ckpt_step(path):
    m = re.search(r"step(\d+)\.json$", os.path.basename(path))
    return int(m.group(1)) if m else 0


def fmt(v, nd=4):
    return "—" if v is None else f"{v:+.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="append", default=[], help="tag=path (可重复)")
    ap.add_argument("--glob", action="append", default=[],
                    help="tag=pattern (按 step 数值排序; tag 仅作前缀, 每个文件用文件名作模型名)")
    ap.add_argument("--infer", action="append", default=[], help="tag=path (可重复)")
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    probes, infers = {}, {}
    for spec in a.probe:
        tag, p = spec.split("=", 1)
        if os.path.exists(p):
            probes[tag] = load(p)
        else:
            print(f"[warn] probe 不存在: {p}")
    for spec in a.glob:
        prefix, pat = spec.split("=", 1)
        for f in sorted(glob.glob(pat), key=ckpt_step):
            tag = f"{prefix}{ckpt_step(f)}"
            probes[tag] = load(f)
    for spec in a.infer:
        tag, p = spec.split("=", 1)
        if os.path.exists(p):
            infers[tag] = load(p)
        else:
            print(f"[warn] infer 不存在: {p}")

    lines = []
    out = {"probe": {}, "infer": {}}

    def emit(s=""):
        lines.append(s)
        print(s)

    emit("# 各采样 step 增益分析")
    emit()
    emit("口径: `prog_curve_255` = 第 t 步**累积**输出的整图 0-255 像素 L1（probe 512 张）；"
         "`marg` = 上一步累积 − 本步累积（正 = 本步带来误差下降）；"
         "`scale` = 第 t 步**原始增量** mean|pixel_head(Y_t)|（未累加）。")
    emit()
    emit("> 注意: step1 无前序累积，其 `marg` 不可定义（表中记 —）；step1 的贡献用 `scale` 度量。")
    emit()

    # ── 1. 逐模型明细 ──
    emit("## 1. 各模型逐 step 明细")
    emit()
    for tag, d in probes.items():
        steps = d.get("steps") or list(range(1, len(d.get("prog_curve_255", [])) + 1))
        cum = d["prog_curve_255"]
        marg = margins(cum)
        scale = d.get("step_px_scale", [float("nan")] * len(cum))
        rel = [s / scale[0] if scale and scale[0] else float("nan") for s in scale]
        cfg = (f"depth={d.get('decoder_depth')}, stack_dim={d.get('stack_dim')}, "
               f"heads={d.get('heads')}, dropout={d.get('decoder_dropout')}, n={d.get('n_img')}")
        emit(f"### `{tag}`  ({cfg})")
        emit()
        emit("| 采样步 t | 累积 L1 (0-255) | 边际增益 Δ | 增量幅度 scale | scale/scale₁ |")
        emit("|---|---|---|---|---|")
        for i, t in enumerate(steps):
            emit(f"| {t} | {cum[i]:.4f} | {fmt(marg[i])} | {scale[i]:.4f} | {rel[i]:.4f} |")
        tot = cum[0] - cum[-1]
        gain_sum = sum(m for m in marg if m is not None)
        emit()
        emit(f"- step1→末步 总落差 = **{tot:+.4f}**；后 4 步边际增益之和 = **{gain_sum:+.4f}**")
        if abs(tot) > 1e-9:
            emit(f"- 后 4 步占 step1→末步 落差的比例 = {gain_sum / tot * 100:+.1f}%")
        emit()
        out["probe"][tag] = {
            "steps": steps, "cum_px": cum, "marg_px": marg,
            "step_px_scale": scale, "scale_rel": rel,
            "total_drop_step1_to_last": tot, "post_step1_gain_sum": gain_sum,
            "config": {k: d.get(k) for k in
                       ("decoder_depth", "stack_dim", "heads", "decoder_dropout", "n_img")},
        }

    # ── 2. 跨模型对比 ──
    if len(probes) > 1:
        emit("## 2. 跨模型对比")
        emit()
        base_tag = next(iter(probes))
        steps = probes[base_tag].get("steps") or list(range(1, len(probes[base_tag]["prog_curve_255"]) + 1))
        emit("### 2.1 累积 L1 (0-255, 越小越好)")
        emit()
        emit("| 采样步 t | " + " | ".join(f"`{t}`" for t in probes) + " |")
        emit("|---" * (len(probes) + 1) + "|")
        for i, t in enumerate(steps):
            row = []
            for tag, d in probes.items():
                c = d["prog_curve_255"]
                row.append(f"{c[i]:.4f}" if i < len(c) else "—")
            emit(f"| {t} | " + " | ".join(row) + " |")
        emit()
        emit("### 2.2 各 step 边际增益 Δ (正 = 该步降低误差)")
        emit()
        emit("| 采样步 t | " + " | ".join(f"`{t}`" for t in probes) + " |")
        emit("|---" * (len(probes) + 1) + "|")
        for i, t in enumerate(steps):
            row = []
            for tag, d in probes.items():
                m = margins(d["prog_curve_255"])
                row.append(fmt(m[i]) if i < len(m) else "—")
            emit(f"| {t} | " + " | ".join(row) + " |")
        emit()
        emit("### 2.3 各 step 增量幅度 scale (该步『发声』强度)")
        emit()
        emit("| 采样步 t | " + " | ".join(f"`{t}`" for t in probes) + " |")
        emit("|---" * (len(probes) + 1) + "|")
        for i, t in enumerate(steps):
            row = []
            for tag, d in probes.items():
                s = d.get("step_px_scale", [])
                row.append(f"{s[i]:.4f}" if i < len(s) else "—")
            emit(f"| {t} | " + " | ".join(row) + " |")
        emit()

    # ── 3. 全量 test 渐进曲线 (infer_v2_test.py, 3004 张) ──
    if infers:
        emit("## 3. 全量 test (3004 张) 渐进曲线边际增益")
        emit()
        for tag, d in infers.items():
            cur = d.get("step_pixel_l1_255") or []
            steps = d.get("decoder_steps") or list(range(1, len(cur) + 1))
            m = margins(cur)
            emit(f"### `{tag}` — full_norm_l1 = {d.get('full_norm_l1')}, "
                 f"full_pixel_l1_255 = {d.get('full_pixel_l1_255')}")
            emit()
            emit("| 采样步 t | 累积像素 L1 | 边际增益 Δ |")
            emit("|---|---|---|")
            for i, t in enumerate(steps):
                emit(f"| {t} | {cur[i]:.4f} | {fmt(m[i])} |")
            emit()
            out["infer"][tag] = {"steps": steps, "cum_px": cur, "marg_px": m,
                                 "full_norm_l1": d.get("full_norm_l1"),
                                 "full_pixel_l1_255": d.get("full_pixel_l1_255")}
        emit("### 3.x 跨模型边际增益对比 (0-255)")
        emit()
        first = next(iter(infers))
        steps = infers[first].get("decoder_steps") or []
        emit("| 采样步 t | " + " | ".join(f"`{t}`" for t in infers) + " |")
        emit("|---" * (len(infers) + 1) + "|")
        for i, t in enumerate(steps):
            row = []
            for tag, d in infers.items():
                m = margins(d.get("step_pixel_l1_255") or [])
                row.append(fmt(m[i]) if i < len(m) else "—")
            emit(f"| {t} | " + " | ".join(row) + " |")
        emit()

    if a.md:
        with open(a.md, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[save] {a.md}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[save] {a.json}")


if __name__ == "__main__":
    main()
