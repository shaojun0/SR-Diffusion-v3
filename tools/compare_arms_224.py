#!/usr/bin/env python3
"""compare_arms_224.py — 224×126 三条臂的统一对照（base / slice1 / tap）

三条臂**同代码、同数据、同步数、同 seed**，差别只在读窗口口径 / 步集切片：

| 臂 | `--slice_start/--slice_end` | 步数 | 读窗口 |
|---|---|---|---|
| base   | 0 / 12（= 不切片, 全量） | 12 | 平方块, z_s 全末层, 宽 [4,5,7,…,23,1] |
| slice1 | **1 / 12** | **11** | 平方块, 首步并掉 t=1 的窗口 ⇒ 宽 [9,7,…,23,1] |
| tap    | 0 / 12 | 12 | **逐层 tap 金字塔**, 宽 [1,3,…,23] |

用法:
    python tools/compare_arms_224.py                      # 打印全表
    python tools/compare_arms_224.py --plot curves.png    # 另存三臂逐 step 图
    python tools/compare_arms_224.py --arm "name=path.json" ...

跨分辨率对照（2026-09-22 新增列: 输入 / bpp 分母 / RD 端点）:
    python tools/compare_arms_224.py \
      --arm "224 base=.../eval_224_d4/test_224_d4_bptt.json" \
      --train1k "224 base=.../eval_224_d4/train1k_224_d4_bptt.json" \
      --arm "896=.../eval_896_d4/test_896_d4_bptt_slice012.json" \
      --train1k "896=.../eval_896_d4/train1k_896_d4_bptt_slice012.json" \
      --title "224x126 vs 896x504" --plot rd.png
    ⚠️ 跨分辨率时**像素 L1 不可直接比**（bpp 分母差 16×）; 同轴可比的是 bpp/bpp–PSNR。
"""
from __future__ import annotations

import argparse
import json
import os

# 服务器默认路径（本仓库分析脚本惯例：口径写死、可被 --arm 覆盖）
EVAL_BASE = "/root/autodl-tmp/cot_l1/eval_224_d4"
EVAL_S1 = "/root/autodl-tmp/cot_l1/eval_224_d4_slice1"
EVAL_TAP = "/root/autodl-tmp/cot_l1/eval_224_d4_tap"

PRESET = [
    ("base  [0:12] 平方块/末层", f"{EVAL_BASE}/test_224_d4_bptt.json",
     f"{EVAL_BASE}/train1k_224_d4_bptt.json"),
    ("slice1[1:12] 平方块/末层", f"{EVAL_S1}/test_224_d4_bptt_slice1.json",
     f"{EVAL_S1}/train1k_224_d4_bptt_slice1.json"),
    ("tap    [0:12] 逐层金字塔", f"{EVAL_TAP}/test_224_d4_bptt_tap.json",
     f"{EVAL_TAP}/train1k_224_d4_bptt_tap.json"),
]


def load(p):
    with open(p) as f:
        return json.load(f)


def best_idx(d):
    return min(range(len(d["step_pixel_l1_255"])),
               key=lambda i: d["step_pixel_l1_255"][i])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[],
                    help="name=test_json_path（可重复; 不给则用内置三臂 preset）")
    ap.add_argument("--train1k", action="append", default=[],
                    help="name=path（可选, 与 --arm 同名配对）")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--title", default=None,
                    help="图标题（默认写死 224x126; 跨分辨率对照时用这个覆盖）")
    a = ap.parse_args()

    if a.arm:
        arms = [(s.split("=", 1)[0], s.split("=", 1)[1], None) for s in a.arm]
        t1 = dict(s.split("=", 1) for s in a.train1k)
        arms = [(n, p, t1.get(n)) for n, p, _ in arms]
    else:
        arms = PRESET

    data = []
    for name, path, t1path in arms:
        if not os.path.isfile(path):
            print(f"[skip] {name}: {path} 不存在")
            continue
        d = load(path)
        t1 = load(t1path) if (t1path and os.path.isfile(t1path)) else None
        data.append((name, d, t1))

    print("## 总量（construction_site test 3,004）\n")
    print("| 臂 | 输入 | bpp 分母 px | 步数 | layer_tap | K | 像素 L1 (0-255) | 最优 PSNR | 最优 t "
          "| 最优 MS-SSIM | PSNR 跨度 | L1 跨度 | train1k L1 | Δ(train−test) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, d, t1 in data:
        i = best_idx(d)
        lt = d["layer_tap"] if "layer_tap" in d else "—(旧产物)"
        span_p = max(d["step_psnr"]) - min(d["step_psnr"])
        span_l = max(d["step_pixel_l1_255"]) - min(d["step_pixel_l1_255"])
        t1s = f"{t1['full_pixel_l1_255']:.2f}" if t1 else "—"
        dlt = (f"{t1['full_pixel_l1_255'] - d['full_pixel_l1_255']:+.2f}"
               if t1 else "—")
        inp = d.get("input")
        inp = (f"{inp[0]}×{inp[1]}" if isinstance(inp, (list, tuple)) and len(inp) == 2
               else "—")
        print(f"| {name} | {inp} | {d.get('bpp_px', '—')} "
              f"| {len(d['decoder_steps'])} | {lt} | {d['num_specials']} "
              f"| {d['full_pixel_l1_255']:.2f} ± {d['full_pixel_std_255']:.2f} "
              f"| **{max(d['step_psnr']):.2f}** | {d['decoder_steps'][i]} "
              f"| {d['step_ms_ssim'][i]:.4f} | {span_p:.2f} dB | {span_l:.2f} px "
              f"| {t1s} | {dlt} |")

    # 端点（RD 曲线两端 + 最优）：跨分辨率时**L1 不可直接比**, bpp/PSNR 才可同轴
    print("\n## RD 端点（最左 / 最优 / 最右; bpp = β=1 估计, 分母见上表）\n")
    print("| 臂 | 首步 t / tokens / bpp / PSNR | 最优 t / bpp / PSNR | 末步 t / tokens / bpp / PSNR |")
    print("|---|---|---|---|")
    for name, d, _ in data:
        i = best_idx(d)
        t = d["decoder_steps"]
        tk, bp, ps = d["step_tokens"], d["step_bpp_beta1"], d["step_psnr"]
        print(f"| {name} | t={t[0]} / {tk[0]} / {bp[0]:.3f} / {ps[0]:.2f} "
              f"| t={t[i]} / {bp[i]:.3f} / **{ps[i]:.2f}** "
              f"| t={t[-1]} / {tk[-1]} / {bp[-1]:.3f} / {ps[-1]:.2f} |")

    # 逐 step 对照（按 t 对齐; 各臂步集可能不同）
    all_t = sorted({t for _, d, _ in data for t in d["decoder_steps"]})
    print("\n## 逐 step 对照（按 t 对齐; `—` = 该臂没有这一步）\n")
    hdr = "| t | " + " | ".join(f"{n.split()[0]} L1 / PSNR" for n, _, _ in data) + " |"
    print(hdr)
    print("|---|" + "---|" * len(data))
    for t in all_t:
        cells = []
        for _, d, _ in data:
            if t in d["decoder_steps"]:
                i = d["decoder_steps"].index(t)
                cells.append(f"{d['step_pixel_l1_255'][i]:.3f} / {d['step_psnr'][i]:.2f}")
            else:
                cells.append("—")
        print(f"| {t} | " + " | ".join(cells) + " |")

    # token 数（bpp 轴）对照
    print("\n## 同一步的 token 数（bpp 轴; tap 不读 z_cls ⇒ 偏移 0）\n")
    print("| 臂 | " + " | ".join(str(t) for t in all_t) + " |")
    print("|---|" + "---|" * len(all_t))
    for name, d, _ in data:
        tok = {t: d["step_tokens"][i] for i, t in enumerate(d["decoder_steps"])}
        print(f"| {name} | " + " | ".join(str(tok.get(t, "—")) for t in all_t) + " |")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 服务器 matplotlib 无 CJK 字体 ⇒ 图内标签一律 ASCII（中文名只用于终端表格）
        def _ascii(name):
            short = name.split()[0]
            if "金字塔" in name or "tap" in short:
                return f"{short} layer pyramid"
            return f"{short} square/deep"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.8))
        colors = ["tab:gray", "tab:blue", "tab:red", "tab:green"]
        for k, (name, d, _) in enumerate(data):
            c = colors[k % len(colors)]
            lab = _ascii(name)
            ax1.plot(d["step_bpp_beta1"], d["step_psnr"], "-o", ms=4, color=c,
                     label=f"{lab}  best {max(d['step_psnr']):.2f} dB")
            steps_i = list(range(1, len(d["decoder_steps"]) + 1))
            ax2.plot(steps_i, d["step_psnr"], "-o", ms=4, color=c, label=f"{lab} PSNR")
            ax2.plot(steps_i, d["step_pixel_l1_255"], "--s", ms=3, color=c, alpha=.55,
                     label=f"{lab} L1")
        ax1.set_xscale("log")
        ax1.set_xlabel("bpp (beta=1 estimated)")
        ax1.set_ylabel("PSNR (dB)")
        ax1.set_title(a.title or
                      "224x126 RD: base / slice1 / tap (construction_site test n=3004)")
        ax1.grid(alpha=.3, which="both")
        ax1.legend(fontsize=7)
        ax2.set_xlabel("decoder step index (1-based, within each arm's own step set)")
        ax2.set_ylabel("PSNR (dB) / pixel L1 (0-255)")
        ax2.set_title("progressive curves (solid=PSNR, dashed=L1)")
        ax2.grid(alpha=.3)
        ax2.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=150)
        print(f"\n[plot] {a.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
