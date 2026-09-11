#!/usr/bin/env python3
"""SR-Diffusion-v3: self.stack 2x vs blockdiag slice[0:5] 基线 A/B 对比.

输入 = infer_v2_test.py 产出的两个 JSON（同一脚本、同一 test 全量 3004 张），
可选训练 log（解析 eval_recon 曲线）。输出 markdown 表格 + 结论。

用法:
  python ab_compare_stack2x.py \
    --new  output/phase1_v2_stack2x_slice05/infer_test.json \
    --base output/baseline_blockdiag_slice05_infer_test.json \
    --new_log  /root/train_logs/stack2x_slice05.log \
    --base_log /root/train_logs/blockdiag_slice05.log --md AB_stack2x.md
"""
import argparse
import json
import re


def load_json(p):
    with open(p) as f:
        return json.load(f)


def parse_eval_log(path):
    """从训练 log 抓 (step, eval_recon) 序列（Trainer 的 {'eval_recon': ...}）。"""
    if not path:
        return []
    try:
        raw = open(path, errors="ignore").read().replace("\r", "\n")
    except OSError:
        return []
    out = []
    # 形如  2190/8760 [50:57<...]{'eval_loss': '0.4842', 'eval_recon': '0.486', ...}
    for m in re.finditer(
            r"(\d+)/\d+ \[[^]]*\][^\n]*?'eval_recon': '([0-9.]+)'", raw):
        out.append((int(m.group(1)), float(m.group(2))))
    # 末尾无进度条的那条（final eval）
    for m in re.finditer(r"\{'eval_loss': '[0-9.]+', 'eval_recon': '([0-9.]+)'", raw):
        v = float(m.group(1))
        if not out or out[-1][1] != v:
            out.append((None, v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True, help="新模型 infer_test.json")
    ap.add_argument("--base", required=True, help="基线 infer_test.json")
    ap.add_argument("--new_log", default=None)
    ap.add_argument("--base_log", default=None)
    ap.add_argument("--md", default=None, help="可选: 把 markdown 写到该路径")
    ap.add_argument("--plot", default=None, help="可选: 画对比图(PNG)到该路径")
    a = ap.parse_args()

    new, base = load_json(a.new), load_json(a.base)
    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    emit("# A/B: self.stack 2x(d_model 2048/heads 16/depth 4/dropout 0.05) vs 基线 blockdiag slice[0:5]")
    emit()
    emit(f"- 新模型: `{a.new}`")
    emit(f"- 基线:   `{a.base}`")
    emit(f"- test 全量: n={new.get('n')} (基线 n={base.get('n')})，"
         f"N={new.get('num_patches')}, K={new.get('num_specials')}, "
         f"steps={new.get('decoder_steps')}")
    emit()

    emit("## 1. 全量重建（同一推理脚本/口径）")
    emit()
    emit("| 指标 | 基线 blockdiag slice[0:5] | 本次 self.stack 2x | 变化 |")
    emit("|---|---|---|---|")
    for key, label, fmt in (("full_norm_l1", "归一化空间 L1 (eval_recon 口径)", "{:.4f}"),
                            ("full_pixel_l1_255", "0-255 像素 L1", "{:.2f}")):
        bv, nv = base.get(key), new.get(key)
        if bv is None or nv is None:
            emit(f"| {label} | {bv} | {nv} | — |")
            continue
        d = nv - bv
        pct = 100.0 * d / bv if bv else float("nan")
        emit(f"| {label} | {fmt.format(bv)} | {fmt.format(nv)} | "
             f"{d:+.4f} ({pct:+.2f}%) |")
    emit()

    emit("## 2. 渐进曲线（每采样步累积结果, 0-255 像素 L1）")
    emit()
    bs = base.get("step_pixel_l1_255") or []
    ns = new.get("step_pixel_l1_255") or []
    steps = new.get("decoder_steps") or list(range(len(ns)))
    emit("| 采样步 t | 基线 | self.stack 2x | 变化 |")
    emit("|---|---|---|---|")
    for i, t in enumerate(steps):
        if i < len(bs) and i < len(ns):
            emit(f"| {t} | {bs[i]:.2f} | {ns[i]:.2f} | {ns[i] - bs[i]:+.2f} |")
    if ns:
        emit()
        emit(f"- 基线 step1→末步 落差: {bs[0] - bs[-1]:+.2f}" if bs else "")
        emit(f"- 本次 step1→末步 落差: {ns[0] - ns[-1]:+.2f}")
    emit()

    bl, nl = parse_eval_log(a.base_log), parse_eval_log(a.new_log)
    if bl or nl:
        bd = {s: v for s, v in bl}
        nd = {s: v for s, v in nl}
        keys = sorted(set(bd) | set(nd), key=lambda x: (x is None, x if x is not None else 0))
        emit("## 3. 训练中 eval_recon 曲线")
        emit()
        emit("| checkpoint step | 基线 | self.stack 2x |")
        emit("|---|---|---|")
        for s in keys:
            b = f"{bd[s]:.4f}" if s in bd else "—"
            n = f"{nd[s]:.4f}" if s in nd else "—"
            emit(f"| {s if s is not None else 'final'} | {b} | {n} |")
        emit()

    if a.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
            ax = axes[0]
            for lg, name, color in ((a.base_log, "baseline blockdiag", "#7f8c8d"),
                                    (a.new_log, "self.stack 2x", "#c0392b")):
                ev = parse_eval_log(lg)
                pts = [(s, v) for s, v in ev if s is not None]
                if pts:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, "o-", color=color, label=name)
                fin = [v for s, v in ev if s is None]
                if fin:
                    ax.axhline(fin[-1], ls="--", lw=1, color=color)
                    ax.annotate(f"{fin[-1]:.4f}", (1.0, fin[-1]),
                                xycoords=("axes fraction", "data"),
                                ha="right", va="bottom", color=color, fontsize=9)
            ax.set_xlabel("train step")
            ax.set_ylabel("eval_recon (归一化 L1)")
            ax.set_title("训练中 eval_recon")
            ax.legend()
            ax.grid(alpha=0.3)

            ax = axes[1]
            if bs:
                ax.plot(range(1, len(bs) + 1), bs, "s--", color="#7f8c8d",
                        label="baseline blockdiag")
            if ns:
                ax.plot(range(1, len(ns) + 1), ns, "o-", color="#c0392b",
                        label="self.stack 2x")
            ax.set_xlabel("累积采样步序号")
            ax.set_ylabel("0-255 像素 L1")
            ax.set_title("渐进曲线（累积结果）")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(a.plot, dpi=140)
            print(f"[plot] {a.plot}")
        except Exception as e:  # 画图失败不应影响主结论输出
            print(f"[warn] 画图失败: {e}")

    if a.md:
        with open(a.md, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n[save] {a.md}")


if __name__ == "__main__":
    main()
