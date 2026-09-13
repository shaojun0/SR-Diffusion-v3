#!/usr/bin/env python3
"""Generate STEP_GAIN_ALL_ARMS.md — 全部臂 / 全部 ckpt 的各采样 step 边际增益完整对比.

数据源（均为实测原始产物，本目录内）:
  data_stepid/STEP_GAIN_stepid.json              (臂①: baseline / stack2x / stepid_d4 + stepid_d4_step*)
  data_stepidcls/STEP_GAIN_stepidcls.json        (臂②: 四臂 + stepidcls_d4_step*)
  data_stepidcls_stepid/STEP_GAIN_stepidcls_stepid.json (臂④: 五臂 + stepidcls_stepid_d4_step*) ← 可选

口径: cum_px = 第 t 步累积输出的整图 0-255 像素 L1（probe 512 / full test 3004）;
      marg_px = cum[t-1] - cum[t]（正 = 该步降低误差）;
      step_px_scale = 第 t 步原始增量 mean|pixel_head(Y_t)|（未累加, 仅 probe）。
step1 无前序累积, marg 不可定义。总落差 = cum[first] - cum[last] = Σ marg[2:]。

§4 的裁决文字全部由数据计算得到（无手工转录）；「身份 vs 槽位」的 P1/P2/P3 归属
结论写在 EXPERIMENT_step_identity_arm3_sharedcls.md。
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))


def load_arm(subdir, fname):
    p = os.path.join(HERE, subdir, fname)
    if not os.path.exists(p):
        print(f"[skip] {p} 不存在")
        return None
    return json.load(open(p))


A1 = load_arm("data_stepid", "STEP_GAIN_stepid.json")
A2 = load_arm("data_stepidcls", "STEP_GAIN_stepidcls.json")
A3 = load_arm("data_stepidcls_stepid", "STEP_GAIN_stepidcls_stepid.json")

probe, infer = {}, {}
for A in (A1, A2, A3):
    if A:
        probe.update(A["probe"])
        infer.update(A["infer"])

MAIN = [a for a in ["baseline", "stack2x_lr1e4", "stepid_d4", "stepidcls_d4", "stepidcls_stepid_d4"]
        if a in probe]
CKPT = sorted([k for k in probe if re.search(r"_step\d+$", k)],
              key=lambda k: (k.split("_step")[0], int(k.rsplit("_step", 1)[1])))

LABEL = {
    "baseline": "baseline (1024/8/2, drop0, lr1.5e-4)",
    "stack2x_lr1e4": "stack2x (2048/16/4, drop.05, lr1e-4)",
    "stepid_d4": "**臂①** stepid (2048/16/4 + step_embed)",
    "stepidcls_d4": "**臂②** stepidcls (2048/16/4 + per-step [cls])",
    "stepidcls_stepid_d4": "**臂④** stepidcls+stepid (2048/16/4 + per-step [cls] + step_embed)",
}
SHORT = {
    "baseline": "baseline",
    "stack2x_lr1e4": "stack2x",
    "stepid_d4": "臂① stepid",
    "stepidcls_d4": "臂② stepidcls",
    "stepidcls_stepid_d4": "臂④ stepidcls+stepid",
}


def f(x, n=4):
    return "—" if x is None else f"{x:+.{n}f}".replace("+-", "-")


def u(x, n=4):
    return "—" if x is None else f"{x:.{n}f}"


def table(src, steps, arms, key, fmt):
    hdr = "| t | " + " | ".join(LABEL.get(a, f"`{a}`") for a in arms) + " |"
    sep = "|---|" + "---|" * len(arms)
    rows = []
    for i, t in enumerate(steps):
        cells = []
        for a in arms:
            v = src[a][key][i] if key in src[a] else None
            cells.append(fmt(v))
        rows.append(f"| {t} | " + " | ".join(cells) + " |")
    return "\n".join([hdr, sep] + rows)


def step_index(steps, t):
    return steps.index(t) if t in steps else None


def delta_at(src, arm, t):
    d = src.get(arm)
    if not d:
        return None
    i = step_index(d["steps"], t)
    return None if i is None else d["marg_px"][i]


def sign_word(v):
    if v is None:
        return "—"
    if v > 0:
        return "**翻正**"
    if v < 0:
        return "净负"
    return "0"


out = []
n_arms = len([a for a in MAIN if a.startswith("stepid")])
out.append("# 各采样 step 边际增益 — 全部臂完整对比（SR-Diffusion v3, 2026-09-12/13）\n")
out.append("> 由 `gen_step_gain_all.py` 从 `data_stepid/` + `data_stepidcls/` + `data_stepidcls_stepid/` 的")
out.append("> `STEP_GAIN_*.json` 生成，**无手工转录**。口径：`cum` = 第 t 步**累积**输出的整图 0-255 像素 L1（越小越好）；")
out.append("> `Δ` = cum[t−1] − cum[t]（**正 = 该步降低误差**）；`scale` = 第 t 步**原始增量** `mean|pixel_head(Y_t)|`。")
out.append("> step1 无前序累积 ⇒ `Δ` 不可定义。**总落差 = cum[step1] − cum[step25] = Σ Δ(step4..step25)**。\n")
out.append("单变量设计：各臂除「改动」外逐项相同（depth-4 系配置 `stack_dim 2048 / heads 16 / "
           "dropout 0.05 / lr 1.0e-4 / seed 42 / bs16×2卡 / 8760 步 / slice 0:5 (K=35) / blockdiag`）；"
           "`baseline` 为历史 depth-2 对照。臂① 为**零初始化** `step_embed`（默认路径逐位不变，"
           "实测 `max_abs_diff=0.0`）；臂② 为每步独立 `[cls]` 查询槽；臂④ = 臂② **+** 臂① `step_embed`"
           "（零初始化 ⇒ 训练起点逐位等于纯臂②；作用域 = 臂① 原作用域，即该步全部查询行）。\n")

out.append("---\n\n## 1. probe 512 张\n")
out.append("### 1.1 累积 L1（0-255，越小越好）\n")
out.append(table(probe, probe[MAIN[0]]["steps"], MAIN, "cum_px", u) + "\n")
out.append("### 1.2 边际增益 Δ（正 = 该步降低误差）— **核心对比**\n")
out.append(table(probe, probe[MAIN[0]]["steps"], MAIN, "marg_px", f) + "\n")

steps = probe[MAIN[0]]["steps"]
cells = []
for a in MAIN:
    c = probe[a]["cum_px"]
    cells.append(f"**{c[0] - c[-1]:+.4f}**")
out.append("| 总落差 (step1→step25) | " + " | ".join(cells) + " |\n")

out.append("### 1.3 增量幅度 scale（该步『发声』强度）\n")
out.append(table(probe, steps, MAIN, "step_px_scale", u) + "\n")
out.append("### 1.4 scale / scale₁（后 4 步占 step1 的比例）\n")
out.append("| t | " + " | ".join(LABEL.get(a, a) for a in MAIN) + " |")
out.append("|---|" + "---|" * len(MAIN))
for i, t in enumerate(steps):
    cs = []
    for a in MAIN:
        s = probe[a]["step_px_scale"]
        cs.append(f"{s[i] / s[0] * 100:.1f}%" if s[0] else "—")
    out.append(f"| {t} | " + " | ".join(cs) + " |")
out.append("")

out.append("---\n\n## 2. 全量 test 3004 张\n")
out.append("### 2.1 累积像素 L1（0-255）\n")
out.append(table(infer, infer[MAIN[0]]["steps"], MAIN, "cum_px", u) + "\n")
out.append("### 2.2 边际增益 Δ — **核心对比**\n")
out.append(table(infer, infer[MAIN[0]]["steps"], MAIN, "marg_px", f) + "\n")
cells = []
for a in MAIN:
    c = infer[a]["cum_px"]
    cells.append(f"**{c[0] - c[-1]:+.4f}**")
out.append("| 总落差 (step1→step25) | " + " | ".join(cells) + " |\n")
out.append("### 2.3 聚合重建指标\n")
out.append("| 臂 | `full_norm_l1` | 像素 L1 (0-255) | vs baseline | vs stack2x | 填补 stack2x→baseline |")
out.append("|---|---|---|---|---|---|")
base = infer["baseline"]["full_norm_l1"]
s2x = infer["stack2x_lr1e4"]["full_norm_l1"]
gap = s2x - base
for a in MAIN:
    n = infer[a]["full_norm_l1"]
    px = infer[a]["cum_px"][-1]
    vs_b = (n / base - 1) * 100
    vs_s = (n / s2x - 1) * 100
    filled = (s2x - n) / gap * 100 if a != "baseline" and gap else None
    filled_s = f"{filled:.1f}%" if filled is not None else "—"
    out.append(f"| {LABEL.get(a, a)} | {n:.5f} | {px:.3f} | {vs_b:+.2f}% | {vs_s:+.2f}% | {filled_s} |")
out.append("")
out.append(f"> stack2x→baseline 差距 = {gap:.5f}；「填补」列 = (stack2x − 本臂) / 差距。")
fills = [f"{SHORT[a]} {((s2x - infer[a]['full_norm_l1']) / gap * 100):.1f}%"
         for a in MAIN if a.startswith("stepid")]
out.append("> 身份注入臂的填补率: " + "，".join(fills) + "。\n")

out.append("---\n\n## 3. 逐 checkpoint 的 Δ 轨迹（probe 128 张）\n")
if CKPT:
    out.append("### 3.1 边际增益 Δ\n")
    out.append("| t | " + " | ".join(f"`{a}`" for a in CKPT) + " |")
    out.append("|---|" + "---|" * len(CKPT))
    for i, t in enumerate(steps):
        cs = [f(probe[a]["marg_px"][i]) for a in CKPT]
        out.append(f"| {t} | " + " | ".join(cs) + " |")
    out.append("")
    out.append("### 3.2 累积 L1\n")
    out.append(table(probe, steps, CKPT, "cum_px", u) + "\n")
    out.append("### 3.3 总落差（cum[step1] − cum[step25]）\n")
    cells = []
    for a in CKPT:
        c = probe[a]["cum_px"]
        cells.append(f"`{a}` = **{c[0] - c[-1]:+.4f}**")
    out.append("；".join(cells) + "\n")
else:
    out.append("(无 ckpt 数据)\n")

out.append("---\n\n## 4. 读法（全部由数据计算）\n")
out.append("### 4.1 后段符号：Δ16 / Δ25 是否翻正（主判据）\n")
out.append("| 臂 | probe Δ16 | probe Δ25 | 全量 test Δ16 | 全量 test Δ25 |")
out.append("|---|---|---|---|---|")
for a in MAIN:
    out.append("| " + SHORT.get(a, a) + " | "
               + " | ".join([f"{f(delta_at(probe, a, 16))} {sign_word(delta_at(probe, a, 16))}",
                             f"{f(delta_at(probe, a, 25))} {sign_word(delta_at(probe, a, 25))}",
                             f"{f(delta_at(infer, a, 16))} {sign_word(delta_at(infer, a, 16))}",
                             f"{f(delta_at(infer, a, 25))} {sign_word(delta_at(infer, a, 25))}"]) + " |")
out.append("")

out.append("### 4.2 量级：Δ4 占总落差的比例（分工强度的反面）\n")
out.append("| 臂 | 全量 test 总落差 | 全量 test Δ4 | Δ4 占比 | probe 总落差 | probe Δ4 | Δ4 占比 |")
out.append("|---|---|---|---|---|---|---|")
for a in MAIN:
    ci = infer[a]["cum_px"]
    cp = probe[a]["cum_px"]
    ti, tp = ci[0] - ci[-1], cp[0] - cp[-1]
    d4i, d4p = delta_at(infer, a, 4), delta_at(probe, a, 4)
    si = f"{d4i / ti * 100:.1f}%" if ti else "—"
    sp = f"{d4p / tp * 100:.1f}%" if tp else "—"
    out.append(f"| {SHORT.get(a, a)} | {ti:+.4f} | {f(d4i)} | {si} | {tp:+.4f} | {f(d4p)} | {sp} |")
out.append("")

out.append("### 4.3 臂④ 与臂①/臂② 的直接差值（「叠加增益」的证据）\n")
BASE4 = "stepidcls_stepid_d4"
if BASE4 in infer and "stepid_d4" in infer:
    out.append("臂④ = 臂② 每步独立 `[cls]` **+** 臂① `step_embed`；`step_embed` 零初始化 ⇒ 训练起点逐位等于纯臂②。\n")
    out.append("| 对比 | 含义 | 全量 `full_norm_l1` 差 | probe 总落差差 | 全量 总落差差 |")
    out.append("|---|---|---|---|---|")
    for other, meaning in (("stepidcls_d4", "`step_embed`（臂①）的增量，相对纯臂②"),
                           ("stepid_d4", "额外槽位（臂②）的增量，相对纯臂①")):
        if other not in infer:
            continue
        dn = infer[BASE4]["full_norm_l1"] - infer[other]["full_norm_l1"]
        cp4, cpo = probe[BASE4]["cum_px"], probe[other]["cum_px"]
        ci4, cio = infer[BASE4]["cum_px"], infer[other]["cum_px"]
        verdict = "臂④更差" if dn > 0 else "臂④更好"
        out.append(f"| 臂④ − {SHORT[other]} | {meaning} | {dn:+.5f}（{verdict}） | "
                   f"{((cp4[0] - cp4[-1]) - (cpo[0] - cpo[-1])):+.4f} | "
                   f"{((ci4[0] - ci4[-1]) - (cio[0] - cio[-1])):+.4f} |")
    out.append("")
    out.append("> 读法: 差值 ≈ 0 ⇒ 该组件在另一组件已存在时没有独立贡献；负值 = 叠加有正增益。\n")
else:
    out.append("(臂④ 数据未入库 — 待 `data_stepidcls_stepid/` 生成后重跑本脚本)\n")

out.append("### 4.4 聚合排序\n")
ranked = sorted(MAIN, key=lambda a: infer[a]["full_norm_l1"])
out.append("由好到差: " + " < ".join(f"{SHORT.get(a, a)} ({infer[a]['full_norm_l1']:.5f})" for a in ranked)
           + f"。（baseline {base:.5f} 为最佳对照）\n")

open(os.path.join(HERE, "STEP_GAIN_ALL_ARMS.md"), "w").write("\n".join(out))
print("WROTE", os.path.join(HERE, "STEP_GAIN_ALL_ARMS.md"))
print("probe arms:", list(probe.keys()))
print("ckpt arms:", CKPT)
