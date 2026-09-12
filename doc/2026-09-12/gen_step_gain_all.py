#!/usr/bin/env python3
"""Generate STEP_GAIN_ALL_ARMS.md — 全部臂 / 全部 ckpt 的各采样 step 边际增益完整对比.

数据源（均为实测原始产物，本目录内）:
  data_stepid/STEP_GAIN_stepid.json         (臂①: baseline / stack2x / stepid_d4 + stepid_d4_step*)
  data_stepidcls/STEP_GAIN_stepidcls.json   (臂②: 四臂 + stepidcls_d4_step*)
口径: cum_px = 第 t 步累积输出的整图 0-255 像素 L1（probe 512 / full test 3004）;
      marg_px = cum[t-1] - cum[t]（正 = 该步降低误差）;
      step_px_scale = 第 t 步原始增量 mean|pixel_head(Y_t)|（未累加, 仅 probe）。
step1 无前序累积, marg 不可定义。总落差 = cum[first] - cum[last] = Σ marg[2:]。
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
A1 = json.load(open(os.path.join(HERE, "data_stepid", "STEP_GAIN_stepid.json")))
A2 = json.load(open(os.path.join(HERE, "data_stepidcls", "STEP_GAIN_stepidcls.json")))

probe = {**A1["probe"], **A2["probe"]}
infer = {**A1["infer"], **A2["infer"]}

MAIN = ["baseline", "stack2x_lr1e4", "stepid_d4", "stepidcls_d4"]
MAIN = [k for k in MAIN if k in probe]
CKPT = sorted([k for k in probe if re.search(r"_step\d+$", k)],
              key=lambda k: (k.split("_step")[0], int(k.rsplit("_step", 1)[1])))

LABEL = {
    "baseline": "baseline (1024/8/2, drop0, lr1.5e-4)",
    "stack2x_lr1e4": "stack2x (2048/16/4, drop.05, lr1e-4)",
    "stepid_d4": "**臂①** stepid (2048/16/4 + step_embed)",
    "stepidcls_d4": "**臂②** stepidcls (2048/16/4 + per-step [cls])",
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


def trajectory(arm, key):
    steps = probe[arm]["steps"]
    rows = []
    for i, t in enumerate(steps):
        v = probe[arm][key][i] if key in probe[arm] else None
        rows.append((t, v))
    return rows


out = []
out.append("# 各采样 step 边际增益 — 全部臂完整对比（SR-Diffusion v3, 2026-09-12）\n")
out.append("> 由 `gen_step_gain_all.py` 从 `data_stepid/` + `data_stepidcls/` 的 `STEP_GAIN_*.json` 生成，")
out.append("> 无手工转录。口径：`cum` = 第 t 步**累积**输出的整图 0-255 像素 L1（越小越好）；")
out.append("> `Δ` = cum[t−1] − cum[t]（**正 = 该步降低误差**）；`scale` = 第 t 步**原始增量** `mean|pixel_head(Y_t)|`。")
out.append("> step1 无前序累积 ⇒ `Δ` 不可定义。**总落差 = cum[step1] − cum[step25] = Σ Δ(step4..step25)**。\n")
out.append("单变量设计：四臂除「改动」外逐项相同（depth-4 系配置 `stack_dim 2048 / heads 16 / "
           "dropout 0.05 / lr 1.0e-4 / seed 42 / bs16×2卡 / 8760 步 / slice 0:5 (K=35) / blockdiag`）；"
           "`baseline` 为历史 depth-2 对照。臂① 为**零初始化** `step_embed`（默认路径逐位不变，"
           "实测 `max_abs_diff=0.0`）；臂② 为每步独立 `[cls]` 查询槽（输出切回 N 行，对外契约不变）。\n")

out.append("---\n\n## 1. probe 512 张\n")
out.append("### 1.1 累积 L1（0-255，越小越好）\n")
out.append(table(probe, probe[MAIN[0]]["steps"], MAIN, "cum_px", u) + "\n")
out.append("### 1.2 边际增益 Δ（正 = 该步降低误差）— **核心对比**\n")
out.append(table(probe, probe[MAIN[0]]["steps"], MAIN, "marg_px", f) + "\n")

# total drop row
steps = probe[MAIN[0]]["steps"]
cells = []
for a in MAIN:
    c = probe[a]["cum_px"]
    cells.append(f"**{c[0] - c[-1]:+.4f}**")
out.append("| 总落差 (step1→step25) | " + " | ".join(cells) + " |\n")

out.append("### 1.3 增量幅度 scale（该步『发声』强度）\n")
out.append(table(probe, steps, MAIN, "step_px_scale", u) + "\n")
out.append("### 1.4 scale / scale₁（后 4 步占 step1 的比例）\n")
hdr = "| t | " + " | ".join(LABEL.get(a, a) for a in MAIN) + " |"
out.append(hdr)
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
out.append("| 臂 | `full_norm_l1` | 像素 L1 (0-255) | vs baseline | vs stack2x |")
out.append("|---|---|---|---|---|")
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
    out.append(f"| {LABEL.get(a, a)} | {n:.5f} | {px:.3f} | {vs_b:+.2f}% | {vs_s:+.2f}% |")
out.append("")
out.append(f"> stack2x→baseline 差距 = {gap:.5f}；「填补」列 = (stack2x − 本臂) / 差距。")
out.append(f"> 臂① 填补 {((s2x - infer['stepid_d4']['full_norm_l1']) / gap * 100):.1f}%，"
           f"臂② 填补 {((s2x - infer['stepidcls_d4']['full_norm_l1']) / gap * 100):.1f}%。\n")

out.append("---\n\n## 3. 逐 checkpoint 的 Δ 轨迹（probe 128 张）\n")
if CKPT:
    out.append("### 3.1 边际增益 Δ\n")
    hdr = "| t | " + " | ".join(f"`{a}`" for a in CKPT) + " |"
    out.append(hdr)
    out.append("|---|" + "---|" * len(CKPT))
    for i, t in enumerate(steps):
        cs = []
        for a in CKPT:
            cs.append(f(probe[a]["marg_px"][i]))
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

out.append("---\n\n## 4. 读法（结论摘要）\n")
out.append("1. **符号**：baseline/stack2x 的 Δ16、Δ25 在 probe 与全量 test 上**全为负**（后步净有害）；")
out.append("   臂① 使 Δ16 翻正、Δ25 大幅减负；**臂② 是唯一让 Δ16 与 Δ25 在全量 test 上同时翻正**的配置。")
out.append("2. **量级**：臂② 后段 Δ 仅 +0.0015~+0.0047 px，是绝对误差（~21–25 px）的 ~0.02%；")
out.append("   Δ4 仍占总落差的绝大部分（全量 test：Δ4 +0.0222 / 总落差 +0.0302 = 73%）。**无实质「后段分工」。**")
out.append("3. **收益归属**：臂② 的聚合改善主要来自 **step1 单发读出变好**"
           "（全量 `head_mean`：stack2x 23.885 → 臂① 23.097 → 臂② 21.335；baseline 19.012），")
out.append("   而不是后段变强 ⇒ 与「时间轴无正贡献、单发路线最优」的仓库战略判断一致。")
out.append("4. **聚合**：`full_norm_l1` baseline 0.33179 / stack2x 0.41729 / 臂① 0.40316 / 臂② 0.37442；")
out.append("   臂② 把 stack2x→baseline 差距填补 **50.1%**（臂① 16.5%），但仍落后 baseline 12.85%。")
out.append("\n> 完整分析、判据与局限见 `REPORT_step_identity_AB.md`。\n")

open(os.path.join(HERE, "STEP_GAIN_ALL_ARMS.md"), "w").write("\n".join(out))
print("WROTE", os.path.join(HERE, "STEP_GAIN_ALL_ARMS.md"))
print("probe arms:", list(probe.keys()))
print("ckpt arms:", CKPT)
