#!/usr/bin/env python3
"""gen_step_gain_all_v2.py — 含 depth-8 臂① 的「全臂统一对比」(STEP_GAIN_ALL_ARMS_v2.md)。

数据源（均为实测原始产物, 相对本文件所在 doc/ 目录）:
  ../2026-09-12/data_stepid/STEP_GAIN_stepid.json                  (baseline/stack2x/臂①@d4 + ckpt)
  ../2026-09-12/data_stepidcls/STEP_GAIN_stepidcls.json            (…/臂②@d4)
  ../2026-09-12/data_stepidcls_stepid/STEP_GAIN_stepidcls_stepid.json (…/臂④@d4)
  data_stepid_d8/STEP_GAIN_stepid_d8.json                          (…/stack8x/臂①@d8 + ckpt)

口径: cum = 第 t 步累积输出整图 0-255 像素 L1; Δ = cum[t-1]-cum[t] (正=该步降误差);
      scale = 第 t 步原始增量 mean|pixel_head(Y_t)|。全部数字由本脚本从 JSON 计算, 无手工转录。
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.dirname(HERE)  # doc/

SRC = [
    (os.path.join(DOC, "2026-09-12/data_stepid/STEP_GAIN_stepid.json")),
    (os.path.join(DOC, "2026-09-12/data_stepidcls/STEP_GAIN_stepidcls.json")),
    (os.path.join(DOC, "2026-09-12/data_stepidcls_stepid/STEP_GAIN_stepidcls_stepid.json")),
    (os.path.join(HERE, "data_stepid_d8/STEP_GAIN_stepid_d8.json")),
]

probe, infer = {}, {}
for p in SRC:
    if not os.path.exists(p):
        print("[skip]", p); continue
    d = json.load(open(p))
    probe.update(d["probe"]); infer.update(d["infer"])

MAIN = [a for a in ["baseline", "stack2x_lr1e4", "stack8x", "stepid_d4",
                    "stepidcls_d4", "stepidcls_stepid_d4", "stepid_d8"] if a in probe]
LABEL = {
    "baseline": "baseline (1024/8/2, drop0, lr1.5e-4)",
    "stack2x_lr1e4": "stack2x (2048/16/4, drop.05, lr1e-4)",
    "stack8x": "stack8x (2048/16/8, drop.05, lr7.6e-5)",
    "stepid_d4": "臂①@d4 (2048/16/4 + step_embed)",
    "stepidcls_d4": "臂②@d4 (2048/16/4 + per-step [cls])",
    "stepidcls_stepid_d4": "臂④@d4 (per-step [cls] + step_embed)",
    "stepid_d8": "**臂①@d8** (2048/16/8 + step_embed)",
}
SHORT = {"baseline": "baseline", "stack2x_lr1e4": "stack2x", "stack8x": "stack8x",
         "stepid_d4": "臂①@d4", "stepidcls_d4": "臂②@d4",
         "stepidcls_stepid_d4": "臂④@d4", "stepid_d8": "臂①@d8"}


def f(x, n=4): return "—" if x is None else f"{x:+.{n}f}".replace("+-", "-")
def u(x, n=4): return "—" if x is None else f"{x:.{n}f}"


def table(src, steps, arms, key, fmt):
    out = ["| t | " + " | ".join(LABEL.get(a, a) for a in arms) + " |",
           "|---|" + "---|" * len(arms)]
    for i, t in enumerate(steps):
        out.append(f"| {t} | " + " | ".join(fmt(src[a][key][i]) for a in arms) + " |")
    return "\n".join(out)


def delta_at(src, arm, t):
    d = src.get(arm)
    if not d or t not in d["steps"]:
        return None
    return d["marg_px"][d["steps"].index(t)]


def sign(v):
    if v is None: return "—"
    return "**翻正**" if v > 0 else ("净负" if v < 0 else "0")


steps = probe[MAIN[0]]["steps"]
out = []
out.append("# 各采样 step 边际增益 — 全臂统一对比 v2（含 depth-8 臂①，SR-Diffusion v3, 2026-09-14）\n")
out.append("> 由 `gen_step_gain_all_v2.py` 从四个 `data_*/STEP_GAIN_*.json` 生成，**无手工转录**。")
out.append("> 口径：`cum` = 第 t 步**累积**输出整图 0-255 像素 L1（越小越好）；`Δ = cum[t−1] − cum[t]`（**正 = 该步降低误差**）；")
out.append("> `scale` = 第 t 步**原始增量** `mean|pixel_head(Y_t)|`。step1 无前序累积 ⇒ Δ 不可定义。**总落差 = cum[step1] − cum[step25]**。\n")
out.append("单变量设计：depth-4 系各臂除「改动」外逐项相同（`stack_dim 2048 / heads 16 / dropout 0.05 / lr 1.0e-4 / "
           "seed 42 / bs16×2卡 / 8760 步 / slice 0:5 (K=35) / blockdiag`）；depth-8 系两臂为 `lr 7.6e-5 / bs8×accum2 / "
           "8760 步`。`baseline` 为历史 depth-2 对照。臂① 为**零初始化** `step_embed`（训练起点与无注入逐位一致）；"
           "臂② 为每步独立 `[cls]` 查询槽；臂④ = 臂② **+** 臂①。**臂①@d8 与 stack8x 构成唯一单变量对照（仅差 `step_embed`）。**\n")

out.append("---\n\n## 1. probe 512 张\n")
out.append("### 1.1 累积 L1（0-255，越小越好）\n")
out.append(table(probe, steps, MAIN, "cum_px", u) + "\n")
out.append("### 1.2 边际增益 Δ（正 = 该步降低误差）— **核心对比**\n")
out.append(table(probe, steps, MAIN, "marg_px", f) + "\n")
out.append("| 总落差 (step1→step25) | " + " | ".join(
    f"**{probe[a]['cum_px'][0]-probe[a]['cum_px'][-1]:+.4f}**" for a in MAIN) + " |\n")
out.append("### 1.3 增量幅度 scale\n")
out.append(table(probe, steps, MAIN, "step_px_scale", u) + "\n")
out.append("### 1.4 scale / scale₁\n")
out.append("| t | " + " | ".join(LABEL.get(a, a) for a in MAIN) + " |")
out.append("|---|" + "---|" * len(MAIN))
for i, t in enumerate(steps):
    cs = []
    for a in MAIN:
        s = probe[a]["step_px_scale"]
        cs.append(f"{s[i]/s[0]*100:.1f}%" if s[0] else "—")
    out.append(f"| {t} | " + " | ".join(cs) + " |")
out.append("")

out.append("---\n\n## 2. 全量 test 3004 张\n")
out.append("### 2.1 累积像素 L1（0-255）\n")
out.append(table(infer, infer[MAIN[0]]["steps"], MAIN, "cum_px", u) + "\n")
out.append("### 2.2 边际增益 Δ — **核心对比**\n")
out.append(table(infer, infer[MAIN[0]]["steps"], MAIN, "marg_px", f) + "\n")
cells = [f"**{infer[a]['cum_px'][0]-infer[a]['cum_px'][-1]:+.4f}**" for a in MAIN]
out.append("| 总落差 (step1→step25) | " + " | ".join(cells) + " |\n")

out.append("### 2.3 聚合重建指标\n")
out.append("| 臂 | `full_norm_l1` | 像素 L1 (0-255) | vs baseline | vs stack2x | 填补 stack2x→baseline |")
out.append("|---|---|---|---|---|---|")
base = infer["baseline"]["full_norm_l1"]; s2x = infer["stack2x_lr1e4"]["full_norm_l1"]
gap = s2x - base
for a in MAIN:
    n = infer[a]["full_norm_l1"]; px = infer[a]["cum_px"][-1]
    vs_b = (n / base - 1) * 100; vs_s = (n / s2x - 1) * 100
    filled = (s2x - n) / gap * 100 if a != "baseline" and gap else None
    out.append(f"| {LABEL.get(a,a)} | {n:.5f} | {px:.3f} | {vs_b:+.2f}% | {vs_s:+.2f}% | "
               f"{f'{filled:.1f}%' if filled is not None else '—'} |")
out.append("")

if "stack8x" in infer and "stepid_d8" in infer:
    s8, d8 = infer["stack8x"]["full_norm_l1"], infer["stepid_d8"]["full_norm_l1"]
    gap8 = s8 - base
    out.append(f"> **单变量（仅 `step_embed`，depth-8 内部）**: 臂①@d8 vs stack8x = `{d8-s8:+.5f}` "
               f"({(d8/s8-1)*100:+.2f}%)，像素 {infer['stepid_d8']['cum_px'][-1]-infer['stack8x']['cum_px'][-1]:+.3f}。")
    out.append(f"> 臂①@d8 填补 stack8x→baseline 差距 = {(s8-d8)/gap8*100:.1f}%（差距 {gap8:.5f}）。")
    out.append(f"> **跨深度（同补丁 4→8）**: 无注入 stack2x→stack8x = `{s8-s2x:+.5f}` ({(s8/s2x-1)*100:+.2f}%)；"
               f"臂①@d4→臂①@d8 = `{d8-infer['stepid_d4']['full_norm_l1']:+.5f}` "
               f"({(d8/infer['stepid_d4']['full_norm_l1']-1)*100:+.2f}%)。\n")

out.append("---\n\n## 3. 逐 checkpoint 的 Δ 轨迹（probe 128 张）\n")
CKPT = sorted([k for k in probe if re.search(r"_step\d+$", k)],
              key=lambda k: (k.split("_step")[0], int(k.rsplit("_step", 1)[1])))
if CKPT:
    out.append("### 3.1 边际增益 Δ\n")
    out.append("| t | " + " | ".join(f"`{a}`" for a in CKPT) + " |")
    out.append("|---|" + "---|" * len(CKPT))
    for i, t in enumerate(steps):
        out.append(f"| {t} | " + " | ".join(f(probe[a]["marg_px"][i]) for a in CKPT) + " |")
    out.append("")
    out.append("### 3.2 累积 L1\n")
    out.append(table(probe, steps, CKPT, "cum_px", u) + "\n")
    out.append("### 3.3 总落差（cum[step1] − cum[step25]）\n")
    out.append("；".join(f"`{a}` = **{probe[a]['cum_px'][0]-probe[a]['cum_px'][-1]:+.4f}**" for a in CKPT) + "\n")

out.append("---\n\n## 4. 读法（全部由数据计算）\n")
out.append("### 4.1 后段符号：Δ16 / Δ25 是否翻正（主判据）\n")
out.append("| 臂 | probe Δ16 | probe Δ25 | 全量 test Δ16 | 全量 test Δ25 |")
out.append("|---|---|---|---|---|")
for a in MAIN:
    out.append("| " + SHORT.get(a, a) + " | " + " | ".join(
        [f"{f(delta_at(probe,a,16))} {sign(delta_at(probe,a,16))}",
         f"{f(delta_at(probe,a,25))} {sign(delta_at(probe,a,25))}",
         f"{f(delta_at(infer,a,16))} {sign(delta_at(infer,a,16))}",
         f"{f(delta_at(infer,a,25))} {sign(delta_at(infer,a,25))}"]) + " |")
out.append("")
out.append("### 4.2 Δ4 占总落差的比例（分工强度的反面）\n")
out.append("| 臂 | 全量 test 总落差 | 全量 Δ4 | Δ4 占比 | probe 总落差 | probe Δ4 | Δ4 占比 |")
out.append("|---|---|---|---|---|---|---|")
for a in MAIN:
    ci, cp = infer[a]["cum_px"], probe[a]["cum_px"]
    ti, tp = ci[0] - ci[-1], cp[0] - cp[-1]
    d4i, d4p = delta_at(infer, a, 4), delta_at(probe, a, 4)
    si = f"{d4i/ti*100:.1f}%" if ti and d4i is not None else "—"
    sp = f"{d4p/tp*100:.1f}%" if tp and d4p is not None else "—"
    out.append(f"| {SHORT.get(a,a)} | {ti:+.4f} | {f(d4i)} | {si} | {tp:+.4f} | {f(d4p)} | {sp} |")
out.append("")
out.append("### 4.3 聚合排序\n")
ranked = sorted(MAIN, key=lambda a: infer[a]["full_norm_l1"])
out.append("由好到差: " + " < ".join(f"{SHORT.get(a,a)} ({infer[a]['full_norm_l1']:.5f})" for a in ranked)
           + f"。（baseline {base:.5f} 为最佳对照）\n")

open(os.path.join(HERE, "STEP_GAIN_ALL_ARMS_v2.md"), "w").write("\n".join(out))
print("WROTE", os.path.join(HERE, "STEP_GAIN_ALL_ARMS_v2.md"))
print("arms:", MAIN)
print("ckpt:", CKPT)
