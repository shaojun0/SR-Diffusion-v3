# REPORT — 全量 CoT 生成（7 数据集 · 75,717 条）

> 状态：**已完成并归档**。看护于 **2026-09-20 09:13:03** 自动跑完收尾三连并写 `FINALIZE_DONE`；
> 同日 09:2x 人工补跑二次回收（`cot_full_repair2.py`）后重跑 QA/汇总，本报告数字为**二次回收后**的最终值。
> 任务书：`HANDOFF_20260919.md` §1；接手记录：`HANDOFF_20260919_TAKEOVER.md`
> 原始产物：`data/cot_full/{SUMMARY.json,QA.json,SUMMARY.md}`（服务器 `/root/autodl-tmp/cot_out/`）

---

## 1. 一句话结论

7 个数据集 **75,717 / 75,717 条全部生成**，解析成功 **75,713（99.995%）**，
detect 目标数对齐 **37,302 / 37,306（99.99%）**，机械质检门禁**全部通过**（画布 0 错、尺寸 0 错、退化重复 0）；
残留 4 条因 `max_tokens` 截断无法解析（2 个数据集各 2 条 caption），**只能重生成**。

---

## 2. 汇总（`SUMMARY.json`）

| 数据集 | 任务 | 已生成 | 计划 | 解析成功 | 解析失败 | detect 对齐 | 旋转 |
|---|---|---:|---:|---:|---:|---:|---:|
| kevincluo（野火损毁分级） | classify | 18,714 | 18,714 | 18,714 | 0 | — | 37 |
| iluvvatar（木材缺陷） | detect | 18,284 | 18,284 | 18,284 | 0 | 18,284/18,284 | 0 |
| baizhanquan（林火/烟雾） | detect | 10,683 | 10,683 | 10,683 | 0 | 10,682/10,683 | 589 |
| aswin00000（建筑工地） | detect 1,276 + caption 8,737 | 10,013 | 10,013 | 10,011 | 2 | 1,276/1,276 | 1,054 |
| hayden-yuma（道路作业区） | caption | 8,549 | 8,549 | 8,547 | 2 | — | 0 |
| hf-vision（安全帽） | detect | 7,063 | 7,063 | 7,063 | 0 | 7,060/7,063 | 746 |
| chandrabhuma（建筑缺陷VQA） | vqa | 2,411 | 2,411 | 2,411 | 0 | — | 1,079 |
| **合计** | | **75,717** | **75,717** | **75,713** | **4** | **37,302/37,306** | **3,505** |

`bad_json = 0`（无整行坏 JSON）；CoT 长度 中位 **174 字** / 最短 0 / 最长 2,425。

---

## 3. 质检门禁（`QA.json`，全量机检）

| 数据集 | rows | parse✗ | align✗ | 空/过短 | 退化重复 | 缺标签 | 画布✗ | 尺寸✗ | 框✗ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aswin00000 | 10,013 | 2 | 0 | 4 | 0 | 0 | 0 | 0 | 1 |
| baizhanquan | 10,683 | 0 | 1 | 2 | 0 | 1 | 0 | 0 | 8 |
| chandrabhuma | 2,411 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| hayden-yuma | 8,549 | 2 | 0 | 18 | 0 | 0 | 0 | 0 | 0 |
| hf-vision | 7,063 | 0 | 3 | 6 | 0 | 3 | 0 | 0 | 0 |
| iluvvatar | 18,284 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| kevincluo | 18,714 | 0 | 0 | 24 | 0 | 0 | 0 | 0 | 0 |
| **TOTAL** | **75,717** | **4** | **4** | **54** | **0** | **4** | **0** | **0** | **9** |

- **画布✗ = 0 / 尺寸✗ = 0**：manifest 声明画布与实际 JPEG 文件头尺寸**全量核对**（`QA_IMG_SAMPLE` 未设 = 全量），实现的是「读 PIL 头 + 按 id 连 manifest」，不是旧版那个恒真的假检查。
- **退化重复 = 0**：同字符连续 ≥8、或 20–40 字子串重复 ≥5 的样本为 0。
- **框✗ = 9**：detect 的 bbox 越界或 `x1≥x2 / y1≥y2`，集中在 baizhanquan(8) + aswin(1)；均为真值框本身的问题，不是生成问题（生成不写坐标）。
- **空/过短 = 54**：其中 27 条 `cot` 为空、27 条过短；空 CoT 的信息仍在 `gen.spatial/evidence/answer` 里未丢（口径是"不审核直接入库"，**未改数据**）。

---

## 4. 生成口径（2026-09-18 定稿，全程未变）

| 项 | 口径 |
|---|---|
| 画布 | 统一 **12:9 = 1200×900**；横图 letterbox（灰底 127），**竖图顺时针旋转 90°** 后填充（共 3,505 条） |
| 坐标 | bbox 换算到画布并归一化 **0–1000**；**坐标由系统用真值拼接，模型不写坐标** |
| 每图 | **一条 CoT** |
| detect | 先 `spatial`（整幅空间结构 + 方位 + 遮挡依附），再逐目标 `position`（落在什么结构的什么方位、依据），末尾系统拼坐标 |
| caption/vqa/classify | `spatial → evidence → cot → answer` |
| **关键** | detect 提示词**给出真值坐标作定位线索** —— 试点 A/B 实测把通过率从 **73.8% → 99.25%** |
| 审核 | **不审核、直接入库** |
| 跳过 | 6,924 行无标注图（baizhanquan 4,932 + iluvvatar 1,992），未生成"无目标"型 CoT |

---

## 5. 运行史与可靠性

| 时间（CST） | 事件 |
|---|---|
| 09-18 14:51 | 双卡（2×RTX PRO 6000 96GB）两路 Qwen3.8-27B 服务 + 两车道启动（laneA 37,276 / laneB 38,441） |
| 09-18 15:22 | 接手会话部署常驻看护 `cot_full_watch.py`（300s 快照；服务掉线重启、真完成自动收尾、ALL_DONE 计数不符自动续跑） |
| 09-18 19:03 | 一次**驱动重启**（旧 driver tree 被杀后按 §1.4 幂等续跑；**非看护触发**，watch.log 无 WARN/RESUME）。产物中已存在 id 自动跳过，**无重复** |
| 09-19 09:20 | kevincluo 完成（14,746 条 / 51,445 s） |
| 09-19 21:44 | **laneB 全部完成**（iluvvatar → baizhanquan 27,040 s → hf-vision → chandrabhuma 9,667 s），此后单车道 |
| 09-20 01:04 | aswin 完成（10,011 条 / 56,609 s） |
| 09-20 09:11 | hayden 完成 → `LANEA ALL DONE`，**总计 75,717 条收齐** |
| 09-20 09:12:42–09:13:02 | 看护自动执行 repair → qa → aggregate（三步 rc=0）→ 写 **`FINALIZE_DONE`** |
| 09-20 09:13:03 | 看护打印「任务结束，退出」并自行退出 |
| 09-20 09:2x | 人工补跑 `cot_full_repair2.py`（括号配平二次回收 4 条）+ 重跑 qa/aggregate |

- 全程 **42 h 20 min**（09-18 14:51 → 09-20 09:11），平均 **0.497 条/s**。
- **会话中断/工控机重启均不影响任务**：驱动与两路服务都是 `setsid nohup` 独立会话（09-19 19:44 工控机重启过一次，服务器侧零感知）。
- 期间两路服务 RSS 由 ~6 GiB 缓慢涨到 ~15–16 GiB，**未触及看护 20 GiB 告警线**；显存稳定 ~67–76 GiB/卡。
- 未发生 WARN_* / FINALIZE_FAILED。

---

## 6. 残留问题（逐条，含可修性）

| # | 问题 | 数量 | 可修性 |
|---|---|---:|---|
| 1 | `raw` 被 `max_tokens=1500` 截断（模型进入 self-doubt 循环，如"或者，也许我看反了？不…"） | **4**（aswin 2 + hayden 2，均 caption） | **不可机修，只能重生成这 4 条** |
| 2 | detect `objects` 数与真值框数不等 ⇒ 未拼坐标 | **4**（baizhanquan 1 + hf-vision 3） | 可重生成；不影响其余 37,302 条 |
| 3 | 真值 bbox 越界/退化（生成侧无关） | **9** | 属真值质量问题，登记不改 |
| 4 | `cot` 空 / 过短（信息在 `gen` 里未丢） | **54**（27 空 + 27 过短） | 口径为不审核入库，未改；下游若只吃 `cot`，可在 repair 里加带标记的派生回退 |
| 5 | hayden 有 **1,334 行无真值描述**（1,134 空 + 200 过短，含 176 行字面量 `No Description`） | 1,334 | 生成时提示词的"真值描述："为空 ⇒ 模型自编 answer；建议后置按 `gt.text=="" / len<20` 标记或过滤 |
| 6 | 6,924 行无标注图未纳入（按 detect 口径跳过） | 6,924 | 如需"无目标"型 CoT 可后补 |
| 7 | 竖图旋转 90°（用户已确认保留） | 3,505 | 若改纯填充需重跑 |
| 8 | 收尾脚本 `cot_full_repair.py` 无法回收的 8 条中，有 4 条只是**结尾括号写错**（`]]` 应为 `]}` 或漏 `}`） | 4 | **已由新增的 `cot_full_repair2.py` 修复**（带截断守卫：结尾仍在字符串里的一律不修） |

> 修复效果：`parse_fail 8→4`、`align✗ 8→4`、`空/过短 62→54`、`缺标签 8→4`、`detect 对齐 37,298→37,302`。

---

## 7. 复现（服务器 `ssh -p 38024 root@connect.westb.seetacloud.com`）

```bash
export PATH=/root/miniconda3/bin:$PATH
# 续跑生成（幂等：已存在 id 自动跳过）
setsid nohup bash /root/translate/run_cot_full.sh > /root/translate_logs/cot_full/driver_console.log 2>&1 </dev/null &
# 看护（服务重启 + 自动收尾 + 有界续跑）
setsid nohup python3 /root/translate/cot_full_watch.py >/dev/null 2>&1 </dev/null &
# 收尾三连（必须等生成全部结束；顺序不能变）
python3 /root/translate/cot_full_repair.py      # ① 回收解析失败（写 .pre_repair.bak）
python3 /root/translate/cot_full_repair2.py     # ①' 括号配平二次回收（写 .pre_repair2.bak，本会话新增）
python3 /root/translate/cot_full_qa.py          # ② 机械质检门禁 -> cot_out/QA.json
python3 /root/translate/cot_full_aggregate.py   # ③ 汇总 -> cot_out/SUMMARY.json + SUMMARY.md
```

⚠️ 只有 `repair*` 会**重写**产物；必须在生成完全停止后跑（生成器全程持有文件 fd，运行中重写会造成 inode 失联，
事故记录见 `translate_logs/cot_full/INCIDENT_20260918_repair_orphan.md`）。`qa` / `aggregate` 只读。
⚠️ 看护的收尾**只发生一次**（`FINALIZE_ATTEMPTED` 存在即不再执行）；手工重跑需先删该标记。
⚠️ 服务器**没有 `bc`**，驱动末尾的 `TOTAL_LINES.txt` 不会生成，**以 `SUMMARY.json` 为准**。

---

## 8. 后续（本会话新增的一条线）：CoT「国标依据」后置增强

生成完成后，按用户口径给 CoT **后置追加国标依据**（不改原有叙事）：

- **原则**：条款原文是"数据"不是"生成内容"。模型（`deepseek-flash`，带视觉，`reasoning_effort:"none"`）
  只输出「图上可见特征 + 一句事实指认」，**条款号与原文由系统查表字符串拼接**，模型输出里禁止出现 `「」` 或 `GB`。
- **条款集合由隐患描述确定性判定**（用户口径：先看描述说了什么），图像特征只做叠加/校验；
  已纳入 `GB 55034-2022 3.1.2`（安全警示标识，用户 2026-09-20 确认）。
- 已完成：aswin detect **1,276/1,276**（1,253 条已引依据；22 条因证据不支持被门禁拦下）；
  hf-vision `head` **1,319 条**（960 条已引依据；359 条被拦，其中包含**`head` 框并非"未戴安全帽"**的非工地场景 —— 该标签不可直接当违规判据）。
- 无隐患侧（"为什么没有隐患"）：aswin caption 试点 292 条 + Label Studio 复核项目（project id=4）已就绪。
- 细节见后续 `REPORT_standard_basis.md`（待写）。

---

## 9. 归档清单

| 路径 | 内容 |
|---|---|
| `doc/2026-09-18/REPORT_cot_full.md` | 本报告 |
| `doc/2026-09-18/data/cot_full/{SUMMARY.json,QA.json,SUMMARY.md}` | 收尾三连的最终产物（二次回收后） |
| `doc/2026-09-18/data/cot_pilot/{cot_build_full,cot_generate,cot_full_repair,cot_full_repair2,cot_full_qa,cot_full_aggregate,cot_full_watch}.py`、`run_cot_full.sh` | 与服务器**逐字节一致**（md5 核对）的流水线脚本 |
| 服务器 `/root/autodl-tmp/cot_out/` | 7 个 `<dataset>.jsonl` 产物 + `SUMMARY/QA` + `ALL_DONE` |
| 服务器 `/root/translate_logs/cot_full/` | `watch.log`（513 行）、`watch.jsonl`、`lane{A,B}.log`、`finalize.log`、`FINALIZE_DONE`、产物 `.pre_repair*.bak` |
