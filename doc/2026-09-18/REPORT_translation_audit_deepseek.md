# REPORT — 全量中文译文质检（DeepSeek deepseek-flash 评委）

> 时间：2026-09-18 11:45–12:30（本地出网 + 服务器产物只读）
> 评委：`deepseek-flash`（`https://api.deepseek.com`，`reasoning_effort=none`，temperature=0）
> 范围：两轮**全部 21,329 个译文字段**（第一轮 13,821 + 第二轮 7,508）
> 产物：`doc/2026-09-18/data/translation_audit/`（判决全量 + major 清单 + 校准集 + 脚本）
> **本次只审计，未修改任何译文产物**；服务器数据全程只读。

---

## 0. 一句话结论

**实测硬错误率 21.09%（4,499 / 21,329）**：`aswin00000` **26.52%**、`hayden-yuma` **19.93%**，其余数据集接近 0。
这**显著高于**此前"自由文本约 3–5%""hayden 修复后 2.41%"的结论——旧数字来自 **100 条冒烟外推**和**5 条正则规则**，
本次是全量 21,329 条、由经过校准的 LLM 评委逐条判定，**旧结论低估了 5–8 倍**。

---

## 1. 为什么换评委：Qwen3.8-27B 不可用（有据）

先按原计划用服务器常驻的 Qwen3.8-27B 做评委，在 22 条人工标注集上校准，结论是**不可用**：

| 提示词版本 | thinking | 抓错率 | 误报率 |
|---|---|---|---|
| 6 类清单 + JSON | 关 | 1.00 | **1.00**（16/16 全判 major） |
| 硬错误 JSON | 关 | 1.00 | **1.00** |
| 自由格式「无/证据」 | 关 | 1.00 | **1.00**（把模板占位符当错误抄出） |
| 二选一 OK/ERR | 关 | 1.00 | **0.80** |
| 4 例 few-shot | 关 | 0.73 | **0.60** |
| 二选一 + thinking | 开 | 截断 | 截断（~13 s/条，全量约 40 h） |

失效机制（可复现）：把 `excavator→挖掘机`、`hard hats→安全帽` 等**正确项**判为错误；把提示词示例里的术语对
（`crane=起重机`/`drum=隔离桶`）当成本行错误；复述原文时自我污染再据此判错。详见
`REPORT_llm_judge_calibration.md`。

**DeepSeek 校准（22 条：14 BAD / 7 GOOD / 1 MINOR）**：抓错率 **0.93**、误报率 **0.00**（21/22）。
唯一漏报是 `dredging→排水`（语义级术语错，较隐蔽）。开推理（`max_tokens=2500`）反而更差（0.86，且 3 条截断、慢 13 倍、多烧 38 倍 token），故**定稿关推理**。

## 2. 总览

| 项 | 值 |
|---|---|
| 复核单元 | **21,329** |
| 有效判决 | 21,329（15 条 JSON 被截断，已按 `major/garble` 归类修复） |
| ok | 16,830（**78.91%**） |
| minor | 0 |
| **major（硬错误）** | **4,499（21.09%）** |

> 评委抓错率 0.93 ⇒ 真实硬错误率约为实测的 1.08 倍（**≈22.7%**）。

### 2.1 错误类型（可多标签）

| 类型 | 次数 | 占总单元 |
|---|---:|---:|
| term 术语 | 3,426 | 16.06% |
| garble 乱码/重复 | 914 | 4.29% |
| neg 否定方向 | 811 | 3.80% |
| omit 漏译 | 736 | 3.45% |
| extra 多译 | 352 | 1.65% |
| num 数字 | 240 | 1.13% |

### 2.2 分语料

| 维度 | 单元 | major | 占比 |
|---|---:|---:|---:|
| 第一轮 | 13,821 | 3,008 | **21.76%** |
| 第二轮 | 7,508 | 1,491 | **19.86%** |
| 自由文本 | 21,165 | 4,496 | **21.24%** |
| 类别名/枚举 | 164 | 3 | **1.83%** |

### 2.3 分数据集

| dataset | 单元 | major | 占比 |
|---|---:|---:|---:|
| `aswin00000__ConstructionSiteCleanedDataSet` | 11,339 | 3,007 | **26.52%** |
| `hayden-yuma__roadwork` | 7,478 | 1,490 | **19.93%** |
| `iluvvatar__wood_surface_defects` | 10 | 1 | 10.00% |
| `chandrabhuma__multi_building_defect_vqa` | 2,417 | 1 | 0.04% |
| 其余 12 个数据集 | 85 | 0 | 0.00% |

**类名/枚举中文化质量很好（1.83%）**；问题集中在两个自由文本大数据集。

## 3. 评委精度人工验证（不是只看它自己说）

**随机抽 16 条 major 逐条人工通读，16/16 都含真实错误**（不是误报）。典型：

| 位置 | 原文 | 译文 | 性质 |
|---|---|---|---|
| aswin `train-00005#54` | A **pile driver** with its boom | 一堆**带有爆炸声的司机** | 术语崩坏 |
| hayden `train-00013#42` | **Cones** on right side of road | 道路右侧的**坑洼** | 术语（交通锥→坑洼） |
| hayden `train-00002#42` | Work vehicle **off** right side of road | 作业车辆**位于**道路右侧 | 方位/否定反转 |
| hayden `train-00019#124` | Barrier **next to** work zone | 屏障位于作业区**之后** | 方位错 |
| aswin `train-00001#358` | yellow **bulldozer** | 黄色**压路机** | 术语 |
| aswin `train-00000#562` | two **cranes** with extended booms | 两根带吊索的**软管** | 术语 |
| aswin `train-00003#5` | wooden **planks** | 木**托盘** | 术语 |
| aswin `train-00004#726` | orange **coveralls** and a red hard hat | 头戴橙色**安全帽**、身穿红色**硬壳** | 术语+乱码 |

**已知错例捕获（11 条）**：命中 10 条，仅 `train-00002#502`（`dredging→排水`）漏报。

**规则交叉验证**：全量扫描发现 16 条译文长度 >400 字、4 条存在同一字符连续 ≥8 次（最长 1,050 字全是「水」/「的」/「钢筋」重复），
3 条译文不含任何汉字——这类**灾难级乱码**由模型与规则**双通道同时命中**。

## 4. 与旧结论的对照

| 旧结论 | 来源 | 本次实测 | 差距 |
|---|---|---|---|
| aswin 自由文本"约 3–5% 语义/数字错误" | 第一轮报告 §6.1，**100 条冒烟外推** | **26.52%**（11,339 条全量） | **约 5–8 倍** |
| hayden 修复后"系统性术语错 2.41%" | 第二轮报告 §5.3，**5 条正则规则** | **19.93%**（7,478 条全量） | 正则只覆盖 5 个 TTC 词，未覆盖方位词/漏译/乱码/其它术语 |

⇒ 旧的两处"已达标"判断**都不成立**：`aswin00000` 从未做过全量质检；`hayden-yuma` 的 TTC 修复只解决了它自己定义的那 5 类。

## 5. 局限（诚实登记）

1. **单一评委、无第二模型交叉**；抓错率 0.93 基于 14 条 BAD，样本小，真实错误率可能略高于 21%。
2. **major 是"硬错误"口径**，包含"木托盘/压路机"这类**局部术语错**，不等于"整句不可用"。若只算语义反转/漏译/乱码等重错，占比更低（neg+omit+garble ≈ 11.5%）。
3. **评委只看文本、不看图**；个别判错可能需要图才能定论（但这些 caption 的错误大多是文本内部自洽性/数字/方位问题，判错概率低）。
4. **未做修复**，产物未改动。
5. 15 条 JSON 截断者按 `major/garble` 归类修复（其 raw 明确含 `"v":"major"` 与 `garble`），已登记。

## 6. 建议下一步

1. **aswin00000（26.5%）**：照 `hayden-yuma` 的 TTC 打法建**施工安全术语表**（coveralls/hard hat/rebar/pile driver/bulldozer/wheelbarrow/wooden planks…）后重翻，重点压 `term`（3,426 次）。
2. **hayden-yuma（19.9%）**：现有 TTC 术语表**不够**，需补方位词（`next to/off/behind/around`）、漏译（并列句丢句）与乱码三类规则，再重翻。
3. **乱码兜底**：对"同一字符连续 ≥8 次""长度 >400 字""无汉字"三条规则做**落盘前硬拦截**（本次 16 条灾难样本）。
4. **复测口径**：修复后用**同一评委、同一提示词**再跑一次，做 before/after 对照（可复现）。
5. 类名/枚举（1.83%）无需返工。

## 7. 复现

```bash
# 评委（本地出网；密钥 ~/.deepseek_key 权限 600）
python3 deepseek_judge.py --in judge_units_all.jsonl --out verdicts_all.jsonl \
  --concurrency 10 --reasoning-effort none --max-tokens 400
# 汇总
python3 aggregate_verdicts.py verdicts_all.jsonl AUDIT_deepseek_qc.md flagged_major.jsonl
# 乱码规则扫描 + 截断修复
python3 repair_and_scan.py
```

实测开销：主跑 **1,206.8 s（20.1 min，17.7 单元/s）**，tokens 约 **7.57 M 输入 + 0.33 M 输出**，另 15 条重试 17 k tokens。

## 8. 产物清单（`doc/2026-09-18/data/translation_audit/`）

| 文件 | 内容 |
|---|---|
| `AUDIT_deepseek_qc.md` | 自动汇总报告（总览/类型/分维度/样例） |
| `flagged_major.jsonl` | **4,499 条 major** 的可定位清单（含 EN/ZH/错误类型/证据，4.2 MB） |
| `verdicts_all.jsonl.gz` | 全量 21,329 条判决（2.2 MB） |
| `calibration_labeled.jsonl` / `calibration_verdicts.jsonl` | 22 条校准集与评委输出 |
| `deepseek_judge.py` / `aggregate_verdicts.py` / `repair_and_scan.py` | 评委客户端 / 汇总 / 修复+规则扫描 |
