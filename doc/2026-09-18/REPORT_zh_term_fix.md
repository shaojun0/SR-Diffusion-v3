# REPORT — aswin 工地隐患数据「中文译文」术语/缩写修正（R1/R2/R3）

> 日期：**2026-09-18**
> 触发：用户在 Label Studio「aswin 工地隐患 · 规则→国标条款 复核」复核时提出三条质检口径
> 适用数据：`label-studio/data/aswin_hazard_manifest_zh.jsonl`（1,276 行，aswin00000/ConstructionSiteCleanedDataSet 的隐患图 + 真值规则框 + 真值描述中译）
> 结论：**383 处中文译文字段已按三条口径确定性修正**；线上 Label Studio 316 个任务就地更新，**31 条人工标注零丢失**。

---

## 1. 用户口径（本报告的唯一判据）

| 编号 | 口径 | 说明 |
|---|---|---|
| **R1** | 语义 | `not using PPE` 一类**否定**句不得译成「不得使用…」（语义反转）。例：`Person on the left not using PPE.` 原译「左侧人员不得使用PPE。」 |
| **R2** | 术语 | `hard hat` 统一译**安全帽**，不得出现「硬帽」 |
| **R3** | 缩写 | 任何缩写首次出现须在**后紧跟的括号**内给出全称；中文语境下**全称可为英文** |

> R1 的原始出处不是报告而是**用户的真实标注**：Label Studio task 1 的备注写着
> 「`…：左侧人员不得使用PPE。`这里意义不明，是不是左侧人员为配备PPE？」——
> 与本文档 R1 完全对应，可视为口径的一手证据。

---

## 2. 方法：确定性后处理，可复现

不重新调用 LLM（全量重翻是另一条线，见 §6）。新增一个**纯查表/正则**的修正器，
在 `enrich_manifest_zh.py` 落盘前统一过一遍，保证「重新生成 manifest 也不会回退」：

```
verdicts_all.jsonl.gz ──enrich_manifest_zh.py──► caption_zh / reason_zh
                                                   │
                                          zh_fix.fix_row()  ← R1/R2/R3 确定性修正
                                                   ▼
                                    aswin_hazard_manifest_zh.jsonl（已修正）
```

- 修正器：`label-studio/tools/zh_fix.py`（幂等，二次运行 0 改动，已实测）
- 生成脚本：`label-studio/tools/enrich_manifest_zh.py`（新增 `fix_row(r)` 调用）
- 线上回写：`label-studio/tools/update_aswin_ls_zh.py`

---

## 3. 修正结果

| 类别 | 字段数 |
|---|---:|
| R2 `硬帽 → 安全帽` | 353 |
| R3 缩写展开（`SUV` 13 / `XCMG` 3） | 16 |
| R2 同源：残留英文/错译（formwork / hard hats） | 11 |
| 拼写·断词（`KOBEL CO→Kobelco`、`DOOSan→Doosan`） | 2 |
| R1 语义反转（PPE） | 1 |
| **合计** | **383**（`caption_zh` 32 + `reason_zh` 351） |

逐处改动清单：`doc/2026-09-18/data/zh_term_fix/changes.tsv`（`row_id / field / category / before / after`）。
修正前快照：`data/zh_term_fix/aswin_hazard_manifest_zh.prefix.bak.jsonl`。

### 3.1 三条口径的落点示例

| 规则 | 修正前 | 修正后 |
|---|---|---|
| R1 | 左侧人员**不得使用PPE**。 | 左侧人员**未使用个人防护装备（PPE，Personal Protective Equipment）**。 |
| R2 | 中间的工人没有戴**硬帽**。 | 中间的工人没有戴**安全帽**。 |
| R2 | 五名工人穿着安全背心和**硬帽子**。 | 五名工人穿着安全背心和**安全帽**。 |
| R3 | 画面左侧有两辆**SUV**。 | 画面左侧有两辆**运动型多用途汽车（SUV，Sport Utility Vehicle）**。 |
| R3 | 挖掘机侧面印有“**XCMG**”品牌标识。 | 挖掘机侧面印有“**徐工集团（XCMG，Xuzhou Construction Machinery Group）**”品牌标识。 |
| R2 同源 | **formwork** 的工人未佩戴安全帽。 | **模板（formwork）**上的工人未佩戴安全帽。 |
| R2 同源 | 两名工人在**Form沃克**和另一名工人在**厄尔顿路**上… | 两名工人在**模板（formwork）**上，另一名工人在背景中的**土路**上… |
| R2 同源 | 右侧的两名为“**hard hats**”的工人… | 右侧的**坐着**的两名工人…（`sitting` 误译修正） |

### 3.2 校验（全量，已实测）

- `硬帽` 字段残留 **0**（修正前 383+ 处）；
- `不得使用PPE` 残留 **0**；
- `PPE / SUV / XCMG` 在**中文译文**中的未展开残留 **0**；
- 二次运行修正器改动 **0**（幂等）；
- 行数 **1276 → 1276**，`caption_major` 与 `boxes/bbox` 未动。

---

## 4. 线上 Label Studio 就地更新（不删标注）

复核项目已有 **31 条人工标注**（`total_annotations_number=31`）。`seed_aswin_ls.py --recreate`
会**连同标注一起删掉**，因此本轮改用 `update_aswin_ls_zh.py`：按 `row_id` 匹配，
**只 `PATCH` 任务的 `data`（meta_html）**，不触碰 `predictions/annotations`。

实测（`http://127.0.0.1:8085`，project id=3）：

| 项 | 结果 |
|---|---|
| 需更新（meta_html 变化） | **316** |
| 已一致（无需改） | 960 |
| 无匹配 row_id | 0 |
| PATCH 成功 / 失败 | **316 / 0** |
| 项目标注数 before → after | **31 → 31**（零丢失） |
| 全量复核 `meta_html` 含「硬帽」/「不得使用PPE」 | **0 / 0** |

> 其余 14 处 `PPE/SUV/XCMG` 的「未展开」命中**全部位于 meta_html 的 `EN:` 英文原文段**
> （保留原文，不属中文译文），非违规。

---

## 5. 未纳入本轮（诚实登记）

R1/R2/R3 之外的**语义错误**不在本轮判据内，原样保留（manifest 仍以 `caption_major=True` 在界面提示 403 行硬错误）。
抽样残留（修正器**不会**自动改，避免静默改语义）：

| 类型 | 现存译文 | 英文原文 / 应为 |
|---|---|---|
| 名词误译 | 右侧戴着白色安全帽的工人佩戴着一条**发带**。 | `is wearing a slipper` → 拖鞋 |
| 名词误译 | 右侧的工人穿着短裤和**护目镜**。 | `is wearing slippers` → 拖鞋 |
| 名词误译 | 中间的工人…戴着一副**耳套**。 | `is wearing a pair of shorts` → 短裤 |
| 名词误译 | 中间那个人…戴着**一双溜冰鞋**。 | `a pair of slippers` → 拖鞋（row `…#439`） |
| 语序 | 戴红安全帽的**工人中间**没有穿长裤。 | `The worker with a red hard hat in the middle does not have long pants` |
| 漏译 | 工人在**沥青中间**没有戴安全帽。 | `The worker shoveling asphalt in the middle…`（shoveling 漏译） |
| 搭配 | 八名工人头戴橙色安全帽，**脚穿黄色安全帽**。 | 应为「戴着黄色安全帽」，非「脚穿」 |
| 搭配 | 挖掘机位于一台**带有极的构建设备件**的前方。 | 机翻腔，需重译 |

此外以下品牌/专名仍保留拉丁写法（非缩写，按中文惯例可保留；如需中文化可加一轮）：
`Kobelco / Kato / Zoomlion / Doosan / SK200 / W-7`、`Kiosk`。

**建议**：与交接文档 §5 #5 合并做一次 aswin/hayden 的**术语表重翻 + 同一评委同一提示词 before/after**
（评委必须 DeepSeek `deepseek-flash`、`reasoning_effort:"none"`；Qwen3.8-27B 不可当文本评委）。

### 5.1 一个未动的 UI 标签（需用户决策）

Label Studio 的矩形框标签仍为 `rule1 个人防护(PPE)`——按 R3 严格说也缺全称。
**本轮未改**：该值是 label config 的枚举，线上 31 条标注里有 **41 个 `rectanglelabels` 结果项**、
1276 条 prediction 也都引用它；改动需**同步迁移标注 + prediction**，属破坏性操作，不宜擅自做。
如需，建议单独排一次「改 label config + 迁移 41 标注项 + 重发 1276 prediction」的操作。

---

## 6. 复现命令

```bash
cd /home/linaro/dsh/label-studio

# 1) 只看改动（dry-run，不写文件）
python3 tools/zh_fix.py data/aswin_hazard_manifest_zh.jsonl --report

# 2) 从审计产物重生成带修正的 manifest（幂等）
python3 tools/enrich_manifest_zh.py

# 3) 就地回写线上 Label Studio（先 dry-run 再执行；不动标注）
LS_PASSWORD='...' python3 tools/update_aswin_ls_zh.py --dry-run
LS_PASSWORD='...' python3 tools/update_aswin_ls_zh.py
```

---

## 7. 变更文件

| 路径 | 说明 |
|---|---|
| `label-studio/tools/zh_fix.py` | **新增**：R1/R2/R3 确定性修正器（SPECIAL/SIMPLE/ABBREV 三表 + `fix_text/fix_row`） |
| `label-studio/tools/update_aswin_ls_zh.py` | **新增**：线上任务 `data.meta_html` 就地回写（保标注） |
| `label-studio/tools/enrich_manifest_zh.py` | 修改：落盘前调用 `fix_row()` |
| `label-studio/data/aswin_hazard_manifest_zh.jsonl` | 修改：383 处译文字段修正 |
| `doc/2026-09-18/data/zh_term_fix/` | 归档：修正器 / 回写脚本 / `changes.tsv` / 修正前快照 |
