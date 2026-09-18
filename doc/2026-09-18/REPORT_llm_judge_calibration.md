# Qwen3.8-27B 自评质检（LLM-as-judge）校准报告

日期：2026-09-18 ｜ 目的：在全量复核 21,329 个译文字段之前，先验证评委可靠性
服务：GPU0 `:8100`（常驻 PID 643233）+ GPU1 `:8101`（本次新起），greedy（temperature=0），`enable_thinking=False`（除注明外）
校准集：`/root/translate_logs/judge_calib.jsonl`（16 条，人工标注：11 BAD / 4 GOOD / 1 MINOR）

---

## 0. 结论（一句话）

**Qwen3.8-27B 在关闭 thinking 时不能充当可靠评委**：它几乎对每一条都判"有错"，
并且会**编造证据**——包括把正确译文（`excavator→挖掘机`、`hard hats→安全帽`）列为错误、
把提示词里的模板占位符（`<英文片段> || <中文片段>`）原样抄出来当"错误"。
**开启 thinking 后推理质量明显更好，但会被 `max_tokens` 截断（答案没生成），且 ~13 s/条（全量 21,329 条 ≈ 40 h）。**

因此**不建议**用当前形态的 LLM 评委跑全量：产出会是"几乎全部 major"，没有信息量。

---

## 1. 提示词版本对照（16 条校准集）

| 版本 | 题式 | thinking | recall（BAD 被抓到） | 误报率（GOOD 被判错） | 备注 |
|---|---|---|---|---|---|
| V1 | 6 类清单 + JSON | 关 | 1.00 | **1.00** | 16/16 判 major |
| V3 | 硬错误 JSON（强调别误报） | 关 | 1.00 | **1.00** | 把 `钢筋结构`、`硬帽`、`挖掘机` 当错误 |
| V4 | 自由格式「无 / 一行证据」 | 关 | 1.00 | **1.00** | 抄出 `<英文片段> \|\| <中文片段>` 模板 |
| V6 | 二选一 OK/ERR | 关 | 1.00 | **0.80** | 4/5 好例也判 ERR |
| V7 | 4 例 few-shot + OK/ERR | 关 | 0.73 | **0.60** | 目前最好，仍不可用 |
| V6T | 二选一 OK/ERR | 开 | 0.36* | 0.40* | *大量 parse_error：1600 token 被推理耗尽、答案未生成 |

## 2. 压力测试（3 条明显正确 + 3 条人工注入错误，最简题面）

| 用例 | V6 二选一 | V7 few-shot | V1 清单 |
|---|---|---|---|
| GOOD `The rear view of a mobile crane.` → `移动式起重机的后视图。` | ❌ ERR | ✅ OK | ❌ major(term) |
| GOOD `Multiple workers not wearing hard hats nor high-visibility vests.` → `多名工人未佩戴安全帽，也未穿着高能见度背心。` | ❌ ERR | ✅ OK | ❌ major(term) |
| GOOD `A worker with a yellow hard hat is washing the concrete pavement…` → `戴黄色安全帽的工人正用水管冲洗混凝土地面。` | ❌ ERR | ❌ ERR | ❌ major(garble) |
| BAD-num `Eleven people…` → `七人…` | ✅ ERR | ✅ ERR | ⚠️ major（类别乱） |
| BAD-neg `Person on the left not using PPE.` → `左侧人员必须使用PPE。` | ✅ ERR | ✅ ERR | ⚠️ major（类别乱） |
| BAD-term `The back view of an excavator in a wood.` → `一台木工钻床的背面。` | ✅ ERR | ✅ ERR | ⚠️ major（类别乱） |
| **得分** | **3/6** | **5/6** | **0/6** |

## 3. 观察到的失效机制（原文可复现）

1. **模板补全**：`<英文片段> || <中文片段> || <类别>` 被原样输出为"错误"（V4）。
2. **术语表回显**：V1 的提问示例里的 `crane=起重机 / drum=隔离桶` 被当成该行的"错误"吐出。
3. **把"不同措辞"当错误**：`rebar structure→钢筋结构`、`hard hat→安全帽` 被判 term 错误。
4. **复述原文时自我污染**：thinking 文本里把 `sides` 写成 `side`、`workers` 写成 `worker`，
   再基于这个被自己改写的版本判错。
5. **必须找错**的先验：即便题面写"绝大多数译文没有硬错误""漏报可接受、误报不可接受"，仍 100% 判错。

## 4. 建议的替代路径

| 方案 | 覆盖 | 可信度 | 成本 |
|---|---|---|---|
| A. 确定性规则全量审计（数字/否定/乱码重复/术语/漏译/英文残留） | **全量 21,329** | 高（可复现、可逐条定位） | 分钟级，纯读产物 |
| B. 模型只当"候选提出"，规则+人工复核确认 | 全量候选 | 中 | ~2 h |
| C. **VL 多模态复核**（喂图 + caption EN/ZH，thinking 开、大 token 预算） | 抽样 300–500 | 有望最高（caption 对不对要看图） | ~1–3 h |
| D. thinking 开 + max_tokens≈3000 抽样 | 抽样 ~500 | 中高（需先修截断） | ~1 h/卡 |

*本报告所有数字来自服务器实测日志；人工标注（GOOD/BAD）为抽样通读判断。*
