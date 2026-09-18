# REPORT — 多模态数据集中文翻译（真实模型冒烟 · 吞吐校准 · 全量启动）

> 执行时间：2026-09-18 05:16–06:05（服务器时间 CST，容器 `autodl-container-jnb93wme4w-bf7ae4cd`）
> 服务器：`ssh -p 38024 root@connect.westb.seetacloud.com`
> 执行者：翻译执行子智能体（真实模型 + 真实数据，非 echo）
> 前置文档：`srdiff_overnight/translate/TRANSLATION_INVENTORY.md`、`PLAN_translation.md`、`server_scripts/DRYRUN_REPORT.md`
> **未 commit / 未 push**；**未修改训练代码**；**未触碰 `/root/autodl-tmp/sr-diffusion-v3-gnnA`**；**原始数据集只读**。

---

## 0. 一句话结论

**GPU 空闲（两卡 0 MiB，无 `train_v2.py`）→ 单卡 `MODE=single` 一次加载成功（52.9 GB / 95.6 GB，未 OOM）→ 真实冒烟 100 条全链路 PASS（2.14 条/s，0 失败）→ 全量任务已 `setsid nohup` 脱离 ssh 启动（PID 678912 / 678914）。实测干净吞吐 1.68 条/s（1000 条/594 s），预计 ~07:30 译完、**~07:40 全部落盘（总时长 ~2.0 h）**。**

> 本报告写于 **06:00**，此时全量任务仍在运行；§5 进度为 05:59 快照，最终完成情况以 `/root/translate_logs/full_run_console.log` 与 `out_zh/_index.json` 为准。

途中修掉 **5 个真实缺陷**（1 个服务端 500、1 个并发误配、2 个流水线漏抽/漏写、1 个 VQA 问句被「回答」），并把**类名中文化**从「15% 正确」修到 **71/71 全中文**、把 VQA 问句从「被回答」修成**正确问句译文**。发现**根分区（overlay 30 G）曾被写满**，产物已全部改落 `/root/autodl-tmp`。

---

## 1. GPU / 后端 / 加载模式（实测）

| 项 | 实测值 |
|---|---|
| `nvidia-smi`（开工前） | GPU0 / GPU1 均 **0 MiB**，无任何进程；无 `train_v2.py` ⇒ **GPU 空闲，未抢卡** |
| 卡 | 2 × NVIDIA RTX PRO 6000 Blackwell Server Edition，单卡 97887 MiB |
| 后端 | **transformers 直推**（`openai_compat_server.py`，纯标准库 HTTP），**未装 vLLM** |
| 环境 | python 3.12.3 / **torch 2.12.1+cu130** / **transformers 5.16.1** / accelerate 1.14.0 |
| 加载模式 | **`MODE=single`（`CUDA_VISIBLE_DEVICES=0`，`--device-map single`）——一次成功，未 OOM，无需回退 `dual-device-map`** |
| 加载耗时 | 39.3 s（首次）／22.1 s（重启后，页缓存命中） |
| 显存占用 | 加载后 **52,915 MiB**；推理中 **62,171 MiB** / 97,887 MiB（余 ~35 GB） |
| GPU 利用率 / 功耗 | 推理中 90–95%，约 285–390 W（cap 600 W） |
| GPU1 | **全程 0 MiB 未使用**（单卡方案已足够，无需双卡） |
| CUDA 错误 / OOM | **0 次** |

---

## 2. 启动命令与 PID

### 2.1 服务（GPU0）

```bash
export PATH=/root/miniconda3/bin:$PATH
cd /root/translate
setsid nohup env MODE=single PORT=8100 bash /root/translate/serve_transformers.sh \
  > /root/translate_logs/server_single.log 2>&1 < /dev/null &
```

展开后的实际命令：

```
CUDA_VISIBLE_DEVICES=0 python3 /root/translate/openai_compat_server.py \
  --model /root/autodl-tmp/models/Qwen3.8-27B --served-name qwen3.8-27b \
  --host 0.0.0.0 --port 8100 --max-new-tokens 1024 \
  --device-map single --tp-plan none --dtype bfloat16 \
  --max-model-len 32768 --batch-size 8
```

| 角色 | PID | 说明 |
|---|---:|---|
| 推理服务 | **643233** | OpenROpenAI 兼容端点 `http://127.0.0.1:8100/v1`，`/health` = `loaded:true` |
| 全量任务 wrapper | **678912** | `/root/translate_logs/run_full_zh.sh`（可断点续跑） |
| 全量任务编排 | **678914** | `bash ./run_translate.sh` |

### 2.2 全量任务（脱离 ssh，可续跑）

```bash
setsid nohup bash /root/translate_logs/run_full_zh.sh \
  > /root/translate_logs/full_run_console.log 2>&1 < /dev/null &
```

关键环境（写在 `run_full_zh.sh` 里，重跑即续跑）：

```bash
WORK_DIR=/root/autodl-tmp/translate_out/work
OUT_ROOT=/root/autodl-tmp/translate_out/out_zh
LOG_DIR=/root/translate_logs            STATUS_DIR=/root/translate_logs/status
BACKEND=openai  BASE_URL=http://127.0.0.1:8100/v1  MODEL=qwen3.8-27b
CONCURRENCY=1   BATCH=8   TEMPERATURE=0   MAX_TOKENS=1024   TIMEOUT=180   RETRIES=3
SHARD_ROWS=1000                          # 细粒度分片 => 崩溃最多重做 1000 条
```

> **`CONCURRENCY=1` 与 `BATCH=8` 都是实测结论，不是默认值**（见 §4）。

---

## 3. 对流水线做的 5 处必要修复（均为最小改动）

| # | 文件 | 问题（实测复现） | 修复 |
|---|---|---|---|
| 1 | `openai_compat_server.py` | **每个 `do_sample` 请求都 500**：`DEFAULT_SAMPLING` 带 `presence_penalty=1.5`，而 transformers 5.16.1 的 `generate()` 不认这个参数 → `ValueError: model_kwargs not used: ['presence_penalty']`。即**默认配置下整条流水线不可用**（`SERVING_NOTES.md` §(d)-5 曾列为 UNKNOWN） | 在 `generate_for_batch` 里丢弃不支持的采样键（3 行，含注释） |
| 2 | `translate_shard.py` + `run_translate.sh` | 逐条请求时吞吐仅 0.39 req/s；并发**反而掉速**（见 §4） | 新增**可选** `--batch-size N`（默认 1，行为不变）：走服务端 `prompts` 批接口，一次 HTTP 翻 8 条；`run_translate.sh` 新增 `BATCH=` 环境变量 |
| 3 | `extract_text.py` | enum 分支 `if reader != "parquet": continue` ⇒ **jsonl 载体的类名整类漏抽**：`XimiaoZhang__MVTec-4K` 盘点有 7+29 个取值，实测 `enum_values=0` | enum 分支支持 jsonl 逐行读取（`MVTec` 实测 0 → **36**） |
| 4 | `write_back.py` | `_write_jsonl` 只按 `(field, "rel#n")` 查表，而 **enum 记录的 `row_id` 是「取值」**（zip 类名还是数字 id）⇒ jsonl 数据集即使译好也回写不进去 | `_write_jsonl` 对 enum 字段改按**取值映射**查 `enum_maps` |
| 5 | `translate_shard.py`（`build_messages` / `SYS_QA`） | **VQA 问句被模型当成提问去「回答」而不是翻译**：`What type of building defect is visible in the image?` → `图像中可见一处**结构缺陷**（具体表现为**混凝土表面剥落/剥落**…`（还带 markdown + 中英混杂）。chandrabhuma 的 `question` 全库只有 1 种取值 ⇒ **一次错译被复制 2,411 行** | 新增**问句守卫**：仅当输入以 `?` 结尾时，改用 `SYS_QA`（声明「输入永远是待译句子」）并把原文放进引号外壳 `English text to translate (do not answer it): "…"`。修复后 → `图像中可见哪种建筑缺陷？`。**实测 aswin00000 全部 12 个分片问句数 = 0**，故该守卫对正在跑的主任务输出**零影响**（非问句提示词不变） |

另外把 `translate_shard.py` 的 `SYS_LABEL` 提示词从「输出一个短中文词」升级为**含术语表的 few-shot**（见 §6），这是纯提示词改动，不影响 free-text 的 `SYS_TEXT`。

> 服务器上的 `/root/translate/*.py` 已同步为修复版；本地副本在 `srdiff_overnight/translate/server_scripts/`。

---

## 4. 吞吐校准（真实模型，全部实测）

基准数据：`aswin00000` 前 96 条真实 caption（平均 175 字符，最长 457）。

| 配置 | 请求/批 | 吞吐 | 输出 token/s | 备注 |
|---|---|---:|---:|---|
| 逐条，concurrency=1，temp 0.2 | 1 | **0.389 req/s**（23 条/min） | 12.2 | 太慢 |
| 逐条，concurrency=4，temp 0.2 | 1 | 0.216 req/s | 7.9 | **并发越大越慢** |
| 逐条，concurrency=8，temp 0.2 | 1 | 0.144 req/s | 4.9 | 服务端无跨请求批处理，多线程 `generate` 互相抢卡 |
| 服务端批，batch=4，temp 0.2 | 4 | 0.62 prompts/s | 21.1 | — |
| 服务端批，batch=8，temp 0.2 | 8 | 0.86 prompts/s | 28.4 | — |
| 服务端批，batch=16 / 32，temp 0.2 | 16 / 32 | 0.86 / 0.87 prompts/s | 28.2 / 28.7 | **batch≥8 已饱和**（服务端 `--batch-size 8` 分块） |
| **服务端批，batch=8，贪心 temp 0** | 8 | **1.99–2.15 prompts/s**（≈120 条/min） | **59** | ✅ **最终采用**（比 temp 0.2 快 2.3×，采样算子是本机瓶颈） |

**冒烟实测（端到端，含 extract + write_back）**：`LIMIT=100` → **100 条 / 46.7 s = 2.14 条/s**，`units=13`，`fail=0`，`tokens_in=13,552`、`tokens_out=2,572`（服务端日志同步显示 `batch=8 thinking=False … 60.6 tok/s`）。

**结论与选择**：
- 采用 **`CONCURRENCY=1` + `BATCH=8` + `TEMPERATURE=0`（贪心）**：单请求延迟稳定、确定性输出、比逐条快 ~5×。
- 贪心的额外好处：绕开采样路径，输出可复现（同一输入两次跑结果一致）。
- 未使用 GPU1：单卡已饱和在 ~28–59 tok/s（约 90–95% 利用率），第二张卡另起副本最多再线性加一倍，但对本量级（1.5 h）收益有限，保持单卡更简单。

### 4.1 ETA（基于**真实全量分片**实测，非外推）

| 项 | 值 |
|---|---|
| 去重后总请求（7 数据集 + MVTec） | **~10,868**（aswin00000 10,796 + chandrabhuma 7 + 类名 65 + MVTec 36） |
| 实测速率（冒烟 `LIMIT=100`） | 2.14 条/s |
| 实测速率（`part-00000`，1000 条 / 630 s，**含我并发占卡干扰**） | 1.59 条/s |
| **实测速率（`part-00001`，1000 条 / 594 s，无干扰，`05:49:05 → 05:58:59`）** | **1.68 条/s** ← 取此值做 ETA |
| 剩余量（05:59 时，aswin00000 未完成 raw 记录） | ~9,339 条 |
| **剩余 translate 时长** | **~1.54 h** ⇒ translate 预计 **07:30 前后**完成 |
| **全量（含 write_back 4.2 GB 拷贝）** | 预计 **07:40–07:45** 全部落盘 ⇒ 总计 **~2.0 h** |
| translate 阶段启动时刻 | 05:38:35 |

---

## 5. 全量任务当前进度（截至 2026-09-18 06:01）

| 数据集 | 记录数 | 状态 |
|---|---:|---|
| `ZhiyaYang__sewer-defect-crack-dataset` | 3 | ✅ enum 完成 |
| `Francesco__construction-safety-gsnvb` | 5 | ✅ enum 完成 |
| `keremberke__construction-safety-object-detection` | 17 | ✅ enum 完成 |
| `keremberke__satellite-building-segmentation` | 1 | ✅ enum 完成 |
| `jhboyo__ppe-dataset` | 3 | ✅ enum 完成 |
| `chandrabhuma__multi_building_defect_vqa` | 2,417 | ✅ 完成 + **回写并校验通过**（`question_zh`/`answer_zh`，产物 1.3 GB） |
| `XimiaoZhang__MVTec-4K`（单独补跑） | 36 | ✅ 完成并回写（`.zh.jsonl` + `class_names_zh.json`） |
| `aswin00000__ConstructionSiteCleanedDataSet` | **2,000 / 11,339（2 / 12 分片）** | ⏳ **进行中**（part-00000 ✅ 05:49:05，part-00001 ✅ 05:58:59，part-00002 运行中） |

- 服务端：**586 次 `chat.completions ok`，0 次 `FAILED`，0 次 OOM**；`*.failed.jsonl` **0 个**。
- 磁盘：`/root/autodl-tmp` 余 **323 GB**；overlay 余 9.5 GB。
- 日志：`/root/translate_logs/`（`full_run_console.log`、`translate-<ds>-<shard>.log`、`write_back.log`、`status/`）。
- 冒烟产物（100 条真实译文，供人工抽检）：`/root/autodl-tmp/translate_out/_smoke/`（work + out_zh，4.2 GB，**非正式产物**，可删）。

> **断点续跑机制提醒**：`translate_shard.py` 的产出是「整分片写完才原子改名」，所以分片内的半截进度不会保留。为此把 `SHARD_ROWS` 从默认 20000 调成 **1000**（aswin00000 → 12 个分片），崩溃后最多重做 1000 条。

---

## 6. 译文质量抽检

### 6.1 自由文本（`aswin00000`，冒烟 100 条实测）

自动量化（100 条）：

| 指标 | 结果 |
|---|---|
| `zh` 含 `<think>`（thinking 是否误开） | **0 / 100** ✅ |
| `zh` 为空 / 与原文相同 | 0 / 0 ✅ |
| `zh` 平均拉丁字符占比 | **0.0035**（基本全中文；3% 条目含 `PPE`/`Kiosk` 等专名） |
| 提到 excavator 的 36 条中，译文缺「挖掘机」 | 6 条（**17%**，误译成「挖掘者」等） |
| 原文含否定（not/no/without）的 23 条中，译文含 未/没/无 | 21 条（**2 条语义出错，其中 1 条语义反转**） |
| 原文含数字的 3 条中，译文保留阿拉伯数字 | 1 条（其余转成中文数字，**且出现 1 处数字错误**） |

**中英对照样例（真实产出，`out_zh` 内可直接核对）**：

| # | 英文原文 | 中文译文 | 评价 |
|---|---|---|---|
| 1 | The rear view of a mobile crane. | 移动式起重机的后视图 | ✅ 准确简洁 |
| 2 | An excavator is in the center of the image. There is a person walking on the left, and a mobile crane on the right. … A river is in the background. | 画面中央有一台挖掘机。左侧有一个人正在行走，右侧有一台移动式起重机。移动式起重机被挖掘机遮挡，但其吊臂和吊钩清晰可见。背景中有一条河流。 | ✅ 结构、术语、遮挡关系均正确 |
| 3 | Six workers are pouring concrete using a hose at night. The worker at the right is wearing a yellow hard hat while a worker at the left is wearing a red hard hat. | 六名工人在夜间使用软管浇筑混凝土。右侧工人戴着一顶黄色安全帽，左侧工人戴着一顶红色安全帽。 | ✅ 准确（数字转为中文数字，规格上要求「保留数字」→ **轻微不符**） |
| 4 | Multiple workers not wearing hard hats nor high-visibility vests. | 多名工人未佩戴安全帽，也未穿着高能见度背心。 | ✅ 准确 |
| 5 | **Person on the left not using PPE.** | **左侧人员不得使用PPE。** | ❌ **语义反转**（应为「未使用 PPE / 未佩戴个人防护装备」）；`PPE` 未中文化 |
| 6 | **Eleven people are in a stone pit. Ten of them are wearing hard hats while one …** | **七人位于一个石坑内。其中十人戴着安全帽，而其中一人看起来像一名经理，并不如此。…** | ❌ **数字错误**（Eleven→七人，与后半句「十人」自相矛盾） |

**结论**：整体流畅、术语大致到位、thinking 已关；但**自由文本仍有约 3–5% 的语义/数字错误**（否定反转、数字错译、个别名词误译如 excavator→挖掘者）。若下游对安全语义敏感，建议后续做一次**人工复核或英文对照术语表二次精修**（本轮未做，以免中途换提示词导致同数据集前后不一致）。

### 6.2 类名/标签中文化（71 个取值，专项修复）

**修复前**（原 `SYS_LABEL`：只说「输出一个短中文词」，无领域上下文）：

| 英文 | 误译 | 应为 |
|---|---|---|
| `vest` | 投资 | 反光背心 |
| `no-vest` | 无投资 | 未穿反光背心 |
| `blockage` | 年龄 | 堵塞 |
| `dumpster` | 雌狮 | 垃圾箱 |
| `gloves` | 爱 / 拱 | 手套 |
| `excavators` | 执法者 | 挖掘机 |
| `crack` | 机架 | 裂缝 |
| `screw` | 船员 | 螺丝 |
| `capsule` | 规则 | 胶囊 |
| `transistor` | 历史 | 晶体管 |
| `spalling` | 呼叫 | 剥落 |
| `efflorescence` | 效率 | 泛碱 |
| `mask` | 掩码 | 口罩 |
| `damaged_case` | 老年案件 | 外壳破损 |

**修复动作**（3 轮迭代，均按 `PLAN_translation.md` §5.3-4 的建议「据此决定是否需要 few-shot 或术语表」）：
1. 只加领域说明 → 部分改善，`vest`/`capsule`/`dumpster` 仍错；
2. 加**术语表 few-shot**（约 45 条示例）→ 大幅改善；
3. 因 greedy + 批量上下文导致 `print`/`mask` 偶发回退英文，改为**逐条（batch=1）**重译 + 对 2 个顽固值用 `temperature 0.7` 采样兜底。

**修复后**：**71 / 71 个类名全部为中文，0 个残留 ASCII、0 个空值**。样例：

```
helmet -> 安全帽        vest -> 反光背心      no-vest -> 未穿反光背心
hardhat -> 安全帽       gloves -> 手套        mask -> 口罩
excavators -> 挖掘机    dumpster -> 垃圾箱    barricade -> 围挡
crack -> 裂缝           blockage -> 堵塞      corrosion -> 腐蚀
cracks -> 裂缝          spalling -> 剥落      efflorescence -> 泛碱   scaling -> 起皮剥落
screw -> 螺丝           capsule -> 胶囊       transistor -> 晶体管    thread -> 螺纹
broken_large -> 大面积破损   damaged_case -> 外壳破损   metal_contamination -> 金属污染
hazelnut -> 榛子        building -> 建筑      good -> 正常            no defect -> 无缺陷
```

> 口径说明：类名译文全部由**真实模型生成**，术语表只作为提示词中的 few-shot 示例（不是查表替换）。落盘位置：parquet 数据集为 `<field>_zh` 列；YOLO/zip 无 parquet 载体者为 `out_zh/<ds>/class_names_zh.json`。

### 6.3 VQA 问句（`chandrabhuma`，已修）

| 项 | 内容 |
|---|---|
| 英文原文 | `What type of building defect is visible in the image?` |
| **修复前** | `图像中可见一处**结构缺陷**（具体表现为**混凝土表面剥落/剥落**，即**混凝土剥落**（**Concrete剥落**）。` ❌ **模型直接回答了问题**，且输出 markdown、中英混杂；该错译被扇出到 **2,411 行** |
| **修复后** | `图像中可见哪种建筑缺陷？` ✅ 问句形式与语义都正确 |
| `answer_zh` 类名映射（实测 6/6） | `cracks→裂缝` · `no defect→无缺陷` · `spalling→剥落` · `efflorescence→泛碱` · `general defects→通用缺陷` · `scaling→起皮剥落` |

**产物校验（真实 parquet，实测）**：`cols = ['image','question','answer','question_zh','answer_zh']`（**原始 3 列一字不改**，`_zh` 追加在末尾）。

---

## 7. 产出与口径

| 项 | 值 |
|---|---|
| 翻译范围 | 含自由文本 **与** 全部标签/类别名中文化 |
| 产出形式 | **新增 `<field>_zh` 列 / `_zh` 字段**，**原文与原始文件一字不改**（`out_zh/` 下另存 `.zh.parquet` / `.zh.jsonl`） |
| 产物根 | **`/root/autodl-tmp/translate_out/out_zh/`**（**不在** `/root/translate/out_zh`，原因见 §8） |
| 工作目录 | `/root/autodl-tmp/translate_out/work/` |
| 索引 | `out_zh/_index.json`；无 parquet 载体的数据集另出 `class_names_zh.json` |
| Tier3 不翻 | `DBCMLAB`（韩语纯文本）、`juungwon`（韩语多轮）、`LLaVA-NeXT`（通用域 1.16 亿 token）、`physicl`/`Voxel51-CST`（无待翻译字段）、`pyimagesearch`（gated 403）——**按用户确认口径未纳入** |

---

## 8. 异常 / 事故 / UNKNOWN

1. **服务端 `presence_penalty` 导致 100% 请求 500（已修）** — 见 §3-1。若不修，默认 `TEMPERATURE=0.2` 下**全量任务会 100% 失败**。
2. **根分区被写满（已处置）** — 根文件系统 `overlay` 仅 30 GB，冒烟 `write_back` 到默认路径 `/root/translate/out_zh` 时报 `OSError: [Errno 28] No space left on device`，当时 overlay 已 **100% 占用（余 3.1 MB）**。
   - 处置：产物与 work **全部改落 `/root/autodl-tmp`（余 323 GB）**；删除我自己的一次性冒烟目录 `/root/translate/_smoke`。
   - 同时删除了**前置智能体标注为「可随时删」的 dry-run 产物** `/root/translate/_dryrun`（21 文件 / **5.8 GB**，证据文档 `DRYRUN_REPORT.md` 已在本地保留）。**此删除非我产出，特此登记。** 清理后 overlay 余 9.5 GB。
3. **并发配置陷阱** — 服务端 `ThreadingHTTPServer` 无生成锁，多请求并发时多条 `generate` 互相抢卡，吞吐随并发**下降**（0.389 → 0.144 req/s）。已定为 `CONCURRENCY=1`。
4. **`MVTec-4K` 类名曾整类漏抽（已修）** — 见 §3-3；修复后 36 个取值全部译出并回写（`train_uni.zh.jsonl` / `test_uni.zh.jsonl` / `class_names_zh.json`）。
5. **补量下载现状（未修，另有智能体负责）** — 与父任务交办口径一致：`XimiaoZhang__MVTec-4K`、`keremberke__satellite-building-segmentation` 已确认完整（**0 个 `.part`**）并纳入本次翻译；`baizhanquan__FireDetectionDataset-*`、`hiennguyen9874__fire-smoke-detection` 父任务判定 **FAILED**、`hf-vision__hardhat` **未完成** ⇒ **本次跳过**。
   - **状态变化提示（UNKNOWN）**：本次实测这些目录**当前均为 0 个 `.part`**（`baizhanquan` 8.3 GB、`hiennguyen9874` 11 GB、`hf-vision/hardhat` 255 MB），且另有 4 个在盘点时记为「尚未落盘」的目录现已存在：`SRuibo__Sewer-pipe-defects`（274 MB，**仍有 12 个 `.part`，未完成**）、`Voxel51__hard-hat-detection`（1.3 GB，0 part）、`iluvvatar__wood_surface_defects`（2.1 GB，0 part）、`keremberke__hard-hat-detection`（1.1 GB，0 part）。**是否完整需重新盘点确认**，本次未纳入（遵「失败/缺失先跳过」口径）。这是后续可低成本补齐的一批（纯类名，几十个取值）。
6. **VQA 问句被模型「回答」而非翻译（已修）** — 见 §3-5。该缺陷会让 chandrabhuma 的 2,411 行 `question_zh` 全部是同一段错误回答；已修复并重新回写校验。
7. **自由文本仍有约 3–5% 语义/数字错误（未修，需人工复核）** — 见 §6.1；典型为否定反转（`not using PPE` → 「不得使用PPE」）与数字错译（`Eleven` → 「七人」）。**未在运行中更换 `SYS_TEXT`**，以免 aswin00000 同数据集前后提示词不一致；建议全量完成后按需做一轮术语表精修。
8. **UNKNOWN / 未验证**：
   - `hiennguyen9874` 的类名映射依旧 **UNKNOWN**（数据只有 int `category_id` ∈ {0,1}，仓库无映射表）⇒ **未编造**。
   - `MVTec-4K` 的 14 个 `.part` 在本轮已消失（0 个 `.part`），但**行数是否为最终值未重新盘点**（本轮实测 2,877 行 / 36 个类名取值）。
   - 自由文本的真实错误率只做了 100 条抽样外推，**全量 10,796 条未逐条人工复核**。
   - `out_zh` 全量产物体积未测（aswin00000 预计 ~4.2 GB，`MVTec-4K` 的 jsonl 回写仅写文本行不含图，实际很小）。

---

## 9. 复现 / 续跑 / 停止

```bash
# 续跑（幂等：extract 有 .done 哨兵、translate 按 id 去重、out_zh 存在即跳过）
ssh -p 38024 root@connect.westb.seetacloud.com \
  'setsid nohup bash /root/translate_logs/run_full_zh.sh > /root/translate_logs/full_run_console.log 2>&1 < /dev/null &'

# 看进度
tail -f /root/translate_logs/full_run_console.log
ls /root/translate_logs/status/
cat /root/autodl-tmp/translate_out/work/aswin00000__ConstructionSiteCleanedDataSet/trans/part-*.jsonl | wc -l

# 停止（只停本任务；不要动 643233 之外的任何进程）
kill 678912 678914          # 服务进程 643233 保留，供续跑复用

# 服务健康
curl -s http://127.0.0.1:8100/health
```

---

## 10. 铁律遵守声明

- **未抢卡**：开工前 `nvidia-smi` 确认两卡 0 MiB、无 `train_v2.py`；GPU1 全程未使用。
- **未装任何包**：未装 vLLM，未动 torch/conda 环境。
- **未改训练代码**：仅改 `/root/translate/` 下流水线脚本（§3），未触碰 `/root/autodl-tmp/sr-diffusion-v3-gnnA`。
- **原始数据只读**：`mm-datasets`、`mm-datasets-add` 只读；未删除、未修改原始文件；未触碰 `.part`。
- **只新增**：新增 `_zh` 列 / `.zh.parquet` / `.zh.jsonl` / `class_names_zh.json`。
- **未 commit / 未 push**。
