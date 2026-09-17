# Group C — `lmms-lab/LLaVA-NeXT-Data` 数据集分析

- 服务器: `ssh -p 38024 root@connect.westb.seetacloud.com`
- 数据根: `/root/autodl-tmp/mm-datasets/lmms-lab__LLaVA-NeXT-Data/`
- 只读分析；未写入/删除任何远端数据，未遍历整个 137G（仅 parquet metadata + 轻量列读取 + 抽样分片）。
- 磁盘实占: `data/` = 86G，`llava_next_raw_format/` = 51G（合计 ≈137G）。

## 0. 结论速览

| 项目 | 结论 |
|---|---|
| 任务类型 | 视觉指令微调（visual instruction tuning）多轮对话数据，LLaVA/LLaVA-NeXT 训练格式 |
| 数据格式 | 250 个 parquet 分片（每片 3117–3118 行）+ 原始发布格式 `llava_next_raw_format/`（1 个 processed.json + 11 个图片 tar.gz） |
| 总样本数 | **779,289**（250 分片 pyarrow `metadata.num_rows` 精确求和） |
| 单图/多图 | **单图**：`image` 为单个 `struct<bytes, path>`（不是 list）；抽样中每条含图样本恰有 1 个 `<image>` 标记，无一样本 >1 |
| 图像为空比例 | **40,688 / 779,289 = 5.22%** 为纯文本样本（无图，`<image>` 标记数 = 0）；其余 738,601 (94.78%) 单图 |
| 单轮/多轮 | 单轮 = **38.45%**（299,619），多轮(≥2) = **61.55%**（479,670）；均值 6.03 轮，中位 4，最大 275 |
| 子集构成 | coco 50.2%、vg 11.1%、ocr_vqa 10.3%、gqa 9.3%、synthdog-en 5.1%、dvqa 2.6%、chartqa 2.4%、docvqa 1.3%、laion-gpt4v 1.0%、llava_pretrain_lcs558k 0.9%、ai2d 0.4%、sharegpt4v 0.3%（+5.2% 无图样本无子集目录） |

---

## 1. 任务类型：多轮视觉指令对话（LLaVA 格式）

依据：
1. 官方 README.md（数据集内）自述：本数据集是 **LLaVA-NeXT / LLaVA-NeXT(stronger) 指令微调阶段（instruction tuning stage）** 所用的全部数据，共 ~779k 行；任务维基为 "visual instruction-following data"、"visual conversation"、"Multimodal Document/Chart Data"。
2. 字段结构就是 LLaVA 训练格式：`conversations: list<struct<from: string, value: string>>`，`from ∈ {human, gpt}`（全库统计 `human` 与 `gpt` 各 4,700,533 条，严格 1:1 交替）。
3. 图像以文本占位符 `<image>` 出现在首条 human 消息开头，例如 `"<image>\nWhat are the colors of the bus in the image?"` —— 这是 LLaVA 多模态对话模板的标志性写法。
4. README 明确给出把本数据集转换为 LLaVA-NeXT 训练格式（json + image 文件夹）的官方代码。

## 2. 数据格式

### 2.1 `data/` — parquet 分片
- 文件数：**250**，命名 `train-00000-of-00250.parquet` … `train-00249-of-00250.parquet`。
- 每片行数 3117–3118（`MIN 3117, MAX 3118`），HF 元数据声明的 `num_examples = 779289` 与实际求和一致。
- HF parquet schema（`train-00000` 实测）：

```
id: string
conversations: list<element: struct<from: string, value: string>>
  child 0, element: struct<from: string, value: string>
data_source: string
image: struct<bytes: binary, path: string>
```
- `image` 内嵌真实图片字节：抽样前 3 行 `bytes` 长度 189406/199709/205203 与 114291/214720/148352，magic = `ffd8ffe0`（JPEG SOI/APP0），确认是 **JPEG 二进制内嵌**，`path` 只是冗余文件名（如 `000000033471.jpg`，无目录前缀）。

### 2.2 `llava_next_raw_format/` — 数据集原始（LLaVA json + 图片文件夹）格式
`ls` 结果：
- `llava_next_raw_format_processed.json`（844,510,658 B ≈ 805 MB）：**单个 JSON 数组**，每条记录 `{"id", "conversations", "data_source": "llava_next_raw_format", "image": "<相对路径>"}`；`image` 是**路径字符串**（如 `"coco/train2017/000000033471.jpg"`），不是内嵌字节 —— 即 LLaVA 训练用的 json 标注文件。
- `llava_next_raw_format_images_1.tar.gz` … `_11.tar.gz`：11 个图片压缩包（10 个 ≈5.29 GB，`_11` 为 0.37 GB）。tar 内为**按子集分目录的原始图片**，如 `vg/VG_100K/2321062.jpg`、`ocr_vqa/images/62435329.jpg`、`gqa/images/2349510.jpg`、`dvqa/images/bar_train_00157767.png`、`chartqa/train/png/*.png`、`synthdog-en/images/*.jpg`、`coco/train2017/*.jpg`、`laion-gpt4v-20231128/0000N/*`、`llava_pretrain_lcs558k/images/*`、`ai2d/abc_images/*`、`docvqa/images/*`、`sharegpt4v/*`（12 个顶层目录）。
- 与 README 一致：raw format 为 "de-compress it for json file and images with structured folder" 的发布形态。

**raw json 与 parquet 的关系与差异**：raw json 记录数 = **738,590**，全部带 image；parquet = 779,289 行，其中带图 `image.path` 非空 = 738,601（差 11 条，见 §9 限制）。即 **raw format 未包含 40,688 条纯文本样本**（`sharegpt4v` 类文本指令也在其中一部分）。

## 3. 样本数（精确）

- 方法：`pyarrow.parquet.ParquetFile(p).metadata.num_rows` 对 250 个分片求和，**不读数据**。
- 结果：**TOTAL_ROWS = 779289**；`num_files 250`；`MIN 3117, MAX 3118`；头 5 片均 3118，尾 5 片均 3117。
- 与数据集内 README 的 `num_examples: 779289`、`dataset_size: 124810423962.272`（≈124.8 GB 解压后）一致。

**抽样分片**：
- 深度抽样（读全列、逐样本统计）：`train-00000`、`train-00060`、`train-00124`、`train-00249`（共 12,469 样本）。
- 图像路径/ID 抽样：`train-00000 / 00030 / 00060 / 00100 / 00124 / 00180 / 00236 / 00249`。
- 分层抽样（每 10 片取 1，共 25 片，77,929 样本）用于轮数分布交叉验证。
- 全量统计（只读 `id`、`conversations`、`data_source` 轻量列，不读 `image.bytes`）：250/250 分片 → 精确轮数分布与子集构成。

## 4. 单图还是多图：**单图**

- Schema 铁证：`image: struct<bytes: binary, path: string>` —— 是**单个 struct**，不是 `list<struct<...>>`（对比：对话字段才是 `list<element: struct<...>>`）。物理 parquet schema 同样是 `optional group image { optional binary bytes; optional binary path; }`，无 repeated 层。
- 文本侧交叉验证：全库 779,289 条样本中，human 消息里 `<image>` 标记出现次数分布 = **{1: 738,601, 0: 40,688}**，不存在 ≥2 的样本。即每条样本最多一张图，LLaVA-NeXT 的"多图"能力在此指令数据中并未以多图样本形式出现。
- **纯文本（image 为空）比例 = 40,688 / 779,289 = 5.221%**（用 250 个分片 parquet row-group 统计中的 `image.path` null_count 精确求和；其中 `train-00236` 部分为空 167 条，`train-00237`…`train-00249` 共 13 片 100% 为空，其余分片 100% 有图）。
- 分层抽样端点一致：25 片中仅 `train-00240`（属纯文本分片段）为 0 标记，其余 24 片全部恰好 1 标记。

> 注意：`data_source` 字段在 parquet 中**全库恒为 `"llava-next-instruct"`**（779,289/779,289，一次全量扫描确认），不能用来区分子集；raw json 中则恒为 `"llava_next_raw_format"`。

## 5. 单轮还是多轮

### 5.1 `conversations` 结构
`list<struct<from, value>>`，`from ∈ {human, gpt}`（抽样未见 `user`/`assistant`/`system`）。全库 250 分片精确统计：
- `human` 4,700,533 条、`gpt` 4,700,533 条 → 严格成对、始终 human 起头、gpt 收尾，**无 system 轮、无连续同角色**。
- 轮数（= human 消息数 = len(conversations)/2）分布（779,289 样本全量，非抽样）：

| 轮数 | 样本数 | 占比 |
|---|---|---|
| 1 | 299,619 | **38.45%** |
| 5 | 116,455 | 14.94% |
| 20 | 83,807 | 10.75% |
| 4 | 59,062 | 7.58% |
| 3 | 53,889 | 6.92% |
| 2 | 31,162 | 4.00% |
| 8 | 20,465 | 2.63% |
| 7 | 18,411 | 2.36% |
| 6 | 15,607 | 2.00% |
| 9 | 12,297 | 1.58% |

- **单轮(1) = 299,619 (38.45%)；多轮(≥2) = 479,670 (61.55%)**；均值 6.03 轮，中位数 4，最大 275。无 0 轮（空 conversations）样本。
- 交叉验证（分层 25 片 / 77,929 样本）：单轮 38.37%、多轮 61.63%、均值 6.13、中位 4、最大 112 —— 与全量一致。
- 早前 4 片深度抽样因分片与子集强相关而失真，仅作参考：`train-00000` 单轮 0.83%（coco，4–5 轮为主）、`train-00124` 单轮 0.96%（vg/gqa，轮数长尾）、`train-00060` 类似、`train-00249` 单轮 49.12%（纯文本，1–2 轮）。**分片按子集聚集，因此轮数分布高度分片相关**：

| 分片 | 行数 | 平均轮数 | 最大轮数 | 说明 |
|---|---|---|---|---|
| train-00000 | 3118 | 4.54 | 6 | coco/llava-instruct |
| train-00020/30/40/50 | 3118/3118/3117/3117 | 1.00 | 1 | 单轮子集（如 ocr_vqa/textvqa 类） |
| train-00060–00080 | 3117 | 5.3–5.5 | 77–87 | 长多轮子集 |
| train-00120 | 3117 | 17.60 | 112 | 极长多轮 |
| train-00160–00180 | 3117 | ≈19.9 | 20 | vg region-caption 多轮（20 轮固定） |
| train-00230 | 3117 | 1.00 | 1 | 单轮 |
| train-00240 | 3117 | 2.18 | 21 | 纯文本段 |
| train-00249 | 3117 | — | — | 纯文本（单轮 49.1%） |

### 5.2 有图 vs 纯文本 × 单轮/多轮

| 子群 | 多轮(≥2) | 单轮(1) |
|---|---|---|
| 有图 (738,601) | 459,173 | 279,428 |
| 纯文本 (40,688) | 20,497 | 20,191 |

纯文本样本几乎一半是单轮（49.6%），有图样本单轮仅 37.8%。

## 6. 关键字段名 + 真实样例

字段：`id`(string)、`conversations`(list<struct<from,value>>)、`data_source`(string)、`image`(struct<bytes,path>)。
parquet 中 `data_source` 恒为 `llava-next-instruct`；`image.path` 仅为文件名（无子集目录），子集信息在 `id` 前缀或 raw json 的 `image` 相对路径中。

样例 1（coco，`train-00000`，id=`000000033471`，image.path=`000000033471.jpg`，3 轮）：
```
[{"from": "human", "value": "<image>\nWhat are the colors of the bus in the image?\nAnswer the question with GPT-T-COCO format."}, {"from": "gpt", "value": "The bus in the image is white and red."}, {"from": "human", "value": "What feature can be seen on the back of the bus?"}, {"from": "gpt", "valu  ...(截断至 300 字符)
```

样例 2（vg，`train-00180`，id=`VG_100K/2317533`，image.path=`2317533.jpg`，20 轮，region 坐标指令）：
```
[{"from": "human", "value": "<image>\nProvide a short description for the given region.\n[0.060,0.344,0.441,0.696]"}, {"from": "gpt", "value": "A little girl with a dark pink shirt."}, {"from": "human", "value": "[0.270,0.676,0.847,0.868]"}, {"from": "gpt", "value": "4 candles lit on a cake."}, ...(截断)
```

样例 3（synthdog-en，`train-00236`，id=`synthdog_en_00074_005206`，单轮 OCR）：
```
[{"from": "human", "value": "<image>\nOCR this image section by section, from top to bottom, and left to right. Do not insert line breaks in the output text. If a word is split due to a line break in the image, use a space instead."}, {"from": "gpt", "value": "ut. The unsupe rvised equivalent of cla ...(截断)
```

样例 4（纯文本，`train-00249`，id=`M9Nb6So_0`，`image.path = None`）：
```
[{"from": "human", "value": "financial inflation"}, {"from": "gpt", "value": "Content Brief: Understanding Financial Inflation\n\nTop level keywords: inflation, financial inflation, monetary policy, economic growth, central bank ...(截断)
```

raw json 中的对应记录（同一条 id=`000000033471`）：
```json
{"id": "000000033471", "conversations": [...], "data_source": "llava_next_raw_format", "image": "coco/train2017/000000033471.jpg"}
```

## 7. 子集构成

parquet 的 `data_source` 无区分度，`image.path` 也只留文件名，因此子集口径取自 **raw format processed.json 的 `image` 相对路径顶层目录**（流式解析 738,590 条，全部带图；占比分母用 779,289 全量）：

| 子集 | 样本数 | 占带图样本 | 占全库 |
|---|---|---|---|
| coco | 391,219 | 52.97% | **50.20%** |
| vg | 86,417 | 11.70% | **11.09%** |
| ocr_vqa | 80,000 | 10.83% | **10.27%** |
| gqa | 72,140 | 9.77% | **9.26%** |
| synthdog-en | 40,093 | 5.43% | **5.14%** |
| dvqa | 20,000 | 2.71% | 2.57% |
| chartqa | 18,317 | 2.48% | 2.35% |
| docvqa | 10,194 | 1.38% | 1.31% |
| laion-gpt4v-20231128 | 8,048 | 1.09% | 1.03% |
| llava_pretrain_lcs558k | 6,989 | 0.95% | 0.90% |
| ai2d | 3,247 | 0.44% | 0.42% |
| sharegpt4v | 1,926 | 0.26% | 0.25% |
| 无图纯文本（无子集目录） | 40,688 | — | 5.22% |

合计 738,590 + 40,688 = 779,278（余 11 条差异见 §9）。共 12 个子集目录。
parquet 内 `id` 前缀也能部分识别子集（全量 id 模式扫描，779,289 条）：

| id 模式 | 样本数 | 对应子集 |
|---|---|---|
| `\d{12}`（如 `000000033471`） | 381,413 (48.94%) | coco 主 + gqa/ocr_vqa 混合 |
| `\d{1,11}`（如 `2384780`） | 149,771 (19.22%) | gqa / ocr_vqa |
| `VG_100K/*` | 86,417 (11.09%) | vg |
| 短哈希/下划线 id（如 `M9Nb6So_0`, `Tom_Felton_2`） | 40,691 (5.22%) | sharegpt4v / llava-instruct 文本 |
| `synthdog_en_*` | 40,093 (5.14%) | synthdog-en |
| `coco2014_caps_*` | 20,000 (2.57%) | coco2014 caption 子集（单轮 "Provide a one-sentence caption for the provided image."） |
| `bar_train_*.png` | 20,000 (2.57%) | dvqa |
| `chart_qa_two_col_*` | 18,317 (2.35%) | chartqa |
| 其它（docvqa/ai2d/laion 等） | 11,292 + 3,247 + 8,048 | — |

**注意**：`synthdog-en`、`dvqa`、`chartqa`、`ai2d`、`vg`、`coco2014_caps` 的 id 前缀可区分；但 **coco / gqa / ocr_vqa 三者的 id 都是纯数字，无法只靠 id 区分，只能用 raw json 的 `image` 目录**。`laion-gpt4v-*` id 8,048 条可区分。

### 7.1 分片 ↔ 子集聚集关系（按分片主导 id 模式，近似）

分片按子集成块排列（每片 ~3,117 行），因此**单看某几个分片的统计会严重偏离全库**：

```
000-052 (53片) 12位数字(coco/ocr_vqa/gqa)     115-138 (24片) 数字id(gqa/ocr_vqa)
053-054 ( 2片) 数字id                        139-153 (15片) 12位数字
055-055 ( 1片) 短哈希(文本)                   154-181 (28片) vg
056-057 ( 2片) laion-gpt4v                   182-206 (25片) 数字id
058-084 (27片) 12位数字                       207-210 ( 4片) 12位数字
085-090 ( 6片) coco2014_caps                 211-216 ( 6片) dvqa
091-114 (24片) 12位数字                       217-217 ( 1片) ai2d
                                             218-223 ( 6片) chartqa
                                             224-236 (13片) synthdog-en
                                             237-249 (13片) 短哈希(sharegpt4v/纯文本)
```

README 亦说明相对 LLaVA-1.5 的改动：移除 TextCaps，新增 DocVQA、SynDog-EN、ChartQA、DVQA、AI2D，并加入 LAION-GPT-V / ShareGPT-4V 高质量指令数据；因授权问题 **15k 用户指令数据未发布**。

## 8. 证据（命令 + 输出摘要）

```bash
# 1) 结构
ssh -p 38024 root@... 'ls -la <root> && ls <root>/data | wc -l && ls -la <root>/llava_next_raw_format/'
# -> data 250 个 train-XXXXX-of-00250.parquet；raw 目录 11 个 *_images_N.tar.gz + processed.json(844510658 B)
#    data=86G  llava_next_raw_format=51G  (du -sh)

# 2) 行数（仅 metadata，不读数据）
python - <<'PY'
from pyarrow.parquet import ParquetFile; import glob
fs=sorted(glob.glob('.../data/*.parquet')); print(len(fs), sum(ParquetFile(p).metadata.num_rows for p in fs))
PY
# -> 250 779289   (MIN 3117 MAX 3118)

# 3) schema
python - <<'PY'
import pyarrow.parquet as pq; print(pq.ParquetFile('.../train-00000-of-00250.parquet').schema_arrow)
PY
# -> id: string / conversations: list<element: struct<from: string, value: string>>
#    / data_source: string / image: struct<bytes: binary, path: string>
# -> num_rows 3118 row_groups 32

# 4) 全量 data_source + image.path null（row-group 统计求和，250/250 分片可用）
# -> data_source GLOBAL: [('llava-next-instruct', 779289)]
# -> image.path nulls TOTAL: 40688 / 779289 = 5.221%
#    13 个分片 100% null（train-00237..00249），train-00236 部分 null(167)，其余 236 片 0 null

# 5) 全量轮数 + <image> 标记统计（只读 id, conversations）
python - <<'PY'  # 250 分片
# -> TOTAL 779289 ; human 4700533 / gpt 4700533 (1:1)
# -> <image>/sample: {1: 738601, 0: 40688}
# -> 1轮 299619 (38.45%) / 5轮 116455 (14.94%) / 20轮 83807 (10.75%) / 4轮 59062 (7.58%) / 3轮 53889 (6.92%)
# -> single(1)=38.45%  multi(>=2)=61.55%  mean 6.03 median 4 max 275
PY

# 6) raw format: 流式 json 解析 + tar 目录
python - <<'PY'  # json.JSONDecoder.raw_decode 流式，8MB 缓冲
# -> TOTAL_RECORDS 738590 ; 全部含 image，无 null
# -> 子集 top: coco 391219 / vg 86417 / ocr_vqa 80000 / gqa 72140 / synthdog-en 40093 / dvqa 20000 ...
# -> 1轮 37.83% multi 62.17%（仅带图部分）
PY
tar -tzf llava_next_raw_format_images_11.tar.gz | awk -F/ '{print $1}' | sort | uniq -c
# -> vg 624 coco 620 ocr_vqa 577 gqa 489 synthdog-en 291 dvqa 161 chartqa 133 docvqa 75
#    laion-gpt4v-20231128 54 llava_pretrain_lcs558k 49 ai2d 22 sharegpt4v 10

# 7) 内嵌图像字节（只验长度/magic，不打印字节）
# -> bytes_len [189406, 199709, 205203] magic ['ffd8ffe0',...]  path ['000000033471.jpg',...]

# 8) 分层抽样 25 片（77,929 样本）交叉验证
# -> single(1)=38.37% multi=61.63% mean 6.13 median 4 max 112
```

## 9. 限制与不确定项

- **11 条对账差异**：parquet 中 `image.path` 非空 = 738,601，而 raw json 记录 = 738,590（差 11）。可能来源：(a) 少量样本 `image.path` 为空串（非 null，未被 null_count 计入）；(b) raw json 发布端的少量缺漏/去重。未逐条比对 137G 数据，故不强行归因；数量级不影响任何结论（<0.002%）。
- 抽样分片明确列出（§3）；轮数/子集的大盘数字均为**全量精确统计**而非抽样外推。
- 未解压任何 tar.gz（只 `tar -tzf | head/awk`），未读取 `image.bytes` 全量（仅 3 行验 magic）。
- 是否有比 12 个目录更细粒度的来源标注：parquet 无此字段，raw json 的 `data_source` 恒为 `llava_next_raw_format`，故子集最多只能解析到 12 个目录级。
- §7.1 的分片↔子集映射是按每片"主导 id 模式"推断的近似块边界（少数分片为混合子集），非逐行精确归属。
- README 提及的 "15k 用户指令数据未发布" 可见于纯文本/短哈希 id 样本（如 `train-00249` 的 marketing-plan 类提示），但无法从字段上严格区分其来源批次。
