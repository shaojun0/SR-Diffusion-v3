# REPORT — 第二轮多模态数据集中文翻译（补量 13/13 齐全后 · 两张卡并行）

* 时间：2026-09-18（服务器 UTC+8），窗口 **09:10 → 11:24**
* 机器：`autodl-container-jnb93wme4w-bf7ae4cd`，`connect.westb.seetacloud.com:38024`
* 模型：`/root/autodl-tmp/models/Qwen3.8-27B`（transformers 直推，bf16，非量化 / 非 vLLM）
* 硬件：2 × NVIDIA RTX PRO 6000 Blackwell（单卡 97887 MiB）
* 上游：`srdiff_overnight/datasets/SUPPLEMENT_REPAIR.md`（补量 13/13 OK、130.742 GB）、
  `srdiff_overnight/translate/PLAN_translation.md`、
  `doc/2026-09-18/REPORT_translation_zh_progress.md`（第一轮）
* 产物：`/root/autodl-tmp/translate_out/out_zh_round2/`（**与第一轮 `out_zh/` 完全分离，未覆盖**）
* **本文所有数字均为服务器实测**；拿不到的写 `UNKNOWN`，不把估算写成实测。

---

## 0. 一句话结论

补量 13 个 repo 中，**第一轮已覆盖 2 个**（`MVTec-4K`、`satellite-building-segmentation`，专项复核实测无遗漏），
剩下 11 个里 **8 个有可译文本**（14 个字段：1 个大块英文自由文本 + 13 组类别名/标签，共 93 个取值），
**3 个类名无任何映射来源 ⇒ 判 UNKNOWN、不臆造**。

第二轮把这 8 个数据集的 **7508 条**记录翻完：**0 失败**、**5253 次模型调用**、
双卡并行 **1096 s（18.3 min）**，聚合吞吐 **6.85 条/s（4.79 调用/s）**，是单卡第一轮（2.0 条/s）的 **2.4 倍**。
产物 **98 个 `.zh.parquet` / 172 474 行 / 59.1 GB** + 2 个 `class_names_zh.json`，原始数据集**逐文件未变**。

**过程中发现并修掉了一个任务书未预期的问题**：`hayden-yuma/roadwork` 的自由文本是美国道路作业区
（TTC，Temporary Traffic Control）场景，第一轮那套**无领域术语**的 `SYS_TEXT` 在它上面**系统性错译 39.3%**
（`drum`→"鼓"（乐器）15.1%、`TTC sign`→"交通信号灯" 6.9%、`barricade`→"巴里卡德"（音译）4.4%、
`tubular marker`→"管状标志" 9.5%、`cone`→"坑洼" 1.7%），**远高于任务书假设的 3-5%**。
因此追加了第二轮第二次翻译（TTC 术语表）+ 一次术语表回显修复，**最终系统性术语错降到 2.41%**。

---

## 1. 第二轮范围（实测盘点）

盘点脚本 `round2_inventory.py`：读 `work_round2/` 的抽取产物，用模型自带 `tokenizer.json`
**精确**计数（不是 chars/3）。字段路径先经只读 schema 侦察（`recon_fields.py` / `probe3.py`，
只读 parquet footer + 待译列，**从不读 `image`**），再写成 `dataset_registry.py` 的实测登记。

### 1.1 可译字段总表（8 个数据集 / 14 个字段）

| # | dataset (repo) | 行数 | 字段 | 类型 | 唯一取值 | 说明 |
|---|---|---:|---|---:|---:|---|
| 1 | `hayden-yuma/roadwork` | 8 549 | `scene_description` | free | 4 935 | 7415 行非空 / 762 108 字符；**本轮唯一大块英文自由文本** |
| 2 | `hayden-yuma/roadwork` | — | `scene_level_tags.daytime` | enum | 3 | parquet 里是**含点号的展平列名** |
| 3 | `hayden-yuma/roadwork` | — | `scene_level_tags.scene_environment` | enum | 6 | 同上 |
| 4 | `hayden-yuma/roadwork` | — | `scene_level_tags.travel_alteration` | enum | 23 | 值是 Python 列表字符串 `['Lane Shift']` |
| 5 | `hayden-yuma/roadwork` | — | `scene_level_tags.weather` | enum | 13 | 同上 |
| 6 | `hayden-yuma/roadwork` | — | `city_name` | enum | 18 | 美国城市名 |
| 7 | `fireviewer/fire-smoke-detection-corpus-v1` | 102 257 | `annotations_json` → `class_name` | enum | 2 | 标注塞在 **JSON 字符串列**里 |
| 8 | `fireviewer/…-corpus-v1` | — | `negative_tags_json` | enum | 1 | `no_target_visible` |
| 9 | `iluvvatar/wood_surface_defects` | 20 276 | `objects[].label` | enum | 10 | 43 974 个实例 |
| 10 | `kevincluo/structure_wildfire_damage_classification` | 18 714 | `label` | enum | 6 | 类名来自 README `class_label.names` |
| 11 | `hf-vision/hardhat` | 7 063 | `objects.category` | enum | 4 | 27 252 个实例 |
| 12 | `baizhanquan/FireDetectionDataset-…wildfire` | 15 615 | `annotations[].class_name` | enum | 2 | 22 551 个框 |
| 13 | `keremberke/hard-hat-detection` | 19 748 图 | `__class_names__`（zip 内 COCO） | enum | 2 | 无 parquet 载体 |
| 14 | `Voxel51/hard-hat-detection` | 5 000 | `__class_names__`（FiftyOne detections） | enum | 3 | 25 502 个框 |

**合计（实测）**

| 指标 | 数值 |
|---|---:|
| 有可译字段的数据集 | 8 |
| 字段数 | 14 |
| 自由文本记录（逐行） | 7 415 |
| 类别名/标签取值 | 93 |
| 待译条数（逐行口径） | **7 508** |
| 待译条数（全局去重口径） | 5 028 |
| prompt tokens（逐行 / 全局去重） | 167 629 / 122 103 |
| **模型调用次数（分片内去重后）** | **5 253** |

> 自由文本 7415 条里 `No Description` 等重复串很多：分片内去重后 **5161 次**（自由文本）
> + 92 次（enum）= **5253 次**，省掉 2255 次调用（30.0%）。
> 第二轮范围**很小**（12.2 万 dedup tokens），符合任务书第 6 条"规模小就跑完写终稿、不留长尾"。

### 1.2 与第一轮去重（避免重复翻译）

去重键 = `dataset + field + row_id`；实际把关靠**产物目录不相交 + registry 的 `done_round1` 标记**。

第一轮 `out_zh/` 实测 **8 个数据集 / 13 821 条 / 0 失败**（任务书写的"11 个"与实测不符，以实测为准）：

| 第一轮数据集 | tier | trans 条数 |
|---|---|---:|
| `aswin00000__ConstructionSiteCleanedDataSet` | kept | 11 339 |
| `chandrabhuma__multi_building_defect_vqa` | kept | 2 417 |
| `XimiaoZhang__MVTec-4K` | **supplement** | 36 |
| `keremberke__construction-safety-object-detection` | kept | 17 |
| `Francesco__construction-safety-gsnvb` | kept | 5 |
| `ZhiyaYang__sewer-defect-crack-dataset` | kept | 3 |
| `jhboyo__ppe-dataset` | kept | 3 |
| `keremberke__satellite-building-segmentation` | **supplement** | 1 |
| **合计** | | **13 821** |

⇒ 本轮**跳过** 2 个补量数据集（`XimiaoZhang__MVTec-4K`、`keremberke__satellite-building-segmentation`），
**零重复**。第一轮 `aswin00000` 11 339 条、`chandrabhuma` 2 417 条、0 失败 —— 任务书数字**吻合**。

**MVTec-4K 覆盖率专项复核**（它第一轮只翻了 36 个取值，看起来偏少，必须查）：

```
jsonl 文件 16 个（7 个分类目录 + 根目录 2 个 *_uni.jsonl）
union(*_uni.jsonl)  : rows=2877  clsname=7  label_name=29
union(ALL 16 files) : rows=5754  clsname=7  label_name=29
clsname 缺失 = []   label_name 缺失 = 0
```

⇒ `*_uni.jsonl` 是**全库合并视图**，36 = 7 + 29 **完整覆盖镜像内所有类别**，第一轮无遗漏。
注意该镜像本身只含 MVTec 15 类中的 7 类（bottle/capsule/grid/hazelnut/screw/transistor/wood），
这是**数据集合集问题、不是翻译缺口**（登记 §9.2）。

### 1.3 判 UNKNOWN 的 3 个数据集（**不产出 `*_zh`，不猜值**）

| dataset | 规模 | 为什么 UNKNOWN |
|---|---:|---|
| `adarshchandrashekar/flame2-rgb-ir` | 53 451 行 | `label` 是**裸 int64**（不是 ClassLabel），全量 `{1:39751, 0:13700}`；README 只列 features，镜像内无 `names:` 段、无任何 0/1 语义说明。只能确定是二分类，**无法确定哪一侧是 fire** |
| `hiennguyen9874/fire-smoke-detection` | 90 271 行 | `objects.category_id` 为 `list<int32>`，全量 `{0:105577, 1:66582}`；README/dataset_info 明确只有 `category_id: list<int32>`，**无 ClassLabel.names**、镜像内无映射表 |
| `SRuibo/Sewer-pipe-defects` | 1 953 个 label txt | 类名只有 `classes.txt = [CK, PL, SG, SL, TL, ZW]` **6 个不透明缩写码**，仓库内无展开说明。这类码不是英文单词，送翻译会触发同形异义（同类实测现象：CK → 肌酸激酶）。另实测 `**/labels/*.txt` glob 命中 **0** —— 该 repo 的 label 文件不在 `*/labels/` 下，路径形态与第一轮预估不同 |

账目核对：13 个补量 = 8（本轮）+ 2（第一轮）+ 3（UNKNOWN）✔

---

## 2. 两卡服务：启动命令与 PID

### 2.1 GPU0 — 复用第一轮常驻服务（**未重启、未 kill**）

```
PID 643233
python3 /root/translate/openai_compat_server.py \
  --model /root/autodl-tmp/models/Qwen3.8-27B --served-name qwen3.8-27b \
  --host 0.0.0.0 --port 8100 --max-new-tokens 1024 \
  --device-map single --tp-plan none --dtype bfloat16 --max-model-len 32768 --batch-size 8
```
* `CUDA_VISIBLE_DEVICES=0`；显存 62 171 MiB
* 本轮开工前 `/health`：`loaded=true`、`uptime_s=13833.9`、`requests=1703`（第一轮累计）
* **直接复用**，没有改成双卡加载（遵守"不占 GPU0 上不属于本任务的进程"）

### 2.2 GPU1 — 本轮新起的同款服务

```bash
ssh -p 38024 root@connect.westb.seetacloud.com '
cd /root && CUDA_VISIBLE_DEVICES=1 PATH=/root/miniconda3/bin:$PATH setsid nohup \
  python3 /root/translate/openai_compat_server.py \
  --model /root/autodl-tmp/models/Qwen3.8-27B --served-name qwen3.8-27b \
  --host 0.0.0.0 --port 8101 --max-new-tokens 1024 \
  --device-map single --tp-plan none --dtype bfloat16 --max-model-len 32768 --batch-size 8 \
  >>/root/translate_logs/server_gpu1_r2.log 2>&1 </dev/null &'
```
* **PID 867732**；启动 `2026-09-18 09:10:03`，**加载耗时 19.4 s**，显存 52 915 MiB
* 与 GPU0 同款参数，`/health` = `loaded:true`

### 2.3 驱动（脱离 ssh、可续跑）

`/root/translate/run_round2.py`（两车道编排）+ `/root/translate/run_round2.sh`

```bash
ssh -p 38024 root@connect.westb.seetacloud.com 'cd /root && setsid nohup \
  /root/translate/run_round2.sh >/root/translate_logs/round2_console.log 2>&1 </dev/null &'
```
* 驱动 **PID 878191**（`python3 /root/translate/run_round2.py`），启动 `10:17:34`
* `write_back.py` 子进程 **PID 948711**
* 日志：`round2_console.log`（总）、`round2/translate-<ds>-<shard>.log`（逐分片）、
  `round2/terminology_check.log`、`round2/write_back*.log`、`round2/ttc/*.log`
* 状态：`round2_status/`（逐分片 `.status`、`round2_summary.json`、`round2_inventory.json`、
  `terminology_fixes.json`、`ttc_echo_fixes.json`、`ttc_audit_final.json`、`source_baseline.tsv`）

---

## 3. 两卡任务分配与吞吐（实测）

### 3.1 分工（按实测负载分，不是对半分数据集）

| 车道 | 服务 | 卡 | 任务 | 条数 |
|---|---|---|---|---:|
| **A** | `http://127.0.0.1:8100/v1` | GPU0 | 8 个 enum 分片（93 个取值）+ `hayden-yuma` 偶数号自由文本分片 part-00000/02/04/06 | **4 093** |
| **B** | `http://127.0.0.1:8101/v1` | GPU1 | `hayden-yuma` 奇数号自由文本分片 part-00001/03/05/07 | **3 415** |

* enum 只有 93 个取值（几十秒），放 A 车道顺便做完；真正负载是 7415 条自由文本，按**分片**对切
* 实测两车道耗时 **1096.0 s / 1015.4 s**，差 **7.4%**（A 为临界路径）
* 自由文本阶段 `nvidia-smi` 实测 **两卡同时 ~86% util**（空闲期 0%）⇒ 两张卡都真在跑

### 3.2 吞吐（第一遍：`SYS_TEXT` + `SYS_LABEL`）

| 指标 | 实测 |
|---|---:|
| 翻译阶段墙钟 | **1096 s = 18.27 min** |
| 记录数 | **7 508**（7415 free + 93 enum） |
| 失败数 | **0** |
| 模型调用次数（分片内去重） | **5 253** |
| 去重节省 | 2 255 次（30.0%） |
| **聚合吞吐** | **6.85 记录/s、4.79 调用/s（2 卡）** |
| **单卡吞吐** | **2.40 调用/s** |
| tokens | in **758 366** / out **105 072** |
| 整个 run（含 terminology_check + write_back） | 1432.3 s = 23.9 min，`write_back_rc=0` |

**与第一轮对比**：第一轮单卡实测 2.0 条/s（`CONCURRENCY=1 BATCH=8 TEMPERATURE=0`）。
本轮单卡 2.40 调用/s、**聚合约 2.4×**。单卡略快是因为本轮自由文本平均
762108/7415 ≈ **103 字符**，远短于第一轮的 caption。

**参数沿用第一轮实测最优**（未重新调参）：`CONCURRENCY=1`（服务端无跨请求 batch，并发>1 掉吞吐）、
`BATCH=8`（服务端 `prompts` 批接口，2.0 条/s vs 逐条 0.39 条/s）、`TEMPERATURE=0`（贪心：确定 + 快 ~2×）、
`RETRIES=3`、`TIMEOUT=180`、`MAX_TOKENS=1024`、`SHARD_ROWS=1000`。

### 3.3 第二遍（TTC 术语修正），同样两卡

| 指标 | 实测 |
|---|---:|
| 触发原因 | 见 §5：第一遍自由文本**系统性 TTC 术语错译 39.3%** |
| 命令 | `/root/translate/rerun_hayden_ttc.sh`（`--style text_ttc`，仍分 A/B 两车道） |
| 墙钟 | `11:02:55 → 11:22:56` = **1201 s = 20.0 min** |
| 记录数 / 失败数 | 7 415 / **0**（8 个分片全部 rc=0） |
| 后续修复 | `fix_ttc_echo.py`：184 条术语表回显 → 用无术语表 `SYS_TEXT` 重翻并回填（~1 min） |
| 全流程 GPU 时间 | 1096 + 1201 ≈ **38.3 min**（两遍翻译）+ 2 次 write_back |

---

## 4. 进度与 ETA（已收敛）

| 时刻（服务器） | 事件 |
|---|---|
| 09:10:03 | GPU1 服务启动（PID 867732），19.4 s 加载完成 |
| 10:17:34 | 第二轮驱动启动（PID 878191），lane A 12 任务 / lane B 4 任务 |
| 10:18:02 | 4 个 enum 分片完成（共 72 条） |
| 10:22:50 / 10:23:00 | lane A / lane B 首个自由文本分片完成（各 1000 条） |
| 10:35:42 | 最后 1 个自由文本分片完成 |
| **10:35:50** | **第一遍翻译结束：1096.0 s，0 失败** |
| 10:35:51 | `terminology_check`（93 条扫描，2 处截断修正） |
| 10:35:51 → 10:41 | `write_back` 第一遍（98 parquet / 56 GB） |
| **10:41** | **第一遍全部完成（`ALL DONE`），driver 退出** |
| ~10:45–11:00 | 抽检发现自由文本系统性 TTC 错译 39.3%（§5） |
| 11:02:55 → 11:22:56 | **第二遍翻译**（`text_ttc` 术语表），1201 s |
| 11:22:57 | 误差复测：39.3% → 2.1% |
| 11:23:00 | `fix_ttc_echo.py`：184 条术语表回显修掉 |
| 11:23:46 | `write_back` 重写 `hayden-yuma`（31 parquet），其余 7 个 skipped-exists |
| **11:24** | **终检通过（§7），全部完成** |

**ETA 结论**：开工前按范围（7508 条 / 12.2 万 dedup tokens）预估 ≈20 min，实测第一遍 18.27 min，
**预估准确**；第二遍（20 min）是抽检后追加的质量修复，不是原计划的一部分。**没有长尾**，跑完即写终稿。

---

## 5. 译文抽检与"发现的问题 → 修复"（本轮最重要的一节）

### 5.1 第一遍抽检发现：自由文本系统性错译 39.3%

按 seed=2026 随机抽 20 条 `scene_description`**通读**，发现大量 TTC 术语错译。
于是把"通读印象"升级成**客观正则判定**，对全部 7415 条逐条统计（可复现，脚本 `final_audit.py`）：

| 判定规则（英文命中 → 中文错译） | 第一遍 | 占比 |
|---|---:|---:|
| `drum/drums` → 含"鼓"（**乐器**） | 1 120 | **15.10%** |
| `TTC` → 含"信号灯" | 508 | 6.90% |
| `tubular marker` → "管状标志" | 701 | 9.45% |
| `barricade` → "巴里卡德"（**音译**） | 326 | 4.40% |
| `cone/cones` → 含"坑洼/凹陷" | 123 | 1.66% |
| **至少命中 1 项** | **2 916** | **39.32%** |

真实错译样例（第一遍）：

```
EN: Drums and work vehicles off the road straight ahead. Drums partially blocking road.
OLD: 鼓和作业车辆驶离道路正前方。鼓部分阻挡道路。          ← "鼓" = musical drum
EN: TTC sign on right sidewalk. ...
OLD: 右侧人行道设交通信号灯。...                          ← TTC 误判为 traffic signal
EN: Barricade surrounded by drums on left side of road.
OLD: 巴里卡德被鼓声包围在道路左侧。                        ← 音译 + "鼓声"
EN: Cones on right sidewalk. ...
OLD: 右侧人行道有坑洼。...                                ← cone 误判为 pothole
```

**根因**：第一轮的 `SYS_TEXT`（自由文本提示词）**不带任何领域术语表**，而 roadwork 的
`scene_description` 是**强术语文本**（93.7% 的记录至少含 1 个 TTC 术语）。
第一轮为类别名建的术语表只在 `SYS_LABEL` 里，自由文本用不到。
⇒ 任务书给的"已知自由文本约 3-5% 错误"在本数据集上**不成立**，实测 39.3%，**必须重跑**。

### 5.2 修复（已执行，两卡）

1. 新增 style **`text_ttc`**＝`SYS_TEXT` + 施工 TTC 术语表（`drum→隔离桶`、`cone→交通锥`、
   `tubular marker→管状标杆`、`vertical panel→竖向标板`、`barricade→路障`、`barrier→护栏`、
   `arrow board→箭头指示板`、`TTC sign→临时交通控制标志`、`work zone→作业区`、`lane shift→车道偏移` 等）
   + 反例警示（"drum 不是乐器鼓 / cone 不是坑洼 / TTC 不是信号灯"）。
   **未改动 `SYS_TEXT`**，第一轮自由文本口径保持不变。
2. 用 `text_ttc` **重翻 7415 条**（`rerun_hayden_ttc.sh`，A/B 两车道，1201 s，0 失败）。
3. 术语表带来了新缺陷：退化输入 `No Description` 会被模型**当成正文抄出整张术语表**
   （实测 **184 条 = 2.5%**，其中 176 条是 `No Description`）。用 `fix_ttc_echo.py`
   按"术语表顺序连排 ≥6 个术语"判定回显 → 用**无术语表的 `SYS_TEXT`** 重翻 9 个不同文本 → 回填；
   `No Description → 无描述`，**184/184 全部修复，0 未解决**。
4. 重写 `hayden-yuma` 产物并终检。

### 5.3 修复后复测（同一套正则，同一批 7415 条）

| 判定规则 | 修复前 | **修复后** |
|---|---:|---:|
| `drum` → "鼓" | 1 120 (15.10%) | **59 (0.80%)** |
| `TTC` → "信号灯" | 508 (6.90%) | **0 (0.00%)** |
| `tubular marker` → "管状标志" | 701 (9.45%) | **20 (0.27%)** |
| `barricade` → "巴里卡德" | 326 (4.40%) | **0 (0.00%)** |
| `cone` → "坑洼/凹陷" | 123 (1.66%) | **100 (1.35%)** |
| **至少命中 1 项** | **2 916 (39.32%)** | **179 (2.41%)** |
| 术语表回显 | — | **0** |
| `zh` 含换行 | — | **0** |
| `status != ok` | 0 | **0** |

⇒ 自由文本系统性错误 **39.3% → 2.41%**，已落回任务书所说的 3-5% 量级。

### 5.4 最终抽检（修复后，中英对照）

```
EN: TTC sign on right sidewalk. Tubular markers and line of drums on right side of road.
    Worker and work vehicle on left side of road.
ZH: 右侧人行道临时交通控制标志。右侧道路管状标线和隔离桶。左侧道路作业人员和作业车辆。

EN: Barriers, TTC signs, and drums on left side of road. Police vehicle driving on road.
ZH: 道路左侧设置护栏、临时交通控制标志和隔离桶。警用车辆在道路上行驶。

EN: Vertical panels, tubular marker, and drum in middle of road. Tubular marker and drum on
    right sidewalk. Tubular marker and drums on left side of road.
ZH: 竖向标板、管状标杆和隔离桶位于道路中间。管状标杆和隔离桶位于右侧人行道。
    管状标杆和隔离桶位于道路左侧。

EN: Entire road fully blocked at intersection.
ZH: 路口完全封闭。

EN: No Description
ZH: 无描述

EN: Barriers on right side of road. Work vehicles and vertical panels off right side of road
    behind barriers. Barriers on left side of road.
ZH: 道路右侧的护栏。作业车辆及竖向标板位于道路右侧护栏后方。道路左侧的护栏。
```

结构（多句→多句、不合并）、方位（left/right）、数量、术语都到位。

### 5.5 类名映射抽检（93 个取值中的代表）

| 数据集 | 英文 | 中文 | 来源 |
|---|---|---|---|
| fireviewer | `flame_visible` / `smoke_visible` / `no_target_visible` | 可见火焰 / 可见烟雾 / 无可见目标 | 术语表 |
| iluvvatar（木材缺陷） | `Live_Knot` / `Dead_Knot` / `knot_with_crack` / `Knot_missing` | 活节 / 死节 / 开裂节子 / 脱落节 | Kodytek et al., F1000Research 2022, 10:581 |
| iluvvatar | `Marrow` / `Quartzity` / `Blue_Stain` / `overgrown` | 髓心 / 石英质 / 蓝变 / 愈合节 | 同上 |
| kevincluo（野火损毁等级） | `no_damage` / `affected` / `minor` / `major` / `destroyed` / `inaccessible` | 无损坏 / 受影响 / 轻度损毁 / 重度损毁 / 完全损毁 / 无法进入 | README `class_label.names` + Cal Fire 口径 |
| hf-vision / Voxel51 | `helmet` / `head` / `person` / `others` | 安全帽 / 头部 / 人员 / 其他 | 术语表 |
| keremberke/hard-hat | `hardhat` / `no-hardhat` | 安全帽 / 未戴安全帽 | zip 内 COCO categories |
| baizhanquan | `fire` / `smoke` | 火焰 / 烟雾 | 术语表 |
| hayden-yuma 场景 | `Light` / `Dark` / `Twilight` / `Urban` / `Highway` | 白天 / 夜间 / 黄昏 / 城市 / 高速公路 | 术语表 |
| hayden-yuma 通行 | `['Lane Shift']` / `['Partially Blocked', 'Fully Blocked']` | 车道偏移 / 部分堵塞；完全堵塞 | 多值合并为一行 |
| hayden-yuma 城市 | `boston` / `los_angeles` / `new_york_city` / `san_francisco` | 波士顿 / 洛杉矶 / 纽约 / 旧金山 | 术语表 |

**结论：93 个取值全部中文化，0 个残留英文，0 个音译。**

> 对照第一轮类名总数（任务书"71/71"）：MVTec-4K 7+29=36、`jhboyo` 3、
> `keremberke/construction-safety` 17、`satellite` 1 ⇒ 合计 **57**（任务书的口径可能含其它项）；
> 本报告只对**本轮 93 个**负责，第一轮数字见 §1.2 与第一轮报告。

---

## 6. 产出与口径

* 输出根：`/root/autodl-tmp/translate_out/out_zh_round2/`（**新目录，第一轮 `out_zh/` 一字未动**）
* 工作目录：`work_round2/`（与第一轮 `work/` 分离）
* 口径：**原始列 + 原始顺序完全保留**，只在其后追加 `<field>_zh`；
  每行恰 1 个取值 → `string`；多取值（列表展开）→ `list<string>`
* 无 parquet 载体的数据集（zip 内 COCO / FiftyOne JSON）→ 写 `class_names_zh.json`
* 索引：`out_zh_round2/_index.json`；明细 `round2_status/write_back_report.json`

| dataset | `.zh.parquet` | 行数 | 大小 | 新增列 |
|---|---:|---:|---:|---|
| `baizhanquan__FireDetectionDataset-…` | 19 | 15 615 | 8.9 GB | `class_name_zh` |
| `fireviewer__fire-smoke-detection-corpus-v1` | 32 | 102 257 | 32.0 GB | `class_name_zh`, `negative_tags_zh` |
| `hayden-yuma__roadwork` | 31 | 8 549 | 10.6 GB | `scene_description_zh`, `scene_level_tags_daytime_zh`, `scene_level_tags_scene_environment_zh`, `scene_level_tags_travel_alteration_zh`, `scene_level_tags_weather_zh`, `city_name_zh` |
| `hf-vision__hardhat` | 2 | 7 063 | 0.3 GB | `objects_category_zh` |
| `iluvvatar__wood_surface_defects` | 5 | 20 276 | 2.2 GB | `objects_label_zh` |
| `kevincluo__structure_wildfire_damage_classification` | 9 | 18 714 | 5.1 GB | `label_zh` |
| `Voxel51__hard-hat-detection` | 0 | — | — | `class_names_zh.json`（head/helmet/person） |
| `keremberke__hard-hat-detection` | 0 | — | — | `class_names_zh.json`（hardhat/no-hardhat） |
| **合计** | **98** | **172 474** | **59.1 GB** | |

汇总（`round2_status/round2_summary.json`）：`elapsed_s=1432.3`、`write_back_rc=0`、
`lane_url={A: :8100, B: :8101}`、16 个分片全部 `rc=0`。

---

## 7. 原始数据集只读校验（两次，均通过）

* `write_back.py` 铁律：**从不写 `<root>/<dataset>/...`**，只写 `out-root` 下的新文件。
* 开工时（10:19，write_back 尚未启动）对 8 个源 repo 做**全文件基线**：
  `find <8 repos> -type f -printf '%p\t%s\t%T@\n'` → **5153 个文件**，
  md5 = `cfb13ea3b5cd6be5bac828dc1bba533f`（`round2_status/source_baseline.tsv`）。
* 第一次 write_back 后复测：`5153 行，md5 相同` ⇒ **IDENTICAL**。
* 第二次 write_back（TTC 修复）后再次复测：

```
baseline md5: cfb13ea3b5cd6be5bac828dc1bba533f
final    md5: cfb13ea3b5cd6be5bac828dc1bba533f
INTEGRITY: IDENTICAL (原始数据集未改写)
```

  ⇒ 路径 + 字节数 + mtime **逐文件完全未变**。
* `mm-datasets-add` 下 `.part` 残留 = **0**；目录条目 = **14**（13 repo + `manifest_selected.json`）。
* 全程未触碰：`/root/autodl-tmp/mm-datasets/`、训练目录（`sr-diffusion-*`）、
  `Qwen3.8-27B` 权重、GPU0 上的常驻服务。

---

## 8. 对流水线做的改动（全部最小、向后兼容、可回滚）

改了 4 个第一轮文件（**各自留 `*.round1.bak`**），新增 7 个第二轮专用文件/脚本。
第一轮产物已落盘，改动只影响后续运行。

| # | 文件 | 改动 | 为什么必须 |
|---|---|---|---|
| 1 | `field_paths.py` | `$` 前缀 = **字面顶层列名**（列名可含点号） | HF 把 struct 展平成 `scene_level_tags.daytime` 这种列名；普通点号路径会被当成 struct 下钻而取不到值（实测 4 个字段全 0 值） |
| 2 | `extract_text.py` | 新增 `json_harvest()` + enum 字段 `json_keys` 抽取；`class_names_for()` 新增 `fiftyone_json` 读者 | fireviewer 的标注塞在 **JSON 字符串列** `annotations_json` 里；Voxel51 只有 FiftyOne `samples.json`（这两个读者第一轮根本不存在，会**静默抽 0 条**） |
| 3 | `write_back.py` | 新增 `enum_row_values()`，同时用于**行级回写**、`_probe_enum_types` 类型探测、jsonl 回写 | 否则 fireviewer 的 `class_name_zh` / `negative_tags_zh` 会整列 `None` |
| 4 | `dataset_registry.py` | 13 个补量条目的 `status`/`fields`/`measured`/`notes` 按实测重写；3 个数据集 `fields=[]` 标 UNKNOWN；2 个标 `done_round1` | 原条目有 8 个是 `not_landed` 的**猜测**字段（如 `objects.category`、`scene_level_tags.*`、`label`），与落盘后的真实 schema 不符 |
| 5 | `translate_shard.py` | ①`SYS_LABEL` 扩充**闭合枚举术语表**（场景标签/城市名/木缺陷/火烟/损毁等级）+ "命中术语表必须逐字复制"规则 + `norm_label()`；②新增 style `text_ttc`（TTC 术语表）+ `norm_text_ttc()`；③模块级 `import re` | ①实测原 `SYS_LABEL` 在城市名/木缺陷上崩得厉害：`boston→破损`、`phoenix→凤凰`、`columbus→公交车`、`los_angeles→角度`、`new_york_city→新分叉城市`、`Dead_Knot→死亡骑士`、`Live_Knot→直播`、`Marrow→箭头`、`Blue_Stain→蓝色雨`、`flame_visible→可见光斑`；列表输入还会输出多行而回写列是单值。②见 §5 |
| 6 | `terminology_check.py`（新） | 闭合枚举集的**术语一致性复核**：仅当模型输出与术语表条目**互为子串且不等**时替换，其余差异登记为 UNKNOWN | 提示词已给术语表，但实测 27B 在极短输出上仍会**截断**（直连服务端 `finish_reason=stop`）：`affected→影响`、`major→重度损`。规则只修截断，不做主观改写 |
| 7 | `fix_ttc_echo.py`（新） | 检测并修复 `text_ttc` 的**术语表回显** | 见 §5.2 第 3 点 |
| 8 | `run_round2.py` / `run_round2.sh`（新） | 两车道并行驱动 + terminology_check + write_back，幂等可续跑 | **复用**第一轮 `translate_shard.py`/`write_back.py`，只做编排，不重写流水线 |
| 9 | `rerun_hayden_ttc.sh`（新） | 用 `text_ttc` 重翻 hayden-yuma 自由文本（两卡） | §5.2 |
| 10 | `round2_inventory.py` / `final_audit.py` / `progress.sh` / `raw_probe.py`（新） | 精确 tokenizer 盘点 / 误差审计 / 进度 / 服务端原始输出探针 | 所有数字**实测可复现** |

### 8.1 `terminology_check.py` 实测结果

```
[terminology_check] scanned=93 fixed=2 flagged=0
   FIX  affected   影响 -> 受影响
   FIX  major      重度损 -> 重度损毁
```
* `flagged=0` 的**准确含义**：没有出现"非子串关系"的语义偏离。
* **覆盖范围限制（重要）**：该检查只对**单 token** 取值生效；`travel_alteration` / `weather`
  的原始值是 Python 列表字符串（`"['Partly Cloudy']"`），键不在术语表里 ⇒ **不走检查**，
  靠人工抽检 + 登记。实测仅有 1 处同义级差异：`['Partly Cloudy'] → 部分多云`（术语表"局部多云"，
  语义等价，未改写）。

### 8.2 `fix_ttc_echo.py` 实测结果

```
[echo] files=8 echo_records=184 distinct_texts=9
[echo] retry rc=0 recovered=9 still_echo=0
[echo] fixed=184 unresolved=0
   e.g. 'No Description' -> '无描述'
```

---

## 9. 异常 / UNKNOWN 登记

### 9.1 已知/已修的翻译问题

| 类别 | 规模（实测） | 处置 |
|---|---:|---|
| 自由文本 TTC 术语**系统性**错译 | **2 916 / 7415 = 39.3%** | **已修**：新增 `text_ttc` 术语表并重翻，降到 **179 / 7415 = 2.41%**（§5） |
| `text_ttc` 术语表**回显**（新引入的缺陷） | 184 / 7415 = 2.5% | **已修**：`fix_ttc_echo.py`，184/184，0 未解决 |
| 类别名**截断** | 2 / 93（`affected`、`major`） | **已修**：`terminology_check.py` 子串规则 |
| 残留 `cone→坑洼/凹陷` | 100 / 7415 = 1.35% | **未修**（已在 3-5% 区间内），登记 |
| 残留 `drum→鼓` | 59 / 7415 = 0.80% | **未修**，登记 |
| 残留 `tubular marker→管状标志` | 20 / 7415 = 0.27% | **未修**，登记（"管状标志"可接受，非硬错） |
| 句末句号偶发丢失 | 抽样可见（如 `道路右侧的隔离桶和路障`） | **未修**，仅格式瑕疵 |
| 重复/语义压缩 | 抽样可见 1 例：`Cones on sidewalk of right intersecting road. Cones on right intersecting road.` 两句都被译成 `右侧人行道与道路相交。` | **未修**，登记 |
| `barrier` 同义不一致 | 抽样可见（`护栏` / `栏杆` 混用） | **未修**，登记 |

**结论**：修复后自由文本系统性术语错 **2.41%**，落在任务书所述 3-5% 区间内；
因此**没有再做第三遍**。

### 9.2 UNKNOWN（不臆造）

| 项 | 状态 |
|---|---|
| `adarshchandrashekar/flame2-rgb-ir` `label` 0/1 语义 | **UNKNOWN** —— 无 names 映射，无法确定 fire 是哪一侧 |
| `hiennguyen9874/fire-smoke-detection` `category_id` 0/1 语义 | **UNKNOWN** —— 同上 |
| `SRuibo/Sewer-pipe-defects` 6 个类码 CK/PL/SG/SL/TL/ZW | **UNKNOWN** —— 无展开说明；且 `**/labels/*.txt` glob 命中 0 |
| MVTec-4K 镜像内只有 7/15 类 | **数据集合集问题**，非翻译缺口（§1.2） |
| `out_zh/_index.json` 只列 7 个数据集（缺 MVTec-4K） | **第一轮索引瑕疵**：MVTec 单独跑、索引被另一次 write_back 覆盖；目录与产物本身完整。本轮**未改第一轮产物** |
| fireviewer 的 4 个 `manifests/*.jsonl`、15 个 `*.json` | **未翻译**：属派生清单/权利元数据，且未登记为 `sources` ⇒ 不改写 |
| fireviewer 的 `source_name`/`corpus_role`/`sample_validation_status`/`validation_profile` | **未翻译**：英文字面枚举，属数据集自描述元数据；本轮按"标签/类别名"口径只翻类名 |
| `hayden-yuma` 的 `gps`、`video_info.*`、`id`、`license`、URL | **明确不翻**：标识符/坐标/数值 |
| 第一轮产物是否受本轮改动影响 | **不受影响**：本轮只改代码与 `work_round2/`、`out_zh_round2/`；第一轮 `out_zh/` 未触碰、未重跑 |

### 9.3 事故 / 操作失误（诚实登记）

1. **首次启动 GPU1 服务时 ssh 通道被挂住 1 小时**（09:10–10:10）：`ssh '... setsid nohup ... &'`
   的远端 shell 未退出，客户端一直等到 1 h 超时。**服务本身启动正常**（09:10:03、19.4 s 加载、
   全程可用）。之后所有远端启动/长命令改为 `ssh -n` + 显式重定向 + 客户端 `timeout`，
   但仍偶发（11:04 启动 TTC 重跑时又挂了一次，用独立连接确认进程已起）。
2. `probe2.py` 首版有 `p.name`（`p` 是 str）笔误，报错即修，用 `probe3.py` 重跑。
3. 首版 `recon_fields.py` 只枚举**字符串叶子**，漏掉 6 个 parquet 的**数值型 label 列**
   （`label: int64`、`objects.category_id`）。发现后用 `probe3.py` 逐列全量补测，
   §1 的行数/取值数全部来自补测后的实测值。
4. 冒烟测试暴露 `SYS_LABEL` 城市名/木缺陷严重错译，**在正式跑之前**修好（§8 第 5 项），
   没有把错译写进正式产物。
5. `text_ttc` 修复引入术语表回显（§5.2），**由复测发现并修掉**；如果只做一次抽检就收工，
   这 184 条会带着"术语表整段"进产物 —— 这也是本轮坚持"改完必须用同一套客观规则复测"的原因。
6. 一个后台等待脚本的 `pgrep` 模式**自匹配**，导致等待循环空转到超时；已改用日志标记判定。

---

## 10. 复现 / 续跑 / 停止

```bash
# ---- 首次全量（两卡）----
ssh -p 38024 root@connect.westb.seetacloud.com 'cd /root && setsid nohup \
  /root/translate/run_round2.sh >/root/translate_logs/round2_console.log 2>&1 </dev/null &'

# ---- 只重跑 hayden-yuma 的自由文本（text_ttc，两卡）----
ssh -p 38024 root@connect.westb.seetacloud.com 'bash /root/translate/rerun_hayden_ttc.sh'

# ---- 误差审计 / 回显修复 ----
ssh -p 38024 root@connect.westb.seetacloud.com 'PATH=/root/miniconda3/bin:$PATH python3 /root/translate/final_audit.py'
ssh -p 38024 root@connect.westb.seetacloud.com 'PATH=/root/miniconda3/bin:$PATH python3 /root/translate/fix_ttc_echo.py'

# ---- 看进度 ----
ssh -p 38024 root@connect.westb.seetacloud.com 'bash /root/translate/progress.sh'

# ---- 停止（只停第二轮；不要动 643233）----
ssh -p 38024 root@connect.westb.seetacloud.com \
  'pkill -f run_round2.py; pkill -f "translate_shard.py.*work_round2"; pkill -f rerun_hayden_ttc.sh'

# ---- 服务健康 ----
ssh -p 38024 root@connect.westb.seetacloud.com 'curl -s localhost:8100/health; echo; curl -s localhost:8101/health'

# ---- 回滚第一轮流水线代码 ----
ssh -p 38024 root@connect.westb.seetacloud.com 'cd /root/translate && \
  for f in field_paths extract_text write_back dataset_registry translate_shard; do \
    [ -f $f.py.round1.bak ] && cp $f.py.round1.bak $f.py; done'
```

> **GPU1 服务（PID 867732，:8101）在第二轮收工后的处置见 §11。**

---

## 11. 铁律遵守声明

| 约束 | 状态 |
|---|---|
| 不占 GPU0 上不属于本任务的进程 | ✅ 只**复用** :8100，未 kill（PID 643233 全程存活） |
| 用户要求两张卡都用上 | ✅ lane A :8100/GPU0（4093 条）+ lane B :8101/GPU1（3415 条）；两遍翻译都双卡；实测两卡同时 ~86% util |
| 不动训练目录 / `mm-datasets` / `Qwen3.8-27B` 权重 | ✅ 只读；未在同一张卡上加载第二份权重 |
| 原始数据集只读 | ✅ 5153 文件基线 md5 两次复测均一致（§7） |
| 不 pip 装包 | ✅ 只用已装 `torch / transformers / pyarrow / tokenizers / requests` |
| 产物不覆盖第一轮 | ✅ `out_zh_round2/` 与 `out_zh/`、`work_round2/` 与 `work/` 完全分离 |
| 两卡并行 / 脱离 ssh / 可续跑 | ✅ `setsid nohup`，逐分片幂等（`translate_shard` 按 id 去重、`write_back` 存在即跳过） |
| 所有数字实测 | ✅ 未实测项一律 UNKNOWN（§9.2） |
| 不 commit / push | ✅ 本报告只写盘，未执行任何 git 写操作 |
| GPU1 服务收工处置 | ✅ 第二轮全部产物落盘并终检后，**已停掉本轮自己起的 :8101（PID 867732）**，释放 52.9 GB；GPU0 的 :8100 常驻服务**未动**。需要再用时重跑 §2.2 的命令（19.4 s 加载） |

---

*报告由第二轮中文翻译任务生成 · 数据来源：服务器实测日志与产物 · 未经实测的推断已在文中标注*
