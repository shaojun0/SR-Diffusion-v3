# ANALYSIS_mm_datasets_inventory — 12 个多模态数据集盘点总表（2026-09-17）

> 数据来源:服务器 `connect.westb.seetacloud.com:38024` 的只读镜像 `/root/autodl-tmp/mm-datasets/`(12 个 repo 目录)。
> 方法:4 个子智能体分组只读抽样(parquet 仅读 metadata/轻量列、zip 只读成员不解压、图片目录 `find` 计数),命令与输出摘要见各分组报告;**未修改/删除任何服务器数据**。
> 原始证据:`doc/2026-09-17/data/dataset_recon/`(4 份分组报告 + README 溯源说明)。

## 结论(一句话)

镜像 `/root/autodl-tmp/mm-datasets/` 共 **202 GB / 12 个 repo 目录 / 10 个 OK + 2 个 gated(HTTP 403)**;剔除 5.22% 纯文本样本后**含图样本约 80.0 万**(779,289 行中 738,601 行有图),但**任务形态高度异质**——检测/分类 6 个、多轮视觉指令 2 个、VQA 1 个、纯文本 QA 1 个、渲染包 1 个、机器人录制 1 个,其中只有 `aswin00000/ConstructionSiteCleanedDataSet` 是本项目 `construction_site` 训练数据的直接来源。

## 镜像状态

| 项 | 值 |
|---|---|
| 服务器路径 | `/root/autodl-tmp/mm-datasets/<org>__<name>` |
| 总体积 | **202 GB** |
| repo 目录数 | **12** |
| 状态 | **10/12 OK**,**2 个 gated** |
| gated 明细 | `jhboyo/ppe-dataset`、`pyimagesearch/construction-safety-object-detection-paligemma` — 在 hf-mirror 上 **HTTP 403**,服务器无 HF token |

## 主总表(12 行)

| 数据集 | 类型/任务 | 单轮还是多轮 | 单图还是多图 | 样本数 | 关键字段 | 备注 |
|---|---|---|---|---|---|---|
| `aswin00000/ConstructionSiteCleanedDataSet` | 图像描述(caption) + 违规目标检测(bbox) + 文本理由(多任务、非 QA) | N/A(无对话字段) | 单图 | **10013**(train 7009 / test 3004) | `image`, `image_caption`, `rule_1..4_violation` | **正是本项目 `construction_site` 训练数据来源**;10 个 parquet;4 条安全规则共 2078 个违规框;bbox = 归一化 xyxy |
| `chandrabhuma/multi_building_defect_vqa` | VQA(分类式,6 缺陷类) | 单轮 100% | 单图 | **2411**(1928/483) | `image`, `question`, `answer` | 名为 VQA 但 `question` 仅 1 种、`answer` 仅 6 类 ⇒ 等价 6 类图像分类;answer: cracks 789 / no defect 452 / spalling 427 / efflorescence 311 / general defects 264 / scaling 168 |
| `DBCMLAB/Constructionsafety_QApairs` | 纯文本单轮指令 QA(韩语,无图) | 单轮 100% | **无图** | **5114** | `instruction`, `input`(全空), `output` | 唯一无图数据集;`input` 5114/5114 全为空串;第 2365 行(`id=2365`)`instruction`/`output` 为 NULL |
| `Francesco/construction-safety-gsnvb` | 目标检测(COCO,RF100,6 类) | N/A | 单图 | **1206**(997/90/119) | COCO json + 图片 | tar.gz 与 parquet 重复(二者取一即可);7724 个框;`construction-safety` 类 0 实例(实际有效 5 类);bbox = COCO 像素 xywh |
| `jhboyo/ppe-dataset` | 目标检测(YOLO,3 类) | N/A | 单图 | 可用 **15487**(9998/2738/2751) | YOLO txt | ⚠️ **13 个 `.part` 未完成下载**(1 图 + 12 标签);实盘 45,498 框(README 报 60,991,以实盘为准);bbox = YOLO 归一化 cxcywh;HF 端亦 gated |
| `juungwon/LLava-ConstructionSafety_v6` | 多轮视觉指令对话(链式) | 行内单轮 100%,但按图重组 **3–6 轮** | 单图 | **30 行 / 9 张图** | `messages`, `images` | 30 行只对应 9 张不同图(8 张 ×3 行 + 1 张 ×6 行);29 个相邻行对中 20 个(69%)user 文本内嵌上一行 assistant 回答 ⇒ 链式;韩语 |
| `keremberke/construction-safety-object-detection` | 目标检测(COCO,17 类) | N/A | 单图 | **398**(307/57/34) | 图片仍在 zip 内,需解压 | 1093 个框;valid-mini 另有 3 张;类别极不均衡(`barricade`/`dumpster`/`mask`/`mini-van`/`truck` 各仅 1 实例);bbox = COCO 像素 xywh |
| `lmms-lab/LLaVA-NeXT-Data` | 多轮视觉指令对话(LLaVA 格式) | **单轮 38.45% / 多轮 61.55%**(均值 6.03,中位 4,最大 275) | **单图**(5.22% 为纯文本样本) | **779289** | `conversations`, `image` | 137G;`llava_next_raw_format/` 另含 processed.json(738590) + 11 个图片 tar;`data_source` 全库恒为 `llava-next-instruct`;12 个子集 |
| `physicl/indoor-safety-hazard-detection-and-work-zone-monitoring` | 渲染数据包(多 pass,无监督标签) | N/A(`metadata` 全 null) | **同一视角 11 路多 pass 多图** | parquet **400 行** / 磁盘 **4400 PNG** | `image` + 10 个 pass 列, `image_path`, `metadata` | 2048×1536;单相机(`Camera_004`)/单 focal;400/400 行 11 列共享同一 view index;声明的第 12 路 pass `depth_metric` 无列无文件 |
| `pyimagesearch/construction-safety-object-detection-paligemma` | 目标检测(应然,PaliGemma 标签) | BLOCKED | 单图(应然) | 应然 **398**(307/57/34),**NOT VERIFIED** | 仅 `README`,gated 403 | 镜像内只有 `README.md`(1337 B)、`data/` 为空 ⇒ 除 README 声明外一切 **UNKNOWN**,不编造 |
| `Voxel51/Construction-Site-Traversability` | 机器人多模态 MCAP 录制(**无标注**) | N/A | **每条 sample = 1 段多帧 episode(8,696–28,219 帧)** | **4 sample / 4 个 .fo.mcap** | `fiftyone samples.json`, `metadata.json` | traversability 标注不在此镜像(作者另发 `manojkarnekar/construction-traversability-dataset`);`classes`/`mask_targets` 全空;内含 62,995 colour 帧;mcap 需装包才能解码 |
| `ZhiyaYang/sewer-defect-crack-dataset` | 图像分类(3 类) | N/A | 单图 | **585**(409/88/88) | `image`, `label`(0/1/2) | `names: {0: blockage, 1: corrosion, 2: crack}`;实测分布 178/237/170 |

**与参考值核对结果:上表 12 行的样本数/类型/轮数/图数与分组报告逐项一致,无任何不一致项。** 下列三处补充口径与参考值不冲突,但需注意:

1. `chandrabhuma` 参考值记「类型 = VQA」正确;报告补充其 `question` 只有 1 种、`answer` 只有 6 类,**形式上是分类式 VQA,可直接当图像分类用**。故在「按类型」统计中**同时计入 VQA(1) 与分类(2)** 两个口径(见下),这是口径重叠而非矛盾。
2. `jhboyo` 参考值记「可用 15487」正确(严格图片-标签配对);报告另记磁盘图片 15,499 个、标签 15,488 个,README 声称 15,500 ⇒ **严格可用 15,487 对**。另:`DATASET_MIRROR.md` 的 FINAL STATUS 把 `jhboyo` 记为 **BLOCKED — HTTP 403(gated)**,但其落盘文件基本完整(31,004 个文件),两者描述的是不同层面(gated 状态 vs 已落盘数据),以本表为准。
3. `pyimagesearch` 参考值记「应然 398(307/57/34),NOT VERIFIED」正确;报告强调该 398 **来自 README front-matter 自述,未做任何数据校验**。

## 三个维度的统计小结

### ① 按「类型」分类计数

| 类型 | 数量 | 数据集 |
|---|---|---|
| 目标检测 | **4** | aswin00000(附 caption+reason)、Francesco、jhboyo、keremberke |
| 图像分类 | **2** | ZhiyaYang、chandrabhuma(分类式 VQA,见口径重叠说明) |
| VQA | **1** | chandrabhuma |
| 多轮视觉指令对话 | **2** | juungwon、lmms-lab/LLaVA-NeXT-Data |
| 纯文本 QA(无图) | **1** | DBCMLAB |
| 渲染数据包 | **1** | physicl |
| 其它 — 机器人多模态录制 | **1** | Voxel51(无标注) |
| BLOCKED(不可判定) | **1** | pyimagesearch |
| **合计** | **12** | 注:`chandrabhuma` 在「分类」与「VQA」两栏重复计入(分类式 VQA),故分类口径列合计 > 12 |

### ② 单轮 vs 多轮

| 类别 | 数量 | 数据集 |
|---|---|---|
| 单轮 | **5** | DBCMLAB(100%)、chandrabhuma(100%)、Francesco(N/A)、keremberke(N/A)、ZhiyaYang(N/A)、pyimagesearch(应然单轮) —— 其中 Francesco/keremberke/ZhiyaYang 属「无对话字段 ⇒ N/A」,不是「单轮」 |
| 多轮 | **2** | lmms-lab/LLaVA-NeXT-Data(多轮 61.55%)、juungwon(**按图重组** 3–6 轮) |
| 混合 | **1** | lmms-lab/LLaVA-NeXT-Data 单轮 38.45% + 多轮 61.55%(同属「多轮」栏) |
| N/A(无对话字段) | **6** | aswin00000、Francesco、jhboyo、keremberke、physicl、Voxel51、ZhiyaYang |

**「行内轮数」与「按图重组轮数」必须分开说(juungwon)**:该数据集每行 `messages` 长度恒为 2(1 user + 1 assistant)⇒ **按行判定单轮 100%**;但 30 行只对应 9 张不同图,同一图片的连续多行构成一条对话(上一行 assistant 回答被拼接进下一行 user 输入)⇒ **按图片重组为 3–6 轮**(8 张各 3 轮 + 1 张 6 轮)。若下游按「行」训练,它是单轮;按「图」重组才是多轮。

### ③ 单图 vs 多图 vs 无图

| 类别 | 数量 | 数据集 |
|---|---|---|
| 单图(1 图/样本) | **7** | aswin00000、chandrabhuma、Francesco、jhboyo、juungwon、keremberke、ZhiyaYang + lmms-lab(94.78% 单图) |
| 多图 | **0(常规意义)** | 无任何数据集是「1 样本 = N 张同质图」 |
| 无图 | **1** | DBCMLAB(纯文本 QA) |
| BLOCKED / 应然单图 | **1** | pyimagesearch |
| 特殊:多 pass 多图(非多图样本) | **1** | physicl(1 视角 × 11 pass = 4400 PNG;每行 11 个图片列,属**同一视角的多通道 G-buffer**,不是多视角、也不是多图样本) |
| 特殊:多帧 episode(非多图样本) | **1** | Voxel51(1 sample = 1 个 .fo.mcap,内含 8,696–28,219 帧单目时序 + 等量 depth/LiDAR) |

**「多 pass 多图」(physicl)与「多帧 episode」(Voxel51)都不是常规多图样本**:前者的 11 张图是同一视角的 11 个渲染通道(beauty/albedo/depth/normal/material_index/… ),必须先按 `view index` 分组才能还原为「1 个视角」;后者的「图」是容器内的时序帧,需先解码 mcap。二者都**不能**按「1 行 = 1 张图」直接喂给常规多图模型。

## 注意事项 / 风险

1. **gated 2 个**:`jhboyo/ppe-dataset` 与 `pyimagesearch/...-paligemma` 在 hf-mirror 上 HTTP 403,服务器无 HF token。`pyimagesearch` 镜像内**只有 README**,`data/` 为空 ⇒ 其全部字段级结论为 UNKNOWN/not verified。`jhboyo` 虽同被标记 gated,但落盘数据基本完整。
2. **jhboyo 13 个 `.part` 未完成下载**:`train/images/ds2_helmet_jacket_02011.jpg.part`(50,820 B)+ `val/labels/` 下 12 个 `*.txt.part`。严格图片-标签配对数 9998/2738/2751 = **15,487**(README 声称 15,500,实盘缺 1 图 + 12 标签)。
3. **keremberke 未解压**:图片仍在 `train.zip`/`valid.zip`/`test.zip`/`valid-mini.zip` 内(平铺 + `_annotations.coco.json`),使用前需解压(本次分析刻意未解压、未落盘)。
4. **三种 bbox 约定不同,不可混用**:
   - `aswin00000`:归一化 `xyxy`(float,[0,1])—— 判别证据:2078 框中 1619 个满足 `x1+x2>1`(若 xywh 会越界),0 个 `x2<x1`;
   - `keremberke` / `Francesco`:COCO **绝对像素** `xywh`(Francesco 全部 resize 到 640×640);
   - `jhboyo`:YOLO **归一化** `cx cy w h`。
5. **Voxel51 mcap 需装包才能解码**:`import mcap` → `ModuleNotFoundError`(遵守「不装新包」约束,本次仅读 `samples.json`/`metadata.json` 的计数,未解码任何帧)。且该镜像**不含 traversability 标注/掩码**(`classes`/`mask_targets`/`annotation_runs` 全空),标注在另一个 repo `manojkarnekar/construction-traversability-dataset`。
6. **physicl 声明的第 12 路 pass(`depth_metric`)无文件**:`available_passes` 声明 12 个 pass(含 `depth_metric`),但只有 11 个图片列,磁盘上也无 `depth_metric` 的 PNG ⇒ 该 pass 未镜像/未导出。另 `metadata` 列 400/400 全为 `null`,**不能**声称其内含文本或问答。
7. **其它数据质量项**:DBCMLAB 第 2365 行 `instruction`/`output` 为 NULL;Francesco 的 `construction-safety` 类 0 实例(类名预留);keremberke 类别极不均衡(5 类各仅 1 实例);LLaVA-NeXT 的 parquet 带图数(738,601)与 raw json 记录数(738,590)差 **11 条**(<0.002%,未强行归因);LLaVA-NeXT 的轮数分布**高度分片相关**(分片按子集成块排列,单看少数分片会严重偏离全库,如 `train-00000` 单轮仅 0.83%,全库 38.45%)。
8. **`du` 口径**:`DATASET_MIRROR.md` 记载 USB 侧 exFAT 的 `du` 会严重虚高(如 DBCMLAB `du`=13M 而真实 payload 672 KB),本盘点的体积/规模一律以**行数/文件数/真实文件大小**为准。

## 溯源

### 原始证据(服务器,只读)

| 项 | 值 |
|---|---|
| 服务器 | `ssh -p 38024 root@connect.westb.seetacloud.com`(autodl-container-jnb93wme4w-bf7ae4cd) |
| 数据根 | `/root/autodl-tmp/mm-datasets/` |
| Python | `export PATH=/root/miniconda3/bin:$PATH`(pandas/pyarrow,无新装包) |
| 状态文件 | `/root/mm-datasets-status/*.status`;脚本 `/root/mm-datasets-dl.sh`;日志 `/root/mm-datasets-dl.log` |
| 访问方式 | **只读抽样**;未修改/删除任何数据,训练目录与 `/root/autodl-tmp/construction_site` 未被读写 |

### 分组报告(已入库,同目录)

| 报告 | 覆盖数据集 |
|---|---|
| `data/dataset_recon/groupA_qa.md` | DBCMLAB、chandrabhuma、juungwon、ZhiyaYang |
| `data/dataset_recon/groupB_detect_cls.md` | aswin00000、keremberke、jhboyo、Francesco |
| `data/dataset_recon/groupC_llava_next.md` | lmms-lab/LLaVA-NeXT-Data |
| `data/dataset_recon/groupD_render_traversability.md` | physicl、Voxel51、pyimagesearch |
| `data/dataset_recon/README.md` | 上述 4 份报告的来源与对应关系说明 |

### 参考文档

- 镜像复制运行记录:`/home/linaro/dsh/srdiff_overnight/runtime/DATASET_MIRROR.md`(镜像脚本、逐 repo 状态、FINAL STATUS 2026-09-17 11:45)
- 机器可读数据:`doc/2026-09-17/data/dataset_recon/datasets_inventory.json`(12 个对象,数值与本表一致)

> 本次任务**只新增文档**,未改任何代码。
