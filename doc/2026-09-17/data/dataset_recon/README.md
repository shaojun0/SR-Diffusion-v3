# data/dataset_recon — 原始证据溯源说明（2026-09-17）

本目录存放 **2026-09-17 多模态数据集盘点**所用的原始证据(4 份分组证据报告)与机器可读汇总数据。
汇总文档见上级目录:`doc/2026-09-17/ANALYSIS_mm_datasets_inventory.md`。

## 证据来源

| 项 | 值 |
|---|---|
| 服务器 | `ssh -p 38024 root@connect.westb.seetacloud.com`(autodl-container-jnb93wme4w-bf7ae4cd,上海/西部节点) |
| 数据根 | `/root/autodl-tmp/mm-datasets/`(12 个 `<org>__<name>` repo 目录,共 202 GB) |
| 访问方式 | **只读抽样**。未修改/删除任何远端数据;训练目录与 `/root/autodl-tmp/construction_site` 未被读写 |
| Python 环境 | `export PATH=/root/miniconda3/bin:$PATH`(pyarrow/pandas;**未新装任何包**) |
| 抽样手段 | parquet 仅读 `ParquetFile(...).metadata.num_rows` 与少量列/row group(不读 `image.bytes` 全量);zip 仅用 `zipfile` 读成员与 `_annotations.coco.json`(不解压、不落盘);图片目录用 `ls`/`find` 计数;tar 仅 `tar -tzf` 列表;mcap **未解码** |
| 执行者 | 4 个子智能体分组并行只读抽样 |
| 时间 | 2026-09-17(镜像脚本于 2026-09-17 06:35:52 完成) |

## 4 份分组报告与对应数据集

| 报告文件 | 报告标题 | 覆盖数据集 | 说明 |
|---|---|---|---|
| `groupA_qa.md` | Group A — 数据集分析报告 | `DBCMLAB/Constructionsafety_QApairs`、`chandrabhuma/multi_building_defect_vqa`、`juungwon/LLava-ConstructionSafety_v6`、`ZhiyaYang/sewer-defect-crack-dataset` | 4 个数据集全部 ≤ 5114 行 ⇒ **对全部行**统计轮数/字段分布(非抽样外推);图数/链式对话/标签分布逐项核验 |
| `groupB_detect_cls.md` | Group B — 检测 / 分类类数据集分析 | `aswin00000/ConstructionSiteCleanedDataSet`、`keremberke/construction-safety-object-detection`、`jhboyo/ppe-dataset`、`Francesco/construction-safety-gsnvb` | 4 个检测类;含 bbox 编码判别(aswin 遍历全部 10013 行只读 rule 列)、zip 内 COCO json 只读解析、YOLO 标签计数与 `.part` 定位、tar 成员清单 |
| `groupC_llava_next.md` | Group C — `lmms-lab/LLaVA-NeXT-Data` 数据集分析 | `lmms-lab/LLaVA-NeXT-Data` | 单数据集单报告。轮数/子集/`<image>` 标记均为 **250/250 分片全量精确统计**;另有分层抽样(25 片 77,929 样本)交叉验证;raw format 流式解析 processed.json(738,590 条);未解压任何 tar.gz |
| `groupD_render_traversability.md` | Group D — Render Data-Pack & Traversability | `physicl/indoor-safety-hazard-detection-and-work-zone-monitoring`、`Voxel51/Construction-Site-Traversability`、`pyimagesearch/construction-safety-object-detection-paligemma` | 渲染包(同视角校验 + 4400 URL 逐条落盘校验)、机器人 MCAP 录制(仅读 counts 未解码帧)、gated BLOCKED 项(仅 README 可得) |

报告的原始副本位于本地 `srdiff_overnight/datasets/`(**未改动**,本目录为 `cp -p` 复制件,md5 逐字节一致):

| 文件 | md5 |
|---|---|
| `groupA_qa.md` | `b177fafbaef95111a2e0059b54c6c0f8` |
| `groupB_detect_cls.md` | `cd62d2aaeeb1d9ec324af7848264331d` |
| `groupC_llava_next.md` | `5c1f6cb0f6a8046b75fb711073ae50d5` |
| `groupD_render_traversability.md` | `22240fbb62c52bcbc17d4fc309165db5` |

## 本目录内容

| 文件 | 说明 |
|---|---|
| `groupA_qa.md` … `groupD_render_traversability.md` | 4 份分组证据报告原始副本(保留原文件名) |
| `datasets_inventory.json` | 机器可读总表:**12 个对象的 JSON 数组**,字段 `repo` / `local_dir` / `size_human` / `task_type` / `turns` / `turns_detail` / `images_per_sample` / `images_detail` / `num_samples` / `splits` / `key_fields` / `notes` / `evidence_md`。数值与 `ANALYSIS_mm_datasets_inventory.md` 主总表逐项一致;`evidence_md` 指回本目录的报告 |
| `README.md` | 本文件(溯源说明) |

## 交叉引用与已知口径差异

- 镜像复制运行记录(逐 repo 状态、FINAL STATUS 2026-09-17 11:45):`/home/linaro/dsh/srdiff_overnight/runtime/DATASET_MIRROR.md`
- **`DATASET_MIRROR.md` 的 FINAL STATUS 把 `jhboyo/ppe-dataset` 记为 `BLOCKED — HTTP 403 (gated)`,而 Group B 报告实测其落盘 31,004 个文件、可用 15,487 对样本**。两者不矛盾:前者描述 **HF 端 gated 状态**,后者描述**已落盘数据的实际可用量**。总表以落盘实测为准。
- `du -sh` 在 USB(exFAT)侧严重虚高,本盘点一律以**行数/文件数/真实文件大小**为准,不用 `du` 值。
- `pyimagesearch/...-paligemma` 的 398(307/57/34)来自 **README front-matter 自述**,NOT VERIFIED;该 repo 数据未镜像(gated/HTTP 403),**无任何字段级证据**。
