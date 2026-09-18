# 数据集筛选与补量归档（2026-09-17 → 09-18）

> 背景：从 USB 镜像到服务器的 12 个 HuggingFace 多模态数据集（`/root/autodl-tmp/mm-datasets/`，202 GB），
> 按用户四条标准筛选，再补量到 ~140 GB。本目录是该过程的证据与脚本。

## 1. 筛选判定（标准：剔除「多轮对话 / 无标注 / 非图像 / 非中英语言」）

判定脚本：`scripts/filter_check.py`（原 `srdiff_overnight/runtime/filter_check.py`）。

**保留 6 个 = 8.109 GB**

| repo | 体量 | 类型 | 保留理由 |
|---|---:|---|---|
| `aswin00000/ConstructionSiteCleanedDataSet` | 4.498 G | caption + 违规检测框 + 理由 | 有标注、单图、无对话、英文 |
| `jhboyo/ppe-dataset` | 1.863 G | YOLO 目标检测 | 有标注、单图、无文本 |
| `chandrabhuma/multi_building_defect_vqa` | 1.344 G | VQA（分类式 6 类） | 有标注、单图、单轮、英文 |
| `ZhiyaYang/sewer-defect-crack-dataset` | 0.226 G | 图像分类（3 类） | 有标注、单图、无文本 |
| `Francesco/construction-safety-gsnvb` | 0.151 G | COCO 目标检测（6 类） | 有标注、单图、无文本 |
| `keremberke/construction-safety-object-detection` | 0.028 G | COCO 目标检测（17 类） | 有标注、单图、无文本 |

**剔除 6 个 = 208.455 GB**

| repo | 体量 | 剔除理由 |
|---|---:|---|
| `lmms-lab/LLaVA-NeXT-Data` | 146.251 G | 多轮对话（多轮占 61.55%） |
| `physicl/indoor-safety-hazard-detection-…` | 44.047 G | 无标注（渲染包，`metadata` 全 null） |
| `Voxel51/Construction-Site-Traversability` | 18.142 G | 无标注（MCAP 录制，标注在别处） |
| `juungwon/LLava-ConstructionSafety_v6` | 0.014 G | 多轮对话 |
| `DBCMLAB/Constructionsafety_QApairs` | 0.001 G | 非图像（纯文本、韩语） |
| `pyimagesearch/…-paligemma` | 0.000 G | gated 无数据（镜像内仅 README） |

**保留语料 = 8.109 GB；距 140 GB 需补 ≈131.9 GB。**
（说明：仅做「从语料中剔除」判定，**未删除**服务器/USB 上任何原始文件。）

## 2. 补量：新增 13 个同域数据集 = 130.742 GB

选型与候选淘汰见 `supplement_plan.md`；精确清单（repo / 大小 / 任务类型 / 语言 / gated）见 `manifest_selected.json`。
硬标准：含图像 + 有标注 + 非多轮 + 中文/英文或无文本 + hf-mirror 可下（非 gated）。
最终：**保留 8.109 + 补量 130.742 ≈ 138.85 GB**（≈140 G 目标）。

## 3. 下载与修复

- **首轮失败根因**（见 `SUPPLEMENT_REPAIR.md`）：
  1. `mm-datasets-add.sh:66-74` 的重试分支**漏了 `curl -C -`** ⇒ 失败后把 `.part` 从 0 重传（73 次 RESUME-FAIL / 73 次 FAIL，7.5 h 零增长）；触发原因是 Xet CAS bridge 中途断连（`CURLE_PARTIAL_FILE` rc=18，124 次）。
  2. 4 个 shard 的 `ONLY=` 列表与 `IDX%4` 不匹配 ⇒ **8 个 repo 从未被任何 shard 认领**（0 条日志）。
- **修复**：新下载器 `scripts/mm_add2.py` + `scripts/mm-datasets-add2.sh`——每次都 `curl -L --fail -C -`、失败保留 `.part` 续传、尺寸精确的 `.part` 直接改名、停滞检测替代次数上限、429 全局退避、mirror 原生 Link 分页逐文件校验；幂等可续跑。
- **最终校验**（`repair_verify_20260918.json`）：13/13 repo，**15,717 文件**，tree bytes = local bytes = **130,741,857,441 B**，wrong/extra/missing 全 0，`.part` 0。
- 耗时 92m44s，均速 11.5 MB/s（峰值 17.7 MB/s）。

## 4. 本目录文件索引

| 文件 | 内容 |
|---|---|
| `README.md` | 本文件（筛选判定 + 总量 + 结论） |
| `supplement_plan.md` | 补量候选表、被排除候选与原因、凑量算术 |
| `manifest_selected.json` | 最终选中的 13 个 repo（repo/大小/类型/语言/gated） |
| `SUPPLEMENT_DOWNLOAD.md` | 首轮下载报告（含 ADDENDUM 最终状态） |
| `SUPPLEMENT_REPAIR.md` | rc=18 根因诊断 + 新下载器做法 + 校验结果 |
| `repair_verify_20260918.json` | 逐 repo 逐文件校验（tree/local bytes、missing/extra） |
| `scripts/filter_check.py` | 四条标准的筛选判定脚本（保留/剔除表） |
| `scripts/` 其余 | 下载/校验/选型脚本（首轮 `mm-datasets-add.sh`、`mm_verify.py`、`verify.py`、`sweep.py`；修复轮 `mm_add2.py`、`mm-datasets-add2.sh`） |

## 5. 关联文档

- 数据集盘点（12 个原始镜像）：`doc/2026-09-17/ANALYSIS_mm_datasets_inventory.md` 与 `doc/2026-09-17/data/dataset_recon/`
- 中文翻译（对保留语料的 `_zh` 化）：`doc/2026-09-18/REPORT_translation_zh_progress.md`
