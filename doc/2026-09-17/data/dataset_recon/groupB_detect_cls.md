# Group B — 检测 / 分类类数据集分析

- 服务器: `ssh -p 38024 root@connect.westb.seetacloud.com`
- 数据根: `/root/autodl-tmp/mm-datasets/`
- Python: `export PATH=/root/miniconda3/bin:$PATH`
- 分析方式: 只读。parquet 仅读 `ParquetFile.metadata.num_rows` 与少量列/batch；zip 仅用 `zipfile` 读取成员与 `_annotations.coco.json`（不解压、不落盘）；图片文件夹用 `ls`/`find` 计数。未修改/删除任何数据。
- 时间: 分析于镜像快照 `2025-09-17`

## 总表

| # | 数据集 | 任务类型 | 单轮/多轮 | 单图/多图 | 样本数 (train/val/test) | 类别数 | 格式 |
|---|--------|----------|-----------|-----------|--------------------------|--------|------|
| 1 | aswin00000__ConstructionSiteCleanedDataSet | 图像描述 + 违规目标检测(带文本理由) | **N/A（无 QA/对话字段）**，仅 1 条 caption + 违规 reason | 单图 (1 图/样本) | **10013** (train 7009 / test 3004；无 val) | 4 条安全规则(非类别)，共 2078 个违规框 | 10 个 parquet |
| 2 | keremberke__construction-safety-object-detection | 目标检测 (COCO) | N/A（无任何文本字段） | 单图 (1 图/样本) | **398** (train 307 / valid 57 / test 34；另有 valid-mini 3) | 17 类 | 4 个 zip（未解压） |
| 3 | jhboyo__ppe-dataset | 目标检测 (YOLO) | N/A（无任何文本字段） | 单图 (1 图/样本) | **15487 可用**（train 9998 / val 2738 / test 2751）；含未完成下载 13 个 `.part` | 3 类 | 图片文件夹 + txt 标签 |
| 4 | Francesco__construction-safety-gsnvb | 目标检测 (COCO) | N/A（无任何文本字段） | 单图 (1 图/样本) | **1206** (train 997 / validation 90 / test 119) | 6 类，共 7724 个框 | 3 个 parquet + `dataset.tar.gz` + `dataset_info.json` |

**结论摘要**：4 个数据集全部是「单图 + 非多轮」的 CV 数据集，3 个是纯目标检测（无任何文本字段），1 个（aswin）是「caption + 违规框 + 违规理由」的多任务数据集，但也没有问答/对话结构。

---

## 1. aswin00000__ConstructionSiteCleanedDataSet

### 1.1 任务类型
**图像描述（captioning）+ 违规目标检测（grounded rule-violation detection）+ 文本理由生成**，多任务混合。

依据：
- 每行有 `image`（1 张 JPEG）与 `image_caption`（1 条英文场景描述）→ 是图像描述任务。
- 另有 4 个结构体字段 `rule_1_violation` … `rule_4_violation`，每个含 `bounding_box`（违规区域的框）+ `reason`（自然语言违规理由）→ 是带文本解释的目标检测/grounding，不是普通分类。
- 无 `question`/`answer`/`conversations` 等字段，故不是 VQA、也不是多轮指令。

4 条规则的语义（由 `reason` 文本归纳）：
| 字段 | 规则含义 | 有 reason 的行数 |
|---|---|---|
| `rule_1_violation` | PPE 未佩戴（安全帽 / 高可视背心） | 1000 |
| `rule_2_violation` | 高处作业未使用安全带 (safety harness) | 84 |
| `rule_3_violation` | 临边/洞口/基坑缺少防护 (edge / opening protection) | 172 |
| `rule_4_violation` | 人员离运行中的挖机等机械过近 | 70 |

### 1.2 数据格式
parquet。目录层级：
```
aswin00000__ConstructionSiteCleanedDataSet/
├── README.md
├── .gitattributes
└── data/
    ├── train-00000-of-00007.parquet … train-00006-of-00007.parquet  (7 个)
    └── test-00000-of-00003.parquet  … test-00002-of-00003.parquet   (3 个)
```
共 10 个 parquet，`download_size = 4497699882` B ≈ 4.19 GiB（与「4.2G」一致）。

### 1.3 样本数（精确）
| split | 文件 | 行数 |
|---|---|---|
| train | train-00000 … 00006 | 1002, 1002, 1001, 1001, 1001, 1001, 1001 = **7009** |
| test | test-00000 … 00002 | 1002, 1001, 1001 = **3004** |
| **合计** | 10 文件 | **10013** |
无 validation split（与 README 中 `splits` 只列 train/test 一致）。

### 1.4 单图 / 多图
**单图，1 张/样本。** `image` 字段类型是 `struct<bytes: binary, path: string>`（单个 struct，不是 `list<struct>`），每行恰好 1 张图。所有 split 合计 image 数 = 行数 = 10013。
图片内嵌于 parquet（`bytes`，如 `0000001.jpg` 长度 303238 B），`path` 只是原始文件名。

### 1.5 单轮 / 多轮
**N/A**。该数据集**没有** question/answer/conversations/轮次类字段，只有：
- `image_caption`：每行 1 条描述（非 QA）；
- `rule_N_violation.reason`：每行每规则最多 1 条违规理由（非 QA）。

因此不存在「轮数分布」可统计；不存在任何问答文本字段。属于「有文本但非对话」的数据集。

### 1.6 标签体系
不是分类类别体系，而是 **4 条安全规则（rule_1…rule_4）+ 违规区域框 + 文本理由**。可视为 4 个二值/多标签检测头。
- 类别数：4（规则数）；如按「是否违规」看为二分类，按规则看为 4 个独立标签。
- bbox 格式：**归一化 `[x_min, y_min, x_max, y_max]`，坐标值域 [0,1]**，每个 bbox 4 个 float。
  判别证据：2078 个框中 **1619 个满足 `x1+x2>1` 或 `y1+y2>1`**（若为 xywh 会越界，不合理），而 **0 个满足 `x2<x1` 或 `y2<y1`**（若为 xyxy 是合理顺序）→ 只能是 **xyxy**。
- 标注密度：全部 10013 行中 **1277 行** 至少含 1 条违规 reason；框总数 **2078**（每 (行,规则) 的框数分布：1 个框 146 次、2 个 20 次、3 个 6 次、4 个 4 次、5 个 1 次）。
- 每行 4 个规则字段都可能为 `null`（合规样本）。

### 1.7 关键字段名
`image` (struct: `bytes`, `path`)、`image_caption` (string)、`rule_1_violation` / `rule_2_violation` / `rule_3_violation` / `rule_4_violation` (struct: `bounding_box: list<list<double>>`, `reason: string`)。

真实样例（第 1 个 split 的 row 0，截断）：
```
image = {path: "0000001.jpg", bytes: 303238 B}
image_caption = "The rear view of a mobile crane."
rule_1_violation = null ; rule_2_violation = null ; rule_3_violation = null ; rule_4_violation = null
```
含违规的样例（截断 ~200 字符）：
```
image_caption = "An excavator is in the center of the image. There is a person walking on the left, and a mobile crane on the right. The"
rule_1_violation = {"bounding_box": [[0.14, 0.59, 0.19, 0.7]], "reason": "Person on the left not using PPE."}
```

### 1.8 证据
```bash
# 行数 + schema
python - <<'EOF'
import pyarrow.parquet as pq, glob, os
files=sorted(glob.glob('/root/autodl-tmp/mm-datasets/aswin00000__ConstructionSiteCleanedDataSet/data/*.parquet'))
for f in files: print(os.path.basename(f), pq.ParquetFile(f).metadata.num_rows)
print(pq.ParquetFile(files[0]).schema_arrow)
EOF
```
输出摘要：
```
test-00000-of-00003.parquet rows= 1002
test-00001-of-00003.parquet rows= 1001
test-00002-of-00003.parquet rows= 1001
train-00000..00006  rows= 1002,1002,1001,1001,1001,1001,1001
TOTALS {'test': 3004, 'train': 7009} GRAND 10013
image: struct<bytes: binary, path: string>
image_caption: string
rule_1_violation: struct<bounding_box: list<element: list<element: double>>, reason: string>
...(rule_2/3/4 同构)
```
```bash
# bbox 编码判别 + 覆盖率（遍历全部 10013 行，只读 4 个 rule 列）
# 输出: total bboxes 2078 ; x1+x2>1 or y1+y2>1: 1619 ; x2<x1 or y2<y1: 0
#       rows total 10013 ; rows with >=1 rule violation reason: 1277
# caption 非空: 10013 / 10013
# per-rule 非空 reason: rule_1 1000, rule_2 84, rule_3 172, rule_4 70
```

---

## 2. keremberke__construction-safety-object-detection

### 2.1 任务类型
**目标检测（object detection）**，Roboflow COCO 格式导出。
依据：`construction-safety-object-detection.py` 的 `_info()` 定义 `objects` 为 `Sequence{id, area, bbox(length=4), category: ClassLabel}`；数据为 `_annotations.coco.json` + 图片；无任何文本问答字段。

### 2.2 数据格式
- **zip（原始 Roboflow 导出，未解压）+ 元数据文件**。
- 目录层级：
```
keremberke__construction-safety-object-detection/
├── data/
│   ├── train.zip        (22,259,647 B)
│   ├── valid.zip        ( 3,338,795 B)
│   ├── test.zip         ( 1,796,467 B)
│   └── valid-mini.zip   (   166,069 B)
├── split_name_to_num_samples.json
├── construction-safety-object-detection.py   (HF dataset loader)
├── README.md / README.roboflow.txt / README.dataset.txt
├── thumbnail.jpg
└── .gitattributes
```
- 每个 zip 内部为平铺结构：`<image files>` + `_annotations.coco.json`（COCO 格式）。图片**未解压到磁盘**，仅存在于 zip 内。

### 2.3 样本数（精确，实测 zip 内）
| split | zip | 图片数 | 标注数 | json 中 `images` | `split_name_to_num_samples.json` |
|---|---|---|---|---|---|
| train | train.zip | 307 | 799 | 307 | 307 |
| valid | valid.zip | 57 | 221 | 57 | 57 |
| test | test.zip | 34 | 73 | 34 | 34 |
| **合计(full)** | — | **398** | **1093** | 398 | 398 |
| valid-mini | valid-mini.zip | 3 | 6 | 3 | —（mini 配置的三个 split 都用它） |
与 `README.roboflow.txt`「It includes 398 images」一致。zip 成员数 = 图片数 + 1（`_annotations.coco.json`），train.zip 308 = 307+1，valid.zip 58，test.zip 35，valid-mini.zip 4。

### 2.4 单图 / 多图
**单图，1 张/样本。** 证据：COCO json 中 `images` 条目数 == 图片文件数（307/57/34），每个 `annotations[i].image_id` 指向单张图；loader 逐个文件名 `yield` 一条样本，含单个 `image` 字段。总图片数 398 == 总样本数 398。

### 2.5 单轮 / 多轮
**N/A。** 该数据集**完全没有任何文本/问答字段**（特征只有 `image_id, image, width, height, objects`），不存在轮数或问答统计。

### 2.6 标签体系
**17 类**（loader 中 `_CATEGORIES`，与 json 中 17 个 categories 完全一致，按 cat id 0–16 顺序）：
`barricade, dumpster, excavators, gloves, hardhat, mask, no-hardhat, no-mask, no-safety vest, person, safety net, safety shoes, safety vest, dump truck, mini-van, truck, wheel loader`

- bbox 格式：**COCO `[x, y, width, height]`，绝对像素**（非归一化）。
  样例 `[238, 22, 44, 34]`（图片 359×270）、`[285, 125, 379, 587]`（图片 888×864）→ 符合像素绝对坐标 xywh。
- 每类实例数（train / valid / test）：
  - train: barricade 1, dumpster 1, excavators 107, gloves 11, hardhat 289, mask 1, no-hardhat 92, no-mask 36, no-safety vest 21, person 42, safety net 3, safety shoes 9, safety vest 45, dump truck 77, mini-van 1, truck 1, wheel loader 62（共 799）
  - valid: excavators 3, gloves 5, hardhat 115, no-hardhat 9, no-mask 14, no-safety vest 14, person 23, safety shoes 13, safety vest 6, dump truck 8, wheel loader 11（共 221）
  - test: gloves 2, hardhat 32, no-hardhat 7, no-mask 2, no-safety vest 3, person 4, safety vest 2, dump truck 12, wheel loader 9（共 73）
  - 类别极不均衡：`hardhat` 最多，`barricade/dumpster/mask/mini-van/truck` 各仅 1 个实例（仅在 train）。

### 2.7 关键字段名
COCO json：`images[{id, file_name, width, height, license, date_captured}]`、`annotations[{id, image_id, category_id, bbox, area, segmentation, iscrowd}]`、`categories[{id, name, supercategory}]`。
HF loader 输出特征：`image_id, image, width, height, objects[{id, area, bbox, category}]`。

真实样例（截断 ~200 字符）：
```
annotations[0] = {"id": 0, "image_id": 0, "category_id": 4, "bbox": [238, 22, 44, 34], "area": 1496, "segmentation": [], "iscrowd": 0}
images[0]      = {"id": 0, "license": 1, "file_name": "005453_jpg.rf.b288a868fed6d94a795c5d93e3e627ea.jpg", "height": 270, "width": 359, "date_captured": "2022-12-29T11:22:28+00:00"}
```

### 2.8 证据
```bash
# 只读 zip 成员与 COCO json，不解压
python - <<'EOF'
import zipfile, json
D='/root/autodl-tmp/mm-datasets/keremberke__construction-safety-object-detection/data/'
for z in ['train.zip','valid.zip','test.zip','valid-mini.zip']:
    zf=zipfile.ZipFile(D+z); names=zf.namelist()
    imgs=[n for n in names if n.lower().endswith(('.jpg','.jpeg','.png'))]
    a=json.loads(zf.read('_annotations.coco.json'))
    print(z,'entries=',len(names),'images=',len(imgs),'jsonimages=',len(a['images']),
          'annots=',len(a['annotations']),'cats=',len(a['categories']))
    print('  cats:',[c['name'] for c in a['categories']])
    print('  sample annot:',json.dumps(a['annotations'][0])[:220])
EOF
```
输出摘要：
```
train.zip entries= 308 images= 307 jsonimages= 307 annots= 799 cats= 17
valid.zip entries= 58  images= 57  jsonimages= 57  annots= 221 cats= 17
test.zip  entries= 35  images= 34  jsonimages= 34  annots= 73  cats= 17
valid-mini.zip entries= 4 images= 3 jsonimages= 3 annots= 6 cats= 17
cat names: ['barricade','dumpster','excavators','gloves','hardhat','mask','no-hardhat','no-mask','no-safety vest','person','safety net','safety shoes','safety vest','dump truck','mini-van','truck','wheel loader']
sample annot: {"id": 0, "image_id": 0, "category_id": 4, "bbox": [238, 22, 44, 34], "area": 1496, ...}
```
另有 `split_name_to_num_samples.json` = `{"train": 307, "valid": 57, "test": 34}`（与实测一致）。

---

## 3. jhboyo__ppe-dataset

### 3.1 任务类型
**目标检测（object detection），YOLO 格式**（PPE 佩戴检测）。
依据：目录为 `images/` + `labels/` 成对结构；标签文件为 YOLO txt（`class_id x_center y_center width height`）；README 明确 `task_categories: object-detection`、`YOLOv8 최적화 포맷`。无任何文本问答字段。

### 3.2 数据格式
**图片文件夹 + YOLO txt 标签**。目录层级：
```
jhboyo__ppe-dataset/
├── README.md
├── .gitattributes
├── train/
│   ├── images/   (9999 个条目: 6781 .jpg + 3217 .png + 1 .part)
│   ├── labels/   (9999 个 .txt)
│   └── labels.cache        (3,551,713 B, YOLO 缓存)
├── val/
│   ├── images/   (2750: 1845 .jpg + 905 .png)
│   ├── labels/   (2738 .txt + 12 .part)
│   └── labels.cache        (967,708 B)
└── test/
    ├── images/   (2751: 1873 .jpg + 878 .png)
    └── labels/   (2751 .txt)
```
无 `data.yaml` / `dataset.yaml`（类名只在 README 中给出）。
注意：该 repo 在 hf-mirror 上标 gated，但落盘文件基本完整（见 3.3 的 `.part` 说明）。

### 3.3 样本数（以实际落盘为准）
| split | 图片(jpg+png) | 标签(.txt) | 可用成对样本 | 空标签文件 |
|---|---|---|---|---|
| train | **9998** | 9999 | **9998** | 27 |
| val | 2750 | **2738** | **2738** | 14 |
| test | **2751** | **2751** | **2751** | 8 |
| **合计** | 15499 | 15488 | **15487** | 49 |
- 全部文件数 `find ... -type f | wc -l` = **31004**（含 2 个 `labels.cache`、README、.gitattributes）。
- 未完成下载残留（`.part`，**不可直接当样本用**）：
  - `train/images/ds2_helmet_jacket_02011.jpg.part`（50,820 B）→ 该图不完整（对应 `train/labels/ds2_helmet_jacket_02011.txt` 存在，172 B，故 tight 配对时该 pair 应剔除）。
  - `val/labels/` 下 12 个 `*.txt.part`（ds1_hard_hat_workers2858/2866/2917/2972/2923/2877/2965/515/4986/4907/4994、ds2_helmet_jacket_01648）→ 这 12 个 val 图片可用但标签不完整。
- 与 README 声称的 9,999 / 2,750 / 2,751（合计 15,500）相比，实盘缺 1 张 train 图与 12 个 val 标签 → **严格可用 15487 对**。README 中的对象数统计（39,455 / 10,674 / 10,862，合计 60,991）与实盘不符，实盘对象总数为 **45,498**（见 3.6），README 数字偏大，应以实盘为准。

### 3.4 单图 / 多图
**单图，1 张/样本。** 证据：`images/` 与 `labels/` 同名配对（1:1），每个样本 = 1 个图片文件（`.jpg`/`.png`）+ 1 个同名 `.txt`；文件名前缀区分来源（`ds1_` = Hard Hat Detection，`ds2_` = Safety Helmet and Reflective Jacket）。总图片文件数 15,499（含 1 个 `.part`）即样本量级。

### 3.5 单轮 / 多轮
**N/A。** 数据集**完全没有任何文本/问答字段**：图片 + YOLO 数值标签，无 caption / question / answer。不存在轮数统计。

### 3.6 标签体系
**3 类**（README 明确，标签中出现的 class id 只有 0/1/2，实测吻合）：
| class id | 名称 | 含义 |
|---|---|---|
| 0 | helmet | 已戴安全帽 |
| 1 | head | 未戴安全帽（仅头部） |
| 2 | vest | 已穿反光安全背心 |

- bbox 格式：**YOLO 归一化 `class_id x_center y_center width height`**（值域 [0,1]）。
- 实盘实例数（按 split，仅统计完整 `.txt`）：
| split | helmet(0) | head(1) | vest(2) | 对象合计 |
|---|---|---|---|---|
| train | 18,964 | 3,404 | 7,116 | **29,484** |
| val | 5,020 | 1,034 | 1,840 | **7,894** |
| test | 5,196 | 893 | 2,031 | **8,120** |
| **合计** | **29,180** | **5,331** | **10,987** | **45,498** |
- 49 个空标签文件（27 train / 14 val / 8 test）= 纯背景图（无目标），用于 YOLO 的 background/negative 样本。

### 3.7 关键字段名
无结构化字段（非 parquet/json）。样本 = `{split}/images/<name>.{jpg,png}` 与 `{split}/labels/<name>.txt` 同名配对；txt 每行 5 个 token：`class_id x_center y_center width height`。

真实样例（`train/labels/` 中某个文件，5 行标注之一，截断）：
```
0 0.865385 0.508434 0.230769 0.202410
0 0.438702 0.326506 0.257212 0.248193
0 0.097356 0.344578 0.194712 0.226506
```
（对应图片如 `train/images/ds1_hard_hat_workers0.png`）

### 3.8 证据
```bash
D=/root/autodl-tmp/mm-datasets/jhboyo__ppe-dataset
for s in train val test; do
  echo "== $s"; echo -n "images: "; ls $D/$s/images | wc -l
  echo -n "labels: "; ls $D/$s/labels | wc -l
  ls $D/$s/images | sed 's/.*\.//' | sort | uniq -c
  cat $D/$s/labels/*.txt | awk '{print $1}' | sort -n | uniq -c
  echo -n "empty label files: "; find $D/$s/labels -name '*.txt' -empty | wc -l
done
find $D -type f | wc -l
find $D -name '*.part' -exec ls -la {} \;
```
输出摘要：
```
train: images 9999 (6781 jpg, 3217 png, 1 part) ; labels 9999 txt ; class 0:18964 1:3404 2:7116 ; empty 27
val:   images 2750 (1845 jpg, 905 png)          ; labels 2738 txt + 12 part ; class 0:5020 1:1034 2:1840 ; empty 14
test:  images 2751 (1873 jpg, 878 png)          ; labels 2751 txt ; class 0:5196 1:893 2:2031 ; empty 8
total files: 31004
.part: train/images/ds2_helmet_jacket_02011.jpg.part (50820 B)
       val/labels/*.txt.part × 12 (ds1_hard_hat_workers2858/2866/2917/2972/2923/2877/2965/515/4986/4907/4994, ds2_helmet_jacket_01648)
```
README 依据：
```
task_categories: [object-detection] ; tags: [yolo, ppe-detection, safety, computer-vision, construction]
Classes: 0 helmet / 1 head / 2 vest
Data Format: class_id x_center y_center width height
```

---

## 4. Francesco__construction-safety-gsnvb

### 4.1 任务类型
**目标检测（object detection）**，Roboflow 100 (RF100) 子集 `construction-safety-gsnvb`。
依据：`dataset_info.json` 描述「This dataset was exported via roboflow.com … Construction-safety are annotated in COCO format」，特征含 `width/height/objects{id,area,bbox,category}`；无任何文本问答字段。

### 4.2 数据格式
**3 个 parquet（HF 转换版）+ `dataset.tar.gz`（原始 Roboflow 导出）+ `dataset_info.json`**。目录层级：
```
Francesco__construction-safety-gsnvb/
├── data/
│   ├── train-00000-of-00001-363ce03539eb0073.parquet
│   ├── validation-00000-of-00001-fc7c3e3c4e5acfa5.parquet
│   └── test-00000-of-00001-2a010722bb743ffa.parquet
├── dataset_info.json        (2,658 B)
├── dataset.tar.gz           (75,194,830 B ≈ 71.7 MiB)
├── README.md
└── .gitattributes
```
`dataset.tar.gz` 内部（前 2 级目录，绝对路径前缀 `home/zuppif/Documents/Work/RoboFlow/ODinW-RF100-challenge/rf100/construction-safety-gsnvb/`）：
```
construction-safety-gsnvb/
├── README.dataset.txt, README.roboflow.txt
├── train/  _annotations.coco.json + <997 张 .jpg>
├── valid/  _annotations.coco.json + <90 张 .jpg>
└── test/   _annotations.coco.json + <119 张 .jpg>
```
tar 成员总数 1217 = 1206 张图 + 3 个 COCO json + 2 个 README + 6 个目录条目。**tar 内图片与 parquet 内嵌图片是同一批 Roboflow 数据**（parquet 是 tar 的 HF parquet 化版本，`download_checksums` 也指向该 tar.gz）。

### 4.3 样本数（精确）
| split | parquet 行数 | dataset_info.json `num_examples` |
|---|---|---|
| train | **997** | 997 |
| validation | **90** | 90 |
| test | **119** | 119 |
| **合计** | **1206** | 1206 |
与 `dataset.tar.gz` 内 1206 张图片、README 的「It includes 1206 images」三方一致。`download_size` 75,194,830 B，`dataset_size` 75,740,588 B。

### 4.4 单图 / 多图
**单图，1 张/样本。** 证据：schema 中 `image: struct<bytes: binary, path: string>`（单 struct，非 list）；每行同时给出该图 `image_id/width/height`；tar 内每张图对应 COCO `images` 中一条记录。图片数 1206 == 行数 1206。

### 4.5 单轮 / 多轮
**N/A。** 数据集**完全没有文本/问答字段**（特征：`image_id, image, width, height, objects`），无 caption、无 QA、无对话，故轮数分布不适用。

### 4.6 标签体系
**6 类**（`dataset_info.json` 的 `objects.category` ClassLabel `names`）：
`construction-safety`(0), `helmet`(1), `no-helmet`(2), `no-vest`(3), `person`(4), `vest`(5)`

- bbox 格式：**COCO `[x, y, width, height]`，绝对像素**（整体 `[0,640]` 范围内）。
  证据：`image` 采样 `{"id":[99],"area":[140681],"bbox":[[53.0, 0.0, 512.5, 274.5]],"category":[1]}`；`width,height` 分布为 `{(640,640): 1206}`（全部 resize 到 640×640，与 dataset_info「Resize to 640x640 (Stretch)」一致）。
- 实盘实例数（3 个 split 合计）：`person` 2817, `helmet` 2543, `vest` 1343, `no-vest` 892, `no-helmet` 129, `construction-safety` **0**（该类别在 json/parquet 中从未出现）→ 实际有效类别 5 个，总框数 **7724**。
- 每图对象数分布（min 1，max 42，均值 ≈ 6.4）：1:12, 2:78, 3:396, 4:57, 5:51, 6:243, 7:30, 8:32, 9:122, 10:16, 11:17, 12:41, 13:12, 14:20, 15:24, 16:7, 17:3, 18:9, 19:1, 20:4, 21:9, 22:1, 23:3, 24:3, 25:3, 26:1, 27:3, 28:1, 30:2, 31:1, 34:1, 36:1, 39:1, 42:1。**没有空标注图**（每图至少 1 个框）。

### 4.7 关键字段名
`image_id` (int64)、`image` (struct: `bytes`, `path`)、`width` (int32)、`height` (int32)、`objects` (struct: `id: list<int64>`, `area: list<int64>`, `bbox: list<fixed_size_list<float>[4]>`, `category: list<int64>`)。

真实样例（validation 第 0 行，截断 ~200 字符）：
```
image = {path: "ppe_0062_jpg.rf.2b77684acf9ffd2ba956ea61c27c7185.jpg", bytes: 32407 B}
image_id = 20 ; width = 640 ; height = 640
objects = {"id": [99], "area": [140681], "bbox": [[53.0, 0.0, 512.5, 274.5]], "category": [1]}   # category 1 = helmet
```

### 4.8 证据
```bash
# 行数 + schema
python - <<'EOF'
import pyarrow.parquet as pq, glob, os
files=sorted(glob.glob('/root/autodl-tmp/mm-datasets/Francesco__construction-safety-gsnvb/data/*.parquet'))
for f in files: print(os.path.basename(f), pq.ParquetFile(f).metadata.num_rows)
print(pq.ParquetFile(files[1]).schema_arrow)
print(open('/root/autodl-tmp/mm-datasets/Francesco__construction-safety-gsnvb/dataset_info.json').read()[:800])
EOF
# 类别/框统计（iter_batches 只读 objects/width/height，不落盘）
# tar 内容清单
tar tzf .../dataset.tar.gz | head -15 ; tar tzf ... | wc -l ; tar tzf ... | grep -Eci '\.(jpg|jpeg|png)$'
```
输出摘要：
```
test-...-2a010722bb743ffa.parquet rows= 119
train-...-363ce03539eb0073.parquet rows= 997
validation-...-fc7c3e3c4e5acfa5.parquet rows= 90
TOTALS {'test': 119, 'train': 997, 'validation': 90} GRAND 1206
image_id: int64
image: struct<bytes: binary, path: string>
width: int32 ; height: int32
objects: struct<id: list<item: int64>, area: list<item: int64>, bbox: list<item: fixed_size_list<item: float>[4]>, category: list<item: int64>>
category names: ["construction-safety","helmet","no-helmet","no-vest","person","vest"]
category counts: {'helmet': 2543, 'vest': 1343, 'person': 2817, 'no-vest': 892, 'no-helmet': 129}
width,height dist: {(640, 640): 1206}
objects sample: {"id": [99], "area": [140681], "bbox": [[53.0, 0.0, 512.5, 274.5]], "category": [1]}
tar entries= 1217 ; image files in tar= 1206
dataset_info.json: "It includes 1206 images. Construction-safety are annotated in COCO format." ; splits train 997 / validation 90 / test 119
```

---

## 附录：跨数据集对比要点（供下游使用）

1. **全部为单图、非多轮**：4 个数据集的每个样本都恰好 1 张图；无任何 QA/对话字段（aswin 有 caption+reason 文本，但非对话）。
2. **三种 bbox 约定并存**，不可混用：
   - aswin: 归一化 `xyxy`（float, [0,1]）
   - keremberke / Francesco: COCO 绝对像素 `xywh`
   - jhboyo: YOLO 归一化 `cx cy w h`
3. **类别规模**：3 (jhboyo) < 6 (Francesco，实际 5 类有样本) < 17 (keremberke)。
4. **规模**：jhboyo 15,487 可用图 (45,498 框) > aswin 10,013 图 (2,078 违规框) > Francesco 1,206 图 (7,724 框) > keremberke 398 图 (1,093 框)。
5. **落盘注意事项**：
   - keremberke 图片仍在 zip 内，使用前需解压（本次分析刻意未解压）。
   - jhboyo 有 13 个 `.part` 未完成下载（1 图 + 12 标签），train/val 严格配对数为 9998 / 2738。
   - Francesco 的 `dataset.tar.gz` 与 parquet 内容重复，二者取一即可。
   - Francesco 的 `construction-safety` 类别在 3 个 split 中均为 0 实例（类名预留但无样本）。
