# Group D — Render Data-Pack & Traversability (mirror analysis)

Server: `ssh -p 38024 root@connect.westb.seetacloud.com`
Data root: `/root/autodl-tmp/mm-datasets/`
Python: `export PATH=/root/miniconda3/bin:$PATH` (pandas/pyarrow only, no new installs)
Access: read-only. No data modified or deleted; training dirs untouched.

---

## 1. `physicl__indoor-safety-hazard-detection-and-work-zone-monitoring` (42G)

### 1.1 任务类型
**渲染数据包 (render data-pack) — 合成多 pass 渲染，非监督视觉任务。**
依据：
- README front-matter tags 明写 `synthetic-data`, `rendering`, `data-pack`；config 名为 `renders`。
- README 原文：「Each row represents one render view. The `image` column contains a stable URL to the primary render image... Additional render passes are exposed as URL columns.」
- 全库没有 label / bbox / mask / QA 列；`metadata`（README 称 "Core `renderOutput.metadata` serialized as JSON"）在全部 400 行均为 `null`（见 1.5）。
- 它**不是**分类/检测/分割/VQA 数据集；可用作扩散/渲染/多通道重建类任务的素材（beauty + albedo + depth + normal + material_index + attenuation + uvw 等 G-buffer pass 对）。

### 1.2 数据格式
HuggingFace `parquet`（仅 1 个 204 KB 的 viewer parquet，只存 URL/元数据）+ 42G 独立 PNG 图片目录。

目录层级：
```
physicl__indoor-safety-hazard-detection-and-work-zone-monitoring/
├── README.md              (978 B)
├── .gitattributes
├── data/                  (42G, 4400 个 .png, 扁平单层)
│   ├── 003cb9d0c486ac41-M02_Kitchen_100x_dl_0000_F1_colored_depth_0006.png
│   ├── 0019ca0eb7fa2caf-M02_Kitchen_100x_0000_F2_albedo_0005.png
│   └── ... (4400 files)
└── viewer/
    └── renders/
        └── train-00000-of-00001.parquet   (204 KB, 400 rows, 1 row group)
```
注意：parquet 里**没有图片字节**，只有 `https://huggingface.co/.../resolve/<commit>/data/<name>.png` URL 与相对路径 `image_path`；真实像素在 `data/` 的 4400 个 PNG 里（`data_commit_sha = f36d576f3c2ef481d30bcef6dbac7e7cefd024cf`）。PNG 分辨率见 1.6（2048×1536）。

### 1.3 样本数
| 量 | 值 | 证据 |
|---|---|---|
| parquet 行数 | **400** | `ParquetFile(...).metadata.num_rows` → 400；row_groups=1 |
| PNG 文件数 | **4400** | `find … -name "*.png" \| wc -l` → 4400 |
| 每行图片 URL 列 | **11**（全部非空） | 11 × 400 = 4400，与磁盘文件数完全吻合 |
| URL 指向文件缺失数 | **0** | 4400 个 URL 逐条 `os.path.exists` 全部命中 |

### 1.4 单图还是多图 —— **同一视角的多 pass 多图（11 通道/视图）**
这是一个 view（`file_name` 尾号 `_NNNN` 是 view index）× **11 个渲染通道**，而不是多视角多帧。

图片相关列共 **11** 个（`f.schema_arrow.names` 顺序）：
1. `image` — 主图 / **beauty** 主渲染（README 明示）
2. `normal_world_gl_srgb`
3. `normal_world_dx_linear`
4. `depth_rel`
5. `colored_depth`
6. `material_index`
7. `albedo`
8. `attenuation`
9. `uvw`
10. `normal_world_dx_srgb`
11. `normal_world_gl_linear`

（另有 `available_passes` 字符串列，声明 12 个 pass：`albedo, attenuation, beauty, colored_depth, depth_metric, depth_rel, material_index, normal_world_dx_linear, normal_world_dx_srgb, normal_world_gl_linear, normal_world_gl_srgb, uvw`。**声明了 12 个但只暴露 11 列**：`beauty` 即 `image` 列，剩 `depth_metric` 没有对应列、磁盘上也没有 `depth_metric` 的 PNG —— 该 pass 未镜像/未导出。）

**关键证据（同视角而非多视角）：** 400/400 行的 11 个图片列文件名共享**同一个尾部 view index** `_NNNN`（`bad=0`），例如 row 0 的 11 个 basename 全部以 `_0001.png` 结尾，仅 pass 名不同：
```
M02_Kitchen_100x_0000_0001.png                          <- image (beauty)
M02_Kitchen_100x_0000_normal_world_gl_srgb_0001.png
M02_Kitchen_100x_0000_normal_world_dx_linear_0001.png
M02_Kitchen_100x_0000_depth_rel_0001.png
M02_Kitchen_100x_0000_colored_depth_0001.png
M02_Kitchen_100x_0000_material_index_0001.png
M02_Kitchen_100x_0000_albedo_0001.png
M02_Kitchen_100x_0000_attenuation_0001.png
M02_Kitchen_100x_0000_uvw_0001.png
M02_Kitchen_100x_0000_normal_world_dx_srgb_0001.png
M02_Kitchen_100x_0000_normal_world_gl_linear_0001.png
```
场景结构：单一环境 `M02_Kitchen_100x`，4 个组合分支（`_0000` / `_0000_F2` / `_dl_0000` / `_dl_0000_F1`，各 100 行 × view index 0001–0100）。`camera_name` 恒为 `Camera_004`（唯一），`focal` 恒为 `24.95833396911621`，`environment_id` 全为 null。也就是说：**同一台相机、同一视角，横向扩展 11 个 G-buffer pass**，不存在多相机/多视角。

### 1.5 单轮还是多轮 —— **N/A（纯视觉渲染包，无任何对话/QA）**
- `metadata` 列虽然在 schema 中是 `string`（README 称序列化 JSON），但**在 400 行中 400 行全为 `null`**：`non-null metadata: 0`，metadata key 集为空 `[]`，QA-like key 为空 `[]`。
- 因此**不能**声称 metadata 内含文本或问答；本数据集无单轮/多轮概念 → **N/A**。

### 1.6 关键字段名 + 真实样例（截断 ~250 字符）
关键字段：`id, name, data_pack_id, version_id, version_number, combination_id, source_combination_id, environment_id, camera_name, light_id, focal, file_name, available_passes, image, image_path, data_commit_sha, metadata, normal_world_gl_srgb, normal_world_dx_linear, depth_rel, colored_depth, material_index, albedo, attenuation, uvw, normal_world_dx_srgb, normal_world_gl_linear`

Row 0 真实样例（单行、截断）：
```
id                              = 3d9150e0-ae54-456a-a014-fd9c979ad7ff
name                            = Indoor Safety Hazard Detection & Work-Zone Monitoring
version_number                  = 1
combination_id                  = 1b501341-eeeb-47df-9682-e44810ccf9e6
camera_name                     = Camera_004
focal                           = 24.95833396911621
file_name                       = M02_Kitchen_100x_0000_0001.png
available_passes                = ["albedo", "attenuation", "beauty", "colored_depth", "depth_metric", "depth_rel", "material_index", "normal_world_dx_linear", "normal_world_dx_srgb", "normal_world_gl_linear", "normal_world_gl_srgb", "uvw"]
image                           = https://huggingface.co/datasets/physicl/indoor-safety-hazard-detection-and-work-zone-monitoring/resolve/f36d576f3c2ef481d30bcef6dbac7e7cefd024cf/data/c370d27a164de683-M02_Kitchen_100x_0000_0001.png
image_path                      = data/c370d27a164de683-M02_Kitchen_100x_0000_0001.png
data_commit_sha                 = f36d576f3c2ef481d30bcef6dbac7e7cefd024cf
metadata                        = None
normal_world_gl_srgb            = .../data/81722f9ed328635b-M02_Kitchen_100x_0000_normal_world_gl_srgb_0001.png
depth_rel                       = .../data/f59a3717ef67d6a3-M02_Kitchen_100x_0000_depth_rel_0001.png
albedo                          = .../data/5c1e46f1a5fae5af-M02_Kitchen_100x_0000_albedo_0001.png
```
磁盘 PNG 实际尺寸（读 PNG IHDR）：**2048 × 1536**；单文件 9–19 MB。

### 1.7 证据（命令 + 输出摘要）
```bash
# schema / 行数
ParquetFile(p).metadata.num_rows            -> NUM_ROWS: 400 ; NUM_ROW_GROUPS: 1
f.schema_arrow.names                        -> ['id','name',...,'metadata','normal_world_gl_srgb',
                                                'normal_world_dx_linear','depth_rel','colored_depth',
                                                'material_index','albedo','attenuation','uvw',
                                                'normal_world_dx_srgb','normal_world_gl_linear']
# 非空与 pass 统计
for c in imgcols: non-null                  -> 11 个图片列均 400/400 非空
metadata non-null                           -> total rows: 400  non-null metadata: 0
available_passes distinct                   -> 1（400 行全同，12 个 pass 名）
distinct image-URL cols per row             -> {11: 400}
# 同视角校验 + 磁盘校验
rows sharing SAME trailing view-index       -> 400 / 400   bad: 0
total image URLs checked: 4400   missing on disk: 0
view groups                                 -> {'M02_Kitchen_100x_0000':100, '..._0000_F2':100,
                                                '..._dl_0000':100, '..._dl_0000_F1':100}
# 目录
find … -name "*.png" | wc -l                -> 4400
ls viewer/renders/                          -> train-00000-of-00001.parquet (208633 B)
du -sh data viewer                          -> 42G  data ; 204K  viewer
PNG IHDR                                    -> 2048 1536
```

---

## 2. `Voxel51__Construction-Site-Traversability` (17G)

### 2.1 任务类型
**机器人多模态传感器录制（FiftyOne native multimodal MCAP episode）— 不是可通行性分割/标注数据集。**
依据：
- `fiftyone.yml`: `format: FiftyOneDataset`，`media_type: multimodal`（metadata.json 同），tags = `mcap, robotics, lidar, depth, imu, gnss, odometry, traversability`。
- README「What you get」逐条列出的是 MCAP 频道：`/camera`(640×480 CompressedImage)、`/depth`(640×400 CompressedImage)、`/lidar-points`(foxglove.PointCloud)、`/imu.plot`、`/livox-imu.plot`、`/odometry`、`/wheel-odometry.plot`、`/gnss`、`/camera-calibration`、`/tf`、`/session`。
- **没有标签**：`metadata.json` 中 `classes = {}`、`mask_targets = {}`、`annotation_runs = {}`、`runs = {}`、`label_schemas = {}`、`frame_fields = []`；`samples.json` 每个 sample 只有计数/元数据字段，无 mask/detection/segmentation 字段。
- README 明确把标注分离出去：*"The curated annotated frames, camera calibration and the trained segmentation model that accompany these recordings are published separately by the authors at `manojkarnekar/construction-traversability-dataset`."* → 本镜像里**没有**那份带标注的可通行性数据。
- 结论：task type = **其它（robotics 多模态录制 / 原始传感器流）**；名称里的 "Traversability" 只是用途声明。**本镜像不含 traversability 分割或标注**（若需要标注需另取 `manojkarnekar/construction-traversability-dataset`，不在本次镜像内 → 该部分 UNKNOWN/UNVERIFIED）。

### 2.2 数据格式
`fiftyone json + 二进制媒体目录`（FiftyOneDataset 导出：`samples.json` + `metadata.json` + `fiftyone.yml` + `calibration/*.yaml` + `data/**/*.fo.mcap`）。

目录层级：
```
Voxel51__Construction-Site-Traversability/
├── README.md
├── .gitattributes
├── fiftyone.yml
├── metadata.json                 (8328 B, FiftyOne dataset 元数据)
├── samples.json                  (3105 B, 4 个 sample)
├── preview.gif                   (1784674 B)
├── calibration/
│   ├── camera_intrinsics.yaml        (OAK-D RGB, 640x480, fx=513.8646...)
│   └── lidar_camera_extrinsics.yaml  (base_footprint / livox_frame / oakd_rgb_frame)
└── data/
    ├── site1_session01/site1_session01.fo.mcap   (8410848174 B ≈ 8.4 GB)
    ├── site1_session02/site1_session02.fo.mcap   (2441287673 B ≈ 2.4 GB)
    ├── site2_session01/site2_session01.fo.mcap   (4778893653 B ≈ 4.8 GB)
    └── site2_session02/site2_session02.fo.mcap   (2508896631 B ≈ 2.5 GB)
```
`data/` 下**只有 4 个 .mcap 文件**，没有任何独立图片文件。

### 2.3 样本数
| 量 | 值 | 证据 |
|---|---|---|
| `samples.json` sample 数 | **4** | `len(json.load(...)["samples"])` → 4 |
| mcap episode 文件数 | **4** | `find data -type f \| wc -l` → 4 |
| 等效图片帧（媒体内，未解码） | 62,995 colour + 62,931 range | README 汇总；samples.json 每 episode 计数之和 = 8696+8854+17226+28219 = **62,995** |
| LiDAR sweeps | 62,995（0.82B points） | 同上 |
| IMU samples | 1,888,884 | README |
| GNSS fixes | 4,986 | README |

### 2.4 单图还是多图 —— **每 sample = 1 个多传感器、多帧 episode（远多于 1 张图）**
`filepath` 证据（`samples.json`，4 条）：
```
data/site2_session02/site2_session02.fo.mcap   num_camera_frames=8696   num_depth_frames=8695   num_lidar_scans=8696
data/site1_session02/site1_session02.fo.mcap   num_camera_frames=8854   num_depth_frames=8854   num_lidar_scans=8855
data/site2_session01/site2_session01.fo.mcap   num_camera_frames=17226  num_depth_frames=17201  num_lidar_scans=17225
data/site1_session01/site1_session01.fo.mcap   num_camera_frames=28219  num_depth_frames=28181  num_lidar_scans=28219
```
- `filepath` 指向**单个 `.fo.mcap` 容器**（`_media_type: "multimodal"`），里面按时间轴装着**数千至数万帧**：每 episode 8,696–28,219 张 colour 帧 + 几乎同数的 depth 帧 + 同数 LiDAR sweep。
- 相机方面是 **1 个相机 / 多帧时序**（OAK-D RGB **单目**，README: "the OAK-D colour camera"、`camera_name: oakd_rgb`；`camera_intrinsics.yaml` 也只有一台 OAK-D RGB 的 640×480 内参），并非多相机阵列。
- 所以：不是「1 sample = 1 图」，而是「1 sample = 一条长时序多模态 episode（单相机多帧 + LiDAR + IMU + GNSS + 里程计）」。
- 帧率：colour 与 LiDAR 各 10 Hz，range 相机 20 Hz 抽帧到 10 Hz（README「Notes on the conversion」）。

### 2.5 单轮还是多轮 —— **N/A**（纯传感器录制；`samples.json` 无任何 prompt/question/answer/caption 字段，`sample_fields` 全是计数与元数据）

### 2.6 关键字段名 + 真实样例（截断 ~250 字符）
字段：`filepath, sequence, site, session, recorded, scene, num_camera_frames, num_depth_frames, num_lidar_scans, num_lidar_points, num_imu_samples, num_livox_imu_samples, num_odometry_poses, num_gnss_fixes, has_gnss, odometry_path_m, duration`

`metadata.json` 顶层：`_id, name, slug, version, created_at, last_modified_at, last_deletion_at, last_loaded_at, sample_collection_name, persistent, media_type, group_media_types, tags, info, app_config, classes, default_classes, mask_targets, default_mask_targets, skeletons, camera_intrinsics, static_transforms, sample_fields, frame_fields, saved_views, workspaces, annotation_runs, brain_methods, evaluations, runs, active_label_schemas, label_schemas, frame_label_schemas`

`samples.json` 首条真实样例（截断）：
```
{"_id": {"$oid": "6aa42a60bd04feee501d7809"}, "filepath": "data/site2_session02/site2_session02.fo.mcap",
 "tags": [], "_media_type": "multimodal", "_rand": 0.9997072649998512, "sequence": "site2_session02",
 "site": "site2", "session": "session02", "recorded": "2026-08-05", "scene": "site2 session02",
 "num_camera_frames": 8696, "num_depth_frames": 8695, "num_lidar_scans": 8696,
 "num_lidar_points": 134346063, "num_imu_samples": 86945, "num_livox_imu_samples": 173954,
 "num_odometry_poses": 86944, "num_gnss_fixes": 870, "has_gnss": true,
 "odometry_path_m": 538.3991972183628, "duration": 869.618607579, ...}
```

### 2.7 证据（命令 + 输出摘要）
```bash
ls -la $R                                    -> samples.json 3105 B, metadata.json 8328 B, fiftyone.yml 553 B
find $R/data -maxdepth 2                     -> 4 个 site*_session*/*.fo.mcap，无图片文件
find $R/data -type f | wc -l                 -> 4
find $R/data -type f -printf "%s %p\n"       -> 8410848174 site1_session01.fo.mcap, 4778893653 site2_session01,
                                                2508896631 site2_session02, 2441287673 site1_session02  (≈17G)
len(json.load(samples.json)["samples"])      -> Voxel51 sample count: 4
metadata.json media_type                     -> multimodal ; group_media_types -> {}
classes={} mask_targets={} annotation_runs={} runs={} label_schemas={} frame_fields=[]
sample_fields                                -> ['_dataset_id','filepath','_rand','id','created_at','_media_type',
                                                'metadata','last_modified_at','tags','sequence','site','session',
                                                'recorded','scene','num_camera_frames','num_depth_frames',
                                                'num_lidar_scans','num_lidar_points','num_imu_samples',
                                                'num_livox_imu_samples','num_odometry_poses','num_gnss_fixes',
                                                'has_gnss','odometry_path_m','duration']
calibration/camera_intrinsics.yaml           -> camera_name: oakd_rgb ; image_width: 640 ; image_height: 480 ; fx 513.8646
# 未解码帧：mcap 包不可用 -> `import mcap` -> ModuleNotFoundError（未安装，遵守不装新包约束）
```

---

## 3. `pyimagesearch__construction-safety-object-detection-paligemma` (4.0K) — **BLOCKED / 不可判定（数据未镜像）**

### 3.1 状态
**BLOCKED（数据文件缺失，HF 镜像端 gated → HTTP 403）**。本地镜像目录只有 README.md，`data/` 为空目录：`ls -la $R` → 仅 `README.md (1337 B)` + 空 `data/`；`find $R -type f` → 只有 README.md；无 `.gitattributes`。因此除 README 声明外**一切字段级/样本级结论均不可验证**。

### 3.2 任务类型（依 README 应然，NOT VERIFIED）
**目标检测（object detection）→ 转成 PaliGemma 指令格式（detection-as-text / vision-language）**。
依据（README front-matter 特征声明）：
- 特征 `image` (`dtype: image`)、`width`/`height`、`objects`（`sequence`，含 `id`、`area`、`bbox` (float32 ×4)、`category` class_label）。
- `category` 17 类（construction safety PPE/机械）：`barricade, dumpster, excavators, gloves, hardhat, mask, no-hardhat, no-mask, no-safety vest, person, safety net, safety shoes, safety vest, dump truck, mini-van, truck, wheel loader`。
- 另有 `paligemma_labels` (`dtype: string`) —— 说明是检测标注被序列化为 PaliGemma 风格文本标签（具体模板文本**未知**，因文件不可得）。

### 3.3 数据格式（应然，NOT VERIFIED）
Parquet（README configs 声明 `data/train-*`、`data/validation-*`、`data/test-*`）。目录层级应然：
```
pyimagesearch__construction-safety-object-detection-paligemma/
├── README.md
└── data/          <- 空（train-*/validation-*/test-* parquet 缺失，镜像 403）
```

### 3.4 样本数（README 声明，NOT VERIFIED）
| split | num_examples | num_bytes |
|---|---|---|
| train | **307** | 22,414,831 |
| validation | **57** | 3,378,116 |
| test | **34** | 1,825,733 |
| 合计 | **398** | `dataset_size: 27,618,680`, `download_size: 27,597,011` |

（README 另注 `size_categories` 未给；以上为 front-matter `dataset_info.splits` 自述值，未经数据校验。）

### 3.5 单图还是多图（应然，NOT VERIFIED）
特征是单个 `image` (`dtype: image`) 列 → **(应然) 每样本 1 张图**；`objects` 是该图上的多个 bbox+category 标注（多目标，不是多图）。

### 3.6 单轮还是多轮（应然，NOT VERIFIED）
`paligemma_labels` 为**单个 string** → **(应然) 单轮**（一图一文本，检测标签/指令）。是否存在多轮对话**不可判定**（无数据、无模板文本）。

### 3.7 关键字段名 + 样例
关键字段（README）：`image_id (int64)`, `image (image)`, `width (int32)`, `height (int32)`, `objects[].id/area/bbox[4]/category`, `paligemma_labels (string)`。
**真实样例：UNKNOWN** —— parquet 缺失（gated/HTTP 403），无法读取任何一行；不编造。

### 3.8 证据
```bash
ls -la $R                 -> total 8 ; README.md 1337 B ; data/ (drwxr-xr-x 2, size 10 => 空)
find $R -type f           -> 只有 README.md（无 parquet）
cat $R/README.md          -> dataset_info.features = [image_id, image, width, height, objects[...], paligemma_labels]
                             splits: train 307 / validation 57 / test 34
                             configs: data_files -> data/train-*, data/validation-*, data/test-*
# HF 端 data/ 为 gated，未镜像 -> HTTP 403（本次未做网络重试，按任务前提记为 BLOCKED）
```

---

## 总表

| 数据集 | 任务类型 | 单轮/多轮 | 单图/多图 | 样本数 | 格式 | 状态 |
|---|---|---|---|---|---|---|
| `physicl__indoor-safety-hazard-detection-and-work-zone-monitoring` | **渲染数据包**（合成多 pass 渲染，无监督标签） | **N/A**（`metadata` 列 400/400 全为 null，无 QA） | **同一视角的 11 路多 pass 多图**（1 beauty 主图 + 10 pass；camera/focal 唯一；400/400 行 11 列共享同一 view index） | **400 行 / 4400 PNG**（11×400） | 1 个 viewer parquet（URL+元数据，204 KB）+ 42G 扁平 `data/*.png`（2048×1536） | ✅ 已核验 |
| `Voxel51__Construction-Site-Traversability` | **其它：机器人多模态 MCAP 录制**（原始传感器流）；**不含** traversability 分割/标注（标注另发布） | **N/A**（无文本/QA 字段） | **每 sample = 1 条多模态多帧 episode**（单目 OAK-D 8,696–28,219 帧 + 等量 depth/LiDAR + IMU/GNSS），非单图 | **4 samples / 4 个 .fo.mcap**（内含 62,995 colour 帧 + 62,931 range 帧） | fiftyone `samples.json`+`metadata.json`+`fiftyone.yml`+`calibration/*.yaml` + 17G `data/**/*.fo.mcap` | ✅ 已核验（帧内容未解码，仅计数） |
| `pyimagesearch__construction-safety-object-detection-paligemma` | (应然) **目标检测 → PaliGemma 文本标签** | (应然) **单轮**（`paligemma_labels` 单 string）；多轮不可判定 | (应然) **1 图/样本**（单 `image` 列 + 多 bbox） | (README) train 307 / val 57 / test 34 = **398** | (应然) parquet `data/{train,validation,test}-*` | ⛔ **BLOCKED**：数据未镜像（gated/403），`data/` 为空；除 README 声明外 **UNKNOWN/不可判定** |

### 关键结论
1. **physicl = 渲染数据包**，400 行 × 11 pass = 4400 图；核心特征是 **「1 视角多通道」**，绝不能当成 400 张图的普通图像集，也不能当成多视角多图；`depth_metric` 在 `available_passes` 中被声明但**无对应列/文件**。
2. **Voxel51 = 机器人多模态录制**，4 个 MCAP episode；**名称为 traversability，但本镜像无任何标注/掩码**，带标注版本在另一个 repo (`manojkarnekar/construction-traversability-dataset`)。单样本图数不是 1，而是每个 episode 8k–28k 帧。
3. **pyimagesearch 数据不可得（BLOCKED）**，只能按 README 推断「检测 + PaliGemma 单轮单图」，全部标 UNKNOWN/not verified。
