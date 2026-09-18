# Supplement plan — add ≈132 GB of construction / industrial-safety multimodal data

Run: 2026-09-17 · Server `connect.westb.seetacloud.com:38024` (autodl-container-jnb93wme4w-bf7ae4cd)
Destination: **`/root/autodl-tmp/mm-datasets-add/<org>__<name>/`** (new dir; existing `/root/autodl-tmp/mm-datasets/` untouched)
Transport: direct `curl -L -C -` against `https://hf-mirror.com/datasets/<repo>/resolve/main/<path>` (verified method: HTTP 206, resume works)
File plan: `/root/mm_tree_jobs.py` (mirror-native Link pagination — `huggingface_hub` pagination is broken for >1000-file repos here)

## 1. Selection (13 repos, 130.74 GB)

Sizes are **measured**, not estimated: sum of `lfs.size`/`size` over the mirror tree of the current `main`
revision (`/root/mm-verify.jsonl`), cross-checked with an HTTP `Range: bytes=0-1023` probe (all `206`).

| # | repo | GB (mirror main rev) | files | task | supervision | language | license | gated |
|---|---|---:|---:|---|---|---|---|---|
| 1 | `XimiaoZhang/MVTec-4K` | 45.69 | 6600 | industrial anomaly detection + defect segmentation (4K) | 844 `ground_truth/*_mask.png` + `.jsonl` | none | apache-2.0 | no |
| 2 | `fireviewer/fire-smoke-detection-corpus-v1` | 32.22 | 50 | fire/smoke object detection (curated multi-source) | parquet bbox+class, 15 manifests | English metadata | other (per-sample CC-BY-SA-4.0 seen) | no |
| 3 | `hiennguyen9874/fire-smoke-detection` | 11.57 | 26 | fire/smoke object detection | `objects.bbox` + `category_id` (72 159 train / 18 112 val) | none | UNKNOWN | no |
| 4 | `hayden-yuma/roadwork` | 10.52 | 33 | roadwork / work-zone scene understanding | English `scene_description` + `scene_level_tags.*` | en | UNKNOWN | no |
| 5 | `adarshchandrashekar/flame2-rgb-ir` | 8.97 | 20 | flame/fire classification, RGB + infrared pair | `label` int64 (53 451 samples) | none | UNKNOWN | no |
| 6 | `baizhanquan/FireDetectionDataset-flame-forest-flameye-wildfire` | 8.90 | 22 | wildfire fire/smoke object detection | YOLO `annotations[class_id,x,y,w,h]` (15 615 imgs) | en | cc-by-4.0 | no |
| 7 | `kevincluo/structure_wildfire_damage_classification` | 5.08 | 11 | structural damage classification (6 classes) | `label` class_label | en | cc-by-4.0 | no |
| 8 | `SRuibo/Sewer-pipe-defects` | 2.37 | 3907 | sewer pipe defect detection | 1 953 YOLO `labels/*.txt` + `classes.txt` | none | cc-by-4.0 | no |
| 9 | `iluvvatar/wood_surface_defects` | 2.20 | 7 | wood surface defect detection | `objects[bb,label]` (20 276 samples) | none | cc-by-4.0 | no |
| 10 | `Voxel51/hard-hat-detection` | 1.33 | 5006 | hard-hat / PPE object detection | `samples.json` detections (Helmet/Person/Head) | en | cc0-1.0 | no |
| 11 | `keremberke/hard-hat-detection` | 1.12 | 11 | hard-hat / PPE object detection | Roboflow COCO+YOLO zips | none | UNKNOWN (Roboflow CC-BY-4.0) | no |
| 12 | `keremberke/satellite-building-segmentation` | 0.49 | 11 | building segmentation (infrastructure) | Roboflow masks | none | UNKNOWN (Roboflow CC-BY-4.0) | no |
| 13 | `hf-vision/hardhat` | 0.27 | 4 | hard-hat / PPE object detection | `objects.bbox` + `category` (5297 + 1766) | none | UNKNOWN | no |
| | **TOTAL** | **130.74** | | | | | | |

Machine-readable list: `/root/autodl-tmp/mm-datasets-add/manifest_selected.json` (on server) and
`datasets/supplement/manifest_selected.json` (local, identical).

### Hard-criteria check (all 13 pass)
1. **Images** — every repo stores real images (png/jpg/tif/tgz-of-images) as the payload; no URL-only or video-only repos.
2. **Supervision** — detection boxes (#2,3,6,8,9,10,11,13), segmentation masks (#1,12), classification labels (#5,7), captions/tags (#4). No pure image packs.
3. **Not multi-turn** — none of the 13 has a conversation/QA field; features lists were read in full (e.g. `hiennguyen9874/fire-smoke-detection` = `image_id,image,width,height,objects`; `fireviewer` = image + metadata columns, no dialogue).
4. **Language en/zh or no text** — text-bearing repos are English (`roadwork` scene descriptions sampled via datasets-server: *"Four lane highway with jersey barriers. TTC message board and vertical panels on side of the road."*; `baizhanquan`/`kevincluo` declare `language: en`). The rest carry no text at all. Nothing Korean/Japanese/Arabic.
5. **hf-mirror reachable, non-gated, no 403** — `gated=false` on the mirror API for all 13, tree listing `200`, range probe `206`. `Piyush152/ppe-detection` and `qxnam/ppe-detection` were dropped precisely because their probe was `403`.

### Arithmetic
* existing kept corpus (after the four-criteria filter): **8.109 GB**
* new download: **130.74 GB** (target ≈132 GB, accepted 120–145 GB)
* resulting retained corpus: **8.109 + 130.74 = 138.85 GB ≈ 140 GB** ✔
* server free space before launch: **485 GB** (`/dev/md0` 1.1 T, 566 G used, 54 %) → no disk risk.

## 2. Candidates examined and rejected

Each was checked on the mirror (`/api/datasets/<id>`, tree listing, README, and where useful HTTP-range zip/parquet probes).

| repo | mirror size | reason for rejection |
|---|---:|---|
| `Piyush152/ppe-detection` | 14.79 GB (4 978 png) | **0 label files** and range probe **HTTP 403** → fails criteria 2 and 5 |
| `qxnam/ppe-detection` | 19.89 GB (4 981 files) | identical size/file-count to the above → same unlabeled pack, duplicate |
| `toth235a/concrete_crack_combined` | 2.27 GB | `Public_all.zip` = 7 056 png + 7 056 jpg, **0 label/annotation entries** → pure image pack (criterion 2) |
| `htfhgf/flame_test` | 19.25 GB | Chinese class folders (点火/熄火/稳定燃烧) but 12.24 GB of it is an unidentifiable `vim完整版本.tgz`; 2 downloads, labels unverifiable → rejected |
| `mountainmoon2000/wildfire_mm` | 33.50 GB | labels present (FASDD_CV.zip: 95 317 txt + 95 314 xml + COCO json) **but FASDD is already inside `fireviewer` corpus** (`source_name: fasdd`) → duplicate content |
| `nicolas93/test_roadwork` | 26.41 GB | English prompt / image-to-image **synthetic generation** data (i2i parquet + `prompt`/`prompt_long` json) — valid caption supervision, but kept the real `hayden-yuma/roadwork` instead; held as alternate |
| `Voxel51/mvtec-ad`, `TheoM55/mvtec_anomaly_detection`, `foersben/mvtec-ad`, `introvoyz041/mvtec-ad`, `OctorVX/mvtec-ad`, `box0602/mvtec-ad`, `pmrozo/mvtec-ad` | 5.27 GB each | duplicates of MVTec-AD already covered by `MVTec-4K`; Voxel51 copy carries labels only via `samples.json` |
| `Hajorda/flameye-wildfire-detection` | 8.90 GB | **byte-identical** tree to `baizhanquan/...` (same 19 parquet shards, same per-file sizes) → duplicate, one kept |
| `MahedixHasan/pose-guided-fall-detection-icta2026` | 13.89 GB | paper/code repo whose payload is a **video** archive (`data.zip`, MCFD AVI) → no per-sample images (criterion 1) |
| `romainpuech/wildfire-drone-routing-data` | 17.65 GB | `.npy` rasters + shapefiles + XML, not images with labels (criteria 1–2) |
| `physicl/indoor-anomaly-detection-path-obstruction-monitoring` | 33.13 GB | same publisher/format as the already-DROPped `physicl/indoor-safety-hazard-...` (无标注) → consistent exclusion |
| `CypressLI/3Dlarge_construction` | 81.92 GB | COLMAP/pixsfm 3-D reconstructions (`colmap_results.zip`, `*.tgz`) → not labeled image samples |
| `fucthin/fire_smoke` | 328.39 GB | misleading name: a bundle of Flickr30k/GQA/Objects365/cc3m tars; 100 GB+ single point and not fire-specific |
| `dalle-mini/open-images` 1 036 GB · `vikhyatk/openimages-bbox` 609 GB · `finedet/openimages` 597 GB · `Salesforce/blip3-grounding-50m` 163 GB · `XAI/OpenImages-Inpainted` 131 GB | ≥131 GB each | **100 GB+ single points** → excluded by the download-risk rule |
| `Obscure-Entropy/ImageCaptioning_EN-HU` 892 GB · `visheratin/laion-coco-nllb` 416 GB · `ituperceptron/image-captioning-turkish` 75 GB | | non-en/zh text languages (criterion 4), or 100 GB+ |
| `Peacockery/weapon-detection-runs-backup-2026-07-24` / `-workerssd-backup-...` | 79.44 / 42.44 GB | "backup" dumps with 17 772 files, weapon domain (security, not construction) |
| `lvlm-anomaly-detection/datasets` 26.5 GB · `sehbeygi79/Anomaly-Detection-Datasets` 19.8 GB · `gagan0716/...` 15.7 GB · `shxhe/AnomalyDetection` 10.1 GB · `meksamiao/mmad-anomaly-detection` 11.1 GB | | heterogeneous dataset dumps with unverifiable supervision → not selected |
| `pppenner/edge-agent-reasoning-websearch-260k` | 56.68 GB | text-generation/QA web-agent traces → multi-turn dialogue (criterion 3) |
| medical / plant-disease / autonomous-driving / satellite corpora (chest X-ray, NIH, KITTI-Depth, plantvillage, …) | — | off-domain; **not needed** — in-domain supply alone exceeded 130 GB |

### Good in-domain alternates held in reserve (verified reachable, not downloaded)
`Junhan0518/Traffic_Cone` (18.41 GB), `himanshu1257/industrial-defect-dataset` (10.39 GB),
`rohanath/insulator-defect-detection` (9.67 GB), `akiii1234/RoadDamageDetection` (8.97 GB),
`pjramg/main_ppe_subset` (7.09 GB), `nicolas93/test_roadwork` (26.41 GB), `mountainmoon2000/wildfire_mm` (33.50 GB).
They can be appended by adding lines to `REPOS=()` in `/root/mm-datasets-add.sh` and re-running it (it is resumable).

## 3. Why a repo's `usedStorage` ≠ real download size
The mirror's `/api/datasets/<id>` `usedStorage` counts every stored revision, so it over-reports the current
`main` revision — e.g. `fireviewer` 78.10 GB vs **32.22 GB** real, `Voxel51/mvtec-ad` 15.47 vs 5.27,
`hf-vision/hardhat` 2.69 vs 0.27. All figures in this plan use the **tree sum of the current revision**,
which is exactly what `curl` will fetch.
