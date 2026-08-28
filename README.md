# SR-Diffusion-v3

**项目目标（权威版，2026-08-28，详见 [`doc/2026-08-28/GOAL_compression_for_nlp.md`](doc/2026-08-28/GOAL_compression_for_nlp.md)）**：
通过 **token 压缩**训练编码器的**联想能力**（把图像信息压进少量 special token z_s），训练完成后**冻结编码器**，作为 `model.py` 的编码器进行 **NLP 解码训练**（Qwen 生成中文工地描述/隐患）。

⚠️ **像素重建只是 Phase 1 的训练脚手架（代理任务），不是最终目标**——验收标准是 Phase 2 冻结编码器后的文字生成质量，不是像素 L1。`doc/2026-08-27/DIAGNOSIS_clarity.md` 的"追清晰度"分析在新目标下需按 GOAL 文档 §3 重新解读。

Phase 1 架构：DINOv2-large(不冻结) → ReEncoder(因果 specials 前缀链) → OutputQueryDecoder(输出查询注意力 + KV 因果 + 平方采样) → PixelHead → **重建原始像素（脚手架）**。

当前为最小可运行主干：工地图像素重建（448×252 → 576 patches）。实验文档与历史脚本按日期归档在 `doc/`。

## 架构

```
448×252 原图 (1600:900 画布预处理)
    │
    ▼  [DINOv2-large, 不冻结]  (304M)
patch 特征 (B, 577, 1024)
    │
    ▼  [ReEncoder 4层]  [cls; specials; patches] 因果 specials 块掩码
z_cls, z_s (B, 577, 1024)
    │
    ▼  [OutputQueryDecoder]  平方采样 25 步, KV 因果前缀
F_hat (B, 576, 1024) 特征
    │
    ▼  [PixelHead]  Linear 1024→588
像素 patch (B, 576, 588) → 重建 448×252
```

**目标 = 原始像素 pixel_values**（L1 平权全覆盖，所有采样步 mean）。2026-08-27 重大修复：此前监督 DINO patch 特征会退化（工地图特征空间近常数，学质心即低 L1，假收敛）；像素目标有真实空间结构，强制模型保留空间信息。

## 快速开始

```bash
# 1. 训练（2 GPU DDP, 全 fp32, 平权, bs16/卡）
NUM_GPUS=2 ./run_v2_train.sh \
    --data_dir /root/autodl-tmp/construction_site \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --output_dir output/phase1_v2_pixel_fp32 --epochs 40

# 2. 推理测试（全量 test, fp32, 像素 L1 + 渐进曲线）
python infer_v2_test.py \
    --data_dir /root/autodl-tmp/construction_site \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --final_model output/phase1_v2_pixel_fp32/final_model.pt \
    --output output/phase1_v2_pixel_fp32/infer_test.json

# 3. 重建可视化（原图 vs 各采样步）
python visualize_recon_pixel.py \
    --data_dir /root/autodl-tmp/construction_site \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --final_model output/phase1_v2_pixel_fp32/final_model.pt \
    --out output/phase1_v2_pixel_fp32/recon_visual.png
```

自检: `python model_v2.py`（形状 / 块掩码 / 梯度 / PixelHead / eval 同路径）。

## 目录结构

```
├── model_v2.py               # 核心模型（像素目标, 平权损失）
├── data_v2.py                # 数据管线（1600:900 画布 → 448×252）
├── train_v2.py               # 训练（HF Trainer, fp32, 平权, bs16）
├── run_v2_train.sh           # 训练入口（NUM_GPUS=n）
├── infer_v2_test.py          # 推理测试（像素 L1 + 渐进曲线）
├── visualize_recon_pixel.py  # 重建可视化
├── model.py / train.py       # v1 模型（SVD 思路, 存档）
└── doc/                      # 实验文档与历史脚本（按日期归档）
    ├── 2026-08-26/           # 第一轮: 特征目标 + 泄露排查
    ├── 2026-08-27/           # 像素目标训练 + 清晰度诊断
    └── 2026-08-28/           # 目标权威版: token 压缩练联想 → 冻结接 NLP
```

## 已知结果（2026-08-27 像素目标版，脚手架口径）

> 按项目目标（GOAL 文档 §2），以下像素指标是**训练压力/脚手架有效性**的度量，
> **不是验收标准**；验收标准是 Phase 2 冻结编码器后的 NLP 文字生成质量。

- 全量重建像素 L1 (0-255) = **23.41**（全图平均色参照 ≈61，改善 2.6×）→ 像素目标修复成功，编码器确实保留了空间信息
- 渐进曲线: t=0（仅 1 键）L1=60（粗糙）→ t≥1 L1=22.7（精细）→ **联想能力已成立**（2 键≈577 键）
- 重建偏平滑（边缘 ≈ 原图 1/3）：按新目标**不追清晰度**（差距主要是块内高频纹理，与语义无关）；机制分析见 `doc/2026-08-27/DIAGNOSIS_clarity.md`，新目标下的重新解读见 GOAL 文档 §3
- 未决: 压缩率 K（32/64/128）、DINO 冻结策略、是否叠加文字 CE——见 GOAL 文档 §4

## 环境

- torch ≥ 2.12, transformers 5.x, accelerate, datasets, safetensors（`requirements.txt`）
- 数据: parquet（image 列），预处理确定性（最优角旋转 + 等比缩放 + 1600:900 填充）
- 模型: DINOv2-large（ModelScope 下载, `/root/autodl-tmp/models/dinov2-large`）
