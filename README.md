# SR-Diffusion-v3（main 分支: v2 最小可运行代码 + 全部实验文档）

**项目目标（权威版）**: 通过 **token 压缩**训练编码器的**联想能力**（把图像信息压进少量 special token z_s，同时根据图片的信息量动态调整token的数量），训练完成后冻结编码器做 **Phase 2 NLP**（Qwen 生成工地描述/隐患）。像素重建是 Phase 1 的训练脚手架 + "信息保持"直接探针，不是最终目标；验收 = Phase 2 文字生成质量（详见 [`doc/2026-08-28/GOAL_compression_for_nlp.md`](doc/2026-08-28/GOAL_compression_for_nlp.md) §2）。

---

## 0. 2026-09-04 仓库整理说明（重要，先读）

本次对仓库做了**结构性迁移**，后续分析/复现前请先理解：

| 分支 | 内容 | 说明 |
|---|---|---|
| **main**（本分支） | ① **v2 最小可运行代码** ② `doc/` 全部实验/分析文档 | 日常开发/分析主分支 |
| **test** | main 曾有的**除文档外的全部内容**归档：v1/v3/v4/v5 代码、`run_v2_boundary.sh`、`ds_config_zero2.json`、旧版 README 等 | **历史代码只在这里**（25 个根文件，无 doc/） |
| dev | **已删除** | 旧"v2 最小可运行集"分支，tip = `04cfc02`（本地安全 tag `pre_reorg_dev`） |

- v2 最小可运行集 = `model_v2.py`、`train_v2.py`、`infer_v2_test.py`、`data_v2.py`、`visualize_recon_pixel.py`、`run_v2_train.sh`、`requirements.txt`、`.gitignore`。
- 其余曾经在 main 的一切（v1 `model.py/train.py`、v3/v4/v5 全套、boundary 实验脚本、deepspeed 配置、旧 README）都移到了 **test**。
- 文档（`doc/<日期>/…`）**只保留在 main**；test 上没有 doc/。

### ⚠️ 迁移可能对后续分析带来的问题 & 解决方案

1. **文档里的复现命令引用了 main 上已不存在的文件**：`doc/` 大量实验报告（尤其 2026-09-01/09-02 及更早）的复现命令引用 `train_v3.py / train_v4.py / train_v5.py / infer_v3..5.py / model_v3..5.py / visualize_v3..5.py / run_v2_boundary.sh`（如"按 train_v4.py 配方重训"等），这些文件现在**只在 test 分支**。
   - 方案：复现历史实验 → `git checkout test` 后在 test 里跑（对照 main 的 doc 阅读）；只取个别文件 → `git show test:<path>` 或 `git checkout test -- <path>`；要基于旧代码开新实验 → `git branch <新分支> test`。
2. **文档头部标注的代码 commit 与 main 当前 HEAD 可能不一致**：每份报告头部都记录了当时运行代码的 commit（如 `b295016`/`66aa9d2`/`36cf777`）。机制分析/复现时要**以文档头部的 commit 为准**，而不是假设 main HEAD == 当时代码。
   - 方案：需要"当时的代码" → `git checkout <文档标注的 commit>`（v2 代码全历史都在 main 的分支历史上）；v2 代码自 `b295016` 起语义兼容，`36cf777` 加的 `SRV2_MEMORY_OPEN`（读侧掩码开关）默认关闭、不影响旧配置复现。
3. **被删/被重写的分支内容**：dev 已删、test 被重写（旧 test tip = `7fdb98a`）。本地保留安全 tag `pre_reorg_dev` / `pre_reorg_test` 可随时找回旧 tip；远程侧被删对象 GitHub 会在一段时间后 GC——需要长期保留时把本地 tag push 回远程，或在整理前 `git clone --mirror` 全量备份。
4. **文档只留在 main**：在 test 上开发/复现实验后，新的报告与分析**请提交回 main 的 `doc/<日期>/`**（`git checkout main` → 写文档 → push），避免文档在两边分裂；v2 主线代码改动提交回 main，只属于归档代码的改动留 test。
5. **若将来需要"多代代码同仓"分析**：从 test 取回文件（`git checkout test -- <files>`）或把 test 并入新分支即可；不建议直接改回整理前的 main 结构（会再次混淆"当前 v2 代码"与"历史代码"）。

---

## 1. v2 是什么（本分支保留的代码）

register 式（2026-08-28 起唯一路径，`model_v2.py` 头部 docstring 是权威说明）：
- specials 作为额外 token 直接拼进 DINOv2-large 输入序列 `[cls; specials(K); patches(N)]`，DINO 24 层全双向算出 z_s（register token 式）。
- register 数 K（num_specials）与 patch 数 N 解耦：**K 由"最终生效采样步集"自动推导**（`K = min(max_t((⌊√t⌋+1)²−1), N)`，无"花瓶 register"）。
- 解码器 = OutputQueryDecoder（输出查询注意力 + 分块读侧 mask + 查询自注意力**块因果 tgt_mask** + 2 层 PixelHead MLP）→ 像素重建。
- 损失 = 每个采样步累加结果的平权全覆盖像素 L1，**梯度按步解耦**；K 压缩（如 K=63）是练联想的主要杠杆。
- 实验开关：`SRV2_MEMORY_OPEN=1`（读侧掩码全开、仅留因果 mask，2026-09-04 实验用，默认关）。

## 2. 快速开始

```bash
# 训练（2 GPU DDP, 全 fp32; 单卡冒烟加 --smoke --limit 32 --max_steps 3）
NUM_GPUS=2 ./run_v2_train.sh \
    --data_dir /root/autodl-tmp/construction_site \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --output_dir output/phase1_v2 --epochs 40

# 全量 test 推理（slice 参数必须与训练一致）
python infer_v2_test.py --data_dir /root/autodl-tmp/construction_site \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --final_model output/phase1_v2/final_model.pt \
    --output output/phase1_v2/infer_test.json
```

自检: `python model_v2.py`（形状 / 块掩码 / 梯度 / PixelHead / eval 同路径，应输出 `ALL CHECKS PASSED`）。

环境：torch≥2.0 / transformers / accelerate（见 `requirements.txt`）；DINOv2-large 离线权重 + `HF_HUB_OFFLINE=1`（服务器路径 `/root/autodl-tmp/models/dinov2-large`）。

## 3. 文档导航（doc/ 按日期归档，读最新在前）

- 权威目标: `doc/2026-08-28/GOAL_compression_for_nlp.md`
- 全版本机制分析（"为什么曲线全平"、下一步选项）: `doc/2026-09-03/ANALYSIS_v2_story_and_next.md`、`doc/2026-09-04/ANALYSIS_v2_three_configs.md`
- 最新实验（2026-09-04）: `doc/2026-09-04/REPORT_v2_block_slice05.md`（K=35 分块读，L1 20.55）、`doc/2026-09-04/REPORT_v2_slice05_memory_open.md`（读侧全开训练塌缩）、`doc/2026-09-04/ANALYZE_v2_slice05_exp1_leakage.md`（泄露专项探针）
- 历史版本（v1/v3/v4/v5）代码与复现入口: 见 **test** 分支（本文档 §0）
