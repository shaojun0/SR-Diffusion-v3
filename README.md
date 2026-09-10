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
- 解码器 = OutputQueryDecoder（输出查询注意力 + 分块读侧 mask + 查询自注意力**块对角 tgt_mask**（默认）+ 2 层 PixelHead MLP）→ 像素重建。
- 损失 = 每个采样步累加结果的平权全覆盖像素 L1，**梯度按步解耦**；K 压缩（如 K=63）是练联想的主要杠杆。
- 实验开关：`query_mask_mode`（`train_v2.py --query_mask_mode`，默认 **`blockdiag`**）——查询自注意力语义。`blockdiag` = 块对角，步间自注意力完全隔离（`memory_mask` 的"每步只见自己的块"在整条前向路径上字面成立，跨步梯度回流切断）；`causal` = 历史行为（块下三角，3151bab 之前全部产物的口径，**复现历史结果须显式传**）。详见 `doc/2026-09-10/DESIGN_query_mask_mode.md`。
  - 本项**不改任何权重形状** ⇒ 旧 checkpoint 双向可载，但模式错了 `strict load` 不报错、只会静默算错；消费方（`infer_v2_test.py` / `visualize_recon_pixel.py`）一律以 `model_info.json` 记录值为准，无该字段的旧产物 fallback 到 `causal`。
  - 早期文档提到的 `SRV2_MEMORY_OPEN`（读侧掩码总开关）**已不在代码里**（`0a1ee45` 回滚时移除），勿再使用。

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
- 最新实验（2026-09-10，新默认 blockdiag 单卡复跑 + 后步坍缩探针）: `doc/2026-09-10/REPORT_v2_blockdiag_slice05.md`（slice[0:5] K=35 单卡 bs=32，`eval_recon` 0.3584→**0.3318**）、`doc/2026-09-10/PROBE_v2_step_collapse_blockdiag.md`（探针 `probe_step_collapse.py`：**step1~5 仍坍缩且 blockdiag 更甚**；证明该增益来自单发通路而非后步分工）
- 最新机制分析（2026-09-10，块级共享位置编码 `3151bab` 塌缩归因：三路独立取证 + 交叉验证 + 对抗裁决）: `doc/2026-09-10/ANALYSIS_k3_posenc_failure.md`；配套设计与实测 `doc/2026-09-10/DESIGN_query_mask_mode.md`（`query_mask_mode` 开关，自检 `doc/2026-09-10/smoke_query_mask_mode.py`）
- 机制分析（2026-09-07，后步归零的最短因果链，见 §4）: `doc/2026-09-07/ANALYSIS_k3_why_later_steps_zero.md`、`doc/2026-09-07/DESIGN_v2_region_loss.md`（P1 设计，实现已还原见 §4.1）
- 全版本机制分析（"为什么曲线全平"、下一步选项）: `doc/2026-09-03/ANALYSIS_v2_story_and_next.md`、`doc/2026-09-04/ANALYSIS_v2_three_configs.md`
- 最新实验（2026-09-04）: `doc/2026-09-04/REPORT_v2_block_slice05.md`（K=35 分块读，L1 20.55）、`doc/2026-09-04/REPORT_v2_slice05_memory_open.md`（读侧全开训练塌缩）、`doc/2026-09-04/ANALYZE_v2_slice05_exp1_leakage.md`（泄露专项探针）
- 历史版本（v1/v3/v4/v5）代码与复现入口: 见 **test** 分支（本文档 §0）

---

## 4. 阶段性目标 / 未决问题（2026-09-10 更新）

### 4.1 待决：让"后步真干活"（方案未定，留档待后续设计）

**现象与机制裁决**（详见 `doc/2026-09-07/ANALYSIS_k3_why_later_steps_zero.md`）：
v2 时序解码 5 步渐进曲线永远全平、后几步输出量级≈0、整图由 step-1 一肩扛。
最短因果链 = 旧损失把每个采样步的累加结果都监督成整图 + 按步解耦 ⇒ step-1 的子任务=完整重建，
后步的子任务=预测 step-1 没做出来的残差 ⇒ 残差从后步可读的键里不可预测 ⇒ L1 最优预测=零
（条件中位数）⇒ 后步输出≈0 ⇒ 后区 register 收不到有效读出梯度 ⇒ 键的读出内容高度相似 ⇒
残差更不可预测——**"零增量自锁"**。这是自洽均衡，不是某一层单独的病（梯度确实会传回可训的
DINO，但读出上限 ~19 使"键分化"的收益≈0，键收到的主要是草稿纸式的趋同梯度，而非"携带可读
差异内容"的压力）。

**两条可干预的耦合**（2026-09-10 复核后的表述；早期报告用"双对称简并解 / 破缺对称性"来描述它，
该框架**已撤下**，理由见本小节的判据修正）：

| 耦合 | 表现 | 干预手段（候选） |
|---|---|---|
| **步间任务同构** | 每步监督在函数形式上等价（都做 `L1(累加_t, 整图)`、平权、各收 1 份梯度）——"哪一步干活"对损失中性，step-1 可以一肩扛完全部 | **P1 分区域掩码损失**：step t 只监督自己对应的 patch 行区，制造 step-1 抢不走的**私有目标**；让损失本身产生 register 分工压力 |
| **键的内容可寻址性不足** | 后区键的读出内容高度相似（slice27_v2 位置 16..63 两两 cos 0.963–0.974，采样种子 cos 0.998+），解码器难以**按内容指名**某一位 | **P2 F1/E2' 编码侧注入**：special 输入拼 `Linear(patch_feat)`，让键有"各自携带逐 patch 内容"的可能（可叠正交/负余弦/使用率正则） |

- 要点：① 两者需**同时**处理——只给私有目标则键仍无可读内容，只改键则损失仍奖励后步归零；② 手段不是外部扰动/正则"推一把"，而是改损失/改输入通路，让均衡点本身失稳、让分工方向长出梯度；③ 已证伪的单侧手段：去解耦/后步加权（17.46 vs 18.77）、课程 fade-in（只推迟同一均衡）、残差目标显式化（单独无效——零仍是残差不可预测时的最优）。
- 读出上限提醒：F3/F4 固定模板读出上限 ~19（S25/S64 单步独占梯度也只有 19.15/19.58）；若要超 ~19，还要逐 patch 内容句柄（E2 类），另议。

**判据修正：「键相似度」不能当目标或观测特征（2026-09-10）**
早期把"键 cos 跌破 0.9、`z_s` within-std 从 ~0.14 回升"当作"键塌缩被打破"的观测特征——该判据**不成立**：

- 高块内相似度在**健康**模型里普遍存在：09-02 健康 K=576 模型（像素 L1 **17.46**）块内两两 cos = **0.915**、跨块 0.905；
- slice05 的**健康逐位置基线**本身 `z_s` within-std 已只有 **0.053**，为 slice27_v2（0.143）与 P（0.171–0.178）的 **1/3**；
- 且解码器 `pos_embed` 是**加性**偏移，块内键 `z_s[j] + pos_p` 因内容不同而彼此不同——"键表示趋同"在措辞上也不成立。

⇒ 键相似度**既不必要也不充分**，不能单独作为判据或优化目标。仍然可用的判据（全部用**匹配步**，且历史对比一律用 `eval_recon`，勿用 `eval_loss`——口径已随 region_loss 变化）：
① 后步 `\|W·Y_t\| ≫ 0.015`；② 累加曲线出真阶梯（覆盖区单调降）且 `eval_recon` 不劣化；③ 各步**隔离** L1 在自己区域上相近（不再 39 vs 61）。
（早期框架同时声称"交换任意键系统不变"——该对称性在当前逐位置 `pos_embed` 下**不存在**：置换 register 会改变其键与查询种子。仅在 `3151bab` 的块级共享变体下，解码器读出路径才对**块内非种子成员**的置换精确不变；这条精确不变性同时意味着"塌缩一旦发生就没有免费的恢复梯度"，正是 `ANALYSIS_k3_posenc_failure.md` §2.3 实测的 `0.00e+00`。详见该文档 §2.2/§2.4/§2.5。）

**状态：留档待决。** 2026-09-07 用户暂无时间，解决路径后续再想；本条目只记录框架与判据，不作实施承诺。
P1 曾实现并通过本地自检 + 服务器数值冒烟（`doc/2026-09-07/DESIGN_v2_region_loss.md`，`region_loss=True` 默认），但**代码已还原**（HEAD `a8eeabc` "原提交不符合实际需要，暂时还原"）——当前 `model_v2.py` 仍是旧"每步整图"平权损失；若走渐进路线，按该 doc 重新落地即可（~20 行核心改动）。

**2026-09-10 补充证据（掩码侧已排除）**：`query_mask_mode` 默认翻转为 `blockdiag` 后按 slice[0:5] K=35 单卡 bs=32 复跑，`eval_recon` 0.3584→**0.3318**（`doc/2026-09-10/REPORT_v2_blockdiag_slice05.md`）；但同尺探针（`doc/2026-09-10/PROBE_v2_step_collapse_blockdiag.md`）实测 `step_px_scale` = `[1.0273, 0.0395, 0.0349, 0.0342, 0.0335]`（causal 对照 `[1.0072, 0.0630, 0.0584, 0.0572, 0.0563]`）——
**step1~5 仍然坍缩，且 blockdiag 下后步相对量级从 5.6–6.3% 降到 3.3–3.8%**，区域×步矩阵两臂都是五行逐位相同。⇒ 上述增益来自**单发通路收敛更好**，不是后步分工被激活；**掩码开关不是本条的杠杆**，与 `ANALYSIS_k3` §5 预判一致。附带一条判据层实证：blockdiag 的逐块 register cos 从 `[0.619, 0.854, 0.990, 0.998, 0.999]` 变"健康"到 `[0.734, 0.876, 0.918, 0.916, 0.926]`、`within-std` 0.054→0.095，而后步输出反而**更接近零**——实测支持本节"键相似度既不必要也不充分"的判据修正。

**战略上下文**：渐进阶梯**不是** GOAL 验收项（`doc/2026-08-28/GOAL_compression_for_nlp.md`），Phase 2 一次性消费全部 K token；v4 单发（8.26）已是仓库最佳。除非"token 增量性"叙事本身成为目标，本条目可长期冻结，不挡主路线。

> 相关文档：`doc/2026-09-07/ANALYSIS_k3_why_later_steps_zero.md`（主分析）、`doc/2026-09-07/ANALYSIS_k3_slice_infodiff.md` / `doc/2026-09-07/REPORT_v2_E1_probe.md`（信息侧证据）、`doc/2026-09-07/DESIGN_v2_region_loss.md`（P1 设计与实现记录）
