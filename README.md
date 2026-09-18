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
- 解码器 = OutputQueryDecoder（**循环读出，唯一路径**：step1 查询 = `query_base`，step t≥2 查询 = `query_base + 上一步输出`（裸加，无投影/LayerNorm）；读侧按块切片 `A[:, lo:hi+1]`，每步只读自己那块 `z_s`，首步含 `z_cls` 及前缀；**循环 carry 被 `detach`** ⇒ 步间无梯度回流）+ 2 层 PixelHead MLP → 像素重建。注意：循环版**不新增任何参数**（没有 `rec_*` 参数）。
- 损失 = **直接预测口径**（2026-09-15 起）：每个采样步的输出 `Y_t` **各自直接**过 PixelHead 预测整图（无累加/集成），`loss = mean_t L1(PixelHead(Y_t), target)`（全轨迹深监督、各步平权）；`F_hat = Y_pix[:, -1]`（最后一步的直接预测），监控量 `recon` = 它的 L1（= loss 的最后一项）。旧的"累加结果"口径（`mean_t L1(PixelHead(Σ_{i≤t}Y_i), target)`）与 `--loss_mode` / `--loss_decouple` 两个开关**已整块删除**（传即 `TypeError`，不静默忽略；复现从 git 取回，旧细节见 `doc/2026-09-15/DESIGN_v2_recurrent.md` §2.7）。K 压缩（如 K=63）是练联想的主要杠杆。
- **循环架构的"旋钮"已全部删除（2026-09-15，文档更正）**：`--recurrent` 总开关与 `--recurrent_state/--recurrent_fuse/--recurrent_memory/--recurrent_step_embed/--recurrent_detach/--recurrent_gate_init` 子开关、以及 `rec_proj`/`rec_norm`/`rec_step_embed`/`rec_gate` 参数**都已随并行路径整块删除**，`train_v2.py` 传进去直接 `TypeError`。当前硬编码的语义 = 旧实验里的 `state=increment` + `fuse=add`（裸加，去掉 zero-init Linear 与 LayerNorm）+ `memory=block`（首步含前缀）+ 无 step embed + **`detach=True`（截断 BPTT）**。设计文档 `doc/2026-09-15/DESIGN_v2_recurrent.md`（§2.4/§2.5/§2.5.2/§2.7）记录的是**带开关**的那一版（默认 `recurrent_detach=False` = 整条循环反传 BPTT），只能当历史设计参考，不要当成当前行为。
  - **detach 的实测后果（2026-09-15 复核）**：① `∂L_t/∂Y_{t-1} = 0` ⇒ 循环只有**前向**耦合，"后期步基于当前画布做残差修正"缺梯度支撑（没有任何损失项要求上一步输出成为对下一步有用的草稿）；② `∂L_t/∂query_base = 0`（t≥1），`query_base` 只从 step0（读窗口最小那一步）的损失收梯度。要换 BPTT 口径只能改 `model_v2.py:OutputQueryDecoder.forward` 里那一行 `(self.query_base + Y).detach()`（已无开关）。
- **已删除（2026-09-15, 需复现请从 git 取回）**：并行解码路径（|T| 步一次算完 + 跨步 `tgt_mask`）与 `--query_mask_mode` 开关——循环架构是**唯一路径**。`model_v2.py` 不再有 `query_mask_mode`/`tgt_mask`（传即 `TypeError`，不静默忽略），`build_causal_query_mask` 仅作历史遗留纯函数供诊断脚本引用。
  - ⚠️ **checkpoint 兼容性更正**：循环版**不新增参数**，与并行时代**默认配置**的 state_dict 逐 key 逐形状完全相同（实测 44 keys 全等；只有旧 `recurrent=True` 那一支才多 `decoder.rec_proj/rec_norm`）。所以**旧 `final_model.pt` 用当前代码 `strict load` 不会报错**，会按新语义静默算错——这正是"静默算错"型风险，消费方不能依赖 load 报错来区分代际。需要旧产物就用 git 取回当时的 `model_v2.py`（`git log -- model_v2.py`，最后一个含并行路径的提交 = `043ef2a`），不要拿旧权重在当前代码上推理。
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

- **方案A 8760 正式实验（2026-09-18，`sum+replace` 负结果）**: `doc/2026-09-18/REPORT_gnn_schemeA_8760.md` —— bs16+ga2 / 8760 步 / seed42，K35/48/99 比同 K 锚点（bs32 ga1、`off`）差 **+13.25% / +17.10% / +21.90%**（16.02 / 15.40 / 15.70 px），**逐步落差塌缩为 +0.04%~+0.13%**（锚点 −1.72%/−7.89%/−6.75%，即末步反而略差 ⇒ 精炼消失）；**机制更正**：`sum+replace` 只替换槽位 0（`z_slots=cat([z, bank[:,:K-1]])`，`model_v2.py:752-754`），其余 K−1 仍为 register，**不是**"1 向量顶掉 K 个 token"；**缺口**：无同口径 `off` 对照、无设计文档 §10 E3 指定的 `proto(k=K)` 臂 ⇒ **不足以判定方案A 本身**，只证明该集成方式在 8760 步下劣于现有 register z_s
- **中文翻译 · 第二轮（2026-09-18，补量 13 个 repo，双卡）**: `doc/2026-09-18/REPORT_translation_zh_round2.md` —— GPU0 复用 :8100、GPU1 新起 :8101 并行，聚合 **6.85 条/s**（单卡 2.0 的 2.4×，两卡均 86% util）；覆盖 8 个可译 repo（`hayden-yuma` `scene_description` **7,415 条自由文本** + 93 个类名/标签取值），3 个因**无映射来源**判 UNKNOWN（flame2 label 0/1、hiennguyen category_id、SRuibo 缩写码）；产物 `out_zh_round2/` **98 个 `.zh.parquet` + 2 个 `class_names_zh.json`，172,474 行 / 59.1 GB**；**质量告警与修复**：`hayden-yuma` 是 TTC 强术语文本，首轮**系统性错译 39.3%**（drum→鼓(乐器) 15.1%、TTC sign→交通信号灯 6.9%、barricade→巴里卡德 4.4%），加 TTC 术语表重翻后降至 **2.41%**，并修掉术语表回显 184 条；原始数据集 5,153 文件 md5 两次复测一致（未改写）
- **中文翻译 · 第一轮（2026-09-18，Qwen3.8-27B，`_zh` 化）**: `doc/2026-09-18/REPORT_translation_zh_progress.md` —— transformers 5.16.1 直推、单卡 MODE=single 一次加载成功（52.9/95.6 GB，未装 vLLM）；服务端 `prompts` 批接口 BATCH=8 实测 **1.68 条/s**（逐条 0.39、高并发更慢）；`aswin00000` 11,339 + `chandrabhuma` 2,417 条、0 失败、**类名 71/71 全中文**，schema = 原列 + `<field>_zh`（保留原文）；产物 `/root/autodl-tmp/translate_out/out_zh`（9.9 G / 48 jsonl）；**已知局限**：自由文本约 **3–5% 语义/数字错误**未修
- **数据集筛选与补量归档（2026-09-17→18）**: `doc/2026-09-18/data/dataset_supplement/README.md` —— 按四条标准（剔除多轮对话/无标注/非图像/非中英）判定：**保留 6 个 = 8.109 GB**、剔除 6 个 = 208.455 GB（**未删除**任何原文件）；补量 13 个同域数据集 = **130.742 GB**（校验 15,717 文件 tree==local bytes==130,741,857,441 B，0 missing）⇒ 保留语料 **≈138.85 GB**；含 rc=18 根因（重试分支漏 `curl -C -` 从 0 重传 + 4 分片 `ONLY/IDX%4` 不匹配使 8 repo 无人认领）与修复下载器 `mm_add2.py`
- **方案A 图嵌入落地实验（2026-09-17，GNN + 置换不变 Readout）**: `doc/2026-09-17/DESIGN_IMPL_gnn_schemeA.md` —— 纯 PyTorch 实现 `model_gnn_schemeA.py`（kNN 图 + GIN/GCN/SAGE + sum→1 向量 z / attention→K 原型；`--gnn_mode {off,sum,proto}`、`--gnn_inject {replace,concat}`），基线 `bptt@d76f0c5`（代码在服务器 clone `feat-gnn-schemeA`=`f24b863`+`793e99a`，**未 push**）；**E2 置换不变性**：sum z 单位球漂移 **1.92e-07**、k 原型多重集 split-half Hausdorff **0**，对照 OrderSensitiveMLP**+PE 1.61e-03** vs −PE 2.59e-07（差 3–4 数量级），但**现有 register z_s 仅漂 8.9e-07 ⇒ 不能声称其显著漂移**；**K=35×2000 步全量 test**：off 26.2784 px/0.4576 → **sum 23.6203/0.4118（−10.1%）**、proto 24.0418/0.4187（−8.5%）；**诚实边界**：缺 null 对照（图的贡献未隔离）、proto 原型区分度坍塌、未跑满 8760 步、未做纯内容特征口径
- **多模态数据集盘点（2026-09-17，12 个 mm-datasets 总表）**: `doc/2026-09-17/ANALYSIS_mm_datasets_inventory.md` —— 服务器镜像 `/root/autodl-tmp/mm-datasets/` **202 GB / 12 repo / 10 OK + 2 gated**；逐项给出类型（4 检测 / 2 分类 / 2 多轮视觉指令 / 1 VQA / 1 纯文本 QA / 1 渲染包 / 1 机器人录制 / 1 BLOCKED）、单轮多轮、单图多图与规模，**含图样本约 80 万+**（LLaVA-NeXT 779,289 行中 738,601 行有图）；机器可读 JSON 与 4 份原始分组证据入库 `doc/2026-09-17/data/dataset_recon/`
- **K-sweep 汇总（2026-09-16，BPTT 口径 8 点压缩-质量曲线，单卡 bs32）**: `doc/2026-09-16/REPORT_ksweep_bptt.md` —— K=15→120 像素 L1 **15.02 → 12.51**（norm 0.2612 → 0.2175；7 个单卡点，**K=63 非单调异常**：14.06 px 反差于 K=48）；含 2 个 2 卡 DDP 锚点（K=35 / K=576）与 detach 基线，用 k35c 同 K 控制把单卡↔DDP 口径差量化为 **21.62% / 3.9030 px** ⇒ 原“单卡 K=99/K=120 反超 DDP K=576”的**反转被推翻**；**Phase-1 判定**：K≈128（K=120）最优且干净、K≈32（K=35c）可用、**K≈64（K=63）存疑须重跑**
- **逐步细化形态分析（2026-09-16，BPTT K-sweep）**: `doc/2026-09-16/ANALYSIS_ksweep_stepwise_refinement.md` —— 逐步增量分解：**有**真细化（detach 全平 → BPTT 连续正增量），但落差 **≥88% 在前 3–4 步**、增量按 ~3–4 倍几何衰减、**末 2–3 步转负（过冲）**；仅 **K≥48** 明显（K≤35 ≤1.8%），K=63 折返；⇒ K 的有效容量远小于名义 K
- 权威目标: `doc/2026-08-28/GOAL_compression_for_nlp.md`
- **★ 结果总账（2026-09-16，循环 carry BPTT）**: `doc/2026-09-16/RESULTS_v2_bptt.md` —— **先读这份**（实验矩阵 / 主结果 / 逐步曲线 / 可写进论文的四个点 / 一个被静默作废的判据 / 诚实的边界；分实验详报与原始 json 见其内部链接）。核心：单变量去掉循环 carry 的 `detach` ⇒ 全量 test 像素 L1 23.72 → **16.42**（**−30.8%**），24 步逐步曲线由"平"转"降"（19.10 → 16.38），显存/步速**逐位不变**
- **BPTT vs detach 对照实验（2026-09-16，循环 carry 反传）**: `doc/2026-09-16/REPORT_v2_bptt_vs_detach.md`（只去掉 `model_v2.py:528` 的 `.detach()` ⇒ 全量 test 像素 L1 23.72 → **16.42**、`full_norm_l1` 0.4127 → **0.2857**，**−30.8%**；24 步逐步曲线由"平"转"降"、**后步坍缩均衡被打破**（但饱和仍早，见该文 §3）；显存/步速与 detach 版**逐位相同** ⇒ 不是拿算力换的。附件 `doc/2026-09-16/model_v2_bptt.patch`。**本实验未改仓库默认行为**）
- **slice[0:5] + BPTT（2026-09-16，5 步操作点）**: `doc/2026-09-16/REPORT_v2_slice05_bptt.md`（slice[0:5] → `K=35` → 5 步 `[1,4,9,16,25]`；全量 test `full_norm_l1` **0.314064** / 像素 L1 18.05 ± 6.50，比 2026-09-10 历史基线 0.3318 低 5.3%（**注意跨代 + 跨损失口径**，只能算提示）；但逐步 L1 仅改善 **1.3%** ⇒ **后步在该操作点仍未分工**（对比 24 步全轨迹的 14.2%）。另记重要方法学结论：`DESIGN_v2_recurrent.md` §6 判据①（`step_px_scale`）在"直接预测"损失口径下**已失效**，实测各步恒 ≈1.03）
- **CPU 侧独立验证（2026-09-16，离线/无 GPU/无权重）**: `doc/2026-09-16/ANALYSIS_v2_bptt_cpu_verify.md`（在**仓库自己的解码器**上实测两条结构性事实：`out_t = t·query_base + ΣΔ`（裸加 carry，随步号 24× 线性累积）、`∂L_t/∂query_base = 0`（t≥1，`query_base` 只被读窗口最小的 step1 塑造）；把 −30.8% 拆成"与 carry 无关的整体提升 **−19.2%** × 轨迹精修 **−14.0%**"；CPU 微型复现台四臂重现"detach 平 / BPTT 降"（末步 −37.8%），并用**冻结编码器**那一刀把核心机制定位到解码器侧信用分配；含梯度扇出 9.49 vs 2.50 的操作点解释、末步读窗口退化为 1 token、`F_hat` 口径复核建议。脚本/数据同目录归档）
- 架构现状（2026-09-15，**当前代码 = 循环 + 直接预测损失**）: `doc/2026-09-15/DESIGN_v2_recurrent.md`（设计过程与带开关那一版的细节；**当前实现是其中"无开关"的子集**，差异见 §1 —— 尤其它记的默认 `recurrent_detach=False`（BPTT）**不是**当前行为）
- 解码器/可视化修复（2026-09-15）: `model_v2.py` 的 `patches_to_image`（target_pix 布局的唯一反变换 + 自检 §1b 往返断言）；`infer_v2_test.py`/`visualize_recon_pixel.py` 一律调用它
- 实验（2026-09-10，blockdiag 单卡复跑 + 后步坍缩探针）: `doc/2026-09-10/REPORT_v2_blockdiag_slice05.md`（slice[0:5] K=35 单卡 bs=32，`eval_recon` 0.3584→**0.3318**）、`doc/2026-09-10/PROBE_v2_step_collapse_blockdiag.md`（探针 `probe_step_collapse.py`：**step1~5 仍坍缩且 blockdiag 更甚**；证明该增益来自单发通路而非后步分工）
- 机制分析（2026-09-10，块级共享位置编码 `3151bab` 塌缩归因：三路独立取证 + 交叉验证 + 对抗裁决）: `doc/2026-09-10/ANALYSIS_k3_posenc_failure.md`；配套设计与实测 `doc/2026-09-10/DESIGN_query_mask_mode.md`（`query_mask_mode` 开关，**该开关已删除**，自检 `doc/2026-09-10/smoke_query_mask_mode.py`）
- 机制分析（2026-09-07，后步归零的最短因果链，见 §4）: `doc/2026-09-07/ANALYSIS_k3_why_later_steps_zero.md`、`doc/2026-09-07/DESIGN_v2_region_loss.md`（P1 设计，实现已还原见 §4.1）
- 全版本机制分析（"为什么曲线全平"、下一步选项）: `doc/2026-09-03/ANALYSIS_v2_story_and_next.md`、`doc/2026-09-04/ANALYSIS_v2_three_configs.md`
- 实验（2026-09-04）: `doc/2026-09-04/REPORT_v2_block_slice05.md`（K=35 分块读，L1 20.55）、`doc/2026-09-04/REPORT_v2_slice05_memory_open.md`（读侧全开训练塌缩）、`doc/2026-09-04/ANALYZE_v2_slice05_exp1_leakage.md`（泄露专项探针）
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
| **步间任务同构** | 每步监督在函数形式上等价（当时口径都做 `L1(累加_t, 整图)`；**现为** `L1(PixelHead(Y_t), 整图)`、仍平权、各收 1 份梯度）——"哪一步干活"对损失中性，step-1 可以一肩扛完全部 | **P1 分区域掩码损失**：step t 只监督自己对应的 patch 行区，制造 step-1 抢不走的**私有目标**；让损失本身产生 register 分工压力 |
| **键的内容可寻址性不足** | 后区键的读出内容高度相似（slice27_v2 位置 16..63 两两 cos 0.963–0.974，采样种子 cos 0.998+），解码器难以**按内容指名**某一位 | **P2 F1/E2' 编码侧注入**：special 输入拼 `Linear(patch_feat)`，让键有"各自携带逐 patch 内容"的可能（可叠正交/负余弦/使用率正则） |

- 要点：① 两者需**同时**处理——只给私有目标则键仍无可读内容，只改键则损失仍奖励后步归零；② 手段不是外部扰动/正则"推一把"，而是改损失/改输入通路，让均衡点本身失稳、让分工方向长出梯度；③ 已证伪的单侧手段：去解耦/后步加权（17.46 vs 18.77）、课程 fade-in（只推迟同一均衡）、残差目标显式化（单独无效——零仍是残差不可预测时的最优）。
- 读出上限提醒：F3/F4 固定模板读出上限 ~19（S25/S64 单步独占梯度也只有 19.15/19.58）；若要超 ~19，还要逐 patch 内容句柄（E2 类），另议。

**判据修正：「键相似度」不能当目标或观测特征（2026-09-10）**
早期把"键 cos 跌破 0.9、`z_s` within-std 从 ~0.14 回升"当作"键塌缩被打破"的观测特征——该判据**不成立**：

- 高块内相似度在**健康**模型里普遍存在：09-02 健康 K=576 模型（像素 L1 **17.46**）块内两两 cos = **0.915**、跨块 0.905；
- slice05 的**健康逐位置基线**本身 `z_s` within-std 已只有 **0.053**，为 slice27_v2（0.143）与 P（0.171–0.178）的 **1/3**；
- 且解码器 `pos_embed` 是**加性**偏移，块内键 `z_s[j] + pos_p` 因内容不同而彼此不同——"键表示趋同"在措辞上也不成立。

⇒ 键相似度**既不必要也不充分**，不能单独作为判据或优化目标。仍然可用的判据（全部用**匹配步**，且历史对比一律用 `eval_recon`，勿用 `eval_loss`——口径已随 region_loss 变化）：
① 后步 `\|W·Y_t\| ≫ 0.015`；② 渐进曲线出真阶梯（覆盖区单调降；当时口径为累加曲线，现为每步直接预测曲线）且 `eval_recon` 不劣化；③ 各步**隔离** L1 在自己区域上相近（不再 39 vs 61）。
（早期框架同时声称"交换任意键系统不变"——该对称性在当前逐位置 `pos_embed` 下**不存在**：置换 register 会改变其键与查询种子。仅在 `3151bab` 的块级共享变体下，解码器读出路径才对**块内非种子成员**的置换精确不变；这条精确不变性同时意味着"塌缩一旦发生就没有免费的恢复梯度"，正是 `ANALYSIS_k3_posenc_failure.md` §2.3 实测的 `0.00e+00`。详见该文档 §2.2/§2.4/§2.5。）

**状态：留档待决。** 2026-09-07 用户暂无时间，解决路径后续再想；本条目只记录框架与判据，不作实施承诺。
P1 曾实现并通过本地自检 + 服务器数值冒烟（`doc/2026-09-07/DESIGN_v2_region_loss.md`，`region_loss=True` 默认），但**代码已还原**（HEAD `a8eeabc` "原提交不符合实际需要，暂时还原"）。**口径现状（2026-09-15 更正）**：当前 `model_v2.py` 的损失既不是 P1 分区损失，也不是早期的 cumsum 累加口径，而是**每步直接预测**（`mean_t L1(PixelHead(Y_t), target)`，见 §1）；共同点只有"每步整图、各步平权"。若走渐进路线，按该 doc 重新落地 P1 即可（~20 行核心改动）。

**2026-09-10 补充证据（掩码侧已排除）**：`query_mask_mode` 默认翻转为 `blockdiag` 后按 slice[0:5] K=35 单卡 bs=32 复跑，`eval_recon` 0.3584→**0.3318**（`doc/2026-09-10/REPORT_v2_blockdiag_slice05.md`）；但同尺探针（`doc/2026-09-10/PROBE_v2_step_collapse_blockdiag.md`）实测 `step_px_scale` = `[1.0273, 0.0395, 0.0349, 0.0342, 0.0335]`（causal 对照 `[1.0072, 0.0630, 0.0584, 0.0572, 0.0563]`）——
**step1~5 仍然坍缩，且 blockdiag 下后步相对量级从 5.6–6.3% 降到 3.3–3.8%**，区域×步矩阵两臂都是五行逐位相同。⇒ 上述增益来自**单发通路收敛更好**，不是后步分工被激活；**掩码开关不是本条的杠杆**，与 `ANALYSIS_k3` §5 预判一致。附带一条判据层实证：blockdiag 的逐块 register cos 从 `[0.619, 0.854, 0.990, 0.998, 0.999]` 变"健康"到 `[0.734, 0.876, 0.918, 0.916, 0.926]`、`within-std` 0.054→0.095，而后步输出反而**更接近零**——实测支持本节"键相似度既不必要也不充分"的判据修正。

**2026-09-15 循环架构成为唯一路径（并行路径已删除）**：旧架构在 `blockdiag` 下"步 t 的查询行看不到其它步、`memory_mask` 又只给它自己那块 `z_s`"⇒ 每步的可用信息只有一块键 + 静态 `query_base`，用一块键重建整图不成立 ⇒ 最优解退化为"交给 step-1、自己输出≈0"（§1 诊断的结构性缺口）。解法 = 把解码器改为**顺序循环**：step1 查询 = `query_base`（去掉 `A_t`），step t≥2 查询 = `query_base + fuse(上一步的输出)`；历史输出经循环显式携带 ⇒ 后步可基于"当前画到哪"做残差修正。**掩码侧**：跨步 `tgt_mask` 删除（其职责被循环状态 `h` 取代，单步内 N 行全双向自注意力自动等价 `blockdiag` 对角块）；读侧 `memory_mask` 保留、升为三档 `--recurrent_memory`（`block`/`prefix`/`open`，**open 需配 `--recurrent_step_embed`**——实测 open + zero-init + 无步信号时各步输出逐位相同）。代价 = 时间（步间顺序依赖，无法并行算完 T 步）。**并行路径与 `--recurrent`/`--query_mask_mode` 开关随后整块删除**（循环为唯一路径，`git log -- model_v2.py` 可找回），细节/自检/判据见 **`doc/2026-09-15/DESIGN_v2_recurrent.md`**（§2.5 掩码怎么变、§2.5.2 open 的步退化）。**损失侧**：该轮训练损失为**累加口径**（逐步深监督，`F_hat` 的 L1 只是监控量 `recon`）；该口径连同 `--loss_decouple` / `--loss_mode` 开关已于 2026-09-15 **整块删除**——现行口径 = 每步 `Y_t` 各自直接过 PixelHead 预测整图（`mean_t L1(PixelHead(Y_t), target)`），`F_hat = Y_pix[:, -1]`；旧口径细节见该 doc §2.7。
**2026-09-15 复核更正（以本节 §1 为准）**：最终落地的循环是**无开关**版——`recurrent_*` 全删、carry 硬编码 `detach`，所以上面"历史输出经循环显式携带 ⇒ 后步可基于当前画布做残差修正"只有**前向**成立：实测 `∂L_t/∂Y_{t-1}=0`（无 BPTT）、`∂L_t/∂query_base=0`（t≥1，`query_base` 只由 step0 塑造）。设计 doc 里默认的整条循环反传（`recurrent_detach=False`）与三档读窗口（`prefix`/`open`）**都不是当前代码的行为**，只是历史设计记录。

**战略上下文**：渐进阶梯**不是** GOAL 验收项（`doc/2026-08-28/GOAL_compression_for_nlp.md`），Phase 2 一次性消费全部 K token；v4 单发（8.26）已是仓库最佳。除非"token 增量性"叙事本身成为目标，本条目可长期冻结，不挡主路线。

> 相关文档：`doc/2026-09-07/ANALYSIS_k3_why_later_steps_zero.md`（主分析）、`doc/2026-09-07/ANALYSIS_k3_slice_infodiff.md` / `doc/2026-09-07/REPORT_v2_E1_probe.md`（信息侧证据）、`doc/2026-09-07/DESIGN_v2_region_loss.md`（P1 设计与实现记录）、`doc/2026-09-15/DESIGN_v2_recurrent.md`（循环架构）
