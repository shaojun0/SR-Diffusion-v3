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
- 解码器 = OutputQueryDecoder（**循环读出，唯一路径**：step1 查询 = `query_base`，step t≥2 查询 = `query_base + 上一步输出`（裸加，无投影/LayerNorm）；读侧按块切片 `A[:, lo:hi+1]`，每步只读自己那块 `z_s`，**循环 carry 走 BPTT**（**唯一路径**，2026-09-23 起 `carry_detach` 开关已整块删除；历史 detach 口径实测曲线全平，见 `doc/2026-09-22/REPORT_batch0_rd_earlystop.md` §2b）+ 2 层 PixelHead MLP → 像素重建。注意：循环版**不新增任何参数**（没有 `rec_*` 参数）。
- **读窗口 = 平方块（唯一口径）**：步 t 读 `z_s[k²:(k+1)²−1]`（`k=⌊√t⌋`；首步 `lo=0` 另含 `z_cls`），`z_s` 全部来自 DINOv2 **末层**。⚠️ 2026-09-22 的 **`--layer_tap` 逐层 tap（特征金字塔）已于 2026-09-23 整块从 main 移除**（用户口径：main 只保留主线目标，实验性分支不进主线）——代码/脚本已删，历史设计与实测（tap 画质输 0.80 dB、但 PSNR 跨度 0.30→1.69 dB）见 `doc/2026-09-22/DESIGN_layer_tap_pyramid.md` / `REPORT_layer_tap_pyramid_224.md`（只作存档）。
- **A 相 warm start（2026-09-23 新增 `--warm_steps`，默认 0=关）**：前 N 个优化步把 `decoder.steps` 设成 `[K]`（**只跑 1 步、一次读完全部 `z_s` + `z_cls`**），到点由 `WarmStartSwitchCallback` 自动切回真采样步集跑多步循环。动机（实测）：多步循环的早期步只读到 `2k+1` 个 token，其条件最优解接近"预测均值"；从零直接训多步会停在该平凡解（曲线平）。224² 实测 A 相要 **1500~2000 步**才逃逸，`--warm_steps 1000→3000` 使 16 步曲线由平变单调降（最优 PSNR 14.26→**18.69**, +4.43 dB）⇒ `doc/2026-09-23/REPORT_224_A3000_verdict.md`。
- 损失 = **直接预测口径**（2026-09-15 起）：每个采样步的输出 `Y_t` **各自直接**过 PixelHead 预测整图（无累加/集成），`loss = mean_t L1(PixelHead(Y_t), target)`（全轨迹深监督、各步平权）；`F_hat = Y_pix[:, -1]`（最后一步的直接预测），监控量 `recon` = 它的 L1（= loss 的最后一项）。旧的"累加结果"口径（`mean_t L1(PixelHead(Σ_{i≤t}Y_i), target)`）与 `--loss_mode` / `--loss_decouple` 两个开关**已整块删除**（传即 `TypeError`，不静默忽略；复现从 git 取回，旧细节见 `doc/2026-09-15/DESIGN_v2_recurrent.md` §2.7）。K 压缩（如 K=63）是练联想的主要杠杆。
- **循环架构的"旋钮"已全部删除（2026-09-15，文档更正）**：`--recurrent` 总开关与 `--recurrent_state/--recurrent_fuse/--recurrent_memory/--recurrent_step_embed/--recurrent_detach/--recurrent_gate_init` 子开关、以及 `rec_proj`/`rec_norm`/`rec_step_embed`/`rec_gate` 参数**都已随并行路径整块删除**，`train_v2.py` 传进去直接 `TypeError`。当前硬编码的语义 = 旧实验里的 `state=increment` + `fuse=add`（裸加，去掉 zero-init Linear 与 LayerNorm）+ `memory=block`（首步含前缀）+ 无 step embed + **carry = BPTT**（**唯一路径**：2026-09-22 起为默认，2026-09-23 起 `carry_detach` 开关整块删除）。设计文档 `doc/2026-09-15/DESIGN_v2_recurrent.md`（§2.4/§2.5/§2.5.2/§2.7）记录的是**带开关**的那一版（默认 `recurrent_detach=False` = 整条循环反传 BPTT），只能当历史设计参考，不要当成当前行为。
  - **detach 的实测后果（2026-09-15 复核；⚠️ 只适用于 detach 口径，当前默认是 BPTT）**：① `∂L_t/∂Y_{t-1} = 0` ⇒ 循环只有**前向**耦合，"后期步基于当前画布做残差修正"缺梯度支撑（没有任何损失项要求上一步输出成为对下一步有用的草稿）；② `∂L_t/∂query_base = 0`（t≥1），`query_base` 只从 step0（读窗口最小那一步）的损失收梯度。（该 detach 口径已于 2026-09-23 随开关一起删除，当前只有 BPTT。）**BPTT 的实测收益**：全量 test 像素 L1 23.72 → **16.42**（−30.8%，`doc/2026-09-16/REPORT_v2_bptt_vs_detach.md`）；224×126 上 detach 的 RD 曲线是**一条平线**而 BPTT 有阶梯（`doc/2026-09-22/REPORT_batch0_rd_earlystop.md` §2b）。
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

- **★★ 修正配方全分辨率重跑（A 相 3000 + B 相 7000 = 总 1 万步）+「逃逸步数 vs N」实测 · 实测（2026-09-23）**: `doc/2026-09-23/REPORT_rerun_A3000_10k.md` —— 用户口径「336 开始重新补消融实验，step 加到 1 万，warm_steps=3000」；A800 / 5 臂 / **1 h 59 m**。**① 全线翻盘**：把 224² 判决配方推广到 28→336，5/5 分辨率都由平线变单调降，最好 PSNR 相对旧放大预算臂 **+3.39 / +5.38 / +6.82 / +5.33 / +4.26 dB**（28/56/112/224/336）；112² 与 JPEG 的差从 −8.02 收到 **−1.20 dB**，28²/56² 反超 JPEG **+5.2 dB**。**② ★ A 相逃逸步数随 N 单调增长（实测曲线）**（首次超过 per-patch DC 基线的 checkpoint）：N=4 → **≤500**、N=16 → **500–1000**、N=64 → **500–1000**、N=256 → **1500–2000**、N=576 → **2000–2500** ⇒ 旧配方恒 1000 步对 224²/336² **根本不够**（把交接 §2c 从单点推断升级为 N 上实测，顺带完成交接 §7②）。**③ B 相延长有实质收益**：224² B 相 5000→7000（8000→10000 总步）best PSNR 18.69→**19.59（+0.90 dB）**。**④ ⚠️ 336² 是唯一没跨过平凡解线的点**：best PSNR 13.36→**17.62（+4.26 dB）**、best L1 38.22→21.76、24 步里 21 步严格降，但 **17.62 < DC 基线 17.83（−0.21 dB）**，且 B6000 17.51 → B6500 17.59 → 末 17.62 **仍在涨（≈+0.08 dB/500 步）未收敛**（外推还需 ~1500–2500 步 [推断]）。**⑤ "切块读"的代价随 \|T\| 单调增大**：A 相（单步全读）上限 − B 相最好 = +0.55（28² 末步退化 artifact）/ −0.99 / −1.55 / −1.80 / **−1.88 dB**（56/112/224/336），且 A→B 冷切换瞬时破坏随 N 放大（56² L1 8.09→28.62、336² 17.07→44.37，但都能恢复）⇒ 交接 §7⑤「步间损失加权」拿到定量动机。**⑥ 工具**：新增 `tools/{run_rerun_A3000_10k.sh,per_patch_baseline.py,compare_rerun_A3000_10k.py}`（基线复现旧报告 15.32/16.04/16.69/17.35/17.83 逐位一致）；产物 `doc/2026-09-23/rerun_A3000_10k/`。⚠️ 边界：单 seed / patch 读出（非 register）/ β=1 名义 bpp / **336² 未收敛（17.62 是 7000 步下界）**；新臂写 `out_A3000_10k/`、**未覆盖** `out_224_A3000/`
- **★★ main 代码清理：移除 layer_tap 特征金字塔 + carry_detach 开关，新增 `--warm_steps` · 2026-09-23**: 用户口径「main 是我们的主目标，其他方法会干扰我们的主目标」——`model_v2.py` 删 `layer_tap_groups()` / `OutputQueryDecoder.layer_tap` / `SRPhase1V2.layer_tap` 与 tap 分支 / 自检 §2d–§2f（**−297/+67 行**）；`carry_detach` 开关整块删除 ⇒ **carry = BPTT 唯一路径**（`Y = query_base + Y`，不 detach，无开关）；`train_v2.py` 删 `--layer_tap`、**新增 `--warm_steps`**（`WarmStartSwitchCallback`：前 N 个优化步 `decoder.steps=[K]` 单步全读，到点自动切回真步集）；`infer_v2_test.py` 删 tap 兜底（旧产物带 `layer_tap` 字段会打一条"已移除"警告，bpp 偏移恒为 1）；删 `tools/run_224_d4_bptt_tap.sh`、`tools/eval_224_d4_bptt_tap.sh`、`tools/compare_layer_tap.py`；`tools/run_sweep.sh` 删 detach 对照臂、`tools/sweep_res_train.py` 删 `--detach`（传即 argparse 报错，不静默忽略）。⚠️ **权重形状完全不变**（layer_tap 与 detach 都不新增参数）⇒ 旧 checkpoint 仍能 strict load，但 2026-09-22 及以前的 tap/detach 产物**必须用当时 commit 的代码推理**，否则会静默算错；`doc/2026-09-22/{DESIGN,REPORT}_layer_tap_pyramid*.md` 已加归档横幅、只作历史
- **★★ 分辨率扫描「只有 56² 达标」根因复核 + 224² 判决实验 · 实测（2026-09-23）**: `doc/2026-09-23/ANALYSIS_res_sweep_rootcause.md` + `doc/2026-09-23/REPORT_224_A3000_verdict.md` —— **★ 推翻 [`doc/2026-09-22/REPORT_resolution_sweep.md`](doc/2026-09-22/REPORT_resolution_sweep.md) 的主结论「多步递归训不起来」。** ① **像素数不是自变量**：扫描的 336² 与主线 448×252 **像素数（112,896）、N（576）、\|T\|（24）完全相同**，却一个平线（L1 39.61→38.71）一个达标（19.10→16.38）⇒ 变量在配方不在分辨率；② 高分辨率臂其实是**停在平凡解（预测均值）**：224²/336² 最好 PSNR 13.85/13.61 dB **低于** per-patch 均值基线 17.35/17.83 dB；③ **近因 = 两相配方的预算与切换**：`--warm_steps`（A 相＝单步全读；**不是学习率 warmup**，LR warmup 是 `--warmup`）在所有分辨率恒 1000 步，而 224² 的 A 相实测要 **1500→2000 步**才逃出平凡解（`224_diag` hist：it1000 L1 36.08/14.19 dB → it2000 18.08/19.34 dB）⇒ 旧 B 相是从「还只会涂平均色」的模型起步的；④ **判决实验**（224²，唯一变量 A 相 1000→3000、B 相 5000、A800/fp32/893s）：16 步曲线从平线变**单调降**，最好 L1 **34.02→19.38（−43.0%）**、PSNR **14.26→18.69（+4.43 dB）**、**重新超过** per-patch 均值基线 **+1.34 dB**；⑤ **同一个 B 任务、同样步数、只有起点不同**：起点糊（A=1000）B 相 5000 步只降 **3.3 L1**，起点锐（A=3000）降 **19.7 L1**（6×）⇒ A 相梯度能出平凡解、B 相出不去（B 相是「切块**读**、整图**预测**」，早期步被平权监督要求画锐图，梯度指向糊）；⑥ CPU 微型复现台（`doc/2026-09-23/toy_zmode_ab.py`，N=K=64/\|T\|=8，只换 z_s 读出段）：A→B 冷切换瞬时破坏 register 1.25× vs **patch 13.8×**，900 步后两者都能降 ⇒ patch 读出不是死因、但放大切换破坏；⑦ 仍需修：**carry 未归一化**（实测 24 步内 ‖Y‖ 长 **43×**）、**K=N 末步窗口退化为 1 token**、bpp 用名义 `t+1` 而非实际 `(k+1)²`（56² 最好步 1.224 vs **1.959**，对 JPEG 的 Δ 应再差 1.5/0.8/0.5 dB）；⑧ 下一步（**未做**）：28/56/112/336 补 A 相逃逸曲线、`--warm_steps 0`（去相位）臂、把 step-1 的固定模板查询/损失按窗口加权；复现命令与边界见报告 §5/§6
- **★ 新服务器验收：AutoDL bjb1 / 1× A800-80GB · 实测（2026-09-23）**: `doc/2026-09-23/SERVER_bjb1_a800_test.md` —— `ssh -p 54066 root@connect.bjb1.seetacloud.com`；1×A800-SXM4-80GB（sm_80）/144 核/1007 GB/`/root/autodl-tmp` 50 GB；已装 transformers **5.17.0** + accelerate 1.15.0 + bitsandbytes 0.50.2（AdamW8bit 实测可用）；网络：pip 用清华源（20 MB/s，aliyun 默认源仅 0.5 MB/s）、GitHub 需 `source /etc/network_turbo`（**与 pip 同壳会让 pip 失败**）、HF 用 `hf-mirror.com`；公共盘 `/autodl-pub/data` 含 DIV2K（zip 需解压）；`python model_v2.py` **ALL CHECKS PASSED**、224² 端到端 GPU 冒烟 9 s；吞吐（DINOv2-small/d2/patch 读出/bs16）：28² 31.7、56² 23.7、112² 15.6、224² 3.29、336² 1.08 it/s（fp16 3.2×）；**只有 1 张卡** ⇒ `run_v2_train.sh` 要 `NUM_GPUS=1`
- **★★ 896×504 高分辨率臂 + 448×252 原始配方对照臂（都用 fp16 + 8-bit AdamW）· 实测（2026-09-22 夜 → 09-23 05:03）**: `doc/2026-09-22/REPORT_res896_448_fp16_8bit.md` —— 用户口径：「基于 BPTT 跑 layer=4 / slice=0~12 / **(448×2)×(252×2)**」+「试试 fp16，优化器用 8 位的 adamw」；随后「再跑一下**原始的 BPTT + fp16 + 8bit** 作为对照（448×252）」。**① 896 臂**（N=2304 / `slice[0:12]` ⇒ 12 步 / **K=168**（K/N=7.3%）/ d4 / 5h39m38s）：test 3,004 像素 L1 **15.85 ± 5.76**、最优 PSNR **19.10** @t=121、MS-SSIM 0.5864、**曲线真的在降**（首→末 −2.82 px / −19.1%，单调前缀 11/12，oracle 上限 +0.111 dB）；**但整条曲线只覆盖 0.005–0.329 bpp**，同 bpp 下**落后 WebP 最多 −11.25 dB**、JPEG −8.67 dB ⇒ 12 步的固定 token 预算（≤145 token）对 45 万像素**量级不够**。**② 448 臂**（用户选 B 读法 = 原始配方 d2/全 24 步/K=576，只换精度优化器，1h24m52s）**——本轮最重要的结果**：与历史 fp32 BPTT **同配方同 seed**（`doc/2026-09-16/data/bptt24_bptt_infer_test.json`，16.422 px、跨度 **−2.674 px / −14.0%**）相比，fp16+8bit 得到 **18.04 px（+1.62 px / +9.9%）但跨度只有 −0.071 px（−0.4%）**，**与历史 `detach`（关掉 BPTT）对照臂的 +0.072 px 在数值上已无法区分** ⇒ **把 BPTT 的精修收益整块抹掉了**（训练快 3.26×；但 fp16 与 8-bit 一起换的，**不能单独归因**）。机制诊断（`tools/diag_step_spread.py`）证明 448 是「**轨迹在动但不降误差**」（首→末挪 1.62 px、改善仅 0.080 px）而**不是**「输出不动」；896 是「在动且降误差」（−19.1%）。**诚实边界**：精度/优化器同时变、单 seed、bpp 是三套不同分母的 β=1 估计 ⇒ 224/448/896 **目前不是一条可排序的分辨率曲线**；C2b 仍不成立（天花板 ≤0.111 dB）。**下一步最省的一步**：448 上补 `fp16+torch-fused` 与 `fp32+8bit` 两臂分离变量。工具 `tools/run_896_d4_bptt_slice012.sh`、`tools/run_448_d2_bptt_fp16_8bit.sh`、`tools/plot_l1_curves.py`、`tools/diag_step_spread.py`；设计 `doc/2026-09-22/DESIGN_res896_d4_bptt_slice012.md`；图 `data/curves_448_precision.png`、`data/curves_3arms_res.png`

- **★ 分辨率扫描 28→336（小模型探针，逐 step L1 变化率 + JPEG 同图对比）· 实测（2026-09-22）**: `doc/2026-09-22/REPORT_resolution_sweep.md` —— 用户口径「图片的大小从 32×32 到更大，看看其各个 step 的 L1 变化率，并跟 jpeg 比比」。小模型（**DINOv2-small 384** / d2 / BPTT / DIV2K 800-100 / 直接 resize 到 S×S，**AutoDL 4090**）在 **28/56/112/224/336** 各训一个，逐 step 出 nested RD，并在同一批图、同一 PSNR 口径、同一 bpp 分母下跑 JPEG/WebP。**★ 主结论：等预算下"变化率随分辨率雪崩"是预算混入，真瓶颈是「多步递归训不起来」** —— 224px 的**单步+全读 21.59 dB**（平凡 per-patch 均值 17.35 dB）而**同分辨率 16 步循环只有 14.26 dB** ⇒ 编码器/解码器/像素头有能力，是 16 步循环训不起来。**其余读数**：① 56px 上 BPTT 首步 23.32 → 最好 **16.13 px（改善 30.8%，20.80 dB / vs JPEG +0.48）**，detach **首步反而更好（21.52）但轨迹不涨**（16.6%，19.94 dB）⇒ 自然图像上也复现「后步修正前步」；② **K=N 时末步窗口恒退化为 1 个 token**（块宽 3,5,7,…）⇒ 28px 的"第 2 步"改善恰好 **0.0%**，**"|T| 步"严重高估有效轨迹**（与 `doc/2026-09-16/ANALYSIS_v2_bptt_cpu_verify.md` §2.4 同一现象）；③ **极低码率独占区在低分辨率下最宽**：28px 上 JPEG 最低只能到 ~6.5 bpp，我们覆盖 **0.98–2.45 bpp**；④ 112px 极吃预算（2500→3800 步：15.90→**17.90 dB**）⇒ 等预算表的跨分辨率绝对高低不可比。**⚠️ 口径边界**：**z_s 用 patch token 而非 register**（register 版 56px/900 步 loss 0.50 vs patch 版 0.256，路由从零学不动）、无 letterbox 画布、β=1 估计 bpp（无熵编码）、112px 起欠训、单 seed ⇒ **不是仓库主线的复现**，只作「分辨率 / 步数」机制探针。**下一步最该做的**：修多步递归在 |T| 大时训不起来（步间损失加权 / 课程式增长 |T| / 加深解码器），并把分辨率扫描改成**等保真度**再比 RD。脚本 `tools/sweep_res_{train,jpeg,prep,report}.py` + `tools/run_sweep{,2}.sh`；产物 `doc/2026-09-22/res_sweep/`（原始 json / 图 / 自动汇总 md）
- **★ 采样步集切片 `[0:12]` → `[1:12]`（去掉 t=1）· 实测（2026-09-22 晚）**: `doc/2026-09-22/REPORT_slice1_224.md` —— 用户口径「跑 batch0 §10 那个实验，只把 step 从 0~12 改成 1~12」。**零代码改动**（切片是既有 CLI `--slice_start/--slice_end`）；与基线 `out_224_d4_bptt` 单变量对照（唯一差别 = 切片）：11 步 `[4,9,…,144]`、**K 仍是 144**、首步窗口由 `A[0:3]` 并成 `A[0:8]`、`A[0..144]` 仍被完整覆盖（无花瓶）。结果：**−0.38 dB**（像素 L1 12.74±4.61 vs 12.02±4.50；最优 PSNR 20.99 vs 21.37；MS-SSIM 0.7356 vs 0.7731），**每个 t 都输**；且**丢掉整条 RD 曲线最左端的操作点**（2 token / **0.073 bpp** / 21.06 dB → 5 token / 0.181 bpp / 20.37 dB）⇒ 对"极低码率独占区"是净损失。**机制（本轮最有价值）**：t≥9 各步读窗口两臂**逐位相同**却处处差 0.69–0.99 px，且"冷启动首步"用**更大**窗口（9 位）反而更差（14.02 vs 12.67 px）⇒ **BPTT 让采样步构成串联草稿链，删一步的代价会摊到所有后续步**；`decoder_steps` **不是**"可自由裁剪的 token 预算表"。收益仅 +3.8% 步速（2.38→2.47 it/s）；自适应天花板 0.042→0.105 dB（**C2b 仍不成立**）。三臂对照图 `doc/2026-09-22/data/curves_3arms_224.png`、表 `data/COMPARE_3arms_224.md`、工具 `tools/compare_arms_224.py`、脚本 `tools/run_224_d4_bptt_slice1.sh`
- **★ DINOv2 逐层 tap（金字塔读出）· 设计与实测（2026-09-22）** ⚠️ **已于 2026-09-23 从 main 整块移除（本文只作存档）**: `doc/2026-09-22/REPORT_layer_tap_pyramid_224.md` + `doc/2026-09-22/DESIGN_layer_tap_pyramid.md` —— 用户口径：12 步 ↔ DINOv2-large 24 层「每 2 层一个 step」，从顶往下第 g 组读出 `2g−1` 个 register（1,3,…,23，合计 144=K）并**读出即从序列删除**（"用掉的向量不进入下一层"），解码器第 i 步只读第 i 组、carry 走 BPTT。代码 `--layer_tap`（默认关，**不新增任何参数**；口径由 `model_info.json` 驱动推理侧）。**单变量对照**（唯一差别 `--layer_tap`；construction_site test 3,004 / 224×126 / d4 / BPTT / 8,760 步）：**画质输 0.80 dB**（像素 L1 13.34±4.85 vs 12.02±4.50；PSNR 20.57 vs 21.37；MS-SSIM 0.7097 vs 0.7731；**每个 t 都输**，eval 轨迹全程 ≈1.11× 且两侧都已收敛 ⇒ 结构性）；**但曲线从平线变成真有斜率**（PSNR 跨度 0.30→1.69 dB、L1 跨度 0.67→3.98 px、首末步逐图相关 0.9994→0.9623、oracle 自适应天花板 +0.042→+0.269 dB；实际 ε 判据最好仅 +0.03 dB ⇒ **C2b 仍不成立**，只能当 analysis/ablation）。**机制**：那 `2g−1` 个是 **register 而非 patch**，底部 23 个只走 2 层 ⇒ 是"没被语义化的全局摘要"、**不是**浅层高频 ⇒ 并没有真的把浅层纹理接进解码器；末步 `F_hat` 吃最浅一组反不如基线"1 个走满 24 层的 register"。脚本 `tools/run_224_d4_bptt_tap.sh` / `tools/eval_224_d4_bptt_tap.sh` / `tools/compare_layer_tap.py`；图 `doc/2026-09-22/data/curves_tap_vs_base.png`
- **批 0 实测 · RD/PSNR/早停（2026-09-22，零训练）**: `doc/2026-09-22/REPORT_batch0_rd_earlystop.md` —— 一个 checkpoint 出完整 `bpp–PSNR / bpp–MS-SSIM` + 逐图逐 step 曲线（**C2a 成立**）；**但 C2b 不成立**（逐图自适应天花板 +0.13 dB、实际 ε 最好 +0.02）；**E1-lite 传统 codec 对照：PSNR 轴完败**（224×126 对 JPEG −0.4~−14.2 dB、对 WebP −5.3~−15.9 dB，且我们 0.3 bpp 后曲线就平）⇒ 确定性 RD 这条线应放弃，剩下的牌只有"极低码率独占区"与 **E8 任务保真**；含 224×126/d4/BPTT 操作点（L1 12.02 / PSNR 21.35）与"曲线比 448 更平（72× 码率换 +0.30 dB）"的诊断，以及 **BPTT vs detach 同代码对照（detach 的 RD 曲线是一条平线）**
- **曲线为何平坦：「纹理偏浅层」vs「解码器路由」对照（2026-09-22）**: `doc/2026-09-22/ANALYSIS_texture_vs_decoder_routing.md` —— 用户方向对（浅层 patch embed 本身不丢信息、高频是在中深层被语义化抹掉的），**但根因排序不在这里**：8-27 诊断把第一根因判给**解码器的逐 patch 信息路由缺陷**（P1-1 `q_patch` / P1-2 special 输入绑定 patch 特征，可回收 12.9 L1），而这两条**至今未落地**（`grep q_patch model_v2.py` 无命中）；当时代码**只用末层、没有任何浅层 tap**。**2026-09-22 晚的实测反证**：把浅层接进来（`--layer_tap`）画质**反而 −0.80 dB** ⇒ 支持"先修路由、再谈 tap"的排序
- **BPTT 送到编码器的信号有多弱？（2026-09-22，CPU 探针）**: `doc/2026-09-22/REPORT_crossattn_encoder_grad.md` —— 结论方向相反：BPTT 让编码器信号**变强**（`‖∂L/∂z_s‖` **×5.0**；扇出 1→**9.49** @24 步/K=576），且逐 step 梯度**相干**（两两 cos 均值 +0.493，不是相互抵消）；真正弱的是 ① `mean_t` 平权使交付的末步 `F_hat` 只占 `1/|T|`、② 编码器每参数梯度仍只有解码器的 **≈1/3** ⇒ "给编码器补信号"应靠损失加权或给 `z_s` 一条不经交叉注意力的直接监督，**不是**挪到交叉注意力
- **论文实验方案 / 文献与数据集 / 拼接式自注意力对照（2026-09-22）**: `doc/2026-09-22/PLAN_paper_experiments.md`（实验矩阵、§0.1 早停零成本分析、E3 真 bpp、**E8 任务保真判别树**）、`doc/2026-09-22/LITREVIEW_codec_papers_datasets.md`（可对标论文 / bpp 口径 / 数据集摘录）、`doc/2026-09-22/ANALYSIS_attn_concat_vs_cross.md`（拼接式自注意力 vs 交叉注意力：**掩码拼接逐位等价**，无掩码不等价——预算 + 梯度耦合）
- **aswin 工地隐患中译 · 术语/缩写修正（2026-09-18，R1/R2/R3）**: `doc/2026-09-18/REPORT_zh_term_fix.md` —— 用户复核 Label Studio 时提出三条口径：**R1** `not using PPE` 类否定句不得译成「不得使用…」（原译「左侧人员不得使用PPE。」语义反转）、**R2** `hard hat` 统一「安全帽」不出现「硬帽」、**R3** 任何缩写首次出现须括号附全称（全称可为英文）；新增**确定性修正器** `zh_fix.py`（幂等）并接入 `enrich_manifest_zh.py`，全量修正 **383 处**（硬帽 353 / 缩写 16 / 同源残留英文 11 / 拼写 2 / R1 1）⇒ `硬帽`、`不得使用PPE` 残留 **0**；线上 Label Studio **316 个任务就地 PATCH meta_html**，**31 条人工标注零丢失**。脚本与逐处 diff `doc/2026-09-18/data/zh_term_fix/`
- **思维链（CoT）试点 · 口径验证（2026-09-18，7 数据集 × 100 张）**: `doc/2026-09-18/REPORT_cot_pilot.md` —— Qwen3.8-27B 双路（`:8100`/`:8101`）生成 + DeepSeek `deepseek-flash` **视觉**复核；图像统一 **12:9 画布 1200×900**（横图 letterbox 灰底 127、**竖图顺时针旋转 90°**），bbox 归一化 **0–1000**，detect 口径 = `spatial` + 逐目标 `position` + **坐标由系统拼接真值**；**700 → 697 条可用（99.6%）**、detect **400/400 目标数对齐**、DS 复核 **694/697（99.6%，两次取严）**；**关键 A/B**：提示词中给出真值坐标作定位线索 ⇒ detect 通过率 **73.8% → 99.25%**（hf-vision 54%→99%、iluvvatar 60%→100%）；并定位/修复致命基础设施故障：**本地 Qwen3.8-27B 视觉权重被抹零**（33 个张量整层为 0，含 `visual.patch_embed`、`visual.blocks.11/12`）⇒ 模型"看不见图"，从 ModelScope 重下 shard1 修复（诊断脚本 `vision_check.py` / `image_format_probe.py`）。数据与脚本 `doc/2026-09-18/data/cot_pilot/`
- **全量 CoT 生成 · 完成（2026-09-20，7 数据集 75,717 条）**: `doc/2026-09-18/REPORT_cot_full.md` —— 双卡两路 Qwen3.8-27B、两车道，**75,717/75,717 全部生成**、解析成功 **75,713（99.995%）**、detect 目标数对齐 **37,302/37,306**；机械门禁 **画布✗0 / 尺寸✗0（全量核对 JPEG 头）/ 退化重复 0**；残留：**4 条** `max_tokens` 截断（不可机修，只能重生成）+ 4 条 detect 错位 + 9 条真值框非法 + 54 条空/过短；全程 **42h20m**（09-18 14:51 → 09-20 09:11，均速 **0.497 条/s**），会话中断与工控机重启均无损（`setsid nohup` + 常驻看护 `cot_full_watch.py` 自动收尾 `FINALIZE_DONE`）；新增 `cot_full_repair2.py`（括号配平二次回收 4 条，带截断守卫）。产物 `doc/2026-09-18/data/cot_full/{SUMMARY.json,QA.json,SUMMARY.md}` + 脚本（与服务器 **md5 一致**）`doc/2026-09-18/data/cot_pilot/`
- **全量中文译文质检 · 结论推翻旧"3–5%"（2026-09-18，21,329 个字段）**: `doc/2026-09-18/REPORT_translation_audit_deepseek.md` —— 评委 DeepSeek `deepseek-flash`（必须 `reasoning_effort:"none"`：开推理反而慢 13×、token 38×；它**支持图片**而 `deepseek-v4-pro` 不支持）；22 条标注集校准**抓错率 0.93 / 误报率 0.00**，另抽 16 条 major 人工通读 **16/16 都含真实错误**；全量 **major 4,499 / 21,329 = 21.09%**（`aswin00000` 26.52%、`hayden-yuma` 19.93%、类名 1.83%、其余 ≈0）⇒ 旧"3–5%"低估 **5–8 倍**；产物含 **4,499 条可定位清单** `flagged_major.jsonl`（`doc/2026-09-18/data/translation_audit/`）
- **Qwen3.8-27B 不能当文本评委（2026-09-18，LLM-as-judge 校准）**: `doc/2026-09-18/REPORT_llm_judge_calibration.md` —— 6 个提示词版本在 16 条校准集上**误报率 0.60–1.00**：几乎对每条都判 major，且会**编造证据**（把正确译文 `excavator→挖掘机`、`hard hats→安全帽` 判错，把模板占位符 `<英文片段> || <中文片段>` 当错误抄出）；开 thinking 质量好转但被 `max_tokens` 截断、~13 s/条（全量 ≈40 h）⇒ **不可用**，故全量质检改用 DeepSeek
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

**2026-09-22 新证据（BPTT 默认化 + 逐层 tap 实测，两条都与本节的"钥匙"有关）**：
1. **carry 改 BPTT 后"平线"部分被推翻**：仓库默认已从 `detach` 切到 **BPTT**（§1），实测 224×126 上 detach 的 RD 曲线是**一条平线**（跑满 576 与 t=1 的 PSNR 相同甚至更低），BPTT 有真阶梯（`doc/2026-09-22/REPORT_batch0_rd_earlystop.md` §2b；全量 448 上像素 L1 23.72→16.42）。⇒ 本节早期"后步输出≈0 / 零增量自锁"的结论**是在 detach 口径下得到的**，换成 BPTT 后已不再是当前现象；判据请按 BPTT 口径重述。
2. **把"向量来自哪一层"多样化，能买到斜率但买不到画质**：`--layer_tap`（逐层金字塔，§1）把 PSNR 跨度从 0.30 dB 提到 **1.69 dB**、首末步逐图相关从 0.9994 降到 **0.9623**（曲线不再只是整体平移），但绝对画质**输 0.80 dB**，且 oracle 自适应天花板仍只有 **+0.269 dB**（`doc/2026-09-22/REPORT_layer_tap_pyramid_224.md`）。**这正好落在本节的第二个耦合上**：那 `2g−1` 个是 register 而非 patch，底部 23 个只走 2 层 ⇒ 仍是"内容可寻址性"不足，只是换了个深度 ⇒ **本节的 P2（special 输入拼 `Linear(patch_feat)`，让键各自携带逐 patch 内容）依旧是未被触碰的那根杠杆**；P1（分区域掩码损失）同样未做。
   > ⚠️ **2026-09-23 更新**：`--layer_tap` 已按用户口径整块从 main 移除（§1）；本条只作历史证据保留。

**2026-09-23 新证据（A 相预算 = 分辨率扫描"平线"的近因）**：`doc/2026-09-23/` 复核发现原分辨率扫描的高分辨率臂**停在平凡解（预测均值）**，近因是两相配方的 A 相（`--warm_steps`，单步全读）在所有分辨率恒 1000 步、而 224² 实测要 1500~2000 步才逃逸；把 A 相提到 3000 步后同一架构的 16 步曲线由平变单调降（最优 PSNR 14.26→**18.69**）。⇒ 本节早期"后步输出≈0 / 零增量自锁"的结论在 BPTT + 足够 A 相预算下**已不是当前现象**（`doc/2026-09-23/REPORT_224_A3000_verdict.md`、`ANALYSIS_res_sweep_rootcause.md`）。

**战略上下文**：渐进阶梯**不是** GOAL 验收项（`doc/2026-08-28/GOAL_compression_for_nlp.md`），Phase 2 一次性消费全部 K token；v4 单发（8.26）已是仓库最佳。除非"token 增量性"叙事本身成为目标，本条目可长期冻结，不挡主路线。

> 相关文档：`doc/2026-09-07/ANALYSIS_k3_why_later_steps_zero.md`（主分析）、`doc/2026-09-07/ANALYSIS_k3_slice_infodiff.md` / `doc/2026-09-07/REPORT_v2_E1_probe.md`（信息侧证据）、`doc/2026-09-07/DESIGN_v2_region_loss.md`（P1 设计与实现记录）、`doc/2026-09-15/DESIGN_v2_recurrent.md`（循环架构）
