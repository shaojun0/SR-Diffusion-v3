# SR-Diffusion v3 — v2 K 自动推导（无花瓶 register, K=99）完整 5 步模型 训练+推理报告

> 日期: 2026-09-02 ｜ 分支: `main`（HEAD `b653847`, K=num_specials 由最终采样步集推导）
> 服务器: `connect.westd.seetacloud.com:25398`, 2× RTX PRO 6000 (97GB), torch 2.12.1+cu130, transformers 5.16.1
> 性质: `doc/2026-09-02/DESIGN_v2_num_specials_from_max_steps.md`（去花瓶 register, K 解耦）落地后的**第一次完整 5 步训练**；与旧花瓶式 K=576 block 基线（全量 L1=17.46）做**单变量对比**——同预算（8760 步 / 有效 bs32 / 全 fp32 / seed 42 / decoder_depth 2 / slice [4:9]），唯一变量 = K（576→99，编码器 register 数 = 解码器实际读窗口）。
> 判据: 无花瓶版若在相同预算下 ≥ 花瓶版 → 花瓶 register 的端到端净效应为正（或至少砍掉它们同时砍掉了编码器有用容量），K 解耦未带来重建收益。

---

## 0. 结论速览（TL;DR）

1. **训练完成**（8760 步，2 卡 1h49m，exit 正常），全量 3004 张 test：像素 L1 = **18.533**（归一化 0.3236）。
2. **比旧花瓶式 K=576 block 基线差**: 18.533 vs 17.460 = **+1.073（+6.1%）**——同预算单变量下，去花瓶后重建反而变差。
3. **5 步渐进曲线依旧全平**: [18.509, 18.507, 18.511, 18.520, 18.533]（极差仅 0.026，与旧版 17.41→17.46 同一形态）——K 解耦**没有**改变"渐进机制失效/曲线全平"现象。
4. **训练期 eval 轨迹对比**: 新 0.4820→0.4047→0.3460→0.3242→0.3231 vs 旧 0.4783→0.4199→0.3436→0.3059→0.3045：前 6000 步几乎持平，**6000→8760 段被旧版甩开**（8000 步时 0.3242 vs 0.3059）。
5. **机制判读**: 两种 K 下**解码器读窗口完全相同**（都只 attend 位置 1..99 + z_cls，K=576 时 100..576 的键被掩码屏蔽、pos_embed 从不训练）；唯一结构性差异在编码器侧——K=576 时 100..576 的"花瓶" register 仍作为 DINO 24 层全双向注意力的**中间 token** 参与计算 z_s[1..99]（前向/反向跨 token 耦合，DESIGN §1 已实测梯度 9.81 vs 6.35、扰动 ~68%）。端到端看这些花瓶 register 提供的是**编码侧附加容量/工作记忆**：砍掉后 z_s[1..99] 由更少的 token 算出来，重建净损失 +1.07。**"花瓶干扰训练动力学"的局部测量（DESIGN §1）没有转化为端到端收益。**
6. 对本系列的意义：K 自动推导/去花瓶**不是**重建 L1 的修复路径；曲线全平的根因（损失前置最优 + 首步前缀 + 查询自注意力混叠，见 ANALYSIS_v2_all_failures）在 K 解耦后原样保留——E2（逐 patch 内容查询）仍是下一步主修复。

---

## 1. 背景与目的

- `DESIGN_v2_num_specials_from_max_steps.md`（b653847）认为 register 式里"编码器生成 N 个 register、训练只读 slice 内的部分"存在**花瓶 register**：位置 100..N 从不被解码器读，但经 DINO 全双向注意力耦合读窗口（局部证据: 花瓶区梯度 |Σ|=9.81 > 读窗口 6.35；逐维扰动花瓶输入 → 读窗口输出变化 ~68%），应消除而非容忍。
- 修复 = K 与 N 解耦，K = `min(max_t((⌊√t⌋+1)²−1), N)`。本配置 slice [4:9] → steps=[25,36,49,64,81] → **K=99**（覆盖解码器全部可读位置 1..99），DINO 序列 1153→676 token。
- 待回答: 无花瓶版在**同预算**下能否达到/超过旧花瓶式 K=576 的 17.46？此报告即该单变量实验。

## 2. 配置（与旧 block 基线的对照）

| 项 | 旧 block（K=576, 花瓶式, 09-01） | 本次 block-k99（K=99, 自动推导） |
|---|---|---|
| 代码 | HEAD 之前 register-only | **HEAD b653847**（derive_num_specials/select_steps） |
| slice → steps | [4:9] → [25,36,49,64,81] | 同 |
| K（num_specials） | 576（=N, 花瓶 100..576 无人读） | **99**（自动推导, 无花瓶） |
| DINO 输入序列 | 1+576+576 = **1153** token | 1+99+576 = **676** token |
| 可训练参数 | 340.3M | 339.4M |
| 有效 batch / 总步数 | 2×16=32 / 8760 | 同 |
| epochs / lr / wd / warmup / seed | 40 / 1.5e-4 / 0.01 / 3% / 42 | 同 |
| 精度 / decoder_depth / 输入 | fp32 / 2 / 448×252→576 patches | 同 |
| 墙钟（2 卡） | ~2h45m | **1h49m**（序列短 ~41%） |

## 3. 结果（全量 3004 张 test，fp32 推理）

| 指标 | 旧 block（K=576） | **block-k99（K=99）** | Δ |
|---|---|---|---|
| 归一化 L1 | 0.3054 | **0.3236** | +0.0182 |
| 像素 L1 (0-255) | 17.460 (±6.16) | **18.533 (±6.51)** | **+1.073（+6.1%）** |
| 5 步渐进像素 L1（step_pixel_l1_255） | [17.415, 17.411, 17.415, 17.435, 17.460] | **[18.509, 18.507, 18.511, 18.520, 18.533]** | 形态相同（全平） |
| 前段/后段均值 | 17.419 / 17.431 | 18.512 / 18.518 | — |
| 最少/最多步累积差 | 0.045 | **0.024** | — |

训练期 eval_loss 轨迹（归一化空间，n=3004，同预算逐步对比）：

| step | 旧 block（K=576） | block-k99（K=99） | 差值 |
|---|---|---|---|
| 2000 | 0.4783 | 0.4820 | +0.004 |
| 4000 | 0.4199 | 0.4047 | **−0.015**（新略优） |
| 6000 | 0.3436 | 0.3460 | +0.002 |
| 8000 | 0.3059 | 0.3242 | **+0.018**（旧反超） |
| 8760 (final) | **0.3045** | **0.3231** | **+0.019** |

（8000/8760 两点：n=3004 下像素 L1 均值 SE≈0.12，Δ1.07 ≈ 9 SE——统计上确凿。）

## 4. 判读

1. **K 解耦（去花瓶）端到端净效应为负**（+1.07 L1，≈旧值的 +6%）。"花瓶 register" 虽在局部测量（梯度/扰动耦合）中显示干扰特征，但端到端删除它们同时删掉了编码器侧的有用容量：K=576 时花瓶 register 是 DINO 24 层双向注意力里的**额外中间 token/工作记忆**，其隐状态经逐层注意力参与 z_s[1..99] 的计算（这正是 DESIGN §1 实测到的"前向/反向耦合"），因而有正的贡献。
2. **解码器功能等价**：两版解码器 attend 的键完全相同（1..99 + z_cls；K=576 的 100..576 键全被掩码、对应 pos_embed 从不训练，ANALYSIS_v2_block_params_deep §2.2 已证 83% 位置不训练）。故 +1.07 全部归因于**编码器 register 数**（99 vs 576）→ z_s 计算容量。
3. **曲线全平原样保留**：极差 0.024–0.026（新旧同形态）。这与 ANALYSIS_v2_all_failures 的结论一致——全平由损失结构（累加 + 首步前缀规则 + 查询间自注意力混叠）强制，与 K/花瓶无关；K 解耦不是修复路径。
4. **尾段收敛被甩开**：新模型 4000 步前略优（0.4047 vs 0.4199），6000 步后旧模型持续反超（8000: 0.3059 vs 0.3242）——更少 token 的编码路径在后期容量上吃亏。
5. 与 S25/S64 单步消融（REPORT_v2_single_step_retrain, 花瓶式 K=576）关系：旧花瓶式单步 S25=19.15 / S64=19.58；本次给出**花瓶式 5 步 17.46 vs 无花瓶 K=99 5 步 18.53** 的直接对照——"K 解耦→无花瓶→更干净训练"的假设在两个层级（单步/5 步）都未转化为更低的 L1。

## 5. 对后续实验的建议

1. **K 扫描（DESIGN §6.2 的权威版）**: 同 slice [4:9]、同预算下扫 K ∈ {64, 99(自动), 128, 256, 576(花瓶)}，画 "K × 重建 L1" 曲线——判定 register 数对重建的真实贡献曲线（本报告只给了两个端点，K=99 与 K=576）。
2. **花瓶净贡献的归因实验**: 在 K=576 花瓶版上做"花瓶 register 输入置零/随机"消融，测读窗口 z_s[1..99] 输出与最终 L1 的变化，把 DESIGN §1 的局部耦合（~68% 扰动）升级为端到端贡献量化。
3. **曲线全平的主修复不变**: E2（逐 patch 内容查询 `Q = A_t + query_base + Linear(z_s[k])`）+ 损失结构改造（每步独立预测），与 K 无关——K 解耦实验证明此路不通后应回到 E2 主线。

## 附：相关文件与复现

- 服务器产物: `/root/autodl-tmp/sr-diffusion-v2-k99/output/phase1_v2_block_k99/`（`final_model.pt` fp32 / `infer_test.json` / `model_info.json`（num_specials=99）/ `args.json` / `checkpoint-{2000,4000,6000,8000,8760}` / `recon_visual_k99.png`）
- 训练日志: 服务器 `/root/train_logs/train_block_k99.log`（1h49m, 8760 步, exit 正常）
- 代码: 服务器 `/root/autodl-tmp/sr-diffusion-v2-k99/`（b653847 同步版, 新建干净目录, 旧 v2-new 未动）
- 复现训练:
  ```bash
  cd /root/autodl-tmp/sr-diffusion-v2-k99
  accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --same_network train_v2.py \
    --data_dir /root/autodl-tmp/construction_site --dino_dir /root/autodl-tmp/models/dinov2-large \
    --output_dir output/phase1_v2_block_k99 --model_input 448x252 --canvas 1600x900 --angle_step 0.5 \
    --epochs 40 --max_steps 8760 --batch_size 16 --lr 1.5e-4 --weight_decay 0.01 \
    --warmup_ratio 0.03 --grad_clip 1.0 --num_workers 8 --eval_every 2000 --save_every 2000 \
    --seed 42 --heads 8 --mlp_ratio 4.0 --decoder_depth 2 --slice_start 4 --slice_end 9
  ```
- 复现推理（K 自动从 model_info.json 对齐，需与训练一致的 slice）:
  ```bash
  python infer_v2_test.py --data_dir /root/autodl-tmp/construction_site \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --final_model output/phase1_v2_block_k99/final_model.pt \
    --output output/phase1_v2_block_k99/infer_test.json --slice_start 4 --slice_end 9
  ```
