# SR-Diffusion v3 — v2 `self.stack` 放大 2×（d_model 2048 / heads 16 / depth 4 / dropout 0.05）实验报告

> 日期: 2026-09-11 ｜ 分支: `main` ｜ 服务器: 2× RTX PRO 6000 (97GB), 2 卡 DDP
> 代码: `c9e3229`（模型/推理改动）+ `b819f7c`（可视化看板）
> 服务器目录: `/root/autodl-tmp/sr-diffusion-v3-stack2x`
> 对照基线: `doc/2026-09-10/REPORT_v2_blockdiag_slice05.md`（`eval_recon` **0.3318**）
> 看板: TensorBoard `:6006` / Streamlit `:6008`（见文末）

---

## 0. 结论速览（TL;DR）

**结论：按本次指定配置（`self.stack` 的 d_model/heads/depth 全部 2×、dropout=0.05，lr 保持不变 = 1.5e-4），
模型在 warmup 结束前后即进入退化吸引子（塌缩），不是"收敛慢"。**

硬证据（全部为实测）：

1. **训练 loss 反升后冻结**：step 200 `loss=1.056 / grad_norm=0.327`（与基线 1.052/0.352 同级）→
   step 400 `loss=1.151 / grad_norm=0.007`，此后直到 step≈2500 始终在 **1.14–1.16 / 0.002–0.013**
   之间抖动（基线同期已降到 **0.477 / grad≈1.1**）。塌缩发生在 warmup 峰值（`warmup=262` 步、
   lr=1.5e-4）附近。
2. **首个 eval（step 2000）**：`eval_recon = 1.139`，基线同点 `0.486`（差 2.3×）。
3. **`checkpoint-2000` 探针（step-collapse）**：解码器输出与 register 同时退化：
   - `step_px_scale = [0.1814, 0.1814, 0.1814, 0.1814, 0.1814]` —— **5 个采样步完全相同**，
     模型已"看不见"采样步；基线为 `[1.027, 0.0395, 0.0349, 0.0342, 0.0335]`（step1 主导、后步 3–4%）。
   - `prog_curve_255 ≈ 69.33`（5 步全等）≈ 全图平均色基线（≈61）→ 输出退化为常量图。
   - `E_px` 的 5 行逐位相同（step 维度消失），仅按 region 呈 `[91.4, 80.6, 65.9, 57.8, 51.0]`，
     即"对每个区域输出同一个常量"。
   - `z_s_within_std = 0.0197`（基线 0.0951）、块内 cos = `[0.996, 0.999, 0.998, 1.000, 0.992]`
     → DINO 侧的 register/specials 也塌成近同一向量。
4. 与仓库历史多次记录的「塌缩吸引子」（块级共享位置编码 `3151bab`、读侧全开等）**同型**：
   loss 平坦 + grad≈0 + 后步与 register 同步退化。

**原配置（lr=1.5e-4）训练在 step≈2500 手动停止**（塌缩自 step 400 起已持续 2000+ 步，无恢复迹象）。
随后按根因结论把峰值 lr 降到 1.0e-4 **完整重跑了 8760 步**（见 **§8**）：**塌缩消失，但最终
`eval_recon 0.4174` 仍显著劣于基线 0.3318（+25.8%）** ⇒ 本任务上"放大 self.stack 2×（含 dropout 0.05）"
不是有效方向。本次改动本身**不污染历史配置**：`stack_dim=0` 默认路径与 HEAD `6915f6b` 的
state_dict 键/值与 forward 输出**逐位相同**（见 §3），旧 checkpoint 双向可载。

---

## 1. 动机与改动

`OutputQueryDecoder` 的解码器堆叠 `self.stack` 原为:

```python
nn.TransformerDecoder(
    nn.TransformerDecoderLayer(d_model=dim, nhead=heads,
                               dim_feedforward=int(dim * mlp_ratio),
                               dropout=0.0, ...),
    num_layers=depth)
```

其中 `dim = dino.config.hidden_size = 1024`（DINOv2-large 输出维，模型其余部分
`pos_embed` / `query_base` / `PixelHead` 全部绑死在该维），`heads=8`、`depth=2`、
`dropout=0.0`（**硬编码**）。

本次把 `self.stack` 的容量翻倍，并给解码器引入轻度 dropout：

| 项 | 基线 | 本次 |
|---|---|---|
| `d_model`（= `self.stack` 的） | 1024 | **2048** |
| `heads` (`nhead`) | 8 | **16** |
| `depth` (`num_layers`) | 2 | **4** |
| `dropout` | 0.0 | **0.05** |
| `dim_feedforward` | 4096 | 8192（随 `d_model` 等比，`mlp_ratio=4` 语义不变） |

## 2. 实现细节（前后 Linear 投影）

`dim` 被 DINO 锁死、不能跟着变，所以 `self.stack` 独立加宽后必须在两侧对齐维度：

- `stack_in:  nn.Linear(dim, stack_dim)`（`1024 → 2048`）：查询 `Y=(B,|T|·N,dim)` 与
  memory `A=(B,S,dim)` **共用**同一个投影送入 `self.stack`；
- `stack_out: nn.Linear(stack_dim, dim)`（`2048 → 1024`）：`self.stack` 输出投影回
  `dim`，交回 `PixelHead` / 逐时刻累加损失（`last_Y`、`F_hat`、`Y_pix` 形状与语义不变）。

新增构造参数（默认值保证旧行为逐位不变）：

- `OutputQueryDecoder(..., stack_dim=None, dropout=0.0)`
- `SRPhase1V2(..., stack_dim=0, decoder_dropout=0.0)`
- CLI: `train_v2.py --stack_dim 2048 --decoder_dropout 0.05`（`--stack_dim 0` ⇒ 与原实现一致）

`stack_dim == dim`（默认）时 `stack_in` / `stack_out` 是 `nn.Identity`，**不新增任何参数**。
`model_info.json` 额外记录 `stack_dim` / `decoder_dropout`，推理侧据此重建。

## 3. 兼容性验证（本地）

1. `python model_v2.py` 全量自检通过，含新增的加宽路径检查（d_model 64→128、投影形状、
   depth=4、dropout=0.05、save/load 往返）。
2. **默认路径与 HEAD 逐位一致**：对 `OutputQueryDecoder`（`stack_dim=dim`）用同 seed 构造，
   `state_dict` 键集合与值、以及 forward 输出 `torch.equal` 全相等 ⇒ 旧 checkpoint 行为不变。
3. 服务器 smoke（单卡 `--smoke --limit 32 --max_steps 3`）通过；落盘权重形状核对：
   `decoder.stack_in.weight (2048,1024)`、`decoder.stack_out.weight (1024,2048)`、
   `decoder.stack.layers.{0,3}.linear1.weight (8192,2048)`；`model_info.json` 记录
   `heads=16, decoder_depth=4, stack_dim=2048, decoder_dropout=0.05`（共 522 个张量，
   可训练参数 **581.1M**，基线 ~355M）。

## 4. 训练配置（与基线唯一变量 = self.stack 规模 + dropout）

| 项 | 值 |
|---|---|
| 数据 | `construction_site` 7009 train / 3004 test（448×252, N=576 patches） |
| 架构 | DINOv2-large(dim=1024, 不冻结) → OutputQueryDecoder(**stack_dim=2048, 4 层, 16 头, dropout 0.05**) → PixelHead |
| 采样切片 | `slice_start=0 / slice_end=5` ⇒ steps `[1,4,9,16,25]`, K=35 |
| 掩码 | `query_mask_mode=blockdiag`（默认, 未显式传） |
| 优化 | fp32, lr=1.5e-4 cosine, warmup_ratio=0.03(=262 步), wd=0.01, grad_clip=1.0, seed=42 |
| 预算 | 40 epochs / 8760 步, 2 卡 DDP, bs=16/卡（全局 32, 与基线一致） |
| 实测吞吐 | 1.96 s/it（两卡均 100%, 65.9/97 GB 显存） |
| 输出 | `output/phase1_v2_stack2x_slice05/`（checkpoint-2000 + 日志 `stack2x_slice05.log`） |

## 5. 训练结果（于 step≈2500 提前停止）

### 5.1 train loss / grad_norm 轨迹（同 step 与基线对照）

| step | 本次 loss | 本次 grad_norm | 基线 loss | 基线 grad_norm |
|---|---|---|---|---|
| 200 | 1.0560 | 0.327 | 1.0520 | 0.352 |
| 400 | **1.1510** | **0.007** | 0.6505 | 1.594 |
| 600 | 1.1590 | 0.007 | 0.5915 | 0.995 |
| 800 | 1.1500 | 0.005 | 0.5578 | 1.524 |
| 1000 | 1.1610 | 0.005 | 0.5236 | 1.116 |
| 1200 | 1.1610 | 0.013 | 0.5246 | 0.902 |
| 1400 | 1.1480 | 0.002 | 0.5246 | 1.239 |
| 1600 | 1.1540 | 0.004 | 0.5159 | 1.464 |
| 1800 | 1.1420 | 0.005 | 0.5100 | 2.112 |
| 2000 | 1.1380 | 0.004 | 0.4766 | 1.116 |
| 2500（停止） | 1.1440 | 0.002 | —（继续训练至 8760） | — |

> 判据：健康训练的 grad_norm 量级为 O(1)（基线 0.9–2.1）。本次自 step 400 起降到
> **1e-2 以下并再未恢复**，loss 同时反向升高后冻结 ⇒ 典型的参数更新停滞（塌缩），
> 不是"更大模型收敛慢"（收敛慢应表现为 loss 持续下降、grad 保持量级）。

### 5.2 eval_recon

| step | 本次 | 基线 |
|---|---|---|
| 2000 | **1.139** | 0.486 |
| 4000 | 未到（训练已停） | 0.4199 |
| 6000 | — | 0.3525 |
| 8000 | — | 0.3325 |
| 8760 (final) | — | **0.3318** |

## 6. 塌缩证据（`checkpoint-2000` step-collapse 探针）

同一份探针脚本（`doc/2026-09-11/probe_step_collapse.py`），基线用其历史探针（512 张）：

| 指标 | 基线 blockdiag slice[0:5] | 本次 self.stack 2×（ckpt-2000） |
|---|---|---|
| `step_px_scale`（step1~5 未累加增量均值） | `[1.027, 0.0395, 0.0349, 0.0342, 0.0335]` | `[0.1814, 0.1814, 0.1814, 0.1814, 0.1814]` |
| `prog_curve_255`（累积重建 L1） | `[19.86, 19.84, 19.84, 19.84, 19.85]` | `[69.33, 69.33, 69.33, 69.33, 69.33]` |
| `z_s_within_std` | 0.0951 | **0.0197** |
| 块内 cos | `[0.734, 0.876, 0.918, 0.916, 0.926]` | `[0.996, 0.999, 0.998, 1.000, 0.992]` |
| `E_px` 行间（step 维）差异 | 有（后步≈持平但非恒等） | **零**（5 行逐位相同） |

- `step_px_scale` 5 步全等 ⇒ 解码器对采样步（即对 `z_s` 的不同块）完全无差别；
- `prog_curve ≈ 69.3` ≈ 全图平均色基线 ⇒ 输出退化为常量；
- `z_s_within_std` 掉到基线的 1/5、块内 cos≈1 ⇒ DINO 侧 register 也同步塌缩；
- 与 §5 的 grad≈0 一致：整条读出链路（DINO specials → 解码器 → PixelHead）落到常量解。

> 备注：探针目录已产出 `output/probe/probe_stack2x_slice05_step2000.json`，
> 看板会自动把它与基线曲线并列显示（`step_px_scale` 与 `E_px` 热力图）。

## 7. 结论与后续建议

1. **直接放大 2× 不可行（在 lr 不变的前提下）**：更大的 `self.stack`（2048/16/4 + dropout 0.05）
   在同一 lr=1.5e-4 下于 warmup 峰值附近塌缩到常量解，`eval_recon` 1.139 ≫ 基线 0.3318。
2. **最可能的主因是 lr 未随宽度缩放**：塌缩点正好是 lr 首次达峰（262 步）之后；
   常规做法是宽度 ×2 时把 lr 降到约 `1/√2`–`1/2`（即 7.5e-5–1.1e-4），或延长 warmup。
   `dropout=0.05` 引入的噪声可能进一步把模型推向"预测条件中位数（常量）"的宽平坦解，需单独消融。
3. **建议的下一步（如要继续）**：
   - 变量隔离：先只加 `depth 2→4`（历史 `d4` 已验证 18.77→18.25，正向），再单独加 width；
   - lr 扫描：`7.5e-5` / `1.0e-4`，`dropout=0`；
   - 早停判据：训练前 500 步内若 `grad_norm < 1e-2` 且 loss 不降，判定塌缩、立即终止，省 GPU；
   - 保留本次的 `--stack_dim` 接口即可，无需改代码。
4. **本次代码改动本身是安全的**：默认路径逐位不变（§3），加宽/ dropout 仅由 CLI 开关触发，
   已提交 `c9e3229`。

## 8. 修复版复跑（峰值 lr 1.5e-4 → 1.0e-4）：塌缩消失，但仍劣于基线

依据 §7 与根因分析（`doc/2026-09-11/ANALYSIS_stack2x_collapse_rootcause.md`：触发变量是
**d_model 宽度 × lr 调度**），把峰值 lr 降到 **1.0e-4** 重跑完整 8760 步；其余逐项不变
（2048/16/4、dropout 0.05、blockdiag slice[0:5]、K=35、2 卡 × bs16、seed 42）。

### 8.1 训练健康（对比塌缩 run）

| step | 塌缩 run (lr=1.5e-4) | 修复 run (lr=1.0e-4) |
|---|---|---|
| 200 | loss 1.056 / grad 0.327 | 正常下降 |
| 572 | 1.150 / 0.006（已塌） | **0.815 / 7.24** |
| 4286 | —（已停） | 0.455 / 1.6–2.0 |
| 8760 | — | 完整跑完（~4h59m），守卫未触发 |

`eval_recon` 曲线（训练内 eval，全量 3004 张）：

| step | 基线 blockdiag | 塌缩 run | **修复 run** |
|---|---|---|---|
| 2000 | 0.4860 | 1.139 | **0.5112** |
| 4000 | 0.4199 | 1.139（冻结） | **0.4655** |
| 6000 | 0.3525 | — | **0.4376** |
| 8000 | 0.3325 | — | **0.4191** |
| 8760 (final) | **0.3318** | — | **0.4174** |

### 8.2 全量 test 推理（同一脚本、同口径）

| 指标 | 基线 | 修复 run | 变化 |
|---|---|---|---|
| 归一化空间 L1（eval_recon 口径） | **0.3318** | 0.4173 | **+0.0855（+25.8%）** |
| 0-255 像素 L1 | 19.02 | 23.78 | +4.76（+25.0%） |
| 渐进曲线（5 采样步累积, 0-255） | `[19.02, 19.01, 19.01, 19.01, 19.02]` | `[24.22, 23.78, 23.77, 23.77, 23.78]` | — |

### 8.3 step-collapse 探针（512 张，同一脚本）

| 指标 | 基线 | 塌缩 run (ckpt-2000) | **修复 run（final）** |
|---|---|---|---|
| `step_px_scale` | `[1.027, 0.0395, 0.0349, 0.0342, 0.0335]` | `[0.1814]×5` | `[0.969, 0.0778, 0.0686, 0.0701, 0.0711]` |
| `prog_curve_255` | `[19.86, …]` | `[69.33]×5` | `[25.29, 24.94, 24.93, 24.93, 24.93]` |
| `z_s_within_std` | 0.0951 | 0.0197 | 0.0614 |

⇒ 修复 run **没有塌缩**：step1 幅度正常（0.969）、5 步不再全等、register 未被钉死；
但整体重建质量明显低于基线。

### 8.4 随训练进程（各 checkpoint 探针，`limit=128`）

| ckpt step | `step_px_scale` (s1..s5) | `prog_curve_255` | `z_s_within_std` |
|---|---|---|---|
| 2000 | `[0.863, 0.069, 0.070, 0.070, 0.070]` | `[29.31, …]` | 0.0707 |
| 4000 | `[0.935, 0.067, 0.065, 0.065, 0.065]` | `[27.46, …]` | 0.0632 |
| 6000 | `[0.951, 0.066, 0.065, 0.065, 0.065]` | `[26.25, …]` | 0.0657 |
| 8000 | `[0.974, 0.082, 0.070, 0.070, 0.071]` | `[25.29, …]` | 0.0613 |
| 8760 | `[0.976, 0.081, 0.070, 0.070, 0.072]` | `[25.23, …]` | 0.0609 |

step1 幅度随训练上升、重建 L1 单调下降（29.3→25.2），与 §8.1 的 eval 曲线一致。

### 8.5 结论（修复版）

1. **lr=1.0e-4 成功消除塌缩**（final `eval_recon` 0.4174 vs 塌缩 1.139），验证了 §7/根因分析的归因。
2. **但 2× self.stack 在健康训练下仍显著劣于基线**：全量 0.4173 vs 0.3318（**+25.8%**），
   且在 4000/6000/8000 步的差距稳定在 0.08–0.09、无收敛趋势。
3. **归因注意（混淆变量）**：修复 run 相对基线同时变了 *宽度/深度/heads + dropout 0.05 + lr*，不是单变量对照；
   差距可能部分来自 dropout（基线 dropout=0）。若要判定"加宽本身是否有益"，需补 `dropout=0` 的同配置对照。
4. **综合结论**：本任务上「把 `self.stack` 放大 2×（含 dropout 0.05）」**不是有效方向** ——
   沿用原 lr 会塌缩；降 lr 后虽健康，但仍不如原 2 层/dim1024 基线。

产物：`doc/2026-09-11/data/`（infer/probe JSON、`AB_lr1e4.md`、`AB_lr1e4.png`）；看板 `:6006`/`:6008` 三路对照。

---

## 附 A: 复现命令

```bash
# 本次（会塌缩）配置
accelerate launch --multi_gpu --num_processes 2 --num_machines 1 --same_network train_v2.py \
  --data_dir /root/autodl-tmp/construction_site --dino_dir /root/autodl-tmp/models/dinov2-large \
  --output_dir output/phase1_v2_stack2x_slice05 --model_input 448x252 --canvas 1600x900 --angle_step 0.5 \
  --epochs 40 --max_steps 8760 --batch_size 16 --grad_accum 1 \
  --lr 1.5e-4 --weight_decay 0.01 --warmup_ratio 0.03 --grad_clip 1.0 \
  --num_workers 8 --eval_every 2000 --save_every 2000 --log_every 20 --seed 42 \
  --heads 16 --mlp_ratio 4.0 --decoder_depth 4 --stack_dim 2048 --decoder_dropout 0.05 \
  --slice_start 0 --slice_end 5

# 塌缩探针（对 checkpoint，需先转 safetensors→pt）
python probe_step_collapse.py --ckpt <ckpt>/final_model.pt --tag stack2x_ckpt2000 --limit 128 --out out.json
```

## 附 B: 指标可视化看板（开源方案）

- **TensorBoard**：`https://u849831-me4w-bf7ae4cd.westb.seetacloud.com:8443`（:6006）
- **Streamlit 交互看板**：`https://uu849831-me4w-bf7ae4cd.westb.seetacloud.com:8443`（:6008）
  - 图：loss / grad_norm / lr / eval_recon（含基线 0.3318 虚线）；探针 `step_px_scale`(step1~5)、
    `prog_curve_255`、`E_px` step×region 热力图、`z_s` 统计；数据源状态表。
  - 源码：`tools/metrics_dash/`（commit `b819f7c`），45 s 准实时刷新，数据源缺失时按预期路径提示。

## 附 C: 后续根因与超参分析（同日完成）

- **根因排查（受控消融）**：`doc/2026-09-11/ANALYSIS_stack2x_collapse_rootcause.md`
  - 结论：不是纯参数量问题 / 不是 dropout / 不是实现 bug（四重证据：hook 计数、手写参考 maxdiff=0、
    `stack_dim=0` 逐位相同、权重审计 stack_in/out 未塌），而是 **`lr × 解码器宽度` 的超参交互**；
    触发变量是 **d_model 宽度**（仅 depth、仅 dropout、仅 heads 均不塌，去掉 dropout 照样塌）。
  - 失效时间线：step240（lr≈1.37e-4）灾难性更新 → 解码器幅度爆涨 → step290 **DINO register 先塌** →
    step330 pixel_head 输出被压平 → 梯度全消。
  - 修复佐证：峰值 lr **7.5e-5–1.0e-4** 不塌，且 `lr=7.5e-5` 的 600 步末 loss **0.5445 为 10 臂最低**
    （基线同口径 0.5723）⇒ **加宽本身有收益，只是不能沿用 1.5e-4**。
  - 诊断脚本/数据：`tools/diag_*.py`、`tools/diag_out/*.json`。
- **Step Law 超参评估**：`doc/2026-09-11/ANALYSIS_lr_batch_vs_steplaw.md`
  - `η_opt = 1.79·N^(−0.713)·D^(0.307)`、`B_opt = 0.58·D^0.571`（arXiv:2503.04715v7，含正文笔误更正）。
  - 放大 N 后 η_opt 比值 ×0.685（全模型口径）~×0.227（模块口径）⇒ 应降至 1.03e-4~3.4e-5；
    与实测安全边界 (1.0e-4, 1.5e-4] 吻合。**batch 与 N 无关，32 保持不变是对的。**
