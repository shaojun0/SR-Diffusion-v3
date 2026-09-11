# SR-Diffusion v3 学习率 / batch size 设计的 Step Law 评估

> 日期: 2026-09-11 ｜ 评估对象: `phase1_v2_stack2x_slice05`（`self.stack` 放大 2×）
> 论文: **Predictable Scale: Part I, Step Law** — [arXiv:2503.04715v7](https://arxiv.org/abs/2503.04715v7)（HTML v7: <https://arxiv.org/html/2503.04715v7>，PDF: <https://arxiv.org/pdf/2503.04715>）
> 官方仓库: <https://github.com/step-law/steplaw>
> 对照实验报告: `doc/2026-09-11/REPORT_v2_stack2x_slice05.md`；基线 `doc/2026-09-10/REPORT_v2_blockdiag_slice05.md`
>
> 方法说明: 本报告**未跑训练**（GPU 被根因排查子智能体占用）。所有论文公式/系数均**抓取原文核对**（arXiv HTML v7 正文，非记忆）；所有参数量均**从服务器 checkpoint 逐张量实测**；所有训练曲线均**从服务器日志重解析**。服务器仅只读访问，未改动任何文件。

---

## 1. TL;DR

| 问题 | 结论 | 置信度 |
|---|---|---|
| 本项目 `lr=1.5e-4` 本身是否合理？ | **按 Step Law 的绝对值口径：偏小**。全模型 N 口径下 η_opt ≈ **3.5e-4 ~ 5.1e-4**，实际 1.5e-4 只有它的 **0.30~0.43×**。但该绝对预测是 **D 外推 12 倍**（论文最小 D=2e9 token，本项目 D=1.7e8），**不可直接采信**。 | 低（绝对值） |
| 放大解码器后 lr 该不该改？ | **该改，方向明确是"调小"**。η_opt ∝ N^-0.713：N×1.70（全模型口径）→ η_opt×**0.685**；N×8.0（只看被放大的模块口径）→ η_opt×**0.227**。对应 lr 应从 1.5e-4 降到 **1.03e-4**（温和口径）或 **3.4e-5**（激进口径）。**"放大解码器但 lr 完全不变"确实违反 Step Law 的处方。** | 中（方向+比值） |
| batch=32 是否合理？ | **合理**。B_opt = 0.58·D^0.571 → **29,252 token/步 ≈ 47.8 张图**（本项目 D 口径），实际 32 张 = **0.67×B_opt**，落在论文自证的"宽最优平台"内。且论文用回归证明 **B_opt 与 N 无关**（log N 系数 −0.087, p=0.257, 95% CI [−0.241, 0.067]），**所以放大解码器时 batch 不该动、也确实没动 —— 这一步是对的**。 | 中高 |
| Step Law 对 step≈400 塌缩的**解释力** | **低–中（LOW–MEDIUM）**。<br>· **中**的部分：Step Law 的**比值**预测（η_opt×0.685，即安全上界下移 ~1.46×）与实测吻合 —— 基线在 1.5e-4 稳定跑完 8760 步，加宽模型在 lr≈**1.37e-4**（step 240，warmup 峰值 262 步前 22 步）就出现 grad_norm 尖峰 2.32 并永久失稳。<br>· **低**的部分：论文自身数据显示 lr 偏离最优 1.46× 只值 **+0.2~0.3% loss**（宽最优），而它的发散区在 **>3~4×** η_opt；而 Step Law 对加宽模型的**绝对** η_opt 预测（3.47e-4）**高于** 1.5e-4，按公式 1.5e-4 本来"应该安全"。**塌缩是结构性退化吸引子（grad→0.007 的梯度消失型塌缩），不是"lr 超最优"的典型形态。** | — |

**一句话**：**batch 不用改（论文说 B 只随 D 变，D 没变）；lr 应该改（η_opt ∝ N^-0.713），但 Step Law 只能给出"该降 1.5~4.4×"这个量级与方向，解释不了"为什么是硬塌缩而不是温和的 loss 变差"。**

---

## 2. Step Law 公式原文摘录（v7）

### 2.1 主公式（正文 Eq. (1)，第 1 节 Introduction；§3.4.1 Eq. (6) 同式）

> 摘自 §1 "Our main contributions"（arXiv:2503.04715v7）：
>
> "Step Law demonstrates that the optimal batch size exhibits a primary dependence on dataset size D, while the optimal learning rate manifests a joint dependence on both model parameters N and dataset size D:
>
> 　　　　**η(N,D) = 1.79 N^(−0.713) D^(0.307)**  　　(1)
>
> 　　　　**B(D) = 0.58 D^(0.571)**  　　　　　　　　(1)"

> 摘自 §3.4.1 Scaling Laws（Eq. (6)）：
>
> "the scaling law for hyperparameters can be described by the following power-law relationships:
>
> 　　　　η(N,D) = c·N^α·D^β, 　B(D) = d·D^γ  　(6)
>
> where c, α, β, d, and γ are constants... Notably, the proposed form B(D) assumes that the optimal batch size is independent of N. **This assumption is statistically validated through regression analysis in Appendix A.5.**"

### 2.2 系数表（§3.4.2 Table 2 "Fitted power-law coefficients for hyperparameter scaling laws"）

| Parameter | α | β | γ | c | d |
|---|---|---|---|---|---|
| Fitted value（论文 Table 2 原文） | **−0.713** | **0.307** | **0.571** | **1.79** | **0.58** |

**另外用仓库 `data/1004_fitted_lr_bs_scaling_model_parameters.csv`（1000 个 bootstrap 拟合）独立复核，与 Table 2 完全一致：**

| 量 | 论文 Table 2 | 仓库 1000-bootstrap 均值 | bootstrap 95% CI |
|---|---|---|---|
| `exp(lr_intercept)` = c | 1.79 | **1.7973** | — |
| α (lr_coefN) | −0.713 | **−0.7129** | [−0.7744, −0.6554] |
| β (lr_coefD) | 0.307 | **0.3075** | [0.2712, 0.3451] |
| `exp(bs_intercept)` = d | 0.58 | **0.5807** | — |
| γ (bs_coefD) | 0.571 | **0.5709** | [0.5340, 0.6055] |

> 仓库 README 明确写出拟合形式（与式(6)一致，自然对数/幂次）：
> `lr = exp(intercept) * N^coefN * D^coefD`；`bs = exp(intercept) * D^coefD`
> 并且**仓库里的 `bs` 是"每卡序列数"级，需 `bs*seq_len` 换算为 token 级**（`code/fit_tool.py` 第 266 行：`merged_data['bs'] = merged_data.eval("bs*seq_len") # change to token level`）。

### 2.3 单位与口径（Appendix A.1 Notation 原文）

> - "**D**: Dataset size in tokens."
> - "**N**: Number of **non-embedding** parameters in the model."
> - "**BS**: Batch size (**in tokens**)."
> - "**η(N,D)**: Optimal **peak** learning rate for a given parameter count N and dataset size D."
> - "**B(N,D)**: Optimal batch size (in tokens) for given parameter N and dataset size D."

**⇒ 单位约定（无任何隐藏归一化）：** N 用**非嵌入参数个数**（原始计数，不是百万）；D 用**训练 token 总数**（原始计数）；B 用**全局 batch 的 token 数/优化步**；η 是**峰值学习率**（cosine 调度里的 lr_max）。本报告即按此代入。

### 2.4 拟合域 / 有效范围（§3.4.1 原文）

> "For the parameter count N, we set up seven experiments spanning **60M, 120M, 210M, 270M, 430M, 540M, and 1B parameters**. ... we conducted experiments across five different data scales D: **2B, 4B, 8B, 20B, and 100B tokens**. Notably, we specifically reserved the **1B parameter and 100B token** settings as test points."

- 另有 Appendix A.3 Table 5（18 个 dense 配置）实测 N ∈ [2.15e8, 1.07e9]，D ∈ [4e9, 1e11]；Table 6（16 个 MoE 配置）N≈2.15e9，D ∈ [2e9, 2e10]。
- 训练总投入：3,700 个 LLM / ~100T token / ~1M H800 GPU·hour（Abstract）。

**⇒ 有效外推区间：N ≈ [60M, 1.07e9]（非嵌入参数）、D ≈ [2e9, 1e11]（token）。**

### 2.5 「宽最优」的原文（§3.3.1）

> "both the learning rate and batch size exhibit **convex relationships** with the training loss under fixed model parameters and dataset size conditions. ... we observe that the loss surface demonstrates a **stable region around the optimal configuration**, evidenced by the plateau-like behavior ... This stability provides **practical tolerance for small deviations** in hyperparameter selection while maintaining near-optimal performance."

> Abstract: "the hyperparameter landscape exhibits **convexity with a broad optimum** ... our estimated optima deviate from the global best performance found via exhaustive search by merely **0.094%** on the test set."

### 2.6 ⚠️ 论文内部一处**自相矛盾**（必须标注）

§3.4.1 正文写：*"As visualized in Fig. 5(b), we find that for each data scale D, the **optimal LR increases with model size N**."*

**但**公式(1)/Table 1/Table 2 的 N 指数是 **−0.713（负）**，即 η_opt **随 N 下降**。二者矛盾。

**判定：以公式和数据为准，正文那句话是笔误。** 三条独立证据：
1. **仓库 released 原始 grid 数据**（`data/dense_lr_bs_loss.csv`，1911 点）：固定 D=2e10 时，N=2.15e8 → lr_opt=3.91e-3，N=1.07e9 → lr_opt=1.38e-3，**随 N 下降**（5× N 对应 2.83× 降幅 = 0.353×；公式 5^−0.713 = 0.317×，吻合到 ~11%）。
2. **我对该 1911 点数据独立重拟合**（对 17 个 (N,D) 组取 argmin 后 OLS）：`lr = 30.1·N^(−0.824)·D^(0.288)`，N 指数同样为负；`bs = 3.42·D^(0.498)`（BS 只留 D）。重拟合指数 −0.824 vs 论文 −0.713 的差异来自 LR 网格是 2 倍步长的粗网格 + 只取 argmin，但**符号与量级一致**。
3. 论文自己 Table 1 列的对比方法（Microsoft Law `N^-0.23 D^-0.32`、Porian Law `3.7N^-0.36`）**也都是随 N 下降**；Table 3 的 6.5B MoE 预测值 2.12e-4 也低于 1B 模型在 D=1e10 时的 8.4e-4。

**这一条对本任务至关重要**：如果误信正文那句话，结论会 180° 反转（会认为"放大 N 应该调大 lr"）。**正确方向是调小。**

---

## 3. 代入本项目计算

### 3.1 参数量：服务器 checkpoint 逐张量实测（非日志抄录）

用 `safetensors.safe_open` 遍历 `checkpoint-2000/model.safetensors`，用 `torch.load` 遍历基线 `final_model.pt`：

| 模块 | 基线（2 层 / d1024 / ff4096） | 本次（4 层 / d2048 / ff8192） | 倍数 |
|---|---|---|---|
| `dinov2.encoder` | 302.359 M | 302.359 M | 1.00× |
| `dinov2.embeddings` | 2.007 M | 2.007 M | 1.00× |
| `dinov2.layernorm` | 0.002 M | 0.002 M | 1.00× |
| **DINO 小计** | **304.368 M** | **304.368 M** | 1.00× |
| `decoder.stack` | **33.593 M** | **268.591 M** | **8.00×** |
| `decoder.stack_in` + `stack_out` | —（Identity） | 2.099 + 2.098 M | 新增 |
| `decoder.query_base` | 0.590 M | 0.590 M | 1.00× |
| `decoder.pos_embed` | 0.037 M | 0.037 M | 1.00× |
| `pixel_head.net` | 3.304 M | 3.304 M | 1.00× |
| `special_bank`（token+pos） | 0.037 M | 0.037 M | 1.00× |
| **decoder 捆绑**（stack+query+pos） | **34.220 M** | **273.415 M** | **7.99×** |
| **非 DINO 小计** | **37.560 M** | **276.755 M** | **7.37×** |
| **总可训练参数** | **341.929 M** | **581.124 M** | **1.70×** |

> 注：任务简报里的"38M → 310M"与实测"37.56M → 276.76M"口径略有差别（可能把 `dinov2.embeddings` 算进了非 DINO 侧）；本报告一律用实测值。日志里打印的 "581.1M" 与实测 581.124M 一致 ✓。

### 3.2 D 的口径（token 数）

数据 `construction_site`：7009 训练图 / 3004 测试图，448×252 → patch 14 → **576 patch**；K = `derive_num_specials(576, [1,4,9,16,25]) = (isqrt(25)+1)²−1 = 35`（已在 `model_v2.py` 核对），故**每图 token = 1(cls) + 35(specials) + 576(patches) = 612** ✓。

| 口径 | 公式 | 数值 |
|---|---|---|
| **D1（主口径）** 编码器过一遍的 token | 7009 × 612 × 40 | **1.7158e8** |
| D2（仅承载 loss 的 patch token） | 7009 × 576 × 40 | 1.6149e8 |
| D3（若把 5 个采样步的解码器前向各算一遍） | 5 × D1 | 8.579e8 |
| 参考：论文拟合域下界 | §3.4.1 | 2e9 |

**D 该怎么算（方法论讨论）：**
- 论文的 D = "Dataset size in tokens" = **训练中每个 token 被消费一次的总数**。本项目里每个 token 就是一路进 DINOv2 encoder 的 patch/cls/register 单元，所以 **D1 是最贴近论文定义的**（612 token/图 × 40 遍）。
- **不应**用 D3 当主口径：论文的 D 数的是"数据 token"，不是"前向计算量"；5 个采样步是同一张图的重复前向，不是新数据。
- **更该担心的是 D 被高估**：612 个 token 里 576 个 patch 在空间上高度相关，且只有 **7009 张唯一图**（40 epoch 是重复采样）。论文的 D 隐含假设 token 近似可加地提供信息量；这里"有效独立 token 数"远低于 1.7e8。**而 β=+0.307>0，D 变小 ⇒ η_opt 更小 ⇒ 更该降 lr**，方向与下面的结论一致（见 §3.4 敏感性）。
- 用 BS 换算的自洽性检查：D1 / (8760 步 × 32 图/步 × 612) = 1.7158e8 / 1.7156e8 = **1.000** ✓。

### 3.3 代入结果（主口径 D = D1 = 1.7158e8，附 bootstrap 95% CI）

η_opt 用 1000 个 bootstrap 模型逐一代入取中位数与 2.5/97.5 分位：

| N 口径 | N_base | **η_opt(base)** | N_wide | **η_opt(wide)** | η_opt 比值 (N_w/N_b)^α | 实际 1.5e-4 ÷ η_opt(wide) |
|---|---|---|---|---|---|---|
| **(a) 全模型可训练参数**（论文口径） | 341.93 M | **5.05e-4** [4.27e-4, 5.94e-4] | 581.12 M | **3.47e-4** [2.89e-4, 4.11e-4] | **×0.685**（−31%） | **0.43×（偏小 2.3×）** |
| (b) 仅 `decoder.stack` | 33.59 M | 2.64e-3 [2.18e-3, 3.15e-3] | 268.59 M | 6.01e-4 [5.10e-4, 7.01e-4] | **×0.227**（−77%） | 0.25×（偏小 4.0×） |
| (b′) 仅 decoder 捆绑 | 34.22 M | 2.60e-3 [2.15e-3, 3.11e-3] | 273.42 M | 5.93e-4 [5.03e-4, 6.92e-4] | ×0.227 | 0.25× |
| (c) 非 DINO 全部 | 37.56 M | 2.44e-3 [2.01e-3, 2.90e-3] | 276.76 M | 5.88e-4 [4.99e-4, 6.87e-4] | ×0.241（−76%） | 0.26× |

**B_opt（与 N 无关）：**

| D 口径 | D | **B_opt (token/步)** | **B_opt（图/步，seq=612）** | 实际 32 图 ÷ B_opt |
|---|---|---|---|---|
| D1（主） | 1.7158e8 | **29,252** [24,779, 34,640] | **47.8** [40.5, 56.6] | **0.67×** |
| D2 | 1.6149e8 | 28,248 [23,877, 33,529] | 46.2 [39.0, 54.8] | 0.69× |
| D3 | 8.579e8 | 73,289 [65,414, 81,964] | 119.8 [106.9, 133.9] | 0.27× |

> 对照：反解"B_opt 恰好等于 32 图"的 D = **8.55e7 token**，正好是本项目 D1 的一半 —— 即本项目的 batch 相当于"按数据量减半后的最优 batch"，符合 0.67×。

**用官方工具复算（`code/fit_tool.py pred-opt-lr-bs`）得到同一结果**，可作为可复现证据：

```bash
python code/fit_tool.py pred-opt-lr-bs 3.4193e8 1.7158e8 612   # → lr 5.05e-4, bs 29252 (=47.8 imgs)
python code/fit_tool.py pred-opt-lr-bs 5.8112e8 1.7158e8 612   # → lr 3.47e-4, bs 29252 (=47.8 imgs)
python code/fit_tool.py pred-opt-lr-bs 2.6859e8 1.7158e8 612   # → lr 6.01e-4
```

### 3.4 D 口径敏感性（说明结论稳健性）

| 假设 D | D 值 | η_opt(wide, N=581M) | 实际/η_opt | B_opt（图） |
|---|---|---|---|---|
| 有效独立 token 极保守 | 1e7 | 1.452e-4 | **1.03×** | 9.4 |
| D2（仅 patch） | 1.61e8 | 3.410e-4 | 0.44× | 46.1 |
| D1（主） | 1.7158e8 | 3.474e-4 | 0.43× | 47.7 |
| D3（×5 采样步） | 8.58e8 | 5.694e-4 | 0.26× | 119.6 |
| 论文拟合域下界 | 2e9 | 7.384e-4 | 0.20× | 193.9 |

**关键性质：η_opt 的 N 比值 (N_w/N_b)^α 与 D 完全无关**，所以"放大解码器就该按 ×0.685（或 ×0.227）降 lr"这条结论**不受 D 口径争议影响** —— 这是本报告最稳的一条。绝对值的量级则对 D 敏感（1.45e-4 ~ 7.9e-4），所以不应拿绝对值当判据。

---

## 4. 关键判断：放大 N 后 η_opt 怎么变？能否解释塌缩？

### 4.1 定量：η_opt 随 N 的指数与倍数

由 Eq.(1)，η_opt ∝ **N^(−0.713)**（bootstrap 95% CI [−0.774, −0.655]）：

| N 增大倍数 | η_opt 倍数 | 说明 | lr 应从 1.5e-4 改为 |
|---|---|---|---|
| ×1.70 | **×0.685（−31%）** | 论文口径：总非嵌入参数 341.9M → 581.1M | **1.03e-4** |
| ×7.37 | ×0.241（−76%） | 非 DINO 部分 37.6M → 276.8M | 3.6e-5 |
| **×8.00** | **×0.227（−77%）** | 被放大的 `decoder.stack` 33.6M → 268.6M | **3.4e-5** |

**"放大解码器但 lr 不变"是否违反 Step Law？→ 是。** 论文的口径是**整模型**非嵌入参数，对应 **×0.685（−31%）**，即 lr 应从 1.5e-4 降到 ~1.0e-4。保持 1.5e-4 相当于在 η_opt 处方上**超出 1.46×**。

**多方法交叉印证（都指向"该降"，只在幅度上分歧）：**

| 迁移启发式 | 对本次改动（width×2, depth×2, ff×2）的 lr 倍数 | 出处/理由 |
|---|---|---|
| Step Law（论文口径，N×1.70） | **0.685** | 本报告 §3.3 |
| Step Law（模块口径，N×8.0） | 0.227 | 同上（N 口径误用，仅作上界参考） |
| μP 宽度律（lr ∝ 1/width） | 0.5 | width 1024→2048 |
| 经验共识区间 | **0.23 ~ 0.7** | 即 lr 该落到 **3.4e-5 ~ 1.0e-4** |

### 4.2 变化幅度足以解释塌缩吗？—— 需要拆成两个问题

**实测时间线（从 `/root/train_logs/stack2x_slice05.log` 逐条重解析，共 125 条记录）：**

| step | loss | grad_norm | lr | 判读 |
|---|---|---|---|---|
| 20 | 1.253 | 0.362 | 1.09e-5 | warmup |
| 100 | 1.069 | 0.235 | 1.03e-4 | **健康下降** |
| 200 | 1.056 | 0.327 | 1.14e-4 | **健康（基线同点 1.052）** |
| 220 | 1.046 | 0.551 | 1.25e-4 | **健康，本 run 最低点** |
| **240** | **1.136** | **2.321** | **1.368e-4** | **⚠️ grad 尖峰 + loss 跳升** |
| **260** | **1.227** | 0.555 | 1.483e-4 | **⚠️ loss 最高点** |
| 280 | 1.180 | 0.0650 | 1.5e-4（峰值） | warmup 结束（第 262 步）|
| 300 → 2500 | 1.14 ~ 1.17 冻结 | **0.0014 ~ 0.013** | 1.5e-4 → 1.26e-4 | **永久塌缩；eval_recon@2000 = 1.139** |
| 基线 step 400 | 0.6505 | 1.594 | — | 基线同点正常下降 |
| 基线 final | 0.4379（train） | — | — | **eval_recon 0.3318** |

**问题 A：「1.46× 的 lr 偏差」是不是死因？→ 大概率不是主因，但可能是扣扳机的那一下。**

1. **论文自己的数据说：1.46× 偏差只值 +0.2~0.3% loss。** 我用仓库 released 的 1911 点真实 grid（17 个 (N,D) 组，每组在最优 bs 上扫 lr）算出**有损容差曲线**（相对 loss 增量）：

| lr / lr_opt | 中位数 Δloss | 均值 | 最差 |
|---|---|---|---|
| 0.12× ~ 0.25× | +1.35% | +1.39% | +1.98% |
| 0.25× ~ 0.35× | +0.85% | +0.92% | +1.53% |
| 0.35× ~ 0.50× | +0.51% | +0.60% | +1.14% |
| **0.50× ~ 0.71×** | **+0.24%** | +0.29% | +0.76% |
| **0.71× ~ 1.00×** | **+0.13%** | +0.12% | +0.26% |
| 1.00× ~ 1.41× | +0.00% | +0.09% | +0.87% |
| **1.41× ~ 2.00×** | **+0.23%** | +0.44% | +1.35% |
| 2.00× ~ 2.83× | +0.98% | +1.33% | +3.27% |
| 2.83× ~ 4.00× | +2.56% | +2.93% | +6.43% |
| **4.00× ~ 8.00×** | **+8.19%** | +66.7% | **+183%（发散）** |

  ⇒ 论文的**宽最优平台在 [0.5×, 2×] 内几乎零代价**（±0.25%）；**发散区在 >3~4×**。**1.46× 落在平台内部**——按论文的凸性声明，它**不可能**单独把模型打进 hard collapse。
  ⇒ 反过来，**4.4×（模块口径处方）才落在论文数据里的发散区**。

2. **塌缩的形态不是"lr 过大"，而是"梯度消失型退化"。** lr 过大导致的是 grad_norm 爆掉 / loss 冲 NaN / 参数发散；这里实测是 **grad_norm 从 0.33 掉到 0.0014~0.013 并锁死**、loss 冻在 **1.15 ≈ 5 步累加的"常量输出"地板**（探针：5 个采样步输出完全相同、`z_s_within_std` 0.0197 vs 基线 0.0951、解码器+register 同步退化）。**这是退化吸引子（saddle / 表示塌缩），不是发散。**

3. **所以：Step Law 解释得了"为什么同样的 1.5e-4 在基线安全、在加宽后不再安全"（安全边界按 η_opt 同比下移 ~1.46×，实测失稳点 1.37e-4 < 1.5e-4，吻合）；解释不了"为什么失稳形式是硬塌缩"。**

4. **一个反向检验（说明绝对外推确实失效）**：若按容差曲线把"发散临界 ≈ 3~4× η_opt"反推，本次塌缩意味着加宽模型的**真实有效 η_opt ≲ 1.5e-4 / 3.5 ≈ 4.3e-5**。而 Step Law 两个口径都给出 **3.5e-4 ~ 6.0e-4**（高估 8~14×）。⇒ **绝对值外推在本项目（D 低 12 倍、ViT+L1 像素重建、部分结构）不可用**，这也是"解释力只能给低–中"的核心原因。

### 4.3 结论

> **η_opt 随 N 的指数 α = −0.713。N 增大 1.70× ⇒ η_opt ×0.685（−31%）；N 增大 8× ⇒ η_opt ×0.227（−77%）。"放大解码器但 lr 不变"确实偏离处方 1.46×（论文口径）到 4.4×（模块口径）。这个幅度足以让"基线安全、加宽不安全"（安全边界同步下移），但不足以在论文自己的宽最优/凸性图景里解释一次硬塌缩 —— Step Law 对塌缩的解释力评为「低–中」：方向与比值对得上，机理与绝对量级对不上。**

---

## 5. batch size 合理性分析

### 5.1 结论：`batch=32`（全局，2×16）**合理，且"放大解码器不动 batch"是正确决策**

1. **代入值**：B_opt(D1) = **29,252 token/步 = 47.8 图/步**（95% CI 40.5~56.6 图），实际 32 图 = **0.67×**。在论文的宽最优平台（±1.4~2×几乎零代价）内。
2. **B_opt 与 N 无关，论文有统计检验**（Appendix A.5）：
   > "The D-only formulation achieves nearly identical explanatory power as the full model (**R²=0.821 vs 0.823**), while the N-only model performs poorly (R²=−0.032)..."
   > Table 9: `log N` 系数 **−0.087**，std.err 0.075，**t=−1.158，p=0.257，95% CI [−0.241, 0.067]（跨 0）**；`log D` 系数 0.580，p<0.001。
   我在 released 数据上直接验证：**D=2e10 时，N=2.15e8 与 N=1.07e9（相差 5 倍）的最优 bs_tok 完全相同 = 524,288**。
   ⇒ **N 从 342M 变到 581M，B_opt 不变。所以"batch 保持 32"完全正确。**
3. **batch 太小导致梯度噪声大？** 定量依据：项目实际 **B/D = 19,584 / 1.7158e8 = 1.14e-4**；论文 B_opt/D = 0.58·D^(−0.429) = **1.71e-4**（D1 口径）。比值 0.67，即比最优小 1.5×，对应容差曲线的 "+0.24%" 档。**没有"batch 严重偏小"的问题。**
4. **batch 太大导致样本效率低？** 反方向同样不成立：要触发"过大"，需要 B ≫ B_opt（>2~3×），本项目是 0.67×，方向相反。
5. **可选微调**：若想正好踩在 B_opt 上，需 ~48 图/步。但显存实测 65.9/97 GB @ bs16/卡，48 图（24/卡）会逼近上限，且收益仅 ~0.2% loss 量级 —— **不值得动**。保持 32。

### 5.2 口径提示（诚实性）

若采用 D3 口径（把 5 个采样步计成 5 遍数据，8.58e8），B_opt 变成 ~120 图，则 32 图只有 0.27×，会落到 "+0.85~1.35% 损失" 档。**但 D3 口径本身不符合论文对 D 的定义**（同一张图的重复前向不是新 token）。所以主口径取 D1，结论是"合理"。**这是我给出 batch 结论时唯一实质性的不确定来源。**

---

## 6. 外推局限与置信度

### 6.1 论文的拟合域 vs 本项目（差多少倍）

| 维度 | 论文拟合域 | 本项目 | 偏离 |
|---|---|---|---|
| N（非嵌入参数量） | 60 M ~ 1.07 B | **341.9 M / 581.1 M**（全模型口径） | **在域内** ✓ |
| N（若只看解码器/新增模块） | 同上 | 33.6 M / 268.6 M | 下界外 1.8× ✗ |
| D（token） | **2e9 ~ 1e11** | **1.716e8** | **下界外 11.7×** ✗✗ |
| 架构 | decoder-only Transformer（dense + MoE） | **ViT(DINOv2-large) encoder + TransformerDecoder cross-attention + PixelHead** | 结构不同 ✗ |
| 损失 | 下一代 token **cross-entropy**（语言） | **像素 L1**，5 个采样步累加，block-diagonal 掩码 | 完全不同 ✗ |
| token 性质 | 文本 token，近似 i.i.d. | 576 patch/图，强空间相关；仅 7009 张唯一图 | 有效 D 远小于名义 D ✗ |
| 训练结构 | 全参数从头训练 | **DINOv2 预训练权重不冻结 + 新模块随机初始化**（混合初始化） | 不同 ✗ |
| 优化器/精度 | 未在论文中作为变量研究（H800 集群标准配方） | AdamW, wd=0.01, fp32, grad_clip=1.0 | 不同 |
| batch 单位 | token | 已按 612 tok/图 换算 | 可对齐 ✓ |

### 6.2 哪些能套、哪些不能套

**可以谨慎借用（置信度 中）：**
- **η_opt ∝ N^(−0.713) 的"比值/方向"**：因为它是**同 D 下不同 N 的比值**，D 的口径争议完全抵消；而且论文 §3.5 声称该指数在 dense/MoE/不同 shape/不同数据配方间是不变量（"topological invariance"，六个 430M 不同 shape 模型的最优 LR-BS 落在同一窄带）。**用它来指导"放大 N 后该怎么改 lr"，方向可靠。**
- **B_opt 只依赖 D、与 N 无关**（A.5 有回归 + 我在 released 数据上直接验证）：**用于"放大解码器时 batch 不需要动"，可靠。**

**不能套用（置信度 低）：**
- **η_opt 与 B_opt 的绝对值**：D 外推 11.7×，且损失/架构/初始化范式完全不同。§4.2 第 4 点已用实测反证：绝对预测高估 8~14×。
- **"最优区间很宽"的边界形状**：论文的凸性/宽最优是在它的损失-超参面上测的（且是 cross-entropy + 全参训练）。本项目在 lr≈1.37e-4 就出现硬塌缩，说明**真实面的结构（至少稳定性边界）与论文的宽最优图景不同**。
- **lr 与 batch 的联合最优**：论文的 η_opt 是在它自己的最优 BS 附近测的；本项目 batch 是 0.67×B_opt，lr-batch 联合面可能有轻微位移（论文的 η、B 二式是可分离形式，未建模交互项）。
- **论文未建模"稳定性/发散边界"**：Step Law 拟合的是 **loss 最优**，不是 **发散临界**。本项目踩的是后者。

### 6.3 置信度自评

| 结论 | 置信度 | 理由 |
|---|---|---|
| 论文公式与系数（1.79/−0.713/0.307, 0.58/0.571） | **高** | 原文三处（Eq.1、Table 1、Table 2）+ 仓库 1000-bootstrap CSV 完全一致 |
| §3.4.1 正文"optimal LR increases with N"是笔误 | **高** | 公式、Table 1/2、released 数据、我的独立重拟合（N^-0.824）四重印证为负指数 |
| 参数量/倍数（581.12M、decoder.stack ×8.00） | **高** | 直接遍历 checkpoint 张量 |
| D1 = 1.7158e8 及其口径合理性 | **中高** | 与论文 D 定义一致；但"有效独立 token"低于名义值是合理质疑 |
| "放大解码器该降 lr" + 幅度 1.5×~4.4× | **中** | 比值结论与 D 口径无关；幅度区间依赖 N 口径选择 |
| batch=32 合理、不该随 N 改 | **中高** | B 与 N 无关有论文回归 + 数据双证；绝对比值 0.67× 依赖 D 口径 |
| Step Law 对塌缩的解释力 = 低–中 | **中** | 比值与失稳点吻合（支持"中"）；形态与绝对量级不符（支持"低"） |
| 建议的具体 lr 数值（1.0e-4 / 7.5e-5） | **中低** | 基于比值处方 + 实测失稳点 1.37e-4 + 容差曲线；**必须靠 §7 短扫确认** |

---

## 7. 建议：下一次重跑

### 7.1 单点推荐（若只跑一次）

| 项 | 基线/本次 | **建议** | 依据 |
|---|---|---|---|
| **peak lr** | 1.5e-4 | **1.0e-4**（保守可取 **7.5e-5**） | Step Law 比值处方 ×0.685 → 1.03e-4；且实测失稳点为 1.37e-4，1.0e-4 留出 1.37× 余量 |
| **全局 batch** | 32（2×16） | **32 不变** | B_opt 只随 D 变，D 未变；A.5 证明与 N 无关 |
| warmup | ratio 0.03（262 步） | **ratio 0.05~0.06（438~526 步）** | 塌缩精确发生在 warmup 峰值附近第 240~262 步；拉长 warmup 让 Adam 二阶矩更稳。**这是独立于 Step Law 的工程保险** |
| grad_clip | 1.0 | 1.0 不变，但**加尖峰早停**（见 7.2 判据） | 本次 grad_norm 尖峰 2.32 未被 clip 拦住（clip 阈值 1.0 只限制步长，不防退化吸引子） |
| cosine `min_lr` | 衰减到 0（HF 默认） | 可选：`min_lr ≈ 1e-5` | 论文 §3.3.2 主张固定 `lr_min=1e-5` 优于 `lr_max/10`（避免高 peak 时末端 lr 偏大）。本项目 lr_max=1e-4 时 `lr_max/10` 恰好就是 1e-5，**影响很小，非必需** |
| dropout | 0 → **0.05（与 lr 同时改了）** | **建议单独消融**：先固定 lr=1.0e-4 跑 `dropout=0.05` vs `0.0` | 本次是**两个变量同时改**，dropout 是混淆项，必须解耦 |

### 7.2 短程 lr×batch 扫描方案（推荐执行）

**网格**：lr ∈ {5e-5, 7.5e-5, 1.0e-4, 1.5e-4} × 全局 batch ∈ {16, 32} = **8 个 run**，每个 **600 步**（覆盖 warmup 峰值并多跑 ~340 步，正好跨过本次塌缩点 240~262）。

- 成本：实测 1.96 s/it（2 卡）→ 单 run ≈ 20 min；8 run 串行 ≈ 2.7 h；若每 run 单卡并 2 路并行 ≈ **1.4 h**（bs16/卡 时 1 卡显存 65.9GB，97GB 上限内可行）。
- 其余超参全部锁定为：`warmup_ratio=0.03`（保留原设定以便与本次对比）、`wd=0.01`、`grad_clip=1.0`、`fp32`、`seed=42`、`slice 0:5`、`blockdiag`、`stack_dim=2048, depth=4, heads=16, dropout=0.05`（dropout 单独一轮再消融）。
- 加一个 **lr=1.5e-4 + 2 层 decoder（stack_dim=0）** 的对照 run 作为"健康参照"（600 步时应 ≈0.59）。
- 建议每 20 步（`log_every` 现值）额外记录 **DINO 侧 vs decoder 侧的 grad_norm 分组**，用于定位塌缩起源。

**判据（按优先级）**：

1. **硬失败（立即判负，不必跑完）**：`grad_norm < 1e-2` 连续 ≥3 条日志记录（即 ≥60 步）；或 loss 升破 step-200 的值且在 60 步内不回落。
2. **健康判据**：step 600 的 loss 必须 **低于 step 200 的 loss** 且仍单调下降；grad_norm 全程保持在 **0.1 ~ 3** 量级（本次健康段实测 0.23~1.7）。
3. **择优**：在存活者中取 step 600 loss 最低者；若两个 lr 接近（差 <0.5%），取**更小的 lr**（宽最优平台内，小 lr 的方差与风险更低）。
4. **确认**：用胜出配置跑到 step 2000 做一次 `eval_recon`，与基线同点 **0.486**、基线终点 **0.3318** 对比。若 step 600 已明显优于"1.5e-4 塌缩版"的 1.14，即可判定修复。

**预期**：最优落在 **7.5e-5 ~ 1.0e-4**；1.5e-4 复现塌缩；5e-5 健康但 600 步 loss 偏高（对应 §4.2 容差曲线 0.35~0.5× 档，约 +0.5% 长期代价）。若 **1.0e-4 也塌缩**，则说明死因**不是 lr**（或不仅是 lr），应转入 dropout / 初始化（`stack_in`/`stack_out` 的默认 kaiming 初始化 + 无归一化投影）/ 混合初始化范式的排查 —— 那正是根因排查子智能体的领域。

### 7.3 明确不做的事

- 不动 batch（论文与数据都支持"B 不随 N 变"）。
- 不因为"Step Law 预测 η_opt=3.5e-4 > 1.5e-4"就去**调大** lr —— 那是 D 外推 11.7× 的不可信绝对值。
- 不做全量 40 epoch 重跑，先 600 步筛选（本次塌缩在 400 步内即已判定，600 步足够）。

---

## 8. 引用

- 论文（v7, 2025-08-19）: Li, Zheng, Wang, Zhang, Wang, Xuyang, Fan, Ding, Wang, Ding, Zhou, Zhang, Jiang. *Predictable Scale: Part I, Step Law – Optimal Hyperparameter Scaling Law in Large Language Model Pre-training.* [arXiv:2503.04715v7](https://arxiv.org/abs/2503.04715v7) ｜ [HTML v7](https://arxiv.org/html/2503.04715v7) ｜ [PDF](https://arxiv.org/pdf/2503.04715) ｜ [abs](https://arxiv.org/abs/2503.04715)
  - Eq. (1) / Eq. (6) / Table 1 / Table 2 / §3.3.1 / §3.3.2 / §3.4.1 / §3.4.2 / §3.5.1 / Appendix A.1 / A.3 (Table 5, Table 6) / A.5 (Table 8, Table 9) / A.9
- 官方仓库与工具: <https://github.com/step-law/steplaw>
  - `code/fit_tool.py`（`pred-opt-lr-bs`，把 bs 由样本级 `bs*seq_len` 换算到 token 级）
  - `data/1004_fitted_lr_bs_scaling_model_parameters.csv`（1000 bootstrap 拟合系数）
  - `data/dense_lr_bs_loss.csv`（1911 点真实 grid，本报告 §4.2 容差曲线与 §2.6 方向验证的数据源）
- 项目内部: `doc/2026-09-11/REPORT_v2_stack2x_slice05.md`、`doc/2026-09-10/REPORT_v2_blockdiag_slice05.md`
- 服务器只读数据源: `/root/autodl-tmp/sr-diffusion-v3-stack2x/output/phase1_v2_stack2x_slice05/args.json`、`.../checkpoint-2000/model.safetensors`、`/root/autodl-tmp/sr-diffusion-v3-qmask/output/phase1_v2_block_slice05_blockdiag/final_model.pt`、`/root/train_logs/stack2x_slice05.log`、`/root/train_logs/blockdiag_slice05.log`

---

## 附：本报告用到的可复现命令

```bash
# 1) 抓论文原文（HTML v7）并抽公式
curl -sL -A "Mozilla/5.0" https://arxiv.org/html/2503.04715v7 -o steplaw.html

# 2) 官方仓库
git clone --depth 1 https://github.com/step-law/steplaw.git

# 3) 复核系数（1000 bootstrap 均值 = Table 2）
python -c "import pandas as pd,numpy as np; d=pd.read_csv('data/1004_fitted_lr_bs_scaling_model_parameters.csv');
print(np.exp(d.lr_intercept.mean()), d.lr_coefN.mean(), d.lr_coefD.mean(), np.exp(d.bs_intercept.mean()), d.bs_coefD.mean())"
# -> 1.7973, -0.7129, 0.3075, 0.5807, 0.5709

# 4) 实测参数量
ssh -p 38024 root@connect.westb.seetacloud.com \
  'export PATH=/root/miniconda3/bin:$PATH; python3 -c "from safetensors import safe_open; ..."'
# -> widened TOTAL 581.124M / decoder.stack 268.591M ; baseline TOTAL 341.929M / decoder.stack 33.593M
```
