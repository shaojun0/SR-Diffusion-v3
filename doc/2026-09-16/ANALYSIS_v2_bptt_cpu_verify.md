# CPU 侧独立验证：循环 carry detach / BPTT —— 机制复核与 −30.8% 的构成（2026-09-16）

> 本文是**离线、纯 CPU**（无 GPU / 无 DINOv2 权重 / 无数据集）的独立验证，对象 = ① 仓库自己的解码器代码 ② 同日已入库的原始推理 json。
> **不改任何训练口径、不改 `model_v2.py` / `train_v2.py`、不新增训练**；脚本与数据归档在本目录。
> 配套：[`REPORT_v2_bptt_vs_detach.md`](REPORT_v2_bptt_vs_detach.md)（24 步 detach vs BPTT 对照）、[`REPORT_v2_slice05_bptt.md`](REPORT_v2_slice05_bptt.md)（slice05 臂）、[`RESULTS_v2_bptt.md`](RESULTS_v2_bptt.md)（结果总账）。

- 代码版本：`main` HEAD `ea13add`（工作树 clean）
- 环境：Windows / Python 3.9 / torch 2.8.0+**cpu** / 8 线程
- 新增：`cpu_verify_decoder_facts.py`、`cpu_analyze_bptt_result.py`、`cpu_probe_bptt_toy.py`、`data/toy_bptt_cpu_results.json`、`data/bptt_readout_decomposition.json`

## 0. 结论（TL;DR）

1. **两条结构性事实在仓库自己的解码器上成立**（§1，数值实测）：① 裸加 carry + pre-LN 残差 ⇒ `out_t = t·query_base + ΣΔ`（零权重下 `max|out_t − t·q| = 0`，第 24 步恰为 24 倍）；② 现行代码下 `∂L_t/∂query_base = 0`（t≥1），`query_base` 只被 step1 塑造。
2. **−30.8% 可拆成两个独立效应**（§2，只用已入库 json 重算）：与 carry 无关的整体提升 **−19.2%** × 轨迹精修 **−14.0%** ≈ **−30.6%**（headline −30.8%）。即"打破后步坍缩"只是其中**较小**的那一半。
3. **CPU 微型复现台重现了主现象**（§3，合成数据、8 步）：detach ⇒ 轨迹增益 2.2~2.6%（平）；BPTT ⇒ 17.4~42.1%（降）；可训编码器时末步 **−37.8%**（真实 −30.8%）。**并给出真实 run 没做的那一刀**：冻结编码器时 BPTT 仍能把轨迹从 2.2% 修到 17.4% ⇒ 核心机制在**解码器侧信用分配**；但此时 step1 反而变差、净收益只剩 −9.2% ⇒ 真实 run 里"连 step1 都提升 19%"那一半**指向编码器侧，本文无法证实**（§4 给出判别实验）。
4. **"梯度扇出"是操作点依赖的算术原因**（§2.4）：detach 下每个被读到的键只进 1 个损失项，BPTT 下进 **9.49**（24 步）/ **2.50**（slice05）= 相差 **3.80 倍**。

## 1. 在仓库自己的解码器上证实的两条事实

脚本：`cpu_verify_decoder_facts.py`（CPU 数秒，不需要数据/权重）。取真实操作点 `K=N=576`（24 步）。

### 1.1 事实 A：读出特征随步号线性累积

`OutputQueryDecoder` 的 stack 全部置零后只剩残差恒等通路，第 t 步输出应恰为 `t·query_base`：

| 步 | `max\|out_t − t·query_base\|` |
|---|---|
| 1 | 0.000e+00 |
| 2 | 0.000e+00 |
| 6 | 2.980e-08 |
| 13 | 1.192e-07 |
| 24 | 4.768e-07 |

`|out_24| / |out_1| = 24.0x` ⇒ **`out_t = t·query_base + Σ_i Δ_i`**。carry 是裸加、递推到 PixelHead 之间没有任何 LayerNorm/投影（设计如此，见 `DESIGN_v2_recurrent.md`），于是"沿用上一步画布"（Δ≈0）在通路上是**免费解**。
可验证的预测（**本文未测**）：若尾部漂移有害，规范化 carry（LayerNorm / 门控）是候选干预。

### 1.2 事实 B：detach 下 `query_base` 只被 step1 塑造

现行代码 `Y = (self.query_base + Y).detach()` ⇒ t≥1 的查询含被 detach 的项：

| 步 | `\|∂L_t/∂query_base\|` |
|---|---|
| 1 | 1.13e-04 |
| 2 / 3 / 6 / 24 | **0** |

独立复现了 `REPORT_v2_bptt_vs_detach.md` §5 的判据（该文报 step3 = 0、BPTT 下 17.30）。含义：detach 时"每步恰收 1 份梯度"**不是中性性质**，而是把 24 步变成 24 个互不相干的平权子任务，且种子 `query_base` 只由读窗口最小那一步（0.7% 的键）决定。

## 2. 已入库原始数据的重新拆分（不新增实验）

脚本：`cpu_analyze_bptt_result.py`，输入 `data/bptt24_detach_infer_test.json` / `data/bptt24_bptt_infer_test.json` / `data/slice05_bptt_infer_test.json`。

### 2.1 headline 对读出点不敏感，但会把两个效应混在一起

| 读出点 | A detach | B BPTT | 相对 |
|---|---|---|---|
| step 1 | 23.6466 | 19.0960 | **−19.24%** |
| 最好步 | 23.6426 | 16.3768 | −30.73% |
| 全步均值 | 23.6592 | 16.6416 | −29.66% |
| 末步（= `F_hat`） | 23.7190 | 16.4219 | **−30.76%** |

### 2.2 乘法分解

```
与 carry 无关的部分 (step1 vs step1) : −19.24%
轨迹部分       (B 的首步→末步递降)   : −14.00%
乘积                                  : −30.55%   (headline −30.76%)
```

**step1 的查询就是 `query_base`，不含任何 carry** ⇒ 它的 −19.2% 不可能来自"后步精修"。

### 2.3 轨迹形状：detach 是定点，BPTT 是几何收敛

| 配置 | min | max | 展幅 | 首→最好 | 最好→末 |
|---|---|---|---|---|---|
| A detach | 23.6426 | 23.7190 | 0.0764（**0.32%**） | +0.02% | −0.32% |
| B BPTT | 16.3768 | 19.0960 | 2.7192（16.60%） | **+14.24%** | −0.28% |

B 的逐步增量几何衰减（比值 ≈0.4~0.7）：**首→次一步拿到 52.4%** 的总增益，前 2 次步进 73.6%、前 3 次 83.2%、前 4 次 88.8%、前 8 次 96.9%、前 16 次 100%；**第 19 步起转为轻微变差**（−0.0013 → −0.0150）。

⇒ "BPTT 补上了梯度通路"为真，但**递进分工仍远不充分**（与 `REPORT_v2_bptt_vs_detach.md` §3 一致）；且**两个 run 的尾部都在向上漂移**，而 `F_hat` 恰好取在这个尾部。

### 2.4 读窗口 / 累计读入 / 梯度扇出（操作点依赖的算术原因）

| 操作点 | K | 步数 | 总读入 token | 每键平均损失项数 detach → BPTT |
|---|---|---|---|---|
| 24 步全轨迹（A/B） | 576 | 24 | 577 | 1 → **9.49**（×9.5） |
| slice[0:5]（C） | 35 | 5 | 36 | 1 → **2.50**（×2.5） |

块窗口互不重叠 ⇒ detach 下每个键只被"读它的那一步"更新；BPTT 下还会被其后所有步经 carry 更新（精确 = `Σ_{j=1..|T|} (|T|−j+1)·size_j / Σ_j size_j`，`size_j` = 第 j 步的读窗口宽度）。24 步的逐步表（窗口/累计读入/A、B 逐步 L1）见 `data/bptt_readout_decomposition.json`。两条要点：

- **末步 t=576 的窗口退化为 1 个 token**（`K=N` 时 `min((k+1)²−1,K)` 被截断：末块本应 49 列，实际只剩 1 列），而 B 的末步仍拿到接近最优的 16.42 ⇒ **BPTT 学到的是"对画布迭代精修"，不是"逐步读入更多键"**。
- `F_hat` 取的正是这个退化的末步 ⇒ 指标口径值得复核（建议同时报最好步 / 全步均值）。

### 2.5 训练期 eval 曲线：不是"收敛更慢"，是形态变化

| | 区间改善（每 2000 步） | 相邻比值 | 几何外推极限 |
|---|---|---|---|
| A | −0.0697 / −0.0314 / −0.0132 | 0.45 / 0.42 | ≈0.383~0.411（按 r∈[0.2,0.7]） |
| B | −0.0595 / **−0.0729** / −0.0380 | **1.23** / 0.52 | —（比值 >1 ⇒ 非几何，中段在加速） |

A 的衰减是干净几何（比值恒定），外推到收敛仍 ≈0.38~0.41，比 B 实际的 0.288 差 25~33% ⇒ 在合理外推下**不能用"A 只是训得慢"解释**（不是严格证明：单 seed、未跑更长训练）。

## 3. CPU 微型复现台（四臂）

脚本：`cpu_probe_bptt_toy.py`；数据：`data/toy_bptt_cpu_results.json`。
配置：`N=K=64`、8 步（`[1,4,9,16,25,36,49,64]`）、`D=32`、编码/解码各 2 层 pre-LN、**裸加 carry + 块读窗口 + `mean_t L1` + 读数取末步**（全部照 `model_v2.py` 语义）；A 相 600 步 warm start（单步 + 全读；test L1 0.6498，平凡解 0.8101），B 相 900 步（lr 5e-4、bs 32、seed 42）。

| 配置 | step1 | 最好步 | 末步 | 轨迹增益 | 末步 vs detach |
|---|---|---|---|---|---|
| 冻结编码器 + detach | 0.5865 | 0.5735 | 0.5806 | **2.2%** | — |
| 冻结编码器 + BPTT | 0.6297 | 0.5202 | 0.5271 | **17.4%** | −9.2% |
| 可训编码器 + detach | 0.3737 | 0.3639 | 0.3773 | **2.6%** | — |
| 可训编码器 + BPTT | 0.3925 | 0.2273 | 0.2346 | **42.1%** | **−37.8%** |

（carry 位移 `mean|Y_t^pix − Y_{t−1}^pix|`、逐段耗时见 json；BPTT 臂的位移约为 detach 的 2.7 倍。）

**三条读数**：

1. **主现象重现**：detach 两臂几乎平（2.2~2.6%，与真实 A 的 0.02% 同型）；BPTT 两臂显著递降。可训编码器时末步 **−37.8%**，与真实 −30.8% 同量级。
2. **冻结编码器那一刀**：编码器固定时 BPTT 仍把轨迹从 2.2% 修到 17.4% ⇒ **"carry 变成精修算子"这条机制成立在解码器侧**，不需要编码器学习。但此时 BPTT 的 step1 反而变差（0.5865 → 0.6297），净收益只剩 −9.2%。
3. ⇒ **真实 run 的 −19.2% "整体提升"无法用 toy 解释**：toy 里 BPTT 从未让 step1 变好；真实 run 的 step1 提升只能来自编码器侧的读出梯度变宽（§2.4 的扇出 ×9.5）。**这是待验证假设，不是结论。**

## 4. 边界（不能说 / 待验证）

1. **toy 是合成数据 + 从零训练的小模型**，只能证明机制"充分/可复现"，**不能证明真实 run 的成因就是它**。
2. **"step1 的 −19% 来自编码器侧"仍是假设**。判别实验（最便宜、最判别的一刀）：**冻结 DINOv2 重跑 A/B** —— 差距塌掉 ⇒ 主因在编码器侧；差距还在 ⇒ 在解码器侧信用分配。
3. **第一版 toy 失败并已弃用**：不 warm start、直接从头训整条 pipeline 时，四臂全部停在平凡解（test L1 ≈ 0.81 = 预测均值），对 detach/BPTT **无判别力**。说明该类 toy 必须先给"编码器可用特征"的起点，否则优化困难会淹没变量。记录在此以免后人重复。
4. **toy 里两臂的 step1 都没有变好**，与真实 run 的 step1 −19.2% **方向相反** ⇒ 真实 run 里必然有 toy 未复现的因素（预训练编码器规模、24 步 vs 8 步、真实数据的空间结构）。
5. 结论建立在**单 seed（真实 run）+ 单 seed（toy）**上，无误差棒。
6. **不新增任何训练、不改仓库默认行为**（`model_v2.py` 仍是 detach 版）：§2 的数字全部可由已入库 json 重算，§1/§3 是 CPU 上的独立构造。

## 5. 复现

```bash
git checkout ea13add          # 或任何含 2026-09-16 文档的提交

# §1 两条结构性事实（CPU 数秒）
python doc/2026-09-16/cpu_verify_decoder_facts.py

# §2 只用已入库 json 重算（CPU 数秒）
python doc/2026-09-16/cpu_analyze_bptt_result.py

# §3 微型复现台：--quick 冒烟约 1 分钟；完整四臂 8 线程约 28 分钟
python doc/2026-09-16/cpu_probe_bptt_toy.py --quick
python doc/2026-09-16/cpu_probe_bptt_toy.py
```

## 6. 文件清单

| 文件 | 内容 |
|---|---|
| `cpu_verify_decoder_facts.py` | §1：零权重下 `out_t = t·query_base`；`∂L_t/∂query_base`（t≥1 恰 0） |
| `cpu_analyze_bptt_result.py` | §2：读数点 / 乘法分解 / 轨迹形状 / 梯度扇出 / eval 曲线外推 |
| `cpu_probe_bptt_toy.py` | §3：四臂微型复现台（含 `--quick` 冒烟） |
| `data/bptt_readout_decomposition.json` | §2 的结构化输出 |
| `data/toy_bptt_cpu_results.json` | §3 四臂原始结果（曲线 / carry 位移 / 耗时；含 provenance 注） |
| `data/bptt24_*_infer_test.json`、`data/slice05_bptt_infer_test.json` | 输入（同日已入库，**未改动**） |

## 7. 附：与本次验证直接相关的代码状态问题（**建议另文归档**）

验证 §1.2 时顺带确认了一件影响"基线"定性的事：**现行 detach 不是设计文档的默认值，而是一次重构的遗留**。

| 时间（2026-09-15） | 事件 |
|---|---|
| 10:01–10:18 | 循环架构引入（带全套开关）。`DESIGN_v2_recurrent.md` 写明 **`recurrent_detach=False`（默认）= 整条 BPTT**；`=True` 是"截断 BPTT（省显存/更稳），但**失去长程信用分配**"；并写明 detach 的原始动机是**并行架构**的"每步恰收 1 份梯度、避免 t=0 三角失衡"，**"循环架构建议试 false"** |
| 13:46 | `ac91173`「**架构变更**」（message 4 字、无正文）一次删掉并行路径 + 全部开关（`--recurrent*` / `--query_mask_mode` / `--loss_mode` / `--loss_decouple`）、把损失改成"直接预测"、并把 `(query_base + Y).detach()` 硬编码为唯一路径 —— 即文档标为"失去长程信用分配"的那一支 |
| 同一 commit | 其 README **仍在描述 `--recurrent_detach`（截断 BPTT）这个开关**（`git show ac91173:README.md` 有、`git show ac91173:model_v2.py` 无） |
| 当日晚些 | `033cb27` 才补注释，写明"代价（实测）"与"与设计文档 §2.4 记的默认**不一致**" |
| 09-16 | 本文与 `RESULTS_v2_bptt.md` 量化代价 |

**detach 原有的两个理由在 09-15 当天同时失效**：① "切累加恒等捷径"针对的是 `cumsum` 路径，而该路径随损失口径改成"直接预测"被整块删除；② "省显存"被实测否定（87153 MiB/卡、1.83 s/it **逐位相同**）。

⇒ 因此 `run A` 不适合当"架构的自然基线"；−30.8% 的合适表述是"**把设计文档本来要的默认值找回来**"。相关工作（**本文未做**）：`ac91173` 同时删掉的 `prefix`/`open` 读窗口、`fuse`、`step_embed` 等开关，哪些是已证伪的、哪些是没试完就被删的，文档没有交代。

⇒ 建议单开 `ANALYSIS_detach_default_drift.md` 归档（逐条 `git show` 证据），并补一条"语义静默变更"的防线：**自检里用行为层断言 `∂L_{t≥1}/∂query_base ≠ 0` 替换现在那条对 detach 不敏感的结构层检查**（`python model_v2.py` 打不打 patch 都 `ALL CHECKS PASSED`，见 `REPORT_v2_bptt_vs_detach.md` §5）。

> 相关文档：[`REPORT_v2_bptt_vs_detach.md`](REPORT_v2_bptt_vs_detach.md)（§3 逐步曲线、§4 显存/步速、§5 自检陷阱、§7 建议后续）、[`REPORT_v2_slice05_bptt.md`](REPORT_v2_slice05_bptt.md)（§3 `step_px_scale` 判据失效）、[`RESULTS_v2_bptt.md`](RESULTS_v2_bptt.md)（总账）、`doc/2026-09-15/DESIGN_v2_recurrent.md`（§2.4/§2.7 设计意图）。
