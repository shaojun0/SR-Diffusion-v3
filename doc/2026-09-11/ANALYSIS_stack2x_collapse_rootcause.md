# SR-Diffusion-v3 — `self.stack` 放大 2× 早期塌缩 **根因排查**（受控实验）

> 日期: 2026-09-11 ｜ 服务器: 2× RTX PRO 6000 (97GB) ｜ 每臂单卡 bs16, 600 步
> 被查代码（服务器运行副本）: `/root/autodl-tmp/sr-diffusion-v3-stack2x`（`model_v2.py` / `train_v2.py`）
> 基线代码: `/root/autodl-tmp/sr-diffusion-v3-qmask`；基线 checkpoint: `output/phase1_v2_block_slice05_blockdiag/final_model.pt`
> 本次塌缩 run: `output/phase1_v2_stack2x_slice05`（checkpoint-2000, 真实 args 见其 `args.json`）
> **本文未改动 `model_v2.py` / `train_v2.py` 任何一行**；所有诊断脚本为新增文件（见 §7）

---

## 0. 结论速览（TL;DR）

**"只把 `OutputQueryDecoder.self.stack` 参数量放大 2 倍"本身并不会导致塌缩；塌缩是"解码器 d_model 加宽（1024→2048）+ 本次 LR 调度（262 步 warmup 后峰值 lr=1.5e-4）"的交互失稳，是一个超参问题，不是纯参数量问题，不是 dropout 问题，也不是实现 bug。**

受控实验（同 seed/同数据序，仅改一个变量）给出的三条硬结论：

1. **同一份加宽代码、同一 seed、同样的 1.5e-4 峰值 lr，只把 warmup 从 262 步改成 18 步 → 完全不塌缩，且 600 步末 loss 0.5603（比基线同口径 0.5723 还低）。** 而 warmup=262 的臂在 step 240 出现 loss 尖峰、step 290 起 grad_norm 冻结在 1e-2 以下。⇒ 架构本身能训，是**调度×宽度**的交互。
2. **加宽里真正致命的是 d_model 宽度，不是 depth、不是 dropout、不是 heads**：`width only`（2048/2/16, F_wr）与 `width+heads8`（2048/2/8, W8_wr）都塌缩；`depth 2→4 + dropout 0.05`（0/4/8, E_wr）与 `baseline+dropout`（G_wr）都健康；**去掉 dropout 后（2048/4/16/0, C_wr）照样塌缩** ⇒ dropout 无关。
3. **把峰值 lr 按宽度缩放就修好了，且加宽后反而更强**：lr=1.0e-4（D1_wr, ×0.667）与 lr=7.5e-5（D_wr, ×0.5）在 warmup=262 下都不塌缩；其中 **D_wr 600 步末 loss 0.5445 是全部 10 臂最低**（基线 A_wr 0.5723）。
   Step Law 的相对处方 `η_opt ∝ N^-0.713`：全模型 341.9M→581.1M ⇒ ×0.685 ⇒ 1.5e-4×0.685=**1.03e-4**，与实测安全边界 (1.0e-4, 1.5e-4] 一致。

**失效机理（时间线实测，非猜测）**：warmup 把 lr 峰值推到 step 262；加宽后的模型在 step≈240、lr≈1.37e-4 时进入"尖锐相"，此时一次灾难性更新使**解码器输出幅度爆涨**（`Y_abs_mean` 1.0→12.5，`query_base` 范数 +102%），紧接着**DINO register 表示先塌**（`z_s` 跨 register std 0.183→0.0017，step 290），pixel_head 输出随后被压成逐 patch 常量（跨 patch std 0.0054→6.0e-5，91×），梯度随之消失 ⇒ 冻结在"预测全图均色"的常数吸引子（prog_curve≈64–69/255，loss≈1.20）。

---

## 1. 复现：短程实验必须先对齐 warmup

任务书的参考模板用 `--max_steps 600 --warmup_ratio 0.03`。注意代码里
`total_steps = args.max_steps or ...`，`warmup_steps = int(total_steps * warmup_ratio)`
⇒ 模板下 **warmup = 18 步**，而真实 run（`max_steps=8760, warmup_ratio=0.03`）是 **262 步**。
**用模板跑 B 臂不会塌缩**（见 §2 B 行）：loss@400=0.6250、grad 1.41、600 步末 0.5603，一切正常。

要复现真实塌缩，必须让 LR 轨迹在 step 240–262 附近与真实 run 一致。warmup 占 max_steps 的比例
反推：`262/600 = 0.4367`。用 `--warmup_ratio 0.4367` 即复现（Bwr）：

| 量 | 真实 run（2 卡, bs32, warmup262, 8760 步） | 短程 Bwr（1 卡, bs16, warmup262, 600 步） |
|---|---|---|
| lr≈1.37e-4 的 step | 240（grad_norm 尖峰 **2.321**） | 240（loss 1.048→1.170） |
| step 400 loss / grad_norm | **1.151 / 0.007** | 1.1603 / 0.00569 |
| step 500 loss / grad_norm | 1.147 / 0.005 | 1.1670 / 0.00578 |
| 之后 | 冻结 1.14–1.16 / 0.002–0.013 至 2500 步 | 冻结 ~1.16 / ~0.007 至 600 步 |
| `step_px_scale`（探针） | `[0.1814]×5` | `[0.0226]×5` |
| `prog_curve_255` | `[69.33]×5`（≈全图均色） | `[64.45]×5` |
| `z_s_within_std` | 0.0197（基线 0.0951） | **0.0017** |
| 块内 cos | `[0.996,0.999,0.998,1.000,0.992]` | `[1.0]×5` |

⇒ **600 步单卡即可稳定复现**（训练动力学 + 塌缩探针双重对齐）。这也是后续所有消融采用的调度。

---

## 2. 消融矩阵（每臂 600 步, 单卡 bs16, seed 42, grad_clip 1.0, wd 0.01）

统一配置除下表列出的变量外全部相同：`slice[0:5]`（K=35, steps [1,4,9,16,25]）、`query_mask_mode=blockdiag`、
`--eval_every 100000 --save_every 100000 --log_every 10`（不加 `--smoke`）、`mlp_ratio 4.0`。
`warmup_ratio*600` 给出 warmup 步数。判据：**末段 grad_norm<1e-2 且末段 loss 高于前 100 步均值 = 塌缩**。

| 臂 | stack_dim | depth | heads | dropout | lr | warmup | loss@400 | grad@400 | loss@500 | grad@500 | best loss | 末段 mean grad | 塌缩 | 日志 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **A_wr** 基线架构 | 0 | 2 | 8 | 0 | 1.5e-4 | 262 | 0.7610 | 2.900 | 0.6090 | 1.176 | 0.5723@590 | 1.200 | ❌ | `/root/train_logs/stack2x_diag_A_wr.log` |
| **B** 完整改动·**短 warmup** | 2048 | 4 | 16 | 0.05 | 1.5e-4 | **18** | 0.6250 | 1.409 | 0.5952 | 1.537 | 0.5603@590 | 0.868 | ❌ | `/root/train_logs/stack2x_diag_B.log` |
| **Bwr** 完整改动·对齐 warmup | 2048 | 4 | 16 | 0.05 | 1.5e-4 | 262 | **1.1603** | **0.0057** | **1.1670** | **0.0058** | 1.0552@220 | **0.0072** | ✅ | `/root/train_logs/stack2x_diag_Bwr.log` |
| **C_wr** 完整 − dropout | 2048 | 4 | 16 | **0** | 1.5e-4 | 262 | **1.1609** | **0.0050** | **1.1681** | **0.0059** | 1.0556@190 | **0.0073** | ✅ | `/root/train_logs/stack2x_diag_C_wr.log` |
| **D_wr** 完整·**lr 7.5e-5** | 2048 | 4 | 16 | 0.05 | **7.5e-5** | 262 | 0.6374 | 1.179 | 0.5680 | 0.885 | **0.5445@590** | 0.597 | ❌ | `/root/train_logs/stack2x_diag_D_wr.log` |
| **D1_wr** 完整·**lr 1.0e-4** | 2048 | 4 | 16 | 0.05 | **1.0e-4** | 262 | 0.7301 | 1.903 | 0.6141 | 1.257 | 0.5721@590 | 0.723 | ❌ | `/root/train_logs/stack2x_diag_D1_wr.log` |
| **E_wr** 仅 depth 2→4(+dropout) | **0** | 4 | 8 | 0.05 | 1.5e-4 | 262 | 0.6551 | 1.926 | 0.6298 | 0.986 | 0.5854@590 | 0.492 | ❌ | `/root/train_logs/stack2x_diag_E_wr.log` |
| **F_wr** 仅 width/heads | 2048 | **2** | 16 | 0.05 | 1.5e-4 | 262 | **1.1607** | **0.0052** | **1.1676** | **0.0059** | 1.0454@230 | **0.0085** | ✅ | `/root/train_logs/stack2x_diag_F_wr.log` |
| **G_wr** 仅 dropout | **0** | 2 | 8 | 0.05 | 1.5e-4 | 262 | 0.8290 | 2.612 | 0.6427 | 0.351 | 0.6050@590 | 0.334 | ❌ | `/root/train_logs/stack2x_diag_G_wr.log` |
| **W8_wr** 仅 width(去 heads 干扰) | 2048 | 2 | **8** | **0** | 1.5e-4 | 262 | **1.1611** | **0.0060** | **1.1685** | **0.0082** | 1.0704@220 | **0.0104** | ✅(软) | `/root/train_logs/stack2x_diag_W8_wr.log` |

> W8_wr 末段 grad 均值 0.0104 刚好压在 1e-2 判据线上（step 400 为 0.0060），loss 在 step 220 触底 1.0704 后回升并冻结 —— 与 Bwr/C_wr/F_wr 是同一个吸引子，判为塌缩。

**读法（单变量结论）**

- 参数量：E_wr 增加 depth（参数更多）**不塌** ⇒ 不是"参数变多"本身。
- dropout：G_wr（基线+dropout）不塌；**C_wr（完整−dropout）照样塌** ⇒ dropout 既非充分也非必要。
- depth：E_wr（仅 depth 2→4 + dropout）不塌，甚至比基线略好 ⇒ depth 无关。
- heads：F_wr(16 头) 与 W8_wr(8 头，head_dim 128→256) 都塌 ⇒ heads 无关。
- **width**：F_wr / W8_wr（仅 d_model 1024→2048，含 `stack_in/stack_out` 与 dim_feedforward 4096→8192）稳定塌缩 ⇒ **触发变量 = `stack_dim` 加宽**。
- **lr**：Bwr(1.5e-4) 塌；D1_wr(1.0e-4)、D_wr(7.5e-5) 不塌 ⇒ 峰值 lr 是可操作的关键变量，**加宽后的临界峰值 lr ∈ (1.0e-4, 1.5e-4]**。
- **调度**：B(短 warmup, 同 lr 1.5e-4, 同架构) 不塌 ⇒ 光有 lr 数值还不够，**"lr 何时达到峰值"同样是触发条件**。

**汇总产物**：`/root/train_logs/stack2x_diag_summary.json`（含每臂逐 10 步曲线与每步 trainer_state 精确 step），
生成脚本 `tools/diag_collect.py`（优先读 checkpoint 的 `trainer_state.json`，避免 stdout 块缓冲导致的 step 错配）。

---

## 3. 是不是实现 bug？—— 不是（静态 + 数值双证据）

被查代码（服务器 `model_v2.py`）：

```python
423:  self.stack_dim = int(dim if not stack_dim else stack_dim)
424:  self.stack = nn.TransformerDecoder(
425:      nn.TransformerDecoderLayer(
426:          d_model=self.stack_dim, nhead=heads,
427:          dim_feedforward=int(self.stack_dim * mlp_ratio),
428:          dropout=dropout, activation="gelu", batch_first=True,
429:          norm_first=True),
430:      num_layers=depth)
432:  self.stack_in  = (nn.Identity() if self.stack_dim == dim else nn.Linear(dim, self.stack_dim))
434:  self.stack_out = (nn.Identity() if self.stack_dim == dim else nn.Linear(self.stack_dim, dim))
...
460:  Y = self.stack(self.stack_in(Y), self.stack_in(A),
461:                 memory_mask=mask, tgt_mask=tgt_mask)
462:  Y = self.stack_out(Y)
```

### 3.1 静态核对

| 检查项 | 结论 |
|---|---|
| `stack_in` 是否只作用一次 | ✅ 每个张量恰一次：`Y`(查询) 一次、`A`(memory) 一次，共 2 次/前向；`Y` 与 `A` 用同一投影（与基线"两者同空间"的语义一致） |
| 输出是否只过一次 `stack_out` | ✅ 1 次/前向 |
| 是否重复投影 / 维度顺序是否错 | ✅ 无 |
| 掩码形状是否与基线一致 | ✅ `memory_mask`=(2880, 36)=(\|T\|·N, K+1)，`tgt_mask`=(2880, 2880)=(\|T\|·N,\|T\|·N)，blockdiag 生效；掩码由未改动的 `build_block_mask`/`build_causal_query_mask` 生成 |
| `dim_feedforward` 语义 | ✅ `int(stack_dim*mlp_ratio)`，`mlp_ratio` 语义不变 |
| `Identity` 分支 | ✅ `stack_dim=0` 与 `stack_dim=dim=1024` 二者 state_dict 键/值逐位相同、forward 输出 maxdiff **0.0** |

### 3.2 数值核对（`tools/diag_impl_check.py`，输出 `/root/train_logs/stack2x_impl_check.json`）

1. **forward 计数（forward hook）**：`stack_in` 4 次 / `stack_out` 2 次 —— 该脚本共跑了 2 次 decoder 前向 ⇒ **每次前向 `stack_in`×2、`stack_out`×1**，与文档语义完全一致。
2. **手写参考实现 vs 模型 forward**：按 docstring 语义独立重写 `stack_out(stack(stack_in(Y), stack_in(A), memory_mask, tgt_mask))`，与 `OutputQueryDecoder.forward` 输出 **maxdiff = 0.0（逐位一致）** ⇒ 没有隐藏的额外算子、没有双重投影。
3. **初始化不是常量输出**（同 seed、同 batch，DINO 侧逐位相同 `z_cls/z_s` maxdiff=0.0）：

   | | `Y` 跨 patch std | `F_hat` 跨 patch std | `Y_abs_mean` | loss |
   |---|---|---|---|---|
   | 基线 (0/2/8/0) | 0.02052 | 0.02475 | 1.636 | 1.662 |
   | 加宽 (2048/4/16/0.05) | 0.00765 | 0.00915 | 1.073 | **1.347** |

   ⇒ 加宽模型 init **不是**常量函数（std 0.0077≠0），loss 反而更低。**没有"初始化即常量"的实现/初始化 bug。**
   （可注意的现象：init 时加宽模型输出跨 patch 变化只有基线的 ~1/2.7、更接近"均色解"，这降低了塌缩后的逃逸余量，但不是触发原因——同一 init 在短 warmup 下训到 0.5603。）

4. **逐模块 grad norm（init，同一 batch，clip 前）**：加宽模型各模块梯度量级只是基线的 ~0.5×，没有模块在 init 就"死"（`stack_in` 0.252、`stack_out` 0.348、`stack.L0..L3` 0.13–0.19、`pixel_head` 1.21、`dinov2.L0` 0.050）。⇒ **init 无死梯度**。

5. **checkpoint 权重审计（塌缩 ckpt = `output/diag_Bwr/final_model.pt`）**：

   | 权重 | init ‖W‖ | ckpt ‖W‖ | 相对变化 | cos(init,ckpt) |
   |---|---|---|---|---|
   | `decoder.stack_in.weight` | 26.131 | 26.175 | 0.062 | 0.998 |
   | `decoder.stack_out.weight` | 18.486 | 18.459 | 0.101 | 0.995 |
   | `decoder.query_base` | 15.373 | 22.013 | **1.021** | 0.701 |
   | `decoder.pos_embed` | 3.852 | 3.885 | 0.124 | 0.992 |
   | `special_bank.pos` / `.token` | 3.806 / 0.643 | 3.822 / 0.639 | 0.062 / 0.042 | 0.998 / 0.999 |

   ⇒ **`stack_in`/`stack_out` 根本没塌**（6–10% 变化、cos≈0.995–0.998，范数不缩不爆）；变化最大的是 `query_base`（范数 +102%）。
   **"新加投影缩到 0/变常量"这一假设被证伪。**

### 3.3 决定性反证

**同一份代码、同一个 600 步短程口径，只换 warmup（B vs Bwr）就从"正常训练到 0.5603"变成"冻结在 1.16"**。
实现 bug 不会对 LR 调度形状如此敏感；而且 §3.2 的 forward 逐位一致性检查已经排除语义错误。
⇒ **实现正确；问题在超参（lr）×架构（宽度）的交互。**
（附带好处：加宽路径的 `stack_dim=0` 默认分支与历史 `state_dict`/forward 逐位一致，旧 checkpoint 双向可载。）

---

## 4. 塌缩方向：谁先死（逐 10 步时间线）

脚本 `tools/diag_timeline.py`（自包含最小训练循环，严格复制 `train_v2.py` 配方：AdamW(0.9,0.999,eps1e-8,wd0.01) +
`get_scheduler("cosine",warmup,total)` + `clip_grad_norm_(1.0)` + 先 `opt.step()` 后 `sched.step()`），
配 Bwr 架构 + `warmup=262/600`，每 10 步在**固定探针 batch**上记录 `z_s`/解码器输出/逐模块梯度。
原始数据：`/root/train_logs/stack2x_timeline_Bwr.json`，日志 `/root/train_logs/stack2x_timeline_Bwr.log`。

| step | lr | train loss | grad_raw | `z_s`跨register std | `z_s`跨图 std | `Y`跨patch std | `Y_abs_mean` | `F_hat`跨patch std | probe loss | `gn_dino.L0` | `gn_special_bank` | `gn_pixel_head` | `gn_stack_out` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 1.397 | 1.462 | 0.103 | 1.077 | 0.0077 | 1.04 | 0.0091 | 1.415 | 0.0397 | – | 0.964 | 0.267 |
| 100 | 5.7e-5 | 1.094 | 0.462 | 0.189 | 0.250 | 0.0073 | – | 0.0053 | 1.142 | 0.0135 | – | 0.408 | 0.095 |
| 200 | 1.14e-4 | 1.020 | 0.418 | 0.092 | 0.073 | 0.0096 | 1.45 | 0.0052 | 1.107 | 0.0193 | 0.0176 | 0.371 | 0.069 |
| **240** | **1.368e-4** | **1.170** | 0.272 | 0.132 | 0.076 | 0.0218 | 2.09 | 0.0155 | 1.126 | 0.0053 | 0.0030 | 0.255 | 0.058 |
| 250 | 1.426e-4 | 1.058 | 0.269 | 0.123 | 0.091 | 0.0411 | – | 0.0387 | 1.099 | 0.0072 | 0.0037 | 0.249 | 0.052 |
| 260 | 1.483e-4 | 1.228 | 0.456 | 0.055 | 0.037 | 0.0358 | – | 0.0320 | 1.214 | 0.0113 | 0.0071 | 0.382 | 0.126 |
| 270 | **1.498e-4** | 1.129 | 0.790 | 0.220 | 0.327 | 0.0546 | – | 0.0391 | 1.198 | 0.0710 | 0.0825 | 0.738 | 0.161 |
| 280 | 1.491e-4 | 1.192 | 0.753 | 0.041 | 0.179 | **0.0660** | **3.62** | 0.0392 | 1.182 | 0.0096 | 0.0072 | 0.737 | 0.129 |
| **290** | 1.477e-4 | 1.154 | 0.605 | **0.0058** | 0.036 | 0.0328 | **6.96** | 0.0085 | **1.201** | **0.0002** | **0.0001** | 0.597 | 0.097 |
| 300 | 1.456e-4 | 1.147 | 0.736 | 0.0052 | 0.038 | 0.0376 | – | 0.0065 | **1.204** | 0.0009 | 0.0004 | 0.725 | 0.117 |
| 310 | 1.430e-4 | 1.113 | 0.998 | 0.0062 | 0.054 | 0.0419 | – | 0.0017 | **1.205** | 0.0076 | 0.0007 | 0.989 | 0.118 |
| **320** | 1.397e-4 | 1.119 | **0.0080** | 0.0060 | 0.049 | 0.0440 | – | 0.0015 | 1.204 | 0.0000 | 0.0000 | **0.0078** | **0.0020** |
| 330 | 1.359e-4 | 1.119 | 0.109 | 0.0062 | 0.049 | 0.0409 | **9.98** | **0.00004** | 1.204 | 0.0000 | 0.0000 | 0.108 | 0.016 |
| 400 | 9.70e-5 | 1.113 | 0.0044 | 0.0055 | 0.054 | 0.0356 | – | 0.00004 | 1.202 | 0.0000 | 0.0000 | 0.0043 | 0.0010 |
| 600 | ~0 | 1.209 | 0.0091 | 0.0055 | 0.056 | 0.0356 | **12.47** | 0.00009 | 1.201 | 0.0000 | 0.0000 | 0.0091 | 0.0011 |

**方向结论（按时间先后）**

1. **触发事件：step 240, lr=1.368e-4 的一次灾难性更新** —— loss 由 1.048 跳到 1.170（真实 2 卡 run 于同一 step 出现 grad_norm 尖峰 2.321）。
2. **先"爆"的是解码器幅度，不是先"死"**（step 250–280）：`Y` 跨 patch std 0.015→0.066、`Y_abs_mean` 2.09→3.62、逐模块 grad 反而冲高（`special_bank` 0.0825、`dino.L0` 0.0710、`pixel_head` 0.738）。这是 **loss 上升型发散**，不是梯度消失。
3. **第一个塌掉的表示是 DINO register 分支**（step 290）：`z_s` 跨 register std 0.0406→**0.0058** 并永久钉死；同一步 `gn_dino.L0` 0.0096→**0.0002**、`gn_special_bank` 0.0072→**0.0001** —— **DINO 侧梯度先死**，而 `pixel_head`(0.597)/`stack_out`(0.097) 仍有大梯度。
4. **pixel_head 输出随后被彻底压平**（step 330）：`F_hat` 跨 patch std 0.0017→**4e-5**（最终 91× 塌缩），probe loss 从 step 290 起逐位冻结在 1.201–1.205（= 常数预测器的损失）。
5. **全部梯度消失**（step 320 起）：连 `pixel_head`/`stack_out` 也降到 1e-3 以下。
6. 结果：`Y` 仍带 ~0.036 的 patch 间差异、但幅度涨到 `Y_abs_mean`≈12.5（`query_base` 范数 +102%），pixel_head 把它压成逐 patch 常量 ⇒ 5 个采样步输出**逐位相同**（`step_px_scale=[0.0226]×5`，`E_px` 5 行相同）。

**来源归因（2×2 交叉换权重，`tools/diag_attribution.py`，`/root/train_logs/stack2x_attribution.json`）**：
把 "DINO 权重" 与 "special_bank(token/pos)" 各自换回初始值：

| 组合 | `z_s` 跨 register std |
|---|---|
| DINO_ck + sp_ck（塌缩模型） | 0.00180 |
| DINO_**init** + sp_**init** | **0.18337** |
| DINO_**init** + sp_ck | **0.19164**（健康） |
| DINO_ck + sp_**init** | **0.00164**（塌缩） |

`special_bank.pos` 跨 register std 0.01996→0.02004、范数 3.806→3.822（几乎未动）；
DINO 权重整体相对变化仅 **1.68%**。⇒ **register 塌缩 100% 归因于 DINO 权重被那一次发散更新打坏，而不是 special_bank**。
同时 `dY(shuffle z_s registers)`：健康 z_s 时 0.223、塌缩 z_s 时 0.0040 ⇒ memory 条件化实质失效。

---

## 5. 修复假设（均有短程实验佐证）

| 修复 | 证据 | 建议 |
|---|---|---|
| **① lr 按宽度/参数量缩放（首选）** | Bwr 1.5e-4 塌；**D1_wr 1.0e-4（×0.667）不塌**（末 0.5721）、**D_wr 7.5e-5（×0.5）不塌且末 loss 0.5445 全场最低**（优于基线 0.5723） | 加宽 2× 时峰值 lr 取 **7.5e-5–1.0e-4**。Step Law 相对处方 ×0.685→1.03e-4，与实测边界一致；batch 无需改（仍在 B_opt 平台内） |
| **② 缩短/重塑 warmup，别让 lr 峰值落在"尖锐相"** | **B（同架构，warmup 18）不塌，600 步末 0.5603**，优于基线 0.5723；而 warmup 262 的 Bwr 塌 | 若保持 lr=1.5e-4，应显著缩短 warmup（如 warmup_ratio ≤0.05，即峰值在 step≲40 出现）；等价地可用"先高后降"的调度 |
| **③ 保留 depth 加宽，只放弃 width（若不改 lr）** | **E_wr（0/4/8/0.05，仅 depth 2→4）健康且略优于基线**（0.5854 vs 0.5723）；depth 增加参数量但不塌 | 若必须不动 lr，则"加 depth 不加 width"是安全的容量提升路径 |
| **④ dropout 不是问题，可保留** | G_wr（基线+0.05）健康；C_wr（完整−dropout）照样塌 | dropout 0.05 可继续用（可能还有正则收益），不必为塌缩背锅 |
| **⑤ 不建议改 `stack_in/stack_out` 初始化或 `dim_feedforward`** | 权重审计显示两者**根本没塌**（6–10% 变化）；init 输出也不是常量；唯一显著变化是 `query_base` 范数 +102%（是发散的**结果**而非原因） | 无需为"混合初始化范式"投入；先做 ①/② |
| **⑥ 早停守卫（工程兜底）** | 塌缩前 5 步内 loss 由 1.05→1.17、lr 逼近峰值 | 训练脚本加：若前 ~500 步内出现 `grad_norm<1e-2 且 loss 反升超过前 100 步均值`，判定塌缩、立即停止 |

**推荐组合**：`stack_dim=2048 + depth=4 + heads=16 + dropout=0.05 + 峰值 lr≈1.0e-4`（或 7.5e-5）。
按本次 600 步短程，它在 step 590 达到 0.5721–0.5445，**同时低于基线同口径 0.5723** ——
即"加宽解码器"本身是有收益的，只要 lr 不沿用为小解码器调的 1.5e-4。

---

## 6. 置信度与局限

**置信度：高（对"触发变量"与"非实现 bug"）；中（对精确临界 lr）。**

- 高置信的依据：同 seed、同数据序、单变量、每臂 600 步；B vs Bwr 只差一个 warmup 得到"健康/塌缩"二元结果；F_wr/W8_wr vs E_wr/G_wr 把 width、depth、dropout、heads 一一分离；实现层有 hook 计数 + 手写参考逐位一致 + init 统计 + 权重审计四重证据；短程 run 与真实 2 卡 run 在 loss/grad 轨迹和探针指标上双重对齐。
- 局限：
  1. **短程 600 步 ≠ 全程 8760 步**。绝对 loss 不可与完整训练对比（本报告只做臂间相对比较）；"塌缩/健康"判据基于 600 步内的 grad_norm 量级与 loss 是否回升。
  2. **单卡 bs16 vs 真实 2 卡 bs32**：梯度噪声更大，塌缩可能更早/更易发生；但 B/Bwr 同口径对照抵消了该差异，且 Bwr 复现了真实 run 的 step 240 尖峰与 step 400 的 grad≈5e-3。
  3. **只跑了 seed 42**（塌缩是该 seed 下的确定性结果，未做多 seed 稳定性统计）；"临界 lr∈(1.0e-4,1.5e-4]"是单点 bracketing，未做细扫（如 1.2e-4/1.3e-4）。
  4. 短程实验用 `max_steps=600` 改变了 cosine 的**尾部**（真实 run 的 lr 在 step 262 后近似恒定在 1.5e-4，短程 run 会继续衰减）。但塌缩发生在 step 240–330、lr 仍接近峰值处，两者重合，故结论不受尾部差异影响。
  5. `diag_impl_check.py` 的 DINO 相对变化子检查曾因 key 前缀（`dinov2.`）比对写错而输出 0；正确值取自 `diag_attribution.py` 的 `dino_rel_change=0.0168`（已修正脚本并重跑）。
  6. Step Law 的**绝对** η_opt 预测（加宽后 3.47e-4）高于 1.5e-4，据此单看无法解释"硬塌缩"；本报告只采信其**相对**处方方向（N↑ ⇒ lr↓），并用实验独立标定了安全区间。

---

## 7. 产物与复现

### 7.1 远端原始产物

| 类型 | 路径 |
|---|---|
| 每臂训练日志 | `/root/train_logs/stack2x_diag_{A_wr,B,Bwr,C_wr,D_wr,D1_wr,E_wr,F_wr,G_wr,W8_wr}.log` |
| 每臂输出目录（含 `args.json` / `model_info.json` / `checkpoint-600/trainer_state.json`） | `/root/autodl-tmp/sr-diffusion-v3-stack2x/output/diag_<arm>/` |
| 消融汇总（含逐步曲线） | `/root/train_logs/stack2x_diag_summary.json` |
| 时间线（谁先死） | `/root/train_logs/stack2x_timeline_Bwr.json`, `.log` |
| 实现核对 | `/root/train_logs/stack2x_impl_check.json`, `.log` |
| 塌缩来源归因 | `/root/train_logs/stack2x_attribution.json`, `.log` |
| 塌缩探针（短程 Bwr） | `/root/train_logs/probe_stack2x_Bwr_diag.json`, `.log` |

### 7.2 新增诊断脚本（本地仓库 `SR-Diffusion-v3/tools/`，同步到服务器 `stack2x/` 根目录运行）

- `diag_collect.py` — 汇总消融矩阵（优先读 `trainer_state.json` 的精确 step）
- `diag_impl_check.py` — 静态+数值实现核对（Identity 等价 / hook 计数 / 手写参考 / init 统计 / 逐模块 grad / ckpt 审计 / 激活跨 patch std 扫描）
- `diag_timeline.py` — 逐 10 步训练时间线（`z_s`/解码器输出/逐模块梯度）
- `diag_attribution.py` — DINO vs special_bank 的 2×2 塌缩来源归因
- `diag_lr_sens.py` — 一步 AdamW 的临界 lr 探针（辅助）
- `/root/run_diag_arm.sh` — 单臂训练封装

> 未改动 `model_v2.py` / `train_v2.py`；未执行任何 `git commit/push`；未触碰 6006/6008 服务与 `/root/tf-logs*`。

### 7.3 复现命令（塌缩臂 Bwr；把 STACK_DIM/DEPTH/HEADS/DROPOUT/LR/WARMUP 换成表中该行即为对应臂）

```bash
export PATH=/root/miniconda3/bin:$PATH; export HF_HUB_OFFLINE=1; export PYTHONUNBUFFERED=1
cd /root/autodl-tmp/sr-diffusion-v3-stack2x
CUDA_VISIBLE_DEVICES=0 python train_v2.py \
  --data_dir /root/autodl-tmp/construction_site --dino_dir /root/autodl-tmp/models/dinov2-large \
  --output_dir output/diag_Bwr --model_input 448x252 --canvas 1600x900 --angle_step 0.5 \
  --epochs 40 --max_steps 600 --batch_size 16 --grad_accum 1 \
  --lr 1.5e-4 --weight_decay 0.01 --warmup_ratio 0.4367 --grad_clip 1.0 \
  --num_workers 8 --eval_limit 32 --eval_every 100000 --save_every 100000 --log_every 10 \
  --seed 42 --heads 16 --mlp_ratio 4.0 --decoder_depth 4 --stack_dim 2048 --decoder_dropout 0.05 \
  --slice_start 0 --slice_end 5 > /root/train_logs/stack2x_diag_Bwr.log 2>&1
```

> **注意**：`--max_steps` 会改变 `warmup_steps=int(max_steps*warmup_ratio)`。要复现真实 run 的 262 步 warmup，
> 短程 run 必须用 `--warmup_ratio 0.4367`（262/600），**不要照抄模板里的 0.03**（那只有 18 步 warmup，不会塌缩）。
