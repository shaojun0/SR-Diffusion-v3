# DESIGN — v2 循环架构（上一步的输出作为下一步的输入）

- 日期: 2026-09-15
- 代码基线: HEAD `1d908a8`（本次改动尚未提交）+ `model_v2.py` / `train_v2.py` / `infer_v2_test.py` / `visualize_recon_pixel.py`
- 状态: 已实现 + 自检通过 + 本地 CPU 数值冒烟通过；**待服务器训练验证**
- 一句话: 把解码器从"一次并行算完 T 步 + 输出累加"改成**顺序循环**——step1 查询 = `query_base`（去掉原式里的 `A_t`），step t≥2 查询 = `query_base + fuse(上一步的输出)`；每步仍只读自己那块 `z_s`（`memory_mask` 不变），历史输出经循环携带。**以时间换跨步信息流。**

---

## 1. 为什么要换这个方案（诊断）

仓库已有结论（`README.md` §4.1、`doc/2026-09-07/ANALYSIS_k3_why_later_steps_zero.md`）：

- 旧损失把**每个采样步的累加结果都监督成整图** + 按步解耦 ⇒ step-1 的子任务就是完整重建，后步的子任务 = 预测 step-1 没做出来的残差；残差从后步**可读的键**里不可预测 ⇒ L1 最优预测 = 零 ⇒ 后区 register 收不到有效读出梯度 ⇒ **"零增量自锁"**。
- 2026-09-10 的掩码实验进一步确认：`query_mask_mode` 从 `causal` 翻到 `blockdiag` 后 `eval_recon` 0.3584→0.3318（单发通路更好），但探针实测后步相对量级反而更小（`step_px_scale` causal `[1.0072, 0.0630, 0.0584, 0.0572, 0.0563]` → blockdiag `[1.0273, 0.0395, 0.0349, 0.0342, 0.0335]`）。**掩码开关不是杠杆。**

本次要补的一条结构性缺口（本设计直接针对）：

> 旧架构的 `blockdiag` 下，**步 t 的查询行看不到任何其它步的查询行**（`tgt_mask` 块对角），而 `memory_mask` 又规定**步 t 只读自己那块 `z_s`**。于是步 t 的**全部可用信息 = 自己那一块键 + 全局静态 `query_base` 模板**。用一块键去重建整图在信息上就不成立，最优解自然退化为"交给 step-1，自己输出 ≈ 0"。

也就是说：旧架构把"渐进"寄托在 `memory_mask` 的列数递增上，却在查询侧把步与步之间的**前向信息流**切断了（`blockdiag` 的设计初衷是切断 loss 的跨步**梯度**回流，用的是自注意力，但它同时切断了前向**内容**流）。`causal` 允许"前步→后步"泄露，后步量级确实略大（5.6–6.3% vs 3.3–3.8%），但那是被掩码顺带允许的副产品，不是显式设计。

**循环架构把"上一步输出"显式地变成下一步的输入**，让步 t 能基于"当前画到哪"再决定补什么——残差从"无条件不可预测"变成"以当前估计为条件的可预测修正"。

---

## 2. 方案与精确语义

### 2.1 用户原始设想 → 实现

| 用户描述 | 本实现 |
|---|---|
| step1 `Y = (A_t.unsqueeze(2) + query_base)` 的 `A_t` 去掉，仅保留 `query_base` | `q_1 = query_base`（`A_t` 不再进查询；`z_s` 仍作为 cross-attention 的 memory） |
| step2 = step1 的输出 + `query_base` | `q_2 = query_base + fuse(Y_1)` ∵ `h_1 = Y_1` |
| 上一步输出作为下一步输入 | `q_t = query_base + fuse(state_{t-1})`，`state` 见下 |

### 2.2 循环体（`OutputQueryDecoder._forward_recurrent`）

```
h_0 = 0                                          # (B,N,D) 累积估计
for i, t in enumerate(steps):                    # t ∈ {1,4,9,16,25,...}
    q_i = query_base                             # (1,N,D) → (B,N,D)
    if i > 0:
        state = h_{i-1}                if recurrent_state == "cumulative"
              = Y_{i-1}                if recurrent_state == "increment"
        q_i = q_i + fuse(state)
    Y_i = stack_out( stack( stack_in(q_i), stack_in(A),
                            memory_mask=block_mask_row_i ) )
    h_i = h_{i-1} + Y_i
```

- `A = [z_cls; z_s] + pos_embed`（与并行路径完全一致）。
- `block_mask_row_i` = `build_block_mask(...)` 的第 i 段（**与并行路径同一个掩码、同一语义**：每步只读自己那块 `z_s`；第一个采样步加前缀）。⇒ 每步读的键**没有变多**，变量只有"查询里多了历史输出"。
- `fuse(state)`：
  - `recurrent_fuse="proj"`（默认）: `rec_proj(rec_norm(state))`，`rec_proj = nn.Linear(dim, dim)` **zero-init**，`rec_norm = nn.LayerNorm(dim)`；
  - `recurrent_fuse="add"`: `rec_gate * rec_norm(state)`，`rec_gate` 是标量 `Parameter`（init = `recurrent_gate_init`，默认 0.0）。
- `recurrent_state`：
  - `"cumulative"`（默认）= `h_{i-1} = Σ_{j<i} Y_j`，即**上一步被监督的那个累积估计**（迭代细化：每步看着当前画到哪再做修正）；
  - `"increment"` = `Y_{i-1}`（更字面的"上一步的输出"）。

### 2.3 为什么用 zero-init + LayerNorm

- `query_base` 的元素尺度是 `0.02`；`h` 的尺度随步数增长。直接相加会让反馈在初始化时压过 `query_base`（第 1 层 `norm_first` 的 LayerNorm 只抹掉整体尺度，抹不掉两项的相对比例），训练早期易炸。
- `proj` 分支用 `LayerNorm(h)` 把尺度钉住，再用 **zero-init Linear** 控制"反馈开多大"：初始化时反馈恰为 0 ⇒ **全体步 = 纯 `query_base` 读出**（这正是 v4 那条被实测证明最好（8.26）的读出形式）+ 各自的分块 memory；循环随训练长出。
- 自检已实测：zero-init 下 `rec_proj` **能收到非零梯度**（`doc` §5），即"循环开关打不开"的死开关风险不存在。注意 `rec_norm` 在 zero-init 时故意收不到梯度（`∂L/∂(W·LN(h)) = 0` 当 `W=0`），`rec_proj` 迈出第一步后它才通——这是正常现象，不是 bug。

### 2.4 梯度与"按步解耦"

- `recurrent_detach=False`（默认）：**整条循环反传（BPTT）**。末步损失能经 `h` 推动步 1 的 register（自检 §6.4 实测 `> 0`），而这正是旧 `blockdiag` 结构上恰为 0 的通路。
- `recurrent_detach=True`：喂给下一步前 detach ⇒ 截断 BPTT（省显存/更稳），末步损失回到步 1 register 的梯度**恰为 0**（自检 §6.4 实测 `== 0`）。
- `decode` 的 `carry.detach()`（累加损失路径）**保持不变**：它只切"损失→更早 Y"的直接通路；循环内部 `q_t ← h_{t-1}` 的通路不受影响。因此"每步直接损失仍平权 1/|T|"的性质保留，同时多了 BPTT 的长程信用分配。这一条在循环模式下**不再**满足旧的 §3b 恒等式（`g_total[:,n] == 仅第 n 步损失的梯度`）——那是有意为之，自检 §3b 只在非循环模型上跑。

### 2.5 掩码在循环架构下怎么变（**关键，容易误解**）

循环架构下**两张掩码的命运不同**——一张被"删掉"，一张保留但语义升级：

| 掩码 | 旧架构（并行）作用 | 循环架构 | 为什么 |
|---|---|---|---|
| `tgt_mask`（查询自注意力, 跨步） | `blockdiag` = 步间查询行互不可见; `causal` = 步 t 可见步 ≤t（后步靠这个"累积前缀泄露"拿到历史） | **删除**（`self.tgt_mask = None`） | 循环路径每次只把**一步**的 N 行送进 stack，张量里根本没有其它步的查询行 ⇒ 跨步 `tgt_mask` 无从谈起。**跨步信息改由循环状态 `h` 显式携带**——这正是本方案的核心替换。`blockdiag` 的效果自动成立（单步内 N 行全双向自注意力 = 原来 blockdiag 的对角块） |
| `memory_mask`（cross-attn 读哪些 `z_s` 列） | 每步只读自己那块 `z_s`（第一个采样步额外含 `z_cls`） | **保留**，但升为可配的三档读窗口 `recurrent_memory` | 它管的是"键的信息怎么分给步"，与并行/循环无关。旧架构"每步瞎"是因为**读窗口窄 + 跨步自注意力又被切断**（两块键看不到整图）；循环补上了后一半，前一半按需可放宽 |

**所以"掩码要不要改"的准确答案**：跨步那张**必须改**（改成循环状态，等价于删掉 `tgt_mask`）；读窗口那张**不必改**，但值得试——它是本方案唯一的读侧自由度，故实现成 `recurrent_memory`：

- `"block"`（默认）：与并行路径**逐位复用 `build_block_mask`**（自检里直接 `torch.equal` 校验）。保持"每个 register 只被一个步读"的分工压力（`REPORT_v2_slice05_memory_open` 的结论：键被多步共享会摊薄梯度、导致趋同塌缩）。此时循环仍是**真顺序循环**：步 t 读自己那块 + 通过 `h` 看全部历史输出。
- `"prefix"`：步 t 读 `z_cls + z_s[1..自己块末]`（累积前缀）。每步可**重读**此前的键，补偿 `h` 只是 D 维有损摘要。代价是前面块的键被后面所有步共享 ⇒ 分工压力下降（走向 `open` 那一侧的中间档）。
- `"open"`：**删掉 `memory_mask`**，每步 cross-attention 都读全部 `z_s`。此时不再是"分块渐进读"，而是**对同一份键的迭代细化**——交叉注意力本身进入循环（链: `cross-attn_{t-1} → Y_{t-1} → h_{t-1} → q_t → cross-attn_t`）。**但有一个必须处理的陷阱，见 §2.5.2。**

> ⚠️ 修正一处早先的过度警告: `REPORT_v2_slice05_memory_open` 的塌缩证据来自**并行**架构（T 步同时算、每个键被 T 步 × N 行共享 ⇒ 单键梯度占比 1/36 ⇒ 键趋同）。循环架构里各步经 `h` **顺序化**，机制不同，所以 `open` 不是"禁用项"，而是一个值得单跑的臂——前提是配 `recurrent_step_embed`。

### 2.5.2 `open` 的步退化（实测）与 `recurrent_step_embed` 修复

删掉 `memory_mask` 后，若不开步身份信号，**各步在 zero-init 时会逐位相同**。本地实测（`N=36, K=18, steps=[1,4,9]`，`rec_proj=0`，各步与 step-1 输出最大差）：

| `recurrent_memory` | 各步 vs step1 最大差 | 说明 |
|---|---|---|
| `block` | `[2.30, 2.73]` | 各步 memory 不同 ⇒ 天然不退化 |
| `prefix` | `[1.14, 1.75]` | 同上（读窗口随步变宽） |
| `open` | `[0.0, 0.0]` | **逐位相同**：查询形式与 memory 都不随步变，模型起步 = 同一输出的 \|T\| 份拷贝 |

为什么难逃逸: 唯一随步变的是 `h_{t-1} = Σ_{i<t}Y_i`，而它在进查询前过了 `LayerNorm`——`LN(c·Y) = LN(Y)`（c>0），尺度信息被抹掉；`rec_proj` 又是 zero-init，所以对 `rec_proj` 的梯度里"步与步的差别"是二阶小量。等价地，此时模型对"用哪一步"完全对称。

**修复 = `--recurrent_step_embed`**：加一个 `(|T|, D)` 的 zero-init 可学习逐采样步偏置到查询上。
- 初始化时它也是 0，所以不扰动任何基线；
- 但它的梯度 `∂L/∂q_t` **按步不同**（loss 是 `mean_t L1(cumsum_t, target)`，`cumsum_t = t·Y`），⇒ **第一次更新就一阶打破对称**，不依赖那个二阶信号。
- 自检 §6.8 同时验证了三件事: `open` 无步信号各步逐位相同、`block` 不退化、`step_embed` 各步梯度互不相同且生效后各步输出不同。

> 代价: `+ |T|·D` 参数（`T=5, D=1024` → 5K，可忽略）。它**只**在 `recurrent=True` 且显式开启时创建 ⇒ 不开不影响任何旧权重。
>
> 注: 这与 `doc/2026-09-12/EXPERIMENT_step_identity_arms.md`（并行架构下步身份"相互干扰"）不矛盾——那是**并行**架构里的结论；在 `open` + 循环下步身份不是"锦上添花"而是**打破退化的必要条件**。

### 2.5.1 `query_mask_mode` 在循环模式下的地位

不参与前向（保留参数只为构造兼容 / `model_info.json` 记录）。循环模式不再有"块间泄露"的歧义——信息流就是显式的 `h`。

> 关于"是不是偷偷并行化了"：**没有**。`_forward_recurrent` 是 Python `for` 循环，第 i 步的 `self.stack(stack_in(q), mem, memory_mask=row)` 里 `q` 依赖第 i−1 步的 `Y`（即 `h`），步间是硬数据依赖，无法把 `|T|` 折进 batch 维一次算完。**并行只发生在单步内部**（该步 N 行查询一次前向）。`--recurrent_detach` 也不把它变并行——它只切梯度，不切前向依赖。

### 2.6 代价（"以牺牲时间"）

并行路径一次算 `|T|·N` 行；循环路径分 `|T|` 次、每次 `N` 行。**FLOPs 相近，但步间顺序依赖 ⇒ 无法并行**，wall-clock 变长（Python 循环 + 无法把 `|T|` 折进 batch 维）。激活显存与并行路径同量级（总行数相同），`recurrent_detach=True` 可进一步降。

### 2.7 损失口径（**先确认事实，再谈要不要改**）

**事实（`SRPhase1V2.decode`）**：训练 `loss` **一直是累加口径**，不是"直接监督 `F_hat`"：

```python
Y_cum = [0, cumsum(Y)[:-1]].detach() + Y      # 数值上 = Σ_{i≤t} Y_i
Y_pix = pixel_head(Y_cum)                     # 每步累加结果的像素
F_pix = Y_pix[:, -1]                          # F_hat = Σ_t Y_t
per_step = L1(Y_pix, target).mean(dim=(0,2,3))  # (|T|,)
loss   = per_step.mean()                      # ← 训练损失: 每个采样步的累加结果都监督成整图
recon  = L1(F_pix, target)                    # ← 只是**监控量**（以及 §6 判据）
```

所以"直接输出 `F_hat`"的那个量是 **`recon`**——它只用于 eval/日志，**不是**被优化的目标。`F_hat = PixelHead(Σ_t Y_t)` 是推理/可视化用的集成结果，它与训练的深监督并不冲突。

**但在循环架构下，损失确实有一个值得改的点**——不是口径，而是**累加路径的梯度**：

- 默认 `loss_decouple=True`（carry `detach`）的动机是并行架构的"每步恰收 1 份梯度、避免 t=0 的三角失衡"。循环架构里它带来一个副作用：
  - 累加恒等捷径被 `detach` 切断；
  - 跨步梯度只剩解码器循环那条（`q_t ← h_{t-1}`），而 `rec_proj` **zero-init ⇒ 该通路在初始化时恰为 0**；
  - ⇒ **各步初始时是独立受训的**（自检 §6.9 实测：末步损失对早期步的梯度**恰为 0**）。循环的耦合要等 `rec_proj` 长起来才建立。
- `loss_decouple=False`（朴素 `cumsum`）保留恒等捷径 ⇒ 一开始就是耦合的 **BPTT 深监督**（标准 iterative-refinement 做法），末步损失能直接推动步 0（实测 `> 0`）。

两个开关（都**不改变任何权重形状**，只改监督；`model_info.json` 记录）：

| 开关 | 取值 | 含义 |
|---|---|---|
| `--loss_mode` | `cumulative`（默认=历史）/ `final` | `cumulative` = `mean_t L1(Σ_{i≤t}Y_i, target)` 全轨迹深监督; `final` = 只监督 `F_hat`（= `recon`） |
| `--loss_decouple` | `true`（默认=历史）/ `false` | 累加 carry 是否 `detach`；**循环架构建议试 `false`** |

**建议**：第一臂仍用默认（`cumulative` + `decouple=true`）以做受控对照；若出现"后步几乎不动/循环学不起来"，再单跑一臂 `--loss_decouple false`。**不建议**先上 `--loss_mode final`：那就回到"后步输出≈0"的原始失败模式（没有逐步监督，后步没有动力做残差），而且它正是把 `recon` 当损失——即本文件 §1 诊断里要避免的形态。

> 注：`loss_mode` / `loss_decouple` 只影响训练损失，`F_hat` / `Y_pix` / `recon` 在任何模式下逐位相同 ⇒ 推理/可视化不必对齐（`model_info.json` 仅作留档）。
> 另一条正交的损失杠杆是 **P1 区域损失**（`doc/2026-09-07/DESIGN_v2_region_loss.md`，代码已还原）：让 step t 只监督自己那块 patch 区，从根上消掉"步间任务同构"。它和本方案可叠加，但一次只改一个变量。

---

## 3. 接口

### 3.1 `OutputQueryDecoder` / `SRPhase1V2` 新参数

| 参数 | 默认 | 含义 |
|---|---|---|
| `recurrent` | `False` | 总开关。**关闭时不建任何 `rec_*` 子模块/参数** ⇒ state_dict 键与历史实现逐位一致，旧 checkpoint `strict=True` 继续可载 |
| `recurrent_state` | `"cumulative"` | `"cumulative"` / `"increment"`，见 §2.2 |
| `recurrent_fuse` | `"proj"` | `"proj"`（zero-init Linear）/ `"add"`（标量门控） |
| `recurrent_memory` | `"block"` | 循环路径读窗口：`"block"`（=并行同掩码）/ `"prefix"`（累积前缀）/ `"open"`（**删掉 memory_mask**, 交叉注意力进循环; 需配 step_embed） |
| `recurrent_step_embed` | `False` | 逐采样步 zero-init 偏置（`|T|×D`）；`open` 时**必需**（破步退化, §2.5.2） |
| `recurrent_detach` | `False` | 是否截断 BPTT |
| `recurrent_gate_init` | `0.0` | `fuse="add"` 的标量门初值 |
| `loss_mode` | `"cumulative"` | 训练损失口径：`"cumulative"`（=历史, 逐步累加深监督）/ `"final"`（只监督 `F_hat`）。**不是**权重, 只改监督（§2.7） |
| `loss_decouple` | `True` | 累加 carry 是否 `detach`。`True`=历史; **循环架构建议试 `False`**（§2.7） |

### 3.2 CLI / model_info.json

训练：

```bash
NUM_GPUS=2 ./run_v2_train.sh \
    --data_dir  /root/autodl-tmp/construction_site \
    --dino_dir  /root/autodl-tmp/models/dinov2-large \
    --output_dir output/phase1_v2_recurrent \
    --recurrent \
    --slice_start 0 --slice_end 5 \
    --decoder_depth 2 --num_specials 0
# 可选: --recurrent_state {cumulative,increment}  --recurrent_fuse {proj,add}
#       --recurrent_memory {block,prefix,open}  --recurrent_step_embed
#       --recurrent_detach  --recurrent_gate_init 0.0
#       --loss_mode {cumulative,final}  --loss_decouple {true,false}
```

**三臂建议（受控）**：
1. `--recurrent`（memory=`block`，不加 step_embed）：只改循环这一个变量，对照基线 0.3318。
2. `--recurrent --recurrent_memory prefix`：单独验"读窗口放宽"。
3. `--recurrent --recurrent_memory open --recurrent_step_embed`：**删掉 memory_mask**，交叉注意力进循环 = 对全部键的迭代细化。`open` 不配 `step_embed` 会步退化（§2.5.2），train 会打 warning。

**损失臂（正交, 单独跑）**：`--recurrent --loss_decouple false`（保留跨步恒等捷径, 一开始就做耦合 BPTT 深监督）。不要先上 `--loss_mode final`（§2.7）。

推理 / 可视化**不用传** `--recurrent`：一律读 `model_info.json`（新增字段 `recurrent` / `recurrent_state` / `recurrent_fuse` / `recurrent_memory` / `recurrent_step_embed` / `recurrent_detach` / `recurrent_gate_init` / `loss_mode` / `loss_decouple`）。CLI 只作 fallback 与显式覆盖告警。`loss_*` 只影响训练损失, 推理不必对齐（`model_info.json` 仅留档）。

> ⚠️ `recurrent` 会新增 `rec_*` 参数 ⇒ 与非循环 checkpoint **形状不兼容**，传错时 `strict=True` 会**明确报错**（这是好事，不会静默算错）。但 `recurrent_state` / `recurrent_fuse` 不改权重形状 ⇒ 传错只静默算错，消费方必须以 `model_info.json` 为准。

---

## 4. 与既有实验的关系

| 既有结论 | 与循环架构的关系 |
|---|---|
| `query_mask_mode`（blockdiag/causal）不是"后步干活"的杠杆 | 一致。本方案不依赖该开关；循环路径直接不构造 `tgt_mask` |
| 步身份注入（step_embed / per-step `[cls]`）在旧架构上相互干扰（`doc/2026-09-12/EXPERIMENT_step_identity_arms.md`） | 本方案**先不加**步身份。循环顺序本身 + 分块 memory 已给足步序信号；若训练后仍看不出步分工，再考虑叠加（届时单臂验证） |
| P1 区域损失（每步只监督自己区域，`doc/2026-09-07/DESIGN_v2_region_loss.md`，代码已还原） | **正交**，可叠加。本方案先单独验证循环本身；P1 是解决"步间任务同构"的另一条杠杆 |
| v4 单发（纯 `query_base` 读出）是仓库最佳（8.26） | 循环架构在 zero-init 时**正是** v4 式读出 + 分块 memory + 输出累加；循环是长在它上面的增量改进，初始化不劣化 |

---

## 5. 自检与本地验证（已完成）

`python model_v2.py` → `ALL CHECKS PASSED`。新增 §6 覆盖：

1. 默认关闭：state_dict 不含任何 `.rec_*` 键（旧 checkpoint 兼容）。
2. 循环模型形状 / `tgt_mask is None` / `attn_mask` 正确；`rec_proj` zero-init。
3. **step1 无反馈**：把 `rec_proj` 从 0 改成随机，第 1 步输出**逐位不变**（恰 0），第 ≥2 步改变。
4. **跨步信息流（核心）**：扰动步 1 的 `z_s` 块（位置 1,2）——并行 `blockdiag` 路径后期步逐位不变（§2b 已断言），循环路径**每个后期步都变**。
5. **跨步梯度回流**：BPTT 下末步损失能推动步 1 register（`> 0`）；`recurrent_detach=True` 下恰为 0。
6. zero-init `rec_proj` 收到非零梯度（循环可被打开）+ `query_base`/`special_bank`/`PixelHead` 全通。
7. `increment` / `add` 两种配置能跑；非法 `recurrent_state` / `recurrent_fuse` / `recurrent_memory` 立刻报错。
8. 读窗口三档：`block` 与 `build_block_mask` **逐位相同**（`torch.equal`）；`prefix` 每步允许 `[:hi+1]` 且严格宽于 `block`；`open` 全允许；三档整模型前向都通。
9. §6.8：`open` + zero-init + 无步信号 ⇒ 各步输出**逐位相同**（实测退化）; `block` 不退化; `recurrent_step_embed` 各步梯度互不相同（一阶破对称）且生效后各步输出不同; 未开启时不新增 `rec_step_embed` 键。
10. §6.9 损失口径：默认仍是 `cumulative` + `loss_decouple=True` 且 `loss == mean_t`；`loss_mode="final"` 时 `loss == recon`；`loss_decouple` 只改梯度图不改数值；**循环 + `decouple=True` + `rec_proj=0` 时末步损失对早期 `Y` 的梯度恰为 0，`decouple=False` 时非零**；非法 `loss_mode` 报错。

另有本地 CPU 冒烟（`N=576, K=35, steps=[1,4,9,16,25], B=2`）：

- zero-init 下，循环 forward **逐步等价于**"`query=query_base` + 该步掩码行"的单步解码（误差 `<1e-5`）；
- 打开反馈后 step1 逐位不变、后续步改变；
- 全模型 `loss.backward()` 通，`rec_proj` 梯度范数和 ≈134.6（确实被打开）；
- 读窗口实测（`N=576, K=35, steps=[1,4,9,16,25]`，每行允许列数）: `block` 首/末步 = 4/11 列、`prefix` = 4/36、`open` = 36/36（`S=K+1=36`），三档反向均通。

**回归**：用 `git show HEAD:model_v2.py` 的旧模块与新模块，同种子、同输入、同权重下 `F_hat` / `loss` / `Y` 的 `float64` 求和**完全相等**，`state_dict` 键数一致（50）⇒ 非循环路径逐位不变。

---

## 6. 训练时看什么（判据，沿用仓库既有口径）

一律用**匹配步**、历史对比用 `eval_recon`（勿用 `eval_loss`）：

1. 后步量级：`probe_step_collapse.py` 的 `step_px_scale` 应显著脱离 `[~1, ~0.04, ~0.03, ...]`（旧值见 §1）；目标 = 后步 `> 0.015`。
2. 渐进曲线：出现真阶梯（覆盖区单调降）且 `eval_recon` 不劣化（对照基线：blockdiag slice[0:5] K=35 `eval_recon` 0.3318）。
3. 各步隔离 L1 在自己区域上相近（不再 39 vs 61）。
4. 监控 `decoder.rec_proj.weight` 的范数与 `rec_norm` 梯度：若训练全程 `rec_proj` 范数 ≈ 0，说明循环没被用起来（退化回 v4 式单发），此时可改 `--recurrent_fuse add --recurrent_gate_init 0.1` 或加步身份/区域损失。

**预期**：至少应看到后步量级上升（循环给了它们"看到当前估计"的通路）；能否把 `eval_recon` 显著压过 0.3318 取决于键里是否真有可读增量——本方案解决的是"信息流被切断"，不直接解决"键内容冗余"（那是 P2/编码侧的事，`README.md` §4.1）。

---

## 7. 改动文件

- `model_v2.py`: `OutputQueryDecoder` 的 recurrent 路径 + `SRPhase1V2` 透传 + `loss_mode`/`loss_decouple` + 自检 §6 + 模块/类 docstring。
- `train_v2.py`: CLI（`--recurrent*` / `--loss_mode` / `--loss_decouple`）/ 构造透传 / 启动打印 / `model_info.json` 新字段 / 头部说明。
- `infer_v2_test.py`: CLI + `model_info.json` 优先解析 + 构造透传 + 加载打印 + 一致性检查。
- `visualize_recon_pixel.py`: CLI + `model_info.json` 解析 + 构造透传 + 打印。
- 工作区（非仓库）: `probe_step_collapse.py`（探针支持循环字段）、`run_recurrent_slice05.sh`（对照臂入口）。
- 本文件。
