# DESIGN · 解码器查询自注意力掩码开关 `query_mask_mode`

> 日期: 2026-09-10 ｜ 改动: `model_v2.py`（`build_causal_query_mask` / `OutputQueryDecoder` / `SRPhase1V2`）+ `train_v2.py` / `infer_v2_test.py` CLI
> 默认值 `"causal"` 与历史行为**逐位相同**，本次改动不改变任何既有训练结果。

---

## 1. 动机：一个已经写了一半、只实现了一半的意图

`SRPhase1V2.decode` 的 docstring（`model_v2.py`）明确写了梯度按步解耦的意图：

> **梯度按步解耦**: 数值上仍 `Y_cum = cumsum(Y)`（F_hat 不变）, 但 `carry = [0, cumsum(Y)[:-1]]` 整体 detach + 自己的预测——每个 `Y_t` 只从自己那一步的损失收 1 份梯度（平权 `1/|T|`）, 不再从所有 ≥t 的累加位置收梯度（否则 t=0 有 `|T|` 份梯度动力, 学乱）。

`carry.detach()` 只砍掉了**数值累加**这条路径。**注意力那条路径没有被砍**，实测（N=576, K=35, D=64, `steps=[1,4,9,16,25]`，按 `1/|T|` 平权求和的真实梯度口径）：

| 损失项 | → 「块 1 种子」\|g\| | → 「块 1 非种子成员」\|g\| |
|---|---|---|
| `L_1` | 1.87e-01 | 2.99e-02 |
| `L_4` | 5.25e-02 | 1.53e-03 |
| `L_9` | 4.07e-02 | 1.27e-03 |
| `L_16` | 1.98e-02 | 6.43e-04 |
| `L_25` | 1.56e-02 | 6.35e-04 |

⇒ 块 1 的 register 仍被后面 4 个损失一起推，正是上面 docstring 说要避免的"`t=0` 有 `|T|` 份梯度动力"。

---

## 2. 前向侧：`memory_mask` 的隔离承诺从第二层起不成立

`build_block_mask` 的语义是"每步只见自己的 `z_s` 块"。但**实测的依赖集是累积前缀**：

```
step  1 -> z_s[0..2]     = 块1
step  4 -> z_s[0..7]     = 块1 + 块2
step  9 -> z_s[0..14]    = 块1..3
step 16 -> z_s[0..23]    = 块1..4
step 25 -> z_s[0..34]    = 块1..5（全部）
```

机理（`depth=2` 两层栈 × 块因果 `tgt_mask`）：第一层里步 1 的行做了 cross-attention（读了自己整块），**这些行已携带块 1 的完整信息**；第二层里步 4 的行 self-attend 到步 1 的行 → 块 1 整块漏进步 4。

量级（改块 1 的种子）：步 4/9/16/25 输出变化 `0.677 / 0.465 / 0.359 / 0.202`（对自身尺度 11.8 即 5.7%/3.9%/3.0%/1.7%）。

置换检验独立确认同一结构（置换块内**非种子**成员，`after` 版输出变化）：

| 置换 | step 1 | step 4 | step 9 | step 16 | step 25 |
|---|---|---|---|---|---|
| 块 1 | 8.3e-07 | 7.2e-07 | 7.2e-07 | 4.8e-07 | 7.2e-07 |
| 块 3 | **0** | **0** | 9.5e-07 | 7.2e-07 | 4.8e-07 |
| 块 5 | **0** | **0** | **0** | **0** | 9.5e-07 |

（其中 `7.15e-07` 与 `doc/2026-09-10/ANALYSIS_k3_posenc_failure.md` §2.2 报的 `7.15e-7` 一致，说明口径对齐。）

结论：`tgt_mask` 只保证"后步不泄露进前步"；反方向（前步 → 后步）的泄露被允许且实际发生——这与 `build_causal_query_mask` 的块下三角定义一致，属**设计行为**，但使"每步只见自己的块"和"梯度按步解耦"两个表述都需要加范围限定。

---

## 3. 改动：掩码语义参数化

```python
QUERY_MASK_MODES = ("causal", "blockdiag")

build_causal_query_mask(num_steps, num_queries, device=None, mode="causal")
```

| mode | 掩码 | 语义 |
|---|---|---|
| `"causal"`（默认） | 块下三角：步 `t` 可见步 `≤ t` | 历史行为，**逐位不变**（自检断言保证） |
| `"blockdiag"` | 块对角：步 `t` 只可见自己那 `N` 行 | 步间在自注意力上完全隔离 |

函数名保留历史名（4 处 md 文档 + 探针脚本按此名引用）；两种语义由 `mode` 承载，`build_causal_query_mask` 的 docstring 已写明该模式名在 `blockdiag` 下不再是因果语义。

**参数贯通**：`OutputQueryDecoder(query_mask_mode=...)` → `SRPhase1V2(query_mask_mode=...)` → `train_v2.py --query_mask_mode` / `infer_v2_test.py --query_mask_mode`。非法值直接 `AssertionError`，不静默退化。

**向后兼容性**：
- 不改任何权重形状（实测 `state_dict` 键集合与形状逐项相同），旧 checkpoint 双向可载；
- 因此 `model_info.json` **必须记录** `query_mask_mode`，且 `infer_v2_test.py` 按"model_info.json 优先 → CLI → causal"解析并告警——本项不一致**不会**触发 `strict load` 形状错，只会静默算错，是比 K 错更隐蔽的坑。

---

## 4. `blockdiag` 做到了什么

1. `memory_mask` 的"每步只见自己的 `z_s` 块"在整条前向路径上**字面成立**（实测：改块 `k` 只影响步 `k`，其余步逐位恰好 `0.0`）。
2. loss 的跨步梯度回流被**恰好**切断：`blockdiag` 下最后一步的损失对更早块 register 的梯度为 `0.0e+00`（`causal` 下为 `3.4e+01` 量级）。与 `carry.detach()` 合起来，`decode` 声明的"每步只从自己那一步的损失收梯度"才真正成立。
3. **归因变干净**：`block k ↔ 步 k` 从"累积前缀耦合"变成严格一一对应，register 级实验可直接归因。

---

## 5. `blockdiag` **没有**做到什么（重要限定）

**它不修 register 塌缩。** 实测：块 1 完全塌缩（`z_s[0]=z_s[1]=z_s[2]`）时，按真实损失加权求和的**块内非种子成员梯度差**：

```
causal    : 0.000e+00
blockdiag : 0.000e+00      ← 一样锁死
```

原因：塌缩相关的对称群 `S_{|block|−1}`（块内非种子成员之间的置换）只由**自己那块的 cross-attention + 块内位置编码**决定；跨步自注意力改变的是**种子**与整块"代表输出"的梯度，而种子本就被排除在该对称群之外。
⇒ 要治塌缩必须动 **memory 侧（块内位置寻址）**，见 `doc/2026-09-10/ANALYSIS_k3_posenc_failure.md` §7 的 P1–P5。

**渐进语义不受影响**：渐进性来自读侧 `memory_mask` 的允许列数随步单调递增（4→5→7→9→11），不依赖 `tgt_mask`；`blockdiag` 不破坏它。

**表达力确实变了**：从"5 步串行、共享权重的循环结构（信息沿步向前流动）"变成"5 个共享权重的并行预测头 + 输出累加"。参数效率不降（权重仍共享），但每步的有效输入从累积前缀缩回自己那一块。这是**必须用实验决定**的问题，不能靠审美定。

---

## 6. 验证

**模型自检**（`python model_v2.py`，全部通过）：新增 §2b 段，覆盖
- 默认 `mode` 与历史块下三角**逐位相同**；
- `blockdiag` 对角块全允许 + 其余全 `-inf` + 是 `causal` 的真子集；
- 非法 `mode` 报错；
- 解码器级隔离性（`causal` 泄露 vs `blockdiag` 恰好 0）；
- 梯度级切断（`causal` 非零 vs `blockdiag` 恰好 0）。

**独立 smoke test**（`python doc/2026-09-10/smoke_query_mask_mode.py`，CPU 秒级，无需 DINO 权重/GPU）：形状 + 互载 + 掩码结构 + 逐块隔离性矩阵 + 梯度矩阵，输出形如

```
[B] 改哪块      step 1  step 4  step 9  step 16
    causal 块1   4.4e+00  2.9e-01  2.4e-01  1.3e-01
    blockd 块1   4.4e+00  0.0e+00  0.0e+00  0.0e+00
[C] causal    : 块1(种子)=3.4e+01  块2(种子)=3.9e+01  块3(种子)=3.6e+01
    blockdiag : 块1(种子)=0.0e+00  块2(种子)=0.0e+00  块3(种子)=0.0e+00
```

---

## 7. 若要跑 A/B 对照（协议要求）

本开关是**移除一条通路**的干净消融，但**必须配 P0 协议**，否则会重蹈 `REPORT_v2_block_slice05_posenc.md:13` 那个"同预算唯一变量"自相矛盾的覆辙（见 `ANALYSIS_k3_posenc_failure.md` §4.1 列出的 4 项运行层混淆）：

- 两臂同 `max_steps`（否则 `warmup` 262 vs 48、cosine 相位不同，端点不可比）；
- `eval_every` 加密到 **step 250–450 每 20 步**（A 臂只有 1 个 eval 点就是教训）；
- 判据全部用**匹配步**的 train loss + grad_norm，不用端点 eval；
- **≥3 seed**（n=1/臂只能得"共现"，得不出"塌缩率"）；
- 记录 `‖Δθ‖`、Adam `v` 范数，避免再次陷入"梯度小 ⇒ 参数不动"的未验证推断。

预期：若 `blockdiag` 对塌缩**中性**（§5 实测支持），两臂应同样健康或同样塌；若 `blockdiag` 反而更稳，说明跨步耦合本身在贡献不稳定，那是一条新线索。
