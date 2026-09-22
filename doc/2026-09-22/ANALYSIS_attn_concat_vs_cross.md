# ANALYSIS — 拼接式自注意力 vs 交叉注意力：拼接后"里面"到底有没有交叉注意力？（2026-09-22）

> 起因（用户 2026-09-22 提问）：**"把交叉注意力的两个向量组拼接在一起做自注意力，
> 那么这个自注意力是否包含着交叉注意力的内容？"**
> 关联：`doc/2026-09-22/probe_crossattn_encoder_grad.py`（同一晚在问"解码器 → 编码器的学习信号"），
> 两者问的是**同一块代码**：`model_v2.OutputQueryDecoder` 既有 `self_attn`（吃 `[z_cls; z_s]`），
> 又有 `multihead_attn`（Q 是 query，KV 是 `A_in[:, lo:hi+1]` 切片）。
>
> 探针：`doc/2026-09-22/probe_attn_concat_vs_cross.py`（纯 CPU、秒级、不依赖 DINOv2）
> 数据：`doc/2026-09-22/data/probe_attn_concat_vs_cross.json`
>
> **一句话答案：注意力矩阵的"交叉子块"就是交叉注意力；但整段输出不是交叉注意力 ——
> 差在 softmax 的归一化范围（注意力预算）与梯度耦合。加掩码后两者逐位相等。**

---

## 0. 结论（先给结果，D=64 / H=4 / n=5 / m=7 / 随机权重 / seed=42）

设 $X\in\mathbb{R}^{n\times d}$（query 组）、$Y\in\mathbb{R}^{m\times d}$（memory 组）。

| # | 命题 | 实测 | 判定 |
|---|---|---|---|
| **A** | 拼接 + **掩码 self 块** ≡ 标准交叉注意力（Q=X, KV=Y） | `max\|Δ\| = 0.0`（逐位） | ✅ **严格等价** |
| **B** | 拼接（无掩码）≠ 标准交叉注意力 | `max\|Δ\| = 0.337`，相对误差 **57.9%** | ❌ 不等价 |
| **C** | 无掩码时 self 块会**抢走**注意力预算 | X 行：cross **58.9%** / self **41.1%**（Y 行对偶 41.1%/58.9%） | ⚠️ 预算被切分 |
| **D** | 无掩码时 $S_{xx}$ 的梯度会串到 $S_{xy}$ 上 | $\|\partial L/\partial S_{xy}\|$：无掩码 **0.0596** vs 掩码 **0.0991**（−40%）；且 $\|\partial L/\partial S_{xx}\|$ 不为 0（0.0162） | ⚠️ 梯度耦合 |
| **E** | 想靠调尺度 $\lambda$ 把预算"拨"给 cross 很难 | $\lambda\!:0.25\to4.0$ 时 X 行 cross 占比仅 $0.574\to0.726$；**对偶方向完全不动**（Y 行恒 0.411） | ⚠️ 不可解耦 |

**对最初提问的直接回答**：
1. "包含着交叉注意力的**内容**" —— ✅ 成立，如果指的是 $A$ 的交叉子块 $A_{xy}=A[:n,\,n:]$。
   把它的行做归一化后，它就是一次合法的"X 查 Y"的注意力分布。
2. "这个自注意力**就是**交叉注意力" —— ❌ 不成立。无掩码时它的分母是 $n+m$ 个位置（含 self），
   并且 $S_{xx}$ 通过 softmax 的雅可比把梯度灌进 $S_{xy}$。
3. **想要"就是"，加掩码即可**（命题 A，`max|Δ|` 精确为 0，不是近似）。

---

## 1. 为什么是"子块相等、整体不等"（块分解）

拼接后只做一次 softmax 行归一化（暂时省略缩放 $1/\sqrt{d_h}$）：

$$
S=\begin{bmatrix}XW_q\\YW_q\end{bmatrix}
\begin{bmatrix}W_k^\top X^\top & W_k^\top Y^\top\end{bmatrix}
=\begin{bmatrix}S_{xx} & S_{xy}\\ \hline S_{yx} & S_{yy}\end{bmatrix},
\qquad A=\operatorname{softmax}_{\text{row}}(S)
$$

输出（$Z_x$ 为 $X$ 那一路）：

$$
\begin{bmatrix}Z_x\\ Z_y\end{bmatrix}
=A\begin{bmatrix}XW_v\\YW_v\end{bmatrix}
=\begin{bmatrix}
\underbrace{A_{xx}(XW_v)}_{\text{自注意力}} + \underbrace{\mathbf{A_{xy}(YW_v)}}_{\text{交叉项}}\\[4pt]
\underbrace{\mathbf{A_{yx}(XW_v)}}_{\text{交叉项}} + \underbrace{A_{yy}(YW_v)}_{\text{自注意力}}
\end{bmatrix}
$$

加粗两项就是"交叉注意力的内容"，形状与语义都对得上（$X$ 去查 $Y$、$Y$ 去查 $X$，
即双向的 cross-attn）。**但整体输出不等于** $\operatorname{softmax}(S_{xy})YW_v$：

| 差异来源 | 标准交叉注意力 | 拼接式自注意力（无掩码） |
|---|---|---|
| 归一化范围 | 只跨 $Y$ 的 $m$ 个位置 | 跨 $X\cup Y$ 的 $n+m$ 个位置，分母含 $\sum_i e^{S_{xx}[i,\cdot]}$ |
| 行权重之和 | $\sum_j A_{xy}[i,j]=1$ | $\sum_j A_{xy}[i,j]+\sum_j A_{xx}[i,j]=1$（**此消彼长**） |
| 反向传播 | $\partial L/\partial S_{xy}$ 只由 $A_{xy}$ 与其上游决定 | 稠密 Jacobian：$S_{xx}$ 的变化经分母改变 $A_{xy}$ |
| 参数 | 通常 Q / KV 各一套投影 | 常共享一套（本探针即"块共享"口径，唯一变量=softmax 范围） |

> **"包含"是集合意义上的包含，不是等价。** 类比：$\operatorname{softmax}$ 是把 $n+m$ 个数
> 拉成概率，你从里面切出 $m$ 个数，它们**不再是**那 $m$ 个数自己的 softmax。

---

## 2. 实测细节

### A/B. 等价性与差异（`probe_attn_concat_vs_cross.py`）

- **A（掩码）**：把 $S_{xx},S_{yy}$ 置 $-\infty$，行 softmax 只在另一块上归一化，
  等价于 $\operatorname{softmax}(S_{xy})$。实测与 `CrossAttn`（q_proj/k_proj/v_proj/out_proj 分离调用）
  的输出 `max|Δ| = 0.0`，**浮点意义上逐位相同**（不是 "1e-7 近似"）。
- **B（无掩码）**：同一组权重、同一输入，`max|Δ| = 0.337`，相对误差 $57.9\%$。
  这个量级说明：**无掩码拼接不能当作交叉注意力的替身**。

### C/E. 注意力预算

无掩码时 X 行的 softmax 概率质量被分成两块，实测 cross : self ≈ **58.9 : 41.1**。
这个比例**不是超参，而是 $S_{xx}$ 与 $S_{xy}$ 量级的隐式函数** —— 探针 E 扫了
乘在 $S_{xy}$ 上的尺度 $\lambda$：

| $\lambda$ | 0.25 | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---:|---:|---:|---:|---:|
| X 行 cross 占比 | 0.574 | 0.578 | 0.589 | 0.626 | 0.726 |
| Y 行 cross 占比 | 0.411 | 0.411 | 0.411 | 0.411 | 0.411 |

两点值得写进工程备忘：
1. $\lambda$ 放大 **16 倍**，cross 占比只从 0.57 抬到 0.73 —— 在 $m$ 较小（此处 $m=7$）
   且 logits 量级相近时，single-softmax 的预算**很"黏"**，不容易拨动。
2. Y 行占比**完全不动**：因为 $\lambda$ 只乘了 $S_{xy}$，$S_{yx}$ 没动。
   这暴露了更本质的问题：**共享 softmax 把两个方向绑在一起**，
   想独立控制"X→Y 有多强"和"Y→X 有多强"用单个 $\lambda$ 做不到（$_xy$ 与 $_{yx}$ 需各自的尺度）。

### D. 梯度耦合（这是比"输出不等"更隐蔽的一条）

构造 $L=\langle A_{xy}, G\rangle$ 反传，看 $\partial L/\partial S$：

| | $\|\partial L/\partial S_{xy}\|$ | $\|\partial L/\partial S_{xx}\|$ |
|---|---:|---:|
| 无掩码拼接 | 0.0596 | **0.0162（≠0）** |
| 掩码拼接 | 0.0991 | **0.0（精确）** |

- **无掩码下，即使损失只定义在交叉项上，梯度也会流进 $S_{xx}$**，而且 $S_{xy}$ 上拿到的梯度
  比掩码版**小 40%**（被分母"稀释"）。工程含义：self 与 cross 互相拖累，
  自注意力学歪会直接污染交叉注意力的学习信号。
- **掩码下耦合被精确切断**（$S_{xx}$ 的梯度恰为 0），与命题 A 的等价性一致。

---

## 3. 什么时候用哪个（决策表）

| 目标 | 建议 | 依据 |
|---|---|---|
| 省一次 kernel / 双向融合 / 复用现成 self-attn 实现 | **拼接 + 掩码**（精确等价，双向一次算完） | 命题 A |
| decoder→encoder 这种"交叉必须独占"的场景 | **掩码版**；**绝不要无掩码拼接** | 命题 B/C |
| 需要对 self/cross 强度做**可控**解耦 | 独立 cross-attn 模块；或拼接但**不共享** $W_q,W_k,W_v$ 并给两个方向**各自的**温度 $\lambda_{xy},\lambda_{yx}$ | 命题 C/E |
| 多模态双向融合（图↔文、隐变量↔条件） | 无掩码拼接常可用，甚至因正则化更稳；但要接受"预算此消彼长 + 梯度耦合" | 命题 C/D |
| 想用微调把"拼接"逼成"交叉" | 别这么做：$\lambda$ 扫到 16× 也只挪 15 个百分点 | 命题 E |

**与 v3 代码的对应**（`model_v2.OutputQueryDecoder.forward`，源码 496–529 行）：
现在是"拼接 self-attn（`[z_cls; z_s]` 共 $1+K$ 个 token）
+ 独立 `multihead_attn`（Q=`Y`，KV=`A_in[:, lo:hi+1]`，$lo/hi$ 由步 $t$ 的平方块算出）"
两条路并存 —— 等价于**已经选了"独立模块"这一列**。若为省算力想把 `multihead_attn`
并进 `self_attn`，注意三点：

1. 每个采样步的读窗口是**块级硬约束**（步 $t$ 只读自己那块，首步 `lo=0` 含 `z_cls`），
   并进拼接版后必须为**每个步**构造跨块掩码；否则一步就会读到**全部 K 个 z_s**，
   逐步采样（`t=1,4,9,…,K`）从"按块增量读"退化为"每步读全量"。
2. 无掩码拼接时，`z_cls/z_s` 的块内交互会与读窗口**争抢同一份注意力预算**（命题 C），
   而读窗口的分配本来应由 `lo/hi` 决定 —— 口径从"结构决定"变成"学出来"。
3. 掩码版本（命题 A）可以做到**与现有 `multihead_attn` 逐位等价**，才是安全的合并方式。

---

## 4. 局限（别过度外推）

1. 本探针是**机制级**验证：随机权重、$d=64$、$n=5$、$m=7$、CPU、seed=42。
   它证明的是"等价/不等价"这一类**结构性命题**，不是"训练后效果差多少"。
2. 命题 A 的逐位相等成立的前提是：**同一组投影**、**同一个 $d_h$**、掩码精确置 $-\infty$。
   若 self/cross 用不同投影，掩码拼接仍等价于"Q/KV 分离的交叉注意力"，但需按实际
   投影对齐（`CrossAttn` 已支持）。
3. softmax 数值细节：掩码用 `masked_fill(-inf)` 后 `softmax` 是标准做法；
   探针未涉及 flash-attn 的实现差异（不同 kernel 对 $-\infty$ 的处理可能引入极小偏差）。
4. 未测**位置编码/相对偏置**跨块参与的情形：若 pos 在跨块也生效，
   会再加一层不可分离的耦合，等价性结论需按掩码后重算。

---

## 5. 复现

```bash
# 秒级，CPU，无额外依赖（torch 2.8.0+cpu 已在本机可用）
python3 doc/2026-09-22/probe_attn_concat_vs_cross.py
# → doc/2026-09-22/data/probe_attn_concat_vs_cross.json
```

输出字段与本文表格对应：`A_masked_exact`（命题 A）、`B_unmasked_*`（B）、
`C_attention_budget`（C）、`D_grad`（D）、`E_budget_sweep`（E）。
改动 shape 只改 `main()` 里的 `D,H,n,m`（或加 CLI 参数），结论方向不随尺寸变化。
