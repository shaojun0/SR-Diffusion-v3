# SR-Diffusion v3 — E1 线性探针报告：z_s/解码路径还剩多少"可被利用"的像素信息

> 日期: 2026-09-07 ｜ 性质: 零训练推理期探针（服务器 2× RTX PRO 6000, ~2 min 总时, 无训练）
> 触发: `doc/2026-09-04/ANALYSIS_v2_info_left_and_decoder.md` §5（E1 设计, 此前"服务器已关机"未跑）+ 用户 2026-09-07 指示执行
> 脚本: `doc/2026-09-07/probe_e1.py`（服务器副本 `/root/autodl-tmp/sr-diffusion-v2-k99/probe_e1.py`, 与 `probe_slice05_exp1.py` 同口径）
> 数据: `doc/2026-09-07/data/probe_e1_slice27_v2.json`、`probe_e1_slice05.json`
> 证据分级沿用项目惯例: [数据]=本次实测 ｜ [推断]=推断

---

## 0. TL;DR

1. **编码器输出层的 patch 级信息极其充足**：与训练同分布（register 同序列过 DINO 24 层）的 patch 特征 P，线性解码即可达像素 L1 = **7.55（slice27_v2）/ 7.82（slice05）**（512 子集, 0-255）——优于 v4 的最终 8.26，更远优于旧冻结 DINO 时代的 ≈9.8。
2. **解码器最终累加特征 h（= z_s 经注意力读出后的 patch 对齐特征, PixelHead 之前）线性解码 = 21.95 / 23.06，≈ 完整模型（含 MLP 头）的 20.64 / 21.30** → **PixelHead 不是瓶颈**；解码器特征携带的信息已被压到 ~模型自身水平。
3. **P(7.5) → h(22) 之间丢失 ≈14.5 个像素 L1**：丢发生在 **register 路由 + 注意力读出 + 时间轴累加**这一跳（该跳只消费 K 个 register, 从不直接看 P），而不是信息源、也不是最后投影。
4. **register 塌缩程度与 L1 直接对应**：z_s 空间内方差 within-std = slice27_v2 **0.143** vs slice05 **0.053**（P 两者均 ≈0.17–0.18）——slice05（K=35, 前缀仅 4 键）register 塌缩 3 倍于 slice27_v2，与其更差的 L1（21.3 vs 20.6）及"多样性预算全押 step-1 读区"旧判读一致。
5. **对"14.28→20：信息没被利用 or 架构学不出来?"的回答（实证版）**：
   - "架构学不出来"在**编码器输出层不成立**——P 里线性躺着 7.5 的信息；
   - "信息没被利用"的精确位置 = **P 级 patch 信息没被接进/读出 register 路径**（v2 因果/分块/固定模板读出路线在训练压力下把 register 养成近冗余簇, h 只能到 ~22）；
   - **不是**"z_s 里明明有 9–14 的信息、只差解码器线性取用"——h 已到该栈的读出上限（线性≈模型自身）；
   - v4=8.26 证明同一编码器架构在"单发读出+正确训练压力"下能把这 7.5 的信息搬进 K register。学不出来的是**特定读出路线**（时间轴+分块+累加平权损失）下的这一跳。

## 1. 方法（与 ANALYSIS §5 的差异说明）

- 目标模型: `phase1_v2_block_slice27_v2`（K=63, steps [9,16,25,36,49]）与 `phase1_v2_block_slice05`（K=35, steps [1,4,9,16,25]），代码 = git `36cf777`（服务器 v2-k99 目录, 与训练时代逐字一致）。
- 512 图子集 = 与 09-04 探针相同的确定性前 512 条 test（limit=512, 无 shuffle）。
- 三个特征层:
  - **P** = 与训练同分布的 patch 段（`encode_with_patch` 复刻 `_encode_register` 的 DINO 24 层 forward, 取 `seq[:, 1+K:]`; 非 HF `dinov2(x)` 干净序列——后者缺 register token, 分布不同）;
  - **h** = `Y.cumsum(dim=1)[:, -1]`（解码器每步输出特征累加的最终值, 与 F_hat 同源、未过 PixelHead; 每图 576 行 patch 对齐）;
  - **z_s** = register 段 `seq[:, 1:1+K]`（**K 个全图 token, 与 576 patch 不对齐**）。
- 线性解码 L1: `np.linalg.lstsq`（float64, rcond=1e-6）特征→patch 像素（0-255, 反归一化+clip）, 与 `trace_info_pixel.py` 同口径。判读参照 ANALYSIS §5 表。
- **z_s 探针不适定, 置 None**：z_s 是 K=63 个全图 register（每图 63×1024 维）而目标是 576 patch——逐 patch 线性映射参数（63×1024→576×588）在 512 样本下病态/无意义; 旧 `trace_info_pixel.py` 的"z_s 逐 patch 对齐"是旧 ReEncoder 架构（N token）产物, register 架构不适用。register 携带的可及信息改由 **h**（=z_s 经真实读出路径后的 patch 对齐特征）刻画, 意义更直接。

## 2. 结果 [数据]

| 量 (512 子集, 像素 L1 0-255) | slice27_v2 (K=63) | slice05 (K=35) |
|---|---|---|
| 完整模型解码（PixelHead, 参照） | **20.640** | **21.299** |
| 5 步渐进曲线 | [20.64, 20.63, 20.63, 20.63, 20.64] 全平 | [21.31, 21.29, 21.29, 21.29, 21.30] 全平 |
| **P（DINO patch 特征）→ 线性** | **7.545** | **7.824** |
| **h（Y_cum[-1], 解码读出特征）→ 线性** | **21.952** | **23.058** |
| 平均色基线 | 60.48 | 60.48 |
| within-std: P / z_s / h | 0.178 / **0.143** / 2.94 | 0.171 / **0.053** / 5.06 |

- h 线性 ≈ 完整模型 L1（±0.3–1.7）→ 线性读出已触到该模型特征空间的顶, MLP 头只带来 ~0.3–1.7 增益。
- P 线性远低于两者 → 编码器输出层确实"躺"着大量可线性还原的像素信息（7.5–7.8）。
- slice05 的 z_s within-std 0.053 ≈ slice27_v2 的 1/3 → 更小 K/更短前缀下 register 内容更塌缩（同批探针的 cos 结构: 1..3 读区 0.62 vs 尾部 0.85–0.999, 采样种子 0.999, 见 09-04 probe json）[数据]。

## 3. 判读

### 3.1 信息在哪一层、丢在哪一跳 [数据]
P(7.5) → h(22) 的 ~14.5 差 = register 压缩/读出/累加路径的净损失。h≈trained → **解码器已把 z_s 用到了它自身上限**, 不在 z_s 里剩一堆"等一个更好的头"。真正未被利用的, 是**编码器输出层 P 的 patch 级信息——它根本没进 register 的可用通道**（解码器只读 z_s）。

### 3.2 register 塌缩与"分工压力" [数据+推断]
slice05 z_s within-std 0.053（vs P 0.171）= register 之间高度同质; slice27_v2 0.143 相对健康但仍 < P。结合 09-04 探针 cos 结构, 因果/分块训练只给 step-1 前缀区强梯度 → 多样性预算集中前缀、后区塌缩为冗余簇 [推断, 与旧判读一致]。h 的线性上限 ~21–23 与 S25/S64 单步直读锚点 19–20 同量级, 说明"固定模板读 register"在这个训练状态下无法展开 P 级细节。

### 3.3 对 ANALYSIS §5 判读表的实证落点
- "z_s→像素 ≈9–14 → 解码浪费"：**不成立**（该判读把旧架构 patch 对齐 z_s 误搬到 register 架构）。h≈trained 证明解码侧没浪费线性可及的信息。
- "≈19–20 → 解码器没浪费多少"：**成立且更强**（h≈trained）。
- 真正结论应改写为：**信息在 P（7.5）, 卡在 register 路由/读出这一跳（→h 22）**。修法顺位不变：E2'/F1（把 P 注入 specials, 让 register 真正分工）或 v4 式单发读出（实证 8.26）, 均已在 ANALYSIS/方案矩阵。

### 3.4 对 09-04 探针 json 的一个取证备注
服务器 `output/probe_slice05_exp1_results.json`（09-04 现场产物, 非 doc 提交版）中 'trained' 项（norm 0.35988 / 像素 20.640）与本次 slice27_v2 的 trained 数值逐位一致, 而 doc 提交版（`doc/2026-09-04/probe_slice05_exp1_results.json`）= slice05 真值 21.299 —— 说明服务器工作副本曾被某个 slice27_v2 的探针运行覆盖/错标, 引用 512 子集"trained=20.64"时应以 doc 提交版与本次 E1 为准（slice05=21.30, slice27_v2=20.64）。

## 4. 结论与下一步

1. **"信息没被利用"位置锁定 = 编码器输出 P → register z_s 路由/读出**; E1 给出数字: P 线性 7.5 vs 解码后 h 22。
2. **若走 v4 式单发读出（推荐, GOAL 对齐）**: 直接用 v2 编码器配置单变量对照（~2 GPU·h）验证能否进 8–14 —— E1 显示 P 有 7.5 的信息 + v4 实证 8.26, 预期乐观。
3. **若坚持时间轴/渐进**: E2'/F1 编码侧注入是让 register 分工的正路; 另可加 register 正交/反塌缩正则（防 slice05 式 within-std 0.053 退化）。
4. E1 全部为只读/零训练; 无代码改动; 服务器产物: `output/probe_e1_{slice27_v2,slice05}.json`（另存 doc 数据目录）。

## 5. 参考

- 设计: `doc/2026-09-04/ANALYSIS_v2_info_left_and_decoder.md` §5
- 旧口径: `doc/2026-08-27/trace_info_pixel.py`（旧 ReEncoder 架构, z_s 逐 patch 对齐）
- 同子集旧探针: `doc/2026-09-04/probe_slice05_exp1_results.json`（slice05 真值 21.299）、`PROBE_v2_slice27_v2_mechanism.md`
- 独立复核: `doc/2026-09-07/ANALYSIS_k3_14to20_review.md`（Kimi K3）
