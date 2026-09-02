# v2 num_specials(K) 由解码器最终采样步集推导 — 消除"花瓶 register"设计

> 日期: 2026-09-02 ｜ 分支: `main`（HEAD 1c5b3c2 之上）｜ 性质: 设计 + 实现记录
> 涉及: `model_v2.py` / `train_v2.py` / `infer_v2_test.py` / `visualize_recon_pixel.py` / `run_v2_boundary.sh`
> 背景: `doc/2026-09-02/REPORT_v2_single_step_retrain.md`（S25/S64 单步重训练消融）、`doc/2026-09-02/ANALYSIS_v2_all_failures.md`

---

## 0. TL;DR

- **问题**：register 式模型里编码器固定生成 N 个 register，但训练只读部分采样步时（如 slice `[4:9]` → steps=`[25,36,49,64,81]`），位置 `100..N` 的 register **从不被解码器读**（"花瓶 register"）。实验证明它们不是摆设：参与训练动力学并干扰读窗口。
- **修复**：K（num_specials）与 N（num_patches）**解耦**——K = 解码器实际读取的 register 范围，由**最终生效采样步集**自动推导：
  `K = min( max_{t∈steps}((⌊√t⌋+1)²−1), N )`。编码器 register 数 = 解码器读的范围，**不存在花瓶**；K<N 时同时实现真压缩。
- **向后兼容**：无 decoder_steps / 无 skip/max 切片（默认全量）时 K=N，与历史行为一致；旧 checkpoint 需在**同 K** 下才能 strict load。

---

## 1. 动机：花瓶 register 的耦合证据

register 式（`model_v2.py` HEAD @1c5b3c2）的现状：

- `SpecialTokenBank(num_patches=N)` 生成 N 个 register；DINO 输入序列 `[cls; specials(N); patches(N)]` = 2N+1 token；
- `OutputQueryDecoder` 只在采样步（`square_block_starts(N)` 切片后的子集）读 z_s 前缀；`build_block_mask` 里每步只见自己的平方块；
- **当采样步只覆盖部分 register 时，其余 register 编码器仍生成、解码器从不读**——它们是"花瓶"。

花瓶不是摆设，证据（用户实验 + 本地 toy 复现）：

| 测量 | 读窗口 register（pos 被读） | 花瓶 register（pos 从不被读） |
|---|---|---|
| 训练梯度（读窗口内参数梯度 |Σ|） | 6.35 | **9.81**（更大） |
| 逐维扰动花瓶输入 → 读窗口输出的相对变化 | — | **~68%** |

机制：DINO 24 层内**全双向注意力**（HF 无 token 级 mask API，register 惯例），前向/反向都跨全部 token 耦合——花瓶 register 虽不在解码器掩码的"读窗口"内，却经中间层与读窗口 register 共享注意力路径，收到大梯度、扰动传播回读窗口（本地 toy 已复现该耦合）。结论：**花瓶参与训练动力学并干扰读窗口，应消除而非容忍**。

**设计结论**：K 与 N 解耦——N 只管输入 patch 数与解码查询行数（每步输出 N 个 patch 预测不变），K 只管键数/register 数，且 K = 解码器实际读的范围（自动由最终采样步集推导），编码器侧不产生任何解码器读不到的 register。

---

## 2. K 的推导公式（权威）

令最终生效的采样步集为 steps（显式 decoder_steps，或 `square_block_starts` 切片后的子集）。步 t 所在的平方块 k=⌊√t⌋ 的**块终点** = `(k+1)²−1`：

```
K = min( max_{t∈steps} ( (⌊√t⌋+1)² − 1 ), num_patches )
```

实现于 `model_v2.py::derive_num_specials(num_patches, steps)`（模块级 helper）；steps 为空 → 返回 num_patches。

### 2.1 示例

| 配置 | 最终 steps | 推导 | K |
|---|---|---|---|
| N=576, 默认全量（无 decoder_steps/slice） | `square_block_starts(576)` = [1,4,…,529] | max t=529 → (24)²−1=575 → min(575,576) | **576 = N**（向后兼容） |
| N=576, slice [4:9] | [25,36,49,64,81] | max t=81 → (9+1)²−1=99 → min(99,576) | **99**（读 1..99 全被覆盖） |
| N=576, 显式 steps=[64]（S64 实验） | [64] | (⌊√64⌋+1)²−1 = 80 | **80** |
| N=16, slice [1:3] | [4,9] | max t=9 → (3+1)²−1=15 → min(15,16) | **15** |

### 2.2 数学性质与限制

- **K ≥ max(steps)**：对任意 t≥1，(⌊√t⌋+1)²−1 ≥ t（因 t < (⌊√t⌋+1)²），故 K ≥ max(steps)——每个采样步 t 本身都是合法的 z_s 位置（z_s 共 K 个，位置 1..K，0 是 z_cls）。
- **square 连续切片全覆盖**：对 `square_block_starts` 的连续切片（如 [4:9]、[1:3]），register 1..K 的每个位置至少被某个步允许（第一个步的前缀规则覆盖开头、各块连续铺满）——**无花瓶**（自检含覆盖断言）。
- **限制**：显式**非平方/非连续** steps 不保证全覆盖（如 steps=[32,64] 时 36..63 无人读）。这是显式步集的固有语义（用户显式指定步即承担该语义），文档注明即可；自动推导分支（默认）只出现在 square 切片上，恒无花瓶。
- K 推导**只依赖步集本身，与 skip_steps 起点无关**（skip/max 只决定"从哪个块开始切"，K 由切完后的实际步值决定）。

---

## 3. 改动清单

### 3.1 `model_v2.py`

| 项 | 改动 |
|---|---|
| 文件头 docstring | 架构段更新为 register 式 + K 推导（公式 + 花瓶动机 + 示例）；序列 = 1+K+N token |
| `SpecialTokenBank.__init__` | 参数 `num_patches` → **`num_tokens`**（K 与 N 解耦后不再恒等；无外部直接构造者，doc 历史脚本只引用 `model.special_bank` 属性，保留不变） |
| `square_block_starts` | 不变（已通用：入参 = 分块计划上界） |
| **新增 `derive_num_specials(num_patches, steps)`** | K 推导 helper（公式见 §2，docstring 写全动机/公式/示例） |
| **新增 `select_steps(num_patches, decoder_steps, skip_steps, max_steps)`** | 最终生效步集 helper：显式原样 / 默认计划 + skip/max 切片；`SRPhase1V2`（上界=N，切片先于 K 推导）与 `OutputQueryDecoder` 独立用法（上界=K）共用，两处结果一致 |
| `build_block_mask` | 参数 `num_patches` → `num_tokens`（语义 = z_s token 数 K，掩码列数 = K+1）；新增 `num_queries`（默认 = num_tokens，K<N 时解码器传 N——掩码高度按每步 N 行 patch 查询铺） |
| `OutputQueryDecoder.__init__` | 新增 `num_specials: Optional[int] = None`（None → num_patches，旧行为）；`S = num_specials+1`、`pos_embed (1,S,D)`；`query_base` 行数 = N 不变；steps 走 `select_steps(K, …)`；断言 steps 非空、0≤s≤K |
| `OutputQueryDecoder.forward` | 断言 z_s 列数 == num_specials；掩码 `build_block_mask(num_specials, steps, num_queries=N)` |
| `SRPhase1V2.__init__` | 新增 `num_specials=None`（None=自动）；先 `steps_selected = select_steps(N, …)`，再 `K = derive_num_specials(N, steps_selected)`（显式则用给定 K）；断言 1≤K≤N、max(steps)≤K（K 过小给清晰报错）；`self.num_specials=K`；`SpecialTokenBank(num_tokens=K)`；decoder 收显式 `steps=steps_selected` + `num_specials=K`（构造方式写入 docstring） |
| `_encode_register` | 序列 `[cls; specials(K); patches(N)]`；返回 `z_s = seq[:,1:1+K]` |
| `__main__` 自检 | 保留全量默认 K==N 断言；新增 2b（slice[1:3]→K=15：z_s 形状 / 掩码 (|T|·16, 16) / 覆盖断言 / 梯度全通）+ 2c（显式 K=8+steps=[4]；K 过小报错信息核对） |

### 3.2 `train_v2.py`（修复与 model_v2.py 的接口脱节）

- **删除** `--reencoder_depth` / `--no_causal_specials` / `--register_specials` 与 `model.init_reencoder_from_dino(...)`（register-only model_v2.py 无这些接口，旧代码直接 TypeError/AttributeError）。
- 文件头 docstring 对旧 ReEncoder 架构、"`--num_specials`（默认 128）/ `--loss_min_t`（默认 5）"的过时描述全部改写为真实描述（这两个参数旧 parse_args 里根本不存在）。
- **新增** `--num_specials type=int default=0`（0 = 自动按公式由 slice_end/max_steps 推导）；传 `num_specials=(args.num_specials or None)`。
- 打印：specials 数改打 `model.num_specials`（不再打 num_patches）；序列 token 数 = `1+K+N`。
- `model_info.json` **新增 `"num_specials": model.num_specials`**（推理/可视化按它对齐权重形状——K 错了 checkpoint 形状就对不上，strict load 即崩）；删除 `reencoder_depth`/`causal_specials`/`register_specials` 字段。

### 3.3 `infer_v2_test.py` / `visualize_recon_pixel.py`

- 删除 `--reencoder_depth` / `--register_specials`（已删接口），模型构造不再传。
- K 复现训练值：**优先读 model_info.json 的 `num_specials`**（新训练必写）；没有则 `--num_specials` CLI（默认 0=auto，与训练一致的 slice/decoder_steps 下自动推导结果相同）。
- 加载后与 model_info.json 对比 num_specials / decoder_steps，不一致给出 warn；旧产物（无 num_specials 字段）打印提示：如需复现旧 K=N 权重，显式 `--num_specials N`。
- infer 输出 json 增加 `num_specials`。

### 3.4 `run_v2_boundary.sh`

- 删除 `--loss_min_t 5` / `--register_specials`；保留 `--num_specials 64 --decoder_steps "32,64"`（显式 K=64，max(step)=64 ≤ K ✓）；头注释更新为 2026-09-02 语义。

### 3.5 未动

- `doc/2026-08-26`、`doc/2026-08-27` 下历史诊断脚本（check_leak.py / cos_eval.py / pixel_recon_check.py / param_count.py / verify_visual.py / trace_info_pixel.py / visualize_recon.py / visualize_constant_check.py / sim_vs_t.py / infer_k_sweep.py 等）：只引用 `model.special_bank` 属性与**旧 API**（reencoder_depth/k_list 等），在 HEAD（register-only）上本就已崩——历史产物，不在本改动范围，报告里列出即可。
- v3/v4/v5 任何文件不动。

---

## 4. CLI 用法与语义

```bash
# 自动推导（默认, 推荐）: K = derive_num_specials(N, 最终采样步集)
python train_v2.py --data_dir ... --dino_dir ... \
    --slice_start 4 --slice_end 9          # N=576 → steps=[25,36,49,64,81] → K=99

# 全量默认（无 slice/decoder_steps）: K=N（历史行为, 向后兼容）
python train_v2.py --data_dir ... --dino_dir ...

# 显式覆盖（复现旧 checkpoint / 手动限 K）: 断言 max(采样步) ≤ K
python train_v2.py --data_dir ... --num_specials 64 --decoder_steps "32,64"

# 推理/可视化: 不传 K（读 model_info.json）即可; 旧产物用 --num_specials N
python infer_v2_test.py --data_dir ... --final_model output/.../final_model.pt ...
python visualize_recon_pixel.py --data_dir ... --final_model output/.../final_model.pt ...
```

模型侧断言（`SRPhase1V2.__init__`）：
- `1 ≤ K ≤ N`；
- `max(steps_selected) ≤ K`——显式 K 过小时报错信息明确（"num_specials(K)=… 过小: 采样步最大 … > K"），提示加大 K 或缩小 steps/slice。

---

## 5. 兼容性说明

- **权重形状随 K 变**：`special_bank.pos (1,K,D)`、`decoder.pos_embed (1,K+1,D)`、DINO 输入序列 1+K+N。同 N 不同 K 的 checkpoint 之间 strict load 形状不匹配。
- **旧 checkpoint**：register 时代（HEAD 之前）任何训练都是 K=N=576（含 slice [4:9]、S25、S64 实验——它们的权重仍是 K=N）。**同 K 才能 load**：
  - 默认全量训练的新 checkpoint：K=N，与旧全量模型互载 OK；
  - 旧**切片**训练产物：须显式 `--num_specials 576`（=N）复现旧 K（新默认自动推导会给 99≠576）；
  - 旧产物 model_info.json 无 `num_specials` 字段，infer/visualize 已打印此提示。
- 反向：用旧工具链读新 checkpoint 同样会因形状不同失败（本仓库推理入口均已按 model_info.json 对齐，正常路径不会踩到）。

---

## 6. 后续建议实验（可选）

1. **重跑 S25/S64 式单步消融**（`REPORT_v2_single_step_retrain.md` 实验二）：现在单步 steps=[25] → K=35、steps=[64] → K=80（而非花瓶式 K=576）——训练更干净（无花瓶梯度干扰），验证 K 解耦是否改善单步重建（旧 19.15/19.58）。
2. **K 扫描对照 v3/v4**：同 slice 下自动 K（如 [4:9]→K=99）与显式更小 K（64/128）对比重建 L1，把"K 压缩 × 重建质量"曲线与 v3（BLIP-2）/v4（registerK）对齐口径。
3. 读窗口寄存器经扰动/梯度度量复测（花瓶证据 §1 的干净版）：K 解耦后扰动任意 register 都落在读窗口内，测量其梯度/敏感性分布是否更均匀。

---

## 7. 诚实边界（本次改动不承诺的）

- **花瓶干扰从未被量化到"端到端损失贡献"**：§1 的 6.35 vs 9.81 与 ~68% 是耦合强度证据，不是"花瓶贡献了多少最终像素 L1"的归因实验。本次改动**消除花瓶通道本身**（结构上不可能再有），但不承诺单独解决 **S64(19.58) vs v4(8.26) 的全部差距**——该差距的其余变量仍在：MLP 头 / 解码器结构（v2 OutputQueryDecoder vs v4 方案）/ 分组学习率等，均不在本改动范围。
- 显式非 square steps 不保证 register 全覆盖（§2.2），该语义由显式步集承担。

---

## 8. 相关文件

- `model_v2.py`（`derive_num_specials` / `select_steps` / `build_block_mask` / `OutputQueryDecoder` / `SRPhase1V2`；文件头架构段）
- `train_v2.py`（`--num_specials`；`model_info.json` 加 `num_specials`）
- `infer_v2_test.py` / `visualize_recon_pixel.py`（model_info.json → K 对齐）
- `run_v2_boundary.sh`（旧参数清理）
- 背景：`doc/2026-09-02/REPORT_v2_single_step_retrain.md`、`doc/2026-09-02/ANALYSIS_v2_all_failures.md`
