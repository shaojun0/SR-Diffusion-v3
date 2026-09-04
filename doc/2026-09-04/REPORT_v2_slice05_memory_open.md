# SR-Diffusion v3 — v2 slice05 实验（exp2）：打开读侧（memory/前缀）mask 仅保留查询侧因果 mask —— 训练塌缩实证

> 日期: 2026-09-04 ｜ 分支: `main`（代码 HEAD `36cf777` = 66aa9d2 + `SRV2_MEMORY_OPEN` 读侧掩码总开关, 默认关）
> 服务器: `connect.westd.seetacloud.com:11228`（2× RTX PRO 6000, torch 2.12.1+cu130）
> 性质: 用户指示——"exp1（slice05 因果+分块）跑完后, 无论结果如何, 再实验一下**打开前缀/读侧 mask、仅保留因果 mask**"。即 `doc/2026-09-04/ANALYSIS_v2_three_configs.md` §2.1 **配置① 读法 A**（memory_mask 全 0, 每步全键可及含 z_cls; tgt_mask 块因果保留）的训练实证——该分析标注为 [推断] 预期 ≈19.2–19.5, 本报告给出实测。与 exp1 **完全同预算、唯一变量 = 读侧掩码放开**。两个实验完成后统一写文档（本报告 + `REPORT_v2_block_slice05.md`）。

---

## 0. 结论速览（TL;DR）

1. **训练塌缩, 未收敛**: 全量 3004 test 像素 L1 = **64.62（±10.79）≈ 全图平均色基线水平**（对照 exp1 分块读 20.55）。5 步渐进曲线全部 64.62（完全全平）。
2. **eval_loss 从 epoch≈2 起冻结在 1.138 直到 8760 步**: [1.139, 1.138, 1.138, 1.138, 1.138] @ [2000..8760]（exp1 同期 0.50→0.36）。train loss 短暂学到 0.899（epoch 1.37, lr 峰值附近）后**回升并落入无梯度平台**: grad_norm 从 ~0.5–2 崩到 **0.001–0.006** 且再未恢复。
3. **非实现 bug**（checkpoint 复验 [数据]）: checkpoint-4000 权重在 open/closed 读侧掩码下 eval 均为 1.1209（512 子集）== 训练日志同口径 → 掩码确实全开且生效, 权重确实退化为"读什么键都一样"的平凡解。
4. **与 ANALYSIS_v2_three_configs §2.1 的 [推断] 预期相悖**: 全键可及 + 因果在 slice05/K=35 上不是"回到 ~19.2 读出上限", 而是**分工压力消失 → 训练崩坏**。注意: 该文档预测针对 slice27 窗口（step-1 16→64 键）且明确标注[推断]未训练; 本实测只覆盖 slice05/K=35/该 lr 调度, 外推需谨慎。
5. **机制判读 [推断]**: 读侧放开后每个 register 被全部步、全部查询行共享 → 交叉注意力摊薄 → 单键梯度微弱 → register 失去"排他读出→内容分工/专业化"的训练压力 → z_s 不携带可读差异 → 解码器退化为近常数输出; 峰值 lr 附近进入该退化吸引子后 L1 本应给的 ±1 梯度经塌缩注意力/退化输出回传后湮灭（grad_norm≈0.001）, cosine 衰减下无逃逸动力。
6. 系列意义 [推断]: "掩码/读侧/梯度侧改动变不出不存在的分工"（09-04 ANALYSIS §3）得到更极端的数据支持——读侧从"分块"走向"全开"不是中性/微正, 而是**训练动力学级负向**; 渐进语义的分工只能来自编码侧注入（E2'/F1）或单发读出（v4 式）。

## 1. 背景与目的

- exp1（`REPORT_v2_block_slice05.md`）: slice05 因果+分块读, K=35, 全量 L1 20.554（全因果族最差, 无泄露, "step-1 4 键单步读出"上限）。
- 09-04 分析: 用户问的三配置中, 配置① = "memory 掩码放开（全键可及）, 查询自注意力保留块因果"（读法 A: build_block_mask → 全 0）; 推理期探针曾做同权重 eval 干预（mem_all_open, slice27_v2 权重 20.64→21.84, OOD 伪影不能当训练结论）; 文档标注训练期预期 [推断] ≈19.2–19.5, "真正归因只能重训"。
- 本实验: 把读法 A 落实为一次同预算重训——回答"step-1 可及键从 4 扩到 36（全键）, 训练能否用上更多键把 L1 从 20.55 拉回 ~19.x"。

## 2. 改动与配置

### 2.1 代码改动（36cf777, vs exp1 的 66aa9d2）

| 项 | exp1（66aa9d2） | exp2（36cf777 + env） |
|---|---|---|
| 读侧 memory_mask | 分块 + 首步前缀（-inf 屏蔽） | **全 0 = 每步全键可及（含位置 0 = z_cls）** |
| 查询侧 tgt_mask | 块因果 | 块因果（不变） |
| 梯度按步解耦 / PixelHead / 损失 | 不变 | 不变 |
| 开关 | — | `SRV2_MEMORY_OPEN=1`（build_block_mask 早退返回全零; 默认关向后兼容） |
| 记录 | model_info.json 无该字段 | model_info.json **memory_open=true**; infer 侧按 model_info 强制对齐 |

### 2.2 训练配置（与 exp1 完全同预算, 唯一变量 = 读侧掩码）

slice [0:5] → steps=[1,4,9,16,25], K=35; 2×16 有效 bs32 / 8760 步 / 40 epochs / 全 fp32 / lr 1.5e-4 / wd 0.01 / warmup 3% (262 步) / seed 42 / decoder_depth 2 / 448×252; 墙钟 ~1h46m（11:52→13:38）。

## 3. 结果

### 3.1 eval_loss 轨迹（归一化, n=3004）与 exp1 对照

| step | exp1（分块读） | **exp2（读侧全开）** |
|---|---|---|
| 2000 | 0.5037 | **1.139** |
| 4000 | 0.4225 | **1.138** |
| 6000 | 0.3923 | **1.138** |
| 8000 | 0.3596 | **1.138** |
| 8760 (final) | 0.3583 | **1.138** |

### 3.2 train loss / grad_norm 轨迹（关键形态）

| epoch | train loss | grad_norm | 说明 |
|---|---|---|---|
| 0.09 (step 20) | 1.413 | 0.61 | 初始水平 |
| 0.46–1.37 | 1.09 → **0.899** | 0.29 → 1.98 | lr 爬升至峰值, 短暂学到粗结构 |
| 1.83 (step ~400) | 1.153 | 0.007 | 峰值 lr 后**跳回并进入退化平台** |
| 2.28–18.26 | 1.14–1.17 | **0.001–0.006** | 梯度消失, 冻结, 再未恢复 |

### 3.3 全量 3004 test 推理（fp32, memory_open=True 与训练对齐）

- 像素 L1 (0-255) = **64.62 ± 10.79**（≈ 全图平均色基线 ~61–65 量级 → 平凡解）
- 归一化 L1 = 1.1382; 渐进曲线 = [64.62 ×5]（完全全平）
- 复验: checkpoint-4000 在 open/closed 掩码下 eval 均 1.1209（512 子集）== 训练日志 → 掩码生效、权重平凡 [数据]

## 4. 判读

### 4.1 为什么"放开读侧"会导致训练塌缩（[推断], 需 lr 消融进一步证实）

分块读侧的关键作用不只是"信息隔离", 更是**分工压力**: 每个 register 只被一个步（的一批查询行）读取 → 该 register 的梯度来自固定、少量查询行 → 单键梯度占比高 → register 被迫携带该批查询可用的内容（专业化）。读侧全开后: 每个键被 5 步 × 576 行共享, 注意力在 36 键上摊薄, 单键梯度占比 ≈1/36 → 任何单键都不值得携带独特内容 → 全体 z_s 趋向共享冗余（甚至训练把信息全塞进 z_cls / query_base 模板）→ 解码器输出退化为近常数; 该退化态是自锁吸引子: 键越相似 → 注意力越均匀 → 梯度越弱。exp2 在峰值 lr 附近（epoch ~1.8）落入该吸引子, grad_norm≈0.001（L1 的 ±1 梯度被塌缩注意力/退化输出湮灭）, cosine 衰减下无法逃逸 [推断]。

### 4.2 与文档预期的关系（诚实归因）

- `ANALYSIS_v2_three_configs.md` §2.1 对配置①读法 A 的预期 "≈19.2–19.5 全平, 比现状好 ~0.5" 是 [推断]（基于 eval 干预伪影 + S25/S64 上限外推）, **未料及训练动力学塌缩**; 本实测 [数据] 显示该预期在该配置族上不成立（至少在 slice05/K=35/lr 1.5e-4 下）。
- 可能混淆变量: ① 窗口是 slice05 而非文档假设的 slice27（exp1 也证明 K=35 首步前缀只有 4 键, 分工本就更弱）; ② 峰值 lr 1.5e-4 对开放注意力态是否过大未消融（lr 更低或 warmup 更长是否避免入坑未知）。两者都未跑——用户指示本双实验完成后收尾, 不追加。文档已按 [推断] 分级, 本报告结论同样只覆盖实测配置。

### 4.3 对系列的含义

- "读侧从分块走向全开" = 负方向（exp1 20.55 → exp2 64.6）; "梯度再平衡不是杠杆"（09-01）与"掩码侧改动平移不了曲线形态"（09-03）在"训练能否收敛"这一更底层的问题上同样成立 [数据]。
- 渐进语义若仍要追, 分工只能来自**编码侧注入**（E2'/F1: special 输入拼 Linear(patch_feat)）或**放弃时间轴走单发读出**（v4 式 K≈64）——与本系列 09-04 探针/分析结论一致 [推断]。
- 附注: exp2 同时证明"分块读"在因果代码下虽浪费（后步/尾部 register 白跑）但至少保住了训练稳定性——它是"分工压力"的最小可行实现 [推断]。

## 5. 产物

- 训练: `/root/autodl-tmp/sr-diffusion-v2-k99/output/phase1_v2_block_slice05_open/`（final_model.pt 1.37G / model_info.json(memory_open=true) / args.json / checkpoint-2000..8000）
- 推理: 同目录 `infer_test.json`（n=3004; full_pixel_l1_255=64.62, full_pixel_std_255=10.79, 曲线 64.62×5）
- 复验脚本: 服务器 `eval_ckpt_check.py`（checkpoint-4000 open/closed 对比）
- 对照: exp1 `REPORT_v2_block_slice05.md`; 配置分析 `ANALYSIS_v2_three_configs.md` §2.1/§3

## 6. 复现

```bash
export SRV2_MEMORY_OPEN=1   # 训练与推理同开关; infer 侧也可不设(自动按 model_info 对齐)
# 训练同 exp1 命令, 仅 --output_dir output/phase1_v2_block_slice05_open;
CUDA_VISIBLE_DEVICES=0 python infer_v2_test.py --data_dir /root/autodl-tmp/construction_site \
  --dino_dir /root/autodl-tmp/models/dinov2-large \
  --final_model output/phase1_v2_block_slice05_open/final_model.pt \
  --output output/phase1_v2_block_slice05_open/infer_test.json --slice_start 0 --slice_end 5
```
