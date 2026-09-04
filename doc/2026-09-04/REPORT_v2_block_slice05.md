# SR-Diffusion v3 — v2 slice05 实验（exp1）：slice [0:5]（K=35）因果 tgt_mask + 分块读侧，同预算全量训练推理

> 日期: 2026-09-04 ｜ 分支: `main`（训练代码 HEAD `66aa9d2` = `b295016` 语义等价, 仅 Y_cum 行格式差异; 推理代码 `36cf777` = 66aa9d2 + `SRV2_MEMORY_OPEN` 开关, exp1 未设该开关=默认关, 行为一致）
> 服务器: `connect.westd.seetacloud.com:11228`（2× RTX PRO 6000, torch 2.12.1+cu130, transformers 5.16.1）
> 性质: 按任务文档命名惯例 "slice05" = slice_start=0 / slice_end=5（沿用 slice27=[2:7] 的切片口径; 仓库 66aa9d2 文档未定义 slice05, 无冲突）→ steps=[1,4,9,16,25], K 自动 = (⌊√25⌋+1)²−1 = **35**。与历史 slice27/slice27_v2 **完全同预算**, 唯一变量 = slice 窗口（K 35 vs 63）。这是"因果 tgt_mask + 2 层 PixelHead"代码（b295016 起）下 K 更小的第二个数据点, 也是用户指示的**双实验消融（本报告 exp1: 读侧分块; 配套 exp2: 读侧全开仅留因果 mask）**的前半。泄露专项分析见同目录 `ANALYZE_v2_slice05_exp1_leakage.md`。

---

## 0. 结论速览（TL;DR）

1. **全量 3004 test 像素 L1 = 20.554（±7.11）**——slice27_v2（K=63, 同代码）=19.878 差 +0.68（≈5 SE, 可信）; 旧 slice27（K=63, 无因果旧代码）=14.284 差 +6.27。**slice05 是 v2 因果族（K=63/35）与全部单读锚点里最差的一个**, 方向与"step-1 可读键数"严格单调（4 键 < 16 键 < 36/80 键单步, 见 §4）。
2. **渐进曲线依旧全平**: [20.559, 20.549, 20.546, 20.548, 20.554]（极差 0.013）——后 4 步合计影响 < 0.02（<0.1%）, "第一步扛全部"在更小 K 上重演。
3. **无信息泄露**（专项探针 [数据]）: 把 z_s[4..35]（step-1 读区外全部键）置 0 → step-1 输出逐位不变（max|ΔY₁|=0.0）; 删掉后 4 步查询行 → Y₁ 只动 1.7e-4。20.554 = "step-1 用 4 键（z_cls+z_s[1..3]）单步读出"的诚实上限。
4. **为什么 K=35 反而更差**: 唯一有效杠杆 step-1 的可读键从 16 个（前缀 0..15）缩到 4 个（前缀 0..3）; slice27_v2 里保持多样的位置 4..15 在 exp1 里落入"只有输出≈0 的后步能读"的塌缩尾区（z_s[4..35] intra cos 0.85–0.999, 9..35 全 ≥0.99）, 其内容对输出零贡献。
5. **机制浪费比 K=63 更极端**: 35 个 register 里 31 个塌缩冗余; 采样种子 A_t=z_s[4/9/16/25] 两两 cos=0.9986–0.9999; step-1 读区 z_s[1..3] intra 0.62 是全文件多样性最高处——模型把多样性预算全押在唯一被有效读取的 3 个 register 上 [数据]。
6. 判读: 曲线形态/分工修复仍不在掩码侧（与系列既有结论一致）; 本数据点把"压缩方向"的收益边界标定到 K≈63（K=99 18.53 → K=63 14.28(旧代码)/19.88(因果) → K=35 20.55: 因果代码下继续压 K 是**负**方向）。

## 1. 背景与目的

- 系列: v2 register 式（specials 进 DINO 序列, K=num_specials 由最终采样步集自动推导, 无花瓶）, 分块读侧 memory_mask + 查询侧块因果 tgt_mask + 2 层 PixelHead + 累加平权全覆盖损失 + 梯度按步解耦（HEAD 自 b295016 起）。
- 命名: "slice27" = slice_start=2 / slice_end=7 → steps [9,16,25,36,49], K=63。本次任务 "slice05" = slice_start=0 / slice_end=5 → **steps [1,4,9,16,25], K=35**（读 register 1..35, 块 [1,3]/[4,8]/[9,15]/[16,24]/[25,35], 首步前缀 0..3）。
- 目的: 同预算下把压缩窗口从 [2:7]（K=63）压到 [0:5]（K=35）——单变量 = 窗口/K, 回答: K 继续压缩在因果代码下是正是负、曲线形态是否变化; 顺带在更小 K 上复查"渐进全平/泄露"机制结论。
- 依据文档: `doc/2026-09-02/DESIGN_v2_num_specials_from_max_steps.md`（K 推导）; `doc/2026-09-03/REPORT_v2_slice27_causal_mask.md`（slice27_v2 同代码对照）; `doc/2026-09-04/PROBE_v2_slice27_v2_mechanism.md`（探针口径）; 泄露专项 `ANALYZE_v2_slice05_exp1_leakage.md`。

## 2. 配置

| 项 | slice27（旧, b653847） | slice27_v2（b295016） | **slice05（本报告, 66aa9d2）** |
|---|---|---|---|
| slice / decoder_steps | [2:7] / [9,16,25,36,49] | [2:7] / [9,16,25,36,49] | **[0:5] / [1,4,9,16,25]** |
| K (num_specials) | 63 | 63 | **35**（自动, 无花瓶） |
| 读侧 memory_mask | 分块+首步前缀 | 分块+首步前缀 | 分块+首步前缀（同左） |
| 查询侧 tgt_mask | 无（全双向） | 块因果 | 块因果 |
| PixelHead | 单层 Linear | 2 层 MLP | 2 层 MLP |
| step-1 可读键 | 0..15（16 键） | 0..15（16 键） | **0..3（4 键）** |
| 预算 | 8760 步 / bs32 有效 / fp32 / seed42 / decoder_depth 2 | 同左 | 同左（墙钟 1h45m36s） |
| 代码 HEAD | b653847 | b295016 | 66aa9d2（= b295016 语义等价） |

训练: `accelerate launch --multi_gpu --num_processes 2 train_v2.py --data_dir /root/autodl-tmp/construction_site --dino_dir /root/autodl-tmp/models/dinov2-large --output_dir output/phase1_v2_block_slice05 --model_input 448x252 --canvas 1600x900 --angle_step 0.5 --epochs 40 --max_steps 8760 --batch_size 16 --grad_accum 1 --lr 1.5e-4 --weight_decay 0.01 --warmup_ratio 0.03 --grad_clip 1.0 --num_workers 8 --limit 0 --eval_limit 0 --eval_every 2000 --save_every 2000 --log_every 20 --seed 42 --heads 8 --mlp_ratio 4.0 --decoder_depth 2 --slice_start 0 --slice_end 5`（342.0M 可训练参数; DINO 序列 1+35+576=612 token; warmup 262 步）。

## 3. 结果（全量 3004 张 test, fp32 推理）

| 指标 | block K=576 [4:9] | K=99 [4:9] | slice27（K=63, 旧码） | slice27_v2（K=63, 因果） | **slice05（K=35, 因果）** |
|---|---|---|---|---|---|
| 像素 L1 (0-255) | 17.460 | 18.533 | **14.284** (±5.20) | 19.878 (±6.96) | **20.554 (±7.11)** |
| 归一化 L1 | — | — | 0.2493 | 0.3468 | **0.3585** |
| 渐进曲线（极差） | 全平 | 全平 | 全平 (0.02) | 全平 (0.013) | **全平 (0.013)** |

训练期 eval_loss（归一化, n=3004; 2000/4000/6000/8000/8760）:

| step | slice27（旧） | slice27_v2 | **slice05** |
|---|---|---|---|
| 2000 | 0.4540 | 0.5002 | **0.5037** |
| 4000 | 0.3079 | 0.4395 | **0.4225** |
| 6000 | 0.2616 | 0.3743 | **0.3923** |
| 8000 | 0.2495 | 0.3475 | **0.3596** |
| 8760 (final) | 0.2490 | 0.3466 | **0.3583** |

（eval 曲线形态与 slice27_v2 同型、整体右移 ~0.01; 像素 L1 差 0.68 ≈ 5 SE(n=3004, SE≈0.13), 方向可信。）

渐进曲线（0-255, 累积结果）: [20.5587, 20.5489, 20.5456, 20.5476, 20.5544]; 逐步像素量级 step_px_scale = [1.007, 0.063, 0.058, 0.057, 0.056]（后 4 步只有 Y₁ 的 5.6–6.3%）。

## 4. 判读

### 4.1 数量级: 20.554 = "4 键单步读出"的诚实结果（无泄露）

全量 3004 锚点链（全部同代或单步独立重训）: **4 键 20.554 > 16 键 19.878(slice27_v2) > 36 键 19.15(S25) ≈ 80 键 19.58(S64)**——严格单调于 step-1 可读键数。专项探针（ANALYZE doc）用数值不变性封死泄露通道: 读侧/tgt 掩码真实生效, step-1 只依赖自己的 4 键 [数据]。若存在"跨步偷看抬高质量", exp1 应 ≥ slice27_v2（同代码代, 唯一变量=窗口/K）, 实际相反。

### 4.2 为什么压缩到 K=35 是负方向

- slice27_v2 里 step-1 的前缀 0..15 中, 位置 4..15 保持内容多样（intra cos 0.686–0.838）并被 step-1 有效读取; exp1 里这批位置落入 4..35 的塌缩尾区（intra 0.85–0.999, 9..35 全 ≥0.99）, 只被输出≈0 的后步读到——step-1 实际只拿到 1..3 三个多样 register [数据+推断]。
- 即: 压缩窗口前移把"首步前缀"从 16 键削到 4 键, 而因果训练又主动把未被 step-1 读的键压成冗余簇 → K 越小, step-1 可用的多样键越少 → L1 越差。因果代码下 K=63→35 是负方向（+0.68）; 对照旧无因果代码 K=576→63 曾是大正方向（17.46→14.28）——压缩收益在旧代码里来自"强制分工/去花瓶", 因果代码已无该收益可挖, 只剩 step-1 前缀变短的成本 [推断]。

### 4.3 曲线全平与"后步白跑"在 K=35 上重演且更极端

Y₂₊≈0（量级 6%）、种子 z_s[4/9/16/25] 两两 cos 0.9986–0.9999、31/35 register 塌缩（≥0.99）——"放弃后步"的机制在更小 K 上推得更彻底; 曲线形态依旧由损失结构（累加平权）+ z_s 冗余决定, 与 K 大小无关 [数据]。渐进语义的修复仍需编码侧逐 patch 注入（E2'/F1）或单发读出（v4 式）, 本数据点不改写该结论 [推断]。

## 5. 产物

- 训练: `/root/autodl-tmp/sr-diffusion-v2-k99/output/phase1_v2_block_slice05/`（final_model.pt 1.37G / model_info.json / args.json / checkpoint-2000..8000）
- 推理: 同目录 `infer_test.json`（n=3004; full_pixel_l1_255=20.5544, full_pixel_std_255=7.105, step_pixel_l1_255 见 §3, K=35）
- 探针/泄露分析: `doc/2026-09-04/ANALYZE_v2_slice05_exp1_leakage.md` + `probe_slice05_exp1_results.json`; 服务器 `output/probe_slice05_exp1_results.json`
- 配套 exp2（读侧全开仅留因果 mask）: `doc/2026-09-04/REPORT_v2_slice05_memory_open.md`

## 6. 复现

```bash
# 训练（同 §2 命令）; 推理（slice 参数必须与训练一致, 否则 K 断言崩）:
cd /root/autodl-tmp/sr-diffusion-v2-k99
CUDA_VISIBLE_DEVICES=0 python infer_v2_test.py --data_dir /root/autodl-tmp/construction_site \
  --dino_dir /root/autodl-tmp/models/dinov2-large \
  --final_model output/phase1_v2_block_slice05/final_model.pt \
  --output output/phase1_v2_block_slice05/infer_test.json --slice_start 0 --slice_end 5
```
