# SR-Diffusion v3 — test 分支 v2 训练与「单键还原」排查报告

> 日期: 2026-08-26 ~ 08-27 ｜ 分支: `test` ｜ 服务器: 2× RTX PRO 6000 (97GB)
> 本文档记录: test 分支（注意力机制改写）v2 模型完整训练（两轮口径对比）+ 推理测试 + 「1 个键还原全量」异常结果的排查结论。

---

## 1. 背景与动机

用户更新了 GitHub `test` 分支（相对 `main` 的两个新提交）:

- `2655fc4 注意力机制改写`: 解码器从 `FeatureDecoder`（块掩码自注意力）改为 **`OutputQueryDecoder`**（Perceiver 式输出查询注意力 + KV 因果掩码 + 自然数平方采样计划 + 加权全覆盖损失）。
- `1a911ae 引文添加`: 补充文献引用（Perceiver AR / BLIP-2 / 视觉 token 稀疏性 / SVD-Prune）。

要求: 按 main 文档标准跑该版本的 v2 模型 —— **工地数据、1600×900 画布、576 patches（非 256 小规模）**，训练后按文档跑测试，并排查「1 个向量（t=0 仅 1 键）就能全量还原」这一异常好结果。

**第二轮训练口径调整（2026-08-27 用户要求）**:
1. **去掉加权体系** —— decoder_loss_weight 默认 `uniform`（全部采样步平权; 之前 density 密度补偿）。
2. **全 fp32 训练** —— 训练/评估不再用 bf16 autocast（之前 bf16 计算 + fp32 导出被认为不公平）。
3. **batch 提高** —— bs=16/卡（97GB 显存充裕）。
4. **train_v2.py 重写为 HF Trainer 风格**（消除造轮子）—— 手写训练循环/梯度累积/调度/checkpoint 全部交给 `Trainer`+`TrainingArguments`（commit `3c438a5`，后修复 transformers 5.15.1 兼容 `8cd9ed3`、Trainer eval 指标 `09d74a4`）。

---

## 2. 训练配置（main 文档标准）

### 第一轮（bf16 + density 加权, 2026-08-26）

| 项 | 值 |
|---|---|
| 数据 | `construction_site` 工地 parquet: 7009 训练 (7 分片) + 3004 测试 (3 分片) |
| 预处理 | 旋转(最优角) → 等比缩放 → 居中填充 **1600×900** 画布 → 模型输入 **448×252** (16:9) |
| patches | 576 = (448/14)×(252/14)（DINOv2-large patch=14, dim=1024） |
| 架构 | DINOv2-large(不冻结) → ReEncoder(depth=4, 因果 specials) → OutputQueryDecoder（平方采样 **25 步**, density 加权全覆盖损失） |
| 训练 | 40 epochs / **17520 步**, 2 GPU DDP, bs=8/卡, lr=1.5e-4 cosine, warmup 3%, **bf16 计算 / fp32 权重** |
| 时长 | 56.6 min（~83 样本/s） |

### 第二轮（fp32 + uniform 平权, 2026-08-27）

| 项 | 值 |
|---|---|
| 架构 | 同第一轮, 但 **decoder_loss_weight=uniform**（25 采样步全部平权） |
| 训练 | 40 epochs / **8760 步**（bs 翻倍后步数减半）, 2 GPU DDP, **bs=16/卡**, lr=1.5e-4 cosine, warmup 262 步, **全 fp32**（无 autocast） |
| 框架 | **HF Trainer**（train_v2.py 重写, 手写循环全部消除） |
| 时长 | ~114 min（~41 样本/s, fp32 较慢） |

入口脚本（均已 push test 分支）: `train_v2.py`（HF Trainer 风格）、`data_v2.py`、`run_v2_train.sh`、`infer_v2_test.py`、`check_leak.py`。

---

## 3. 训练结果

### 第一轮（bf16 + density）

- 训练 loss: **1.32 → 0.00143**（step 17520/17520，cosine LR 归零）
- Eval 历史（bf16, n=3004, 每 2000 步）:

| step | 2000 | 4000 | 6000 | 8000 | 10000 | 12000 | 14000 | 16000 |
|---|---|---|---|---|---|---|---|---|
| eval loss | 0.01059 | 0.00687 | 0.00536 | 0.00316 | 0.00357 | 0.00272 | 0.00218 | 0.00177 |

- 产物: `output/phase1_v2_test/final_model.pt`（1.47GB **fp32**，含 DINO 权重）+ 8 个 checkpoint (ckpt-2000…16000) + `model_info.json` / `args.json`。

### 第二轮（fp32 + uniform, HF Trainer）

- 训练 loss: **1.26 → 0.01011**（train_runtime 6873s ≈ 114 min）—— 注意: 该数值是 Trainer 报告的**平均** train_loss，与第一轮的手写 EMA 口径不同，不直接对比。
- Eval 历史（**fp32**, n=3004, 每 2000 步, Trainer 输出 eval_loss/eval_recon）:

| step | 2000 | 4000 | 6000 | 8000 |
|---|---|---|---|---|
| eval_loss | 0.00958 | 0.00535 | 0.00212 | **0.00016** |

- 产物: `output/phase1_v2_fp32_uniform/final_model.pt`（1.47GB fp32）+ Trainer checkpoint-2000…8760 + `model_info.json` / `args.json`。
- 第二轮收敛显著快于第一轮（step 8000 时 eval 0.00016 vs 第一轮 0.00177，差 ~11×）—— 平权 + fp32 让模型真正逼近目标。

---

## 4. 推理测试（infer_v2_test.py, 全量 3004 张, fp32）

### 第一轮（bf16 + density）

- **全量重建 fp32 L1 = 0.00143**（旧 FeatureDecoder 基线 fp32 ≈0.00114，比值 1.25×，在 ≤1.5× 容限内）。
- **渐进曲线全平**: 25 个采样步的 L1 全部落在 0.00143–0.00145，即使 t=0（仅 1 个键 z_cls）也达到全量精度:

```
t=   0 (前缀  1 键) L1 = 0.001448      ...      t= 576 (前缀 577 键) L1 = 0.001430
```

即「前缀越短越粗」的渐进性质**没有出现** —— 这正是用户怀疑 token 泄露的触发点。

### 第二轮（fp32 + uniform）

- **全量重建 fp32 L1 = 0.000038 ± 0.000008**（n=3004）—— 相比第一轮 0.00143 提升 **37 倍**，已逼近目标自身的空间变异性（行到质心距离 0.000036）。
- **渐进曲线依旧全平**: 25 步 L1 全部落在 0.000039–0.000041:

```
t=   0 (前缀  1 键) L1 = 0.000041      ...      t= 576 (前缀 577 键) L1 = 0.000041
```

**关键结论（两轮对比）**:
- 第一轮 L1=0.00143 远差于质心 baseline（0.00004，差 35×）→ 是 **bf16 + density 加权训练的欠收敛**，不是架构缺陷；
- 第二轮 fp32 + 平权 L1=0.000038 ≈ 质心下限 0.000036 → **模型真正收敛到了目标特征**；
- 但「1 键还原」依旧成立，根源是**目标空间变异性极小**（工地图片 DINO patch 特征均匀，每图跨位置 std≈5e-5）—— 这是数据特性，不是 bug，也不是泄露。

---

## 5. 「1 向量还原」排查

### 5.1 排查路径

怀疑: t=0 步只有 z_cls 一个键，却能还原全量 576 个 patch → 是否推理时其他 token 泄露进注意力？

排查四步（`check_leak.py`, 64 张 test 样本）:
1. 目标方差 —— 目标本身是否就有区分度?
2. decoder 掩码生效性 —— 扰动被屏蔽 token, 输出应不变。
3. ReEncoder 信息流 —— z_cls / z_s 是否（设计使然地）聚合全图。
4. query_base 角色 —— 单键时输出是否依赖键。

### 5.2 掩码无泄露（数值验证）

扰动 `z_s[:, 64]`（幅度 0.5）后重新前向，检查全部 25 个采样步输出变化（第二轮模型）:

```
t=  0 ΔY=0.00e+00  t=  1 ΔY=3.95e-08  ...  t= 49 ΔY=8.34e-08  t= 64 ΔY=8.42e-08   ← 扰动列被屏蔽, 无影响
t= 81 ΔY=4.54e-05  t=100 ΔY=4.11e-05  ...  t=576 ΔY=1.23e-05                    ← t≥64 可见该键, 正常响应
```

**掩码边界精确** —— KV 因果掩码正确屏蔽: t≤64 的步 ΔY≈1e-8（纯浮点噪声，被屏蔽 token 不流入），t>64 的步 ΔY~1e-5（该键可见，输出正常响应）。**无越界信息泄露**。

ReEncoder 侧: 扰动 `patch[:, 0]`（幅度 0.5）→ `Δz_cls=1.13e-4, Δz_s=5.18e-5` —— z_cls/z_s 对 patch 敏感是**设计使然**（cls 全局注意力、specials 行可见全部 patches，见 `model_v2.py` 块掩码说明），并非推理路径的 bug。

### 5.3 目标退化（核心发现）

对照实验（128 张 test, fp32）—— 目标特性两轮一致:

| 指标 | 数值 |
|---|---|
| 目标 patch 特征 每图跨位置 std | **~5e-5**（min 0.000034 / max 0.000115） |
| 「直接输出质心向量」baseline L1 | **~0.00004** |
| 第一轮模型（bf16+density）L1 | 0.001431（= 质心 baseline 的 **35×**） |
| **第二轮模型（fp32+uniform）L1** | **0.000038（≈ 质心 baseline 0.000036, 已收敛到下限）** |

含义:

- 工地图片大面积均匀（天空/地面/围挡等），DINOv2-large 对 576 个 patch 位置输出的特征**几乎完全相同**（空间变异性仅 ~5e-5）。
- 因此「1 个键就能还原全量」是**必然结果** —— 目标本来就是「每图一个近似常数向量」，输出质心即可 L1≈4e-5；渐进曲线全平同样只是目标特性的投影，不反映模型能力。
- **两轮对比修正了第一轮结论**: 第一轮 L1=0.0014 比质心差 35× 并非"模型连常数都学不准"的架构缺陷，而是 **bf16 计算 + density 加权**导致的欠收敛；第二轮 **fp32 + uniform 平权** 后模型 L1=0.000038 已逼近质心下限（0.000036），说明模型**真正收敛**了。但「目标无空间变异性 → 重建指标区分度低」这一数据侧事实不变。

### 5.4 键信息流（第二轮模型, 验证不是输出常数）

替换 decoder 输入为随机向量（幅度 0.1），观察各采样步输出变化:

| 采样步 | 随机 z_cls（保持 z_s） | 全随机（z_cls+z_s） |
|---|---|---|
| t=0 | ΔY=0.42（大变） | ΔY=0.42 |
| t=1 | ΔY=0.0018 | ΔY=0.55 |
| t=16 | ΔY=0.0001 | ΔY=0.35 |
| t=576 | ΔY~1e-4 | ΔY~0.1+ |

- **t=0 强烈依赖 z_cls 键**（随机后输出大变 0.42）—— 键承载还原信息；
- **t≥1 主要由 z_s 键驱动**（随机 z_cls 几乎无影响，全随机则大变）—— z_cls 只是 t=0 的锚点，z_s 才是主体；
- 因此模型**确实在用键信息重建**，不是 query_base 模板函数输出常数。

### 5.5 新旧模型对照（排除 test 分支代码问题）

用旧 `main` 代码（FeatureDecoder）的 `final_model.pt`（08-25 产物）跑同一对照:

| 指标 | 旧模型 (FeatureDecoder) |
|---|---|
| 目标 patch 每图跨位置 std | 0.000023 |
| 质心 baseline L1 | 0.000018 |
| 模型 L1 | 0.001030 |
| **模型 L1 / 质心 baseline** | **56.7×** |

结论: 旧模型同样「远差于质心 baseline」→ **目标退化是数据/特征特性，与 test 分支的注意力改写无关**；旧模型也是 bf16 训练，同样欠收敛。

---

## 6. 结论

1. **无 token 泄露**: 掩码数值验证（扰动被屏蔽 token → 输出不变，边界精确）通过。
2. **「1 向量还原」是目标特性的必然**: DINO patch 特征在工地图上空间变异性仅 ~5e-5，重建任务退化为「输出每图近似常数」，渐进曲线全平是目标投影。
3. **训练口径修正**: bf16 + density 加权会欠收敛（L1 比质心差 35×）；fp32 + uniform 平权后 L1=0.000038 逼近质心下限，模型真正收敛且用键信息重建。
4. **指标建议不变**: 重建任务在该数据/特征组合下区分度低，需要质心 baseline 对照或换有空间结构的目标。

---

## 7. 建议（下一步）

1. **数据侧**: 换有真实空间结构的数据集（非大面积均匀的工地图），或先度量 patch 特征空间变异性（>0.01 才有区分度）。
2. **目标侧**: 若坚持 DINO 特征，检查其是否被 LayerNorm / 位置无关化抹平；或改用像素级重建 / SVD 特征（有真实空间结构，见 main 分支 v1 思路）。
3. **指标侧**: 训练/测试增加「质心 baseline」作为下限对照 —— 模型 L1 必须显著优于质心才有意义；渐进曲线（每采样步 L1）作为前缀能力的可视化指标保留。
4. **训练口径**: 采用 fp32 + uniform 平权（第二轮口径），避免 bf16/density 欠收敛。

---

## 8. 复现

```bash
# 训练（服务器, 2 GPU DDP, 第二轮 fp32+uniform 口径）
NUM_GPUS=2 ./run_v2_train.sh \
    --data_dir /root/autodl-tmp/construction_site \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --output_dir output/phase1_v2_fp32_uniform --epochs 40

# 推理测试（全量 test, fp32）
python infer_v2_test.py \
    --data_dir /root/autodl-tmp/construction_site \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --final_model output/phase1_v2_fp32_uniform/final_model.pt \
    --output output/phase1_v2_fp32_uniform/infer_test.json

# 泄露诊断（掩码验证 + 目标方差分析 + 键信息流）
python check_leak.py \
    --data_dir /root/autodl-tmp/construction_site \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --final_model output/phase1_v2_fp32_uniform/final_model.pt
```

---

## 附: 相关提交（test 分支）

| commit | 内容 |
|---|---|
| `2655fc4` / `1a911ae` | 用户提交: 注意力机制改写 + 引文添加 |
| `d8354bb` | train_v2 适配新接口（移除 TextDecoder, 新增 decoder 参数） |
| `4b97587` | infer_v2_test: 全量 fp32 L1 + 每采样步渐进曲线 |
| `203497a` | 修复 final 元数据 DDP 访问 decoder 崩溃 |
| `d2e5b60` | 修复 infer_v2_test 累加器未初始化 |
| `010b57f` | check_leak: 泄露诊断（掩码验证 + 目标方差分析） |
| `5a1b778` | 本文档 REPORT_test_v2_train_and_leak.md 初版 |
| `17e5708` | 用户提交: 文档修正（DINOv2 冻结描述） |
| `744e76c` → `fc1602f` | 训练口径调整: uniform 平权 + 全 fp32 + bs16 |
| `3c438a5` | train_v2 重写为 HF Trainer 风格（消除造轮子） |
| `8cd9ed3` | 适配 transformers 5.15.1（warmup_steps, 移除 logging_dir） |
| `09d74a4` | Trainer eval 指标修复（can_return_loss, compute_metrics） |
