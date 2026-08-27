# SR-Diffusion v3 — test 分支 v2 训练与「单键还原」排查报告

> 日期: 2026-08-26 ~ 08-27 ｜ 分支: `test` ｜ 服务器: 2× RTX PRO 6000 (97GB), torch 2.13
> 本文档记录: test 分支（注意力机制改写）v2 模型完整训练 + 推理测试 + 「1 个键还原全量」异常结果的排查结论。

---

## 1. 背景与动机

用户更新了 GitHub `test` 分支（相对 `main` 的两个新提交）:

- `2655fc4 注意力机制改写`: 解码器从 `FeatureDecoder`（块掩码自注意力）改为 **`OutputQueryDecoder`**（Perceiver 式输出查询注意力 + KV 因果掩码 + 自然数平方采样计划 + 加权全覆盖损失）。
- `1a911ae 引文添加`: 补充文献引用（Perceiver AR / BLIP-2 / 视觉 token 稀疏性 / SVD-Prune）。

要求: 按 main 文档标准跑该版本的 v2 模型 —— **工地数据、1600×900 画布、576 patches（非 256 小规模）**，训练后按文档跑测试，并排查「1 个向量（t=0 仅 1 键）就能全量还原」这一异常好结果。

---

## 2. 训练配置（main 文档标准）

| 项 | 值 |
|---|---|
| 数据 | `construction_site` 工地 parquet: 7009 训练 (7 分片) + 3004 测试 (3 分片) |
| 预处理 | 旋转(最优角) → 等比缩放 → 居中填充 **1600×900** 画布 → 模型输入 **448×252** (16:9) |
| patches | 576 = (448/14)×(252/14)（DINOv2-large patch=14, dim=1024） |
| 架构 | DINOv2-large(不冻结) → ReEncoder(depth=4, 因果 specials) → OutputQueryDecoder（平方采样 **25 步**, density 加权全覆盖损失） |
| 训练 | 40 epochs / **17520 步**, 2 GPU DDP, bs=8/卡, lr=1.5e-4 cosine, warmup 3%, bf16 计算 / fp32 权重 |
| 时长 | 56.6 min（~83 样本/s） |

入口脚本（均已 push test 分支）: `train_v2.py`（适配新接口，移除 TextDecoder，新增 heads/mlp_ratio/causal_specials/decoder_steps/decoder_loss_weight）、`data_v2.py`、`run_v2_train.sh`。

---

## 3. 训练结果

- 训练 loss: **1.32 → 0.00143**（step 17520/17520，cosine LR 归零）
- Eval 历史（bf16, n=3004, 每 2000 步）:

| step | 2000 | 4000 | 6000 | 8000 | 10000 | 12000 | 14000 | 16000 |
|---|---|---|---|---|---|---|---|---|
| eval loss | 0.01059 | 0.00687 | 0.00536 | 0.00316 | 0.00357 | 0.00272 | 0.00218 | 0.00177 |

- 产物: `output/phase1_v2_test/final_model.pt`（1.47GB **fp32**，含 DINO 权重）+ 8 个 checkpoint (ckpt-2000…16000) + `model_info.json` / `args.json`。
- 已知问题: 训练收尾写 `model_info.json` 时因 DDP 包装模型访问 `model.decoder` 崩溃（`final_model.pt` 已先保存，模型完好）；已修复（`203497a`: 先 `unwrap_model` 再取 `decoder.steps`）。

---

## 4. 推理测试（infer_v2_test.py, 全量 3004 张, fp32）

- **全量重建 fp32 L1 = 0.00143**（旧 FeatureDecoder 基线 fp32 ≈0.00114，比值 1.25×，在 ≤1.5× 容限内）。
- **渐进曲线全平**: 25 个采样步的 L1 全部落在 0.00143–0.00145，即使 t=0（仅 1 个键 z_cls）也达到全量精度:

```
t=   0 (前缀  1 键) L1 = 0.001448      ...      t= 576 (前缀 577 键) L1 = 0.001430
```

即「前缀越短越粗」的渐进性质**没有出现** —— 这正是用户怀疑 token 泄露的触发点。

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

扰动 `z_s[:, 64]`（幅度 0.5）后重新前向，检查全部 25 个采样步输出变化:

```
t=  0 ΔY=0.00e+00  t=  1 ΔY=3.43e-08  t=  4 ΔY=4.90e-08  ...  t=576 ΔY=3.67e-08
```

**所有步 ΔY ≈ 1e-8（纯浮点噪声）** —— KV 因果掩码正确屏蔽，被屏蔽 token 完全不流入输出，**无越界信息泄露**。

ReEncoder 侧: 扰动 `patch[:, 0]`（幅度 0.5）→ `Δz_cls=6.7e-5, Δz_s=4.75e-5` —— z_cls/z_s 对 patch 敏感是**设计使然**（cls 全局注意力、specials 行可见全部 patches，见 `model_v2.py` 块掩码说明），并非推理路径的 bug。

### 5.3 目标退化（核心发现）

对照实验（128 张 test, fp32）:

| 指标 | 数值 |
|---|---|
| 目标 patch 特征 每图跨位置 std | **0.000057**（min 0.000034 / max 0.000115） |
| 「直接输出质心向量」baseline L1 | **0.000040** |
| 模型实际重建 L1 | 0.001431 |
| **模型 L1 / 质心 baseline** | **35.4×** |

含义:

- 工地图片大面积均匀（天空/地面/围挡等），DINOv2-large 对 576 个 patch 位置输出的特征**几乎完全相同**（空间变异性仅 ~5e-5）。
- 因此「1 个键就能还原全量」是**必然结果** —— 目标本来就是「每图一个近似常数向量」，输出质心即可 L1≈4e-5；渐进曲线全平同样只是目标特性的投影，不反映模型能力。
- **更扎心**: 模型 L1=0.0014 比「输出质心」还差 **35 倍** —— 模型连常数都没有学准，重建指标在「工地数据 + DINO 特征」组合下**已失去意义**（t=0 步输出行间 std=0.000000，即 576 行输出完全相同，而目标行间 std 也只有 5.7e-5，二者同为常数级，仅偏置不同）。

### 5.4 新旧模型对照（排除 test 分支代码问题）

用旧 `main` 代码（FeatureDecoder）的 `final_model.pt`（08-25 产物）跑同一对照:

| 指标 | 旧模型 (FeatureDecoder) |
|---|---|
| 目标 patch 每图跨位置 std | 0.000023 |
| 质心 baseline L1 | 0.000018 |
| 模型 L1 | 0.001030 |
| **模型 L1 / 质心 baseline** | **56.7×** |

结论: 新旧模型同样「远差于质心 baseline」→ **目标退化是数据/特征特性，与 test 分支的注意力改写无关**。

---

## 6. 结论

1. **无 token 泄露**: 掩码数值验证（扰动被屏蔽 token → 输出不变）通过。
2. **「1 向量还原」是目标退化的必然**: DINO patch 特征在工地图上空间变异性仅 ~5e-5，重建任务实际退化为「输出每图常数」。
3. **模型连质心都没学到**: 模型 L1 比「直接输出质心」差 35–57×，说明当前重建指标无法区分模型好坏，需要更换目标/数据或增加 baseline 对照。

---

## 7. 建议（下一步）

1. **数据侧**: 换有真实空间结构的数据集（非大面积均匀的工地图），或先度量 patch 特征空间变异性（>0.01 才有区分度）。
2. **目标侧**: 若坚持 DINO 特征，检查其是否被 LayerNorm / 位置无关化抹平；或改用像素级重建 / SVD 特征（有真实空间结构，见 main 分支 v1 思路）。
3. **指标侧**: 训练/测试增加「质心 baseline」作为下限对照 —— 模型 L1 必须显著优于质心才有意义；渐进曲线（每采样步 L1）作为前缀能力的可视化指标保留。

---

## 8. 复现

```bash
# 训练（服务器, 2 GPU DDP）
NUM_GPUS=2 ./run_v2_train.sh \
    --data_dir /root/autodl-tmp/construction_site \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --output_dir output/phase1_v2_test --epochs 40

# 推理测试（全量 test, fp32）
python infer_v2_test.py \
    --data_dir /root/autodl-tmp/construction_site \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --final_model output/phase1_v2_test/final_model.pt \
    --output output/phase1_v2_test/infer_test.json

# 泄露诊断（掩码验证 + 目标方差分析）
python check_leak.py \
    --data_dir /root/autodl-tmp/construction_site \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --final_model output/phase1_v2_test/final_model.pt
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
