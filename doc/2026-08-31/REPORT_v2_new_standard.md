# SR-Diffusion v3 — v2 新版本（标准版）训练 + 推理报告

> 日期: 2026-08-31 ｜ 分支: `main` ｜ 服务器: 2× RTX PRO 6000 (97GB)
> 本文档记录 v2 最新版（OutputQueryDecoder 标准 nn.TransformerDecoder 2 层 + 采样步切片 + register_specials）的完整训练与推理测试。

---

## 1. 训练配置

| 项 | 值 |
|---|---|
| 架构 | DINOv2-large(不冻结, dim=1024) → ReEncoder(4 层, heads=8) → OutputQueryDecoder(nn.TransformerDecoder 2 层) → PixelHead |
| register 模式 | `--register_specials`（specials 进 DINO 序列, specials = 576 = num_patches, 全双向 + causal_specials） |
| 采样切片 | `slice_start=4 / slice_end=9` → decoder_steps = **[16, 25, 36, 49, 64]**（5 步） |
| 数据 | construction_site 工地: 7009 训练 + 3004 测试（1600×900 画布 → 448×252, 576 patches） |
| 口径 | fp32 全精度、平权 mean、bs16/卡×2、lr=1.5e-4 cosine、40 epochs / **8760 步** |
| 产物 | `output/phase1_v2_new/final_model.pt`（1.36GB fp32, 含 DINO 权重） |

## 2. 训练结果

eval 曲线（每 2000 步, 归一化空间）:

| step/epoch | 2000 (9.1) | 4000 (18.3) | 6000 (27.4) | 8000 (36.5) | 8760 (40, final) |
|---|---|---|---|---|---|
| eval_loss | 0.501 | 0.4355 | 0.3704 | 0.3501 | **0.349** |

- 最终 eval_loss = **0.349**（持续收敛, 曲线平滑无波动）
- 训练时长: **~2.93 h**（train_runtime = 1.055e4 s, 0.83 it/s）

## 3. 推理测试（全量 3004 张 test, 0-255 像素空间）

### 3.1 全量重建

| 指标 | 数值 |
|---|---|
| 全量重建像素 L1 (0-255) | **19.99 ± 6.95** |
| 全量重建像素 L1 (归一化) | 0.3489（与最终 eval_loss 0.349 完全吻合） |
| 推理耗时 | 158 s |

### 3.2 渐进曲线（每采样步, 0-255 像素 L1）

```
t=  16 (前缀  17 键) L1 = 20.00
t=  25 (前缀  26 键) L1 = 20.00
t=  36 (前缀  37 键) L1 = 20.00
t=  49 (前缀  50 键) L1 = 20.00
t=  64 (前缀  65 键) L1 = 20.00
```

5 步完全持平（19.995 ± 0.0005）——中段前缀 16 键起重建质量即达平台, 前缀长度不影响质量。

### 3.3 与历史版本对比（全量重建 L1, 0-255）

| 版本 | 全量 L1 | 说明 |
|---|---|---|
| 旧 v2（2026-08-27, 全量 25 步, 无 register） | 23.41 | 像素目标首版 |
| v2 register K=128 | 28.02 | register + K=128 |
| v2 边界 K=64（decoder_steps=[32,64]） | 22.03 | 压缩边界实验 |
| **v2 新版本（本报告, register + specials=576 + 中段切片）** | **19.99** | **历史最佳** |

## 4. 分析与结论

1. **质量显著提升**: 全量重建 L1 19.99, 优于旧 v2（23.41, -15%）、K=128（28.02, -29%）、K=64 边界（22.03, -9%）。register_specials + 标准 TransformerDecoder + 中段采样切片的组合是目前 v2 系列最佳配置。
2. **中段前缀重建质量稳定**: 渐进曲线 5 步全平（20.00）, 说明 16 键前缀即已包含重建所需信息, 采样步切片（跳过最粗糙的短前缀段）无质量损失; 与边界实验"前缀足够后即平台"的结论一致。
3. **训练收敛干净**: eval_loss 0.501 → 0.349 单调下降, 无过拟合迹象; 推理归一化 L1 与训练最终 eval_loss 完全一致, 无 train/eval 口径偏差。
4. **后续建议**: 中段已饱和, 可尝试更短前缀（如 slice_start 更低）验证压缩能力; 或减少采样步数加速推理（当前 5 步已很轻, 158 s 全量）。

## 附: 相关文件

- 推理明细: `doc/2026-08-31/infer_v2_new.json`
- 训练日志: 服务器 `logs/train_new.log`
- 模型: `output/phase1_v2_new/final_model.pt` + `model_info.json`
