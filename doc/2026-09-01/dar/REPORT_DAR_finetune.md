# SR-Diffusion v3 — D-AR 微调实验（construction_site 工地数据）

> 日期: 2026-09-01 ｜ 服务器: 2× RTX PRO 6000 (97GB) ｜ 数据: construction_site 1000 张
> 本文档记录用工地数据微调外部 D-AR（Diffusion via Autoregressive Models, ICLR 2026, https://github.com/showlab/D-AR）的两阶段实验：顺序扩散 tokenizer 微调 + AR 生成模型微调。

---

## 1. Tokenizer 微调

| 项 | 值 |
|---|---|
| 模型 | 官方 tokenizer_v1.pt（307.4M VQModel, codebook 16384×8, 256-res） |
| 数据 | 1000 张工地图（ImageFolder, 448×252 → random crop 256） |
| 配置 | 2 卡 bf16, bs128, lr 1e-4, EMA, perceptual/dino 0.5 |
| 时长 | 280 步 / ~7 分钟 |
| 结果 | Train Loss 0.3749 → **0.3453**；codebook usage 0.44 → **0.97** |

## 2. D-AR 微调

| 项 | 值 |
|---|---|
| 模型 | GPT-L（343M），初始化自官方 D-AR-L-360K.pt（裸 state_dict, 已兼容） |
| Tokenizer | 微调产物 tokenizer_v1_ft.pt（EMA 权重） |
| 配置 | 2 卡 bf16, bs64, lr 1e-4, EMA, num-classes 1, --no-compile |
| 时长 | 3000 步 / ~12 分钟 |
| 结果 | loss 6.95 → **1.4788**（相对随机 ln(16384)=9.70 大幅下降） |

## 3. 评估

### Tokenizer 重建（1000 张, 256-res, EMA 权重）
| 指标 | 微调前 | 微调后 | Δ |
|---|---|---|---|
| PSNR | 18.022 | **18.170** | +0.148 |
| SSIM | 0.5176 | **0.5259** | +0.0083 |

（两套实现交叉验证一致；可视化: dar/montage_before.png / montage_after.png）

### D-AR 生成采样
- 16 张 256-res 网格 2 份（seed 2 / seed 42）: dar/sample_dar_seed2.png / sample_dar_seed42.png
- 统计: 暖灰调（RGB≈0.58/0.56/0.54），平均亮度 0.56，符合工地场景；**主观质量需人工查看**
- FID 未评估（无参考分布；工地数据无 ImageNet 参考集）

## 4. 结论

1. **两阶段微调全部跑通**（tokenizer 280 步 + D-AR 3000 步），全部 7+1 处环境/代码问题已修复（补丁脚本在 repo 根 patch_*.py）。
2. **Tokenizer 提升有限（+0.15 PSNR）**：280 步轻量适配属预期；要更大收益需全量 10013 张重训或加 epochs。
3. **D-AR 适配显著**：loss 6.95 → 1.48，模型已学会工地 code 分布；类嵌入随机初始化，样本质量待人工确认后可加训/上 GPT-XL。
4. **评估口径**：重建指标可靠（两套一致）；FID 不适用。

## 5. 相关文件

- 完整报告: `dar/FINETUNE_RESULT.md`（含产物清单、7 处问题修复细节）
- 采样图: `dar/sample_dar_seed2.png` / `sample_dar_seed42.png`
- 重建对照: `dar/montage_before.png` / `montage_after.png`
- 服务器产物: /root/autodl-tmp/D-AR/temp/tokenizer_v1_ft.pt（4.97GB）、temp/dar_ft.pt（5.47GB）
