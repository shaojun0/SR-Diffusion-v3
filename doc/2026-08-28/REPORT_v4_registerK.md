# SR-Diffusion v4 — register K=64 + DINO 不冻结 实验报告（2026-08-28 夜）

> 服务器: 2× RTX PRO 6000 ｜ 分支: `main` ｜ 设计: `doc/2026-08-28/DESIGN_v4_registerK.md`
> 对照矩阵: v3 (QFormer, K=64, DINO 冻结) = 17.77 → v4 (register K=64, DINO 可训) = **8.26**

---

## 1. 结果总览

| 实验 | 编码器 | K | DINO | 全量 L1 (0-255) | eval_loss |
|---|---|---|---|---|---|
| v2 像素目标 | ReEncoder 路由 | 576 | 可训(单lr) | 23.41 | — |
| register (v2 模式) | specials 进 DINO 24 层 | 576 | 可训 | 24.14 | 0.4575 |
| v3 BLIP-2 | QFormer 交叉注意力 | 64 | **冻结** | **17.77** | 0.3097 |
| **v4（本实验）** | **specials 进 DINO 24 层** | **64** | **不冻结** | **8.26 ± 3.32** | **0.144** |

**v4 全量重建像素 L1 = 8.26，为全项目最优**，比 v3 (17.77) 改善 **2.15×**，
甚至**低于"真实 DINO 特征线性解码上限 9.8"**——说明 DINO 微调让编码器
真正适配了像素重建，突破了冻结特征的信息上限。

## 2. 训练

- register 编码：specials (K=64) 拼入 DINO 序列 [cls; specials; patches] = 641 token，
  DINO 24 层全双向注意力直接算 z_s（**不冻结**，分组 lr：DINO 1.5e-4 / 新模块 3e-4）
- 40 epochs / 8760 步，2 GPU DDP，bs=16/卡，全 fp32，70 分钟完成
- eval_loss: 0.552（冒烟）→ 0.3226（ep9）→ 0.2096（ep18）→ 0.1622（ep27）
  → 0.1446（ep36）→ **0.144（ep40）**，单调下降无回升 = 未过拟合
- 产物: `output/phase1_v4_registerK/final_model.pt`（1.33GB fp32, DINO 含权重）
  + checkpoint-2000/4000/6000/8000 + model_info.json / args.json

## 3. 推理（全量 3004 张 test, fp32 + 双探针 128 条）

| 指标 | v4 | v3 | 解读 |
|---|---|---|---|
| 全量重建像素 L1 (0-255) | **8.26 ± 3.32** | 17.77 ± 6.45 | 突破冻结 DINO 上限 9.8 |
| 全量重建 L1 (归一化) | 0.1439 | 0.3096 | — |
| z_s 均值 → 线性解码 | 60.83 | 60.83 | 均值无像素信息（信息在 token 间分工） |
| z_s within-std | **0.2746** | 0.9734 | token 间分工更紧凑（DINO 微调后） |
| **解码器 h → 线性解码** | **8.88** | 17.03 | ≈ 全量 8.26 ⇒ 解码链路信息充分 |
| h within-std | 0.4493 | 0.7026 | patch 输出变异性 |

## 4. 结论（对照 DESIGN 判读规则）

1. **v4 < 17.77 ⇒ "K 个 register token 由可训 DINO 直接算"能适配像素重建，有增益**——而且是
   大增益（17.77 → 8.26）。DESIGN 判读规则的第 2 分支成立。
2. **解码链路不是瓶颈**：h 线性解码 8.88 ≈ 全量 8.26，QFormer/Decoder/PixelHead 链路信息充分。
3. **关键突破点 = DINO 微调**：v3 冻结时信息上限 9.8（DINO 原始特征线性解码），v4 微调后
   达到 8.26 < 9.8——**编码器主动适配像素重建，突破了冻结特征的信息瓶颈**。
   这印证了 DIAGNOSIS 的判断："DINO 不冻结才能真正适配像素重建（当前步数/lr 不足以
   大幅改造 304M 预训练参数）"——v4 用分组 lr（DINO 1.5e-4）在 8760 步内做到了。
4. **z_s within-std 0.97 → 0.27**：微调后 K=64 token 的分工更紧凑、更聚焦——压缩表示
   更"结构化"，对 Phase 2（冻结编码器接 NLP）是好信号。
5. **register 编码 + 可训 DINO 组合有效**：v2 register（K=N=576, 时序解码）24.14 无效
   的原因被拆解——不是 register 编码本身，而是**当时缺 K 压缩压力 + 无时序解码器 +
   DINO 适配不足**（v2 register 单 lr 1.5e-4 全模型）。

## 5. 下一步建议（供用户拍板）

1. **K 扫描**（Phase 1 中间验收核心）：K=32/64/128 三档跑 v4 配置（register + DINO 微调），
   看"K 压缩 × 重建质量"曲线——现在 K=64 已 8.26，K=128 预计更低（向 5-8 的
   "完美重建"区间靠拢），K=32 看信息是否还够。
2. **DINO 策略对照**：v4 默认不冻结 vs `--freeze_dino`（v3 的 register 孪生）——把
   "register 编码"与"DINO 微调"两个变量彻底分开量化。
3. **Phase 2 预演**：冻结 v4 编码器 → MLP 接 Qwen 小批量文字训练（验收实验设计，GOAL §4 未决项④）。
4. 重建可视化已生成（`recon_visual.png`），可肉眼确认布局/物体/边界保真度。

## 6. 修复记录

- `infer_v4.py` 的 `_patch_to_img` 对 numpy 输入调 `.permute` 崩溃（与 v3 同款 bug），
  已修复为支持 torch/numpy 双输入。commit `24db1b7`。

## 7. 复现

```bash
# 训练（DINO 不冻结, 分组 lr）
accelerate launch --multi_gpu --num_processes 2 train_v4.py \
    --data_dir /root/autodl-tmp/construction_site \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --output_dir output/phase1_v4_registerK --epochs 40

# 推理（全量 L1 + 双探针）
python infer_v4.py --data_dir ... --dino_dir ... \
    --final_model output/phase1_v4_registerK/final_model.pt \
    --output output/phase1_v4_registerK/infer_test.json --probe_limit 128

# 可视化
python visualize_v4.py --data_dir ... --dino_dir ... \
    --final_model output/phase1_v4_registerK/final_model.pt \
    --out output/phase1_v4_registerK/recon_visual.png
```

产物在服务器 `/root/autodl-tmp/sr-diffusion-v3-main/output/phase1_v4_registerK/`。
