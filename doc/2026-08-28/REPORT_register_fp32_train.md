# SR-Diffusion v3 — register_specials 全 fp32 平权训练报告（2026-08-28）

> 分支: `main`（test 已并入）｜ 服务器: 2× RTX PRO 6000, torch 2.12.1（新实例, 端口 35056）
> 目标: 按用户要求（平权 / fp32 / 更高 batch）跑 v2 重建训练, 验证 dfb993b `register_specials`
> 修复（specials 直接进 DINO 24 层, 修 DIAGNOSIS_clarity.md F1/F2）对**重建清晰度**的影响。

## 1. 训练配置（用户三项要求已落实）

| 项 | 值 |
|---|---|
| 分支/commit | `main` @ dfb993b（改进模型）+ 4e081e1 + 37e7b04（eval 修复） |
| 架构 | **register_specials=True**: specials 作为额外 token 拼进 DINO 输入序列（1153 token 全双向, 24 层直接算 z_s）, 无 ReEncoder（316.2M 可训练参数） |
| 目标 | 原始像素 patch (B,576,588), **平权 mean_t L1**（去掉加权体系 ✅） |
| 精度 | **全 fp32**（无 bf16/autocast ✅） |
| batch | **16/卡 × 2 卡**（✅ 提高; 峰值显存 53-58GB/卡, 97GB 卡充裕） |
| 训练 | 40 epochs / 8760 步, lr=1.5e-4 cosine, warmup 262, HF Trainer DDP |
| 时长 | ~2h40m（register 模式比 ReEncoder 慢 ~2.5×, 符合文档预期） |

## 2. Eval 历史（归一化空间, 全量 3004 test）

| step | register 版 | 旧版（非 register）参照 |
|---|---|---|
| 2000 | 1.054 | 0.5294 |
| 4000 | 0.5337 | 0.4735 |
| 6000 | 0.4774 | 0.4471 |
| 8000 | 0.4587 | 0.4236 |
| 8760 | **0.4575** | 0.4230 |

register 版起步慢（DINO 24 层直接吃 1153 token, 热身期长）但收敛趋势一致, 最终持平旧版。

## 3. 推理测试（fp32, 全量 3004 test, 0-255 像素空间）

- 全量重建 L1 = **24.14 ± 8.13**（旧版 23.41, 持平）
- 渐进曲线: t=0 L1=85.5（粗糙）→ t≥1 L1=23.6（平台）→ **联想能力成立**（2 键≈全量）
- **t≥1 平台依旧存在**（23.57~23.6 恒定）

## 4. 清晰度对比（同口径 RGB 全通道, 与旧 verify_visual 一致, 24 张 test）

| 指标 | 旧版（非 register） | register 版 |
|---|---|---|
| 全局 std 保留 | 87% | 84% |
| 边缘保留 | 20% | 20% |
| 像素 L1 | 22.9 | 23.9 |

**结论: register_specials（编码侧 F1/F2 修复）没有改变重建清晰度** —— 边缘仍只保留
~20%（低通签名依旧）。验证了 DIAGNOSIS_clarity.md §4 的预判: 瓶颈在**解码侧 F3/F4**
（查询身份只来自固定 query_base 模板、输出=共享键凸组合）, 编码侧修复不解决清晰度。

## 5. 建议（下一步, 按 DIAGNOSIS §7）

1. **改解码器为"逐 patch 内容查询"**: `Q = W_q(A_t) + query_base + Linear(z_s[k])`
   一行改动, 预期 t≥1 平台消失、全量 L1 从 ~23 降到 ≤16（回收 12.9 的大头）。
2. PixelHead 换小 CNN/转置卷积恢复高频; 3. 叠加感知损失（LPIPS, 在修路由之后）。

## 6. 工程备注

- **eval OOM 修复（两处, 已 push main）**: ① eval_accumulation_steps=2（分批移 CPU,
  缓解）; ② **根治**: `SRPhase1V2Trainer.prediction_step` ignore `Y_pix/target_pix`
  （渐进曲线由 infer_v2_test 单独测）——eval 预测从 ~100GB 降到 ~4GB, eval 从 30 分钟
  崩溃 → 90 秒完成。
- 产物: `output/phase1_v2_reg/final_model.pt`（1.26GB fp32）+ infer_test.json + recon_visual.png
