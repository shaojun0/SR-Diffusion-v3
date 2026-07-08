# SR-Diffusion v2

SVD + DINOv2-giant → SD 2.1 U-Net Diffusion 超分辨率重建

## 架构

```
HR Image (1024²)
  │
  ├─→ SVD → top-k 左奇异向量 → DINOv2 Encoder → SVD tokens (1536d)
  │                                              │
  │                                     TokenProjector
  │                                              ↓
  │                                    cross-attn tokens (1024d)
  │
  ├─→ bicubic↓↑ → VAE Enc → latent_cond
  │
  └─→ VAE Enc → latent → +noise → [noisy;cond] → SD 2.1 U-Net
                                                   ↓ (cross-attn)
                                              noise_pred
```

Loss: MSE(noise_pred, noise) — standard DDPM diffusion loss

## 参数量

| 模块 | Params |
|------|--------|
| DINOv2-giant | 1140M |
| SD 2.1 U-Net | 866M |
| VAE | 84M |
| TokenProjector | ~1M |
| **Total** | **2.09B** |

全参数训练，fp32 + 8-bit AdamW

## 训练配置

- 数据: DIV2K 800张 1024×1024
- BS=1, grad_accum=4
- LR=5e-5, CosineAnnealingWarmRestarts
- 100 epochs, ~20,000 steps
- GPU: H800 80GB, 峰值显存 ~70GB

## 文件

- `model_v2.py` — 模型定义（Config, DINOv2, TokenProjector, DiffusionDecoder, SRDiffusion）
- `train_v2.py` — 训练脚本（Dataset, 训练循环, 评估）

## 已知问题

- 初版训练缺少 `loss.backward()`（已修复），首个完整训练 run 正在 H800 上运行
