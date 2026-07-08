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

Loss: ε-MSE + 0.5·x₀-MSE — noise prediction + latent KL constraint

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
- GPU: H800 80GB / RTX PRO 6000 96GB

## 文件

| 文件 | 说明 |
|------|------|
| `model_v2.py` | v2 模型 (nn.Module, pretrained 直接加载) |
| `model_v3.py` | v3 模型 (PreTrainedModel, AutoModel 兼容, save/load 一体化) |
| `train_v2.py` | v2 训练脚本 (HF Trainer) |

## model_v3 新特性

继承 `PreTrainedModel`，支持标准 HuggingFace 工作流：

```python
# 新建 + 加载预训练权重
config = SRDiffusionConfig()
model = SRDiffusion(config)
model.build_model(dino_dir="dinov2-giant",
                  sd_model_id="stabilityai/stable-diffusion-2-1")

# 一键保存
model.save_pretrained("./checkpoint")

# 一键加载 — 无需手动拼 DINO/SD/VAE
from transformers import AutoModel
model = AutoModel.from_pretrained("./checkpoint", trust_remote_code=True)
```

### v3 相比 v2 的改进

- **AutoModel 兼容**: 继承 PreTrainedModel + PretrainedConfig，`from_pretrained` 直接加载
- **cond_fusion 模块**: `noisy + cond (8ch) → fusion → 4ch`，替代 `conv_in` 直接修改（更易序列化）
- **NoiseSchedule buffers**: `register_buffer` 确保 save/load 一致
- **Config 自包含**: `SRDiffusionConfig` 嵌套 DINO + U-Net + VAE 的完整 config，`config.json` 单文件包含全部架构信息

## 已知问题

- 初版训练缺少 `loss.backward()`（已修复），首个完整训练 run 正在 H800 上运行
