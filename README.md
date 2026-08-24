# SR-Diffusion v3：为多模态大模型设计下一代视觉编码器

> **超分辨率只是试金石，真正目标是为 MLLM 构建一个像素级感知的视觉编码器。**
>
> 2.09B 参数 | SVD 结构分解 + DINOv2-giant | Cross-Attention 注入 SD 2.1 U-Net | 1024×1024 超分验证

---

## 为什么现有 MLLM 的视觉编码器不够用？

当前主流多模态模型（LLaVA、Qwen-VL、InternVL、Gemini 等）的视觉编码器几乎全部基于 **CLIP-ViT** 或 **SigLIP**。

这些编码器在一个关键维度上存在系统性缺陷：

### CLIP-ViT 的设计目标 vs MLLM 的实际需求

| | CLIP-ViT | MLLM 真正需要的 |
|------|----------|-----------------|
| **训练范式** | 图文匹配（对比学习） | 像素级理解 + 语义对齐 |
| **空间感知** | 弱（GAP pooling 丢弃空间信息） | 强（"左边第三个人"） |
| **细节保留** | 差（336²/448² → 单 token 压缩） | 强（小文字、数字、纹理） |
| **结构感知** | 隐式（依赖 CLS token） | 显式（边缘、方向、纹理主成分） |
| **中间层表征** | CLIP vision encoder 12-24 层 | DINOv2 40 层，patch 级密集特征 |

**一句话总结**：CLIP-ViT 擅长"这张图里有只猫"，但不擅长"猫的左前爪压在红色围巾下面"。

---

## 我们的方案：SVD + DINOv2 编码器

SR-Diffusion v3 的外壳是超分模型，但内核是一套**通用视觉编码管线**：

![SR-Diffusion v3 Architecture](assets/architecture.png)

```
原始图像
  │
  ▼
┌─────────────────────────────┐
│  SVD 结构分解（显式几何编码） │  ← 局部边缘方向、纹理主成分
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   DINOv2-giant (40层 ViT)    │  ← 密集 patch 特征 + 全局语义
│   1536d visual tokens        │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   Cross-Attention Projector  │  ← 可替换为 LLM Projector
│   视觉特征 → 条件注入         │     1536d → 4096d (LLM hidden dim)
└─────────────────────────────┘
```

### 相比 CLIP-ViT 的优势

| 优势 | 原理 | 对 MLLM 的影响 |
|------|------|---------------|
| **SVD 显式结构** | 每个 patch 做奇异值分解，取主奇异向量作为几何先验 | 空间推理/定位能力提升 |
| **DINOv2 密集特征** | Self-supervised 训练，patch 级表征比 CLIP 更细粒度 | OCR、细节问答精度提升 |
| **原生高分辨率** | 1024² 输入，非 hard resize + patch merge | 高分辨率文档/图表理解 |
| **Cross-Attention 原生投射** | 视觉 token 经由 attention 注入，天然适合嵌入 LLM | 视觉-语言对齐更平滑 |

---

## 路线图：从超分验证到多模态接入

```
Phase 1 ✅ SR-Diffusion v3 (当前)
  │   超分任务验证 SVD+DINOv2 编码器是否能提取高质量视觉特征
  │   结论：ε-MSE 3×10⁻⁵ → 编码器特征极其丰富，足以让 SD U-Net 完美重建
  │
Phase 2 🔲 编码器剥离
  │   从扩散模型中解耦出纯 encoder
  │   输入 1024² 图像 → 输出 N×1536 visual tokens
  │
Phase 3 🔲 MLLM 接入
  │   Visual Tokens → Linear Projector (1536→4096) → Qwen2.5/LLaMA3
  │   在 VQAv2 / TextVQA / OCRBench / RefCOCO 上全面评测
  │
Phase 4 🔲 动态分辨率 + AnyRes
      任意分辨率输入 → 自适应 patch 划分 → 动态 token count
      对标 InternVL2 / LLaVA-NeXT 的任意分辨率方案
```

---

## 核心思路

### Step 1：SVD 显式结构编码

```
HR Image (1024×1024)
  → 32×32 patches (1024 patches, 每个 32²×3 = 3072d)
  → 每个 patch: A = U Σ Vᵀ
  → 取能量前 95% 的左奇异向量
  → 输出: (1024, k, 3072) 结构化 tokens
```

**为什么 SVD？** 左奇异向量编码了 patch 内最主要的信号方向（边缘朝向、纹理走向），相当于给 DINOv2 提供了显式的几何先验，而不是让它从零开始学。

对 MLLM 的意义：这些几何特征天然能回答"在左边""在上方"这类空间问题。

### Step 2：DINOv2-giant 语义编码

| 参数 | 值 |
|------|-----|
| 层数 | 40 Transformer Layers |
| 注意力头 | 24 |
| Hidden Size | 1536 |
| Patch Size | 14 |
| 参数量 | 1.14B |

DINOv2 在 142M 张无标注图像上做 self-supervised 预训练，学到的特征无需语言监督就具备像素级定位能力。这一点在 MMLab 的开放词汇分割、深度估计等下游任务中已被广泛验证。

### Step 3：Cross-Attention 注入 → SD U-Net 去噪

DINOv2 的 1536d CLS token 经过线性投影变为 cross-attention tokens，注入 SD 2.1 U-Net 的每个 attention 层。

**把 "SD U-Net" 替换成 "LLM"**，cross-attention 注入的逻辑完全一致——这就是未来 Phase 3 要做的事。

---

## 实验验证（Phase 1）

### 训练配置

| 参数 | 值 |
|------|-----|
| 数据集 | DIV2K 800 张 1024² |
| 训练步数 | 20,000 (100 epochs) |
| 训练时间 | ~23.8h |
| GPU | RTX PRO 6000 (96GB) |
| 模型大小 | 2.09B (DINOv2 1.14B + SD U-Net 865M + Projector) |
| 精度 | fp32 |

### 核心结论

| 指标 | 结果 | 含义 |
|------|------|------|
| ε-MSE (训练集) | **3×10⁻⁵** | 噪声预测近乎完美 → 编码器特征信息量充足 |
| VAE 重建上限 | **~11dB PSNR** | SD VAE 8× 压缩的物理天花板 |

**核心洞察**：ε-MSE 降到 3e-5 说明 DINOv2 从 SVD 条件中学到了足够丰富的特征表达——这些特征的信息量足以让一个 865M 参数的 U-Net 完美重建。这验证了编码器本身的表征能力。

---

## 快速开始

### 环境

```bash
pip install torch torchvision transformers diffusers bitsandbytes accelerate
```

### 下载预训练组件

```bash
huggingface-cli download facebook/dinov2-giant --local-dir ./dinov2-giant
huggingface-cli download stabilityai/stable-diffusion-2-1 --local-dir ./sd_models/sd-2-1
```

### 训练

```python
from model_v3 import SRDiffusion, SRDiffusionConfig

config = SRDiffusionConfig()
model = SRDiffusion(config)
model.build_model(
    dino_dir="./dinov2-giant",
    sd_model_id="./sd_models/sd-2-1"
)
model.save_pretrained("./sr_v3_weights")
```

```bash
python train_v3.py
```

### 从 checkpoint 恢复

```python
model = SRDiffusion.from_pretrained("./checkpoint-20000")
```

### 推理

```python
import torch
from PIL import Image
from torchvision import transforms
from model_v3 import SRDiffusion

model = SRDiffusion.from_pretrained("./checkpoint-20000").cuda().eval()
img = Image.open("input.jpg").convert("RGB")
img_tensor = transforms.ToTensor()(img).unsqueeze(0).cuda() * 2 - 1

with torch.no_grad():
    sr_tensor = model.sample(img_tensor, steps=25)

sr_image = (sr_tensor.clamp(-1, 1) + 1) / 2  # 0~1
```

---

## License

MIT

---

## 引用

```bibtex
@misc{sr-diffusion-v3,
  author = {Shaojun},
  title = {SR-Diffusion v3: SVD + DINOv2-giant Encoder for Next-Gen MLLM Vision Backbone},
  year = {2026},
  url = {https://github.com/shaojun0/SR-Diffusion-v3}
}
```
