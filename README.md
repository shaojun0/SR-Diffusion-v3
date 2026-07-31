# SR-Qwen-VL v10

**SVD(1024×1024) → (2n,1024) matrix → DINO Encoder → MLP → Qwen3.5-4B → Text**

PreTrainedModel 兼容 — save_pretrained / from_pretrained / HF Trainer.

## 架构

```
1024×1024 原图
    │
    ▼  [离线预处理]
SVD on 32×32 patches → (2n, 1024) matrix
  · n = 能量截断(99%) ∩ [32, 128]
  · 前 n 行: U[:,:n]^T  (行空间本征向量)
  · 后 n 行: V[:,:n]^T  (列空间本征向量)
    │
    ▼  [训练]
SVD Proj: 1024 → 1536 + CLS + pos
    │
DINOv2-giant Encoder 40层 Transformer ❄️
    │
MLP: 1536 → 5120 → 2560
    │
Qwen3.5-4B → 中文描述
```

**无图片输入** — DINO 直接消费 SVD 结构 token，不经过 patch embedding。

## 快速开始

```bash
# 1. 预计算 SVD 矩阵
python preprocess_svd.py

# 2. 训练
python train.py
```

## 推理

```python
from model import SRQwenVLConfig, SRQwenVLv10
model = SRQwenVLv10.from_pretrained("./output/final")
model.cuda()
text = model.generate(svd_matrix, prompt="描述这张图片：")
```
