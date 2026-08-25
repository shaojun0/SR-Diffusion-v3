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

---

## Phase 1 v2 — 特征压缩自监督训练（DINOv2-large 不冻结）

架构（model_v2.py）: DINOv2-large(可训练) → ReEncoder → FeatureDecoder,
唯一损失 = L1 重建 DINO patch 特征。

数据管线（data_v2.py）: 原图 → **旋转到最优角度 → 等比缩放 → 居中填充
到 1600×900（16:9）画布**。轮廓不变形（只做刚体旋转 + 均匀缩放），
信息量最大化（最优角使画布内内容面积最大）→ 16:9 模型输入 (448×252)。

多卡训练（train_v2.py + run_v2_train.sh）:

```bash
# 双卡（纯重建: L = L1 重建 DINO patch 特征）
NUM_GPUS=2 ./run_v2_train.sh \
    --data_dir /root/autodl-tmp/construction_site \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --output_dir output/phase1_v2

# 双卡（多任务: + 文字自回归, L = L1 + CE, 中文 caption 为目标）
NUM_GPUS=2 ./run_v2_train.sh \
    --data_dir /root/autodl-tmp/construction_site \
    --dino_dir /root/autodl-tmp/models/dinov2-large \
    --qwen_dir /root/autodl-tmp/models/Qwen3.8-27B --text_decoder \
    --output_dir output/phase1_v2_text
```

要点:
- DINOv2-large 权重来自 ModelScope `facebook/dinov2-large`（HF 官方站不可达）。
- bf16 用显式 `torch.autocast`（规避 accelerate prepare 自动包装 +
  torch 2.13 混合设备 conv dtype 报错）；权重保持 fp32。
- 加载后移除 DINO `mask_token`（本任务不传 bool_masked_pos, 该参数不参与
  前向, DDP 会报未用参数）。
- 文字模式: Qwen 词表 embedding 从 safetensors 分片直接读取（不加载全
  模型），默认冻结（--unfreeze_text_embed 解冻）; 优化器只收可训练参数。
- 检查点: `accelerate.save_state` → `output_dir/ckpt-<step>`；续训
  `--resume output_dir/ckpt-<step>`。最终推理权重: `output_dir/final_model.pt`
  （bf16 state_dict, 含 DINO + 可选 TextDecoder）。

