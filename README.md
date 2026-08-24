# SR-Qwen-VL v10

**SVD(1024×1024) → (2n,1024) matrix → DINO Encoder → MLP → Qwen3.5-4B → Text**

PreTrainedModel 兼容 — save_pretrained / from_pretrained / HF Trainer.

## Phase 1 v3 — 拉格朗日预算约束（新增，推荐）

目标（约束优化形式）: **在最小化还原损失的同时最小化输入到解码器的 token 数量**

```
min_θ  L_recon(θ)          L_recon = L1(F_hat, F_patch)  特征空间还原损失
s.t.   R(θ) ≤ R_target     R = k/N                       送入 decoder 的 token 比例
```

拉格朗日函数 + 对偶上升（Uzawa）求解:

```
L(θ, λ) = L_recon + λ·(R − R_target) + (ρ/2)·ReLU(R − R_target)²
原步:  θ ← θ − η·∇L(θ, λ)                    （λ 视为常数，优化器做）
对偶步: λ ← clamp(λ + η_λ·(R_ema − R_target), 0, λ_max)   （每步自动调整）
```

λ 不再是手调超参——它自行增长/衰减直到 **R ≈ R_target 被满足**（收敛时
λ* = KKT 乘子 = 再省一个 token 的边际还原代价）。v1/v2 的固定 λ_rate 惩罚
需要网格搜索 λ 才能打到指定预算（太小不压缩、太大会塌缩，v1 为此补了
信任域/死区铰链/力放大一堆机制）；v3 把这些全部替换成一个对偶变量。

```bash
python train_phase1_v3.py --data_dir /path/to/imagenet \
    --rate_target 0.25 --stage1_steps 800 --anneal_steps 2000 \
    --output_dir output/phase1_v3
```

想描 R-D 帕累托前沿（sweep λ 而非约束）：`--use_lagrangian 0 --lambda_fixed 0.1`
想逐位置内容感知选择：`--select_on zs`（默认）；纯全局描述子：`--select_on cls`。

```bash
python smoke_test_phase1_v3.py   # 形状/梯度/对偶动力学/门控语义/KKT 教科书例
python eval_phase1_v3.py <ckpt_dir> --rate_target 0.25
```

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
