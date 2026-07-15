# SVD + ViT 的自适应图像 Token 压缩：让简单图片少占显存，复杂图片多存细节

> 多模态大模型正在吞噬世界，但每张图的 token 数都是固定的。一张白墙和一张工地全景，凭什么吃同样的算力？本文提出一种基于 SVD 分解的自适应 token 压缩方案。

---

## 1. 固定 Token 数的原罪

当前主流多模态 LLM（LLaVA、Qwen-VL、InternVL）处理图像的流程：

```
Image → ViT Encoder → N 个 patch tokens → Projector → LLM
```

不管输入是一张**纯白壁纸**还是一张**春节火车站**，ViT 都输出固定数量的 token（比如 576 或 1024 个）。问题来了：

- 白墙的信息熵 ≈ 1 个 token
- 火车站的语义密度 → 可能需要上百个 token

**固定 K 压缩本质上是在浪费计算**：简单图片冗余 token 拉低吞吐，复杂图片 token 不够丢失细节。

能不能让 K 随图片内容**自适应变化**？

---

## 2. 思路：用 SVD 的能量分布判断图片复杂度

### 2.1 直觉

对一张 1024×1024 的灰度图做 SVD：

$$A = U \Sigma V^T$$

$\Sigma$ 的对角线是奇异值 $\sigma_1 \geq \sigma_2 \geq \dots \geq \sigma_{1024}$。奇异值的衰减速度反映了图像的**结构复杂度**：

| 图片类型 | 奇异值分布 | 95% 能量所需 K |
|---------|-----------|--------------|
| 纯色/渐变 | $\sigma_1$ 极大，其余接近 0 | K ≈ 8-16 |
| 自然风景 | 中等衰减 | K ≈ 30-50 |
| 复杂纹理/城市 | 缓慢衰减 | K ≈ 60-80 |

**奇异值的累计能量曲线直接告诉你："这张图需要多少个自由度来描述"。**

### 2.2 动态 K 的选择

```python
s2 = singular_values ** 2          # 能量 = 奇异值平方
cumsum = s2.cumsum() / s2.sum()    # 累计能量占比
K = argmin(cumsum > threshold)      # 取刚好超过阈值的位置
```

threshold=0.95 时：白墙 ≈ 12 个, DIV2K 训练图 ≈ 45-64 个, 工地全景 ≈ 70-90 个。

---

## 3. 架构设计

我们不只拿 SVD 算 K，而是把 SVD 的**左奇异向量（特征向量）**直接喂给 ViT 做 token。

### 3.1 Pipeline

```
Image (1024×1024)
  │
  ├──→ resize 448² → DINOv2 patch_embed → 1024 patch tokens (1536d)
  │    (整图特征，提供全局上下文)
  │
  └──→ grayscale → SVD → n 个左奇异向量 → Linear Proj → eig tokens (1536d)
       (结构特征，数量动态)
       
[CLS, eig_1, ..., eig_n, patch_1, ..., patch_1024]
  │
  ▼
DINOv2-Giant Encoder (40层 Self-Attention)
  │   eig tokens 通过 attention 从 patches 中提取信息
  ▼
取前 n 个 eig token 位置的输出 (丢弃 patch outputs)
  │
  ▼
Projector → cross-attention tokens → Diffusion U-Net / LLM
```

### 3.2 为什么取 eig token 位置？


DINOv2 Encoder 的 self-attention 让每个 token 都能 attend 到所有 patch。**eig token 天然是信息瓶颈**——它们数量少（n ≪ 1024），但通过 40 层 attention 从 1024 个 patch 中聚合了最关键的视觉信息。

整个机制类似于：
- **patches = 全书内容**（1024 个 token）
- **eig tokens = 章节摘要**（n 个 token，每个对应一个结构主成分）
- **attention = 写摘要的过程**（eig token "读取"所有 patch 后生成压缩表示）
- **最终输出 = 只保留 n 个摘要**（丢弃原书）

### 3.3 和现有方案对比

| 方案 | K 是否固定 | 图片感知 | 压缩方式 |
|------|-----------|---------|---------|
| LLaVA (固定 patch) | ✅ 固定 | ❌ | 无压缩 |
| Q-Former | ✅ 固定 | ❌ | Learnable queries |
| Perceiver Resampler | ✅ 固定 | ❌ | Learnable latents |
| Token Merging (ToMe) | ✅ 固定 | ⚠️ 后处理 | 合并相似 patch |
| **SVD Bottleneck (本文)** | ❌ **自适应** | ✅ 结构感知 | SVD 主成分 + attention |

---

## 4. 为什么用 SVD 而不是别的？

### 4.1 对比：DINOv2 patch norm

之前想过直接用 DINOv2 patch 的 L2 norm 来选 top-K：

```python
importance = patch_features.norm(dim=-1)   # 计算每个 patch 的重要性
K_patches = topk(importance, K)            # 取 top-K
```

问题：**DINOv2 的 patch norm 反映的是"语义显著度"，不是"结构复杂度"**。白墙上的一个黑点会有高 norm，但结构上确实简单。

SVD 的奇异值直接描述了**矩阵的秩/自由度**，和结构复杂度有数学上的对应关系。

### 4.2 对比：随机选 K

随机选 K 个 patch → 白墙和工地各随机选 32 个 → 白墙 32 个大概率全是冗余，工地 32 个大概率不够。

**SVD 的 top-K 特征向量不是随机选的——它们是对矩阵重建贡献最大的方向。**

---

## 5. 训练与验证

当前实现基于 SR-Diffusion v4 框架，训练目标：

$$\mathcal{L} = \underbrace{\|\epsilon_\theta(z_t, t, c) - \epsilon\|^2}_{\text{noise prediction}} + \lambda \cdot \underbrace{\|x_0^{\text{pred}} - x_0\|^2}_{\text{latent constraint}}$$

其中 $c$ 就是 n 个 eig token 经过 projector 后的 cross-attention embedding。

训练后通过**消融实验**验证自适应 K 的效果：
- 固定 K=16 / K=32 / K=64 vs 自适应 K (threshold=0.95)
- 指标：PSNR、SSIM、LPIPS、FID

---

## 6. 展望：从超分到多模态

当前用在 SR 任务上验证，但核心模块 `DinoEncoderV4` 可以直接替换任何多模态 LLM 中的 ViT：

```python
class MLLMVisionTower(nn.Module):
    def __init__(self):
        self.encoder = DinoEncoderV4()      # 自适应 token 压缩
        self.projector = MLP(1536 → LLM_dim) # 投影到 LLM 空间
    
    def forward(self, img):
        eig_tokens, n = svd_eigenvectors(img)
        features = self.encoder(img, eig_tokens)  # (B, n_max, 1536)
        mask = torch.arange(n_max) < n[:, None]   # 动态 mask
        
        # LLM 侧: 对 masked token 做 attention
        return self.projector(features), mask
```

LLM 通过 attention mask 处理变长输入，简单图 token 少 → 推理快 → 省算力，复杂图 token 多 → 细节全 → 效果好。

---

## 7. 代码

完整实现已开源: [github.com/shaojun0/SR-Diffusion-v3](https://github.com/shaojun0/SR-Diffusion-v3)

核心模块 `model_v4.py`：

```python
# SVD 特征向量提取
def svd_eigenvectors(img, energy_threshold=0.95, max_n=64, min_n=16):
    gray = rgb2gray(img)
    u, s, _ = torch.linalg.svd(gray)
    s2 = s ** 2
    n = searchsorted(s2.cumsum(0), threshold * s2.sum()) + 1
    n = clamp(n, min_n, max_n)
    return u[:, :n].T, n

# DINOv2 + SVD bottleneck
class DinoEncoderV4(nn.Module):
    def forward(self, img, eig_tokens):
        patches = self.patch_embed(resize(img, 448))     # 1024 tokens
        eig_proj = self.eig_proj(eig_tokens)              # n tokens
        tokens = [cls, eig_proj, patches]                 # 1+n+1024
        output = self.transformer(tokens)                 # 40-layer self-attn
        return output[:, 1:1+n]                           # 只取 eig 输出
```

---

## 8. 讨论 & 局限

**优点**:
- ✅ 动态 token 数，理论上有更好的效率-效果 tradeoff
- ✅ SVD 提供数学保证的"最优低秩近似"
- ✅ 模块化设计，可插入任意 ViT + LLM 架构

**局限**:
- ⚠️ 1024×1024 SVD 计算开销 (~50ms on GPU)，需要优化或近似
- ⚠️ SVD 在灰度图上做，忽略了色度信息
- ⚠️ 能量阈值 threshold 是超参，需要根据下游任务调
- ⚠️ 目前只在 SR 任务上验证，多模态 LLM 端到端效果待测

**TODO**:
- [ ] 随机 SVD 近似加速（Nyström / randomized SVD）
- [ ] 彩色 SVD（分通道或 block-wise）
- [ ] 在 LLaVA / Qwen-VL 上做端到端替换实验
- [ ] 对比固定 K vs 自适应 K 的推理吞吐

---

*如果你也在做多模态 token 压缩，或者对 SVD + Attention 的结合有其他想法，欢迎讨论！*

*项目地址: [github.com/shaojun0/SR-Diffusion-v3](https://github.com/shaojun0/SR-Diffusion-v3)*
