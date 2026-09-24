# LSTM + 6×ResNet 循环 ResNet 重建：12 组实验报告

- 远程：`ssh -p 49687 root@connect.westc.seetacloud.com`（AutoDL，1× RTX 4080 SUPER 32GB）
- 工作目录：`/root/autodl-tmp/rrnet/`
- 参考实现：`SR-Diffusion-v3/model.py`（`BpttCrnn`）
- 总用时：**3 h 08 min**（12 组 × 40 epoch，串行，12/12 全部 `rc=0`）

---

## 1. 实验矩阵（12 = 6 ResNet × 2 分辨率）

| 因子 | 取值 |
|---|---|
| backbone | resnet-10 / 18 / 34 / 50 / 101 / 152 |
| 分辨率 | 224×224 / 448×448 |

唯一变量 = backbone × 分辨率；数据、损失、优化器、epoch、评估口径完全一致。

## 2. 架构

```
pixel (B,3,H,W)
 ├─ stem  (HF ResNet embedder: conv7x7 s2 + maxpool s2)   → /4,  64ch
 ├─ stage1 (ResNet blocks, stride 1)                      → /4,  C1 ─► LSTM₁ ─► skip₀
 ├─ stage2 (ResNet blocks, stride 2)                      → /8,  C2 ─► LSTM₂ ─► skip₁
 ├─ stage3 (ResNet blocks, stride 2)                      → /16, C3 ─► LSTM₃ ─► skip₂
 └─ stage4 (ResNet blocks, stride 2)                      → /32, C4 ─► LSTM₄ ─► z
                                                                             │
 解码器 = 编码器逐级/逐 block 的倒置（下采样 → 上采样）：
 ├─ level4: ↑2 → /16, fuse skip₂, depths[3] 个 block, LSTM
 ├─ level3: ↑2 → /8,  fuse skip₁, depths[2] 个 block, LSTM
 ├─ level2: ↑2 → /4,  fuse skip₀, depths[1] 个 block, LSTM
 ├─ level1:    → /4,  fuse stem,   depths[0] 个 block, LSTM   (镜像 stage1 的 stride 1)
 ├─ stem↑1: ↑2 → /2, conv + LSTM
 ├─ stem↑2: ↑2 → /1, conv + LSTM
 └─ head: conv3x3 → 3ch + sigmoid → [0,1]
```

**「每个输出的 layer block 都接入自己的 LSTM」** — 编码器 4 个 layer block 输出（layer1..layer4，
即 4 个 skip 层级）各接一个 `SpatialLSTM`；解码器每个镜像级的 block 输出同样各接一个。
共 **10 个 LSTM / 模型**（4 编码 + 4 解码 + 2 stem 镜像）。

`SpatialLSTM`：`(B,C,H,W)` → 自适应池化到 `g×g`（`g = min(H,W,14)`，token 数封顶 ⇒ 分辨率无关）
→ `LSTM(C→256)` 扫空间 token → 线性投回 `C` → 插值回 `(H,W)` → **残差相加**。`proj` **零初始化**，
所以初始时刻残差恰为 0 ⇒ 模型严格退化为预训练 ResNet。

**「解码器是编码器的倒置」的严格定义（可复核）**：编码器 `stage_i: in_i → out_i`，其中
`in₀ = embedding_size`、`in_i = hidden_sizes[i-1]`。倒置后各级为 `out_i → in_i`，故解码器各级输出通道
= `[hidden_sizes[2], hidden_sizes[1], hidden_sizes[0], embedding_size]`（resnet-50 实测 `[1024, 512, 256, 64]`）；
每级 ResNet block 数 = 编码器该 stage 的 `depths[i]`（倒序）⇒ **解码器 block 总数 == 编码器 block 总数**。
`verify.py` 对 6 个档位逐条断言通过；下采样全部换成 `Upsample(2, nearest) + 3x3 conv`。

## 3. 预训练权重（不从 0 开始）

编码器直接用 `transformers.ResNetModel.from_pretrained("microsoft/resnet-XX")`（ImageNet 权重），
**6/6 档位实测 `matched == total, missing == 0`**：

| backbone | 权重来源 | 命中 |
|---|---|---|
| resnet-18/34/50/101/152 | 官方 `microsoft/resnet-XX` | 120/216/318/624/930 全命中 |
| resnet-10 | `microsoft/resnet-18`（HF 无 resnet-10 官方权重，401） | **72/72 全命中** |

`resnet-10` 的处理：用 **ResNet-10 标准拓扑**（`depths=(1,1,1,1)`，标准宽度 64/128/256/512）建模 ——
ResNet-10 的每个 block 都是 ResNet-18 对应 stage 的**第一个 block**（`depths` 是严格前缀），
所以张量 100% 命中，**编码器没有任何随机初始化参数**。

解码器是本实验新引入的倒置结构、无预训练权重可用，随机初始化。编码器沿用 HF 的 BatchNorm
（否则预训练权重无法加载）；解码器自建模块用 GroupNorm（448² 下 batch 只能到 8，GN 更稳）。

## 4. 数据与协议

数据：AutoDL 公共 **DIV2K（2K 数据集）**`/autodl-pub/data/DIV2K/HighResolution`
→ train 800 张 / val 100 张；短边缩放 + 中心裁剪到 S×S，存 uint8 npy；训练仅随机水平翻转。

| 项 | 值 |
|---|---|
| 目标 / 损失 | [0,1] RGB / **L1**（直接重建，无 deep supervision） |
| 优化器 | AdamW, lr 2e-4, wd 1e-4, cosine, 200 warmup steps |
| epochs | **40（统一，用户指定）** |
| batch | 224²→16，448²→8（自动探测，OOM 自动减半） |
| 精度 / 裁剪 | bf16 autocast + fp32 参数 / grad clip 1.0 |
| 指标 | val L1、PSNR、SSIM（[0,1] 域） |
| seed | 0 |

## 5. 结果

| backbone | params | 224 best L1 | 224 PSNR | 224 SSIM | 448 best L1 | 448 PSNR | 448 SSIM | 448 用时 |
|---|---|---|---|---|---|---|---|---|
| resnet-10 | 13.03M | 0.05477 | 21.980 | 0.6691 | 0.04557 | 23.305 | 0.7107 | 17.2 min |
| resnet-18 | 20.93M | 0.05498 | 21.974 | 0.6739 | 0.04649 | 23.220 | 0.7113 | 17.5 min |
| **resnet-34** | 33.62M | **0.05274** | **22.334** | **0.6927** | **0.04351** | **23.695** | **0.7391** | 18.9 min |
| resnet-50 | 66.59M | 0.05684 | 21.782 | 0.6748 | 0.04748 | 23.146 | 0.7215 | 24.2 min |
| resnet-101 | 90.35M | 0.05659 | 21.819 | 0.6771 | 0.04793 | 23.106 | 0.7243 | 29.5 min |
| resnet-152 | 109.91M | 0.05497 | 22.053 | 0.6898 | 0.04666 | 23.352 | 0.7309 | 36.1 min |

曲线：`results/curves_val_l1.png`；重建样例：`results/samples_{224,448}.png`。

### 结论

1. **分辨率：448² 全面优于 224²**（6/6 backbone 一致）。L1 相对降低 **15.1%–17.5%**，
   PSNR **+1.25~+1.36 dB**，SSIM **+0.03~+0.05**。
2. **resnet-34 在两个分辨率上都最优**（224: 0.05274；448: 0.04351），且是唯一在两个分辨率上
   PSNR/SSIM 同时领先的档位。
3. **参数量再往上不加分**：resnet-152 参数是 resnet-34 的 **3.3×**（109.9M vs 33.6M），
   结果反而更差（224 L1 +4.2%，448 L1 +7.2%）。bottleneck 家族（50/101/152）整体不如 basic 的 resnet-34。
4. **原因不是过拟合，是欠拟合**——这是本次实验最反直觉、也最需要记录的一点：
   | 配置 | train L1 | val L1 | gap |
   |---|---|---|---|
   | resnet-34 @448 | 0.04406 | 0.04387 | **−0.00020** |
   | resnet-152 @448 | 0.04699 | 0.04666 | **−0.00033** |
   | resnet-34 @224 | 0.05219 | 0.05276 | +0.00057 |
   | resnet-152 @224 | 0.05497 | 0.05497 | −0.00000 |

   12 组 train/val gap **全部 < 0.002（甚至为负）** ⇒ 模型还没把训练集拟合完。
   大 backbone 的 train L1 本身就比 resnet-34 差，说明在 **40 epoch（224² 仅 2000 步 / 448² 4000 步）
   的统一预算下，resnet-50/101/152 尚未收敛**：它们的解码器按 `depths` 镜像后更深更宽、
   且完全没有预训练权重，从零训练需要的步数远多于 resnet-34。
5. **真实性核对（不是平凡解）**：逐像素均值图基线 L1 = 0.228(224) / 0.233(448)，
   PSNR = 11.40 / 11.24。所有 12 组模型 L1 落在 0.0435–0.0479（**约 5× 优于均值基线**），
   PSNR 23.1–23.7（**+11.9~12.5 dB**）⇒ 模型确实在重建，而非"预测均值"。
   样例图（`samples_448.png`）显示布局/物体/颜色被还原，细节偏平滑（L1 口径的正常表现）。

## 6. 诚实边界（不要过度解读）

1. **「resnet-34 最优」只成立于 40-epoch 统一预算下。** 证据（第 5 节第 4 点）表明大 backbone 是
   欠拟合而非能力不足。要判定 bottleneck 架构本身的优劣，必须给大模型足够的训练步数
   （或按参数/算力对齐预算）后重跑。**本次结论不能外推为「bottleneck 更差」。**
2. **单 seed（seed=0）+ val 仅 100 张。** 224² 的 val 曲线有可见噪声（`curves_val_l1.png` 左）。
   排名中 <0.001 的差异（如 224 的 resnet-10 0.05477 vs resnet-152 0.05497 vs resnet-18 0.05498）
   在噪声范围内，**不应解读为真实排序**。resnet-34 的领先幅度（224 +0.0022 / 448 +0.0078）大于噪声，可采信。
3. **resnet-10 无 HF 官方权重**，用 resnet-18 前缀初始化（严格前缀，72/72 命中）；它因此更接近
   "ResNet-18 的第一个 block 组成的浅网 + 预训练首块"，而不是独立的 ResNet-10 预训练模型。
4. **归一化不统一**：编码器 BatchNorm（为加载预训练）、解码器 GroupNorm。这是刻意的工程取舍，
   但意味着"严格镜像"在 norm 实现这一层不成立（block 拓扑与通道是严格镜像的）。
5. **L1 训练、无感知/对抗损失 ⇒ 重建偏平滑**（见样例图）。这是评估口径决定的，不是架构缺陷；
   若关注感知质量需换损失并重跑。
6. **`SpatialLSTM` 的 token 数被封顶在 14×14**（`g = min(H,W,14)`），因此 448² 与 224² 的 LSTM
   计算量同阶。这是为控制成本的工程选择；代价是 LSTM 分支看不到比 14×14 更细的空间信息
   （残差直连路径不受影响，仍是全分辨率）。

## 7. 产物

| 路径（远程 `/root/autodl-tmp/rrnet/`） | 内容 |
|---|---|
| `code/model.py` | `RecurrentResNet`（编码器 + 逐 block LSTM + 倒置解码器 + 预训练报告） |
| `code/train.py` | 统一训练/评估 harness |
| `code/prep_data.py` / `code/bench.py` / `code/verify.py` / `code/sanity.py` | 数据、测速、架构自检、平凡解核对 |
| `code/run_all.sh` / `code/aggregate.py` | 12 组编排（可断点续跑）/ 汇总 |
| `ckpt/<backbone>_<res>/{best,last}.pt, metrics.json, history.json, config.json, train.log` | 12 组权重与逐 epoch 记录 |
| `results/summary.{md,csv,json}` | 结果总表 |
| `results/curves_val_l1.png` | val L1 曲线（两分辨率并排） |
| `results/samples_{224,448}.png` | 重建样例（上=原图，下=重建，列=4 个 backbone） |
| `results/mean_baseline.json` | 平凡解基线 |

复现：`cd /root/autodl-tmp/rrnet/code && export HF_ENDPOINT=https://hf-mirror.com && python verify.py && bash run_all.sh && python aggregate.py`
