# LSTM + 6×ResNet 循环 ResNet 重建实验（12 组）

参考实现：`SR-Diffusion-v3/model.py`（`BpttCrnn`）。本实验在其思路上做「每层 block 接自己的 LSTM +
解码器为编码器倒置」的架构，跑满 6 档 ResNet × 2 分辨率的 12 组对照。

## 1. 实验矩阵（12 = 6 × 2）

| # | backbone | resolution | 说明 |
|---|---|---|---|
| 1–6 | resnet-10 / 18 / 34 / 50 / 101 / 152 | 224×224 | basic×3 + bottleneck×3 |
| 7–12 | 同上 | 448×448 | 同协议 |

唯一变量 = **backbone 规模 × 分辨率**，其它一切（数据、损失、优化器、epoch、增强、评估）完全相同。

## 2. 架构

```
pixel (B,3,H,W)
  │
  ├─ stem  (HF ResNet embedder: conv7x7 s2 + maxpool s2)          → /4,  64ch
  ├─ stage1 (ResNet blocks, stride 1)                             → /4,  C1 ─► LSTM₁ ─► skip₀
  ├─ stage2 (ResNet blocks, stride 2)                             → /8,  C2 ─► LSTM₂ ─► skip₁
  ├─ stage3 (ResNet blocks, stride 2)                             → /16, C3 ─► LSTM₃ ─► skip₂
  └─ stage4 (ResNet blocks, stride 2)                             → /32, C4 ─► LSTM₄ ─► z
                                                                                    │
  解码器 = 编码器逐级/逐 block 的倒置（下采样 → 上采样）：
  ├─ level4: ↑2 → /16, fuse skip₂, depths[3] 个 block, LSTM
  ├─ level3: ↑2 → /8,  fuse skip₁, depths[2] 个 block, LSTM
  ├─ level2: ↑2 → /4,  fuse skip₀, depths[1] 个 block, LSTM
  ├─ level1:    → /4,  fuse stem,   depths[0] 个 block, LSTM   (镜像 stage1 的 stride 1)
  ├─ stem↑1: ↑2 → /2,  conv + LSTM
  ├─ stem↑2: ↑2 → /1,  conv + LSTM
  └─ head: conv3x3 → 3ch, sigmoid → [0,1]
```

**「resnet 的每一个输出的 layer block 都接入自己的 LSTM」**
编码器 4 个 layer block 输出（layer1..layer4，即 4 个 skip 层级）各接一个 `SpatialLSTM`；
解码器每个镜像级的 block 输出同样各接一个。共 **10 个 LSTM / 模型**（4 编码 + 4 解码 + 2 stem 镜像）。

`SpatialLSTM`：`(B,C,H,W)` → 自适应池化到 `g×g`（`g = min(H,W,14)`，token 数封顶 ⇒ 分辨率无关）
→ `LSTM(C→256)` 扫过空间 token → 线性投回 `C` → 插值回 `(H,W)` → **残差相加**。
`proj` **零初始化**，因此初始时刻残差恰为 0：模型严格退化为预训练 ResNet，
随机初始化的 LSTM 不会破坏 ImageNet 预训练特征。

**「解码器是编码器的倒置」** 的可复核定义：编码器 `stage_i: in_i → out_i`，其中
`in₀ = embedding_size`、`in_i = hidden_sizes[i-1]`；倒置后各级为 `out_i → in_i`，
故解码器各级输出通道 = `[hidden_sizes[2], hidden_sizes[1], hidden_sizes[0], embedding_size]`
（resnet-50 实测 `[1024, 512, 256, 64]`）。每级 ResNet block 数 = 编码器该 stage 的 `depths[i]`
（倒序）⇒ 解码器 block 总数 == 编码器 block 总数。`verify.py` 对 6 个档位逐条断言。

## 3. 预训练权重（不从 0 开始）

编码器直接使用 `transformers.ResNetModel.from_pretrained("microsoft/resnet-XX")`（ImageNet 权重）。

- `resnet-18/34/50/101/152` → 官方 HF 权重。
- `resnet-10` → **HF 上没有官方权重**（`microsoft/resnet-10` 返回 401）。因此用
  **ResNet-10 标准拓扑**（`depths=(1,1,1,1)`，标准宽度 64/128/256/512）建模，并从
  `microsoft/resnet-18` 加载：ResNet-10 的每个 block 都是 ResNet-18 对应 stage 的
  **第一个 block**（`depths` 是严格前缀），因此 **72/72 张量 100% 命中**，不存在随机初始化的编码器参数。
  ⚠️ 诚实标注：这只对**所有 stage 的第一个 block** 成立；本拓扑下 ResNet-10 每 stage 只有 1 个 block，
  所以覆盖率确实是 100%。
- 6 个档位实测 `matched == total，missing == 0`（见每次训练日志的 `load_report` 行）。

解码器无预训练权重可用（它是本实验新引入的倒置结构），随机初始化 + 零初始化残差门。

归一化：编码器沿用 HF 的 BatchNorm（否则预训练权重无法加载）；解码器为自建模块，用
**GroupNorm**（448² 下 batch 只能到 8，GN 对小 batch 更稳，且解码器本身无预训练参数可迁移）。
block 拓扑严格镜像，仅 norm 实现不同。

## 4. 数据

AutoDL 公共数据集 **DIV2K（2K 数据集）** `/autodl-pub/data/DIV2K/HighResolution`：

- `DIV2K_train_HR` 800 张 → train
- `DIV2K_valid_HR` 100 张 → val（只用于评估，不参与训练）

短边缩放到 S → 中心裁剪 S×S（保持长宽比），存为 uint8 npy：`div2k_{train,val}_{224,448}.npy`。
训练增强仅随机水平翻转。

## 5. 统一训练协议（12 组完全一致）

| 项 | 值 |
|---|---|
| 目标 | [0,1] RGB |
| 损失 | L1（直接重建，无 deep supervision） |
| 优化器 | AdamW, lr 2e-4, wd 1e-4, cosine, 200 warmup |
| epochs | **40**（统一） |
| batch | 224²→16, 448²→8（显存自动探测，OOM 自动减半） |
| 精度 | bf16 autocast，fp32 参数 |
| 梯度裁剪 | 1.0 |
| 指标 | val L1、PSNR、SSIM（[0,1] 域） |
| seed | 0 |

## 6. 代码

| 文件 | 作用 |
|---|---|
| `model.py` | `RecurrentResNet`：编码器 + 逐 block LSTM + 倒置解码器 + 预训练加载报告 |
| `prep_data.py` | DIV2K → 定分辨率 npy |
| `train.py` | 统一训练/评估 harness |
| `bench.py` | 12 配置测速 + 显存（排期用） |
| `verify.py` | 断言「倒置镜像 + 预训练 100% 命中 + 初始残差为 0」 |
| `run_all.sh` | 12 组串行编排，可断点续跑（已有 `metrics.json` 则跳过） |
| `aggregate.py` | 汇总 `summary.{json,csv,md}` + `curves_val_l1.png` |

复现：

```bash
cd /root/autodl-tmp/rrnet/code
export HF_ENDPOINT=https://hf-mirror.com
python prep_data.py                 # 数据（已跑）
python verify.py                    # 架构自检
python bench.py                     # 可选：测速
bash run_all.sh                     # 12 组实验
python aggregate.py                 # 汇总
```

## 7. 结果

见 **`REPORT.md`**（完整结果、结论、诚实边界）与 `results/`：

- `results/summary.md` / `summary.csv` / `summary.json` — 12 组总表
- `results/curves_val_l1.png` — val L1 曲线（224² / 448² 并排）
- `results/samples_{224,448}.png` — 重建样例（上=原图，下=重建）
- `results/mean_baseline.json` — 平凡解（预测均值）基线

一句话结论：**resnet-34 在两个分辨率上都最优**；448² 全面优于 224²（L1 −15~18%，PSNR +1.25~1.36 dB）；
参数量超过 resnet-34 后不加分，且 12 组 train/val gap 全部 <0.002 ⇒ 大 backbone 是**欠拟合**
（40 epoch 统一预算不够），不是过拟合。

