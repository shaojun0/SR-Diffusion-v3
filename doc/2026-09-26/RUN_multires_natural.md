# RUN — MultiResSR 自然图像微调（ImageNet/COCO/DIV2K/Flickr2K/OpenImages/Vimeo，2026-09-26）

> 实例：AutoDL `ssh -p 57940 root@connect.westc.seetacloud.com`（`autodl-container-dff34b9125`）
> 硬件：**2× RTX 4080 SUPER 32G**（单卡可见 31.48 GiB）｜`/root/autodl-tmp` = 550 G 数据盘
> 软件：torch **2.12.1+cu130**、transformers **5.17.0**、accelerate 1.15.0、pyarrow 25.0.1
> 代码：`/root/autodl-tmp/srdiff-multi/`（本地 `SR-Diffusion-v3` 同步，新增 3 个文件）

---

## 0. 一句话

**显存够，已经在跑**：4 个解码器 + K=144/12 步 + 编码器原生尺寸（长边 1280 分桶）在
bf16 + 编码器/解码器双梯度检查点下 **峰值 23.5 GiB / 31.48 GiB**，双卡 ~1.25 s/step，
一个 epoch（265,247 张）≈ **11.8 h**。

---

## 1. 本轮的四个关键决定（用户 2026-09-26 拍板）

| 项 | 决定 |
|---|---|
| 编码器输入 | **原生尺寸 + 长边上限 L=1280 + 按 patch 面积分桶** `{576,1024,2176,4096}`（同形状成批） |
| 「最适配分辨率」 | 长宽比最接近（`min|Δlog aspect|`），平手（≤1e-3）时比像素数 |
| 数据量 | 全量 ≈27 万张（6 个源） |
| 初始化 | **DINOv2-large 预训练权重**起步 + specials/4 解码器全新初始化（老实例已回收，历史 ckpt 不可得） |

## 2. 模型（`model_multi.py`）

- 编码器沿用 register 式：`[cls; specials(K=144); patches(N)]` → DINOv2-large 24 层全双向
  → layernorm → `z_cls=seq[:,:1]`, `z_s=seq[:,1:145]`。**N 可变**（不再固定 576）。
- **解码总 token 固定 144**：`square_block_starts(144) = [1,4,9,…,144]` ⇒ **12 步（0~11）**，
  第 12 步读 `z_s[121:144]`，累计读满 144。`step_plan=fixed --block 12` 亦支持，但默认走 square
  （历史 `fixed` 计划在 448 上被证明退化，见 `doc/2026-09-23/REPORT_fixw4_plan.md`）。
- **4 个解码器**（各自独立 `OutputQueryDecoder`，共享同一个 per-patch `PixelHead`）：

| 解码器 | 输出 | patch 数 N_i | 损失权重 |
|---|---|---:|---:|
| 0 | 448×252 | 576 | 2/5（当原图最适配它时）或 1/5 |
| 1 | 252×448 | 576 | 同上 |
| 2 | 224×224 | 256 | 同上 |
| 3 | 448×448 | 1024 | 同上 |

- 损失：`L = Σ_i w_i · mean_t mean_patch L1(PixelHead(Y_i,t), target_i)`，
  `w_best = 0.4`，其余各 `0.2`（**权重和 = 1**，模型内有断言）。
  每步**直接预测**、步间 BPTT（沿用主线口径，无 detach）。
- 参数量：**445.3 M**（DINOv2-large 304 M + 4 解码器 ~141 M）。

## 3. 数据（`data_multi.py`）

| 源 | 盘上位置 | 数量 | 状态 |
|---|---|---:|---|
| ImageNet ILSVRC2012 **val** | `data/imagenet_val` | 50,000 | ✅ |
| COCO2017 train | `data/coco/train2017` | 118,000（cap） | ✅ |
| Vimeo-90k（每 clip 取 im1） | `data/vimeo` | 91,701 | ✅ |
| DIV2K train HR | `data/div2k/DIV2K_train_HR` | 800 | ✅ |
| Flickr2K | `data/flickr2k` | **2,650** | ✅ 全部到齐 |
| OpenImages V7 子集 | `data/openimages` | **15,000**（实下 18,984，cap 到 15k） | ✅ |
| **v2 索引合计** | `data/index_cache_v2.json` | **278,151**（315 个桶，40,243 步/卡/epoch） | ✅ |

> 公共数据盘 `/root/autodl-pub/` 里本来就有 ImageNet/COCO2017/DIV2K/Vimeo-90k（只读挂载），
> Flickr2K 与 OpenImages 不在，需下载。
> ⚠️ 两个坑：① `hf_hub_download` 对本镜像的大文件**会卡死**（0 B + `.lock`），
> 一律改成 `curl -C -` 直连；② hf-mirror 对大文件限速到 ~7 KB/s，
> 走 `source /etc/network_turbo` + `huggingface.co` 直连有 ~5–8 MB/s。

## 4. 显存实测（关键结论）

**单样本成本（编码器，序列 = 1+144+N，DINOv2-large，单卡）**

| 配置 | N=576 | N=1024 | N=2176 | N=4096 | N=8704 | N=14500 |
|---|---|---|---|---|---|---|
| fp32 + ckpt | 2.32 | 2.52 | 2.74 | 3.10 | 3.98 | 5.08 GiB |
| bf16 + ckpt | 1.18 | 1.30 | 1.41 | 1.58 | 2.02 | 2.56 GiB |
| fp32 无 ckpt | 1.87 | 2.28 | 3.39 | 5.24 | 9.69 | 15.33 GiB |
| bf16 无 ckpt | 1.21 | 2.17 | 3.15 | 4.82 | 8.89 | 13.56 GiB |

**4 解码器 + K=144 + 12 步 BPTT（单卡 bs=1）**：fp32 **5.81 GiB** / bf16 **3.01 GiB**
（这是显存主项——与编码器 N 基本无关，因为它由 448×448 那一档主导）。

**端到端（2 卡 DDP, bf16, 双梯度检查点）**

| 配置 | 峰值/卡 | s/step | 结论 |
|---|---|---|---|
| 最大桶 N=4144, bs=2 | 19.15 GiB | 1.7–2.9 | ✅ |
| 常见桶 N=1014, bs=8 | **28.79 GiB** | 2.6–4.6 | ⚠️ 太顶 |
| 混合, budget 8192, bs 上限 8 | 28.87 GiB | 1.7–2.3 | ⚠️ |
| **混合, budget 6144, bs 上限 6（采用）** | **23.52 GiB** | **1.22–1.40** | ✅ 双卡 97–100% util |

⇒ **结论：够用，但必须 bf16 + 双梯度检查点 + 按 patch 预算收缩 batch**。
不用解码器检查点时，最大桶 bs=4 直接 OOM（参数+优化器 ≈7 GiB，解码器激活 ≈2.4 GiB/样本）。

## 5. 训练启动

```bash
bash /root/autodl-tmp/run_multires.sh     # torchrun --nproc_per_node=2
# bs 上限 6 / patch_budget 6144 / grad_ckpt_decoder / bf16
# lr: DINO 1.5e-4, 新模块 3e-4 / wd 0.01 / warmup 3% / clip 1.0
# epochs 1 / max_steps 33860（= 265247 样本一个 epoch 的每卡步数）
# log_every 20 / save_every 2000 → /root/autodl-tmp/output/multires_natural
```

**已观测**（前 140 步）：`loss 1.880 → 0.928`，4 个解码器分项同步下降，`grad_norm` 2–11，
峰值 23.52 GiB，1.22–1.40 s/step ⇒ **ETA ≈ 11.8 h/epoch**。
索引 311 个桶，最大桶 top1 = (252,448) 92,041 张。

**第 1 轮跟踪（step 860）**：`loss 1.850 → 0.846`（best 0.804），
4 个解码器分项 `[0.845, 0.838, 0.828, 0.843]`（**无解码器塌缩，四档同步下降**），
`grad_norm` 11 → 0.7，实测 **1.18 s/step**（75 s 推进 80 步复核过），
峰值稳定 23.52 GiB，双卡 100% util ⇒ **ETA ≈ 11 h/epoch**。
曲线：`doc/2026-09-26/curve_step860.png`（`plot_multires.py` 生成）。

> 注意：单步 loss 抖动大是**预期的**——每个 batch 是不同桶（不同分辨率/不同 patch 数）
> 的不同内容，不是训练不稳。看趋势要看滑动平均。

## 5.1 六个数据源全部并入（第二轮，已启动）

第一轮（26.5 万索引）在 step 3820 时被主动停掉——因为六源索引已就绪，
没有必要再等 10 h 才让 Flickr2K/OpenImages 进场。**当前在跑的就是六源全量**：

```bash
bash /root/autodl-tmp/run_multires_v2.sh
# 索引 index_cache_v2.json（278,151 / 40,243 步每卡）
# --resume output/multires_natural/ckpt_2000.pt --reset_step_on_resume
# output → /root/autodl-tmp/output/multires_natural_v2
```

已实测的启动日志（证据）：

```
[v2] steps_per_rank=40243 resume=.../ckpt_2000.pt
[data] 样本 278151 | 桶 315 种
[resume] ckpt_step=2000 → start_step=0 (reset=True)
[140/40243] loss=0.80315 dec=[0.886, 0.856, 0.859, 0.869] peak=23.52GiB 1.175s/step
```

**续训路径已单独验证**（`verify_ckpt.py`，CPU）：`torch.load` 默认口径可读；
state_dict **597 键全匹配、无 missing/extra/shape mismatch、strict load 通过**；
优化器 `param_groups=['new','dino']`、596 个 Adam 状态项、`decoder.steps=[1..144]` ✓。

**最终状态（2026-09-26 17:50，六源 run）**：step 580，`loss ≈ 0.78`，
`peak 23.52 GiB / 31.48`，**1.15 s/step**，双卡 100% util，日志仅 2 条
DDP grad-stride UserWarning（"This is not an error"）。
⇒ **ETA ≈ 40,243 × 1.15 s ≈ 12.9 h / epoch**。
曲线：`doc/2026-09-26/curve_v2_six_sources.png`（六源）与 `curve_step860.png`（四源预热轮）。

两个必须记住的坑（已修）：
1. `--resume` 若不复位优化步计数，`step` 直接 ≥ `max_steps` **会立刻退出**，
   且 `lr_scale(step)`≈0 ⇒ 新增 `--reset_step_on_resume`（开新一轮 epoch 必带）。
2. `epoch_steps.py` 输出的第 **4** 列才是 steps_per_rank（第 5 列是 global_batches）。

Flickr2K 的教训：**大 zip 走 hf-mirror 会被限速到 ~0.08 MB/s**（11.6 GB 要 19 h），
改用 `pgtkhai/DF2K` 里的**个体 PNG**（6 位文件名 = Flickr2K，4 位 = DIV2K）
+ 8 线程并行，实测 ~10 MB/s 聚合、约 1.5 h 拉完 2,650 张。
OpenImages 同理：`hf_hub_download` 卡死（0 B + `.lock`）⇒ 换 `curl` + 逐 shard 解图。

## 5.2 六源 run 已跑完（2026-09-27 06:12）

`--epochs 1 --max_steps 40243`，**40,242 步 / 12h33m / 1.122 s/step / 峰值 23.52 GiB**，
0 条 traceback / OOM / NaN（只有 2 条 DDP grad-stride UserWarning）。
产物：`final_model.pt` + **20 个 ckpt**（2000…40000），
日志 `output/multires_natural_v2/train_log.jsonl`（4,024 条），曲线 `curve_v2_final_40k.png`。

| 量 | first100 | last100 | best |
|---|---:|---:|---:|
| loss（20 步滑动均值） | 0.7997 | **0.4387** | **0.3683** |
| 4 档分项（last100） | — | 0.4532 / 0.4518 / 0.4473 / 0.4587 | — |
| recon（末步） | — | 0.4355 | — |

换算到 0-255（÷57.6，与历史 448×252 口径一致）：last100 loss ≈ **25.3 px**、
四档分项 **25.8–26.4 px**。参照：常量预测平台 ≈65.6 px、历史主线 square24 BPTT **16.42 px**、
v4 register K=64 **8.26 px**。

**能说的**：四个解码器**全程同步下降、无塌缩**（四档最后相差 <0.012）；
loss 从 0.80 单调趋势降到 0.44，**远离常量吸引子**（1.138）。
**不能说的**：这是**训练集**上、跨桶混合的 20 步滑动均值，**不是留出集评测**；
与 16.42/8.26 px 的历史数字**不同域、不同 K、不同分辨率集合，不可直接比**。
要给出可比的结论，必须补：固定留出集 + 逐分辨率 PSNR/L1 + 逐 step 早停曲线。

## 6. 逐 step 精炼：**有，但极度前置**（COCO2017 val 留出集，n=5000）

评测：`eval_stepwise.py`（训练只用了 COCO **train**2017，val 5,000 张干净留出；
编码器同训练分桶，encode 一次 → 4 个解码器各跑满 12 步 → 每步直接预测反归一化到
0-255 算 L1/PSNR）。曲线：`eval_stepwise_cocoval.png`，原始 json 在服务器
`output/eval_stepwise_cocoval.json`。

| step | t | 累计 z_s | 448×252 | 252×448 | 224×224 | 448×448 | **best-fit** | PSNR_bf |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 3 | 31.00 | 30.88 | 30.11 | 31.49 | **31.25** | 14.91 |
| 2 | 4 | 8 | 28.94 | 28.87 | 28.27 | 29.44 | **29.19** | 15.30 |
| 3 | 9 | 15 | 27.95 | 27.91 | 27.36 | 28.52 | **28.24** | 15.45 |
| 4 | 16 | 24 | 27.29 | 27.25 | 26.74 | 27.82 | **27.56** | 15.58 |
| 6 | 36 | 48 | 26.29 | 26.20 | 25.81 | 26.77 | **26.52** | 15.80 |
| 8 | 64 | 80 | 25.96 | 25.85 | 25.47 | 26.43 | **26.18** | 15.86 |
| 10 | 100 | 120 | 25.88 | 25.76 | 25.39 | 26.34 | **26.10** | 15.87 |
| 11 | 121 | **143** | 25.88 | 25.75 | 25.39 | 26.34 | **26.10** | 15.87 |
| 12 | 144 | 144 | 25.90 | 25.77 | 25.40 | 26.35 | **26.11** | 15.87 |

（均色常数基线 ≈ **52.4 px**）

**四条结论**

1. **精炼是真的、单调的**：best-fit 31.25 → 26.10（−5.15 px），
   12 步逐一递减，是标准**嵌套码**（前缀可解码），4 个分辨率**全部同步单调**。
2. **极度前置**：逐步 ΔL1 = `−2.05, −0.95, −0.68, −0.60, −0.44, −0.23, −0.12, −0.06, −0.03, −0.00, +0.01`；
   降幅占比 **前 2 步 39.9% / 前 3 步 58.4% / 前 4 步 71.6% / 前 6 步 91.7%**。
   第 7→11 步（63→143 token，**多花 80 个 token = +127%**）只买到 **0.20 px**。
3. **末步轻微过冲**：step 12 反而 **+0.01 px**（与历史 `ANALYSIS_ksweep_stepwise_refinement.md`
   "末 2–3 步转负（过冲）"一致）。本配置下**有效预算 ≈143 token**，第 144 个 token 白花。
4. **早停没有收益**：逐图最优停步集中在 9–12（871/1069/1112/1141 张），
   oracle（逐图选最优步）26.059 vs 固定末步 26.110 ⇒ **天花板只有 0.050 px**。
   与 09-22 批 0 的结论一致（"内容自适应早停结构上没赚头，天花板 +0.13 dB"）。

**分辨率差异**：224×224 最好（25.39，target 本身最平滑）、448×448 最差（26.34，1024 patch 最难），
四档差距 <1 px，说明**加权损失把 4 个解码器训得很均衡**（没有哪一档被牺牲）。

**两个必须写清的边界**
- COCO val 与训练集**同域**（train vs val），所以这里的 26.1 px 不能外推到工地/隐患域；
  与历史 `construction_site` test 的 16.42 px（square24 BPTT）**不同域、不同 K、不同分辨率集合，不可直接比**。
- 训练末段 loss 换算 ≈26 px、留出集 26.1 px ⇒ **训练/留出几乎重合，没有过拟合迹象**（但同域，说服力有限）。

### 5.3 读窗口核对（`verify_windows.py`，K=144 / square / 12 步的**实跑**输出）

| step | t | A 窗口 | 读 A | z_s 新增 | **累计 z_s** | z_s 区间 |
|---:|---:|---|---:|---:|---:|---|
| 1 | 1 | [0,3] | 4 | 3 | **3** | 0–2 |
| 2 | 4 | [4,8] | 5 | 5 | **8** | 3–7 |
| 3 | 9 | [9,15] | 7 | 7 | **15** | 8–14 |
| 4 | 16 | [16,24] | 9 | 9 | **24** | 15–23 |
| 5 | 25 | [25,35] | 11 | 11 | **35** | 24–34 |
| 6 | 36 | [36,48] | 13 | 13 | **48** | 35–47 |
| 7 | 49 | [49,63] | 15 | 15 | **63** | 48–62 |
| 8 | 64 | [64,80] | 17 | 17 | **80** | 63–79 |
| 9 | 81 | [81,99] | 19 | 19 | **99** | 80–98 |
| 10 | 100 | [100,120] | 21 | 21 | **120** | 99–119 |
| 11 | 121 | [121,143] | 23 | 23 | **143** | 120–142 |
| 12 | 144 | [144,144] | 1 | 1 | **144** | 143–143 |

**12 个窗口两两不相交，并集恰好 = `z_s[0..143]`**（实跑断言 `seen == set(range(144))` 为 True）。
⇒ 第 12 步**不是冗余步**，它是唯一边读到 `z_s[143]` 的步；也没有任何 token 被读两次。
（更正记录：先说过"第 12 步只读 1 token ⇒ 弱条件"——后半句错，因为解码器是
**循环携带**的 `Y = query_base + Y_prev`，末步 query 已累积前 11 步信息；后又说
"那个 token 第 11 步已读过"——也错，两步窗口不相交。以本表为准。）


## 7. 未完成 / 下一步

1. ~~Flickr2K / OpenImages 并入~~ ✅ 已完成（六源索引 278,151，v2 run 已跑完）。
2. ~~无留出评测~~ ✅ **已补**：COCO2017 **val** 5,000 张（训练只用 train）→ §6。
   ImageNet val 仍被当训练集用；若要再加独立留出，可从 Vimeo/OpenImages 未用部分切。
3. ~~每档单独 PSNR~~ ✅ 已补（§6 表里 4 档各自的 L1/PSNR）。
4. **仍未做**：`fixed --block 12` 对照臂、A 相 A/B、真实 bpp（量化+熵编码）、
   Kodak/Tecnick 标准集对标、`carry` detach 对照。
5. **§6 暴露出的最该做的一件事**：后段 80 个 token 只买到 0.20 px ⇒
   下一步应查**为什么后步不干活**（`mean_t` 平权把后步梯度稀释到 1/12？
   还是 K=144 的信息已经被前 48 个 token 吃干？），
   候选干预：后步损失加权 / 早期步监督低通目标 / 增大 K 到 576 做对照。

## 8. 产物索引

| 文件 | 内容 |
|---|---|
| `model_multi.py` | MultiResSR：可变 N 编码器 + 4 解码器 + 加权 L1 + 解码器检查点 |
| `data_multi.py` | 分桶函数 / 索引构建 / 多源数据集 / BucketBatchSampler（含 patch 预算与 DDP 对齐） |
| `train_multi.py` | 纯 DDP 训练循环（bf16、分组 lr、cosine、断点、`--reset_step_on_resume`） |
| `eval_stepwise.py` | 逐 step 精炼评测（留出集 → 每步 L1/PSNR/最优停步分布/oracle） |
| `plot_stepwise.py` / `plot_multires.py` | 出图 |
| `verify_ckpt.py` / `verify_windows.py` / `epoch_steps.py` / `summarize_v2.py` | 续训加载校验 / 读窗口核对 / epoch 步数 / 训练日志汇总 |
| `probe_vram_multires.py` | 编码器/解码器显存探针（§4 数字的来源） |
| 服务器日志 | `/root/autodl-tmp/logs/{setup,prep_data,oi_curl2,dl_rest,finalize_v2,multires_natural_v2,eval_stepwise}.log` |
