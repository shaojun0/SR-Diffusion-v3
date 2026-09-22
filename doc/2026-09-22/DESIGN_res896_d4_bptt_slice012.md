# DESIGN — 896×504（=(448×2)×(252×2)）/ d4 / BPTT / slice[0:12] / fp16 + 8-bit AdamW（2026-09-22 晚）

> 用户口径（原话）：「**基于 BPTT 跑一个实验 layer=4，slice=0~12，(448×2)×(252×2)**」
> 追加：「**1，但是这次试试 fp16 吧！然后优化器是 8 位的 adamw**」（"1" = 选 40 epoch / 全量配方）。
> 本文件 = 口径定案 + 实现 + 配方 + 显存/步速实测 + 判据边界。**实测结果见同目录
> `REPORT_res896_d4_bptt_slice012.md`**（训练完成、看护评测跑完后写）。
> 启动脚本 `tools/run_896_d4_bptt_slice012.sh` ｜ 看护 `tools/eval_896_d4_bptt_slice012.sh` ｜ 代码 `ed12a66`

---

## 1. 口径定案（两条，都有仓库内部依据）

### 1.1 `(448×2)×(252×2)` = **896×504**

- 字面算术：448×2 = 896、252×2 = 504（16:9，仍是 DINOv2 patch=14 的整数倍）。
- **仓库自己把这一档列为"远期独立项目"**：`doc/2026-08-27/DIAGNOSIS_clarity.md`
  - L194：「若目标本来就是'清晰大图'，可考虑更高分辨率输入（如 **896×504**），但那会改变任务与算力，列为远期。」
  - L252：「输入分辨率 448×252→**896×504（2304 patches）**，改变任务信息量——列为远期独立项目。」
- 与本线已有三臂（`out_224_d4_bptt{,_slice1,_tap}`）的关系：`224×126 = 448/2 × 252/2`；
  本臂 = `448×2 × 252×2` ⇒ 相对 224 是 **边长 ×4 / 像素 ×16**。
- 预处理完全不变：原图 → `fit_to_canvas(1600×900)` → BICUBIC resize 到模型输入；
  **重建目标就是模型输入本身**（不是超分，是同一压缩/重建任务的更高分辨率档）。

### 1.2 `slice=0~12` ⇒ 12 步 `[1,4,…,144]`，**K = 168（不是 144）**

既有 CLI `--slice_start 0 --slice_end 12`（对 `square_block_starts(N)` 切片），本地逐位核对：

| N（分辨率） | 全量步数 | `slice[0:12]` | `K = derive_num_specials` |
|---|---|---|---|
| 144（224×126） | 12 | `[1,4,…,144]` | **144**（被 N 截断） |
| 576（448×252） | 24 | `[1,4,…,144]` | **168** |
| **2304（896×504）** | 48 | `[1,4,…,144]` | **168** |

因为 `K = min(max_t((⌊√t⌋+1)²−1), N)`：`max_t = 144` ⇒ `(12+1)²−1 = 168`，
而 `168 < N=2304` 不再被截断。⇒ 本臂 **K=168**，序列 `1 + 168 + 2304 = 2473` token。
`model_info.json` 写 `slice_start=0 / slice_end=12 / num_specials=168`，推理侧 `_pick` 以它为准（已实测自动对齐）。

> ⚠️ **这条决定了本臂的性质**：896 上 register 与 patch **不再一一对应**
> （K/N = **7.3%**；224 臂是 K=N=144，100%）。本臂实质在测「**12 步的固定 token 预算抬到 16× 像素后还够不够**」，
> 与 224 臂"无花瓶 register"不是同一类操作点。写结论时必须点明。

---

## 2. 配方（`args.json` 逐字核对）

| 项 | 值 |
|---|---|
| 输入 / 画布 | **896×504** / 1600×900（angle_step 0.5） |
| patches / specials / 序列 | 64×36 = **2304** / **168** / **2473** token |
| 采样步 | `slice_start=0, slice_end=12` ⇒ 12 步 `[1,4,9,16,25,36,49,64,81,100,121,144]` |
| 解码器 | `OutputQueryDecoder`，`decoder_depth=4`，`stack_dim=0`（= 1024，未加宽），dropout 0 |
| 读窗口 | 平方块（`layer_tap=False`），`z_s` 全部来自 DINOv2 **末层** |
| carry | **BPTT**（`carry_detach=False`，仓库当前默认） |
| 损失 | 直接预测 `mean_t L1(PixelHead(Y_t), target)`；`F_hat = Y_pix[:, -1]` |
| 精度 | **fp16**（AMP：autocast + GradScaler；**权重与 `final_model.pt` 仍 fp32**） |
| 优化器 | **`adamw_bnb_8bit`**（bitsandbytes 8-bit AdamW，0.50.2） |
| batch | 每卡 **4** × 2 卡 × `grad_accum 4` = **全局 32**（与基线 16×2×1 同） |
| 训练量 | 40 epoch = **8,760 优化步**（= 219 步/epoch × 40）；drop_last；seed 42 |
| lr / wd / clip / warmup | 1.5e-4 cosine / 0.01 / 1.0 / **262 步**（= 3%） |
| 数据 | `construction_site`：train 7,009 / test 3,004；同切分、同预处理 |
| 代码 | `model_v2.py 64fed531…` / `infer_v2_test.py 14421642…` / `data_v2.py d72ba6af…` / `train_v2.py 648222a8…`（见 §5 说明） |

---

## 3. 显存 / 步速实测（探针，单卡 RTX PRO 6000 Blackwell 97 GiB）

| 配置 | peak | 单步 | ms/img |
|---|---|---|---|
| fp32 bs=4（**仅 fwd+bwd**，无 TF32，3 次稳态取最小） | 58.3 GiB | 1.79 s | 448 |
| fp32 bs=4（更早一轮，**开了 TF32**） | 58.6 GiB | 1.29 s | 323 |
| fp32 bs=8（TF32 那轮） | — | **OOM** | — |
| **fp16 + 8bit bs=4（含优化器步 + clip）** | **34.3 GiB** | **0.41 s** | **102** |
| fp16 + 8bit bs=8（含优化器步 + clip） | 66.3 GiB | 0.91 s | 114（**单图反而更慢**） |
| fp16 + 8bit bs=12 / bs=16 | — | **OOM** | — |

⇒ **bs=4/卡 既最省显存又最快**（bs=8 因显存压力单图更慢）；用 `grad_accum 4` 凑回全局 32。
无 BatchNorm ⇒ 累积 4 个 micro-batch 与单批 bs32 数学等价（仅 fp32 舍入差）。
**整跑实测 1.67 s/优化步**（= 4×0.41 + 开销），两卡 100% util、各 40.7 GB 常驻，40 epoch ≈ **4.1 h**。
同配方若退回纯 fp32，按 448 ms/img 推算 = 0.448 × 280,360 图 / 2 卡 ≈ **17–18 h**
⇒ fp16 让 encoder/decoder 吃到 tensor core，是 4× 量级而非"省一点"。

**为什么要动 batch/精度**：896×504 下 2473 token 的激活远超 bs16；`--grad_accum` 是为了在不改全局
batch 的前提下把梯度噪声口径守住，而 fp16+8bit 是为了让这个臂在可接受墙钟内跑完 40 epoch。

---

## 4. 实现（对仓库代码的改动, **默认行为逐位不变**）

`train_v2.py` 新增两个开关（`ed12a66`）：

| 开关 | 默认 | 语义 |
|---|---|---|
| `--fp16` | `False` = 历史「全 fp32, 不套 autocast」 | `TrainingArguments(fp16=True)` ⇒ accelerate `mixed_precision="fp16"` ⇒ autocast + GradScaler；权重仍 fp32；eval 默认 `fp16_full_eval=False` 走 fp32 |
| `--optim` | `""` = **不传 optim**（保持库默认） | 显式传值才覆盖 |

> ⚠️ **踩到的坑（值得记）**：实测 **transformers 5.16.1 的库默认优化器是 `adamw_torch_fused`**，
> 不是 `adamw_torch`。所以 `--optim` 的默认值必须是「留空 = 不传」，否则新开关会把**历史各臂**的
> 优化器悄悄换掉。日志新增 `[train] 精度: … | 优化器: …` 与概览行的
> `实际优化器=… fp16=…`，让报告可直接引用实际生效值。

> ⚠️ **另一个坑**：`train_v2.py` 的 `steps_per_epoch` 公式**不计 `grad_accum`** ⇒ 日志打印
> `total_steps = 35,040`（4× 真实值），`warmup_steps = int(35,040 × warmup_ratio)`。
> **真实优化步数由 `num_train_epochs` 决定 = 8,760（tqdm 分母也是 8,760）**；为让 warmup 等于基线的
> 262 步，脚本显式给 `--warmup_ratio 0.0075`（= 262/35,040），日志已实测 `warmup 262 步`。

---

## 5. 可比性边界（**结论怎么写，先看这段**）

与基线 `out_224_d4_bptt` **逐项相同**的：数据/切分、画布与预处理、BPTT、平方块读窗口、
depth=4、直接预测损失、12 个采样步、40 epoch / 8,760 优化步、全局 batch 32、lr/wd/clip/warmup、
seed 42、优化器家族（AdamW）。

**不同（本臂是 4 变量，不是单变量对照）**：

| # | 变量 | 基线 | 本臂 | 影响 |
|---|---|---|---|---|
| 1 | 输入分辨率 | 224×126（28,224 px） | **896×504（451,584 px）** | 用户要的自变量；bpp 分母随之变 ⇒ **L1 绝对值不可跨分辨率直接比**，但 **bpp–PSNR 可同轴比** |
| 2 | 精度 | 纯 fp32 | **fp16 AMP** | GradScaler 可能因 inf/nan 跳步（日志 loss 尖刺/跳步要记录）；数值与 fp32 不完全一致 |
| 3 | 优化器 | `adamw_torch_fused`（fp32 状态） | **`adamw_bnb_8bit`** | 状态量化 + 可能的 update 差异 |
| 4 | micro-batch | 16/卡 × 1 | **4/卡 × accum 4** | 无 BatchNorm ⇒ 数学等价，仅舍入差 |
| 5 | K/N | 144/144 = 100% | **168/2304 = 7.3%** | **操作点性质不同**（见 §1.2） |

⇒ 能说的 / 不能说的：
- ✅ 同分辨率内的 **bpp–PSNR 曲线形状**（比 224 更陡还是更平）、**每个 t 是否还在改善**、
  **与同分辨率 JPEG/WebP 的对照**、逐图逐 step 的 **oracle 自适应天花板**（C2a/C2b 口径）。
- ❌ 不能把"896 的 L1 比 224 好/差 X dB"直接说成**分辨率的效应**——精度/优化器/micro-batch 三处也在动。
  要归因分辨率，需在 224 上补一臂 **fp16 + 8bit**（≈20–30 min）当锚点。

---

## 6. 判据（结果出来后按这个口径写）

1. **主表**：test 3,004 的 `full_pixel_l1_255 ± std` / 最优 `PSNR` / 最优 `MS-SSIM` / 内容区口径；
   以及与 `train1k`（seed 42 抽 1,000）的 gap ⇒ 过拟合口径。
2. **逐 step 曲线**（12 步）：L1(t) / PSNR(t) / MS-SSIM(t)；首末步**跨图相关**（越高 ⇒ 曲线只是整体平移）。
3. **bpp–PSNR / bpp–MS-SSIM**：β=1 估计 bpp = `(t+1)·1024 / 451,584`；与 JPEG/WebP@896×504 同图。
4. **oracle 自适应天花板**（E4b）：ΔPSNR 与 oracle 凸包 ⇒ C2b 是否成立（224 臂为 **+0.042 dB，不成立**）。
5. **与 224 臂对照**：只比曲线形状与相对位置，并显式声明 §5 的 4 个变量。

---

## 7. 产物 / 复跑

| 项 | 路径 |
|---|---|
| 权重 | `/root/autodl-tmp/construction_site/out_896_d4_bptt_slice012/final_model.pt`（fp32；ckpt 每 2,000 步） |
| 训练日志 | `/root/train_logs/896_d4_bptt_slice012.log` |
| 评测看护日志 | `/root/train_logs/896_eval_watcher.log`（结尾 `EVAL_ALL_DONE`） |
| 评测产物 | `/root/autodl-tmp/cot_l1/eval_896_d4/`：`test_896_d4_bptt_slice012.json`、`train1k_*.json`、`baseline_classic_896.json` |
| 状态快照 | `ssh -p 38024 root@connect.westb.seetacloud.com 'bash /root/train_logs/status_896.sh'` |

```bash
# 训练（服务器, ssh 会挂到超时, exit 124 是假失败）
cd /root/autodl-tmp/sr-diffusion-v3-tap
setsid nohup bash tools/run_896_d4_bptt_slice012.sh >/dev/null 2>&1 </dev/null &
# 看护（单独一条 ssh）
setsid nohup bash tools/eval_896_d4_bptt_slice012.sh >/root/train_logs/896_eval_watcher.log 2>&1 </dev/null &
# 冒烟（单卡 2 步, 记得换 OUT/LOG）
OUT=/root/autodl-tmp/construction_site/out_smoke_896 LOG=/root/train_logs/smoke_896.log \
  NUM_GPUS=1 SMOKE=1 bash tools/run_896_d4_bptt_slice012.sh
# 分析（bpp 分母自动取 json 的 bpp_px ⇒ 896 无需手传 --px）
python tools/analyze_rd_earlystop.py <test_*.json> --save-json rd_896.json --plot rd_896.png
python tools/plot_rd_compare.py --preset --out rd_compare_224_vs_896.png
```
