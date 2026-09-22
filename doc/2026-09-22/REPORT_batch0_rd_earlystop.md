# REPORT — 批 0 实测：E4a/E4b（rate + PSNR + 逐图早停），2026-09-22

> 依据：[`PLAN_paper_experiments.md`](PLAN_paper_experiments.md) 的**批 0**（零训练）。
> 代码：`infer_v2_test.py`（schema v2）、`data_v2.py`（content_mask/image_path）、
> 新工具 `tools/analyze_rd_earlystop.py`。
> 数据：BPTT 模型（`output/phase1_v2_bptt/final_model.pt`）× construction_site test **3,004 张**。

---

## 0. 一句话

批 0 做完了，而且是**零训练**：一个 checkpoint 就出了完整的 `bpp–PSNR / bpp–MS-SSIM` 曲线 + 逐图逐 step 曲线。

- ✅ **C2a 成立**：一个 checkpoint = 一条嵌套码 RD 曲线（前缀即可解码，24 档免费）。
- ❌ **C2b 不成立**：内容自适应早停**结构上没赚头** —— 任意逐图码率分配的天花板只有 **+0.13 dB**，
  实际 ε 判据最好 **+0.02 dB**。原因见 §3：逐图曲线只**整体平移**、不交叉。
- ⚠️ 因此论文的新意必须改写（见 §6）。

---

## 1. 本次改了什么（全是测量层，不动模型/不重训）

| 文件 | 改动 | 对训练的影响 |
|---|---|---|
| `infer_v2_test.py` | `schema_version: 2`：新增 `step_mse_255 / step_psnr / step_psnr_img_mean/std / step_ms_ssim`、内容区口径 `step_*_content*`、平凡基线 `trivial_black_*`、以及**逐图逐 step** `per_image[{path, content_frac, step_l1, step_mse, step_psnr, step_l1_content, step_mse_content, step_ms_ssim}]` | 无（只是推理脚本） |
| `data_v2.py` | `fit_to_canvas(..., return_box=True)`；`V2Collator(return_mask=True)` 额外产出 `content_mask`；`ParquetImageDataset` 额外返回 `image_path` | **无**：两个开关默认 False ⇒ 训练路径逐位不变（已本地断言 `return_box=False/True` 图逐位相同） |
| `tools/analyze_rd_earlystop.py` | **新建**：E4a 固定 t 表、E4b ε 扫描（含滞回）、**拉格朗日 oracle 凸包**、停步分布、前提检查、三张图 | 无 |

MS-SSIM 为**自实现**（5 尺度 / Y 通道 / Gaussian 11,σ=1.5 / 标准权重），因为服务器上
`piq / pytorch_msssim / torchmetrics / skimage / lpips / scipy / cv2` **全部缺失**。自测：同图=1.0。

**回滚点**：服务器 `sr-diffusion-v3-cot/_bak_20260922/`（原版 md5 `5935ef00…` / `3db49f67…`）。

---

## 2. E4a：固定 t 的嵌套码 RD 曲线（n=3,004，448×252=112,896 px，β=1）

| `t` | 1 | 4 | 9 | 16 | 36 | 64 | 144 | 324 | 576 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tokens | 2 | 5 | 10 | 17 | 37 | 65 | 145 | 325 | 577 |
| **bpp** | **0.018** | 0.045 | **0.091** | 0.154 | **0.336** | 0.590 | 1.315 | 2.948 | 5.234 |
| L1 (0-255) | 19.096 | 17.670 | 17.093 | 16.833 | 16.594 | 16.495 | 16.406 | **16.377** | 16.422 |
| **PSNR (dB)** | **18.15** | 18.60 | **18.81** | 18.89 | **18.95** | 18.99 | 19.01 | 19.01 | 19.01 |
| MS-SSIM | 0.5199 | 0.5563 | 0.5765 | 0.5855 | 0.5929 | 0.5954 | 0.5975 | 0.5979 | 0.5971 |

**止损判据（复现 PLAN §0.1）**：L1 单调下降前缀 **18/24**，全局最优 `t=324`；
**跑满 576 比最优差 +0.045 px**（过冲），PSNR 也从 19.01 掉到 19.01/19.00。

**最有用的一个正面结论**：

> `t=9`→**0.091 bpp / 18.81 dB**；`t=36`→**0.336 bpp / 18.95 dB**；
> 从 0.336 bpp 涨到 5.234 bpp（**15.6 倍码率**）只买到 **+0.06 dB**。
> ⇒ 一个 checkpoint 覆盖 0.018–0.336 bpp，把"跑满 K"当工作点是没有意义的。

---

## 3. E4b：内容自适应早停 —— **负结论**（这是本次最重要的发现）

停判据 `t* = min{t_i : rel(t_i) < ε_rel 且 rel(t_{i+1}) < ε_rel}`（滞回 2 步，`min_idx=1`）：

| ε_rel | 平均 bpp | PSNR(agg) | 同 bpp 的固定 t | **ΔPSNR** |
|---|---:|---:|---:|---:|
| 0.2000 | 0.045 | 18.60 | 18.60 | +0.00 |
| 0.0300 | 0.078 | 18.76 | 18.75 | +0.01 |
| 0.0100 | 0.202 | 18.92 | 18.92 | +0.00 |
| 0.0030 | 0.535 | 18.99 | 18.98 | +0.01 |
| 0.0010 | 1.061 | 19.01 | 19.00 | +0.01 |
| 0.0000 | 2.126 | 19.02 | 19.02 | +0.01 |

三条独立证据说明**不是判据没调好**：

1. **逐图 oracle**（每图各自取最优 t）：平均 2.621 bpp / 19.03 dB，同 bpp 固定 t = 19.01 ⇒ **上限 +0.01 dB**；
2. **拉格朗日 oracle 凸包**（任意逐图码率分配的天花板，扫 λ 解 `argmin_k mse_i(k)+λ·bpp_k`）：
   最大 **ΔPSNR = +0.129 dB @ 0.029 bpp** ⇒ 天花板本身就只有 0.13 dB；
3. **曲线形状**：首步 vs 末步 L1 的跨图相关 = **0.9785**，p10/p90 带宽 24.1→24.08 几乎不变
   ⇒ 各图曲线是"整体平移"的扇形，不是交叉 ⇒ 自适应在结构上无空间。
   停步分布也印证：ε=0.05 时 **81% 停在 t=4、18% 停在 t=9**（差一档而已）。

> **判定**：C2b（"内容自适应码率比固定 t 更贴近 RD 凸包"）**不足以当论文贡献**。
> 它可以作为一节 analysis/ablation 诚实写出来（"嵌套码下，内容自适应的上限是 0.13 dB"），
> 但主张必须换成 C2a + rate 账 + 语义（§6）。

---

## 4. padding / 平凡基线（PLAN §5 的两条，实测）

| 项 | 结果 |
|---|---|
| ⚠️ **PLAN 文档写错了**：填充色 | 代码是 **`fill=(0,0,0)` 黑**，不是"灰底 127"（`data_v2.py` `fit_to_canvas` 默认值） |
| construction_site test 内容区占比 | **77.61%**（padding 22.39%） |
| 平凡基线（全黑画布） | 全画布 **L1 97.2 / PSNR 6.02 dB**；内容区 **L1 128.1 / PSNR 4.93 dB** |
| 模型最左端（t=1） | 全画布 18.15 dB ≫ 6.02 dB ⇒ **曲线左端不是 padding 撑起来的** ✅ |
| 但 padding 仍抬高读数 | 全画布 19.01 dB vs **内容区 17.92 dB**（差 ~1.1 dB）⇒ 论文必须**两个口径都报** |

---

## 5. 与文献同表还差什么

| 缺口 | 状态 |
|---|---|
| rate | ✅ 有了**估计 bpp**（β=1，未量化/熵编码）。**真 bpp 仍要 E3**（熵模型） |
| PSNR / MS-SSIM | ✅ 有了（全画布 + 内容区） |
| LPIPS / FID | ❌ 服务器无 `lpips`（且离线）；FID 只在 MS-COCO-30k 算，属 E8 批 |
| 经典 codec 基线（E1） | ❌ **仍然一个都没跑** —— 这现在是**最紧要的一步** |

> **风险提示**：我们的绝对 PSNR 只有 **18–19 dB**。它是不是"有竞争力"，**完全取决于 E1 的
> JPEG/BPG 在同一 bpp 下的数字**。在拿到 E1 之前，不能对 C1 下任何结论。

---

## 6. 论文叙事要改的地方

| Claim | 原设想 | 实测后 |
|---|---|---|
| C1 RD 竞争力 | E1 + E3 + E4a | 不变，**但等 E1** |
| **C2a** 一个模型=嵌套码族 | E4a | ✅ **成立且是主贡献**（24 档一条曲线，前缀可解码；跑满反而更差） |
| ~~C2b 内容自适应~~ | E4b | ❌ **不成立**（天花板 +0.13 dB）⇒ 降为 analysis 一节 |
| C3 语义保持 | E8 | 不变，**现在是最值钱的一步**（也是唯一能和像素 codec 拉开差距的地方） |
| C4 域内训练 | E7/E9/E10 | 不变 |

**新的正面故事**（不需要新训练）：*一个冻结语义编码器 + 一个 checkpoint，直接给出一条
0.018–0.336 bpp 的嵌套码 RD 曲线；用户按预算选档位，多传 15 倍码率只换来 0.06 dB
——所以"该传多少 token"这个问题在这套架构里是**有确定答案的**（t≈9–36）。*

---

## 7. 训练状态（⚠️ 需要你拍板）

- **训练没有停**。原因：`--save_every 10000`，当时 `output_phase1_v2/` 里**只有 args.json**，
  一个 checkpoint 都没有 ⇒ kill = **白扔 3h26m 且无法 resume**；而本实验**不需要**这个模型
  （主图用的是已有的 BPTT 权重）。
- 实测**并发跑零风险**：推理峰值约 4 GB（GPU0 空闲 9.7 GB），训练进程全程存活；
  代价只是训练瞬时速率 1.83 → ~2.6 s/it（25 分钟，约合 10 分钟训练时间）。
- 但**正在跑的这条训练是 `detach`**，按 PLAN §E5 它**不能支撑早停主线**（detach 臂 mpl=2/24）。
  三个选项：
  | 选项 | 代价 | 得到 |
  |---|---|---|
  | (a) 不动，让它跑完（≈09-23 06:00） | 0 | E7 的"数据规模"结论（BASE vs BIG，同为 detach） |
  | (b) 等 step 10000 出 checkpoint-10000（≈16:04）→ 停 → 换 BPTT 重跑 | 丢 0 | 可 resume；但仍是 detach 配方 |
  | (c) 立刻 kill → 改 BPTT（去掉 `model_v2.py:528` 的 `.detach()`）跑 8,760 步（≈5 h） | 丢 3h26m | **一个真正可支撑 C2a 的域内模型**（CoT 68k 上） |

---

## 8. 下一步（按优先级）

1. **E1 经典/预训练 codec 基线**（JPEG/JPEG2000/WebP/AVIF/BPG/VTM + CompressAI 5 个）——
   在**同一 bpp 轴**上和我们的曲线同图 ⇒ 决定 C1 成不成立。**零训练，半天。**
2. **E3 真 bpp**：对实际传输的前缀挂 factorized 熵模型（→ hyperprior）⇒ 才能和 Ballé/ELIC/MLIC++ 同表。
3. **标准集 no-canvas 通路**：Kodak24 + Tecnick100 + CLIC-P41（共 165 张，不旋转不填充）⇒ 与文献同表。
4. **E8 下游语义**：同等 bpp 下 Acc/mAP —— "语义压缩"四个字的唯一证据。
5. 论文改写（§6）+ 补 E7/E10/E11。

---

## 9. 产物

| 产物 | 位置 |
|---|---|
| 逐图逐 step 原始 json（12.6 MB） | 服务器 `/root/autodl-tmp/cot_l1/eval_psnr/bptt_construction_site_test.json` |
| 曲线 json（画图用） | [`data/rd_bptt_construction_site.json`](data/rd_bptt_construction_site.json) |
| 三张图 | [`data/rd_bptt_construction_site.png`](data/rd_bptt_construction_site.png) |
| 分析命令 | `python tools/analyze_rd_earlystop.py <json> --plot out.png --save-json out.json` |
