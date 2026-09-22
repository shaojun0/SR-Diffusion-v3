# 图像压缩论文合集 — 实验数据集与设置摘录（2026-09-22）

> 目的：为后续实验选测试集 / 定对比口径做参照。
> 论文原文：`/home/linaro/.openclaw/workspace-tutor/papers_codec/`（7 篇 PDF + README.md）。
> 提取文本：`/home/linaro/dsh/.tmp/papers_codec_txt/*.txt`（pypdf 抽取，页码已标）。
> 本文件数字均取自各论文正文/附录原文，标注了出处小节。
>
> **文件导航**：§0–§7 = 7 篇论文的数据集与实验设置摘录；§8 = 对我们实验的含义；
> §9 = 纯压缩实验的三档对标论文与最小对标方案；
> **§10 = bpp 口径 + 变码率背景；§11 = 提出的方法「逐 step ΔL1 早停（自适应 token 预算）」**。

---

## 0. 一页速查

| 论文 | 训练集 | 测试/评估集 | 指标 |
|---|---|---|---|
| Ballé 2017 (ICLR) | ImageNet 子集 **6,507** 张 | **Kodak 24**；+ Lena/Barbara/Peppers/Mandrill + 自拍图 | MSE→PSNR、MS-SSIM（RD 曲线） |
| Ballé 2018 (ICLR) | 网络抓取 **≈1M** 张 JPEG | **Kodak 24**；附录 **Tecnick 100** | PSNR、MS-SSIM |
| ELIC (CVPR 2022) | ImageNet 最大 **8,000** 张；另有一组训在 **CLIC-Professional** | **Kodak 24**、**CLIC Professional** | PSNR-BPP、MS-SSIM-BPP、解码延迟 |
| MLIC++ (2023/2025) | **≈10⁵ (100K)** 张：ImageNet + COCO2017 + DIV2K + Flickr2K | **Kodak 24**、**Tecnick 100**、**CLIC Professional Valid 41**；复杂度用 **LIU4K 16 张** | BD-rate、PSNR/MS-SSIM-BPP、MACs/显存/时延 |
| Yang & Mandt (NeurIPS 2023) | **Vimeo-90k**（90k 视频片段，448×256） | **Kodak 24**、**Tecnick 100**、**DIV2K val 100**、**COCO2017 2,695** | 16 种：PSNR/MS-SSIM/LPIPS/FID/KID… |
| Hoogeboom et al. 2023 | 网图 640px crop→用 MSE 自编码器编解码；+ **MS-COCO train 含人子集** | **Kodak 24**、**CLIC20 428**、**CLIC22 test 30**、**MS-COCO 30k（30,000）** | **FID(256² patch)**、PSNR |
| PerCo (SD) 2024 | **OpenImagesV6（9M）**，150k steps（官方 300k 的 50%） | **Kodak 24 (512×768)**、**MSCOCO-30k (30k, 512×512)** | FID、KID、MS-SSIM、LPIPS、CLIP-score、mIoU |

**一句话**：训练集清一色自然图像大数据（ImageNet / COCO / DIV2K / Flickr2K / OpenImages / Vimeo）；
测试端只有 4 个真正通用的集合 —— **Kodak(24)、Tecnick(100)、CLIC Professional、COCO-30k**，
其中 **Kodak 几乎人人必报**，**COCO-30k 是感知指标(FID)的默认大测试集**。

---

## 1. Ballé 2017 — End-to-end Optimized Image Compression
`papers_codec/Balle2017_EndToEnd.pdf`（ICLR 2017，arXiv 1611.01704）

- **训练集**：ImageNet (Deng et al., 2009) 子集，**6,507 张**；每个 λ 单独训一套参数。
  预处理：剔除过饱和图 → 加微量均匀噪声（模拟量化）→ 随机降采样/裁剪到 **256×256**，
  只允许重采样因子 < 0.75，太小的图丢弃。（§4 / 附录）
- **测试集**：**Kodak**（24 张未压缩图，原始 768×512）。
  另加 4 张"经典老图" **Lena / Barbara / Peppers / Mandrill** 和作者自拍照片；确认都未进训练集。
- **指标**：MSE(→PSNR) 与 MS-SSIM 的率失真曲线；灰度例图也来自 Kodak（降采样裁剪到 752×376）。
  （文中另有一处用训练集 1/3（2,169 张）画离散 vs 连续松弛的散点图，用于验证训练目标，非基准。）

## 2. Ballé 2018 — Variational Image Compression with a Scale Hyperprior
`papers_codec/Balle2018_Hyperprior.pdf`（ICLR 2018，arXiv 1802.01436）

- **训练集**：**≈100 万张** 网络抓取彩色 JPEG，原始高/宽 **3,000–5,000 px**；
  剔除过饱和 → 随机降采样到短边 **640–1,200 px** → 随机裁 **256×256**；Adam lr 1e-4。
  共训 **32 个模型**（有/无 hyperprior × {MSE, MS-SSIM} × 8 个 λ）。（§4.2 前一段）
- **测试集**：**Kodak 24**（正文主结果 + 逐图附录 §6.7）；附录 **Tecnick 100**（§6.6）。
- **指标**：PSNR、MS-SSIM（MS-SSIM 转 dB 显示）。

## 3. ELIC — Efficient Learned Image Compression
`papers_codec/ELIC_2022.pdf`（CVPR 2022，arXiv 2203.10886）

- **训练集**：ImageNet 中**最大的 8,000 张**，沿用前作的 noising-downsampling 预处理。
  λ = {4,8,16,32,75,150,300,450}×1e-4（MSE），另有一组 MS-SSIM 模型 λ={3,12,40,120}；
  每模型 2M iterations（消融 1M），batch 16。
  **另有一条实验线：直接在 CLIC-Professional 上训练**并报告其曲线。（§6.1）
- **测试集**：**Kodak 24**（主图 Fig.1/2/4）；**CLIC Professional**（Fig.3，分辨率不一，故只报时延/曲线）；
  消融/示例图用 Kodak 的 **kodim08**。高码率区间另训 λ={0.08,0.16}（Kodak 1.0–1.5 bpp）。
- **指标**：PSNR/MS-SSIM vs BPP，重点是**解码延迟**（Kodak 512×768 约 100 µs）。

## 4. MLIC++ — Linear Complexity Multi-Reference Entropy Modeling
`papers_codec/MLIC++_2023.pdf`（ICML 2023 Workshop → ACM TOMM 2025，arXiv 2307.15421）

- **训练集**：**≈10⁵（100K）张**、分辨率 > 512×512，来自
  **ImageNet + COCO 2017 + DIV2K + Flickr2K**；按 Ballé 2018 的方式随机降采样，
  使短边落在 **512–584 px**（压 JPEG 伪影）。公开在 `Whiteboat/MLIC-Train-100K`。
  训练：λ(MSE)={18,35,67,130,250,483}×1e-4，batch 32，单卡 A100，2M steps；
  前 1.2M steps 裁 256²，之后裁 512²。（§4.1）
- **测试集**：**Kodak 24（768×512）**、**Tecnick 100（1200×1200）**、
  **CLIC Professional Valid 41（≈2048×1440）**。（§4.2）
- **复杂度实验**：从 **LIU4K test** 选 **16 张 > 3584×3584** 的图，center-crop 到
  512²…3584² 八档测 MACs/显存/时延。（§4.3）
- **指标**：以 **BD-rate** 为主（对 VTM-17.0 Intra），另有 PSNR/MS-SSIM-BPP 与速度/显存。

## 5. Yang & Mandt 2023 — Lossy Image Compression with Conditional Diffusion Models
`papers_codec/YangMandt2023_Diffusion_Compression.pdf`（NeurIPS 2023，arXiv 2209.06950）

- **训练集**：**Vimeo-90k**（Xue et al., 2019）：90,000 个视频片段，每段 7 帧、**448×256**；
  每步随机取一帧再随机裁 **256×256**。NSC 对比基线则用 **DIV2K 训练集**训练。（§4 "Model Training"）
- **测试集（4 个，正文明确列出）**（§4 "Test Data"）：
  1. **Kodak** — 24 张，768×512；
  2. **Tecnick** — 100 张自然图，600×600 → 降采样到 **512×512**；
  3. **DIV2K 验证集** — 100 张，短边缩到 768 → center-crop **768×768**；
  4. **COCO2017** — 取所有 >512×512 的 test 图，缩到 **384×384**，共 **2,695 张**。
- **指标**：共 **16 种**（失真类 PSNR/MS-SSIM、感知类 LPIPS/FID/KID 等）；
  另有 ρ∈{0,0.32,0.64,0.9} 的随机性-失真权衡消融；附录给出 4 个数据集各自的 RD(P) 曲线。
  注：文中 CIFAR-10 只出现在**相关工作**里（评 Hoogeboom 2021 的 ADM 无损压缩），不是本文实验集。

## 6. Hoogeboom et al. 2023 — High-Fidelity Image Compression with Score-based Generative Models
`papers_codec/Hoogeboom2023_ScoreBased_Compression.pdf`（arXiv 2305.18231，Google Research）

- **训练集**：① 从**互联网图片**裁 **640px** crops，先用其 MSE 自编码器 encode/decode，
  再去掉 64px 边框得到 **512px** 的 (重建, 原图) 对；② 额外加入
  **MS-COCO 训练集中"含人"的子集**（作者称这一项对 COCO 评测的 FID 很关键：4.46 vs 6.79）。
  训练 2M iterations、batch 256、256px crops。（§5.3）
- **测试集**（§5.2）：
  **Kodak 24**；**CLIC20 全量 428 张**（宽至 2000px）；
  **CLIC22 test 30 张**（长边缩到 2048px，仅作视觉对比）；
  **MS-COCO 30k**（30,000 张 256×256，FID 按整图算）。
- **指标**：**FID（256×256 patch 方式）** + PSNR；探讨噪声 schedule / 采样步数 / γ 对 rate-FID 的影响。
  注：只有 CLIC20 与 MS-COCO 30k 够大能算 FID。

## 7. PerCo (SD) 2024 — Open Perceptual Compression
`papers_codec/PerCo_SD_2024.pdf`（arXiv 2409.20255）

- **训练集**：**OpenImagesV6（9M）**，AdamW，150k steps（= 官方 PerCo 配置 300k 的 50%），
  batch 80（带 LPIPS），8×H100 全精度；baseline PerCo(official) 同为 OpenImagesV6 9M。（Table 1 + §4）
- **测试集**：**Kodak**（24 张，512×768）与 **MSCOCO-30k**（30k 张，512×512）；
  沿用 PerCo 的评测协议。（§4 "Evaluation setup"）
- **指标**：FID、KID（感知）；MS-SSIM、LPIPS（失真）；CLIP-score（与 BLIP-2 生成 caption 的对齐）；
  mIoU（语义保持，ViT-Adapter 分割网）。码率覆盖 0.003–0.1+ bpp，含 0.0036 bpp 超低码率。

---

## 8. 对我们后续实验的直接含义

1. **可比测试集只有 4 个**：`Kodak(24)` / `Tecnick(100)` / `CLIC Professional(41)` / `COCO-30k`。
   我们现在的判据是 `full_pixel_l1_255`，**没有 bpp / PSNR / MS-SSIM**，
   ⇒ 与以上任何一篇都**无法直接对比**；若要挂靠文献，至少得加 **bpp 估计 + PSNR/MS-SSIM**。
2. **分辨率要显式约定**：Kodak 是 768×512、Tecnick 1200²、CLIC-P ≈2048×1440、COCO-30k 是 256²。
   我们的 pipeline 是固定 patch/576 token，评测前必须先定"整图 or center-crop / 缩放到多少"。
3. **要报 FID 就得有大留出集**：PerCo / Hoogeboom / Yang 三家都靠 **MS-COCO 30k** 算 FID
   （Hoogeboom 明说别的集太小算不了）。CoT 全量 test 有 7,570 张，量级接近但分布不同（工程隐患图）。
4. **训练分布差异**：这些论文全在自然图像（ImageNet/COCO/DIV2K/Flickr2K/OpenImages/Vimeo）。
   我们的 `construction_site` / CoT 7 数据集是工程场景 ⇒ 是"域外"，不适合直接抄它们的绝对数值，
   但**测试集可以借**（尤其 Tecnick 100 与 CLIC-P 41 张，成本低、文献可比）。
5. **训练集规模的参照**：Ballé2018 ≈1M、PerCo 9M、MLIC++ 100K、ELIC 8K、Ballé2017 6.5K。
   我们 CoT 全量 train 68,147 ≈ MLIC++ 的 100K 量级，训练侧不算小。
6. **可低成本加的对照实验**：
   - 用 **Tecnick 100 + Kodak 24 + CLIC-P 41**（共 165 张）跑一遍我们的重建，补 PSNR/MS-SSIM/bpp；
   - 若要做生成式/感知压缩叙事，再上 **COCO-30k** 算 FID（需要先实现 bpp 估计）。

---

# 附：纯图像压缩实验的可对标论文（2026-09-22 追加）

> 背景：Phase-2 对话取消，只做图像压缩。问题 = "有没有可直接对比的论文"。
> 结论：**上面 7 篇全部是纯图像压缩**（无一做对话/VQA），但**我们现在的口径还没法对比** —— 缺 bpp。

## 9.1 为什么现在严格说"不可比"

压缩论文的公共语言是 **rate–distortion**：横轴 **bpp**（真实码率），纵轴 **PSNR / MS-SSIM**（确定性码率）
或 **FID / KID / LPIPS**（感知码率）。我们现在：
- **没有 rate 项**：K=576 token 是 latent 维度，不是码率；没有量化+熵编码 ⇒ 没有 bpp；
- **只有 `full_pixel_l1_255`**：L1 与 PSNR 不同（PSNR 由 MSE 定义），别人表里没有这一列。
⇒ 想"可对比"，**必须先补 rate**，其次补 PSNR/MS-SSIM。

## 9.2 三档对标对象

| 档 | 适用形态 | 对标论文 | 公共指标 / 测试集 |
|---|---|---|---|
| **A. 确定性 RD**（主对标） | 加熵模型后成为 learned codec | **Ballé2017、Ballé2018、ELIC、MLIC++**（本目录）；CompressAI 的 `bmshj2018_* / mbt2018* / cheng2020_*` 有预训练权重 | bpp–PSNR / bpp–MS-SSIM；Kodak24 / Tecnick100 / CLIC-P 41 |
| **B. 感知/生成式** | 解码器是扩散/生成式 | **Yang&Mandt 2023、Hoogeboom 2023、PerCo (SD) 2024**（本目录）+ **PerCoV2 (2025, arXiv 2503.09368, 开源 SD3)** + MS-ILLM、DiffC、DiffEIC、PICS | 极低 bpp (0.003–0.1) 下的 **FID/KID/LPIPS**；**MSCOCO-30k** / Kodak |
| **C. tokenizer 类**（与我们架构最同构） | 固定/可变 token 数重建 | **TiTok**（32/64/128 token）、**FlexTok**（ICML 2025, arXiv 2502.13967，1D 可变长 token）、MAGVIT-v2/LFQ、VAR、MaskGIT（见 `doc/2026-09-04/LITREVIEW_v2_ideas.md`） | token 数 vs PSNR/rFID；ImageNet / COCO |

要点：**A 档和我的"固定 K → 像素"最像的不是架构，而是任务**（都是压缩）；
**C 档和我的架构最像**（都是固定少量 token），但它们也都要报 token 数 + 重建质量（PSNR/rFID）。
所以无论走哪档，**bpp/token 数与 PSNR 二者至少各有一个**。

## 9.3 建议的最小对标方案（按成本递增）

1. **零改模型、先拿 baseline（半天）**：在我们 test 集上重跑
   `JPEG / JPEG2000 / BPG / VTM(VVC)` + `CompressAI 预训练模型`，画 bpp–PSNR、bpp–MS-SSIM 曲线。
   → 这是所有压缩论文的标配 baseline，等于自带对照，且**不需要我们训练**。
2. **给模型加 rate（1–2 天）**：K 个 token 当 latent → 量化 → 挂一个 Ballé2018 式 hyperprior 熵模型，
   loss = λ·rate + 失真（失真可保留 L1 或换 MSE）。得到真实 bpp 后即可与 **Ballé2018/ELIC/MLIC++ 同表**。
3. **对齐标准测试集（1 天）**：Kodak24 + Tecnick100 + CLIC-P41（共 165 张）跑我们的模型，
   报告与论文中相同分辨率下的 PSNR/MS-SSIM；这是"和文献同表"的唯一途径。
4. **域内对比（推荐同时做）**：baseline 重跑到 `construction_site` / CoT 测试集上，
   证明在工程场景下我们的方法有价值（论文数字不能直接抄，但足以自证）。

## 9.4 需要注意的坑

- **域外**：7 篇全部训练/评估在自然图像（ImageNet/COCO/DIV2K/Flickr2K/OpenImages/Vimeo）；
  我们的工地/CoT 是分布外 ⇒ 标准集上我们的绝对数字会难看，**别把标准集当主战场，当"可比性证明"**。
- **分辨率**：Kodak 768×512、Tecnick 1200²、CLIC-P ≈2048×1440、COCO-30k 256²。
  必须明确"整图 / center-crop / 缩放"，否则 PSNR 不可比。
- **FID 需要大留出集**：只有 COCO-30k 量级才稳（Hoogeboom 明说别的集太小）。
  CoT test 7,570 张量级接近但分布不同。
- **"不报 bpp 行不行"**：必须有 **rate**（bpp 或 token 数），fixed-K 只是 latent 大小、不构成码率。
  走 C 档（TiTok/FlexTok 那条线）可以拿 **token 数**当 rate 轴；走 A/B 档必须给 **bpp**。

---

## 10. bpp 口径与早停（变码率）—— 2026-09-22 追加

> 缘起：Phase-2 对话取消后讨论"怎么才算压缩实验"。结论：**跑满 576 不是压缩**，
> 但**逐 step ΔL1 早停**把它变成"嵌套码 + 内容自适应码率"，是本项目最值得做的一条实验轴。

### 10.1 bpp 是什么

```
bpp = 压缩后总比特数 / 像素数(H×W)
```

直觉（480×640 = 307,200 像素）：原始 8-bit RGB = **24 bpp**（≈115 KB）；
JPEG 高档 ≈1–2 bpp；BPG/VTM ≈0.1–1 bpp；learned codec ≈0.1–1 bpp；
**PerCo 0.003 bpp ≈ 115 字节**。

> 口径注意：压缩界对 RGB 图按 `总比特/(H×W)`，故未压缩彩图 = 24 bpp；写 "8 bpp" 多指每通道。
> 论文里的 bpp 通常是**熵模型的 `−log₂P(ŷ)` 估计值**，不必真跑算术编码；要报"实际文件大小"才需要。

### 10.2 我们现在是多少（按 v3 输入 448×252 = 112,896 像素，D=1024）

**跑满 K=576**：`576×1024×32/112896 = 167 bpp`（fp32）——比原图还大 7 倍，**这不是压缩**；
即使量化到 1 bit/dim 也有 5.22 bpp。所以"K=576 = 压缩比 1"是错觉。

**但早停改写了这件事**，而且我们的方形分块 schedule 有个很干净的性质：

> 采样步 `[1,4,9,…,576]`，第 `t=(k+1)²` 步读 `z_s[k² … t−1]` 这一块；
> 要跑到第 t 步，前面所有块都得跑，合起来正好覆盖 `z_s[0 … t−1]`
> ⇒ **停在第 t 步 ⇔ 传了 t 个 z_s token**（再加 1 个 `z_cls`）。

| 停在 t | 传 token | bpp @1 bit/dim | bpp @0.5 bit/dim | 所处档位 |
|---:|---:|---:|---:|---|
| 9 | 10 | 0.091 | 0.045 | PerCo/生成式地盘 |
| 16 | 17 | 0.154 | 0.077 | 超低码率 |
| 36 | 37 | 0.336 | 0.168 | 低于 JPEG |
| 64 | 65 | 0.590 | 0.295 | ≈ JPEG 高档 |
| 144 | 145 | 1.315 | 0.658 | 常规 learned codec |
| 576 | 577 | 5.23 | 2.62 | 无压缩意义 |

**含义**：想进 0.1 bpp 档，得停在 **t≤9**（头 3 步）。这直接说明"K 压缩"这条轴要压到多小才有意义。

### 10.3 早停规则必须钉死的三件事

1. **L1 需要 ground truth ⇒ 停止决定只能在编码侧**，且必须把 stop index 传给解码器
   （24 档 ⇒ `log₂24 ≈ 4.6 bit`，可忽略但要计入）。若想让**解码器**自己停，
   得换无参考代理：token 熵 / 置信度 / MaskGIT 式按置信度解锁。
2. **单调性是前提，而我们 doc 里已埋雷**：`LITREVIEW_v2_ideas.md` §3 已指出
   "平方分块没有信息分层、每步无增量"。若 `L1(t)` 后段是平的+噪声，阈值 ε 会把停步点变得随机。
   另外用**绝对** ΔL1 不如用**相对** `ΔL1/L1(t)`，否则 ε 的含义随图像难度漂移。
3. **没有量化+熵模型就只能报 token 数，不能声称 bpp**（token 数本身是 TiTok/FlexTok 的合法轴）。

### 10.4 定位：这是"嵌套码/变码率"，不是补丁

早停让**单个模型**产出一个**嵌套码族** `{1,4,9,…,576}` ⇒ 天然的整条 RD 曲线；
按内容自适应停 ⇒ 可比固定码率得到更优的 RD 凸包。这正是经典**嵌入码**性质在神经侧的对应物。

必须先读/引（否则会被问"这不就是 variable-rate compression 吗"）：

- 经典嵌入码：**EZW / SPIHT / JPEG2000 SNR scalability**（"前缀即可解码"的原始出处）
- **Rippel & Bourdev, ICML 2017, Real-time adaptive image compression**（anytime 渐进 + 单模型变码率的先例）
- **FlexTok**（ICML 2025, arXiv 2502.13967，1D 可变长 token、按 token 预算采样）——最直接的对手
- **TiTok**（32/64/128 token）、**MaskGIT / VAR**（置信度迭代 / 尺度分层）
- variable-rate learned compression（如 User-Guided Variable Rate, CVPRW 2022 CLIC）

> **未检索到完全同款**（用逐 step ΔL1 早停来决定 token 数）。这既说明可能是本项目的新意，
> 也说明上面几篇**必须引、必须做区分**。

### 10.5 零成本验证（用现有产物，不重训）

`eval/**/*.json` 里已有逐图逐 step 的 `step_pixel_l1_255`，可直接算：

1. `ΔL1(t) = L1(t) − L1(t_next)` 曲线 —— **先确认它到底单调不单调、在哪儿趋平**；
2. 对若干 ε：每张图的停步分布 + 平均停步 + 折算 token/bpp；
3. 等价 RD 曲线（token 数 vs L1），以及"早停 vs 固定 t"的差距（自适应到底赚了多少）。

待写工具：`tools/analyze_earlystop.py`（读 `eval/bigdata`、`eval/baseline` 的 json 出上述三张图）。

### 10.6 变码率怎么报对比

扫 ε → 每条 ε 得一对 `(平均 bpp, 平均 PSNR)` → 一条曲线；与固定码率 codec 比时
**必须在同一平均 bpp 下比 PSNR**（或报 **BD-rate**）。单独说"平均省了 X% token"没有意义。
**编码复杂度要单列**（编码侧需跑满 loop 再算 ΔL1），这也正是 MLIC++ 那类表会分别报 Enc./Dec. 时延的原因。

---

## 11. 提出的方法：逐 step ΔL1 早停（自适应 token 预算）—— 2026-09-22

> **用户口径原话：**"576 又不是全用；我可能会判断这一步和下一步的 L1 差值，
> 当差值小于一定的值时就停止。"
>
> 本节把这个想法写成可执行的方法定义；配套的 bpp 换算、风险与对标见 §10。

### 11.1 动机：K=576 只是上界，不是必须全传

我们的解码器本来就满足两个前提，使"跑到第几步"天然成为一个**离散码率档位**：

1. **每步都直接预测整图**：`L = mean_t L1(PixelHead(Y_t), target)`，
   每个 `Y_t` 自身就是一个完整重建（不依赖后续步的累加/集成）；
2. **第 t 步只读自己那块、累计只用前缀**：第 `t=(k+1)²` 步读 `z_s[k² … t−1]`，
   要跑到第 t 步，前面所有块都得跑，合起来正好消费 `z_s[0 … t−1]`。

⇒ 既然收益递减，就**用边际收益决定停在哪一步**，而不是固定跑满 576。

### 11.2 判据（形式化）

采样步序列 `T = [t_1 < t_2 < … < t_M]`（默认 `[1,4,9,…,576]`，`M=24`）。对每张图取：

```
t* = min { t_i ∈ T : |L1(t_i) − L1(t_{i+1})| < ε }            # 绝对阈值
t* = min { t_i ∈ T : (L1(t_i) − L1(t_{i+1})) / L1(t_i) < ε_rel }   # 相对阈值（推荐）
```

即**第一个"再跑一步也几乎没变好"的步**。`ε` 是唯一超参。

### 11.3 编解码流程（伪码）

```
# ── 编码器（有原图 x）─────────────────────────────
z   = DINOv2([cls; specials(K); patches(N)](x))   # z_s = specials 位置输出
A   = [z_cls; z_s]                                # 时序序列, S = K+1
Y   = query_base
for t in T:                                       # 顺序循环，跑到哪算哪
    Y = OutputQueryDecoder.step(Y, A[lo(t):hi(t)+1])   # 每步读自己那块
    l1[t] = L1(PixelHead(Y), x)
    if 满足停判据(t):  t* = t; break
发送 = ( t*, z_cls, z_s[0 : t*-1] )               # 实际码 = 前 t* 个 token + 停步索引

# ── 解码器（无原图）───────────────────────────────
收到 (t*, token 前缀) → 重放同一循环到 t* → 输出 Y_{t*}
```

**传输开销**：`t*` 索引 `log₂M = log₂24 ≈ 4.6 bit`/图。相对整图码率可忽略，
但必须计入——这是"判据需要原图 ⇒ 只能编码侧决定"的直接代价（见 §10.3 第 1 条）。

### 11.4 变体（按实现成本）

| 变体 | 判据 | 说明 |
|---|---|---|
| 绝对阈值 | `ΔL1 < ε` | 最简单；但 ε 的含义随图像难度漂移 |
| **相对阈值（推荐）** | `ΔL1/L1(t) < ε_rel` | 几乎零改造成本，量纲无关 |
| 阈值 + 最小步数 | 先跑满 `t_min` 再开始判 | 防前几步噪声导致停太早 |
| 滞回 | 连续 2 步都低于阈值才停 | 防单点抖动 |
| 目标码率 | 在 `t·D·β ≤ B` 内取最优质 | 便于与固定 bpp 基线对齐 |
| 解码侧代理 | token 熵 / 置信度（MaskGIT 式） | 不必传 `t*`，但要换判据 |

### 11.5 对训练的影响：现有配方已兼容，不必改

`mean_t L1(PixelHead(Y_t), target)` 让**每个采样步都学会"只看自己那块前缀就预测整图"**，
这正是早停所需的训练信号——所以早停是**推理期**的开关，不要求重训。

可选增强（先不做）：随机截断前缀训练 / 嵌套 dropout，直接优化"任意前缀都要好"，
让退化曲线更单调、停判据更稳。若实验显示 `ΔL1(t)` 不单调（§10.3 第 2 条的风险），
再上这一项。

### 11.6 这条线的产出形态

- **一个模型 → 一条 RD 曲线**：扫 `ε` 得 `(平均 token 数/bpp, 平均 L1/PSNR)` 曲线；
- **与固定 t 对照**：量化"按内容自适应停"相对"所有图都停 t"赚了多少（预期在 RD 凸包上更优）；
- **停步分布**：按数据集/难度看 token 分配的偏向，本身就是一个可解释的结果。
