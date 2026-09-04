# SR-Diffusion v3 — 文献调研:别人的思路(2026-09-04)

> 性质: 文献调研（web 核实, 链接可查）, 按本项目五个痛点分组。
> 目的: 为"通过 v2 实现 v1"找外部参照——K 压缩、token 塌缩、渐进语义、
> 逐 patch 读出、像素重建脚手架 vs 语义。全部条目均经 web 检索核实存在。
> ⭐ = 与本项目贴合度最高、最值得精读/借鉴。

---

## 0. 一句话结论

| 痛点 | 文献共识 | 最贴条目 |
|---|---|---|
| 图像→少量 token（K=32/64/128） | 是成熟研究轴: 1D tokenizer 32–128 token 重建+生成 | TiTok ⭐, MAGVIT-v2 ⭐ |
| register/z_s 塌缩冗余（cos 0.97） | register 有明确角色分工（局部/全局）才不塌缩; 码本塌缩有现成药方（LFQ） | Register&CLS 解耦 ⭐, MAGVIT-v2 LFQ ⭐ |
| 渐进/逐 token 解锁（曲线阶梯执念） | "渐进"要按有真实增量轴的尺度组织（分辨率/置信度）, 不是按 patch 计数 | VAR ⭐, MaskGIT ⭐ |
| 解码器读不出逐 patch（F3/F4） | 压缩解码 = bottleneck + 解码器自回归/层次上下文; 身份与内容因子化 | MAGVIT-v2, Minnen 2018 |
| 像素重建脚手架 → 语义（Phase1→2） | 高掩码率重建逼语义（MAE）; 码本用文本编码器对齐（LQAE） | MAE ⭐, LQAE ⭐ |

---

## 1. 痛点①: 把图像压进少量 token（对应 K 压缩实验与 Phase 2 少 token 进 Qwen）

### ⭐ TiTok — "An Image is Worth 32 Tokens for Reconstruction and Generation"
Yu et al., NeurIPS 2024
- 链接: [NeurIPS 论文页](https://papers.nips.cc/paper_files/paper/2024/hash/e91bf7dfba0477554994c6d64833e9d8-Abstract-Conference.html) ｜ [NeurIPS poster](https://neurips.cc/virtual/2024/poster/93338)
- 思路: 把图像压成 **32/64/128 个 1D token**（VQ 码本索引）, 训练目标 = 重建 + 生成,
  轻量解码器从码本还原。
- 对本项目: 你们的 **K=32/64/128 扫描 = TiTok 的实验轴**, 但它用的是 **码本索引 token**
  （每个 token = 码本条目索引 + 位置）, 不是"576 维连续向量当 token"——码本离散化
  强制 token 分工, 是打破你们 register 塌缩（cos 0.97）的**结构性手段**。

### ⭐ MAGVIT-v2 — "Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation"
Yu et al. (Google), arXiv:2310.05737
- 链接: [arXiv](https://arxiv.org/abs/2310.05737)
- 思路: ① **LFQ（Lookup-Free Quantizer）**: 免码本查找的量化, 从根上消除码本塌缩;
  ② **factorized codes（因子化码）**: 每个视觉 token = (空间索引因子 × 内容因子) 拆分,
  自回归逐因子生成。
- 对本项目: 两个直接可借鉴点——**LFQ 类离散化/正则防塌缩**（你们的 register 塌缩 =
  码本塌缩的近亲）; **因子化 = "patch 身份(索引)与内容解耦"的现成实现**, 正是你们
  F3（patch 身份只来自固定 query_base 模板）想修的方向, 文献已证明可行。

### SEED — "SEED: An Image Tokenizer for Large-Scale Vision-Language Models"
Ge et al., ICLR 2024
- 链接: [官方 GitHub/项目](https://github.com/AILab-CVC/SEED)
- 思路: 用**强语义视觉编码器**（CLIP 式中间层特征）做 tokenizer, 量化后 LLM 直接消费,
  少 token 保语义。
- 对本项目: Phase 2 的正面案例——"冻结语义编码器 → 少 token → LLM"已被大规模验证;
  提示 Phase 1 的目标函数或许应混合语义监督（见痛点⑤ LQAE/MAE）, 而不只是像素。

### ⭐ LQAE — "Language Quantized AutoEncoders: Towards Unsupervised Text-Image Alignment"
Liu et al., NeurIPS 2023, arXiv:2302.00902
- 链接: [NeurIPS 页](https://mlanthology.org/neurips/2023/liu2023neurips-language/) ｜ [arXiv PDF](https://export.arxiv.org/pdf/2302.00902)
- 思路: VQ 码本用**冻结文本编码器（CLIP 文本侧）做监督对齐**——视觉码与语言空间对齐,
  无需标注。
- 对本项目: GOAL §4.3"是否叠加文字 CE"的文献级答案——**用文本/CLIP 编码器给 z_s
  加对齐监督是成熟且便宜的手段**, 不必等到 Phase 2 才验证"z_s 适不适合语言头"。

### TokenPacker / Honeybee（VLM 视觉 token 减量的工程先例）
- TokenPacker: arXiv:2407.02392 — [HF 论文页](https://huggingface.co/papers/2407.02392)
  （C-Abstractor: cross-attention 把 576 视觉 token 压到 ~64）
- Honeybee: CVPR 2024 — [CVF 论文 PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Cha_Honeybee_Locality-enhanced_Projector_for_Multimodal_LLM_CVPR_2024_paper.pdf)
  （locality-enhanced projector: 强调**局部性/位置信息**对 token 压缩的重要性）
- 对本项目: "576→64 的 cross-attention abstractor + 局部性增强"与你们的 OutputQueryDecoder
  同源, 且实证**局部/位置信息不可丢**——呼应你们 query_base/pos 的角色, 也提示压缩
  目标应显式保留位置结构（而非让位置信息混进全局摘要）。

### 综述入口: "A Survey on Visual Token Compression for Efficient Vision-Language Models"
Cao, Feng et al.
- 链接: [IEEE](https://ieeexplore.ieee.org/document/11257670) ｜ [Semantic Scholar](https://www.semanticscholar.org/paper/A-Survey-on-Visual-Token-Compression-for-Efficient-Cao-Feng/ae8fdb94800b3a0cd095123b2374a1965e9ce369)
- 用途: 压缩方法的完整分类（选择/合并/抽象/量化…）, 想系统性调研先读它。

---

## 2. 痛点②: register/z_s 塌缩冗余（实测: 位置 16..63 cos≈0.97, 采样种子 cos≈0.999）

### ⭐ "Register and CLS tokens yield a decoupling of local and global features in large ViTs"
Lappe & Giese, 2025
- 链接: [Semantic Scholar](https://www.semanticscholar.org/paper/Register-and-CLS-tokens-yield-a-decoupling-of-local-Lappe-Giese/02520b20a97b3d65ed381c9840797c3ca97c0260)
- 思路: 大 ViT 里 **register token 承担局部/高频信息, CLS 承担全局**——两者角色解耦。
- 对本项目: 直接解释你们的实测——register 16..63 塌缩成"全局摘要簇" = register **失去
  局部分工**的表现; register 不会自动携带逐 patch 内容, 需要显式压力/角色要求。

### "Vision Transformers with Self-Distilled Registers"
Lappe & Giese, 2025, arXiv:2505.21501
- 链接: [ar5iv](https://ar5iv.labs.arxiv.org/html/2505.21501)
- 思路: 自蒸馏让 register 在各层承担明确一致的角色, 提升大 ViT 表现。
- 对本项目: register **需要"被要求的角色"才分化**——支持你们的 F1 方向（给 special
  输入注入 patch 内容/角色先验）, 而不是指望 24 层注意力自己学会分工。

### ToMe / FastV / VisionZip（视觉 token 冗余是普遍事实）
- ToMe, "Token Merging: Your ViT But Faster", Bolya et al. ICLR 2022 — [Semantic Scholar](https://www.semanticscholar.org/paper/Token-Merging%3A-Your-ViT-But-Faster-Bolya-Fu/1dff6b1b35e2d45d4db57c8b4e4395486c3e365f)
  （训练无关 token 合并, 说明多数 token 语义上可合并）
- FastV, "An Image is Worth 1/2 Tokens After Layer 2", ECCV 2024 Oral, arXiv:2403.06764 —
  [ar5iv](https://ar5iv.labs.arxiv.org/html/2403.06764) ｜ [GitHub](https://github.com/pkunlp-icler/FastV)
  （VLM 中视觉 token 在浅层之后大量冗余, 可剪一半仍不掉点）
- VisionZip, "Longer is Better but Not Necessary in Vision Language Models", arXiv:2412.04467 —
  [HF 论文页](https://huggingface.co/papers/2412.04467) ｜ [Semantic Scholar](https://www.semanticscholar.org/paper/VisionZip%3A-Longer-is-Better-but-Not-Necessary-in-Yang-Chen/ba7ae5960415eb6312f41443db2e336db216f509)
  （保留"显著 token"子集即可）
- 对本项目: 佐证 K 压缩方向正确且**选择/剪枝也是可行轴**（你们被 YAGNI 移除的
  选择/预算机制, 文献显示是有效方向, 值得按需复活评估）。

---

## 3. 痛点③: "渐进/逐 token 解锁"（曲线阶梯执念的出路）

### ⭐ VAR — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"
Tian et al., NeurIPS 2024 Oral
- 链接: [NeurIPS](https://neurips.cc/virtual/2024/poster/94115)
- 思路: 生成按**分辨率尺度**逐级预测（1/8 → 1/4 → 1/2 → 1）, 每级是下一级的条件;
  "粗到细"成为真实可加的信息轴, 效果超越同规模扩散。
- 对本项目: 最直接的回答——**"渐进"要按有真实增量信息的轴组织（尺度/分辨率）才成立**;
  你们按 patch 计数平方分块（0..15 → 0..63）没有信息分层, 所以每步无增量。若坚持
  "联想/渐进"叙事, 应把步计划改成**尺度分层**（低分辨率先出、高分辨率后补）,
  而不是键数分层。注意与 GOAL"不追纹理"的张力: 尺度分层补的多是高频（纹理）,
  需自行权衡。

### ⭐ MaskGIT — "Masked Generative Image Transformer"
Chang et al., CVPR 2022
- 链接: [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Chang_MaskGIT_Masked_Generative_Image_Transformer_CVPR_2022_paper.html)
- 思路: 掩码生成 + **按置信度迭代 unmask**——每步解锁模型最有把握的 token,
  逐步细化; 也支持任意掩码率推理。
- 对本项目: "逐步解锁"的**成熟形式 = 按置信度调度, 不是按固定位置块**。若想给 Phase 2
  保留"token 增量"能力, MaskGIT 式置信度迭代是已被验证的机制; 变体可看
  [Halton Scheduler for Masked Generative Image Transformer (ICLR 2025)](https://mlanthology.org/iclr/2025/besnier2025iclr-halton/)。
- 后续相关: [Improved MaskGIT / Token-Critic](https://github.com/ChrisDeverall/Improved-MaskGIT-pytorch)。

### D-AR — "Diffusion via Autoregressive Models"
Gao & Shou, 2025, arXiv:2505.23660（你们引文 [6], 已做过 dar/ 实验）
- 链接: [GitHub](https://github.com/showlab/d-ar) ｜ [Semantic Scholar](https://www.semanticscholar.org/paper/D-AR%3A-Diffusion-via-Autoregressive-Models-Gao-Shou/88c33ce92133b6c4d6d779e9d34eb18c9d9b8c72)
- 思路: 顺序 tokenizer + 自回归扩散。
- 对本项目: 外部模型实测（PSNR+0.15）对本项目验证价值有限; 顺序建模不是工地数据
  重建瓶颈所在, 参考价值低于 VAR/MaskGIT 的"轴"洞察。

---

## 4. 痛点④: 解码器读不出逐 patch（F3/F4, 6 个版本未修）

### "Variational Image Compression with a Scale Hyperprior"
Ballé et al., ICLR 2018
- 链接: [ICLR/MLAnthology](https://mlanthology.org/iclr/2018/balle2018iclr-variational/)
### "Joint Autoregressive and Hierarchical Priors for Learned Image Compression"
Minnen et al., NeurIPS 2018
- 链接: [NeurIPS/MLAnthology](https://mlanthology.org/neurips/2018/minnen2018neurips-joint/)
- 思路（合并看）: 神经压缩 = **图像 → 紧凑 bottleneck 码 → 解码器还原**; 解码器用
  **自回归/层次上下文**逐位置条件化解码; 熵模型给每部分码显式 bit 预算。
- 对本项目: ① 你们"少 token 压整图 + 逐 patch 还原"= 神经压缩框架, 不必自己发明——
  解码器"读不出逐 patch"在压缩文献的解法是**解码器侧上下文条件化**（每位置解码条件
  化于邻域/低分辨率已解码信息, 对应你们的尺度分层与逐 patch 查询注入）; ② 熵模型
  = **显式信息预算**, 可替代启发式 K 扫描, 回答"每个 token 到底该装多少 bit"。
- 注意: 压缩文献目标是码率-失真最优, 你们目标是"少 token 保语义给 Qwen"——
  借鉴结构, 不照搬码率目标。

---

## 5. 痛点⑤: 像素重建脚手架 vs 语义（Phase 1 → Phase 2 的哲学）

### ⭐ MAE — "Masked Autoencoders Are Scalable Vision Learners"
He et al., CVPR 2022
- 链接: [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.html)
- 思路: **75% 高掩码率像素重建**逼出语义编码——遮住大部分输入, 模型必须学会
  "补全 + 联想"而非复制纹理。
- 对本项目: 你们现在全可见 576 patch 平权重建, "联想压力"弱（这也是渐进/信息路由
  迟迟不出现的脚手架侧原因之一）。**低成本实验方向**: 对输入做高比例掩码（遮
  50–75% patch）再重建,"压缩 + 补全"双压力, 与 K 压缩正交叠加——直接对标
  "联想能力"（少信息还原全图）。

### （补充核实）你们引文 [3]/[4] 真实存在, 可精读
- Fan & Tong, "What Do Visual Tokens Really Encode? Uncovering Sparsity and Redundancy in
  Multimodal Large Language Models", arXiv:2603.00510 — [arXiv](https://arxiv.org/abs/2603.00510)
  （"死信息/冗余 token"依据; 2026-03 的分析, 可精读找"哪类视觉 token 冗余、怎么量化"）
- "SVD-Prune: Training-Free Token Pruning For Efficient Vision-Language Models",
  arXiv:2604.11530 — [Semantic Scholar](https://www.semanticscholar.org/paper/SVD-Prune%3A-Training-Free-Token-Pruning-For-Models-Apedo-Poreba/db2eb6e40e5f569049e396bdc0ace8175b20ea82)
  （token 剪枝的 SVD 视角）

---

## 6. 落地建议（按"通过 v2 实现 v1"排序, 均未实施）

1. **读 TiTok + MAGVIT-v2 的实现**（最贴 K 压缩轴）: 评估把 register 连续向量换
   **离散码本索引 token**（VQ/LFQ）对塌缩的疗效——这可能是"v2 一直塌缩"的结构性解。
2. **因子化查询（MAGVIT-v2）或解码器上下文（Minnen）对应你们 E2**: "patch 身份与
   内容解耦"+"解码条件化"是两条成熟路线, 任选一条实现都比继续调掩码有意义。
3. **MAE 式高掩码率输入**是零架构改动的"联想压力"实验（数据侧改 collator 即可）。
4. **LQAE/SEED 式语言对齐监督**回答 GOAL §4.3 悬案: 若 z_s 与文本编码器对齐,
   Phase 2 的适配成本会大幅下降。
5. **渐进叙事若保留**: 借鉴 VAR 尺度分层（而非键数分块）; 或 MaskGIT 置信度迭代
   （而非位置块固定顺序）——并接受它与 GOAL"不追纹理"的张力需要自行权衡。

> 本文件所有条目均经 web 检索核实（2026-09-04）; 深读前请以官方 arXiv/出版页为准。
