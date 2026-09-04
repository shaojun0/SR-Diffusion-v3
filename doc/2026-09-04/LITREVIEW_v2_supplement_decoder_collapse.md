# SR-Diffusion v3 — 文献调研补充:解码器读出/塌缩/分工的新外部参照（子智能体调研）

> 日期: 2026-09-04 ｜ 性质: 文献调研（web 检索核实, 子智能体执行）
> 定位: 补充 `LITREVIEW_v2_ideas.md`（该文档按 5 大痛点覆盖主流路线; 本文档聚焦
> **最近三天新暴露的三个失败模式**的外部参照: ① v4-vs-v2 解码器 11.6 差距
> ② exp2 读侧全开训练塌缩 ③ register 分工压力缺失）, 并新增若干完全未覆盖视角。
> 核实状态: 除个别标注外均 **[已核实链接]**（arxiv.org/abs、会议官方页、HF paper page）;
> [未能核实, 仅摘要] 仅 DDPM / Deeply-Supervised Nets 两处经典编号。
> 最小实验均按本项目设定写（448×252→576 patch、K≈64、7k 工地图）。

---

## 一、按五个失败现象分组推荐

### 现象 1: v4-vs-v2 解码器 11.6 差距（读出器设计 / 共享内容项 / 行间自注意力）

> 疑点: 给所有查询行加同一个 A_t 内容项是否有害? 行间自注意力是否稀释逐 patch 区分?
> 公开文献没有完全同构的消融（Q-Former 查询公式的专门消融论文稀缺, 建议自做 2×2×2 消融）;
> DETR 家族对"查询 = 内容 vs 位置/模板"的研究是最近的参照系。

1. **Conditional DETR for Fast Training Convergence** — Meng et al., ICCV 2021 — [已核实链接](https://arxiv.org/abs/2108.06152)
   - 机制: 查询显式拆成 **content + spatial 分支**, spatial 查询由 content embedding **条件化逐查询生成**（不是给所有查询加同一偏置）——"查询里的内容与空间信息应解耦、内容应做逐行调制而非共享加法"。
   - 对应本项目: v2 的 `q_row = A_t + query_base[row]` 中 A_t 是**所有行共享的加法偏置**——softmax 对共享偏置不敏感于行, 把 576 行一起"拉向"相同键排序, 天然压制逐 patch 区分。正确加法应是 **per-row 调制**（如 `q_row = query_base[row] ⊙ MLP(A_t)` 或让 A_t 只进 value/上下文）。
   - 最小实验: A_t 从"加到所有行"改为"逐行门控/调制", 其余不动, 看 19.878 是否回落。

2. **DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR** — Liu et al., ICLR 2022 — [已核实链接](https://arxiv.org/abs/2201.12329)
   - 机制: 查询 = 纯**位置性** 4D anchor（动态框）, 内容完全由 cross-attention 从键里读——"查询可以不带内容、只带身份, 内容进入的通道应是注意力本身"。
   - 对应本项目: 与实测一致——**v4 的纯 query_base（无内容项）反而更好**（8.26 vs 19.878）; A_t 共享内容项大概率是 v2 的负担。
   - 最小实验: 消融顺序建议 (1) 去掉 A_t → (2) 再去掉行间自注意力 → (3) 再去掉累加/均值损失, 三步分别跑, 定位 11.6 差距的三个候选成分各占多少。

3. **Dual-R-DETR: Resolving Query Competition with Pairwise Routing in Transformer Decoders** — 2025/2026, arXiv:2512.13876 — [已核实链接](https://arxiv.org/abs/2512.13876)
   - 机制: DETR 查询行在共享键上存在 **query competition**: 冗余查询互相争抢、收敛出少数赢家; 提出两两路由到不同键子集解除竞争。
   - 对应本项目: 576 行查询共享全部键 + 行间自注意力 = 教科书式竞争/冗余结构; 若某些行"赢"了, 读出内容趋同。
   - 最小实验: 行间自注意力换成"只与空间近邻行通信"或去掉（≈S25 但保留 A_t）, 与 v4 对比量化行自注意力贡献。

4. **Breaking the Softmax Bottleneck: A High-Rank RNN Language Model** — Yang et al., ICLR 2018 — [已核实链接](https://arxiv.org/abs/1711.03953)
   - 机制（理论透镜）: 从固定"字典"（键）用 softmax 读出的可表达分布集是**低秩受限**的; 解法 = 拆成多个/高秩结构。
   - 对应本项目: 576 行从 ~64 个共享键组合读出 = 低秩读出; 若行查询彼此相似（共享 A_t 偏置）或行数 >> 有效键秩, 读出秩进一步下降 → 可解释"加内容项反而更差"。
   - 最小实验: 打印 576 行 cross-attention 输出矩阵的奇异值谱与行间余弦; 若秩远小于 64 即命中此瓶颈。

### 现象 2: 读侧全开 → 训练塌缩（注意力对称/熵/秩塌缩与药方）

1. **Stabilizing Transformer Training by Preventing Attention Entropy Collapse** — Zhai et al., ICML 2023 — [已核实链接](https://arxiv.org/abs/2303.06296)
   - 机制: 注意力熵在训练早期崩到近 0 是训崩元凶; 稳定注意力（限制 logits 尺度/熵正则）可避免退化吸引子。
   - 对应本项目: "读侧全开 → 键趋同 → 注意力均匀 → 梯度湮灭" 正是熵塌缩的镜像; 给出现成药方。
   - 最小实验: 读侧全开配置下加注意力 logit 缩放/温度或熵正则, 看 64.6 塌缩是否消失。

2. **DeepViT: Towards Deeper Vision Transformer** — Zhou et al., CVPR 2021 — [已核实链接](https://arxiv.org/abs/2103.11886)
   - 机制: 深度 ViT 注意力图 **秩塌缩**（head 趋同、秩下降）; Re-Attention 跨 head 重新混合恢复多样性。
   - 对应本项目: 键→查询注意力若秩塌缩, 输出退化为常数; 提示在读出注意力后显式重建秩/多样性。
   - 最小实验: 鼓励 576×64 注意力矩阵的行秩/奇异值熵, 观察对塌缩的抵抗。

3. **Signal Propagation in Transformers ... Role of Rank Collapse** — Noci et al., NeurIPS 2022 — [已核实链接](https://arxiv.org/abs/2206.03126)
   - 机制: 理论上证明深 Transformer 输出会**收敛到秩 1 吸引子**（所有 token 同方向）, 注意力变成与输入无关的固定点; LN 位置、残差尺度决定塌缩快慢。
   - 对应本项目: "输出秩 1" = 全图平均色; "键趋同" = 固定点——塌缩是**结构性**的, 需结构层面反制。
   - 最小实验: 监控 576 行输出与 z_s 键矩阵的**有效秩/谱**随训练变化, 塌缩先于 loss 跳变出现 → 可做 early-stopping 信号。

4. **Redesigning the Transformer Architecture with Insights from Multi-particle Dynamical Systems** — NeurIPS 2021 — [已核实链接](https://arxiv.org/abs/2109.15142)
   - 机制: token = 相互吸引的粒子, 深堆叠让全体趋向**共识/均匀化**; 提出抑制趋同的动力学修正。
   - 对应本项目: 直接解释"读侧全开时没有理由让键/查询保持独立 → 滑向全图均值"; 药方 = 给粒子加"斥力/个性保持"。
   - 最小实验: 注意力权重上施加 per-key/per-query 多样性正则（拉远行方向）, 可与第 1 条熵稳定叠加。

5. **A Unifying View of Attention Sinks: Two Algorithms, Two Solutions** — 2026, arXiv:2606.08105 — [已核实链接](https://arxiv.org/abs/2606.08105)
   - 机制: 统一视角解释 LLM 中 attention sink（少数 token 吸走大部分注意力）的成因与解法。
   - 对应本项目: register 16..63 塌成"同一份全局摘要"本质 = **register 变成 attention sink**（读出压力只落在 3–15 个 token 上）。
   - 最小实验: 塌缩后检查 z_s 的注意力累积分布是否集中在 1–2 个键上（sink 诊断）。

### 现象 3: register token 分工压力缺失（竞争式赋值 / 槽位机制 / register 专业化）

1. **Object-Centric Learning with Slot Attention** — Locatello et al., NeurIPS 2020 — [已核实链接](https://arxiv.org/abs/2006.15055)
   - 机制: slot 每轮对输入做 **softmax 竞争式赋值**（软 K-means）+ GRU 携带槽历史——迭代竞争 + 身份保留 = 分工引擎, 无显式正则。
   - 对应本项目: register 不分化 = 缺少**竞争性赋值压力**（你的 softmax 在键上归一, 键之间从不竞争）。Slot Attention 是反过来: "键（registers）竞争位置（576 patches）"。
   - 最小实验: Phase 1 解码前把 64 键当 slot 对 576 patch 做一轮迭代竞争读取（小 GRU）, 看 register 两两 cos 是否从 0.96 掉下来。

2. **DINOSAUR: Bridging the Gap to Real-World Object-Centric Learning** — Seitzer et al., CVPR 2023 — [已核实链接](https://arxiv.org/abs/2209.14860)
   - 机制: 在**冻结 DINO 特征**上跑 slot attention, 特征空间重建——竞争聚类在真实图像稳定分出有意义的槽。
   - 对应本项目: (a) DINO 表征上竞争聚类的先例可直接迁移; (b) 印证"只有被有效读取的 register 才分化"; (c) 可把"读出"定义在特征空间。
   - 最小实验: 把 register 分"被读 15 个 + 不被读 49 个"分别加正交/使用正则, 量化读压力对分化的因果贡献。

3. **When Slots Compete: Slot Merging in Object-Centric Learning** — Chatzisavvas et al., 2026, arXiv:2603.11246 — [已核实链接](https://arxiv.org/abs/2603.11246)
   - 机制: 训练中 slot 会悄悄**合并、死亡**, 有效槽数远小于配置值; 提出检测合并并阻止。
   - 对应本项目: "64 个 register 实际只分化出 3–15 个" = 同现象; 给出**合并检测指标与反制损失**模板（按 register 两两余弦检测）。
   - 最小实验: 对 cos>0.95 的 register 对加显式分离损失（负余弦/正交化, 只加在未被读压力覆盖的 16..63）, 看是否破坏"同一份摘要"吸引子。

4. **Vision Transformers Don't Need Trained Registers** — Jiang et al., NeurIPS 2025 (Spotlight) — [已核实链接](https://arxiv.org/abs/2506.08010)
   - 机制: register 收益主要来自"离群吸收"与 token 结构, 不必随 ViT 训练。
   - 对应本项目: register 大量冗余时训练成本可能白花; register 分化与否由"下游怎么用"决定（呼应读压力）。
   - 最小实验: 把 16..63 号 register 冻结或共享初始化（甚至只保留 15 个可训 register 广播）, 看 Phase 1 L1 与 Phase 2 语义是否几乎不变——若不变, K=64 是"假容量", 可直接减到 ~16。

5. **Vision Transformers Need More Than Registers** — 2026, arXiv:2602.22394 — [已核实链接](https://arxiv.org/abs/2602.22394)
   - 机制: register 相关伪影在 **ViT 训练极早期就出现**, 仅加 register 不能解决, 需要正则/初始化/表征约束。
   - 对应本项目: 支撑"register 分工缺失 = 要在训练目标里加压力, 不是换 register 配置"; 提示检查 DINO 微调早期 register 是否已开始趋同。
   - 最小实验: 记录 register 谱/余弦的**训练早期曲线**, 确认趋同发生的时刻与触发条件。

6. **Test-Time Registers as Global Priors for Tokenized Image Generation** — 2026, arXiv:2607.16824 — [已核实链接](https://arxiv.org/abs/2607.16824)
   - 机制: tokenized 图像生成里把 register 当**测试时全局先验**注入生成序列。
   - 对应本项目: register 作为"全局先验"的正面用例, 与 Phase 1→2（冻结 z_s 驱动中文描述）同构; 细读其"register 何时当先验、何时当键"。
   - 最小实验: 与"采样步查询种子寄存器 cos≈0.999"对照, 验证塌缩是否因为被当成了"静态先验"而非"待读内容"。

### 现象 4: 渐进/时间轴叙事失败（每步私有目标 + 真实信息轴）

> 核心判断: "第一步扛全部、后步≈0"是"每步监督同一全图目标 + 累加"的必然结果。
> 外部参照指向共同药方——**每步配"私有监督目标 + 私有输入视图", 信息轴是真实的
> （尺度/残差/噪声级/分辨率）**。

1. **Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution (LapSRN)** — Lai et al., CVPR 2017 — [已核实链接](https://arxiv.org/abs/1704.03915)
   - 机制: 金字塔每级**只监督本级残差带**（GT 取对应尺度）, 每级私有目标 → **不存在"第一步独扛"的解**。
   - 对应本项目: 最贴近的"每步独立目标 + 信息硬截止"实例; 信息轴 = 频带。
   - 最小实验（成本最低之一）: v2 时间轴第 t 步监督目标换成 `GT − Downsample_t(GT)` 尺度带（或不同高斯模糊残差）, 观察"后步≈0"是否消失。

2. **Autoregressive Image Generation using Residual Quantization (RQ-VAE / RQ-Transformer)** — Lee et al., CVPR 2022 — [已核实链接](https://arxiv.org/abs/2203.01941)
   - 机制: 每深度对**上一层残差**再量化, 并对"只用前 d 层码重建"分别算损失; 层级间有真实增量（残差）。
   - 对应本项目: 与"多步累加 + 平权均值"最像又最不像的对照——也是累加、每步有损失, 但**第 d 步的目标是第 d−1 步的残差**（私有信息）。
   - 最小实验: 每步监督改为"该步贡献的残差目标 `L1(residual_t, x − stopgrad(x̂_{t−1}))`"（残差学习 + 分离损失 + stop-gradient）。

3. **Cascaded Diffusion Models for High Fidelity Image Generation** — Ho et al., 2022 — [已核实链接](https://arxiv.org/abs/2106.15282)
   - 机制: 分辨率级联, 每级输入 = 上采样低清图 + 本级噪声, 每级独立训练; **信息轴 = 分辨率**。
   - 对应本项目: "多级各自还原全图"可行——只要**每级输入状态被硬截止**（分辨率/噪声级）; 对照你时间轴"分块/按键数解锁无真实增量"。
   - 最小实验: 第 t 步 cross-attention **输入视图**换成下采样 1/2^t 再上采样的"粗图嵌入"（每步私有上下文）, 监督仍全图 L1。

4. **Progressive Growing of GANs for Improved Quality, Stability, and Variation** — Karras et al., ICLR 2018 — [已核实链接](https://arxiv.org/abs/1710.10196)
   - 机制: 渐进解锁分辨率, 新层 **fade-in（α 从 0 渐增）**, 先训粗层、后层只被允许学增量。
   - 对应本项目: 与"多步累加一起训→后步不学"同病; fade-in 是通用药方。
   - 最小实验: 各步累加权重改成逐步 fade-in（第 t 步输出 × α_t, α 随训练上升）, 阻止"第一步抢跑占满损失预算"。

5. **Denoising Diffusion Probabilistic Models (DDPM)** — Ho et al., 2020 — [已核实链接](https://arxiv.org/abs/2006.11239) [未能核实编号, 仅摘要]
   - 机制（损失结构模板）: 多步共享同一网络, 但每步目标（噪声 ε_t）与输入状态（带噪 x_t）**步私有、由调度硬截止**。
   - 对应本项目: "同架构多步、每步私有监督、输入视图硬截止"的黄金模板; 你的时间轴缺的是"每步私有监督与输入", 不是"多步"本身。
   - 最小实验: 每步给私有"污染视图"（随机 mask/噪声级/模糊级）作为额外条件输入, 监督不变。

> 附注: Deeply-Supervised Nets（Lee et al. 2015, [链接](https://arxiv.org/abs/1409.5185)）可作"中间层挂独立监督锚"起源参照——但它的锚是**错开的中间层表征**, 不是平权同目标; 你方"平权均值损失"正是缺少锚的错开。

### 现象 5: K≈64 压缩下的逐 patch 内容注入

1. **InternVL-X: Advancing and Accelerating InternVL Series with Efficient Visual Token Compression** — Lu et al., 2025, arXiv:2503.21307 — [已核实链接](https://arxiv.org/abs/2503.21307)
   - 机制: VLM 视觉 token 压缩最新实践（压缩率高, 保留语义与空间结构）。
   - 对应本项目: "小 token 预算下如何保留逐位置/逐区域内容"的工程决策, 直接回答"解码器靠什么拿回逐 patch 内容"。
   - 最小实验: 把 64 token 中一部分**固定绑定到空间栅格**（位置锚）, 其余自由全局, 看解码 L1 与 register 分化变化。

2. **Nüwa: Mending the Spatial Integrity Torn by VLM Token Pruning** — Huang et al., 2026, arXiv:2602.02951 — [已核实链接](https://arxiv.org/abs/2602.02951)
   - 机制: 压缩 token 会**撕碎空间完整性**, 需修补（token 携带/恢复空间身份）。
   - 对应本项目: 给 **z_s 的键也加位置码**可能比只给查询加位置更有用。
   - 最小实验: 给 z_s 每键拼接"负责区域"的位置编码（键侧位置化）, 观察 v4 型解码器 L1。

3. **Conditional Latent Coding with Learnable Synthesized Reference for Deep Image Compression** — 2025, arXiv:2502.09971 — [已核实链接](https://arxiv.org/abs/2502.09971)
   - 机制: 解码器不只吃瓶颈码, 还吃**合成/可学习参考**（条件化解码的新形态）。
   - 对应本项目: 可考虑解码器**额外条件化一张粗图**（下采样/均值图）, z_s 只编码语义-细节残差——直接服务"像素重建 = 信息保持探针"的定位（探针应测 z_s 的信息增量, 而非让 z_s 独自扛全部像素）。
   - 最小实验: 给解码器加"粗图旁路"（双线性下采样嵌入作为额外 memory/查询条件）, 测同一 z_s 下 L1 变化——若大幅下降, K≈64 全局 token 本就不该承担逐 patch 像素, 像素探针的"公平性"需重新定义。

4. **Learning with Unmasked Tokens Drives Stronger Vision Learners (LUT)** — Yang et al., ECCV 2024 — [已核实链接](https://arxiv.org/abs/2310.13593)
   - 机制: 掩码建模引入 unmasked-token 分支, 利用**逐位置真实内容线索**, 表征显著变强。
   - 对应本项目: 侧证"拿到位置自己的内容"价值极高; 反面即你的困境——64 个全局 token 让每个 patch 拿不到"自己的内容"。
   - 最小实验: 对照性思考——若 Phase 1 只是探针, 是否允许解码器训练时看到逐 patch 内容的低损版本, 让 z_s 只学"语义压缩"（会改变探针语义, 需谨慎）。

5. **I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture** — Assran et al., CVPR 2023 — [已核实链接](https://arxiv.org/abs/2301.08243)
   - 机制: 预测头用**位置化 target-block 查询**（只含位置身份）对 context encoder 输出做 cross-attention——"纯位置查询 + cross-attention 读出"是成功范式。
   - 对应本项目: v4 的成功（纯 query_base 读出）与 I-JEPA 预测头同构; 前提是**键保留逐位置内容**——你的键是 64 个全局 token, 正是 v4 只到 8.26 的结构性上限。
   - 最小实验: 把"逐位置内容"部分放回键侧——DINO 内保留 patch 级中间特征（如第 24 层前 patch tokens 池化到 8×8=64 个**带位置键**）, 解码器读"带位置的键"。

---

## 二、"本项目清单完全没覆盖"的新视角

1. **Slot Attention / 物体中心表征家族（竞争赋值 = 分工压力）** — Locatello 2020、DINOSAUR 2023、When Slots Compete 2026（见现象 3）: register 不分化问题的机制级答案可能在"让 token 互相竞争、每 token 必须独自解释一部分输入"这一族; 自带"检测合并/死亡"工具箱可搬来监控 64 个 register。
2. **注意力熵/秩塌缩稳定化文献（诊断 + 药方成套）** — Zhai 2023、DeepViT、Noci 2022、multi-particle 2021、attention sinks 2026（见现象 2）: 把"读出器退化/键趋同/输出均值化"变成**可监控（熵/谱/秩）+ 可反制（logit 约束/熵正则/head 混合/动力学修正）**的工程问题。
3. **DETR 家族的"查询竞争/查询去冗余"视角** — Conditional DETR、DAB-DETR、Dual-R-DETR（见现象 1）: 比 Q-Former 文献更适合逐 patch 读出器问题（Q-Former 查询公式专门消融论文未找到）。
4. **MoE 负载均衡损失作为"token 使用平衡"药方** — Switch Transformers, Fedus et al. 2022, [已核实链接](https://arxiv.org/abs/2101.03961): register 16..63 "没人读→塌缩" 与 **MoE 死专家同构**; 可移植"使用率正则 + 使用率统计"到 register/读出键上（约束**读出侧的读取分布**, 与 MAGVIT 码本熵正则机制不同）。
5. **Softmax Bottleneck 理论透镜**（Yang 2018, 见现象 1）: "固定键集 + 共享查询偏置 ⇒ 读出秩受限" 为 v4-v2 反直觉结果提供可检验解释 + 诊断指标（输出矩阵谱）。
6. **DDPM 式"每步私有目标 + 私有输入视图"作为损失结构模板**（见现象 4）: 扩散的成功不只靠残差, 还靠**调度把信息在步间物理切分**。
7. **2025–2026 register 专业化新文献（三篇直接命中现象 3）** — Don't Need Trained Registers (2506.08010)、Need More Than Registers (2602.22394)、Test-Time Registers as Global Priors (2607.16824); 外加 **Text Template Tokens Are Implicit Semantic Registers in Diffusion Transformers** (2607.19139, [已核实链接](https://huggingface.co/papers/2607.19139))——提示**文本/模板 token 会自然充当语义寄存器**, 对 Phase 2"register 内容 → 中文描述"是直接语义桥梁证据。

---

## 三、如果只试三个新想法（按性价比排序）

1. **【定位实验, 成本≈几天】v4/v2 逐项消融定位 11.6 差距**: 从 v2 依次去掉（共享 A_t → 行间自注意力 → 累加/均值损失）, 得三因子各自贡献; 同时打印 576 行读出矩阵谱与行余弦（softmax bottleneck 诊断）。不动 Phase 2 架构, 但决定后续所有时间轴/读出设计的判断依据。
2. **【损失结构, 成本低-中, 对 Phase 2 价值大】"累加 + 平权全图监督" → "每步私有目标"**: 首选 LapSRN 式尺度带或"残差 + stop-gradient 分离损失"（现象 4 第 1/2 条）。成功则"时间轴"才可能成为真正逐级增信息的轴——"信息逐级增加"正是 Phase 2 想从 z_s 读出的语义分层（粗语义→细节）的结构前提。
3. **【分工压力, 成本中, 直接服务 Phase 2】给"不被读的 register"加分工/使用正则 + 熵稳定**: 把 16..63 从"无人读取→塌缩成同一摘要"救出来（正交化/负余弦/Slot 竞争/读出侧使用率平衡）+ Zhai 式熵稳定防读侧塌缩; 监控每 register 被解码器读到的梯度占比。Phase 2 只有一份 z_s, register 是否分化直接决定中文描述的信息容量; 一次实验同时检验现象 2/3 两个假设。

（备选第 4: 现象 5 的"键侧位置化/粗图旁路"探针公平性实验——最能回答"K≈64 像素 L1 上限到底该是多少", 但牵动 Phase 1 探针定义, 建议放主实验后。）

---

## 四、疯狂想法（推测, 未逐条核实文献支持）

1. **读侧做成"反向 Slot Attention"**: 64 个 register（键）"竞争认领"576 个 patch, 谁认领谁负责该 patch 读出——Sinkhorn/温度退火硬-软赋值替代普通 softmax。推测: 同时破坏现象 2 对称吸引子（键不再均分梯度）与现象 3 无分工（键被迫差异化）。
2. **角色先验化 register**: 前 8 个绑定"全局摘要角色"（只被 Phase 2 读, 禁止进像素解码）, 后 56 个只进像素解码——"谁被谁读"做硬性分工轴。推测: 像素解码读压力不再稀释全局摘要 register, Phase 1/2 互不打架。
3. **初始化打破对称**: 塌缩根源之一可能是 z_s 键初始太对称。推测: 键初始化为 **DINO patch 特征（或其 k-means 中心）**而非随机向量, 让每个键一开始就"认识"一块内容, 可能直接跳过退化吸引子——成本几乎为零, 值得先试。
4. **"谁被读"反转为训练信号**: 统计每 register 累积读出权重, 把"读出权重熵"做成软标签（读出多的承载高频/细节, 少的承载低频/语义）, 引导 DINO 自组织分工——比外部正交正则更贴合"只有被读的 register 才分化"。
5. **每步"污染视图"的时间轴（DDPM 移植）**: 第 t 步额外输入模糊/降采样 2^−t 视图嵌入作为查询上下文, 监督仍全图——让"第 t 步该补哪块信息"由输入视图物理决定, 破坏后步≈0 的全局最优。

---

## 附: 核实状态说明

- 除标注外全部 **[已核实链接]**（arxiv.org/abs、NeurIPS/CVPR/ICML 官方页、HF paper page、Semantic Scholar）。
- **[未能核实, 仅摘要]**: DDPM (2006.11239) 与 Deeply-Supervised Nets (1409.5185) 编号为公认编号, 引用前请自行确认。
- **未找到的文献类型**: Q-Former/BLIP-2 查询公式的专门消融论文; "给所有查询行加同一内容向量是否有害"的直接研究——公开文献稀缺, 建议自做消融（即建议 1）。
- 配套文档: `LITREVIEW_v2_ideas.md`（主流路线）; 本文件为解码/塌缩/分工专题补充。
