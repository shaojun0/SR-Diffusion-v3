"""
SR-Diffusion Phase 1 v2 — 无预算简化版（YAGNI: 先不增实体）
=================================================================

⚠️ 项目目标（权威版 2026-08-28 修订 v2, 见 doc/2026-08-28/GOAL_compression_for_nlp.md）:
    本模型是 Phase 1 的**训练脚手架**——用像素重建当代理任务, 训练编码器
    （DINOv2 + ReEncoder + specials）的"token 压缩 + 联想"能力。训练完成后
    **冻结编码器**, 作为 model.py 的编码器接入 Qwen 做 NLP 解码（中文工地
    描述/隐患）。**验收标准是 Phase 2 文字生成质量**。
    像素重建 = **信息保持的直接探针**（语义是像素的函数: 能还原像素 ⇒ z
    携带整图信息 ⇒ 语义信息必然在）——重建质量决定 NLP 天花板, 与 NLP
    不构成对立。推论: 纹理级清晰度（块内高频）按文献 [3] 属"死信息", 不追;
    **Phase 1 中间验收 = K 压缩 × 重建质量**（K=32/64/128 下活信息=布局/
    物体/边界的保真, 当前 K=N=576 零压缩, 尚未验证）; t≥1 平台 = 联想能力
    已成立; 冗余（576 z_s 全是全局摘要副本）是真问题, 压缩率 K 是练联想的
    真正杠杆。

沿革:
    v2 曾含"可学习前缀预算"机制（SelectHead 边界分布 + STE 门控 +
    率惩罚/目标铰链 + pad_token），统一 hard 后发现它是"先增的实体":
    与前向脱节、代码熵增、收益未验证。按 YAGNI 原则整块移除（git
    历史可找回），k 固定 = 全量 N——decoder 恒吃全部 z_s。
    后续若真需要预算（"少留也能重建"），再把 SelectHead/边界分布
    加回来。

当前架构（无选择、无预算）:
    DINOv2 → cls + patch 特征 (B,257,D)
    ReEncoder:    [cls; specials; patches] 因果 specials 块掩码 → z_cls, z_s
    OutputQueryDecoder: [z_cls; z_s] = 时序序列 A(S,H)；在采样时刻
                    T_sub = 各 z_s 分块起点（平方数 1,4,9,…,⌊√N⌋², 自动
                    适配任意 N; 每个分块一个采样步）
                    上输出 (N,D) 矩阵 = 全部 patch 的预测（输出查询注意力,
                    查询基行 k 对应 patch k）；**分块掩码**: 每步只 attend
                    自己的 z_s 块（块 k = [k², min((k+1)²-1, N)]）
                    → F_hat = Σ_t Y_t (B,N,D)（第 n 步结果 = 前 n 步之和）
    L = mean_n L1(cumsum_n(Y_t), patch)   ← 每步累积结果平权全覆盖损失
        （去掉加权体系; 每个采样步的累积重建都监督还原全部 patch）
        （2026-09-03 梯度按步解耦: 数值累加不变, 但 Y_cum = [0,cumsum(Y)[:-1]]
        detach + Y——每个 Y_t 只从自己那一步的损失收 1 份梯度, 平权,
        不再有"t=0 收 |T| 份梯度动力"的三角失衡; 见 SRPhase1V2.decode）

register_specials=True（2026-08-28 新增, 修 F1/F2）:
    specials 不再由 ReEncoder 算, 而是作为额外 token 直接拼进 DINOv2 的
    输入序列 [cls; specials; patches] (2N+1 token), 由 DINO 的 24 层直接
    算出 z_s——register token 式（Darcet et al., "Vision Transformers Need
    Registers"）。special k 的输入仍共享 token+位置编码（SpecialTokenBank,
    无 patch 内容）, 但深层网络直接做"内容路由", 不再依赖 4 层 ReEncoder
    在 1153 长序列上学路由——修 DIAGNOSIS_clarity.md 的 F1（special 无内容
    输入）与 F2（z_s 冗余全局摘要）。该模式下无 ReEncoder（省 51.6M 参数）。
    HF Dinov2Model 无 token 级注意力 mask API（见踩坑记录）, 故 DINO 内用
    全双向注意力（无掩码）——重建任务无时序因果需求, register 惯例亦然;
    解码器的分块掩码仍提供逐步增量语义。注意: 该模式让 z_s[k] 依赖全部
    patch（含 j>k）, "前缀稳定性"约束不再成立——渐进曲线语义由解码器
    分块掩码提供, 与编码器无关。

块状注意力掩码:
    ReEncoder（causal_specials=True 默认）: [cls; specials; patches]
        cls 全局；specials 因果链（special i 只见 specials≤i + 全部
        patches）；patches 全局。掩码（build_prefix_mask）在 forward
        内现算、按实际序列长度构建（牺牲一点速度，换可扩展性）。
    OutputQueryDecoder: [z_cls; z_s] 为时序序列 A=(S,H)，分块掩码——
        每步 t 只 attend 自己的 z_s 分块（块 k = [k², min((k+1)²-1, N)],
        按平方数边界铺满 1..N）; 第一个步可见自己块之前的所有元素
        （含位置 0 = z_cls, 用户需求 2026-08-31）;
        每步由输出查询注意力产生 (N,D) 矩阵 = 全部 patch 的预测
        （每步全覆盖, 见类文档）。结果沿采样步累加:
        F_hat = Σ_t Y_t（第 n 步结果 = 第 n-1 步结果 + 第 n 步预测）。
    时刻采样（显存优化）: 查询只对 T_sub 构造（默认 = 分块起点,
        见 square_block_starts）——Q 从 (S·N,D) 降到 (|T|·N,D)。

踩坑记录（重要）:
    torch 2.x 的 bool 注意力掩码约定是 True=屏蔽（_canonical_mask:
    masked_fill_(mask, -inf)），与直觉相反。初版写成 True=允许导致
    输出不依赖输入、梯度为零（自检抓到），已按 True=屏蔽 实现。
    注意该约定只对 nn.TransformerEncoderLayer/MultiheadAttention/
    TransformerDecoderLayer 成立；
    F.scaled_dot_product_attention 的 bool 掩码实测 True=允许（torch
    2.8 本机验证: mask=[True,False] 只 attend 第一个 key）。OutputQuery
    Decoder 因此统一用加法浮点掩码(-inf=屏蔽)规避歧义。

用法
----
    model = SRPhase1V2(dinov2=dinov2_model, num_patches=256, dim=768)
    out = model(pixel_values)                 # {"loss","recon","F_hat","Y_pix","target_pix"}
    loss = out["loss"]; loss.backward()
    model.eval()                              # 推理同路径

    # register 式（specials 合并进 DINO, 修 F1/F2）:
    model = SRPhase1V2(dinov2=dinov2_model, num_patches=256, dim=768,
                       register_specials=True)   # 无 ReEncoder

    自检: python model_v2.py（形状 / 两处块掩码结构 / 梯度 /
          init_reencoder_from_dino / eval 同路径 / register 模式形状与梯度）

参考资料（防止遗忘参考来源）:
    [1] Hawthorne, C., Jaegle, A., Cangea, C., Borgeaud, S., Nash, C.,
        Malinowski, M., Dieleman, S., Vinyals, O., Botvinick, M.,
        Simon, I., Sheahan, H., Zeghidour, N., Alayrac, J.-B.,
        Carreira, J., & Engel, J. (2022).
        General-purpose, long-context autoregressive modeling with
        Perceiver AR. ICML 2022, PMLR 162, 8535-8558.
        解码器机制来源: 交叉注意力输出查询 + 因果掩码（Perceiver 家族;
        见 OutputQueryDecoder 文档）。
        https://mlanthology.org/icml/2022/hawthorne2022icml-generalpurpose/
    [2] Li, J., Li, D., Savarese, S., & Hoi, S. C. H. (2023).
        BLIP-2: Bootstrapping language-image pre-training with frozen
        image encoders and large language models. ICML 2023.
        可学习查询基（query_base）与 Q-Former 的可学习 query 设计同源;
        冻结视觉编码器 + 可训练查询桥接的思路。
        https://arxiv.org/abs/2301.12597
    [3] Fan, Y., Tong, J., Zhao, A., & Shen, X. (2026).
        What do visual tokens really encode? Uncovering sparsity and
        redundancy in multimodal large language models. arXiv:2603.00510.
        视觉 token 稀疏性/冗余分析——token 压缩与 specials 设计的背景。
        https://arxiv.org/abs/2603.00510
    [4] Apedo, Y., Poreba, M., Szczepanski, M., & Bouchafa, S. (2026).
        Beyond attention scores: SVD-based vision token pruning for
        efficient vision-language models (SVD-Prune). arXiv:2604.11530.
        视觉 token 剪枝——与已移除的"可学习前缀预算"机制相关的文献。
        https://arxiv.org/abs/2604.11530
    [5] Vahdat, A., & Kautz, J. (2020).
        NVAE: A deep hierarchical variational autoencoder. NeurIPS 2020,
        33, 1966-1979.
        "死信息"取舍的对照引用: NVAE 是概率生成模型, 顶层隐变量只承担
        全局信息, 剩余高频细节用**顶层自回归先验**捕获; 本文按 [3] 把
        纹理级高频视为"死信息"故意不追——方向相反, 但处理同一观察
        （隐变量容量该装哪类信息）。注意: NVAE 的"层级"是尺度层级
        （不同分辨率）, 与本文分块掩码（同一序列内平方数前缀切块）
        结构不同, 仅思想同源。
        https://proceedings.neurips.cc/paper/2020/hash/e3b21256183cf7c2c7a66be163579d37-Abstract.html
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Sequence


# ═══════════════════════════════════════════════════════════════
# SpecialTokenBank — 特殊 token 池（输入相同，仅位置编码不同）
# ═══════════════════════════════════════════════════════════════

class SpecialTokenBank(nn.Module):
    """每个 patch 位置一个特殊 token：共享可学习向量 + 逐位置可学习位置编码。

    token (1,1,D) 对所有位置/所有图像完全相同；pos (1,N,D) 提供位置区分。
    与 patch 组合后输入 ReEncoder，输出中特殊 token 位置的表示 z_s 是
    decoder 的输入——图像原始 patch 编码不进入 decoder（只作为编码器
    输入参与聚合）。
    """

    def __init__(self, num_patches: int, dim: int):
        super().__init__()
        self.num_patches = num_patches
        self.token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)          # 共享向量
        self.pos = nn.Parameter(torch.randn(1, num_patches, dim) * 0.02)  # 位置编码

    def forward(self, B: int, device: torch.device) -> Tensor:
        return self.token.expand(B, self.num_patches, -1) + self.pos      # (B,N,D)


# ═══════════════════════════════════════════════════════════════
# build_prefix_mask — 块状前缀掩码（forward 内现算，可扩展）
# ═══════════════════════════════════════════════════════════════

def build_prefix_mask(seq_len: int, z_start: int, z_end: int,
                      device: torch.device = None) -> Tensor:
    """构建块状前缀注意力掩码（torch bool，True=屏蔽）。

    布局: [z 区域 (z_start..z_end-1) | 尾部 (z_end..seq_len-1)]
        · z 行 i: 屏蔽 z 列 (i+1..z_end-1)（因果链），可看尾部（全局）
        · 尾部行: 全开放（全局，见所有人、被所有人见）
    用于 ReEncoder（z=specials 1..N，尾部=patches）与旧版 FeatureDecoder
    （已被 OutputQueryDecoder 取代），传不同 z_start/z_end 即可。

    在 forward 内现算而非 __init__ 缓存: 牺牲一点速度，换可扩展性——
    之后支持变长序列、运行时切换掩码都只需改传参。
    """
    m = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    for i in range(z_start, z_end):
        m[i, i + 1:z_end] = True          # 屏蔽 z 内部"后面的"
    return m


# ═══════════════════════════════════════════════════════════════
# ReEncoder — pass 2, 组合编码器 [cls; specials; patches]
# ═══════════════════════════════════════════════════════════════

class ReEncoder(nn.Module):
    """对 [cls; 特殊 token; patch] 组合序列做自注意力（带块状掩码）。

    输出中特殊 token 位置的表示 z_s 是 decoder 的输入；patch 位置的
    输出直接丢弃。输入始终是完整 2N+1 序列（训练/推理全量计算）。

    块状注意力掩码（causal_specials=True，默认）:
        cls(0)              → 全局（见所有人、被所有人见）
        specials(1..N)      → 只见 cls + specials≤i + 全部 patches（前缀链）
        patches(N+1..2N)    → 全局（图像无时序，全双向）
    即 M[i,j]=1 除 special 行 i 的 special 列 j>i 之外——special p 的编码
    z_s[p] 不依赖任何"后面的 special"（前缀稳定性：直接路径上）。
    掩码在 forward 内现算（build_prefix_mask），按实际序列长度构建。

    注意: patches 仍会关注 specials（M 的 1_{patch×special} 块）——
    若将来要裁 encoder 输入省算力，需同时把 patch→special 关注也屏蔽，
    否则 patch 编码会经 specials 间接变化。

    Input:  x (B, 2N+1, D) = [cls; special_1..N; patch_1..N]
    Output: z (B, 2N+1, D)
    """

    def __init__(self, dim: int = 768, num_patches: int = 256, depth: int = 4,
                 heads: int = 8, mlp_ratio: float = 4.0,
                 causal_specials: bool = True):
        super().__init__()
        self.num_patches = num_patches
        self.causal_specials = causal_specials
        L = 2 * num_patches + 1              # cls + N specials + N patches
        self.pos_embed = nn.Parameter(torch.randn(1, L, dim) * 0.02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        L = x.shape[1]
        # forward 内现算掩码（True=屏蔽; 可扩展: 变长/运行时切换只改传参）
        z_end = self.num_patches + 1 if self.causal_specials else 1
        self.attn_mask = build_prefix_mask(L, 1, z_end, device=x.device)
        x = x + self.pos_embed[:, :L, :]
        for layer in self.layers:
            x = layer(x, src_mask=self.attn_mask)
        return self.norm(x)


# ═══════════════════════════════════════════════════════════════
# square_step_schedule — 时刻采样计划（自然数平方数, 自动适配任意 N）
# ═══════════════════════════════════════════════════════════════

def square_step_schedule(num_patches: int) -> list:
    """生成 decoder 的时刻采样计划 T_sub = {0} ∪ {k² ≤ N} ∪ {N}。

    采样时刻就是自然数平方数: 0, 1, 4, 9, 16, 25, …
    相邻间距是 (k+1)²−k² = 2k+1 的奇数: 1, 3, 5, 7, 9, …（线性递增）——
    所以"前面密、后面疏"，且比幂次计划（间距成倍拉开）温和得多。
    0 和 N 是额外补的: t=0 = 仅 z_cls 时刻; t=N = 全前缀（能力最强）。
    计划完全由 N 推导，自动适配任意序列长度（N=256→17 步,
    N=512→24 步, …），改 N 无需动代码。

    例: square_step_schedule(256)
        → [0,1,4,9,16,25,36,49,64,81,100,121,144,169,196,225,256]

    注: 2026-08-31 起 OutputQueryDecoder 改用分块掩码（每步只 attend
    自己的 z_s 块），采样计划改为 square_block_starts（块起点 = 平方数,
    不含 0/N）; 本函数保留供参考/兼容。
    """
    steps = {0}
    k = 1
    while k * k <= num_patches:
        steps.add(k * k)
        k += 1
    steps.add(num_patches)          # 最后一步（全前缀, 能力最强）总是保留
    return sorted(steps)


# ═══════════════════════════════════════════════════════════════
# square_block_starts — 分块采样计划（块起点 = 平方数, 不含 0/N）
# ═══════════════════════════════════════════════════════════════

def square_block_starts(num_patches: int) -> list:
    """生成 decoder 的采样时刻 T_sub = {k² | 1 ≤ k² ≤ N}。

    2026-08-31 用户需求: 解码器分块掩码机制下, 每个采样步只 attend
    自己的 z_s 块, 块 k = [k², min((k+1)²-1, N)]（块起点 = 平方数）。
    采样步数 = ⌊√N⌋, 自动适配任意 N（N=12→3 步 [1,4,9],
    N=256→16 步, N=576→24 步, …）。

    例: square_block_starts(12) → [1, 4, 9]
    """
    K = math.isqrt(num_patches)
    return [k * k for k in range(1, K + 1)]


# ═══════════════════════════════════════════════════════════════
# build_block_mask — 分块注意力掩码（每步只见自己的 z_s 块）
# ═══════════════════════════════════════════════════════════════

def build_block_mask(num_patches: int, steps: Sequence[int],
                     device: torch.device = None) -> Tensor:
    """构建分块注意力掩码（torch float, -inf=屏蔽, 0=允许）。

    2026-08-31 用户需求（替代原 KV 因果前缀掩码）: 每个采样步 t 只允许
    attend **自己所在的 z_s 块**——块号 k = ⌊√t⌋, 块 k = [k², min((k+1)²-1, N)]
    （按平方数边界铺满 1..N）。例（N=12, steps=[1,4,9]）:
        步 2 (t=4, 块 [4..8])   → [F,F,F,F,T,T,T,T,T,F,F,F,F]
        步 3 (t=9, 块 [9..12])  → [F,F,F,F,F,F,F,F,F,T,T,T,T]
    即"第 n-1 个 mask 的 T 块结束处, 正是第 n 个 mask 的 T 块起点"。

    **第一个步特殊处理（2026-08-31 用户补充）**:
        · 第一个步（步值最小/切片后第一个）能看到**自己块之前的所有元素,
          含位置 0 (z_cls)**——即 row[:hi+1] 全允许（hi = 自己块终点）;
        · 其余步只允许自己的块 [k², min((k+1)²-1, N)], 位置 0 屏蔽。
      理由: 用 skip_steps/max_steps 挑选子区间（如 [4:9]）时, 第一个步
      若只看自己的块, 被切片丢弃的前段 z_s 与 z_cls 信息就完全不被使用;
      让第一个步看到前面所有元素可避免信息丢失。默认全计划（steps=None）
      时第一个步 t=1 同样能看到位置 0/1——用户验收: 第一个步可见 [0,1]。

    返回 (m·N, N+1): 每个采样步的 N 行 patch 查询共享同一掩码行
    （repeat_interleave, 与旧 KV 因果掩码同构）。True=屏蔽 的 bool
    约定容易踩坑（见文件头踩坑记录），故统一返回加法浮点掩码。
    注意: 步值须 ∈ [0, N]（__init__ 已断言）; t=0 无块（空行全 -inf,
    仅当显式传 0 时出现, 默认计划不含 0）。
    """
    S = num_patches + 1                     # z_cls + N 个 z_s
    rows = []
    for i, t in enumerate(steps):
        k = math.isqrt(int(t))              # 块号 = 步值所在的平方块
        hi = min((k + 1) * (k + 1) - 1, num_patches)
        row = torch.full((S,), float("-inf"), device=device)
        if i == 0:
            # 第一个（挑选出的）步: 看到自己块之前的所有元素（含位置 0
            # z_cls）直到块终点 hi——避免切片丢弃的前段信息完全不被使用
            row[:hi + 1] = 0.0
        else:
            lo = max(k * k, 1)              # 位置 0 (z_cls) 屏蔽
            if lo <= hi:
                row[lo:hi + 1] = 0.0        # 只允许自己块内的 z_s
        rows.append(row)
    return torch.stack(rows).repeat_interleave(num_patches, dim=0)


# ═══════════════════════════════════════════════════════════════
# PixelHead — 特征 → 像素 patch 解码头（2026-08-27 目标改为像素）
# ═══════════════════════════════════════════════════════════════

class PixelHead(nn.Module):
    """把每 patch 特征 (B,N,D) 解码回像素 patch (B,N,14*14*3)。

    背景（重大 bug 修复）: 之前监督目标是 DINO patch 特征 —— 工地图
    的特征在空间上近常数（每图跨位置 std≈5e-5），模型学"输出质心"即
    达低 L1，是假收敛（像素解码 L1=64.9≈平均色）。正确目标是**原始
    像素** pixel_values：有真实空间结构，"输出质心"的损失压力会迫使
    模型保留空间信息。上一个智能体已实证真实 DINO 特征可线性解码回
    像素（L1=9.8），信息在编码器里存在，只需可学习解码头。

    输出不加激活: 像素已按 DINO_MEAN/STD 归一化（范围 ≈[-2,2]），
    L1 直接监督归一化空间，评估时再反归一化。
    """

    def __init__(self, dim: int, patch_px: int = 14 * 14 * 3):
        super().__init__()
        self.patch_px = patch_px
        self.proj = nn.Linear(dim, patch_px)   # 特征 → 588 维像素 patch

    def forward(self, feat: Tensor) -> Tensor:
        """feat: (..., N, D) → (..., N, patch_px)"""
        return self.proj(feat)


# ═══════════════════════════════════════════════════════════════
# OutputQueryDecoder — 输出查询注意力解码器（采样时刻上输出全部 patch）
# ═══════════════════════════════════════════════════════════════

class OutputQueryDecoder(nn.Module):
    """把 [z_cls; z_s] 当作时序序列 A=(S,H)，在采样时刻 T_sub 上每个
    时刻输出一个 (N,D) 矩阵 = 全部 patch 的预测，再沿时刻**累加**得到
    F_hat (B,N,D)。

    机制（一次前向，采样时刻 × 所有 patch 行完全并行）:
        A = [z_cls; z_s] + pos_embed      (B, S, D)     ← 上文的 A=(S,H)
        Y = A_t + E                       (B, |T|, N, D)  E=查询基, 行 k↔patch k
        Y = TransformerDecoder(Y, memory=A, memory_mask=掩码)   (B, |T|·N, D)
                                          内部按 num_layers 逐层执行（默认 2）
        Y = Y.reshape(...)                (B, |T|, N, D)  采样时刻 t = 全部 patch
        F_hat = Σ_t Y_t                   (B, N, D)      采样步累加集成

    解码器堆叠用官方标准模块 torch.nn.TransformerDecoder（num_layers 个
    nn.TransformerDecoderLayer 深拷贝同一规格、顺序执行——堆叠循环在标准
    模块内部, 不手写）: 每层 = 自注意力(MHA) + 交叉注意力(MHA, memory=A) +
    FFN + 残差, 全部由标准模块完成（不造轮子）。各层共享同一 memory=A
    （标准解码器式堆叠）。查询基 E 作为输入偏置进首层 q 投影（含 bias),
    与原 W_q(A_t)+E 表达力等价——两者都是「A_t 的线性函数 + 逐 patch
    可学习向量」。
    · 交叉注意力分块掩码走 memory_mask（加法浮点掩码 -inf=屏蔽, 与
      原来直接调 SDPA 语义一致, 规避 bool 约定歧义, 见文件头踩坑记录）。
    · 注意: block 含一层查询间自注意力（tgt 无掩码, 全双向）——这是
      Q-Former / Perceiver 的常规结构（参考 [1][2]）, 与旧版"查询只 attend
      KV 序列"不同, 是有意为之（选官方 block 的固有代价）。
    · 残差 + pre-norm 取代旧版 norm(ffn(Y)) 的 norm-last 无残差结构。

    分块掩码（2026-08-31 用户需求, 替代原 KV 因果前缀掩码）:
        每个采样步 t 只 attend **自己所在的 z_s 分块**（build_block_mask）:
        块号 k = ⌊√t⌋, 块 k = [k², min((k+1)²-1, N)]（平方数边界）。
        例（N=12, 默认计划 [1,4,9]）:
        步 2 (t=4, 块 [4..8])  → [F,F,F,F,T,T,T,T,T,F,F,F,F];
        步 3 (t=9, 块 [9..12]) → [F,F,F,F,F,F,F,F,F,T,T,T,T]。
        即"第 n-1 个 mask 的 T 块结束处, 正是第 n 个 mask 的 T 块起点"。
        **第一个步特殊（2026-08-31 用户补充）**: 第一个步能看到自己块
        **之前的所有元素（含位置 0 = z_cls）**——row[:hi+1] 全允许;
        其余步只允许自己的块。理由: 挑选切片后若第一个步只看自己的块,
        被切掉的前段 z_s 与 z_cls 信息完全不被使用; 默认计划第一个步
        t=1 亦如此（可见位置 0/1, 自检验收）。
        结果**沿采样步累加**: 第 n 步预测的结果 = 第 n-1 步的结果 +
        第 n 步的预测 ⇒ F_hat = Σ_t Y_t（替代旧 mean 集成）。
        **梯度按步解耦（2026-09-03 用户需求）**: 数值累加不变, 但 Y_cum
        实现为 [0, cumsum(Y)[:-1]]（carry, detach）+ Y——每步只用自己的
        预测收梯度（恰 1 份, 平权）, 见 SRPhase1V2.decode。

    时刻采样（显存优化, 默认开启）:
        查询只对 T_sub 构造——Q 从 (S·N,D) 降到 (|T|·N,D)，显存与算力
        同比例下降（N=256: 全量 257 步 → 分块计划 16 步, 约 16×）。
        · 默认计划 = square_block_starts(N)（块起点 = 平方数, 自动适配
          任意 N, 全部块都用）;
        · skip_steps/max_steps（默认 None）可选挑选: 填了即按 Python
          切片取计划子区间（如 skip_steps=4, max_steps=9 ⇔ [4:9]）——
          只保留中段若干块, 其余 z_s 块不 attend（信息不进重建）;
        · 传 steps= 可自定义采样时刻列表（如 [0, 64, 128, 256]）;
        · 注意: 分块掩码下步值须 ∈ [0, N]（超限断言拦截）; 默认计划
          不含 0, 也不支持"全量不采样"退化。
        注意: 未采样时刻不参与损失, 也不出现在 F_hat 累加里。

    损失权重（已移除，2026-08-27 用户要求"去掉加权体系"）:
        density/uniform/capability 加权机制整块删除（YAGNI），全部采样步
        平权 —— loss = mean_n L1(cumsum_n Y, target)。git 历史可找回。

    覆盖语义（每步全覆盖）:
        每个采样时刻 t 都预测全部 N 个 patch（查询基 E 的行 k 对应
        patch k, 对所有 t 相同）; 每个采样时刻的**累加结果**都被监督
        还原全部 patch。

    掩码约定: 统一用加法浮点掩码(-inf=屏蔽)。实测 torch 2.8:
        SDPA 的 bool 掩码 True=允许, 与 TransformerEncoderLayer 的
        True=屏蔽 相反（见文件头踩坑记录）。

    参考: 交叉注意力输出查询范式源自 Perceiver 家族（文件头 [1]
        Perceiver AR / Perceiver IO）; 可学习查询基 query_base 的"行 k↔
        patch k"设计同源于 BLIP-2 Q-Former 的可学习 query（[2]）。
    """

    def __init__(self, dim: int = 768, num_patches: int = 256,
                 mlp_ratio: float = 4.0,
                 steps: Optional[Sequence[int]] = None, heads: int = 8,
                 depth: int = 2, skip_steps: Optional[int] = None,
                 max_steps: Optional[int] = None):
        super().__init__()
        self.num_patches = num_patches
        # 采样计划: 显式 steps（如边界实验 [32,64]）原样使用, 不切片;
        # 默认 square_block_starts(N) = 全部 z_s 块（每块一个采样步）。
        # skip_steps/max_steps（默认 None）可挑选子区间: 填 [4:9] 语义即
        # square_block_starts(N)[4:9]（Python 切片, 步值取中段）——挑选后
        # 每个保留步仍按自身步值归属分块（build_block_mask 按 ⌊√t⌋ 定块）。
        explicit = steps is not None
        if steps is None:
            steps = square_block_starts(num_patches)
        steps = sorted(set(int(s) for s in steps))
        assert steps and all(0 <= s <= num_patches for s in steps), \
            f"steps 越界: {steps} (N={num_patches})"
        if explicit:
            self.steps = steps                 # 显式列表原样（调用方负责）
        else:
            # 可选挑选: None=None 表示不切片（全部分块）; 填了就在
            # 默认分块计划上做 Python 切片（如 [4:9] → 取计划第 4..8 个块）。
            lo = 0 if skip_steps is None else int(skip_steps)
            hi = len(steps) if max_steps is None else int(max_steps)
            assert 0 <= lo < hi <= len(steps), \
                f"skip_steps/max_steps 越界: skip={lo} max={hi} " \
                f"(计划共 {len(steps)} 步 {steps})"
            self.steps = steps[lo:hi]
            assert self.steps, \
                f"切片后无采样步: steps[{lo}:{hi}] of {steps} 为空"
        S = num_patches + 1                                # z_cls + N z_s
        self.query_base = nn.Parameter(torch.randn(num_patches, dim) * 0.02)  # 行 k↔patch k
        # 标准解码器堆叠（不造轮子）: torch.nn.TransformerDecoder 内部按
        # num_layers 深拷贝同一 decoder_layer 规格并顺序执行（含逐层传掩码),
        # 无需手写 for 循环。每层 = 自注意力 + 交叉注意力(memory=A) + FFN +
        # 残差, 各层共享同一 memory=A。输入查询 = 采样时刻状态 + 查询基, 由
        # 首层内置 q 投影（含 bias）统一投影——与原 W_q(A_t)+E 表达力等价。
        self.stack = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True),
            num_layers=depth,
        )
        self.pos_embed = nn.Parameter(torch.randn(1, S, dim) * 0.02)

    def forward(self, z_cls: Tensor, z_s: Tensor) -> Tensor:
        B, N, D = z_s.shape[0], self.num_patches, z_s.shape[-1]
        A = torch.cat([z_cls, z_s], dim=1) + self.pos_embed      # (B,S,D)
        A_t = A[:, self.steps]                                   # (B,|T|,D) 采样时刻
        Y = (A_t.unsqueeze(2) + self.query_base) \
            .reshape(B, len(self.steps) * N, D)                  # 展平 (t,k), t∈T_sub
        # 分块掩码: 每步只见自己的 z_s 块（-inf=屏蔽, 0=允许）; 用户需求
        # 2026-08-31 替代原 KV 因果前缀掩码。
        mask = build_block_mask(N, self.steps, device=A.device)  # (|T|·N, S)
        self.attn_mask = mask                                    # 供自检
        # 标准解码器堆叠（nn.TransformerDecoder 内部逐层执行, 不手写循环）:
        # tgt=查询, memory=A(键值), 分块掩码走 memory_mask; 每层:
        # 自注意力 → 交叉注意力 → FFN（pre-norm + 残差）
        Y = self.stack(Y, A, memory_mask=mask)                   # (B,|T|·N,D)
        Y = Y.reshape(B, len(self.steps), N, D)                  # (B,|T|,N,D)
        self.last_Y = Y                                          # 采样步全部 patch 预测
        return Y.sum(dim=1)                                      # (B,N,D) Σ_t Y_t 累加


# ═══════════════════════════════════════════════════════════════
# SRPhase1V2 — 主模型（自包含；无预算，k 固定全量）
# ═══════════════════════════════════════════════════════════════

class SRPhase1V2(nn.Module):

    def __init__(
        self,
        dinov2: nn.Module,
        num_patches: int = 256,
        dim: int = 768,
        reencoder_depth: int = 4,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        causal_specials: bool = True,
        decoder_steps: Optional[Sequence[int]] = None,
        patch_px: int = 14 * 14 * 3,
        register_specials: bool = False,
        decoder_depth: int = 2,
        skip_steps: Optional[int] = None,
        max_steps: Optional[int] = None,
    ):
        super().__init__()
        self.dinov2 = dinov2
        self.num_patches = num_patches
        self.dim = dim
        self.patch_px = patch_px
        self.register_specials = register_specials

        self.special_bank = SpecialTokenBank(num_patches=num_patches, dim=dim)
        # register 式: specials 直接进 DINO 由深层网络算, 无 ReEncoder
        # （条件初始化, 避免 DDP find_unused_parameters=False 报未用参数）
        self.re_encoder = None
        if not register_specials:
            self.re_encoder = ReEncoder(dim=dim, num_patches=num_patches,
                                        depth=reencoder_depth, heads=heads,
                                        mlp_ratio=mlp_ratio,
                                        causal_specials=causal_specials)
        self.decoder = OutputQueryDecoder(dim=dim, num_patches=num_patches,
                                          mlp_ratio=mlp_ratio, heads=heads,
                                          steps=decoder_steps,
                                          depth=decoder_depth,
                                          skip_steps=skip_steps,
                                          max_steps=max_steps)
        self.pixel_head = PixelHead(dim=dim, patch_px=patch_px)

    def init_reencoder_from_dino(self, num_layers: int = 4):
        """用 DINO 编码器前 num_layers 层 warm-start ReEncoder（结构对齐
        nn.TransformerEncoderLayer ↔ HF DINO layer）。register 模式下无
        ReEncoder, 直接跳过。"""
        if self.register_specials:
            print("[init] register_specials 模式: 无 ReEncoder, 跳过 warm-start")
            return
        depth = min(num_layers, len(self.re_encoder.layers), len(self.dinov2.encoder.layer))
        dino_layers = self.dinov2.encoder.layer
        for i in range(depth):
            src, dst = dino_layers[i], self.re_encoder.layers[i]
            with torch.no_grad():
                dst.norm1.load_state_dict(src.norm1.state_dict())
                dst.norm2.load_state_dict(src.norm2.state_dict())
                dst.self_attn.in_proj_weight.data.copy_(torch.cat([
                    src.attention.attention.query.weight.data,
                    src.attention.attention.key.weight.data,
                    src.attention.attention.value.weight.data,
                ], dim=0))
                dst.self_attn.in_proj_bias.data.copy_(torch.cat([
                    src.attention.attention.query.bias.data,
                    src.attention.attention.key.bias.data,
                    src.attention.attention.value.bias.data,
                ], dim=0))
                dst.self_attn.out_proj.load_state_dict(src.attention.output.dense.state_dict())
                dst.linear1.load_state_dict(src.mlp.fc1.state_dict())
                dst.linear2.load_state_dict(src.mlp.fc2.state_dict())
        print(f"[init] ReEncoder layers 0..{depth-1} warm-started from DINO encoder")

    # ── encode: 输入 → 解码器输入 z_cls, z_s（按 register_specials 分派）──
    def encode(self, pixel_values: Tensor):
        """把输入编码成解码器输入 (z_cls, z_s)，训练/推理同一路径。

        register_specials=False: DINO → [cls; specials; patches] → ReEncoder
        register_specials=True : specials 作为额外 token 拼进 DINO 输入序列,
            由 DINO 的 24 层直接算出 z_s（register token 式, Darcet et al.）——
            修 DIAGNOSIS_clarity.md F1（special 无 patch 内容输入）与 F2
            （z_s 全是冗余全局摘要）。无 ReEncoder。
        """
        if self.register_specials:
            return self._encode_register(pixel_values)
        return self._encode_reencoder(pixel_values)

    def _encode_reencoder(self, pixel_values: Tensor):
        x = pixel_values                                # (B,3,H,W)
        out = self.dinov2(x)
        feats = out.last_hidden_state                   # (B,257,D)
        cls = feats[:, 0]                               # (B,D)
        patch_feat = feats[:, 1:]                       # (B,N,D) 仅作编码输入, 不作监督目标
        # ReEncoder: [cls; specials; patches] → z_cls, z_s
        specials = self.special_bank(x.shape[0], x.device)      # (B,N,D)
        enc_in = torch.cat([cls.unsqueeze(1), specials, patch_feat], dim=1)  # (B,2N+1,D)
        z = self.re_encoder(enc_in)                     # (B,2N+1,D)
        return z[:, 0:1], z[:, 1:1 + self.num_patches]  # (B,1,D), (B,N,D)

    def _encode_register(self, pixel_values: Tensor):
        """register 式: specials 直接进 DINO 输入序列, 由 DINO 深层算 z_s。

        HF Dinov2Model 无 token 级注意力 mask API（见文件头踩坑记录）, 故
        DINO 内用**全双向注意力**（无掩码）——重建任务无时序因果需求,
        register token 惯例亦然; 解码器的分块掩码仍提供逐步增量语义。
        embeddings(pixel_values) 复用 HF 的 cls/patch 嵌入 + 位置编码
        （含其自动插值逻辑）, 与训练现状一致; specials 用 SpecialTokenBank
        （共享 token + 逐位置可学习 pos）。
        """
        x = pixel_values                                # (B,3,H,W)
        emb = self.dinov2.embeddings(x)                 # (B,1+N,D) [cls; patches] + PE
        specials = self.special_bank(x.shape[0], x.device)   # (B,N,D) token+pos
        seq = torch.cat([emb[:, :1], specials, emb[:, 1:]], dim=1)   # (B,2N+1,D)
        for layer in self.dinov2.encoder.layer:         # DINO 24 层（全双向）
            out = layer(seq)
            seq = out[0] if isinstance(out, (tuple, list)) else out
        seq = self.dinov2.layernorm(seq)                # (B,2N+1,D)
        return seq[:, :1], seq[:, 1:1 + self.num_patches]

    # ── decode: 共享解码尾（Decoder → PixelHead → 像素损失）──
    def decode(self, z_cls: Tensor, z_s: Tensor, pixel_values: Tensor) -> dict:
        """解码器 + 像素头 + 平权全覆盖像素 L1（两种模式共用）。

        累加语义（2026-08-31 用户需求）: 分块掩码下每步只用自己的 z_s 块
        预测全部 patch; **结果沿采样步累加**——第 n 步预测的结果 = 第 n-1 步
        的结果 + 第 n 步的预测, 即 Y_cum[:, n] = Σ_{t≤n} Y_t（特征空间累加,
        再统一过 PixelHead——PixelHead 含 bias, 先投影再累加会重复加 bias）。
        **梯度按步解耦（2026-09-03 用户需求）**: 数值上 Y_cum = Σ_{t≤n} Y_t
        不变, 但实现为 carry[n] = [0, cumsum(Y)[:-1]][n] = Σ_{t<n} Y_t（detach）
        + Y[n]——每个 Y_t 只从**自己那一步的损失**收 1 份梯度（平权 1/|T|）,
        不再从所有 ≥t 的累加位置收梯度（否则 t=0 有 |T| 份梯度动力, 学乱）。

        dict: {"loss", "recon", "F_hat"(像素 B,N,588), "Y_pix"(每采样步**累加**
        像素 B,|T|,N,588: Y_pix[:,n] = 前 n 步预测之和的像素), "target_pix"(B,N,588)}
        —— 训练取 loss; 推理取 F_hat / Y_pix / target_pix（全量 L1、渐进曲线、
        可视化同一路径）。
        """
        x = pixel_values
        B, C, H, W = x.shape
        N = self.num_patches
        # OutputQueryDecoder: 采样时刻上每步 (N,D) 全覆盖（返回值=特征累加,
        # 此处只用其 last_Y 副作用; 最终像素累加在下方由 Y_cum 给出）
        self.decoder(z_cls, z_s)                        # 前向, 填充 last_Y
        Y = self.decoder.last_Y                         # (B,|T|,N,D) 每步全部 patch
        # 像素目标: (B,3,H,W) 归一化像素 → (B,N,588) patch
        # 注意布局: DINO 的 patch 顺序是 row-major (先 y 后 x), 这里保持一致
        target_pix = x.reshape(B, C, H // 14, 14, W // 14, 14) \
                      .permute(0, 2, 4, 1, 3, 5) \
                      .reshape(B, N, C * 14 * 14)       # (B,N,588) 归一化像素
        # 特征空间累加 → 统一过 PixelHead（bias 只加一次）。
        # 梯度按步解耦（2026-09-03 用户需求）: 数值上仍 Y_cum[n] = Σ_{t≤n} Y_t
        # （最终 F_hat = Σ_t Y_t 不变）, 但梯度只让每步**自己的预测**通过——
        # 前面步的累加 carry 整体 detach。否则 Y_t 会从所有 ≥t 的累加位置收
        # 梯度（t=0 有 |T| 份梯度动力, 各步梯度动力三角失衡, 学乱）。
        # 实现（用户给定式）: [0, cumsum(last_Y)[:-1]] 无梯度 + last_Y(整张 Y):
        #   carry[n] = Σ_{t<n} Y_t（detach）; + Y[n] ⇒ 每步恰收 1 份梯度。
        Y_cum = torch.cat([torch.zeros_like(Y[:, :1]),
                           Y.cumsum(dim=1)[:, :-1]], dim=1).detach() + Y   # (B,|T|,N,D)
        Y_pix = self.pixel_head(Y_cum)                  # (B,|T|,N,588) 累加像素
        F_pix = Y_pix[:, -1]                            # (B,N,588) 最终 = Σ_t Y_t
        # 平权全覆盖损失: 每个采样步的**累加结果**都还原全部 patch 像素
        per_step = F.l1_loss(Y_pix, target_pix.unsqueeze(1).expand_as(Y_pix),
                             reduction="none").mean(dim=(0, 2, 3))   # (|T|,)
        loss = per_step.mean()                          # 平权（去掉加权体系）
        recon = F.l1_loss(F_pix, target_pix)            # 集成重建（监控用, 归一化空间）
        return {"loss": loss, "recon": recon, "F_hat": F_pix,
                "Y_pix": Y_pix, "target_pix": target_pix}

    # ── forward（训练/推理同一路径，无分支）──
    def forward(self, pixel_values: Tensor) -> dict:
        """pixel_values: (B,3,H,W) 归一化像素 → dict{loss, recon, F_hat,
        Y_pix, target_pix}

        目标（2026-08-27 重大修复）: 监督**原始像素**而非 DINO patch 特征。
        之前监督特征目标退化（工地图特征空间近常数, 学质心即低 L1）;
        像素目标有真实空间结构, 强制模型保留空间信息。
        H,W 须为 14 的倍数（DINO patch=14），patch_px = 14*14*3。

        损失: 每个采样步的**累加结果**（第 n 步结果 = 前 n 步预测之和,
        2026-08-31 用户需求）都监督还原全部 patch 像素 —— 平权全覆盖
        损失 L = mean_n L1(cumsum_n Y_pix, target_pix)（去掉加权体系）。
        F_hat = Σ_t Y_t → PixelHead → 像素; recon 仅作监控。
        """
        x = pixel_values                                # (B,3,H,W)
        B, C, H, W = x.shape
        N = self.num_patches
        assert W % 14 == 0 and H % 14 == 0, "输入须为 14 的倍数"
        assert (W // 14) * (H // 14) == N, \
            f"输入 {W}x{H} 产生 {(W//14)*(H//14)} patches, 但模型 num_patches={N}"
        z_cls, z_s = self.encode(x)
        return self.decode(z_cls, z_s, x)


# ═══════════════════════════════════════════════════════════════
# 自检（python model_v2.py）
#   1. 形状正确性
#   2. 两处块掩码结构（ReEncoder 前缀 / OutputQueryDecoder 分块掩码）
#   3. 梯度流向（整模型可训: ReEncoder / Decoder / SpecialTokenBank）+
#      梯度按步解耦（Y_cum 数值==cumsum, 每步 Y_t 只收自己那一步的梯度）
#   4. eval 同路径
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from types import SimpleNamespace

    torch.manual_seed(0)

    # ── 假的 DINOv2：结构对齐 HF Dinov2Model（供 init_reencoder_from_dino）──
    class _FakeDinoLayer(nn.Module):
        """HF Dinov2EncoderLayer 属性布局（供 init_reencoder_from_dino 拷贝）
        + 真实单头自注意力 forward（供 register 模式直接调用）。"""
        def __init__(self, dim: int, mlp_ratio: float = 4.0):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            att = nn.Module()
            att.attention = nn.Module()
            att.attention.query = nn.Linear(dim, dim)
            att.attention.key = nn.Linear(dim, dim)
            att.attention.value = nn.Linear(dim, dim)
            att.output = nn.Module()
            att.output.dense = nn.Linear(dim, dim)
            self.attention = att
            mlp = nn.Module()
            mlp.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
            mlp.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
            self.mlp = mlp

        def forward(self, hidden_states: Tensor, head_mask=None,
                    output_attentions=False):
            n = self.norm1(hidden_states)
            q = self.attention.attention.query(n)
            k = self.attention.attention.key(n)
            v = self.attention.attention.value(n)
            scores = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
            attn = F.softmax(scores, dim=-1)
            h = hidden_states + self.attention.output.dense(attn @ v)
            h = h + self.mlp.fc2(F.gelu(self.mlp.fc1(self.norm2(h))))
            return (h, None)

    class _FakeEmbeddings(nn.Module):
        """对齐 HF Dinov2Embeddings 的最小结构: conv patch + cls + PE(+插值)。"""
        def __init__(self, dim: int, num_patches: int):
            super().__init__()
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.patch_embeddings = nn.Conv2d(3, dim, kernel_size=14, stride=14)
            self.position_embeddings = nn.Parameter(
                torch.randn(1, num_patches + 1, dim) * 0.02)
            self.dropout = nn.Identity()

        def forward(self, pixel_values: Tensor, bool_masked_pos=None,
                    interpolate_pos_encoding=None) -> Tensor:
            B = pixel_values.shape[0]
            patch = self.patch_embeddings(pixel_values)      # (B,D,h,w)
            patch = patch.flatten(2).transpose(1, 2)         # (B,N,D)
            cls = self.cls_token.expand(B, -1, -1)
            emb = torch.cat([cls, patch], dim=1)             # (B,1+N,D)
            pe = self.position_embeddings
            if emb.shape[1] != pe.shape[1]:                  # 对齐 HF 自动插值
                pe = F.interpolate(pe.transpose(1, 2).unsqueeze(0),
                                   size=(emb.shape[1],), mode="linear",
                                   align_corners=False).squeeze(0).transpose(1, 2)
            return self.dropout(emb + pe)

    class FakeDino(nn.Module):
        def __init__(self, dim: int = 64, n_layers: int = 4, num_patches: int = 16):
            super().__init__()
            self.embeddings = _FakeEmbeddings(dim, num_patches)
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList(
                [_FakeDinoLayer(dim) for _ in range(n_layers)])
            self.layernorm = nn.LayerNorm(dim)
            self._np, self._dim = num_patches, dim
            self._feat = torch.randn(4, num_patches + 1, dim)   # 固定特征（B≤4）

        def forward(self, pixel_values: Tensor):
            B = pixel_values.shape[0]
            return SimpleNamespace(
                last_hidden_state=self._feat[:B].clone())

    N, D = 16, 64
    dino = FakeDino(dim=D, num_patches=N)
    model = SRPhase1V2(dino, num_patches=N, dim=D,
                       reencoder_depth=2,
                       decoder_steps=square_block_starts(N))  # 显式传完整分块计划(不切片)
    model.init_reencoder_from_dino(2)
    # 像素目标绑定输入尺寸: N=16 patches ⇒ 输入须为 4×4 patch = 56×56 (14 的倍数)
    x = torch.randn(2, 3, 56, 56)
    B, C, H, W = x.shape

    # ── 1. 形状: 目标=像素, F_hat 是像素 patch (B,N,588) ──
    PATCH_PX = 14 * 14 * 3
    out = model(x)
    assert out["F_hat"].shape == (2, N, PATCH_PX), out["F_hat"].shape
    assert out["loss"].shape == () and out["recon"].shape == ()
    # 像素目标提取与模型内部一致
    target = x.reshape(B, C, H // 14, 14, W // 14, 14) \
               .permute(0, 2, 4, 1, 3, 5).reshape(B, N, PATCH_PX)
    assert torch.isclose(out["recon"],
                         F.l1_loss(out["F_hat"], target)), "recon 应为像素 L1"
    print(f"[ok] shapes: F_hat{tuple(out['F_hat'].shape)} (像素 {PATCH_PX}D) "
          f"loss={out['loss'].item():.4f}")

    # ── 2. 块状注意力掩码结构（ReEncoder: causal specials）──
    am = model.re_encoder.attn_mask                  # (2N+1, 2N+1) bool, True=屏蔽
    assert am.shape == (2 * N + 1, 2 * N + 1)
    assert not am[0].any()                           # cls 行无屏蔽（全局）
    assert not am[N + 1:].any()                      # patches 行无屏蔽（全双向）
    for i in range(1, N + 1):
        assert not am[i, :i + 1].any()               # special i 可见 cls + specials≤i
        assert am[i, i + 1:N + 1].all()              # 屏蔽后面的 special
        assert not am[i, N + 1:].any()               # special i 可见全部 patches
    # 关掉 causal 时掩码全 False（回退全双向；掩码 forward 内现算，先跑一次）
    m_free = SRPhase1V2(FakeDino(dim=D, num_patches=N), num_patches=N, dim=D,
                        causal_specials=False, reencoder_depth=2,
                        decoder_steps=square_block_starts(N))
    _ = m_free.re_encoder(torch.randn(2, 2 * N + 1, D))
    assert not m_free.re_encoder.attn_mask.any()
    print(f"[ok] ReEncoder block mask: causal_specials=True 结构正确, "
          f"causal_specials=False 全开放回退")

    # ── 3. OutputQueryDecoder: 分块采样计划 + 分块掩码 + 累加结果损失 ──
    T_steps = model.decoder.steps
    assert T_steps == square_block_starts(N), T_steps       # N=16 → [1,4,9,16]
    assert len(model.decoder.stack.layers) == 2, "默认 decoder_depth=2"
    am = model.decoder.attn_mask                             # (|T|·N, N+1) float
    assert am is not None and am.shape == (len(T_steps) * N, N + 1), am.shape
    # 分块掩码: 第一个步可见 0..hi（自己块 + 前面所有元素, 含位置 0 z_cls）;
    # 其余步只允许自己的块 [⌊√t⌋², min((⌊√t⌋+1)²-1, N)], 位置 0 屏蔽
    for ti, t in enumerate(T_steps):
        k = math.isqrt(t)
        hi = min((k + 1) * (k + 1) - 1, N)
        row = am[ti * N]
        if ti == 0:
            assert (row[:hi + 1] == 0).all()                 # 前面所有 + 自己块
            assert (row[hi + 1:] == float("-inf")).all()
        else:
            lo = max(k * k, 1)
            assert row[0] == float("-inf")                   # 其余步 z_cls 屏蔽
            assert (row[:lo] == float("-inf")).all()         # 块前屏蔽
            assert (row[lo:hi + 1] == 0).all()               # 块内允许
            assert (row[hi + 1:] == float("-inf")).all()     # 块后屏蔽
    Y = model.decoder.last_Y                                 # (B,|T|,N,D) 特征
    assert Y.shape == (2, len(T_steps), N, D)
    # 与模型 decode 相同的梯度解耦构造（2026-09-03）: carry 无梯度 + 自己的预测
    Y_cum = torch.cat([torch.zeros_like(Y[:, :1]),
                       Y.cumsum(dim=1)[:, :-1]], dim=1).detach() + Y
    Y_pix = model.pixel_head(Y_cum)                          # (B,|T|,N,588) 累加像素
    assert Y_pix.shape == (2, len(T_steps), N, PATCH_PX)
    # 数值不变性: 与朴素 cumsum 只差 float32 求和顺序噪声（<1e-4 级）;
    # 最终 F_hat 仍是全步之和 Σ_t Y_t
    assert torch.allclose(Y_cum, Y.cumsum(dim=1), atol=1e-4, rtol=1e-4), \
        "梯度解耦构造数值上应≈原 cumsum（float32 求和顺序噪声内）"
    assert torch.isclose(out["F_hat"], Y_pix[:, -1]).all(), "F_hat 应为各步像素累加和"
    per = F.l1_loss(Y_pix, target.unsqueeze(1).expand_as(Y_pix),
                    reduction="none").mean(dim=(0, 2, 3))    # (|T|,) 每步累加结果
    assert torch.isclose(out["loss"], per.mean()), "loss 应为累加结果平权 L1"
    # 计划自动适配任意 N（可扩展性）: 12→3 块, 256→16 块, 512→22 块
    assert square_block_starts(12) == [1, 4, 9], square_block_starts(12)
    assert len(square_block_starts(256)) == 16
    assert len(square_block_starts(512)) == 22
    # 用户给定示例核对（N=12）: 第一个步 t=1 可见前面所有元素（含位置 0
    # z_cls）→ [T,T,T,T,F,F,F,F,F,F,F,F,F] (0..3); 步 2 (t=4) 只看块
    # [4..8] → [F,F,F,F,T,T,T,T,T,F,F,F,F]
    m12 = build_block_mask(12, [1, 4, 9])
    assert m12.shape == (3 * 12, 13), m12.shape
    r1, r2 = m12[0], m12[12]                                 # 步 1 / 步 2 首查询行
    assert (r1[0:4] == 0).all() and (r1[4:] == float("-inf")).all()
    assert (r2[4:9] == 0).all() and (r2[:4] == float("-inf")).all() \
        and (r2[9:] == float("-inf")).all()
    # 可选挑选（skip_steps/max_steps, 默认 None = 全部分块）: [4:9] 切片取中段
    d_slice = OutputQueryDecoder(num_patches=256, dim=D, skip_steps=4, max_steps=9)
    assert d_slice.steps == square_block_starts(256)[4:9], d_slice.steps
    assert d_slice.steps == [25, 36, 49, 64, 81], d_slice.steps
    # 切片后: 第一个步 t=25 可见 0..35（前面所有元素 + 自己块 25..35）;
    # 其余步只允许自己的块（t=81 → [81..99]）
    sm = d_slice.attn_mask if hasattr(d_slice, "attn_mask") else None
    d_slice(torch.randn(1, 1, D), torch.randn(1, 256, D))
    sm = d_slice.attn_mask                                  # (5·256, 257)
    assert sm.shape == (5 * 256, 257), sm.shape
    r_s1, r_s5 = sm[0], sm[4 * 256]                         # t=25 (第一个) / t=81 首查询行
    assert (r_s1[0:36] == 0).all() and (r_s1[36:] == float("-inf")).all()
    assert (r_s5[81:100] == 0).all() and (r_s5[:81] == float("-inf")).all() \
        and (r_s5[100:] == float("-inf")).all()
    # 默认 None = 全部分块（等价不切片）; 验收: 第一个步 t=1 能看到 [0,1]
    d_full = OutputQueryDecoder(num_patches=256, dim=D)
    assert d_full.steps == square_block_starts(256), "默认 None 应为全部分块"
    assert d_full.steps[0] == 1, "默认计划第一个步应为 t=1"
    d_full(torch.randn(1, 1, D), torch.randn(1, 256, D))
    r_def1 = d_full.attn_mask[0]                            # 默认计划第一个步 (t=1)
    assert (r_def1[0:4] == 0).all(), "第一个步应可见位置 0 (z_cls)"
    assert (r_def1[1] == 0).all(), "第一个步应可见位置 1"
    print(f"[ok] OutputQueryDecoder: {len(model.decoder.stack.layers)} 层 "
          f"TransformerDecoder, 分块采样 {len(T_steps)} 步 {T_steps} "
          f"+ 分块掩码(示例核对, 第一个步可见[0,1]) + 可选挑选 [4:9]={d_slice.steps} "
          f"+ 累加结果像素损失正确")

    # ── 4. 梯度流向（整模型可训, 含 PixelHead; 首层自注意力 + 末层交叉/FFN）──
    out["loss"].backward()
    for name, p in [("re_encoder.0", model.re_encoder.layers[0].linear1.weight),
                    ("decoder.stack.layers[0].self_attn.in_proj_weight",
                     model.decoder.stack.layers[0].self_attn.in_proj_weight),
                    ("decoder.stack.layers[-1].multihead_attn.in_proj_weight",
                     model.decoder.stack.layers[-1].multihead_attn.in_proj_weight),
                    ("decoder.stack.layers[-1].linear1.weight",
                     model.decoder.stack.layers[-1].linear1.weight),
                    ("decoder.query_base", model.decoder.query_base),
                    ("special_bank.pos", model.special_bank.pos),
                    ("pixel_head.proj", model.pixel_head.proj.weight)]:
        g = p.grad
        assert g is not None and g.abs().sum() > 0, f"{name} 收不到梯度"
    print(f"[ok] 梯度: ReEncoder/OutputQueryDecoder({len(model.decoder.stack.layers)}×"
          f"TransformerDecoderLayer,query_base)/"
          f"SpecialTokenBank/PixelHead 全部可训 "
          f"(|grad|={model.re_encoder.layers[0].linear1.weight.grad.abs().sum().item():.4f})")

    # ── 4b. 梯度按步解耦（2026-09-03 用户需求）──
    # 数值上 Y_cum == cumsum(Y)（F_hat/损失语义不变）; 但 dL/dY_t 只来自
    # 第 t 步自己的损失项——每步平权 1 份梯度, 不再有"t=0 收 |T| 份"的
    # 三角失衡。校验: (a) 数值恒等; (b) 全量梯度在位置 n == 仅第 n 步损失
    # 的梯度(÷|T|); (c) 第 n 步损失对其他位置 Y_{m≠n} 无梯度（结构上断开）。
    out_b = model(x)
    Y_b = model.decoder.last_Y                              # (B,|T|,N,D)
    Tb = Y_b.shape[1]
    Y_cum_b = torch.cat([torch.zeros_like(Y_b[:, :1]),
                         Y_b.cumsum(dim=1)[:, :-1]], dim=1).detach() + Y_b
    assert torch.allclose(Y_cum_b, Y_b.cumsum(dim=1), atol=1e-4, rtol=1e-4), \
        "梯度解耦后 Y_cum 数值上应≈原 cumsum（float32 求和顺序噪声内）"
    assert torch.isclose(out_b["F_hat"], out_b["Y_pix"][:, -1]).all()
    per_b = F.l1_loss(out_b["Y_pix"],
                      out_b["target_pix"].unsqueeze(1).expand_as(out_b["Y_pix"]),
                      reduction="none").mean(dim=(0, 2, 3))     # (|T|,) 每步损失
    g_total = torch.autograd.grad(out_b["loss"], Y_b, retain_graph=True)[0]
    for n in range(Tb):
        g_n = torch.autograd.grad(per_b[n] / Tb, Y_b,
                                  retain_graph=True)[0]          # 仅第 n 步损失
        others = [m for m in range(Tb) if m != n]
        assert torch.allclose(g_n[:, others],
                              torch.zeros_like(g_n[:, others]), atol=1e-5), \
            f"step {n} 的损失不应给其他步的预测梯度"
        assert torch.allclose(g_total[:, n], g_n[:, n], atol=1e-5), \
            f"step {n} 的 Y 梯度应只来自自己那一步的损失（平权 1/{Tb}）"
    print(f"[ok] 梯度按步解耦: Y_cum 数值==cumsum(F_hat 不变), "
          f"每步 Y_t 恰收 1/{Tb} 份梯度（不再三角失衡）")

    # ── 5. 推理: 同一 forward（eval + no_grad）──
    model.eval()
    with torch.no_grad():
        out_e = model(x)
    assert out_e["F_hat"].shape == (2, N, PATCH_PX)
    print(f"[ok] eval 同路径: loss={out_e['loss'].item():.4f}")

    # ── 6. register 模式: specials 合并进 DINO（修 F1/F2, 2026-08-28）──
    dino_r = FakeDino(dim=D, num_patches=N)
    m_reg = SRPhase1V2(dino_r, num_patches=N, dim=D, register_specials=True,
                       decoder_steps=square_block_starts(N))
    assert m_reg.re_encoder is None, "register 模式不应有 ReEncoder"
    out_r = m_reg(x)
    assert out_r["F_hat"].shape == (2, N, PATCH_PX)
    assert out_r["Y_pix"].shape == (2, len(m_reg.decoder.steps), N, PATCH_PX)
    z_cls, z_s = m_reg.encode(x)
    assert z_cls.shape == (2, 1, D) and z_s.shape == (2, N, D)
    # 梯度: DINO 嵌入(conv+PE+cls) / DINO 层 / special_bank / decoder / pixel_head
    out_r["loss"].backward()
    for name, p in [("dino.embeddings.patch_embeddings.weight",
                     dino_r.embeddings.patch_embeddings.weight),
                    ("dino.embeddings.position_embeddings",
                     dino_r.embeddings.position_embeddings),
                    ("dino.encoder.layer.0.mlp.fc1.weight",
                     dino_r.encoder.layer[0].mlp.fc1.weight),
                    ("special_bank.token", m_reg.special_bank.token),
                    ("decoder.stack.layers[-1].multihead_attn.in_proj_weight",
                     m_reg.decoder.stack.layers[-1].multihead_attn.in_proj_weight),
                    ("decoder.query_base", m_reg.decoder.query_base),
                    ("pixel_head.proj", m_reg.pixel_head.proj.weight)]:
        g = p.grad
        assert g is not None and g.abs().sum() > 0, f"{name} 收不到梯度"
    m_reg.eval()
    with torch.no_grad():
        out_re = m_reg(x)
    assert out_re["F_hat"].shape == (2, N, PATCH_PX)
    print(f"[ok] register_specials: specials 进 DINO 序列(2N+1 token, 全双向) "
          f"无 ReEncoder; 形状/梯度(eval 同路径)正确")

    print("\nALL CHECKS PASSED")
