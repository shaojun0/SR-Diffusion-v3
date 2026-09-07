"""
SR-Diffusion Phase 1 v2 — 训练脚手架（register 式, 无 ReEncoder）
=================================================================

项目目标（权威版见 doc/2026-08-28/GOAL_compression_for_nlp.md）:
    用像素重建当代理任务, 训练编码器（DINOv2 + register specials）的
    "token 压缩 + 联想"能力; 训练完成后冻结编码器接入 Qwen 做 NLP 解码
    （验收标准 = Phase 2 文字生成质量）。像素重建 = 信息保持的直接探针
    （能还原像素 ⇒ z 携带整图信息 ⇒ 语义信息必然在）。纹理级高频按文献
    [3] 属"死信息"不追; 压缩率 K 是练联想的真正杠杆——K 与 N 解耦: N
    只管输入 patch 数与解码查询行数, K（register/specials 数）由解码器
    最终采样步集自动推导（K≤N; K=N 即历史全量行为, K<N 即真压缩）。

沿革: v2 曾含 ReEncoder 路径与"可学习前缀预算"机制, 均已按 YAGNI 整块
    移除（git 历史可找回）; register 式（2026-08-28 起）为唯一路径。

架构（register 式, 无选择、无预算）:
    specials 作为额外 token 直接拼进 DINOv2 输入序列
    [cls; specials(K); patches(N)] (1+K+N token, K 的推导见下), 由 DINO
    24 层直接算出 z_s（register token 式, Darcet et al.）——深层网络做
    "内容路由", 修 F1（special 无内容输入）与 F2（z_s 冗余全局摘要）。
    DINO 内全双向注意力（HF 无 token 级 mask API）; z_s[k] 依赖全部 patch
    （含 j>k）, "前缀稳定性"不成立——渐进语义由解码器分块掩码提供: 读侧
    memory_mask（每步只见自己的块）+ 查询侧 tgt_mask 块因果（步 t 只见
    步 ≤ t 的查询行, 见 OutputQueryDecoder / build_causal_query_mask）。
    无 ReEncoder（省 51.6M 参数）。
    register 数 K（num_specials）= 解码器实际读取的 z_s 范围: 默认由
    "最终生效采样步集"自动推导 K = min( max_{t∈steps}((⌊√t⌋+1)²−1), N )
    （公式/动机/示例见 derive_num_specials 与 doc/2026-09-02/
    DESIGN_v2_num_specials_from_max_steps.md）——消除"花瓶 register":
    编码器生成、解码器从不读、却经 24 层全双向注意力的前向/反向耦合
    参与训练动力学并干扰读窗口的多余位置。全量默认（无 decoder_steps /
    skip/max 切片）时 steps = square_block_starts(N) ⇒ K=N（向后兼容）;
    例: N=576 切片 [4:9] ⇒ steps=[25,36,49,64,81] ⇒ K=99; steps=[64] ⇒ K=80。
    OutputQueryDecoder: [z_cls; z_s] 为时序序列 A (S=K+1); 在采样时刻
                    T_sub = steps（默认 square_block_starts, 自动适配任意
                    上界; SRPhase1V2 里切片先于 K 推导, 上界 = N, 见
                    select_steps）上输出 (N,D) = 全部 patch 的预测（输出
                    查询注意力, 查询基行 k ↔ patch k, 行数 = N 不变）;
                    分块掩码(memory_mask): 每步只 attend 自己的 z_s 块
                    （掩码列数 = K+1）; 查询自注意力块因果（tgt_mask:
                    步 t 只 attend 步 ≤ t 的查询行, 防后步查询内容泄露进
                    前步输出）→ F_hat = Σ_t Y_t（第 n 步结果 = 前 n 步之和）
    L = mean_n L1(cumsum_n(Y_t), patch)   ← 每步累积结果平权全覆盖损失
        （梯度按步解耦: carry 整体 detach + 自己的预测——每步恰收 1 份梯度,
        避免"t=0 收 |T| 份梯度"的三角失衡; 见 SRPhase1V2.decode）

踩坑记录（重要）:
    torch 2.x 的 bool 注意力掩码约定是 True=屏蔽（nn.TransformerEncoder
    Layer / MultiheadAttention / TransformerDecoderLayer）, 与直觉相反;
    F.scaled_dot_product_attention 的 bool 掩码实测 True=允许（torch 2.8
    本机验证）。掩码统一用加法浮点(-inf=屏蔽)规避歧义。

用法
----
    model = SRPhase1V2(dinov2=dinov2_model, num_patches=256, dim=768)
    out = model(pixel_values)                 # {"loss","recon","F_hat","Y_pix","target_pix"}
    loss = out["loss"]; loss.backward()
    model.eval()                              # 推理同路径

    自检: python model_v2.py

参考资料:
    [1] Hawthorne et al. (ICML 2022), Perceiver AR——输出查询 + 因果掩码解码器
        https://mlanthology.org/icml/2022/hawthorne2022icml-generalpurpose/
    [2] Li et al. (ICML 2023), BLIP-2——可学习查询基 / Q-Former 查询桥接
        https://arxiv.org/abs/2301.12597
    [3] Fan et al. (2026)——视觉 token 稀疏性/冗余分析（"死信息"依据）
        https://arxiv.org/abs/2603.00510
    [4] Apedo et al. (2026), SVD-Prune——视觉 token 剪枝（与已移除的预算机制相关）
        https://arxiv.org/abs/2604.11530
    [5] Vahdat & Kautz (NeurIPS 2020), NVAE——隐变量容量取舍对照
        https://proceedings.neurips.cc/paper/2020/hash/e3b21256183cf7c2c7a66be163579d37-Abstract.html
    [6] Gao & Shou (2025), D-AR——顺序扩散 tokenizer;"token 顺序定义渐进细化"参考
        https://arxiv.org/abs/2505.23660
    [7] Liang et al. (ICLR 2022), EViT——注意力分数排序剪枝 + 不重要 token 融合
        （与 [4] 同域; 预算机制回归时的简单替代）
        https://arxiv.org/abs/2202.07815
        https://github.com/youweiliang/evit
    [8] Darcet et al. (ICLR 2024), Vision Transformers Need Registers——register
        token 式出处（架构段所引的 Darcet et al. 即此文）
        https://arxiv.org/abs/2309.16588
    [9] Jaegle et al. (ICML 2022), Perceiver IO——输出查询定义输出结构的通用化;
        query_base 行 k↔patch k 的直接出处
        https://arxiv.org/abs/2107.14795
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Sequence


# ═══ SpecialTokenBank — 特殊 token 池（输入相同, 仅位置编码不同）═══

class SpecialTokenBank(nn.Module):
    """每个 special/register 位置一个特殊 token：共享可学习向量 + 逐位置可学习位置编码。

    token (1,1,D) 对所有位置/所有图像相同; pos (1,K,D) 提供位置区分。
    特殊 token 位置的表示 z_s 是 decoder 输入; 图像 patch 编码不进 decoder
    （只作为编码器输入参与聚合）。K = register/specials 数（= num_specials,
    由解码器最终采样步集推导, 见 derive_num_specials; 历史参数名
    num_patches 已改名为 num_tokens——K 与 N 解耦后两者不再恒等）。
    """

    def __init__(self, num_tokens: int, dim: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)          # 共享向量
        self.pos = nn.Parameter(torch.randn(1, num_tokens, dim) * 0.02)  # 位置编码

    def forward(self, B: int, device: torch.device) -> Tensor:
        return self.token.expand(B, self.num_tokens, -1) + self.pos      # (B,K,D)


# ═══ build_prefix_mask — 块状前缀掩码（仅旧诊断脚本 gen_mask_gif.py 引用）═══

def build_prefix_mask(seq_len: int, z_start: int, z_end: int,
                      device: torch.device = None) -> Tensor:
    """块状前缀注意力掩码（torch bool, True=屏蔽）。

    布局: [z 区域 (z_start..z_end-1) | 尾部 (z_end..seq_len-1)]
        · z 行 i: 屏蔽 z 列 (i+1..z_end-1)（因果链）, 可看尾部（全局）
        · 尾部行: 全开放（全局）
    """
    m = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    for i in range(z_start, z_end):
        m[i, i + 1:z_end] = True          # 屏蔽 z 内部"后面的"
    return m


# ═══ square_block_starts — 分块采样计划（块起点 = 平方数, 自动适配任意 N）═══

def square_block_starts(num_patches: int) -> list:
    """decoder 采样时刻 T_sub = {k² | 1 ≤ k² ≤ N}（块起点 = 平方数）。

    块 k = [k², min((k+1)²-1, N)]; 步数 = ⌊√N⌋, 自动适配任意 N
    （N=256→16 步, N=576→24 步）。例: square_block_starts(12) → [1,4,9]。
    本函数已通用: 入参语义 = "分块计划上界", 不要求与 patch 数 N 恒等
    （见 derive_num_specials / select_steps 的调用方说明）。
    """
    K = math.isqrt(num_patches)
    return [k * k for k in range(1, K + 1)]


# ═══ derive_num_specials — K 由"最终生效采样步集"自动推导（无花瓶）═══

def derive_num_specials(num_patches: int, steps: Sequence[int]) -> int:
    """K = min( max_{t∈steps} ((⌊√t⌋+1)²−1), num_patches ); steps 为空 → num_patches。

    动机（花瓶 register 问题）: register 式里编码器生成 N 个 register, 但
    当训练只读 steps 覆盖的部分 register 时（如 slice [4:9] → steps=
    [25,36,49,64,81], 块 5..9 最大读到位置 99）, 位置 100..N 的 register
    从不被解码器读——它们不是摆设: 24 层全双向注意力的前向/反向耦合使
    花瓶收到大梯度（读窗口 pos 梯度 |Σ|=6.35 vs 花瓶区 9.81）且逐维扰动
    花瓶输入会让读窗口输出变化 ~68%, 干扰训练动力学。修复 = K 与 N 解耦:
    编码器 register 数 = 解码器实际读的范围, 不存在花瓶。K<N 时同时实现
    真压缩（N 只管输入 patch 数与解码查询行数, K 只管键数/register 数）。

    公式: 步 t 所在的平方块 k=⌊√t⌋ 的块终点 = (k+1)²−1;
        K = min( max_{t∈steps} ((⌊√t⌋+1)²−1), num_patches )
    全量默认（无 decoder_steps、无 skip/max 切片）时 steps =
    square_block_starts(num_patches), K 计算结果 = N（保持现状, 向后兼容）。
    例 1: N=576, slice [4:9] → steps=[25,36,49,64,81], max t=81 →
        (9+1)²−1=99 → K=99（读 1..99, 全部被覆盖, 无花瓶）。
    例 2: 显式 steps=[64] → (⌊√64⌋+1)²−1 = 80 → K=80。
    数学性质: K ≥ max(steps); 对 square_block_starts 的连续切片, register
    1..K 每个位置至少被某个步允许（首步前缀规则覆盖开头, 各块连续铺满）。
    显式非平方 steps 不保证全覆盖（文档注明即可）。

    注意: K 推导只依赖步集本身, 与 skip_steps 起点无关。
    """
    if not steps:
        return num_patches
    max_t = max(int(t) for t in steps)
    return min((math.isqrt(max_t) + 1) ** 2 - 1, num_patches)


# ═══ select_steps — 最终生效采样步集（SRPhase1V2 / OutputQueryDecoder 共用）═══

def select_steps(num_patches: int, decoder_steps: Optional[Sequence[int]] = None,
                 skip_steps: Optional[int] = None,
                 max_steps: Optional[int] = None) -> list:
    """最终生效采样步集: 显式 decoder_steps 原样（去重排序, 不切片）; None 时
    默认计划 square_block_starts(num_patches) 再按 skip_steps/max_steps 做
    Python 切片（如 [4:9] 取中段, 每个保留步仍按自身步值归属分块）。

    入参 num_patches 语义 = "默认分块计划上界": SRPhase1V2 在 K 未知时传
    N=num_patches（切片先于 K 推导, 与 train/infer CLI 的 slice 索引一致）;
    OutputQueryDecoder 独立使用时传自己的 num_specials（K, 已推导完毕）。
    两处对同一配置得到同一步集（SRPhase1V2 把选好的 steps 显式传给
    decoder, decoder 只去重排序不再切片）。

    校验: 步集非空; skip/max 切片索引合法; 步值 ∈ [0, num_patches]。
    """
    explicit = decoder_steps is not None
    if explicit:
        base = sorted(set(int(s) for s in decoder_steps))
    else:
        base = square_block_starts(num_patches)
    assert base and all(0 <= s <= num_patches for s in base), \
        f"steps 越界: {base} (上界={num_patches})"
    if explicit:
        return base
    lo = 0 if skip_steps is None else int(skip_steps)
    hi = len(base) if max_steps is None else int(max_steps)
    assert 0 <= lo < hi <= len(base), \
        f"skip_steps/max_steps 越界: skip={lo} max={hi} " \
        f"(计划共 {len(base)} 步 {base})"
    out = base[lo:hi]
    assert out, f"切片后无采样步: base[{lo}:{hi}] of {base} 为空"
    return out


# ═══ build_block_mask — 分块注意力掩码（每步只见自己的 z_s 块）═══

def build_block_mask(num_tokens: int, steps: Sequence[int],
                     num_queries: Optional[int] = None,
                     device: torch.device = None) -> Tensor:
    """分块注意力掩码（torch float, -inf=屏蔽, 0=允许）：每步只见自己的 z_s 块。

    块号 k = ⌊√t⌋, 块 k = [k², min((k+1)²-1, num_tokens)]（平方数边界铺满
    1..num_tokens; num_tokens = register/specials 数 K）。**第一个步特殊**:
    能看到自己块之前的所有元素（含位置 0 = z_cls）——避免
    skip_steps/max_steps 切片丢弃的前段信息完全不被使用; 其余步只允许
    自己的块, 位置 0 屏蔽。

    返回 (|T|·Q, num_tokens+1)：Q = 每采样步的查询行数（默认 = num_tokens,
    即历史 K==N 全量行为; K<N 时解码器须传 num_queries=N——每步 N 行 patch
    查询共享同一掩码行, 掩码高度按查询行数铺, 列数按 K+1 铺）。
    步值须 ∈ [0, num_tokens]（构造方已断言）; t=0 无块（仅显式传 0 时出现,
    默认计划不含 0）。
    """
    S = num_tokens + 1                      # z_cls + K 个 z_s
    Q = num_tokens if num_queries is None else int(num_queries)
    rows = []
    for i, t in enumerate(steps):
        k = math.isqrt(int(t))              # 块号 = 步值所在的平方块
        hi = min((k + 1) * (k + 1) - 1, num_tokens)
        row = torch.full((S,), float("-inf"), device=device)
        if i == 0:
            row[:hi + 1] = 0.0              # 第一个步: 前面所有元素（含 z_cls）到块终点
        else:
            lo = max(k * k, 1)              # 位置 0 (z_cls) 屏蔽
            if lo <= hi:
                row[lo:hi + 1] = 0.0        # 只允许自己块内的 z_s
        rows.append(row)
    return torch.stack(rows).repeat_interleave(Q, dim=0)


# ═══ build_causal_query_mask — 查询自注意力块因果掩码 ═══

def build_causal_query_mask(num_steps: int, num_queries: int,
                            device: torch.device = None) -> Tensor:
    """查询自注意力“块因果”掩码（torch float, -inf=屏蔽, 0=允许）。

    查询行序 = (t,k) 展平（index = t*num_queries + k, 每采样步 N 行 patch
    查询连排）; **块下三角**: 步 t 的行可 attend 步 ≤ t 的所有行——同步内
    全双向（patch 间可交换信息）+ 之前步, 未来步屏蔽。目的: 后步查询内容
    （z_s 采样种子）不再经查询自注意力影响前步输出——渐进语义由
    memory_mask（读侧）+ tgt_mask（查询侧）双重保证。返回 (L,L) 方阵,
    L = num_steps·num_queries。
    """
    T, Q = int(num_steps), int(num_queries)
    L = T * Q
    m = torch.full((L, L), float("-inf"), device=device)
    for t in range(T):
        m[t * Q:(t + 1) * Q, :(t + 1) * Q] = 0.0
    return m


# ═══ PixelHead — 特征 → 像素 patch 解码头 ═══

class PixelHead(nn.Module):
    """每 patch 特征 (B,N,D) → 像素 patch (B,N,14*14*3)。

    2 层 MLP（dim→hidden→588）: 回应"解码器参数量不够欠拟合"的担忧
    （v2 是单层 Linear, 0.6M; 这里是 ~2.1M@dim=1024,hidden=2048, 且带
    非线性）。输出不加激活: 像素按 DINO_MEAN/STD 归一化(范围≈[-2,2]),
    L1 直接监督归一化空间, 评估时再反归一化。
    """

    def __init__(self, dim: int, patch_px: int = 14 * 14 * 3, hidden: int = 2048):
        super().__init__()
        self.patch_px = patch_px
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, patch_px))

    def forward(self, feat: Tensor) -> Tensor:
        """feat: (..., N, D) → (..., N, patch_px)"""
        return self.net(feat)


# ═══ OutputQueryDecoder — 输出查询注意力解码器（采样时刻上输出全部 patch）═══

class OutputQueryDecoder(nn.Module):
    """把 [z_cls; z_s] 当时序序列 A=(S,D), S=num_specials+1=K+1, 在采样时刻
    T_sub 上每步输出 (N,D) = 全部 patch 的预测, 沿时刻累加得 F_hat (B,N,D)。

    机制: A = [z_cls; z_s] + pos_embed; 查询 = A_t + 查询基 E（行 k↔patch k,
    BLIP-2 同源）→ 标准 nn.TransformerDecoder 堆叠（自注意力 + 交叉注意力
    (memory=A) + FFN）→ reshape (B,|T|,N,D), 返回 Y。
    · num_specials（K）= z_s 长度; None = num_patches（历史 K==N 行为）。
      num_patches（N）只决定查询基行数 = 输出 patch 预测数, 两者解耦后
      不再恒等——pos_embed 尺寸 (1,K+1,D), query_base 行数 = N。
    · 分块掩码走 memory_mask（加法浮点, -inf=屏蔽; 见 build_block_mask）,
      列数 = K+1（每步只见自己的 z_s 块）
    · 查询自注意力**块因果**（见 build_causal_query_mask, 加法浮点,
      -inf=屏蔽）: 查询行序 = (t,k) 展平, 步 t 的行可 attend 步 ≤ t 的所有
      行（同步内全双向）, 未来步屏蔽——后步查询内容不再经查询自注意力
      影响前步输出, 渐进语义由 memory_mask（读侧）+ tgt_mask（查询侧）
      双重保证。
    · 结果沿采样步累加; **梯度按步解耦**（carry detach + 自己的预测, 见
      SRPhase1V2.decode）——每步 Y_t 只从自己那一步的损失收 1 份梯度
    · 时刻采样（显存优化, 默认开启）: 查询只对 T_sub 构造——采样计划 =
      select_steps(num_specials, steps, skip_steps, max_steps): 默认计划 =
      square_block_starts(K); skip_steps/max_steps 可选切片挑选; steps= 可
      自定义（原样不切片）。未采样时刻不参与损失, 也不出现在 F_hat 累加里。
      注意: SRPhase1V2 会先按 N 计划选好步再以显式 steps 传入（两处结果
      一致, 见 select_steps 的 docstring）
    · 覆盖语义: 每个采样时刻都预测全部 N 个 patch, 其累加结果都被监督
      还原全部 patch
    """

    def __init__(self, dim: int = 768, num_patches: int = 256,
                 mlp_ratio: float = 4.0,
                 steps: Optional[Sequence[int]] = None, heads: int = 8,
                 depth: int = 2, skip_steps: Optional[int] = None,
                 max_steps: Optional[int] = None,
                 num_specials: Optional[int] = None):
        super().__init__()
        self.num_patches = num_patches                 # 查询基行数 = N（输出 N 个 patch 预测, 不变）
        self.num_specials = num_patches if num_specials is None else int(num_specials)
        # 采样计划: 与 SRPhase1V2 共用 select_steps; 独立使用时默认计划
        # 上界 = K（num_specials）; 显式 steps 原样（调用方负责, 不切片）
        self.steps = select_steps(self.num_specials, steps,
                                  skip_steps, max_steps)
        S = self.num_specials + 1                                # z_cls + K z_s
        self.query_base = nn.Parameter(torch.randn(num_patches, dim) * 0.02)  # 行 k↔patch k
        # 标准解码器堆叠: nn.TransformerDecoder 内部按 num_layers 深拷贝同一
        # decoder_layer 并顺序执行（含逐层传掩码）, 无需手写循环。每层 =
        # 自注意力 + 交叉注意力(memory=A) + FFN + 残差, 各层共享同一 memory
        self.stack = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True),
            num_layers=depth,
        )
        self.pos_embed = nn.Parameter(torch.randn(1, S, dim) * 0.02)

    def forward(self, z_cls: Tensor, z_s: Tensor) -> Tensor:
        B, N, D = z_s.shape[0], self.num_patches, z_s.shape[-1]
        assert z_cls.shape[1] == 1, f"z_cls 应为 1 列, got {z_cls.shape[1]}"
        assert z_s.shape[1] == self.num_specials, \
            f"z_s 列数 {z_s.shape[1]} != num_specials(K)={self.num_specials}"
        A = torch.cat([z_cls, z_s], dim=1) + self.pos_embed      # (B,S,D)
        A_t = A[:, self.steps]                                   # (B,|T|,D) 采样时刻
        Y = (A_t.unsqueeze(2) + self.query_base) \
            .reshape(B, len(self.steps) * N, D)                  # 展平 (t,k), t∈T_sub
        # 分块掩码: 每步只见自己的 z_s 块（-inf=屏蔽, 0=允许）; 列数 =
        # K+1 = num_specials+1, 每步 N 行查询共享同一掩码行
        mask = build_block_mask(self.num_specials, self.steps,
                                num_queries=N, device=A.device)  # (|T|·N, K+1)
        self.attn_mask = mask                                    # 供自检
        # 查询自注意力块因果: 行序 = (t,k) 展平, 步 t 的行只允许步 ≤ t 的
        # 列（同步内全双向）——后步查询内容不泄露进前步输出
        tgt_mask = build_causal_query_mask(len(self.steps), N,
                                                device=A.device)  # (L,L)
        self.tgt_mask = tgt_mask
        Y = self.stack(Y, A, memory_mask=mask,
                       tgt_mask=tgt_mask)                   # (B,|T|·N,D)
        Y = Y.reshape(B, len(self.steps), N, D)                  # (B,|T|,N,D)
        self.last_Y = Y                                          # 采样步全部 patch 预测
        return Y


# ═══ SRPhase1V2 — 主模型（register 式; 无预算; K=num_specials 由步集推导）═══

class SRPhase1V2(nn.Module):
    """register 式主模型。num_specials（K）与 num_patches（N）解耦:
    N = 输入 patch 数 = 解码查询行数 = 输出 patch 预测数（不变）; K =
    register/specials 数 = 解码器实际读取的 z_s 范围。默认（num_specials
    = None）K 由"最终生效采样步集"自动推导: 先按 N 计划选出步集
    steps_selected（显式 decoder_steps 原样; 否则 square_block_starts(N)
    再 skip_steps/max_steps 切片）, 再 K = derive_num_specials(N,
    steps_selected)——与解码器独立用法同一步集（构造时把 steps_selected
    以显式 steps 传给 OutputQueryDecoder, 由后者去重排序, 不再二次切片）。
    显式 num_specials=K 时断言 1≤K≤N 且 max(steps_selected)≤K（K 过小 =
    采样步超出 z_s 范围, 清晰报错; 复现旧 checkpoint 时须显式传回旧 K）。

    K=N（默认全量）序列 = [cls; specials(N); patches(N)] = 2N+1 token, 与
    历史一致; K<N 时序列 = 1+K+N token, register 1..K 全被解码器读（无
    "花瓶 register", 见 derive_num_specials / DESIGN doc）。
    解码器查询自注意力为块因果: 查询行序 = (t,k) 展平, 步 t 只 attend 步
    ≤ t 的查询行（同步内全双向）, 后步查询内容不泄露进前步输出——渐进
    语义由 memory_mask（读侧）+ tgt_mask（查询侧）双重保证（见
    OutputQueryDecoder / build_causal_query_mask）。
    """

    def __init__(
        self,
        dinov2: nn.Module,
        num_patches: int = 256,
        dim: int = 768,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        decoder_steps: Optional[Sequence[int]] = None,
        patch_px: int = 14 * 14 * 3,
        decoder_depth: int = 2,
        skip_steps: Optional[int] = None,
        max_steps: Optional[int] = None,
        num_specials: Optional[int] = None,
    ):
        super().__init__()
        self.dinov2 = dinov2
        self.num_patches = num_patches
        self.dim = dim
        self.patch_px = patch_px

        # 最终生效采样步集: 先按 N 计划选步（K 尚未推导; 与 train/infer
        # CLI 的 slice 索引口径一致, 见 select_steps）; K = 显式 num_specials
        # 或由步集自动推导（公式见 derive_num_specials）
        steps_selected = select_steps(num_patches, decoder_steps,
                                      skip_steps, max_steps)
        if num_specials is None:
            K = derive_num_specials(num_patches, steps_selected)
        else:
            K = int(num_specials)
        assert 1 <= K <= num_patches, \
            f"num_specials 越界: K={K} (须 1 ≤ K ≤ N={num_patches})"
        max_t = max(steps_selected)
        assert max_t <= K, \
            f"num_specials(K)={K} 过小: 采样步最大 {max_t} > K " \
            f"(z_s 只有 {K} 个位置, 步 {max_t} 读不到); 请显式加大 K, 或 " \
            f"缩小 decoder_steps / skip_steps / max_steps"
        self.num_specials = K

        self.special_bank = SpecialTokenBank(num_tokens=K, dim=dim)
        self.decoder = OutputQueryDecoder(dim=dim, num_patches=num_patches,
                                          mlp_ratio=mlp_ratio, heads=heads,
                                          steps=steps_selected,
                                          depth=decoder_depth,
                                          num_specials=K)
        self.pixel_head = PixelHead(dim=dim, patch_px=patch_px)

    # ── encode: 输入 → 解码器输入 z_cls, z_s ──
    def encode(self, pixel_values: Tensor):
        """输入 → 解码器输入 (z_cls, z_s), 训练/推理同一路径。

        register 式: specials 作为额外 token 拼进 DINO 输入序列, 由 DINO
        24 层直接算出 z_s（register token 式, Darcet et al.）——深层网络做
        内容路由, 修 F1（special 无内容输入）与 F2（z_s 冗余全局摘要）。
        """
        return self._encode_register(pixel_values)

    def _encode_register(self, pixel_values: Tensor):
        """specials 直接进 DINO 输入序列 [cls; specials(K); patches(N)]
        (1+K+N token, K=self.num_specials)。

        HF Dinov2Model 无 token 级注意力 mask API, 故 DINO 内全双向注意力
        （重建任务无时序因果需求, register 惯例亦然）; 渐进语义由解码器
        分块掩码提供。embeddings() 复用 HF 的 cls/patch 嵌入 + 位置编码
        （含自动插值逻辑）; specials 用 SpecialTokenBank（共享 token +
        逐位置可学习 pos, 不带 DINO PE——与 patch 的位置关系完全学出）。
        """
        x = pixel_values                                # (B,3,H,W)
        emb = self.dinov2.embeddings(x)                 # (B,1+N,D) [cls; patches] + PE
        specials = self.special_bank(x.shape[0], x.device)   # (B,K,D) token+pos
        seq = torch.cat([emb[:, :1], specials, emb[:, 1:]], dim=1)   # (B,1+K+N,D)
        for layer in self.dinov2.encoder.layer:         # DINO 24 层（全双向）
            out = layer(seq)
            seq = out[0] if isinstance(out, (tuple, list)) else out
        seq = self.dinov2.layernorm(seq)                # (B,1+K+N,D)
        return seq[:, :1], seq[:, 1:1 + self.num_specials]

    # ── decode: 共享解码尾（Decoder → PixelHead → 像素损失）──
    def decode(self, z_cls: Tensor, z_s: Tensor, pixel_values: Tensor) -> dict:
        """解码器 + 像素头 + 平权全覆盖像素 L1。

        **累加语义**: Y_cum[:, n] = Σ_{t≤n} Y_t（特征空间累加再统一过
        PixelHead——PixelHead 含 bias, 先投影再累加会重复加 bias）。
        **梯度按步解耦**: 数值上仍 Y_cum = cumsum(Y)（F_hat 不变）, 但
        carry = [0, cumsum(Y)[:-1]] 整体 detach + 自己的预测——每个 Y_t 只
        从自己那一步的损失收 1 份梯度（平权 1/|T|）, 不再从所有 ≥t 的
        累加位置收梯度（否则 t=0 有 |T| 份梯度动力, 学乱）。

        dict: {"loss", "recon", "F_hat"(像素 B,N,588), "Y_pix"(每采样步累加
        像素 B,|T|,N,588), "target_pix"(B,N,588)}——训练取 loss; 推理取
        F_hat / Y_pix / target_pix（全量 L1、渐进曲线、可视化同一路径）。
        """
        x = pixel_values
        B, C, H, W = x.shape
        N = self.num_patches
        Y = self.decoder(z_cls, z_s)                        # (B,|T|,N,D) 每步全部 patch
        # 像素目标: (B,3,H,W) 归一化像素 → (B,N,588) patch
        # 注意布局: DINO 的 patch 顺序是 row-major (先 y 后 x), 这里保持一致
        target_pix = x.reshape(B, C, H // 14, 14, W // 14, 14) \
                      .permute(0, 2, 4, 1, 3, 5) \
                      .reshape(B, N, C * 14 * 14)       # (B,N,588) 归一化像素
        # 特征空间累加 → 统一过 PixelHead（bias 只加一次）。
        # 梯度按步解耦: carry[n] = Σ_{t<n} Y_t（detach）; + Y[n] ⇒ 每步恰收
        # 1 份梯度（数值上 Y_cum[n] 仍 = Σ_{t≤n} Y_t, F_hat 不变）
        Y_cum = torch.cat([torch.zeros_like(Y[:, :1]),Y.cumsum(dim=1)[:, :-1]], dim=1).detach() + Y   # (B,|T|,N,D)
        Y_pix = self.pixel_head(Y_cum)                  # (B,|T|,N,588) 累加像素
        F_pix = Y_pix[:, -1]                            # (B,N,588) 最终 = Σ_t Y_t
        # 平权全覆盖损失: 每个采样步的累加结果都还原全部 patch 像素
        per_step = F.l1_loss(Y_pix, target_pix.unsqueeze(1).expand_as(Y_pix),
                             reduction="none").mean(dim=(0, 2, 3))   # (|T|,)
        loss = per_step.mean()                          # 平权
        recon = F.l1_loss(F_pix, target_pix)            # 集成重建（监控用, 归一化空间）
        return {"loss": loss, "recon": recon, "F_hat": F_pix,
                "Y_pix": Y_pix, "target_pix": target_pix}

    # ── forward（训练/推理同一路径, 无分支）──
    def forward(self, pixel_values: Tensor) -> dict:
        """pixel_values: (B,3,H,W) 归一化像素 → dict{loss, recon, F_hat,
        Y_pix, target_pix}

        监督**原始像素**而非 DINO patch 特征（特征目标退化: 工地图特征
        空间近常数, 学质心即低 L1 是假收敛）。H,W 须为 14 的倍数。
        损失: 每个采样步的**累加结果**都监督还原全部 patch —— 平权全覆盖
        L = mean_n L1(cumsum_n Y_pix, target_pix)。F_hat = Σ_t Y_t → 像素;
        recon 仅作监控。
        """
        x = pixel_values                                # (B,3,H,W)
        B, C, H, W = x.shape
        N = self.num_patches
        assert W % 14 == 0 and H % 14 == 0, "输入须为 14 的倍数"
        assert (W // 14) * (H // 14) == N, \
            f"输入 {W}x{H} 产生 {(W//14)*(H//14)} patches, 但模型 num_patches={N}"
        z_cls, z_s = self.encode(x)
        return self.decode(z_cls, z_s, x)


# ═══ 自检（python model_v2.py）═══
#   1. 形状正确性（register 式全路径）
#   2. OutputQueryDecoder 分块采样计划 + 分块掩码结构
#   3. 梯度流向（整模型可训）+ 梯度按步解耦（Y_cum 数值==cumsum, 每步只收
#      自己那一步的梯度）
#   4. eval 同路径

if __name__ == "__main__":
    torch.manual_seed(0)

    # ── 假的 DINOv2: 结构对齐 HF Dinov2Model（register 式直接走层）──
    class _FakeDinoLayer(nn.Module):
        """HF Dinov2EncoderLayer 属性布局 + 真实单头自注意力 forward。"""
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
        """最小假 DINO: 只含 register 式用到的 embeddings / encoder.layer / layernorm。"""
        def __init__(self, dim: int = 64, n_layers: int = 4, num_patches: int = 16):
            super().__init__()
            self.embeddings = _FakeEmbeddings(dim, num_patches)
            self.encoder = nn.Module()
            self.encoder.layer = nn.ModuleList(
                [_FakeDinoLayer(dim) for _ in range(n_layers)])
            self.layernorm = nn.LayerNorm(dim)

    N, D = 16, 64
    dino = FakeDino(dim=D, num_patches=N)
    model = SRPhase1V2(dino, num_patches=N, dim=D,
                       decoder_steps=square_block_starts(N))  # 显式传完整分块计划(不切片)
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

    # ── 2. OutputQueryDecoder: 分块采样计划 + 分块掩码(读侧 memory_mask)
    #    + 查询自注意力块因果(tgt_mask) + 累加结果损失 ──
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
    # 查询自注意力块因果: tgt_mask (L,L), 行序 = (t,k) 展平, 步 ti 的行只
    # 允许步 ≤ ti 的列（同步内全双向 + 之前步）, 未来步 -inf
    tg = model.decoder.tgt_mask
    assert tg is not None, "tgt_mask 不应为 None（查询自注意力块因果）"
    assert tg.shape == (len(T_steps) * N, len(T_steps) * N), tg.shape
    for ti in range(len(T_steps)):
        qrow = tg[ti * N]
        assert (qrow[:(ti + 1) * N] == 0).all(), \
            f"步 {ti} 应可见步 ≤{ti} 的查询行"
        assert (qrow[(ti + 1) * N:] == float("-inf")).all(), \
            f"步 {ti} 不应见未来步查询行"
    Y = model.decoder.last_Y                                 # (B,|T|,N,D) 特征
    assert Y.shape == (2, len(T_steps), N, D)
    # 与模型 decode 相同的梯度解耦构造: carry 无梯度 + 自己的预测
    Y_cum = torch.cat([torch.zeros_like(Y[:, :1]),
                       Y.cumsum(dim=1)[:, :-1]], dim=1).detach() + Y
    Y_pix = model.pixel_head(Y_cum)                          # (B,|T|,N,588) 累加像素
    assert Y_pix.shape == (2, len(T_steps), N, PATCH_PX)
    # 数值不变性: 与朴素 cumsum 只差 float32 求和顺序噪声（<1e-4 级）
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
    # 用户给定示例核对（N=12）: 第一个步 t=1 可见 [0..3]; 步 2 (t=4) 只看 [4..8]
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
    # 切片后: 第一个步 t=25 可见 0..35; 其余步只允许自己的块（t=81 → [81..99]）
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
          f"+ 查询自注意力块因果(步 t 只见步 ≤t) "
          f"+ 累加结果像素损失正确")

    # ── 2b. num_specials(K) 与 N 解耦: K 由最终采样步集自动推导（无花瓶）──
    # derive_num_specials 公式核对（权威示例）
    assert derive_num_specials(576, [25, 36, 49, 64, 81]) == 99, \
        "N=576 slice [4:9] → K=99"
    assert derive_num_specials(576, [64]) == 80, "显式 steps=[64] → K=80"
    assert derive_num_specials(16, [1, 4, 9, 16]) == 16, "全量默认 → K=N"
    assert derive_num_specials(16, []) == 16, "空步集 → K=N"
    # 配置: N=16, skip_steps=1 / max_steps=3 → 切片 square_block_starts(16)
    # [1,4,9,16][1:3] = [4,9] → K = min((⌊√9⌋+1)²−1, 16) = min(15,16) = 15
    dino_k = FakeDino(dim=D, num_patches=N)
    m_k = SRPhase1V2(dino_k, num_patches=N, dim=D, skip_steps=1, max_steps=3)
    assert m_k.num_specials == 15, m_k.num_specials
    assert m_k.decoder.steps == [4, 9], m_k.decoder.steps
    assert m_k.special_bank.pos.shape == (1, 15, D), m_k.special_bank.pos.shape
    assert m_k.decoder.pos_embed.shape == (1, 16, D), m_k.decoder.pos_embed.shape
    out_k = m_k(x)                                        # 同 x: (2,3,56,56)
    z_s_k = m_k.encode(x)[1]
    assert z_s_k.shape == (2, 15, D), z_s_k.shape         # 编码器只生成 K=15 个 register
    am_k = m_k.decoder.attn_mask                          # (|T|·N, K+1) = (2·16, 16)
    assert am_k.shape == (2 * N, 16), am_k.shape
    # 覆盖断言: register 位置 1..K 每个位置至少被一个步允许（square 切片成立
    # ——首步前缀规则覆盖开头, 各块连续铺满, 无花瓶）; 列 0 = z_cls 不算
    allowed = torch.zeros(m_k.num_specials, dtype=torch.bool, device=am_k.device)
    for ti in range(len([4, 9])):
        allowed |= am_k[ti * N][1:] == 0                  # 位置 1..K ↔ allowed[0..K-1]
    assert allowed.all(), "square 切片下 register 1..K 应全覆盖（无花瓶）"
    # 梯度仍全通: 显式 K<N 模型的键侧 / 查询侧 / 像素头都要收梯度
    out_k["loss"].backward()
    for name, p in [("m_k.special_bank.pos", m_k.special_bank.pos),
                    ("m_k.decoder.pos_embed", m_k.decoder.pos_embed),
                    ("m_k.decoder.query_base", m_k.decoder.query_base),
                    ("m_k.pixel_head.net", m_k.pixel_head.net[0].weight)]:
        g = p.grad
        assert g is not None and g.abs().sum() > 0, f"{name} 收不到梯度"
    print(f"[ok] num_specials 自动推导: N={N} slice[1:3] → steps=[4,9], "
          f"K={m_k.num_specials}, z_s{z_s_k.shape} 形状对, "
          f"掩码 {tuple(am_k.shape)} "
          f"(每步 {N} 行查询 × K+1 列), register 1..K 全覆盖(无花瓶), "
          f"梯度全通")

    # ── 2c. 显式 num_specials（旧 checkpoint 复现/手动指定路径）──
    dino_e = FakeDino(dim=D, num_patches=N)
    m_e = SRPhase1V2(dino_e, num_patches=N, dim=D, num_specials=8,
                     decoder_steps=[4])                    # 步 4 ≤ K=8
    assert m_e.num_specials == 8 and m_e.decoder.steps == [4], \
        (m_e.num_specials, m_e.decoder.steps)
    z_s_e = m_e.encode(x)[1]
    assert z_s_e.shape == (2, 8, D), z_s_e.shape
    out_e = m_e(x)
    assert out_e["F_hat"].shape == (2, N, PATCH_PX)
    assert m_e.decoder.attn_mask.shape == (1 * N, 9), m_e.decoder.attn_mask.shape
    # 显式 K 过小 → 清晰报错（max(steps) > K）
    try:
        SRPhase1V2(FakeDino(dim=D, num_patches=N), num_patches=N, dim=D,
                   num_specials=4, decoder_steps=[9])
        raise AssertionError("应报 num_specials 过小错误")
    except AssertionError as err:
        assert "过小" in str(err), err
    print(f"[ok] 显式 num_specials: K=8 + steps=[4] → K=8, z_s{z_s_e.shape}, "
          f"K 过小(4 vs steps=[9])报错信息含'过小'")

    # ── 3. 梯度流向（整模型可训, 含 PixelHead; 首层自注意力 + 末层交叉/FFN）──
    out["loss"].backward()
    for name, p in [("dino.embeddings.patch_embeddings.weight",
                     dino.embeddings.patch_embeddings.weight),
                    ("dino.embeddings.position_embeddings",
                     dino.embeddings.position_embeddings),
                    ("dino.encoder.layer.0.mlp.fc1.weight",
                     dino.encoder.layer[0].mlp.fc1.weight),
                    ("decoder.stack.layers[0].self_attn.in_proj_weight",
                     model.decoder.stack.layers[0].self_attn.in_proj_weight),
                    ("decoder.stack.layers[-1].multihead_attn.in_proj_weight",
                     model.decoder.stack.layers[-1].multihead_attn.in_proj_weight),
                    ("decoder.stack.layers[-1].linear1.weight",
                     model.decoder.stack.layers[-1].linear1.weight),
                    ("decoder.query_base", model.decoder.query_base),
                    ("special_bank.pos", model.special_bank.pos),
                    ("pixel_head.net", model.pixel_head.net[0].weight)]:
        g = p.grad
        assert g is not None and g.abs().sum() > 0, f"{name} 收不到梯度"
    print(f"[ok] 梯度: DINO 嵌入/层 + OutputQueryDecoder({len(model.decoder.stack.layers)}×"
          f"TransformerDecoderLayer,query_base)/SpecialTokenBank/PixelHead 全部可训")

    # ── 3b. 梯度按步解耦 ──
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

    # ── 4. 推理: 同一 forward（eval + no_grad）──
    model.eval()
    with torch.no_grad():
        out_e = model(x)
    assert out_e["F_hat"].shape == (2, N, PATCH_PX)
    print(f"[ok] eval 同路径: loss={out_e['loss'].item():.4f}")

    print("\nALL CHECKS PASSED")
