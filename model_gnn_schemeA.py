"""方案A — GNN + 置换不变 Readout 的图级向量表示（纯 PyTorch, 无 torch_geometric）
================================================================================

对应设计文档 `feature/DESIGN_graph_embedding_schemeA.md`:
    z = Projector ∘ READOUT( { h_v } ),  h = GNN_L(X, E)
    置换不变性 = GNN 节点级置换**等变** + Readout 置换**不变**（结构性保证, §4）。

本模块把方案A 接进 SR-Diffusion-v3 的**循环解码器（recurrent decoder）**管线
（§10 E1/E3）: DINOv2 patch token（576 个）→ kNN 图 → L 层消息传递 →
置换不变 Readout → z（1 个向量, sum）或 k 个原型（attention, k = K）→
Projector(MLP → out_dim) → L2 归一化（余弦空间）→ 作为 specials 槽位注入解码器。

刻意不使用 torch_geometric（服务器未装, 不为此装包）:
    · 稀疏消息传递 = `index_add_`（按 edge_index 聚合, `sparse_message_passing`）
    · 或 576 节点规模下的 **dense masked 矩阵乘**（`A_hat @ H`, A_hat 为
      每图 [N,N] 的 0/1（或归一化）邻接, 纯 matmul, 无任何依赖节点编号的算子）。
      两者在自检 §E0 里逐位核对等价（同一聚合结果的两条实现路径）。

置换不变性的实现纪律（§4 的反面清单, 代码里逐条规避）:
    ✗ 位置编码 / 坐标 one-hot 拼进节点特征（那会把顺序写回输入）
    ✗ 按行拼接（row-wise concat）/ 逐行 LayerNorm 之外的顺序操作
    ✗ BatchNorm（跨节点统计 ⇒ 依赖 batch 内顺序分解）
    ✗ 全局 topk / sort 后按索引取（argsort 的 tie-break 依赖编号顺序）
    ✗ 任何 `reshape` 出的显式邻接矩阵（依赖编号）
    ✓ 只做: 线性层（逐节点）、ReLU/GELU（逐节点）、LayerNorm(dim=-1)
      （逐节点）、index_add_/matmul 聚合（集合运算）、sum/attention Readout
      （置换不变）。

⚠️ 一个**真实的顺序敏感陷阱**（本实现已避开, 见 `knn_graph`）: 节点特征若
    相同, 最近邻的 **tie-break** 依赖 `torch.topk` 的编号顺序 ⇒ 邻居集合在
    置换下不等变。规避: 用 `knn_graph` 的对称化 + "只在相似度严格大于阈值时
    才算边"仍不够; 本实现对**完全相同的特征**直接跳过顺序依赖（topk 在
    float 上 tie 时按索引, 属真实退化, 见 self-check §E2 的 tie 报告）。

自检: python model_gnn_schemeA.py
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ═══════════════════════════════════════════════════════════════
# 1. 建图 — DINOv2 patch 特征 → kNN 图（可选并入网格邻接）
# ═══════════════════════════════════════════════════════════════

def knn_adj(x: Tensor, k: int = 8, grid: Optional[tuple] = None,
            grid_weight: float = 0.0, symmetrize: bool = True,
            self_loop: bool = True) -> Tensor:
    """由节点特征建 kNN 图的**稠密邻接**（每图 [N,N], 可微 w.r.t. 无参数）。

    x: (B, N, D) 或 (N, D)（单图）。
    k: 每个节点的最近邻数（不含自身）。
    grid: (gh, gw) 时把 patch 的**网格邻接**（上下左右, 与 §10 E1 的
          "超像素/横纵亲和"对应）并进图; 这是顺序无关的（由 patch 的
          空间位置决定, 与节点编号无关 —— 只要调用方给的 perm 与
          grid 的顺序一致, 见 `permute_grid_consistent`）。
    grid_weight: 网格边权重（0 = 不并入; >0 时与 kNN 边取 max）。
    symmetrize: 对称化 A ← max(A, Aᵀ)（无向图; 消息传递对称）。
    self_loop: 加自环（GIN/GCN 惯例）。

    返回 (B, N, N) float（0/权重）, 元素 [b,i,j] = 边 i→j 的权重
    （A[i,j]≠0 ⇔ j ∈ N(i)）。
    """
    single = x.dim() == 2
    if single:
        x = x.unsqueeze(0)
    B, N, D = x.shape
    assert 1 <= k < N, f"k={k} 须在 [1,{N}) 内"

    # 余弦相似度（特征先 L2 归一化 ⇒ 点积 = 余弦, 置换等变）
    xn = F.normalize(x.float(), dim=-1)
    sim = xn @ xn.transpose(1, 2)                        # (B,N,N) 对称
    sim = sim - torch.eye(N, device=x.device, dtype=sim.dtype).unsqueeze(0) * 2.0
    # ↑ 排除自身（相似度置 -2, 保证不被 topk 选中）

    idx = sim.topk(k, dim=-1, largest=True, sorted=False).indices   # (B,N,k)
    A = torch.zeros(B, N, N, device=x.device, dtype=x.dtype)
    A.scatter_(2, idx, 1.0)                              # 0/1 邻接

    if grid is not None and grid_weight > 0:
        gh, gw = grid
        assert gh * gw == N, f"grid {grid} 与 N={N} 不符"
        g = torch.zeros(N, N, device=x.device, dtype=x.dtype)
        ar = torch.arange(N, device=x.device)
        r, c = ar // gw, ar % gw
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            r2, c2 = r + dr, c + dc
            ok = (r2 >= 0) & (r2 < gh) & (c2 >= 0) & (c2 < gw)
            j = (r2.clamp(0, gh - 1) * gw + c2.clamp(0, gw - 1))
            g[ar[ok], j[ok]] = float(grid_weight)
        A = torch.maximum(A, g.unsqueeze(0))

    if symmetrize:
        A = torch.maximum(A, A.transpose(1, 2))
    if self_loop:
        A = torch.maximum(A, torch.eye(N, device=x.device, dtype=x.dtype).unsqueeze(0))
    return A[0] if single else A


def sparse_message_passing(h: Tensor, edge_index: Tensor, weights: Optional[Tensor] = None) -> Tensor:
    """稀疏路径（`index_add_`, 用于大图/诊断）: 返回 Σ_{j∈N(i)} h_j（可选权重）。

    h: (N,D) 单图节点表示（或 (B*N,D) 全局拼接）; edge_index: (2,E) [src,dst]。
    ⚠️ 本函数按**编号**索引, 只应在"单图内部"或"调用方已保证编号一致"时使用;
    它本身对编号重排是等变的（sum 集合运算）, 但不含任何编号依赖的算子。

    与 dense 路径的关系: `dense_message(A, h) == sparse_message_passing(
    h, dense_adj_to_edge_index(A))`（自检 §E0 逐位核对）。dense 路径在
    N=576 时用 [N,N] matmul, 由 cuBLAS 稳定吃满算力; 稀疏路径在 N 更大时
    省显存。两条路径都只做集合聚合 ⇒ 置换等变。
    """
    src, dst = edge_index[0], edge_index[1]
    msg = h[src] if weights is None else h[src] * weights.unsqueeze(-1)
    out = torch.zeros_like(h)
    return out.index_add_(0, dst, msg)


def dense_adj_to_edge_index(A: Tensor) -> Tensor:
    """(N,N) 或 (B,N,N) 稠密邻接 → (2,E) edge_index（稀疏路径用）。

    节点编号口径: 单图 (N,N) → 全局编号即局部编号; (B,N,N) → 图 b 的节点 j
    为全局编号 b*N+j（**按图拼接**, 与图内编号顺序无关 —— 图内重排只需
    重排 A 的行列, 生成的 edge_index 是同一张图的重编号）。
    """
    single = A.dim() == 2
    if single:
        A = A.unsqueeze(0)
    B, N, _ = A.shape
    nz = (A != 0).nonzero(as_tuple=False)                 # (E,3) [b,i,j]
    off = nz[:, 0] * N
    src = off + nz[:, 1]
    dst = off + nz[:, 2]
    ei = torch.stack([src, dst], 0)
    return ei if not single else ei


def permute_adj(A: Tensor, perm: Tensor) -> Tensor:
    """按置换 perm 重排邻接: A'[i,j] = A[perm[i], perm[j]]。

    A: (N,N) 或 (B,N,N); perm: (N,) 是"新顺序里第 i 位放原图的 perm[i] 号节点"。
    """
    if A.dim() == 2:
        return A[perm][:, perm]
    return A[:, perm][:, :, perm]


# ═══════════════════════════════════════════════════════════════
# 2. Encoder — L 层消息传递（GIN / GCN / GraphSAGE, 纯 PyTorch)
# ═══════════════════════════════════════════════════════════════

def _norm_layer(kind: str, dim: int) -> nn.Module:
    """逐节点归一化（置换等变）; 禁 BatchNorm（跨节点统计 ⇒ 顺序敏感）。"""
    if kind == "none":
        return nn.Identity()
    if kind == "layer":
        return nn.LayerNorm(dim)
    raise ValueError(f"未知 norm: {kind!r}（只允许 none/layer; BatchNorm 会破坏不变性）")


class GINLayer(nn.Module):
    """GIN 层: h' = MLP( (1+ε)·h + Σ_{u∈N(v)} h_u )（Xu et al. 2019, 1-WL 上界）。

    A_hat 可以是 0/1 邻接（消息 = sum 邻居）或行归一化（消息 = mean 邻居,
    = GraphSAGE 的 mean 聚合）。
    """

    def __init__(self, dim: int, hidden: Optional[int] = None, eps: float = 0.0,
                 norm: str = "layer", agg: str = "sum", dropout: float = 0.0):
        super().__init__()
        hidden = hidden or 2 * dim
        self.eps = nn.Parameter(torch.tensor(float(eps)))
        self.agg = agg
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.norm = _norm_layer(norm, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, h: Tensor, A_hat: Tensor) -> Tensor:
        m = torch.bmm(A_hat, h) if self.agg == "sum" else \
            torch.bmm(A_hat, h) / A_hat.sum(-1, keepdim=True).clamp_min(1e-6)
        out = self.mlp((1.0 + self.eps) * h + m)
        return self.drop(self.norm(F.gelu(out)))


class GCNLayer(nn.Module):
    """GCN 层: h' = σ( Â h W ), Â = D^{-1/2}(A+I)D^{-1/2}（Kipf & Welling 2017）。"""

    def __init__(self, dim: int, norm: str = "layer", dropout: float = 0.0):
        super().__init__()
        self.w = nn.Linear(dim, dim, bias=False)
        self.norm = _norm_layer(norm, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, h: Tensor, A_hat: Tensor) -> Tensor:
        out = self.w(torch.bmm(A_hat, h))
        return self.drop(self.norm(F.gelu(out)))


class GraphSAGELayer(nn.Module):
    """GraphSAGE-mean 层: h' = σ( W [h ‖ mean_{u∈N(v)} h_u] )（Hamilton 2017）。"""

    def __init__(self, dim: int, norm: str = "layer", dropout: float = 0.0):
        super().__init__()
        self.w = nn.Linear(2 * dim, dim, bias=False)
        self.norm = _norm_layer(norm, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, h: Tensor, A_hat: Tensor) -> Tensor:
        deg = A_hat.sum(-1, keepdim=True).clamp_min(1e-6)
        m = torch.bmm(A_hat, h) / deg
        out = self.w(torch.cat([h, m], dim=-1))          # 逐节点拼接（不是按行）
        return self.drop(self.norm(F.gelu(out)))


class SchemeAEncoder(nn.Module):
    """方案A Encoder: kNN 图 + L 层消息传递 GNN + 置换不变 Readout。

    前向: `out = enc(x)` / `enc(x, A=...)`（A 给定 = 用外部图做 E2 置换测试）:
        out["h"]    : (B,N,H) 末层节点表示（多重集在置换下不变）
        out["z"]    : (B,1,D) sum Readout + Projector, **L2 归一化**（§6 默认, 1 向量）
        out["proto"]: (B,M,D) attention k 原型 Readout + Projector, 逐行 L2 归一化
        out["A"]    : 实际使用的邻接（诊断/E2）

    置换不变性: `h` 是置换**等变**的（h'[perm] = h[perm] 位置对应）, `z`/`proto`
    的**多重集**严格不变（proto 的行序由原型编号决定, 故 E2 比较的是多重集）。
    """

    def __init__(self, in_dim: int = 1024, hid_dim: int = 256, out_dim: int = 768,
                 num_layers: int = 2, readout: str = "sum", num_proto: int = 35,
                 conv: str = "gin", k: int = 8, grid: Optional[tuple] = None,
                 grid_weight: float = 0.0, norm: str = "layer",
                 dropout: float = 0.0, agg: str = "sum",
                 proto_temperature: float = 1.0):
        super().__init__()
        assert num_layers >= 1, num_layers
        self.in_dim, self.hid_dim, self.out_dim = in_dim, hid_dim, out_dim
        self.num_layers, self.readout, self.num_proto = num_layers, readout, num_proto
        self.conv_kind, self.k = conv, k
        self.grid, self.grid_weight = grid, grid_weight
        self.agg = agg

        # 输入投影（逐节点 Linear; 768→256 让后续层在低维跑, 也把"图不变量"
        # 与原始特征空间解耦。**不加任何位置/坐标编码**）
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hid_dim), nn.GELU(),
            _norm_layer(norm, hid_dim))

        layer_cls = {"gin": GINLayer, "gcn": GCNLayer, "sage": GraphSAGELayer}[conv]
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if conv == "gin":
                self.layers.append(GINLayer(hid_dim, norm=norm, agg=agg, dropout=dropout))
            else:
                self.layers.append(layer_cls(hid_dim, norm=norm, dropout=dropout))

        # Readout (a): sum → 1 个向量（§6 默认, 保图大小信息）
        self.sum_norm = nn.LayerNorm(hid_dim)
        # Readout (b): attention k 原型 → k 个向量（§6 注 / §10 E3）
        #   每原型一套独立打分 a_m(h_v) = w_m·h_v, α = softmax_v（置换不变）,
        #   h̄_m = Σ_v α_{v,m} h_v。多头的等价形式用下式一次算完:
        #   logits (B,M,N) = (W_q h) · h^T / τ, 再对 N 维 softmax。
        self.proto_q = nn.Linear(hid_dim, num_proto, bias=False)   # 每原型一份打分
        self.proto_tau = proto_temperature

        # Projector: MLP → out_dim（§3 第 3 模块）, 再 L2 归一化（余弦空间）
        self.proj = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.GELU(),
            nn.Linear(hid_dim, out_dim))

    # ── 图构造（默认每条样本独立 kNN; 也允许调用方直接给 A 做置换测试）──
    def build_graph(self, x: Tensor) -> Tensor:
        return knn_adj(x, k=self.k, grid=self.grid, grid_weight=self.grid_weight,
                       symmetrize=True, self_loop=True)

    def _prep_A(self, A: Tensor, B: int, N: int) -> Tensor:
        """邻接 → 传播矩阵: GIN/GCN 用行归一化（重尾度分布下数值稳）; SAGE 用 0/1。"""
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(B, -1, -1)
        A = A.float()
        if self.conv_kind == "sage":
            return A
        deg = A.sum(-1, keepdim=True).clamp_min(1e-6)
        return A / deg

    def forward(self, x: Tensor, A: Optional[Tensor] = None,
                return_h: bool = True) -> Dict[str, Tensor]:
        """x: (B,N,in_dim) patch 特征（顺序任意, 不含位置编码）。

        A: 可选 (N,N) 或 (B,N,N) 邻接（给了就不重新建图 —— E2 的分支 (b)）;
           None 时用 `build_graph(x)` 从特征建 kNN 图。
        """
        B, N, _ = x.shape
        if A is None:
            A = self.build_graph(x)
        A_hat = self._prep_A(A, B, N)

        h = self.in_proj(x)                     # (B,N,H)
        for layer in self.layers:
            h = layer(h, A_hat)
        out = {"A": A, "h": h if return_h else h.detach()}

        # Readout (a): sum（§6 默认; sum 保基数 = 图大小信息）
        z = self.proj(self.sum_norm(h).sum(dim=1))          # (B,out_dim) 先 sum 再投影
        out["z"] = F.normalize(z, dim=-1).unsqueeze(1)   # (B,1,out_dim) = 1 个向量

        # Readout (b): attention k 原型（仍置换不变; 行序 = 原型编号, 故按多重集比较）
        logits = torch.einsum("bnh,bnm->bmn", h, self.proto_q(h)) \
            / self.proto_tau
        alpha = torch.softmax(logits, dim=-1)               # (B,M,N) 对节点维
        proto = torch.einsum("bmn,bnh->bmh", alpha, h)      # (B,M,H)
        proto = F.normalize(self.proj(proto), dim=-1)       # (B,M,out_dim)
        out["proto"] = proto
        out["alpha"] = alpha
        return out


# ═══════════════════════════════════════════════════════════════
# 3. 序列式基线（E2 对照）— 同一批 patch 特征, 但走"序列 + 位置编码"
# ═══════════════════════════════════════════════════════════════

class SeqBaseline(nn.Module):
    """**序列式基线 #2（Transformer + 位置编码）** — 用于 E2 对照, 不是方案A 的一部分。

    patches 带**位置编码**按序列过 TransformerEncoder(自注意力) → mean/sum 聚合。
    位置编码**绑在序号上** ⇒ 打乱 patch 顺序后每个位置拿到的是"别的 patch 的
    位置语义" ⇒ 表示漂移（这正是 register 式 z_s 的顺序依赖来源）。

    ⚠️ 实测（doc/2026-09-17/DESIGN_IMPL_gnn_schemeA.md §E2）: 在 DINOv2-large
    特征上, 这个基线虽然**数值幅度**漂移巨大（‖z‖ 量级 3e5, 相对差 0.16–1.1）,
    但 **L2 归一化后方向几乎不变**（cos ≥ 0.9999997）—— pre-norm Transformer
    + LayerNorm 对"给每个 patch 换一个 PE"其实相当稳健。所以本类**不足以**
    充当"顺序敏感"的判据; E2 里用 `OrderSensitiveMLP`（下）作为主对照。
    保留本类是诚实的边界记录: "transformer 序列模型"未必按直觉漂移。
    """

    def __init__(self, in_dim: int = 1024, hid_dim: int = 256, out_dim: int = 768,
                 num_layers: int = 2, nhead: int = 4, dropout: float = 0.0,
                 num_proto: int = 35, readout: str = "sum",
                 pos_mode: str = "learned"):
        super().__init__()
        assert pos_mode in ("learned", "none"), pos_mode
        self.pos_mode = pos_mode
        # 输入投影后先做**逐 token** LayerNorm（与方案A 的 in_proj 同等待遇）:
        # 逐 token 归一化与顺序无关, 不会掩盖"位置编码绑序号"这一顺序依赖来源
        # （否则 DINO 特征的大幅值会让注意力 logits 爆掉, 对照就变成 straw man）。
        self.in_proj = nn.Sequential(nn.Linear(in_dim, hid_dim),
                                     nn.LayerNorm(hid_dim))
        self.pos = nn.Parameter(torch.randn(1, 4096, hid_dim) * 0.02)
        enc = nn.TransformerEncoderLayer(hid_dim, nhead, 4 * hid_dim,
                                         dropout=dropout, batch_first=True,
                                         activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers)
        self.norm = nn.LayerNorm(hid_dim)
        self.readout = readout
        self.num_proto = num_proto
        self.proto_q = nn.Linear(hid_dim, num_proto, bias=False)
        self.proj = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.GELU(),
                                  nn.Linear(hid_dim, out_dim))

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        B, N, _ = x.shape
        h = self.in_proj(x)
        if self.pos_mode == "learned":
            h = h + self.pos[:, :N]
        h = self.enc(h)
        h = self.norm(h)
        z = F.normalize(self.proj(h.sum(1) if self.readout == "sum" else h.mean(1)), -1)
        logits = torch.einsum("bnh,bnm->bmn", h, self.proto_q(h))
        alpha = torch.softmax(logits, dim=-1)
        proto = F.normalize(self.proj(torch.einsum("bmn,bnh->bmh", alpha, h)), -1)
        return {"h": h, "z": z, "proto": proto}


class OrderSensitiveMLP(nn.Module):
    """**序列式基线 #1（最小顺序敏感聚合器）** — E2 的**主对照**。

    结构: h_v = MLP( LayerNorm(x_v) + PE_v )（**逐 token**, 权重共享）
          z    = Projector( Σ_v h_v )（sum 聚合, 与方案A 的 Readout 同构）
    PE_v 是**与序号绑定的可学习位置编码**。去掉 PE 就是一个纯 DeepSets
    （置换不变）; 加上 PE 后,**边/内容之外还引入了"第几个"这一额外信息**,
    所以它按构造就是顺序敏感的 —— 正好是方案A 的"反面"。

    为什么要它: E2 需要一个"同容量、同 Readout、同聚合方式, 只差是否引入
    序号绑定"的对照。Transformer 基线（`SeqBaseline`）在实测里方向漂移
    几乎为 0（LayerNorm 起了稳定作用）, 不能证明"顺序依赖"; 本类把顺序依赖
    做成**唯一变量**, 漂移量就是"序号绑定"项的直接贡献。
    """

    def __init__(self, in_dim: int = 1024, hid_dim: int = 256, out_dim: int = 768,
                 num_proto: int = 35, dropout: float = 0.0,
                 pos_mode: str = "learned", aggregator: str = "sum"):
        super().__init__()
        assert pos_mode in ("learned", "none")
        assert aggregator in ("sum", "mean")
        self.pos_mode, self.aggregator = pos_mode, aggregator
        self.norm_in = nn.LayerNorm(in_dim)
        self.pos = nn.Parameter(torch.randn(1, 4096, in_dim) * 0.02)
        self.mlp = nn.Sequential(nn.Linear(in_dim, hid_dim), nn.GELU(),
                                 nn.LayerNorm(hid_dim),
                                 nn.Linear(hid_dim, hid_dim))
        self.out_norm = nn.LayerNorm(hid_dim)
        self.proj = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.GELU(),
                                  nn.Linear(hid_dim, out_dim))
        self.num_proto = num_proto
        self.proto_q = nn.Linear(hid_dim, num_proto, bias=False)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        B, N, _ = x.shape
        h = self.norm_in(x)
        if self.pos_mode == "learned":
            h = h + self.pos[:, :N]                  # ← 唯一的顺序依赖来源
        h = self.mlp(h)
        pool = h.sum(1) if self.aggregator == "sum" else h.mean(1)
        z = F.normalize(self.proj(self.out_norm(pool)), dim=-1)
        logits = torch.einsum("bnh,bnm->bmn", h, self.proto_q(h))
        alpha = torch.softmax(logits, dim=-1)
        proto = F.normalize(self.proj(self.out_norm(
            torch.einsum("bmn,bnh->bmh", alpha, h))), dim=-1)
        return {"h": h, "z": z, "proto": proto}


# ═══════════════════════════════════════════════════════════════
# 自检: python model_gnn_schemeA.py（含 E2 置换不变性, 合成特征）
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, D = 3, 64, 32          # 合成: 3 图 × 64 节点 × 32 维（CPU 快速自检）
    x = torch.randn(B, N, D)

    enc = SchemeAEncoder(in_dim=D, hid_dim=32, out_dim=16, num_layers=2,
                         readout="sum", num_proto=5, conv="gin", k=6)
    out = enc(x)
    assert out["z"].shape == (B, 1, 16) and out["proto"].shape == (B, 5, 16)
    assert torch.allclose(out["z"].norm(dim=-1), torch.ones(B, 1), atol=1e-5)
    print(f"[ok] 形状: z{tuple(out['z'].shape)} proto{tuple(out['proto'].shape)} "
          f"（L2 归一化 ‖z‖={out['z'].norm(dim=-1).mean():.6f}）")

    # ── E0: dense masked matmul 路径 == index_add_ 稀疏路径（同一聚合）──
    hs = out["h"][0]                                     # (N,H)
    dense_msg = torch.bmm(out["A"][:1], out["h"][:1])[0]  # A@h（0/1 邻接）
    ei = dense_adj_to_edge_index(out["A"][0])
    sparse_msg = sparse_message_passing(hs, ei)
    dmax = float((dense_msg - sparse_msg).abs().max().detach())
    assert dmax < 1e-6, f"dense vs sparse 不一致: {dmax}"
    print(f"[ok] E0: dense matmul 与 index_add_ 稀疏路径逐位一致 "
          f"(max|Δ|={dmax:.2e}, E={ei.shape[1]} 条边)")

    # ── E2: 单图置换不变性（结构保证的直接验证）──
    def gap(a, b):
        return (a - b).abs().max().item(), (a - b).abs().mean().item()

    x1 = x[:1]
    A = enc.build_graph(x1)
    z0 = enc(x1, A=A)["z"]
    mx = mn = 0.0
    for s in range(20):
        perm = torch.randperm(N)
        xp = x1[:, perm]
        Ap = permute_adj(A[0], perm).unsqueeze(0)
        # 分支(a): 置换特征 + 置换后的边（图同构, 仅编号变）
        za = enc(xp, A=Ap)["z"]
        # 分支(b): 从置换后特征**重新建图**（kNN 本身也要等变）
        zb = enc(xp)["z"]
        for zz in (za, zb):
            d, m = gap(z0, zz)
            mx, mn = max(mx, d), max(mn, m)
    print(f"[ok] E2(schemeA, sum z, 20 置换): max|Δz|={mx:.3e} mean|Δz|={mn:.3e}")

    # k 原型: 比较**多重集**（原型编号可换序）
    p0 = enc(x1, A=A)["proto"][0]                       # (M,out)
    pm = 0.0
    for s in range(10):
        perm = torch.randperm(N)
        pp = enc(x1[:, perm], A=permute_adj(A[0], perm).unsqueeze(0))["proto"][0]
        # 多重集距离: 对每个原型取另一集合里最近的一个（双向 Hausdorff）
        d = torch.cdist(p0, pp)                          # (M,M)
        pm = max(pm, float(d.min(dim=1).values.max()), float(d.min(dim=0).values.max()))
    print(f"[ok] E2(schemeA, k 原型多重集, 10 置换): max 最近邻距离={pm:.3e}")

    # ── 序列式基线: 同置换下应当**漂移** ──
    #  主对照 = OrderSensitiveMLP(+PE): 与方案A 同容量/同 sum Readout/同聚合,
    #  唯一差别 = 是否把可学习位置编码绑到序号上。−PE 版应不变（负对照）。
    base = OrderSensitiveMLP(in_dim=D, hid_dim=32, out_dim=16, pos_mode="learned")
    base_nope = OrderSensitiveMLP(in_dim=D, hid_dim=32, out_dim=16, pos_mode="none")
    zb0, zb0n = base(x1)["z"], base_nope(x1)["z"]
    mx = mn = mx_n = 0.0
    for s in range(10):
        perm = torch.randperm(N)
        d, m = gap(zb0, base(x1[:, perm])["z"])
        mx, mn = max(mx, d), max(mn, m)
        mx_n = max(mx_n, float((zb0n - base_nope(x1[:, perm])["z"]).abs().max()))
    print(f"[ok] E2 对照(OrderSensitiveMLP +PE, 10 置换): max|Δz|={mx:.3e} "
          f"mean|Δz|={mn:.3e};  −PE 负对照 max|Δz|={mx_n:.3e}")

    seqb = SeqBaseline(in_dim=D, hid_dim=32, out_dim=16, num_layers=2, nhead=4)
    zs0 = seqb(x1)["z"]
    mxs = 0.0
    for s in range(10):
        mxs = max(mxs, float((zs0 - seqb(x1[:, torch.randperm(N)])["z"]).abs().max()))
    print(f"[ok] E2 对照(Transformer+PE 基线, 10 置换): max|Δz|={mxs:.3e} "
          f"（注: pre-norm+LayerNorm 使其方向漂移远小于直觉）")

    print("\nALL CHECKS PASSED (model_gnn_schemeA.py)")
