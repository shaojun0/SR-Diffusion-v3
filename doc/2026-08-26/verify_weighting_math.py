"""
数值验证: 前缀加权损失 w(k)·L1_k 的数学恒等式与梯度压力
=============================================================
验证内容:
  (1) 正确的 telescoping 恒等式:
        Σ_j w_j·L1_j ≡ (Σ_j w_j)·L1_N + Σ_i (Σ_{j<i} w_j)·Δ_i
      其中 Δ_i = L1_{i-1} − L1_i (第 i 个 token 的边际改善)。
  (2) DESIGN_prefix_weighting.md §3 声称的恒等式
        Σ_j w_j·L1_j ≡ Σ_j (w_j−w_{j+1})·Σ_{i≤j} Δ_i (+边界项)
      数值上验证: 其 RHS = Σ_i w_i·Δ_i, 与 LHS 并不恒等(草稿笔误/方向性错误)。
  (3) 分部求和 (summation by parts) 恒等式:
        Σ_i w_i·Δ_i = w_1·L1_0 − w_N·L1_N + Σ_{j=1}^{N−1} (w_{j+1}−w_j)·L1_j
  (4) 梯度压力: 均匀采样 k∈[1..N] 时 token i 参与前缀 j≥i 的次数 ∝ N−i+1;
      加权损失下 token i 的总梯度压力 ∝ Σ_{j≥i} w_j (w 递减 ⇒ 递减 ⇒ 前置)。
"""
import numpy as np

rng = np.random.default_rng(0)
N = 576

# ── 构造一条"合理"的 L1 曲线(非增)与递减权重 ──
# 用真实 k 扫描锚点 + 单调插值, 保证 Δ_i ≥ 0
anchors_k = np.array([0, 1, 2, 32, 64, 128, 256, 384, 512, 576], dtype=float)
anchors_l = np.array([0.00407, 0.00339, 0.0033, 0.0033, 0.0032,
                      0.0030, 0.0028, 0.0026, 0.0025, 0.00114], dtype=float)
L1 = np.interp(np.arange(N + 1), anchors_k, anchors_l)          # (N+1,) L1_j, j=0..N
L1 = np.maximum.accumulate(L1[::-1])[::-1]                       # 强制非增(数值保险)
assert np.all(np.diff(L1) <= 0 + 1e-12)
Delta = -np.diff(L1)                                             # Δ_i = L1_{i-1} − L1_i ≥ 0, i=1..N

# 递减权重 w_0 ≥ w_1 ≥ ... ≥ w_N ≥ 0 (对前缀损失 L1_j 的权重)
w = 1.0 / (np.arange(N + 1) + 1.0)                               # w_j = 1/(j+1) 递减

print("=" * 78)
print("恒等式 (1): Σ_j w_j·L1_j == (Σw)·L1_N + Σ_i (Σ_{j<i} w_j)·Δ_i")
lhs = float(np.sum(w * L1))
rhs = float(np.sum(w) * L1[N] + np.sum(np.cumsum(w[:-1]) * Delta))
print(f"  LHS = {lhs:.12e}   RHS = {rhs:.12e}   差 = {abs(lhs-rhs):.3e}  "
      f"{'✓ 恒等' if abs(lhs-rhs) < 1e-9 else '✗ 不恒等'}")

print("-" * 78)
print("恒等式 (2) 草稿 §3: Σ_j w_j·L1_j ≡ Σ_j (w_j−w_{j+1})·Σ_{i≤j}Δ_i (+边界项)")
# 按字面实现草稿 RHS: Σ_{j} (w_j−w_{j+1})·(L1_0 − L1_j), j=0..N−1 (Σ_{i≤j}Δ_i = L1_0 − L1_j)
doc_rhs = float(np.sum((w[:-1] - w[1:]) * (L1[0] - L1[:-1])))
# 可验证: 草稿 RHS = Σ_i (w_{i-1} − w_N)·Δ_i (telescoping 于 j), 并非 Σ_i w_i·Δ_i
doc_alt = float(np.sum((w[:-1] - w[-1]) * Delta))
print(f"  LHS = {lhs:.12e}   草稿 RHS(字面) = {doc_rhs:.12e}   差 = {abs(lhs-doc_rhs):.3e}")
print(f"  交叉验证: 草稿 RHS 精确等于 Σ_i (w_{{i}} − w_N)·Δ_i = "
      f"{float(np.sum((w[1:] - w[-1]) * Delta)):.12e} "
      f"{'✓' if abs(doc_rhs - np.sum((w[1:] - w[-1]) * Delta)) < 1e-9 else '✗'}")
print(f"  ⇒ 草稿恒等式数值上不成立(差={abs(lhs-doc_rhs):.3e}≫0)，方向问题见下。")

print("-" * 78)
print("恒等式 (3) 分部求和: Σ_i w_i·Δ_i == w_1·L1_0 − w_N·L1_N + Σ_{j=1}^{N-1}(w_{j+1}−w_j)·L1_j")
lhs3 = float(np.sum(w[1:] * Delta))                                # Σ_{i=1..N} w_i·Δ_i
sbp = float(w[1] * L1[0] - w[N] * L1[N] + np.sum((w[2:] - w[1:-1]) * L1[1:N]))
print(f"  LHS = {lhs3:.12e}   RHS = {sbp:.12e}   差 = {abs(lhs3-sbp):.3e}  "
      f"{'✓ 恒等' if abs(lhs3-sbp) < 1e-9 else '✗ 不恒等'}")

print("-" * 78)
print("方向检查: 递减 w 下, 各 Δ_i 在两种目标中的系数")
c_pref = np.cumsum(w[:-1])               # 加权前缀损失中 Δ_i 的系数 (递增)
c_marg = w[1:]                           # 草稿目标 Σ w_i Δ_i 中 Δ_i 的系数 (递减)
print(f"  Σ_j w_j L1_j 中 Δ_i 系数 Σ_{'{j<i}'} w_j : i=1→{c_pref[0]:.4f}, i=N/2→{c_pref[N//2]:.4f}, i=N→{c_pref[-1]:.4f}  (递增)")
print(f"  Σ_i w_i Δ_i 中 Δ_i 系数 w_i        : i=1→{c_marg[0]:.4f}, i=N/2→{c_marg[N//2]:.4f}, i=N→{c_marg[-1]:.4f}  (递减)")

print("-" * 78)
print("梯度压力 (4a): 均匀采样 k∈[1..N], token i 参与的前缀数 ∝ N−i+1")
counts = np.array([N - i + 1 for i in range(1, N + 1)])
print(f"  token i=1 → {counts[0]}, i=N/2 → {counts[N//2]}, i=N → {counts[-1]}  (严格线性递减)")

print("-" * 78)
print("梯度压力 (4b): 加权损失 Σ_j w_j L1_j 下 token i 的总压力 ∝ Σ_{j≥i} w_j")
pres = np.array([np.sum(w[j:]) for j in range(1, N + 1)])       # Σ_{j=i..N} w_j
print(f"  w_j=1/(j+1): i=1→{pres[0]:.4f}, i=N/2→{pres[N//2]:.4f}, i=N→{pres[-1]:.4f}  (递减, 前置偏置)")
print(f"  w_j≡1 (均匀): i=1→{counts[0]}, i=N→{counts[-1]}  与 (4a) 同形 ✓")
print("=" * 78)

# 附加: 草稿"天然加成"论断的精确形式: 均匀采样的期望损失 = (1/N)Σ_j L1_j, token i 出现频率 = (N−i+1)/N
freq = counts / N
print(f"附加: 均匀采样下 token i 出现在训练损失中的频率 = (N−i+1)/N, "
      f"i=1→{freq[0]:.4f}, i=N→{freq[-1]:.4f}")
