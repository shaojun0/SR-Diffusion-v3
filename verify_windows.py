"""用 model_v2.step_windows 权威核对 K=144 / square 计划的 12 步读窗口与累计 token。"""
import sys

sys.path.insert(0, "/root/autodl-tmp/srdiff-multi")
from model_v2 import square_block_starts, step_windows

K = 144
steps = square_block_starts(K)
wins = step_windows(K, steps, "square", 0)
print("steps =", steps)
seen = set()
cum = 0
for i, ((lo, hi), t) in enumerate(zip(wins, steps)):
    # A 坐标: 0=z_cls, 1..K=z_s[0..K-1]
    zs = list(range(lo - 1, hi)) if lo > 0 else list(range(0, hi))
    zs = [z for z in zs if 0 <= z < K]
    new = [z for z in zs if z not in seen]
    seen.update(zs)
    cum += len(new)
    print(f"step{i+1:2d} t={t:3d} A窗口=[{lo:3d},{hi:3d}] 读A={hi-lo+1:2d} "
          f"z_s新增={len(new):2d} 累计z_s={cum:3d}  "
          f"z_s区间=[{min(zs) if zs else -1},{max(zs) if zs else -1}]")
print("union z_s =", len(seen), "（应=144）; 是否恰好等于 0..143:",
      seen == set(range(K)))
