#!/usr/bin/env python3
"""Comment-only cleanup of the official arm③ (D2) model so code and comments agree."""
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()


def rep(old, new):
    global src
    assert src.count(old) == 1, f"count={src.count(old)} for {old[:80]!r}"
    src = src.replace(old, new)


rep(
    """        # ── 实验臂③(2026-09-12): 臂① 步身份 + 5 步共享同一个 [cls] 槽 ──
        # (a) 共享 [cls]（控制臂）: 只有 1 个向量, 对全部 |T| 步广播 ——
        #     提供"额外可学习槽位"但不携带任何步身份;
        # (b) step_embed（臂① ）: 每步独立可学习身份向量, 零初始化 ⇒ 起点与
        #     "无身份注入"逐位一致; 只加到该步 N 个 patch 查询行, [cls] 行
        #     不加, 从而 [cls] 在 5 步间保持字面共享。
        # 三者构成 2×2 分解: 臂① = 仅身份; 臂② = 仅每步独立槽;
        # 臂③ = 身份 + 共享槽（槽位本身不带身份）。""",
    """        # ── 实验臂③(2026-09-13): 臂① 步身份 + 5 步共享同一个 [cls] 槽 ──
        # (a) 共享 [cls](控制臂): 只有 1 个向量, 对全部 |T| 步广播 —— 提供
        #     "额外可学习槽位", 槽位本身不携带任何步身份;
        # (b) step_embed(臂①): 每步独立可学习身份向量, 零初始化 ⇒ 起点与
        #     "无身份注入"逐位一致; 作用域 = 臂① 的原始作用域, 即该步的
        #     全部查询行(N 个 patch + 1 个共享 [cls])。
        # 2×2 分解: 臂① = 仅身份; 臂② = 仅每步独立槽; 臂③ = 身份 + 共享槽。
        # 实测警示(2026-09-13): 若把 step_embed 收窄成"只加 patch 行"、让
        # [cls] 行在 5 步间字面不变, 训练会落入"步不变退化吸引子"(loss 锁死
        # 1.15 / grad_norm→0.003 / 5 步输出完全相同) —— 见
        # doc/2026-09-12/data_stepidshcls/collapse_screen/。故此处用臂① 原始作用域。""",
)

rep(
    """        # 臂③(a): 臂① 的步身份向量只加到该步 N 个 patch 行, [cls] 行不加
        # （行序 = (t,k) 展平, k∈[0,N) = patch, k=N = [cls]）。零初始化
        # ⇒ 训练起点与"无身份注入"逐位一致, 新增参数 |T|·stack_dim = 10240。
        # 臂③'(D2): 臂① 的原始作用域 = 该步的全部查询行（N patch + 1 shared [cls]）
        # —— 共享的是 [cls] 的"参数"（只有一个向量）, 不是它的输入内容:
        # 步身份仍完全由 step_embed 提供, 槽位本身不携带身份。零初始化 ⇒
        # 起点与无身份注入逐位一致。""",
    """        # 臂③(b): 臂① 的步身份向量加到该步的全部查询行(N patch + 1 [cls]);
        # 行序 = (t,k) 展平, k∈[0,Q), 故 Q 个连续行共享同一 step_embed[t]。
        # 共享的是 [cls] 的"参数"(只有 1 个向量), 不是它的输入内容: 步身份
        # 完全由 step_embed 提供。零初始化 ⇒ 起点与无身份注入逐位一致;
        # 新增参数 |T|·stack_dim = 10240。""",
)

open(path, "w", encoding="utf-8").write(src)
print("CLEANUP_OK", path)
