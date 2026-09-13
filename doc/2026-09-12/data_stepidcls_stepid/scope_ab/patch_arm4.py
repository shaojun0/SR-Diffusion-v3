#!/usr/bin/env python3
"""Arm (4) = arm(2) per-step [cls] + arm(1) additive step_embed.

Applied on top of arm②'s patched model_v2.py. step_embed is zero-init ⇒ the training
start point is bit-identical to pure arm②; its scope is arm①'s original scope, i.e. all
query rows of the step (N patches + that step's own [cls]).
"""
import sys

path = sys.argv[1]
src = open(path, encoding="utf-8").read()


def rep(old, new):
    global src
    assert src.count(old) == 1, f"count={src.count(old)} for {old[:100]!r}"
    src = src.replace(old, new)


rep(
    "        self.step_cls = nn.Parameter(torch.randn(len(self.steps), dim) * 0.02)",
    "        # ── 实验臂④(2026-09-13): 臂② 每步独立 [cls] + 臂① additive step_embed ──\n"
    "        # 两臂各自单独都健康训练(臂① full_norm_l1 0.40316 / 臂② 0.37442);\n"
    "        # 本臂检验二者是否可叠加。step_embed 零初始化 ⇒ 起点逐位等于纯臂②;\n"
    "        # 作用域 = 臂① 的原始作用域, 即该步全部查询行(N patch + 1 个该步私有 [cls])。\n"
    "        self.step_cls = nn.Parameter(torch.randn(len(self.steps), dim) * 0.02)\n"
    "        self.step_embed = nn.Parameter(torch.zeros(len(self.steps), self.stack_dim))",
)

rep(
    "        Y = self.stack(self.stack_in(Y), self.stack_in(A),",
    "        Y = self.stack_in(Y)                                     # (B,|T|·Q,stack_dim)\n"
    "        # 臂④: 臂① 的步身份向量加到该步全部 Q 行(行序 (t,k), k∈[0,Q));\n"
    "        # 零初始化 ⇒ 训练起点与纯臂②逐位一致; 新增参数 |T|·stack_dim = 10240。\n"
    "        se = self.step_embed.repeat_interleave(Q, dim=0)\n"
    "        Y = Y + se.unsqueeze(0)\n"
    "        Y = self.stack(Y, self.stack_in(A),",
)

open(path, "w", encoding="utf-8").write(src)
print("ARM4_PATCH_OK", path)
