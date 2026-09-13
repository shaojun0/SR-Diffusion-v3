#!/usr/bin/env python3
"""Arm (4b) = arm② per-step [cls] + arm① step_embed on arm①'s ORIGINAL row set.

arm① predates the [cls] slot, so its step_embed was applied to the N patch query rows —
that is arm①'s exact row set. When the per-step [cls] row is added (arm②), applying
step_embed to it as well would make that row step_cls[t] + step_embed[t] (redundant, and
empirically unstable: the all-Q-rows variant collapsed 2/2). 4b keeps the [cls] row's
step identity solely in its own step_cls[t].
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
    "        # 作用域 = 臂① 的**原始行集**: 该步的 N 个 patch 查询行（臂① 时代没有\n"
    "        # [cls] 行, 故其作用域恰为 N 行）。[cls] 行的步身份只由它自己的\n"
    "        # step_cls[t] 提供 ⇒ 避免 \"step_cls[t]+step_embed[t]\" 在同一行上冗余。\n"
    "        # （全 Q 行作用域版本实测 2/2 塌缩, 见 stepidcls_stepid_attempt*_collapsed.log）\n"
    "        self.step_cls = nn.Parameter(torch.randn(len(self.steps), dim) * 0.02)\n"
    "        self.step_embed = nn.Parameter(torch.zeros(len(self.steps), self.stack_dim))",
)

rep(
    "        Y = self.stack(self.stack_in(Y), self.stack_in(A),",
    "        Y = self.stack_in(Y)                                     # (B,|T|·Q,stack_dim)\n"
    "        # 臂④: 步身份只加到该步的 N 个 patch 行(行序 (t,k), k∈[0,Q), 末行 k=N 为 [cls]);\n"
    "        # 零初始化 ⇒ 训练起点与纯臂②逐位一致; 新增参数 |T|·stack_dim = 10240。\n"
    "        se = self.step_embed.view(len(self.steps), 1, self.stack_dim).expand(-1, N, -1)\n"
    "        se = torch.cat([se, torch.zeros(len(self.steps), 1, self.stack_dim,\n"
    "                                        device=Y.device, dtype=Y.dtype)], dim=1)\n"
    "        Y = Y + se.reshape(len(self.steps) * Q, self.stack_dim).unsqueeze(0)\n"
    "        Y = self.stack(Y, self.stack_in(A),",
)

open(path, "w", encoding="utf-8").write(src)
print("ARM4B_PATCH_OK", path)
