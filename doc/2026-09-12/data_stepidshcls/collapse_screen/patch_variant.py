#!/usr/bin/env python3
"""Build the two screening variants from the arm3 (v1) model_v2.py.

D1  ctl_sharedcls   : shared [cls] only, NO step_embed  -> is "sharing" alone the trigger?
D2  allrows_sharedcls: shared [cls] + step_embed on ALL query rows (arm①'s original scope)
"""
import sys

variant = sys.argv[1]
path = sys.argv[2]
src = open(path, encoding="utf-8").read()

SE_START = "        se = self.step_embed.view("
SE_END = "        Y = Y + se.reshape(len(self.steps) * Q, self.stack_dim).unsqueeze(0)\n"
PARAM = "        self.step_embed = nn.Parameter(torch.zeros(len(self.steps), self.stack_dim))\n"

i = src.index(SE_START)
j = src.index(SE_END)
j_end = j + len(SE_END)

if variant == "ctl":
    assert src.count(PARAM) == 1
    # slice FIRST (i/j were computed on this exact string), then drop the parameter line
    src = src[:i] + src[j_end:]
    src = src.replace(PARAM, "")
    note = "no step_embed (pure shared-[cls] control)\n"
elif variant == "allrows":
    new = ("        # 臂③'(D2): 臂① 的原始作用域 = 该步的全部查询行（N patch + 1 shared [cls]）\n"
           "        # —— 共享的是 [cls] 的\"参数\"（只有一个向量）, 不是它的输入内容:\n"
           "        # 步身份仍完全由 step_embed 提供, 槽位本身不携带身份。零初始化 ⇒\n"
           "        # 起点与无身份注入逐位一致。\n"
           "        se = self.step_embed.repeat_interleave(Q, dim=0)\n"
           "        Y = Y + se.unsqueeze(0)\n")
    src = src[:i] + new + src[j_end:]
    note = "step_embed on all Q rows (arm① scope)\n"
else:
    raise SystemExit(f"unknown variant {variant}")

open(path, "w", encoding="utf-8").write(src)
open(path + ".variant", "w", encoding="utf-8").write(note)
print("PATCH_OK", variant, path, "|", note.strip())
