"""验证 checkpoint 可被 --resume 路径严格加载（CPU，不占 GPU）。

与 train_multi.py 的加载方式逐字一致：torch.load(path, map_location="cpu")
（不传 weights_only ⇒ 走 torch 2.6+ 的默认 True），再做 strict load_state_dict。
"""
import os
import sys

import torch
from transformers import Dinov2Model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
from model_multi import MultiResSR

path = sys.argv[1]
ck = torch.load(path, map_location="cpu")
print("load ok; keys:", sorted(ck.keys()), "step:", ck.get("step"))

dino = Dinov2Model.from_pretrained("/root/autodl-tmp/models/dinov2-large")
model = MultiResSR(dino, num_specials=int(ck["args"]["num_specials"]),
                   dim=dino.config.hidden_size,
                   heads=int(ck["args"]["heads"]),
                   depth=int(ck["args"]["decoder_depth"]),
                   step_plan=ck["args"]["step_plan"], block=int(ck["args"]["block"]))
sd = model.state_dict()
ck_sd = ck["model"]
missing = [k for k in sd if k not in ck_sd]
extra = [k for k in ck_sd if k not in sd]
shape_bad = [(k, tuple(sd[k].shape), tuple(ck_sd[k].shape))
             for k in sd if k in ck_sd and tuple(sd[k].shape) != tuple(ck_sd[k].shape)]
print(f"state_dict keys: model={len(sd)} ckpt={len(ck_sd)}")
print("missing:", missing[:5], "extra:", extra[:5], "shape_mismatch:", shape_bad[:5])
model.load_state_dict(ck_sd)                     # strict=True 必须通过
og = ck["opt"]["param_groups"]
print(f"STRICT_LOAD_OK step={ck['step']} opt_groups={[g.get('name') for g in og]} "
      f"opt_lr={[g['lr'] for g in og]} adam_state={len(ck['opt']['state'])}")
print("decoder steps:", model.steps)
