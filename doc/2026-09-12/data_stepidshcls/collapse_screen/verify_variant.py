#!/usr/bin/env python3
"""Behavioral check for the two screening variants (D1 ctl / D2 allrows)."""
import importlib.util
import sys

import torch

repo, variant = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("mv", repo + "/model_v2.py")
mv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mv)

dim, N, K, steps = 256, 36, 35, [1, 4, 9, 16, 25]
dec = mv.OutputQueryDecoder(dim=dim, num_patches=N, num_specials=K, steps=steps,
                            heads=8, depth=2, stack_dim=None, dropout=0.0)
dec.eval()
Q = N + 1
torch.manual_seed(0)
z_cls, z_s = torch.randn(2, 1, dim), torch.randn(2, K, dim)
out = dec(z_cls, z_s)
assert tuple(out.shape) == (2, 5, N, dim), out.shape
assert tuple(dec.step_cls.shape) == (1, dim), dec.step_cls.shape
has = hasattr(dec, "step_embed")
print(f"[{variant}] out={tuple(out.shape)} step_cls={tuple(dec.step_cls.shape)} step_embed={has}")

cap = {}
dec.stack.register_forward_pre_hook(
    lambda m, a, k: cap.__setitem__("Y", a[0].detach().clone()), with_kwargs=True)
with torch.no_grad():
    if has:
        dec.step_embed.copy_(
            torch.arange(1, 6, dtype=torch.float32).view(-1, 1).expand(-1, dec.stack_dim))
    dec(z_cls, z_s)
Y = cap["Y"].reshape(2, len(steps), Q, dim)
cls = Y[:, :, N, :]
same_cls = torch.allclose(cls, cls[:, :1].expand_as(cls))
patch_differ = not torch.allclose(Y[:, 0, :N], Y[:, 1, :N])
print(f"  cls rows identical across steps = {same_cls}; patch rows differ per step = {patch_differ}")

loss = out.pow(2).mean() + dec.last_step_cls.pow(2).mean()
loss.backward()
ge = dec.step_embed.grad.abs().sum().item() if has else None
gc = dec.step_cls.grad.abs().sum().item()
print(f"  grad>0: step_cls={gc > 0} step_embed={ge > 0 if has else 'n/a'}")

# expectations
if variant == "ctl":
    assert not has, "D1 must not have step_embed"
    assert same_cls, "D1 shared [cls] input must be identical across steps"
else:
    assert has, "D2 must keep step_embed"
    assert not same_cls, "D2 step_embed must reach the [cls] row too"
print(f"  VERIFY_OK {variant}")
