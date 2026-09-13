#!/usr/bin/env python3
"""Behavioral check for arm④b = per-step [cls] + step_embed on the N patch rows only."""
import importlib.util
import sys

import torch

repo = sys.argv[1]
spec = importlib.util.spec_from_file_location("m4b", repo + "/model_v2.py")
m4b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m4b)

dim, N, K, steps = 256, 36, 35, [1, 4, 9, 16, 25]
dec = m4b.OutputQueryDecoder(dim=dim, num_patches=N, num_specials=K, steps=steps,
                             heads=8, depth=2, stack_dim=None, dropout=0.0)
dec.eval()
Q = N + 1
torch.manual_seed(0)
z_cls, z_s = torch.randn(2, 1, dim), torch.randn(2, K, dim)
out = dec(z_cls, z_s)
assert tuple(out.shape) == (2, 5, N, dim), out.shape
assert tuple(dec.last_step_cls.shape) == (2, 5, dim)
assert tuple(dec.attn_mask.shape) == (5 * Q, K + 1), dec.attn_mask.shape
assert tuple(dec.tgt_mask.shape) == (5 * Q, 5 * Q), dec.tgt_mask.shape
assert tuple(dec.step_cls.shape) == (len(steps), dim)          # arm②: per-step, 5 rows
assert tuple(dec.step_embed.shape) == (len(steps), dim)         # arm①: zero-init
assert torch.equal(dec.step_embed, torch.zeros(len(steps), dim)), "step_embed must be zero-init"

loss = out.pow(2).mean() + dec.last_step_cls.pow(2).mean()
loss.backward()
assert dec.step_embed.grad is not None and dec.step_embed.grad.abs().sum() > 0, "step_embed no grad"
assert dec.step_cls.grad is not None and dec.step_cls.grad.abs().sum() > 0, "step_cls no grad"
print(f"grad_norm step_embed={dec.step_embed.grad.norm():.6f} step_cls={dec.step_cls.grad.norm():.6f}")

cap = {}
dec.stack.register_forward_pre_hook(
    lambda m, a, k: cap.__setitem__("Y", a[0].detach().clone()), with_kwargs=True)

# reference = pure arm② construction (no step_embed)
with torch.no_grad():
    A = torch.cat([z_cls, z_s], dim=1) + dec.pos_embed
    A_t = A[:, dec.steps]
    cls_rows = dec.step_cls.view(1, len(steps), 1, dim).expand(2, -1, -1, -1)
    ref = torch.cat([A_t.unsqueeze(2) + dec.query_base, cls_rows], dim=2).reshape(2, len(steps) * Q, dim)

# (a) zero step_embed ⇒ bit-identical to pure arm②
with torch.no_grad():
    dec.step_embed.zero_()
    dec(z_cls, z_s)
assert torch.allclose(cap["Y"], ref, atol=1e-6), "zero step_embed must reproduce pure arm② exactly"

# (b) non-zero step_embed shifts the N patch rows but leaves the [cls] row untouched
with torch.no_grad():
    dec.step_embed.copy_(torch.arange(1, len(steps) + 1, dtype=torch.float32)
                         .view(-1, 1).expand(-1, dim))
    dec(z_cls, z_s)
Y = cap["Y"].reshape(2, len(steps), Q, dim)
ref4 = ref.reshape(2, len(steps), Q, dim)
se = dec.step_embed.view(1, len(steps), 1, dim)
assert torch.allclose(Y[:, :, :N], ref4[:, :, :N] + se, atol=1e-6), \
    "patch rows must shift by their step's step_embed"
assert torch.allclose(Y[:, :, N], ref4[:, :, N], atol=1e-6), \
    "[cls] row must NOT be shifted by step_embed (identity stays in step_cls[t])"

print("VERIFY_OK arm4b: per-step cls (5,256) + zero-init step_embed (5,256) on the N patch rows; "
      "zero-init == pure arm②; [cls] row untouched by step_embed")
