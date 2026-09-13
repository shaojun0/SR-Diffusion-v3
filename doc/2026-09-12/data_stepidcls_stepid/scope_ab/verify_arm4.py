#!/usr/bin/env python3
"""Behavioral check for arm④ = per-step [cls] + step_embed (all Q rows)."""
import importlib.util
import sys

import torch

repo = sys.argv[1]
spec = importlib.util.spec_from_file_location("m4", repo + "/model_v2.py")
m4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m4)

dim, N, K, steps = 256, 36, 35, [1, 4, 9, 16, 25]
dec = m4.OutputQueryDecoder(dim=dim, num_patches=N, num_specials=K, steps=steps,
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

# arm② property preserved: per-step independent [cls] (5 distinct rows)
assert tuple(dec.step_cls.shape) == (len(steps), dim), dec.step_cls.shape
# arm① property added: zero-init step_embed of size |T| x stack_dim
assert tuple(dec.step_embed.shape) == (len(steps), dim), dec.step_embed.shape
assert torch.equal(dec.step_embed, torch.zeros(len(steps), dim)), "step_embed must be zero-init"

NEG = float("-inf")
assert dec.tgt_mask[0, Q] == NEG and dec.tgt_mask[Q, 0] == NEG  # blockdiag isolation intact
assert dec.tgt_mask[0, 0] == 0.0 and dec.tgt_mask[0, N] == 0.0

loss = out.pow(2).mean() + dec.last_step_cls.pow(2).mean()
loss.backward()
assert dec.step_embed.grad is not None and dec.step_embed.grad.abs().sum() > 0, "step_embed no grad"
assert dec.step_cls.grad is not None and dec.step_cls.grad.abs().sum() > 0, "step_cls no grad"
print(f"grad_norm step_embed={dec.step_embed.grad.norm():.6f} step_cls={dec.step_cls.grad.norm():.6f}")

# capture the tensor fed to self.stack
cap = {}
dec.stack.register_forward_pre_hook(
    lambda m, a, k: cap.__setitem__("Y", a[0].detach().clone()), with_kwargs=True)

# (a) zero step_embed ⇒ bit-identical to pure arm② (per-step [cls]) construction
with torch.no_grad():
    dec.step_embed.zero_()
    dec(z_cls, z_s)
Y0 = cap["Y"].reshape(2, len(steps), Q, dim)
with torch.no_grad():
    A = torch.cat([z_cls, z_s], dim=1) + dec.pos_embed
    A_t = A[:, dec.steps]
    cls_rows = dec.step_cls.view(1, len(steps), 1, dim).expand(2, -1, -1, -1)
    ref = torch.cat([A_t.unsqueeze(2) + dec.query_base, cls_rows], dim=2).reshape(2, len(steps) * Q, dim)
assert torch.allclose(Y0.reshape(2, len(steps) * Q, dim), ref, atol=1e-6), \
    "zero step_embed must reproduce the pure arm② stack input exactly"

# (b) non-zero step_embed shifts each step's WHOLE Q-row block by exactly its own vector
with torch.no_grad():
    dec.step_embed.copy_(torch.arange(1, len(steps) + 1, dtype=torch.float32)
                         .view(-1, 1).expand(-1, dim))
    dec(z_cls, z_s)
Y = cap["Y"].reshape(2, len(steps), Q, dim)
ref4 = ref.reshape(2, len(steps), Q, dim)
expected = ref4 + dec.step_embed.view(1, len(steps), 1, dim)
assert torch.allclose(Y, expected, atol=1e-6), \
    "each step's whole Q-row block must shift by exactly its own step_embed"
assert not torch.allclose(Y[0, 0, N], ref4[0, 0, N]), \
    "step_embed must also reach the private [cls] row"

print("VERIFY_OK arm4: out=(2,5,36,256) per-step cls (5,256) + zero-init step_embed (5,256); "
      "zero-init == pure arm②; identity reaches all Q rows")
