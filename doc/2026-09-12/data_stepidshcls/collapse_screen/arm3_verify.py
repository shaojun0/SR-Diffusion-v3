#!/usr/bin/env python3
"""Self-check for arm (3): shapes / masks / gradients / shared-[cls] semantics.

Usage: python arm3_verify.py <repo_dir>
"""
import importlib.util
import sys

import torch

repo = sys.argv[1]
spec = importlib.util.spec_from_file_location("m3", repo + "/model_v2.py")
m3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m3)

dim, N, K = 256, 36, 35
steps = [1, 4, 9, 16, 25]
dec = m3.OutputQueryDecoder(dim=dim, num_patches=N, num_specials=K, steps=steps,
                            heads=8, depth=2, stack_dim=None, dropout=0.0, mlp_ratio=4.0)
dec.eval()
torch.manual_seed(0)
z_cls = torch.randn(2, 1, dim)
z_s = torch.randn(2, K, dim)
Q = N + 1

out = dec(z_cls, z_s)
assert tuple(out.shape) == (2, 5, N, dim), out.shape
assert tuple(dec.last_step_cls.shape) == (2, 5, dim), dec.last_step_cls.shape
assert tuple(dec.attn_mask.shape) == (5 * Q, K + 1), dec.attn_mask.shape
assert tuple(dec.tgt_mask.shape) == (5 * Q, 5 * Q), dec.tgt_mask.shape

# blockdiag isolation (true for both arm② and arm③)
NEG = float("-inf")
assert dec.tgt_mask[0, Q] == NEG            # step0 patch row cannot see step1
assert dec.tgt_mask[0, 0] == 0.0            # sees own patch row 0
assert dec.tgt_mask[0, N] == 0.0            # sees own block's [cls]
assert dec.tgt_mask[Q, 0] == NEG            # step1 cannot see step0
assert dec.tgt_mask[N, 0] == 0.0            # step0 [cls] sees its own patch rows
assert dec.tgt_mask[N, Q] == NEG            # step0 [cls] must not see step1 rows
assert dec.tgt_mask[N, N] == 0.0            # step0 [cls] sees itself

# parameters
assert tuple(dec.step_cls.shape) == (1, dim), dec.step_cls.shape
assert tuple(dec.step_embed.shape) == (5, dim), dec.step_embed.shape
assert torch.equal(dec.step_embed, torch.zeros(5, dim)), "step_embed must be zero-init"
assert all(dec.attn_mask[i].shape == (K + 1,) for i in range(5 * Q))

# gradients reach both new parameters
loss = out.pow(2).mean() + dec.last_step_cls.pow(2).mean()
loss.backward()
ge = dec.step_embed.grad
gc = dec.step_cls.grad
assert ge is not None and ge.abs().sum() > 0, "step_embed got no gradient"
assert gc is not None and gc.abs().sum() > 0, "step_cls got no gradient"
print(f"grad_norm step_embed={ge.norm().item():.6f} step_cls={gc.norm().item():.6f}")

# shared-[cls] semantics: capture the tensor fed to self.stack and check that
# the [cls] input row is literally identical across the 5 steps, while the patch
# rows carry the per-step identity.
captured = {}


def pre_hook(module, args, kwargs):
    captured["Y"] = args[0].detach().clone()
    return None


handle = dec.stack.register_forward_pre_hook(pre_hook, with_kwargs=True)
with torch.no_grad():
    ident = torch.arange(1, len(steps) + 1, dtype=torch.float32)
    dec.step_embed.copy_(ident.view(-1, 1).expand(-1, dim))
    dec(z_cls, z_s)
Y = captured["Y"].reshape(2, len(steps), Q, dim)
patch_rows = Y[:, :, :N, :]
cls_rows = Y[:, :, N, :]
assert torch.allclose(cls_rows, cls_rows[:, :1].expand_as(cls_rows)), \
    "shared [cls] input rows must be identical across steps"
assert not torch.allclose(patch_rows[:, 0], patch_rows[:, 1]), \
    "patch rows must differ per step (step_embed)"
# [cls] input must be unaffected by step_embed: identical to the raw parameter
assert torch.allclose(cls_rows[0, 0], dec.step_cls[0], atol=1e-6), \
    "step_embed must not be added to the [cls] row"

# zero step_embed ⇒ the tensor fed to self.stack is identical to the plain
# no-identity (shared-[cls]) construction ⇒ zero-init keeps the default path
with torch.no_grad():
    dec.step_embed.zero_()
    dec(z_cls, z_s)
Y0 = captured["Y"].reshape(2, len(steps), Q, dim)
with torch.no_grad():
    A = torch.cat([z_cls, z_s], dim=1) + dec.pos_embed
    A_t = A[:, dec.steps]
    ref_patch = A_t.unsqueeze(2) + dec.query_base            # (B,|T|,N,dim)
    ref_cls = dec.step_cls.view(1, 1, dim).expand(2, len(steps), dim)
assert torch.allclose(Y0[:, :, :N], ref_patch, atol=1e-6), \
    "zero step_embed must not perturb the patch rows"
assert torch.allclose(Y0[:, :, N], ref_cls, atol=1e-6), \
    "zero step_embed must not perturb the shared [cls] row"

print("VERIFY_OK shape=(2,5,36,256) Q=37 attn=(185,36) tgt=(185,185) "
      "shared_cls=True step_embed_zero_init=True")
