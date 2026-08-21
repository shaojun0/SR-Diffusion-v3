"""Smoke test for the modified model_phase1.py (special-token selection v2).

Verifies:
  1. forward shapes on all three gate paths (fixed_tau / trust-region / plain)
  2. hard_mode (inference pruning) shapes
  3. gradient coverage: every trainable param receives a gradient
  4. special tokens are identical across positions except PE (input property)
  5. init_reencoder_from_dino works with transformers 4.57.x Dinov2Model layout
"""
import torch
import torch.nn as nn
from types import SimpleNamespace

import model_phase1 as mp


class FakeDino(nn.Module):
    """Fake frozen DINO: returns random last_hidden_state (B, 257, D)."""
    def __init__(self, dim=192):
        super().__init__()
        self.dim = dim
        self.config = SimpleNamespace(hidden_size=dim, patch_size=14)

    def forward(self, x):
        B = x.shape[0]
        feats = torch.randn(B, 257, self.dim, device=x.device)
        return SimpleNamespace(last_hidden_state=feats)


def check_grads(model, tag):
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not missing, f"[{tag}] params without grad: {missing[:5]}"
    nans = [n for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None
            and not torch.isfinite(p.grad).all()]
    assert not nans, f"[{tag}] NaN grads: {nans[:5]}"
    print(f"  [ok] {tag}: all {sum(1 for p in model.parameters() if p.requires_grad)} "
          f"trainable params have finite grads")


def main():
    torch.manual_seed(0)
    B, N, D = 2, 256, 192

    # ── 1. fake-DINO forward, three gate paths ──
    dino = FakeDino(D)
    model = mp.SRPhase1(dino, num_patches=N, dim=D, T=1.0,
                        lambda_rate=0.1, lambda_ent=0.01)
    x = torch.randn(B, 3, 224, 224)

    # plain path (train)
    model.train()
    out = model(x)
    assert out["F_hat"].shape == (B, N, D), out["F_hat"].shape
    assert out["mask"].shape == (B, N), out["mask"].shape
    print(f"[ok] plain path: loss={out['loss'].item():.4f} "
          f"k_used={out['stats']['k_used_mean'].item():.1f} "
          f"F_hat={tuple(out['F_hat'].shape)}")
    out["loss"].backward()
    check_grads(model, "plain path")

    # special-token input property: identical across positions except pos
    sp = model.special_bank(B, x.device)
    diff_pos = (sp[:, 0] - sp[:, 1]).abs().max().item()
    assert diff_pos > 0, "special tokens should differ across positions (PE)"
    # same position across images must be identical too (shared token + same PE)
    same_pos_across_batch = (sp[0] - sp[1]).abs().max().item()
    assert same_pos_across_batch == 0.0, "special token input must be identical across images"
    print(f"[ok] special tokens: identical across images (max diff {same_pos_across_batch}), "
          f"differ by PE (max diff {diff_pos:.4f})")

    # ── 2. stage-1 fixed_tau + hard_mode ──
    model.zero_grad()
    model.set_stage(1)
    model.eval()
    out = model(x, hard_mode=True)
    assert out["F_hat"].shape == (B, N, D)
    print(f"[ok] stage-1 + hard_mode: k_used={out['stats']['k_used_mean'].item():.1f} "
          f"(should be ~250-256 with tau=-2.0)")
    model.set_stage(2)

    # ── 3. trust-region path ──
    model.zero_grad()
    model.train()
    btr = mp.BudgetTrustRegion(n=N, k_min=8, k_max=250)
    model.enable_trust_region(btr)
    model.set_stage(2)
    model.gate.T = 1.0
    out = model(x)
    assert out["F_hat"].shape == (B, N, D)
    assert "kl" in out["stats"], "trust-region extras should be in stats"
    print(f"[ok] trust-region path: loss={out['loss'].item():.4f} "
          f"k_used={out['stats']['k_used_mean'].item():.1f} "
          f"tau={out['stats']['tau_mean'].item():.3f} kl={out['stats']['kl']:.4f}")
    out["loss"].backward()
    check_grads(model, "trust-region path")

    # trust-region + hard_mode (inference)
    model.eval()
    out = model(x, hard_mode=True)
    assert out["F_hat"].shape == (B, N, D)
    print(f"[ok] trust-region + hard_mode: k_used={out['stats']['k_used_mean'].item():.1f}")

    # ── 4. init_reencoder_from_dino with a real transformers Dinov2Model ──
    try:
        from transformers import Dinov2Config, Dinov2Model
        cfg = Dinov2Config(hidden_size=D, num_hidden_layers=6,
                           num_attention_heads=6, intermediate_size=768,
                           patch_size=14, image_size=224)
        real_dino = Dinov2Model(cfg)
        model2 = mp.SRPhase1(real_dino, num_patches=N, dim=D, T=1.0)
        model2.init_reencoder_from_dino(num_layers=4)
        model2.train()
        out2 = model2(x)
        out2["loss"].backward()
        check_grads(model2, "real-DINO + warm-start")
        print(f"[ok] init_reencoder_from_dino: loss={out2['loss'].item():.4f}")
    except ImportError as e:
        print(f"[skip] transformers not available: {e}")

    # ── 5. batch with different per-image k (hard_mode gather) ──
    model.eval()
    # force a mask that selects a small subset to exercise _gather_selected
    hard = torch.zeros(B, N)
    hard[0, :10] = 1.0
    hard[1, :20] = 1.0
    sel, lengths = model._gather_selected(torch.randn(B, N, D), hard)
    assert sel.shape == (B, 20, D) and lengths.tolist() == [10, 20], (sel.shape, lengths)
    print(f"[ok] _gather_selected: padded to max_k={sel.shape[1]}, "
          f"lengths={lengths.tolist()}")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
