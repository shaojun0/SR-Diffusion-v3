"""Smoke test for model_phase1_v3.py (Lagrangian budget-constrained tokenizer).

Verifies:
  1. forward shapes on all paths (lagrangian / fixed-λ_rate / stage-1 / hard_mode)
  2. gradient coverage: every trainable param receives a finite gradient
  3. gate semantics: z-scored scores ⇒ k(τ) ≈ N·(1−Φ(τ)) at τ ∈ {−1.8, 0, 1.8}
  4. k hard guard [k_min, k_max]
  5. Lagrangian dynamics (unit): λ grows over-budget / decays under-budget / clamps
  6. target anneal: R_target_eff: 1.0 → rate_target over anneal_steps
  7. textbook dual ascent: min ½(x−5)² s.t. x ≤ 3 → x*=3, λ*=2 (KKT)
  8. λ is a state_dict buffer (resume keeps the multiplier)
  9. eval mode freezes the dual update (λ unchanged)
 10. init_reencoder_from_dino with a real transformers Dinov2Model
"""
import torch
import torch.nn as nn
from types import SimpleNamespace

import model_phase1_v3 as m3


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


def check_grads(model, tag, exclude=()):
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and not n.startswith(exclude) and p.grad is None]
    assert not missing, f"[{tag}] params without grad: {missing[:5]}"
    nans = [n for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith(exclude)
            and p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not nans, f"[{tag}] NaN grads: {nans[:5]}"
    print(f"  [ok] {tag}: all {sum(1 for p in model.parameters() if p.requires_grad)} "
          f"trainable params have finite grads")


def test_textbook_dual_ascent():
    """min ½(x−5)² s.t. x ≤ 3 → KKT: x* = 3, λ* = 2."""
    x = torch.tensor(0.0, requires_grad=True)
    lam = torch.tensor(0.0)
    eta_x, eta_l = 0.1, 0.05
    for _ in range(8000):
        loss = 0.5 * (x - 5) ** 2 + lam * (x - 3)       # L = f + λ·g, g = x−3 ≤ 0
        loss.backward()
        with torch.no_grad():
            x.sub_(eta_x * x.grad)
            x.grad = None
            lam = torch.clamp(lam + eta_l * (x - 3), 0.0, 10.0)   # 对偶上升
    assert abs(x.item() - 3.0) < 0.05, f"x*={x.item():.4f} != 3"
    assert abs(lam.item() - 2.0) < 0.3, f"λ*={lam.item():.4f} != 2"
    print(f"  [ok] textbook dual ascent: x*={x.item():.4f} (expect 3), "
          f"λ*={lam.item():.4f} (expect 2) — 拉格朗日方法收敛到 KKT 点")


def test_lagrangian_budget_unit():
    torch.manual_seed(0)
    lag = m3.LagrangianBudget(rate_target=0.25, lambda_init=0.1, eta_lambda=0.05,
                              lambda_max=10.0, rho=1.0, ema_alpha=0.05,
                              anneal_steps=1)           # anneal off: target=0.25 恒定

    # over-budget: R=0.5 持续超预算 → λ 增长（封顶 10）
    lag.reset()
    for _ in range(3000):
        lag.tick()
        t = lag.current_target()
        lag.dual_step(torch.tensor(0.5), t)
    assert lag.lambda_.item() >= 9.5, f"λ should clamp at λ_max, got {lag.lambda_.item():.3f}"
    assert lag.rate_ema.item() > 0.45, f"R_ema should track R_hard≈0.5, got {lag.rate_ema.item():.3f}"
    print(f"  [ok] over-budget: λ → λ_max={lag.lambda_.item():.3f} (clamped), "
          f"R_ema={lag.rate_ema.item():.3f}")

    # under-budget: R=0.1 < 0.25 → λ 衰减到 λ_min=0
    lag.reset()
    for _ in range(3000):
        lag.tick()
        t = lag.current_target()
        lag.dual_step(torch.tensor(0.1), t)
    assert lag.lambda_.item() <= 0.01, f"λ should decay to 0 under-budget, got {lag.lambda_.item():.3f}"
    print(f"  [ok] under-budget: λ → {lag.lambda_.item():.4f} (decayed, 预算未超则乘子归零)")

    # anneal: R_target_eff 从 1.0 退火到 0.25
    lag2 = m3.LagrangianBudget(rate_target=0.25, anneal_steps=1000)
    lag2.reset()
    lag2.tick()
    assert abs(lag2.current_target() - 1.0) < 1e-3, lag2.current_target()
    for _ in range(999):
        lag2.tick()
    assert abs(lag2.current_target() - 0.25) < 1e-3, lag2.current_target()
    lag2.tick()
    assert lag2.current_target() == 0.25
    print(f"  [ok] target anneal: 1.0 → {lag2.current_target():.3f} over {lag2.anneal_steps} steps")


def main():
    torch.manual_seed(0)
    B, N, D = 2, 256, 192
    dino = FakeDino(D)
    x = torch.randn(B, 3, 224, 224)

    # ── 0. textbook dual ascent（拉格朗日方法的正确性根基）──
    print("[0] textbook dual ascent")
    test_textbook_dual_ascent()

    # ── 1. lagrangian budget 单元测试 ──
    print("[1] LagrangianBudget unit")
    test_lagrangian_budget_unit()

    # ── 2. forward: lagrangian path (stage-2, train) ──
    print("[2] lagrangian forward (select_on=zs)")
    lag = m3.LagrangianBudget(rate_target=0.25, lambda_init=0.1, eta_lambda=0.05,
                              rho=1.0, anneal_steps=2000)
    model = m3.SRPhase1V3(dino, num_patches=N, dim=D, T=1.0, lagrangian=lag,
                          lambda_ent=0.01)
    model.train()
    out = model(x)
    assert out["F_hat"].shape == (B, N, D), out["F_hat"].shape
    assert out["mask"].shape == (B, N) and out["p"].shape == (B, N)
    s = out["stats"]
    print(f"  [ok] loss={out['loss'].item():.4f} recon={s['recon_l1'].item():.4f} "
          f"rate={s['rate'].item():.3f} λ={s['lambda'].item():.3f} "
          f"k={s['k_used_mean'].item():.0f}/{N}")
    out["loss"].backward()
    check_grads(model, "lagrangian path")
    # λ 是 buffer: 不进优化器参数列表
    assert all("lagrangian" not in n for n, _ in model.named_parameters()), \
        "λ must not be an optimizer parameter"
    print("  [ok] λ/rate_ema/step are buffers, not parameters")
    # 拉格朗日惩罚项的梯度确实流到 rate_head（τ）与 score_head
    g_rate = model.rate_head.mlp[3].weight.grad
    assert g_rate is not None and g_rate.abs().sum().item() > 0, "τ 路径无梯度"
    print(f"  [ok] rate_head grad (τ 路径) L1={g_rate.abs().sum().item():.6f} > 0")

    # ── 3. select_on=cls path ──
    print("[3] forward (select_on=cls)")
    model.zero_grad()
    model2 = m3.SRPhase1V3(dino, num_patches=N, dim=D, T=1.0, select_on="cls",
                          lagrangian=m3.LagrangianBudget(rate_target=0.25))
    model2.train()
    out2 = model2(x)
    assert out2["F_hat"].shape == (B, N, D)
    out2["loss"].backward()
    check_grads(model2, "select_on=cls")

    # ── 4. stage-1 fixed_tau（纯重建，率项关闭）──
    print("[4] stage-1 fixed_tau")
    model.zero_grad()
    model.set_stage(1)
    assert model.rate_term_on is False and model.fixed_tau == -2.0
    out = model(x)
    assert out["stats"]["rate"].item() > 0.9, f"stage-1 k should be ≈ all, got {out['stats']['rate'].item():.3f}"
    out["loss"].backward()
    # stage-1 不训练 rate_head（τ 固定）——只查其余参数
    check_grads(model, "stage-1 (rate_head excluded)", exclude="rate_head")
    # stage-1 不 tick 对偶步
    step0 = int(model.lagrangian.step.item())
    model(x)
    assert int(model.lagrangian.step.item()) == step0, "stage-1 must not advance dual step"
    print(f"  [ok] stage-1: rate={out['stats']['rate'].item():.3f} (≈1.0), dual step frozen")

    # ── 5. gate semantics: k(τ) ≈ N·(1−Φ(τ)) on z-scored scores ──
    print("[5] gate semantics vs τ (z-scored scores)")
    model.set_stage(2)
    model.train()
    model.rate_head.mlp[3].weight.data.zero_()
    model.rate_head.mlp[3].bias.data.zero_()            # τ=0 → k≈N/2
    B8, N8 = 8, 256
    x8 = torch.randn(B8, 3, 224, 224)
    for tau_val, lo, hi in [(-1.8, 0.90, 1.00), (0.0, 0.42, 0.58), (1.8, 0.005, 0.08)]:
        model.fixed_tau = tau_val
        with torch.no_grad():
            out = model(x8)
        r = out["stats"]["rate"].item()
        assert lo <= r <= hi, f"τ={tau_val}: rate={r:.4f} outside [{lo},{hi}]"
        print(f"  [ok] τ={tau_val:+.1f} → rate={r:.4f} (expect ≈ {1 - 0.5 * (1 + torch.erf(torch.tensor(tau_val) / 2**0.5)).item():.3f})")
    model.fixed_tau = None

    # ── 6. k hard guard ──
    print("[6] k guard [8, 100]")
    model.zero_grad()
    model.k_min, model.k_max = 8, 100
    model.train()
    for _ in range(3):
        out = model(x8)
        kmin, kmax = out["stats"]["k_used_min"].item(), out["stats"]["k_used_max"].item()
        assert 8 <= kmin and kmax <= 100, (kmin, kmax)
    print(f"  [ok] k ∈ [{kmin:.0f}, {kmax:.0f}] within [8, 100]")
    model.k_min, model.k_max = 0, None

    # ── 7. hard_mode inference ──
    print("[7] hard_mode (physical pruning)")
    model.eval()
    out = model(x8, hard_mode=True)
    assert out["F_hat"].shape == (B8, N8, D)
    print(f"  [ok] hard_mode: k={out['stats']['k_used_mean'].item():.0f}, "
          f"F_hat={tuple(out['F_hat'].shape)}")

    # ── 8. eval 冻结对偶更新 ──
    print("[8] eval freezes dual update")
    lam_before = float(model.lagrangian.lambda_.item())
    model(x8)
    model(x8)
    assert float(model.lagrangian.lambda_.item()) == lam_before, "eval must not move λ"
    print(f"  [ok] λ unchanged in eval ({lam_before:.4f})")

    # ── 9. state_dict carries λ (resume keeps multiplier) ──
    print("[9] λ in state_dict")
    sd = model.state_dict()
    for k in ("lagrangian.lambda_", "lagrangian.rate_ema", "lagrangian.step"):
        assert k in sd, f"missing {k}"
    model.train()
    lam0 = float(sd["lagrangian.lambda_"].item())
    m3b = m3.SRPhase1V3(dino, num_patches=N, dim=D,
                        lagrangian=m3.LagrangianBudget(rate_target=0.25))
    m3b.load_state_dict(sd)
    assert float(m3b.lagrangian.lambda_.item()) == lam0
    print(f"  [ok] λ={lam0:.4f} round-trips through state_dict")

    # ── 10. fixed λ_rate ablation path ──
    print("[10] fixed λ_rate path (Pareto-frontier ablation)")
    model.zero_grad()
    mfix = m3.SRPhase1V3(dino, num_patches=N, dim=D, lagrangian=None,
                         lambda_rate=0.5, lambda_ent=0.01)
    mfix.train()
    out = mfix(x)
    out["loss"].backward()
    check_grads(mfix, "fixed λ_rate")
    print(f"  [ok] fixed λ_rate: loss={out['loss'].item():.4f} rate={out['stats']['rate'].item():.3f}")

    # ── 11. real transformers Dinov2Model warm-start ──
    print("[11] init_reencoder_from_dino (real transformers)")
    try:
        from transformers import Dinov2Config, Dinov2Model
        cfg = Dinov2Config(hidden_size=D, num_hidden_layers=6,
                           num_attention_heads=6, intermediate_size=768,
                           patch_size=14, image_size=224)
        real_dino = Dinov2Model(cfg)
        m3c = m3.SRPhase1V3(real_dino, num_patches=N, dim=D,
                            lagrangian=m3.LagrangianBudget(rate_target=0.25))
        m3c.init_reencoder_from_dino(num_layers=4)
        m3c.train()
        out = m3c(x)
        out["loss"].backward()
        check_grads(m3c, "real-DINO + warm-start")
        print(f"  [ok] real-DINO: loss={out['loss'].item():.4f}")
    except ImportError as e:
        print(f"  [skip] transformers not available: {e}")

    # ── 12. 端到端预算控制环: 拉格朗日对偶上升把 k/N 拉到 R_target ──
    # 隔离控制环（无 transformer）: recon = L1(mask·s_gt, s_gt)，每个 token
    # 等权重 → 约束 R ≤ R_target 必须恰好绑定。验证: k/N → R_target 且
    # λ 稳定在 KKT 乘子附近（= 每个 token 的边际还原代价）。
    print("[12] budget control loop: dual ascent drives k/N → R_target")
    torch.manual_seed(0)
    Bc, Nc, Tc = 8, 128, 0.25
    s_gt = torch.randn(Bc, Nc) * 2 + 1.0
    scores = torch.nn.Parameter(torch.randn(Bc, Nc) * 0.5)
    tau_p = torch.nn.Parameter(torch.zeros(Bc, 1))
    lag_c = m3.LagrangianBudget(rate_target=Tc, lambda_init=0.1, eta_lambda=0.05,
                                rho=2.0, anneal_steps=200)
    gate_c = m3.SigmoidGate(T=1.0)
    opt_c = torch.optim.AdamW([scores, tau_p], lr=1e-2)
    for _ in range(600):
        opt_c.zero_grad()
        mask_c, _ = gate_c(scores, tau_p)
        recon_c = torch.nn.functional.l1_loss(mask_c * s_gt, s_gt)
        R_c = mask_c.mean()
        lag_c.tick()
        tg = lag_c.current_target()
        pen_c = lag_c.penalty(R_c, tg)
        lag_c.dual_step(R_c, tg)
        (recon_c + pen_c).backward()
        opt_c.step()
    r_c = (mask_c > 0.5).float().mean().item()
    assert abs(r_c - Tc) < 0.06, f"budget loop rate={r_c:.3f} != target {Tc}"
    assert lag_c.lambda_.item() > 0.5, f"λ should settle at a positive KKT value, got {lag_c.lambda_.item():.3f}"
    print(f"  [ok] k/N={r_c:.3f} → R_target={Tc}, λ={lag_c.lambda_.item():.3f} (KKT 乘子稳定)")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
