#!/usr/bin/env python3
"""compare_layer_tap.py — tap（逐层金字塔）vs 基线（平方块末层）逐 step 对照

两臂除 --layer_tap 外完全相同: construction_site test 3,004 / 224x126 /
K=144 / 12 步 / d4 / BPTT / 8,760 步 / 同 seed。故差异可归因于"向量来自哪一层"。
"""
import json

BASE = "/root/autodl-tmp/cot_l1/eval_224_d4/test_224_d4_bptt.json"
TAP = "/root/autodl-tmp/cot_l1/eval_224_d4_tap/test_224_d4_bptt_tap.json"
B1K = "/root/autodl-tmp/cot_l1/eval_224_d4/train1k_224_d4_bptt.json"
T1K = "/root/autodl-tmp/cot_l1/eval_224_d4_tap/train1k_224_d4_bptt_tap.json"
b, t = json.load(open(BASE)), json.load(open(TAP))
b1, t1 = json.load(open(B1K)), json.load(open(T1K))

print("## 总量（test 3,004）\n")
print("| 量 | 基线（平方块/末层） | tap（逐层金字塔） | Δ (tap−base) |")
print("|---|---|---|---|")
for k, lab in (("full_norm_l1", "full_norm_l1"),
               ("full_pixel_l1_255", "像素 L1 (0-255)"),
               ("full_pixel_std_255", "像素 L1 std")):
    print(f"| {lab} | {b[k]:.4f} | {t[k]:.4f} | {t[k] - b[k]:+.4f} |")
print(f"| layer_tap | {b.get('layer_tap')} | {t.get('layer_tap')} | |")
print(f"| step_tokens | {b['step_tokens']} | {t['step_tokens']} | |")
print(f"| step_bpp | {[round(x, 3) for x in b['step_bpp_beta1']]} | {[round(x, 3) for x in t['step_bpp_beta1']]} | |")

print("\n## 逐 step（test 3,004）\n")
print("| t | base tok | base bpp | tap tok | tap bpp | base L1 | tap L1 | base PSNR | tap PSNR "
      "| base MS-SSIM | tap MS-SSIM | base L1内容 | tap L1内容 | base PSNR内容 | tap PSNR内容 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for i, tt in enumerate(t["decoder_steps"]):
    print(f"| {tt} | {b['step_tokens'][i]} | {b['step_bpp_beta1'][i]:.3f} "
          f"| {t['step_tokens'][i]} | {t['step_bpp_beta1'][i]:.3f} "
          f"| {b['step_pixel_l1_255'][i]:.3f} | {t['step_pixel_l1_255'][i]:.3f} "
          f"| {b['step_psnr'][i]:.2f} | {t['step_psnr'][i]:.2f} "
          f"| {b['step_ms_ssim'][i]:.4f} | {t['step_ms_ssim'][i]:.4f} "
          f"| {b['step_l1_content_255'][i]:.3f} | {t['step_l1_content_255'][i]:.3f} "
          f"| {b['step_psnr_content'][i]:.2f} | {t['step_psnr_content'][i]:.2f} |")

bp = min(range(len(b["step_psnr"])), key=lambda i: b["step_pixel_l1_255"][i])
tp = min(range(len(t["step_psnr"])), key=lambda i: t["step_pixel_l1_255"][i])
print(f"\n- 基线最优: t={b['decoder_steps'][bp]} → L1 {b['step_pixel_l1_255'][bp]:.3f} "
      f"/ PSNR {b['step_psnr'][bp]:.2f} / MS-SSIM {b['step_ms_ssim'][bp]:.4f}")
print(f"- tap 最优: t={t['decoder_steps'][tp]} → L1 {t['step_pixel_l1_255'][tp]:.3f} "
      f"/ PSNR {t['step_psnr'][tp]:.2f} / MS-SSIM {t['step_ms_ssim'][tp]:.4f}")
print(f"- 基线跨度: L1 {max(b['step_pixel_l1_255']) - min(b['step_pixel_l1_255']):.3f} px "
      f"/ PSNR {max(b['step_psnr']) - min(b['step_psnr']):.2f} dB")
print(f"- tap  跨度: L1 {max(t['step_pixel_l1_255']) - min(t['step_pixel_l1_255']):.3f} px "
      f"/ PSNR {max(t['step_psnr']) - min(t['step_psnr']):.2f} dB")

print("\n## train 1,000 抽样（过拟合检查）\n")
print("| 臂 | test L1 | train1k L1 | Δ(train−test) |")
print("|---|---|---|---|")
print(f"| 基线 | {b['full_pixel_l1_255']:.2f} | {b1['full_pixel_l1_255']:.2f} "
      f"| {b1['full_pixel_l1_255'] - b['full_pixel_l1_255']:+.2f} |")
print(f"| tap | {t['full_pixel_l1_255']:.2f} | {t1['full_pixel_l1_255']:.2f} "
      f"| {t1['full_pixel_l1_255'] - t['full_pixel_l1_255']:+.2f} |")


# ── 可选: 画两联图（bpp→PSNR ; step→PSNR/L1）──
def _plot(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8))
    ax1.plot(b["step_bpp_beta1"], b["step_psnr"], "-o", ms=4, color="tab:gray",
             label=f"base 224x126 (square block, final layer)  best {max(b['step_psnr']):.2f} dB")
    ax1.plot(t["step_bpp_beta1"], t["step_psnr"], "-o", ms=4, color="tab:red",
             label=f"TAP 224x126 (layer pyramid)  best {max(t['step_psnr']):.2f} dB")
    ax1.set_xscale("log")
    ax1.set_xlabel("bpp (beta=1 estimated)")
    ax1.set_ylabel("PSNR (dB)")
    ax1.set_title("RD: layer_tap vs baseline (construction_site test, n=3004)")
    ax1.grid(alpha=.3, which="both")
    ax1.legend(fontsize=8)

    x = list(range(1, len(t["decoder_steps"]) + 1))
    ax2.plot(x, b["step_psnr"], "-o", ms=4, color="tab:gray", label="base PSNR (square)")
    ax2.plot(x, t["step_psnr"], "-o", ms=4, color="tab:red", label="TAP PSNR (pyramid)")
    ax2.set_ylabel("PSNR (dB)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_xlabel("decoder step i  (group i = layers 24-2i, 25-2i)")
    ax2.set_xticks(x)
    ax2b = ax2.twinx()
    ax2b.plot(x, b["step_pixel_l1_255"], "--s", ms=4, color="tab:blue", label="base L1")
    ax2b.plot(x, t["step_pixel_l1_255"], "--s", ms=4, color="tab:green", label="TAP L1")
    ax2b.set_ylabel("pixel L1 (0-255)", color="tab:blue")
    ax2b.tick_params(axis="y", labelcolor="tab:blue")
    ax2.set_title("progressive curve: base is a FLAT line, TAP has a real slope")
    ax2.grid(alpha=.3)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[plot] {path}")


if __name__ == "__main__":
    import sys
    if "--plot" in sys.argv:
        _plot(sys.argv[sys.argv.index("--plot") + 1])

