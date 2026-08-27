"""
生成掩码机制可视化动图: mask_mechanism.gif
=============================================
展示 SR-Diffusion phase1 v2 块状注意力掩码机制:
  左上   ReEncoder 掩码(真实 N=576) + 前缀高亮框 + TextDecoder 掩码小图(示意)
  右上   FeatureDecoder 掩码(真实 N=576, 随 k 增长)
  左下   信息流示意(前缀链 + patch 枢纽, N=16 示意)
  右下   实测 L1(k) 曲线(3004 测试图, fp32) + 移动点

用法: python3 gen_mask_gif.py            # 生成 GIF
      PREVIEW=1 python3 gen_mask_gif.py  # 只输出若干帧 PNG 供检查
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.animation import FuncAnimation, PillowWriter

# ── 中文字体 ──
for fp in ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
           "/userdata/texlive/2026/texmf-dist/fonts/opentype/public/fandol/FandolHei-Regular.otf"):
    try:
        font_manager.fontManager.addfont(fp)
    except Exception:
        pass
plt.rcParams["font.family"] = "FandolHei"
plt.rcParams["axes.unicode_minus"] = False

# ── 真实掩码(复用模型实现, 保证与代码一致) ──
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_v2 import build_prefix_mask  # noqa: E402

N = 576

# ── 实测数据(2026-08-25, 3004 测试图, fp32, 输入 448x252) ──
anchor_k = np.array([0, 1, 2, 32, 64, 128, 256, 384, 512, 576], dtype=float)
anchor_l = np.array([0.00407, 0.00339, 0.0033, 0.0033, 0.0032,
                     0.0030, 0.0028, 0.0026, 0.0025, 0.00114], dtype=float)
x_anchor = np.log10(anchor_k + 1.0)          # k=0 → x=0, 避免 log(0)

def l1_at(k):
    return float(np.interp(np.log10(k + 1.0), x_anchor, anchor_l))

# ── 动画阶段: 2 帧/阶段 + 结尾 3 帧停留 ──
stages = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 576]
frame_k = []
for s in stages:
    frame_k += [s, s]
frame_k += [576, 576, 576]                    # 结尾定格 3 帧
STAGE_NOTES = {
    1: "k=1: 只有一个 z_s，decoder 从未见过小前缀",
    32: "平台期: k=1..32 L1≈0.0033（全量的 2.9×）——训练目标问题，非掩码失效",
    64: "k=64: 开始缓慢下降",
    576: "全量 k=576: L1=0.00114（训练时唯一见过的 k）",
}
FINAL_NOTES = [
    "掩码保证“前缀可计算/信息流结构”（必要）；",
    "不保证“前缀有信息/会用前缀”（不充分）——需训练目标配合；",
    "结论: 保留掩码 + 前缀加权训练(w(k)递减) 是配套补丁。",
]

# ── 掩码矩阵(真实) ──
re_mask = build_prefix_mask(2 * N + 1, 1, N + 1).numpy().astype(np.uint8)      # ReEncoder, 固定
text_mask_full = build_prefix_mask(N + 1 + 8, 0, N + 1, tail_causal=True,
                                   z_see_tail=False).numpy().astype(np.uint8)  # TextDecoder 示意(N=16? 见下)

# TextDecoder 用 N=16 小示意(真实序列太长且未训练)
TN = 16
text_mask = build_prefix_mask(TN + 1 + 8, 0, TN + 1, tail_causal=True,
                              z_see_tail=False).numpy().astype(np.uint8)

# ── 图形布局 ──
fig = plt.figure(figsize=(13.5, 8.2), dpi=95)
gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.42, wspace=0.30,
                      left=0.055, right=0.975, top=0.90, bottom=0.075)
ax_re = fig.add_subplot(gs[0, 0])
ax_fd = fig.add_subplot(gs[0, 1])
ax_sc = fig.add_subplot(gs[1, 0])
ax_l1 = fig.add_subplot(gs[1, 1])

# ── 1. ReEncoder 掩码 ──
im_re = ax_re.imshow(re_mask, cmap="gray_r", vmin=0, vmax=1,
                     interpolation="nearest", aspect="auto")
ax_re.set_title("ReEncoder 掩码 (N=576 真实)\nspecials 前缀链: special i 只见 specials≤i + 全部 patches",
                fontsize=9)
ax_re.set_xticks([]); ax_re.set_yticks([])
# 静态区域框
ax_re.add_patch(plt.Rectangle((0.5, 0.5), N - 0.5, N - 0.5, fill=False,
                              edgecolor="#2563eb", lw=1.0, alpha=0.8))
ax_re.text(0.02, 0.985, "specials 块", transform=ax_re.transAxes, fontsize=7,
           color="#2563eb", va="top")
# 前缀高亮框(动画)
re_hl, = ax_re.plot([], [], color="#dc2626", lw=1.6)

# 1b. TextDecoder 掩码小图(inset, N=16 示意)
ax_td = ax_re.inset_axes([0.50, 0.30, 0.46, 0.42])
ax_td.imshow(text_mask, cmap="gray_r", vmin=0, vmax=1,
             interpolation="nearest", aspect="auto")
ax_td.set_title("TextDecoder 掩码示意 (N=16):\nz 不看文字(防泄漏) + 文字下三角", fontsize=7)
ax_td.set_xticks([]); ax_td.set_yticks([])
ax_re.text(0.02, 0.60, "文字行: 见全部 z + 文字≤i", transform=ax_re.transAxes,
           fontsize=6.5, color="#374151")
ax_re.text(0.02, 0.545, "z 行: 因果链, 屏蔽全部文字", transform=ax_re.transAxes,
           fontsize=6.5, color="#374151")

# ── 2. FeatureDecoder 掩码(随 k 增长) ──
im_fd = ax_fd.imshow(np.zeros((2, 2), dtype=np.uint8), cmap="gray_r",
                     vmin=0, vmax=1, interpolation="nearest", aspect="auto")
fd_box = plt.Rectangle((-0.5, -0.5), 1, 1, fill=False,
                       edgecolor="#dc2626", lw=1.2, alpha=0.9)
ax_fd.add_patch(fd_box)
ax_fd.set_title("FeatureDecoder 掩码 (N=576 真实, k 动态)\n输入 [z_cls; z_s(k); patch×576]: z 因果链 + patch 全局",
                fontsize=9)
ax_fd.set_xticks([]); ax_fd.set_yticks([])
fd_note = ax_fd.text(0.5, -0.06, "", transform=ax_fd.transAxes, ha="center",
                     fontsize=8, color="#b91c1c")

# ── 3. 信息流示意 (N=16) ──
ax_sc.set_xlim(0, 1); ax_sc.set_ylim(0, 1); ax_sc.axis("off")
ax_sc.set_title("信息流示意: specials 前缀链 + patch 全局枢纽 (示意 N=16)",
                fontsize=9)
n_sc = 16
xs = np.linspace(0.06, 0.94, n_sc)
y_s = 0.60
y_c = 0.88            # z_cls
y_p = 0.18            # patch 块
# 静态底图: patch 块 + cls 节点
ax_sc.add_patch(FancyBboxPatch((0.30, y_p - 0.07), 0.40, 0.14,
                               boxstyle="round,pad=0.008",
                               fc="#e0f2fe", ec="#0284c7", lw=1.2))
ax_sc.text(0.5, y_p, "patches\n(全部 N 个, 全局双向)", ha="center", va="center",
           fontsize=7.5, color="#075985")
ax_sc.add_patch(plt.Circle((0.5, y_c), 0.024, fc="#fef3c7", ec="#d97706", lw=1.2))
ax_sc.text(0.5, y_c + 0.045, "z_cls", ha="center", fontsize=8, color="#92400e")
# 链边 + 双向 hub 边(静态, 颜色随帧更新)
chain_edges = []
hub_edges = []
cls_edges = []
for i in range(n_sc - 1):
    e = FancyArrowPatch((xs[i], y_s), (xs[i + 1], y_s), arrowstyle="-|>",
                        mutation_scale=8, lw=1.1, color="#9ca3af", alpha=0.85)
    ax_sc.add_patch(e); chain_edges.append(e)
for i in range(n_sc):
    h = FancyArrowPatch((0.5, y_p + 0.075), (xs[i], y_s - 0.03),
                        arrowstyle="<|-|>", mutation_scale=6, lw=0.8,
                        color="#cbd5e1", alpha=0.6)
    ax_sc.add_patch(h); hub_edges.append(h)
    c = FancyArrowPatch((0.5, y_c - 0.03), (xs[i], y_s + 0.03),
                        arrowstyle="-|>", mutation_scale=5, lw=0.7,
                        color="#d1d5db", alpha=0.7)
    ax_sc.add_patch(c); cls_edges.append(c)
sc_nodes = ax_sc.scatter(xs, np.full(n_sc, y_s), s=90, fc="#eef2ff",
                         ec="#6366f1", lw=1.4, zorder=5)
sc_ring, = ax_sc.plot([], [], "o", ms=17, mfc="none", mec="#dc2626", mew=2.4, zorder=6)
sc_note = ax_sc.text(0.5, -0.02, "", transform=ax_sc.transAxes, ha="center",
                     fontsize=8, color="#111827")

# ── 4. 实测 L1 曲线 ──
ax_l1.set_xscale("log")
xgrid = np.linspace(x_anchor[0], x_anchor[-1], 400)
lgrid = np.interp(xgrid, x_anchor, anchor_l)
ax_l1.plot(xgrid, lgrid, color="#0f766e", lw=1.8, zorder=2)
ax_l1.scatter(x_anchor, anchor_l, s=14, color="#0f766e", zorder=3)
ax_l1.axvspan(np.log10(2.0), np.log10(33.0), color="#fca5a5", alpha=0.25, zorder=1)
ax_l1.text(np.log10(2.0), 0.00402, "平台期 k∈[1,32]\n≈0.0033 (全量 2.9×)",
           fontsize=7.5, color="#b91c1c")
ax_l1.annotate("全量 k=576\n→ 0.00114", xy=(np.log10(577.0), 0.00114),
               xytext=(np.log10(40.0), 0.00145), fontsize=8, color="#047857",
               arrowprops=dict(arrowstyle="->", color="#047857", lw=1.0))
tick_k = [0, 1, 2, 8, 32, 128, 512, 576]
ax_l1.set_xticks([np.log10(k + 1) for k in tick_k])
ax_l1.set_xticklabels([str(k) for k in tick_k], fontsize=7)
ax_l1.set_xlabel("前缀长度 k (log 轴)", fontsize=8)
ax_l1.set_ylabel("L1(F_hat, patch)", fontsize=8)
ax_l1.set_title("实测: 还原误差 vs 前缀 k\n(3004 测试图, fp32 权重, 输入 448×252, N=576)",
                fontsize=9)
ax_l1.grid(alpha=0.3, lw=0.5)
dot, = ax_l1.plot([], [], "o", ms=9, color="#dc2626", zorder=5)
vline, = ax_l1.plot([], [], "--", color="#dc2626", lw=1.0, alpha=0.7, zorder=4)
l1_note = ax_l1.text(0.5, -0.14, "", transform=ax_l1.transAxes, ha="center",
                     fontsize=8, color="#111827")

supt = fig.suptitle("", fontsize=11, fontweight="bold", y=0.965)


def draw_frame(idx, rep=0):
    k = frame_k[idx]
    rep = idx % 2                                    # 每阶段 2 帧: 0=常规, 1=脉冲(标记放大)
    note = STAGE_NOTES.get(k, "")
    if idx >= len(frame_k) - len(FINAL_NOTES):
        note = FINAL_NOTES[idx - (len(frame_k) - len(FINAL_NOTES))]
    # ── suptitle ──
    l1k = l1_at(k)
    supt.set_text(f"SR-Diffusion phase1 v2 · 块状注意力掩码机制 · 前缀 k = {k} (k/N = {k/N:.1%}) · L1 = {l1k:.5f}")

    # ── ReEncoder 高亮框: specials 1..k ──
    if k >= 1:
        re_hl.set_data([0.5, 0.5, k + 0.5, k + 0.5, 0.5],
                       [0.5, k + 0.5, k + 0.5, 0.5, 0.5])
    else:
        re_hl.set_data([], [])

    # ── FeatureDecoder 掩码 ──
    fd_mask = build_prefix_mask(N + k + 1, 0, k + 1).numpy().astype(np.uint8)
    im_fd.set_data(fd_mask)
    im_fd.set_extent((-0.5, fd_mask.shape[1] - 0.5, fd_mask.shape[0] - 0.5, -0.5))
    ax_fd.set_xlim(-0.5, fd_mask.shape[1] - 0.5)
    ax_fd.set_ylim(fd_mask.shape[0] - 0.5, -0.5)
    fd_note.set_text(f"序列长度 = 1 + k + N = {N + k + 1}（z 区域 0..{k} 因果链）")
    fd_box.set_width(k + 1)
    fd_box.set_height(k + 1)

    # ── 信息流示意: 激活前缀 k_s = min(16, ceil(k/36)) ──
    k_s = int(min(16, max(1, int(np.ceil(k / 36.0)))))
    for i, e in enumerate(chain_edges):
        active = (i + 1) < k_s
        e.set_color("#dc2626" if active else "#9ca3af")
        e.set_alpha(1.0 if active else 0.6)
    for i, e in enumerate(hub_edges):
        active = i < k_s
        e.set_color("#dc2626" if active else "#cbd5e1")
        e.set_alpha(0.95 if active else 0.35)
        e.set_lw(1.1 if active else 0.8)
    for i, e in enumerate(cls_edges):
        active = i < k_s
        e.set_color("#dc2626" if active else "#d1d5db")
        e.set_alpha(0.95 if active else 0.5)
    alphas = [1.0 if i < k_s else 0.18 for i in range(n_sc)]
    sc_nodes.set_alpha(alphas)
    if k_s <= n_sc:
        sc_ring.set_data([xs[k_s - 1]], [y_s])
    sc_note.set_text(f"z_s[{k_s}] 可见 {{z_cls, z_s[1..{k_s}], 全部 patches}} —— 前缀自洽（直接路径）")

    # ── L1 曲线 ──
    xk = np.log10(k + 1.0)
    dot.set_data([xk], [l1k])
    vline.set_data([xk, xk], [0.0009, l1k])
    dot.set_markersize(9 + 4 * rep)                  # 脉冲: 偶数帧放大
    dot.set_markerfacecolor("#dc2626" if rep == 0 else "#f97316")
    note_default = f"k={k}: L1 = {l1k:.5f}（全量 {anchor_l[-1]:.5f} 的 {l1k / anchor_l[-1]:.1f}×）"
    ax_l1.set_ylim(0.0009, 0.0043)
    l1_note.set_text(note or note_default)

    return (im_re, im_fd, dot, vline, sc_ring, sc_note, fd_note, l1_note, re_hl, supt)


PREVIEW = os.environ.get("PREVIEW") == "1"
if PREVIEW:
    for idx in [0, 10, 20, 37, 38, 39, 40]:
        draw_frame(idx, idx % 2)
        fig.savefig(f"/tmp/mask_preview_{idx:02d}_k{frame_k[idx]}.png")
        print("saved preview frame", idx, "k=", frame_k[idx])
    print("PREVIEW done")
else:
    n_frames = len(frame_k)
    anim = FuncAnimation(fig, draw_frame, frames=range(n_frames), interval=300, blit=False)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mask_mechanism.gif")
    writer = PillowWriter(fps=1000 / 300, )
    anim.save(out, writer=writer)
    print("saved", out, "frames =", n_frames)
