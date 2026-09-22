# DESIGN — DINOv2 逐层 tap（金字塔读出）: 12 步 ↔ 24 层的深浅分工（2026-09-22）

> 需求（用户原话）：「DINOv2-large 一共有 24 层，而我们有 12 个 step（上一个实验），
> 我们刚好每 2 层一个 step，我们从后往前数，最顶层就是 1 个向量，然后其次层是 3 个向量，
> 一直往后推。这些层分别进行迭代交叉注意力跑 BPTT……类似于 yolo，浅层特征做细节描绘，
> 深层特征负责语义构建。用掉的向量不进入下一层。」
>
> 立项动机（仓库内诊断）：[`ANALYSIS_texture_vs_decoder_routing.md`](ANALYSIS_texture_vs_decoder_routing.md)
> §1 —— 当前代码**只用末层**（`model_v2.py:_encode_register` 跑完 24 层 → `layernorm` →
> 取 `z_cls/z_s`），**没有任何浅层 tap**；若纹理真在浅层，我们现在是 100% 丢弃。

---

## 0. 一句话

把 **144 个 register 按深度分给 12 个采样步**：从顶往下第 g 组（= 第 `24−2g`、`24−2g+1` 层）
读出 `2g−1` 个向量（1, 3, 5, …, 23，合计 144 = K = S²），**读出即从序列里删掉**
（"用掉的向量不进入下一层"）；解码器第 i 步只读第 i 组那 `2i−1` 个向量跑交叉注意力，
步间 carry 仍走 **BPTT** ⇒ 顶部少量向量走满 24 层（语义）、底部大量向量只走 2 层（细节）。

**与默认口径的唯一差别是"每个 step 读的那批向量来自哪一层"**；读窗口宽度序列
（1, 3, 5, …）、K=144、步集 `[1,4,…,144]`、损失、BPTT、bpp 轴全部不变 ⇒ 与
`out_224_d4_bptt`（平方块读窗口）构成**单变量对照**。

---

## 1. 口径（2026-09-22 用户四问四答确认）

| 问题 | 结论 |
|---|---|
| 结构 | **12 步 × 2 层**，从顶到底一一对应；向量数 **1,3,5,…,23**，合计 **144 = 12² = K** |
| "向量"是什么 | **special/register token**（现在 `z_s` 的那 144 个位置），不是 patch token、不是池化向量 |
| "用掉的不进入下一层" | **一开始全部插进输入序列**；在每个组的边界把该组读出并**从序列里删除**，剩下的继续往深层走 |
| 交付范围 | 改完代码**直接在服务器上启动训练**（224×126、12 步、BPTT 配方） |

```
DINOv2-large 24 层                    解码器 12 步（从深到浅）
 层 22,23 ──┐ 读出 1 个  ──────────►  step 1 : 1 个向量   （语义／全局）
 层 20,21 ──┤ 读出 3 个  ──────────►  step 2 : 3 个向量
 层 18,19 ──┤ 读出 5 个  ──────────►  step 3 : 5 个向量
    ⋮       │                            ⋮
 层  2, 3 ──┤ 读出 21 个 ──────────►  step 11: 21 个向量
 层  0, 1 ──┘ 读出 23 个 ──────────►  step 12: 23 个向量  （细节／高频）
        每组读出后**立即从序列删除**（后续层看不到它）
```

- 第 g 组占 `z_s[(g−1)² : g²]`，由 `encoder.layer[24−2g]`、`[24−2g+1]` 算出
  （g 从顶往下数；g=12 → 层 0/1）。
- 解码器第 i 步（1-based）读窗口（A 坐标，`A = [z_cls; z_s]`）= `[1+(i−1)², i²]`；
  **本模式不读 `z_cls`**（金字塔口径顶部恰好 1 个向量）。`z_cls` 仍由 `encode()` 返回，
  解码器不使用（API 保持不变）。
- 越靠顶的 register 被删得越晚 ⇒ 走过的层越多 ⇒ 越语义；越靠底的组只走 2 层 ⇒ 越局部。
  这正是"浅层做细节、深层做语义"的落法（对照 YOLO/FPN，但方向一致：深 = 少 token + 强语义）。

---

## 2. 实现（全部在 `model_v2.py`，默认关，向后兼容）

| 位置 | 改动 |
|---|---|
| `layer_tap_groups(steps, num_specials, num_layers=None)` | **新增**：校验 `steps == [1,4,…,S²]`、`K == S²`、`num_layers == 2S`（不合法清晰报错——两模式**权重形状完全相同**，静默跑错不会报错只会算错） |
| `OutputQueryDecoder.__init__` | 新增 `layer_tap: bool = False`；新增 `last_windows`（本次实际读窗口，诊断/自检用） |
| `OutputQueryDecoder.forward` | 读窗口二选一：`layer_tap` → `(1+(i−1)², i²)`；否则 → 历史平方块 `(0/(k²), min((k+1)²−1, K))`。循环/carry/BPTT 逻辑**逐位不变** |
| `SRPhase1V2.__init__` | 新增 `layer_tap`；构造前校验层数 = 2×步数；记录 `tap_groups`；透传给 decoder |
| `SRPhase1V2._encode_register` | 新增 tap 分支：逐层跑，`idx%2==1` 时 `g = S − idx//2`，`taps[g] = layernorm(seq[:, 1+(g−1)² : 1+g²])`，随后 `seq = cat([seq[:, :lo], seq[:, hi:]])` 删除；最后按 G1..GS 拼 `z_s` |
| `train_v2.py` | 新增 `--layer_tap`；透传模型；`model_info.json` 增写 `layer_tap` / `tap_groups`（**必须写**，否则推理侧无法区分） |
| `infer_v2_test.py` | 新增 `--layer_tap` 兜底；`_pick("layer_tap", …)` **以 `model_info.json` 为准**；bpp 轴 `tokens = t + tok_off`（`tok_off = 0` tap / `1` 平方块，因 tap 不读 `z_cls`）；输出 json 增 `layer_tap` |

**设计取舍（写清边界）**：

1. **中间层也套 `dinov2.layernorm`**（末层那个）。理由：各 tap 的尺度口径一致，
   且与历史 `z_s` 分布一致。代价：层归一化参数在浅层语义下被复用（不是独立 LN）。
2. **`z_cls` 在 tap 模式不参与解码**：用户口径是顶部恰好 1 个向量；`z_cls` 与 G1 都走过
   24 层，语义冗余。它仍返回（不破坏 `encode()` 签名）；`cls_token` 参数照样收梯度
   （它仍参与其它 token 的注意力）。
3. **不新增任何参数**：`layer_tap` 是读窗口口径开关，`state_dict` 与默认模式**逐 key
   逐形状相同** ⇒ 旧权重 strict load **不会报错**。这是本改动**最大的风险点**，故：
   `model_info.json` 记录 + 推理侧强制对齐 + 自检里的"非法配置报错"。

---

## 3. 验证（`python model_v2.py`，本地 torch 2.8.0+cpu 与服务器均通过）

| 复核项 | 做法 | 结果 |
|---|---|---|
| 读窗口宽度 | S=2 fake（4 层 / N=4） | `[(1,1),(2,4)]` = 1, 3 个向量；非 tap 同配置 = `[(0,3),(4,4)]`（含 z_cls） |
| 目标配置 | **24 层 / 224×126（144 patches）/ 12 步** | `K=144`, `tap_groups=12`, 窗口 = `[1,3,5,7,9,11,13,15,17,19,21,23]`，无缝覆盖 `z_s[0..143]` 且无重叠（每个 register 恰读 1 次） |
| "读出即删"（前向） | 每层输入序列长度钩子 | `289→289→266→266→245→245→…→146→146`，收尾 `145`（= cls + patches）；即从底往顶每组按 `2g−1` 递减 |
| 真的 tap 中间层 | 共享同一编码器权重的 tap/非 tap 两模型 | `z_s` 不同（否则就是末层读出） |
| "读出即删"（反向, 最强判据） | `carry_detach=True` + 逐步损失反传 | 底组（第 2 步）损失梯度**只回层 0/1** `[6.85, 4.86, 0.0, 0.0]`；顶组（第 1 步）走满全栈 `[3.23, 2.51, 2.01, 1.87]` |
| 梯度 | 整模型 backward | DINO 浅/深层 + `special_bank` + 解码器 + PixelHead 全通 |
| 非法配置 | 步集非平方 / K≠S² / 层数≠2S | 三种都报含 `layer_tap` 的清晰错误（不静默退化） |
| 真实权重冒烟 | 服务器单卡 `--layer_tap --smoke --limit 64 --max_steps 4` | `patches=144, specials=144, 12 步`；打印 `layer_tap=ON: 24 层 = 12 组 × 2 层`；train→eval→`final_model.pt` 全通 |
| 推理链路 | 冒烟模型 + `--limit 16`（**命令行不传 `--layer_tap`**） | 自动从 `model_info.json` 读到 `layer_tap=True`，打印"读窗口口径: layer_tap=True"，bpp 轴 `tokens=t+0` |
| 代码一致 | 本地 md5 ↔ 服务器 md5 | `model_v2.py 64fed531…` / `train_v2.py eeea8d20…` / `infer_v2_test.py 14421642…` 全等 |
| **权重形状**（真实 DINOv2-large） | tap / 默认两模型 `state_dict()` 与参数量对比 | **518 keys 全等 / 375,302,732 参数全等** ⇒ "静默跑错不会报错"的风险真实存在, `model_info.json` 必写 |

---

## 4. 本次实验（已启动）

| 项 | 值 |
|---|---|
| 代码 | 服务器 `/root/autodl-tmp/sr-diffusion-v3-tap/`（本地 main 打包，排除 `doc/ .git/ output/ logs/`） |
| 数据 | `construction_site`（train **7,009** / test **3,004**），与基线同 |
| 输入 | **224×126** ⇒ patches 16×9=**144**；specials K=**144** |
| 采样步 | **12 步** `[1,4,9,16,25,36,49,64,81,100,121,144]` |
| 解码器 | depth=**4**，heads=8，mlp_ratio=4，无加宽（stack_dim=dim=1024） |
| carry | **BPTT**（`carry_detach=False`，仓库当前默认） |
| 预算 | 40 epoch = **8,760 步**；2 卡 × bs16（全局 32）；lr 1.5e-4 cosine + warmup 3% |
| 启动 | 2026-09-22 **17:40 CST**，实测 ≈**2.7 it/s** ⇒ **ETA ≈55 min** |
| 显存 | 两卡各 **23.3 GB**（基线 26.6 GB；tap 逐组删 token ⇒ 编码器序列 289→145 递减，省显存） |
| 产物 | `/root/autodl-tmp/construction_site/out_224_d4_bptt_tap/`（ckpt 每 2,000 步 + `final_model.pt` + `model_info.json`） |
| 日志 | `/root/train_logs/tap_224_d4_bptt.log` |
| 看护 | PID 35110（`ppid=1`，setsid），日志 `/root/train_logs/tap_eval_watcher.log`；训练退出后自动跑 test 3,004 + train 1,000 + JPEG/WebP 基线 |
| 评测产物 | `/root/autodl-tmp/cot_l1/eval_224_d4_tap/` |
| 复跑 | `NUM_GPUS=2 bash tools/run_224_d4_bptt_tap.sh`（脚本内已固定全部超参） |

**对照基线**：`/root/autodl-tmp/construction_site/out_224_d4_bptt/`（同配方、**平方块读窗口 /
末层 z_s**），test 3,004 → L1 **12.02** / PSNR **21.35** / 内容区 PSNR 20.27；
`t=1`（2 token, 0.073 bpp）就 21.06 dB，`t=144`（145 token, 5.26 bpp）21.36 dB
——**72 倍码率换 +0.30 dB**（[`REPORT_batch0_rd_earlystop.md`](REPORT_batch0_rd_earlystop.md) §10）。

⚠️ **bpp 轴口径变化**：基线按 `(t+1)` 个 token 计（首步另读 `z_cls`），tap 模式实际只读
`t` 个 ⇒ 本次 json 的 `step_tokens = t`。画 RD 曲线时两臂必须各用自己的 `step_bpp_beta1`
（已写进 json），**不要**把 `(t+1)` 硬编码到汇总脚本里。

---

## 5. 怎么看结果（判据）

1. **主判据**：test 3,004 的 `full_pixel_l1_255` / PSNR vs 基线 **12.02 / 21.35**。
   若 tap 真把"浅层纹理"接回来，最该动的是 **`step_l1_content_255`（内容区，尤其高频）
   与 MS-SSIM**，而不是全画布 L1（画布 24% 是 padding）。
2. **逐 step 曲线形状**：基线 12 步是"平线"（21.06→21.36）。tap 的语义是"早步语义、
   末步细节"，若成立，曲线**跨度应变大**、且早步（少向量、语义）不该崩。
3. **数据量守恒**：tap 与基线**每步读的 token 数相同**（t 个），所以任何差异都归因于
   "向量来自哪一层"，**不是** token 预算 ⇒ 这是本设计最干净的地方。
4. **别过度解读**：本轮只是 224×126 / 8,760 步 / 单 seed；1 组对照不构成结论。

---

## 6. 后续（若本轮正向）

1. **深度分配消融**：把"1,3,5,…,23 从顶到底"换成反向（顶部 23）、或均匀（12 层×12 层），
   各跑 8,760 步即可分辨"深浅分工"是否真的重要（改 `layer_tap_groups` 的组宽公式 + 读窗口）。
2. **tap 位置消融**：只在 4 个深度抽 4 组 × (2 组宽) 等。
3. **P1-1 逐 patch 内容查询**（[`ANALYSIS_texture_vs_decoder_routing.md`](ANALYSIS_texture_vs_decoder_routing.md) §6）：
   与 tap 正交且排名更高，应当并行推进。
4. **接回 CoT 68k 数据线**：`--layer_tap` 与数据规模无关，可直接套用。
