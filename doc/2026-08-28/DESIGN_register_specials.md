# register_specials — specials 合并进 DINOv2（修 F1/F2，2026-08-28）

> 状态：已实现（`model_v2.py` / `train_v2.py` / `infer_v2_test.py` / `visualize_recon_pixel.py`），待服务器冒烟 + 全量训练验证。
> 依据：`DIAGNOSIS_clarity.md` 根因 1（F1-F4 逐 patch 信息路由缺陷）+ 用户拍板"第 3 档：字面合并进 DINO（register 式）"。

---

## 1. 动机（对应 F1/F2）

- **F1**：旧架构里 special token k 的输入是 `token + pos[k]`（全位置共享 + 位置编码），**不含 patch k 的任何内容**——`z_s[k]` 想带 patch k 信息只能靠 4 层 ReEncoder 在 1153 长序列上学"注意力路由"，是硬学习问题。
- **F2**：ReEncoder 的 specials 行看全部 patches（全局）+ 前缀链 → 576 个 `z_s` 全是"位置标签化的全局摘要"，彼此冗余 → 解码器键集合无信息梯度 → **t≥1 平台**（t=1: 22.93 vs t=576: 22.72，只差 0.21/12.9）。
- 修复方向（本改动）：**specials 作为额外 token 直接拼进 DINOv2 输入序列**，由 DINO 的 24 层（304M）直接算出 `z_s`——深层网络做内容路由，绕开"浅层学 576 路路由"。

## 2. 机制（register token 式，Darcet et al. "Vision Transformers Need Registers"）

```
输入 x (B,3,H,W)
  → dinov2.embeddings(x)                 (B,1+N,D)  [cls; patches] + PE（复用 HF 逻辑, 含自动插值）
  → specials = SpecialTokenBank(B)       (B,N,D)    共享 token + 逐位置可学习 pos（与旧架构同参）
  → seq = [cls; specials; patches]       (B,2N+1,D)  ← specials 拼在 cls 之后、patches 之前
  → DINO 24 层（全双向注意力, 无掩码）     (B,2N+1,D)
  → layernorm → z_cls = seq[:,0], z_s = seq[:,1:1+N]
  → OutputQueryDecoder + PixelHead（与旧架构完全相同, 未动）
```

- **无 ReEncoder**（省 51.6M 参数；条件初始化，避免 DDP `find_unused_parameters=False` 报未用参数）。
- **DINO 内全双向（无掩码）**：HF `Dinov2Model` 没有 token 级注意力 mask API（文件头踩坑记录），且重建任务无时序因果需求；"前缀稳定性"约束本就是为渐进压缩虚构的，渐进语义由解码器的 **KV 因果**提供，与编码器无关。
- **解码器（F3/F4）未动**：保持"一次只改一处"。键（z_s）变有内容后，固定模板 `query_base[k]` 的"选择"才变得有意义——预期 t≥1 平台消失、曲线变阶梯（t=1 粗 / t=576 细）。若 E1 显示信息仍丢在 Decoder，再上 P1-1（逐 patch 内容查询）。

## 3. 代码改动

| 文件 | 改动 |
|---|---|
| `model_v2.py` | `SRPhase1V2(..., register_specials=False)`：条件化 ReEncoder；forward 重构为 `encode()`（按模式分派）+ `decode()`（共享尾，输出新增 `Y_pix`/`target_pix`，供推理/可视化/探针复用同一路径）；自检新增 register 模式检查（形状/梯度/eval） |
| `train_v2.py` | `--register_specials` 开关；`model_info.json` 记录 |
| `infer_v2_test.py` | `--register_specials`；手动复刻的 forward 改为调 `model(x)` |
| `visualize_recon_pixel.py` | 同上 |
| `doc/2026-08-27/trace_info_pixel.py` | 改用 `model.encode()`（E1 探针两种模式通用） |

## 4. 预期与验证（判据）

| 指标 | 旧（ReEncoder, K=N） | 预期（register, K=N） |
|---|---|---|
| 全量像素 L1 (0-255) | 23.41 | **≤16**（可回收 ~12.9 的大头） |
| t≥1 渐进曲线 | 平台（极差 0.21） | **阶梯**（短前缀粗、长前缀细） |
| E1 探针 `z_s`→像素线性 L1 | ~23（丢在编码路由） | **~9.8-13**（键有内容了） |
| 边缘比（邻域差/原图） | ~1/3 | 60%+ |

**验证步骤**：
1. 冒烟：`--register_specials --max_steps 500` 单卡跑通，看 loss 趋势与显存。
2. 全量重训（40 epochs）：预期时长 ~2h → **~5-6h**（DINO 跑 1153 token，注意力 ~4×）。
3. `infer_v2_test.py --register_specials`：全量 L1 + 渐进曲线。
4. `trace_info_pixel.py`：z_s / F_hat 线性解码 L1，定位剩余差距。
5. `visualize_recon_pixel.py --register_specials`：肉眼确认。

## 5. 风险与观察点

- **重新塌缩为全局摘要**：全双向下每个 z_s[k] 都能见全部 patch，若解码器选择压力不足，键可能又趋同（平台复现）。观察 t≥1 曲线即可判定；若复现，下一步是 P1-1（查询侧注入内容）。
- **训练变慢**：DINO 注意力 ~4×，整步 ~2-3×；显存仍充裕（97GB/卡）。
- **与 K 压缩的关系**：本改动仍是 K=N=576（零压缩）。K 压缩（32/64/128）是 Phase 1 核心验收（`GOAL_compression_for_nlp.md`），本改动先验证"键有内容"，压缩实验在其后或并行设计（如 special 数 < N 时 decoder 查询基行数不变，N 行查询从 K 个键读出）。
- 若 register 模式有效，**ReEncoder 整条链路可退役**（含 causal_specials 语义），Phase 2 冻结 DINO+specials → z_s → Qwen 的桥接也少一环。

## 6. 相关文件

- `model_v2.py`（register 分支：`_encode_register`；文件头"register_specials=True"段落）
- `DIAGNOSIS_clarity.md` §3.4.3 / §5 P1 / §6 E1（诊断依据与探针协议）
- `GOAL_compression_for_nlp.md` §2（K 压缩 × 重建质量验收；纹理清晰度非目标）
