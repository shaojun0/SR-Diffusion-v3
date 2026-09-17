# DESIGN_IMPL_gnn_schemeA — 方案A（GNN + 置换不变 Readout）在 SR-Diffusion-v3 的落地与实测

> 日期：2026-09-17 ｜ 分支：`feat-gnn-schemeA` ｜ 基线：`bptt@d76f0c5`（= `main@033cb27` + 循环 carry 不 detach，BPTT 口径）
> 设计文档：`feature/DESIGN_graph_embedding_schemeA.md`（方案A：GNN + 置换不变 Readout 的图级向量表示）
> 状态：**已落地 + 已实跑**（E2 置换不变性有数字；K=35 三臂 2000 步真实训练 + 全量 test 指标）。有明确边界与未完成项，见 §6。

---

## 1. 实现了什么（与设计文档的对应 / 偏离）

### 1.1 对应关系（逐步对照 §3 架构 + §10 E1/E3）

| 设计文档 | 本实现 | 位置 |
|---|---|---|
| §3/§11 Encoder：L 层消息传递 GNN（GCN/SAGE/GIN，hidden 256，L∈{2,3}） | 纯 PyTorch 三种算子：`GINLayer` / `GCNLayer` / `GraphSAGELayer`，默认 GIN、hid=256、L=2 | `model_gnn_schemeA.py` |
| §10 E1：DINOv2 patch 特征建 kNN 图（k 默认 8） | `knn_adj()`：余弦相似度 → topk → 对称化 → 自环；可选并入 patch 网格邻接（超像素/横纵亲和） | 同上 `knn_adj` |
| §6 Readout (a)：sum → **1 个向量 z** | `SchemeAEncoder.forward` 的 `out["z"]`，(B,1,D)，sum 后 LayerNorm → Projector → L2 归一化 | 同上 |
| §6 注 / §10 E3：attention **k 原型** → k 向量（k=K 与 special token 对齐） | `out["proto"]`，(B,K,D)：每原型一套打分 `a_m(h)=w_m·h`，softmax 对**节点维**，仍置换不变 | 同上 |
| §3 Projector：MLP → out_dim，L2 归一化（余弦空间） | `proj = Linear→GELU→Linear`，输出 `F.normalize` | 同上 |
| §4 置换不变性（结构性保证） | 全模块无位置编码/按行拼接/BatchNorm；归约只用 sum / softmax-attention / matmul；`Permute-Invariance` 有实测（§3） | 同上 + `tools/e2_permutation.py` |
| §10 E1：z 作为**额外全局 token** 注入 | `SRPhase1V2` 新增 `gnn_mode`/`gnn_inject`：图表示装进 **specials 槽位**后走原 `_dino_encode`（`[cls; z_slots; patches]`），与 register 完全同构；解码器循环/读窗口**未改一行** | `model_v2.py` `_encode_gnn` |
| §11 伪代码用 `torch_geometric` | **未装 PyG**（服务器无、不为此装包）：dense masked matmul（N=576 规模）+ `index_add_` 稀疏路径，自检 §E0 逐位核对等价 | `model_gnn_schemeA.py` |

### 1.2 自由发挥 / 与设计文档的偏离

1. **注入方式是"装填 specials 槽位"而不是"拼接新 token"**。设计文档说"与现有 `z_s` 并列或替换，二选一做成开关"，我实现成 `--gnn_inject {replace,concat}`：
   - `proto + replace`（推荐，E3 用法）：K 个原型**逐位占满** K 个 specials 槽位 ⇒ 序列长度 `1+K+N` 与 register 式**完全一致**，解码器读窗口（平方块切片）语义不变 ⇒ 与 `z_s` 路径是最干净的等位对照。
   - `sum + replace`：1 个全局向量占槽位 1，其余 `K−1` 个槽位回落到 `SpecialTokenBank`（保证末步读窗口 `hi = K−1…` 仍覆盖 max 采样步，不会把后段步饿死）。
   - `sum + concat`：保留 K 个 register，再追加 1 个全局向量（槽位 K+1）。`proto + concat` 被**显式拒绝**（K 个原型+K 个 register 会超出解码器读窗口，尾巴原型读不到，不如不提供这个会静默失效的组合）。
   - `off` 路径（默认）与历史 register 路径**逐位一致**（同一条 `_dino_encode`，只是 `z_slots` 来源不同）。
2. **GNN 阅读的是 DINO 末层 patch token**（`dinov2.encoder.layer` 24 层输出 + layernorm），不是 `embeddings()` 的浅层投影。理由：§7/§11 说"节点特征先归一化"，深层 token 语义更强；代价是图节点特征本身依赖 patch 顺序之外的信息（DINO 的位置编码仍在 `embeddings` 里，见 §6 边界 3）。
3. **多尺度聚合没做**（§5 提到的"各层 h 一起聚合"）：只用了末层。`SchemeAEncoder.forward` 返回 `out["h"]` 但不做多尺度拼接 —— 留作下一步（§7）。
4. **`--gnn_lr_scale` 优化器分组**：GNN 随机初始化、DINO 是预训练权重，同 lr 下有效步长不同；`train_v2.py` 的 `SRPhase1V2Trainer.create_optimizer` 把 `gnn.*` 单独一组乘 `--gnn_lr_scale`（默认 1.0 = 不改历史行为）。本轮实验全部用 1.0（未调）。
5. **E2 的对照基线做了加强**（设计文档只要求"序列式基线"）：见 §3.3，先试的 Transformer+PE 基线**几乎不漂移**（方向 cos≈0.9999998），不足以证明"顺序依赖"，于是补了唯一变量对照 `OrderSensitiveMLP`（+PE / −PE）。

### 1.3 接口（clone / 分支 / 文件 / CLI）

- clone：`/root/autodl-tmp/sr-diffusion-v3-gnnA`，分支 `feat-gnn-schemeA`，基线 `d76f0c5`，本次提交 **`793e99a`**（功能提交 `f24b863` + 工具提交 `793e99a`；**未 push**，工作区干净）。
- 新增 `model_gnn_schemeA.py`（≈510 行）：`knn_adj` / `sparse_message_passing` / `dense_adj_to_edge_index` / `permute_adj` / `GINLayer` / `GCNLayer` / `GraphSAGELayer` / `SchemeAEncoder` / `SeqBaseline` / `OrderSensitiveMLP`。
- 改 `model_v2.py`：`SRPhase1V2.__init__` 新增 `gnn_mode/gnn_inject/gnn_hid/gnn_layers/gnn_conv/gnn_k/gnn_grid/gnn_grid_weight/gnn_norm/gnn_dropout/gnn_lr_scale`；`encode()` → `_encode_gnn()`；自检 §7 新增方案A 集成回归。
- 改 `train_v2.py`：对应 CLI（`--gnn_*`）+ `model_info.json` 记录方案A 超参 + `create_optimizer` 的 GNN lr 分组。
- 改 `infer_v2_test.py`：`--gnn_*`，且以 `model_info.json` 优先恢复结构超参（方案A 的 K 必须由模型自己按步集推导，传 `--num_specials` 会被拒绝）。
- 新增工具：`tools/e2_permutation.py`（E2 实测）、`tools/e2_table.py`、`tools/compare_infer.py`、`tools/curve.py`。

主要 CLI（训练/推理同一套）：

```bash
# 方案A：k 原型（k=K）替换 specials 槽位 —— E3 用法
--gnn_mode proto --gnn_inject replace --gnn_hid 256 --gnn_layers 2 --gnn_conv gin --gnn_k 8
# 方案A：sum Readout 出 1 个全局向量, 替换槽位 1
--gnn_mode sum   --gnn_inject replace
# 方案A：sum Readout + 保留 K 个 register 并追加 1 个全局向量
--gnn_mode sum   --gnn_inject concat
# 关闭（默认, 与历史逐位一致）
--gnn_mode off
```

---

## 2. 冒烟 / 自检

在 clone 里实跑（`export PATH=/root/miniconda3/bin:$PATH; export HF_HUB_OFFLINE=1`）：

```
$ python model_gnn_schemeA.py
[ok] 形状: z(3, 1, 16) proto(3, 5, 16) （L2 归一化 ‖z‖=1.000000）
[ok] E0: dense matmul 与 index_add_ 稀疏路径逐位一致 (max|Δ|=0.00e+00, E=518 条边)
[ok] E2(schemeA, sum z, 20 置换): max|Δz|=1.192e-07 mean|Δz|=4.808e-08
[ok] E2(schemeA, k 原型多重集, 10 置换): max 最近邻距离=2.124e-07
[ok] E2 对照(OrderSensitiveMLP +PE, 10 置换): max|Δz|=4.527e-03 mean|Δz|=1.994e-03;  −PE 负对照 max|Δz|=1.118e-07
[ok] E2 对照(Transformer+PE 基线, 10 置换): max|Δz|=8.767e-01
ALL CHECKS PASSED (model_gnn_schemeA.py)

$ python model_v2.py            # 原有 6 段 + 新增 §7 方案A 集成
...
[ok] 损失口径: 直接预测 mean_t L1(PixelHead(Y_t), target)（无累加/cumsum）; ...
[ok] 方案A 集成: off 回归 / sum(replace:16槽) / sum(concat:17槽) / proto(16槽) 四条路径形状+梯度全通; 非法组合（proto+concat / 显式K / grid 尺寸）报错
[ok] 方案A 置换不变性回归: 5 随机置换 max|Δz|=8.94e-08

ALL CHECKS PASSED
```

新增回归断言覆盖：`gnn=off` 与历史同构；三条方案A 路径的 z_s 形状、整模型 loss、**GNN 四个子模块都收到非零梯度**；非法组合（`proto+concat`、方案A + 显式 `num_specials`、`gnn_grid` 尺寸与 N 不符）**直接报错而不是静默退化**。

---

## 3. E2：置换不变性实测（**核心**）

### 3.1 口径

- 数据：`/root/autodl-tmp/construction_site` 的 **test 图 8 张**（真实图像，经训练同款预处理 1600×900 画布 → 448×252）。
- 节点特征：**DINOv2-large 末层 patch token**（B,576,1024），fp32，`torch.no_grad`。
- 置换：① **split-half**（前半 288 个 patch 原位、后半 288 个随机打乱）——对"长序列+位置"最苛刻的设计；② S=8 次独立**全排列**取 mean/max。
- 模块：GIN、hid=256、L=2、k=8、K=35（= 训练的 K）。
- 两条 kNN 分支都测：(a) 用置换前算好的边按置换重排（图同构）；(b) 从**置换后的特征重新建 kNN 图**（kNN 本身也必须等变）。
- 度量：`max|Δ|`、`mean|Δ|`、`rel_fro=‖Δ‖_F/‖ref‖_F`、`cos`、**单位球 L2 距离**（Projector 输出是 L2 归一化的，下游是余弦空间，单位球距离才是任务口径）；原型是集合，用**双向最近邻**（多重集距离）。
- 脚本：`tools/e2_permutation.py`；原始 JSON：`/root/train_logs/e2_permutation_k8.json`（GPU0，约 1 分钟）。

### 3.2 数字（split-half 单次 + S=8 全排列 mean/max）

| 变体 | split-half max&#124;Δ&#124; | rel_fro | cos_min | unit-L2_max | S=8 mean / max（单位球 L2） |
|---|---|---|---|---|---|
| **方案A sum z**（重排边） | 2.98e-08 | 1.83e-07 | 1.00000000 | **1.92e-07** | 2.02e-07 / 2.08e-07 |
| **方案A sum z**（重建 kNN） | 2.98e-08 | 1.83e-07 | 1.00000000 | **1.92e-07** | 2.02e-07 / 2.08e-07 |
| **方案A k 原型（多重集）**（重排边） | 0.00e+00（Hausdorff） | 0.00e+00 | — | — | 2.16e-04 / 3.45e-04 |
| **方案A k 原型（多重集）**（重建 kNN） | 0.00e+00（Hausdorff） | 0.00e+00 | — | — | 2.16e-04 / 3.45e-04（同 left，原型集合逐位匹配） |
| 方案A 节点级 h（多重集，重建 kNN） | 1.10e-02（Hausdorff）；双向 NN mean 2.25e-03 | 0.00e+00（‖·‖ 相同） | — | — | 1.14e-02 / 1.24e-02 |
| 对照 A：OrderSensitiveMLP **+PE** | 1.96e-04 | 1.33e-03 | 0.99999869 | **1.61e-03** | 2.41e-03 / 2.47e-03 |
| 对照 A′：OrderSensitiveMLP **−PE**（负对照） | 3.73e-08 | 2.27e-07 | 0.99999994 | 2.59e-07 | 2.55e-07 / 2.75e-07 |
| 对照 B：Transformer+PE+sum 池化 | 2.52e+04 | 3.48e-01 | 0.99999982 | 5.93e-04 | 8.25e-04 / 9.01e-04 |
| 对照 C：DINOv2 **register 式 z_s**（现有管线） | 9.59e-08 | 6.44e-07 | 0.99999994 | **8.88e-07** | 9.86e-07 / 1.34e-06 |

**结论（诚实版）**

1. **方案A 的置换不变性成立，且是机器精度级**：sum 的 z 在 576 节点半图打乱、以及"从打乱后特征重建 kNN 图"两种分支下，单位球距离 ≤ **2.1e-07**（fp32 非结合性量级）；k 原型的**多重集**在 split-half 下**逐位相等**（Hausdorff=0），全排列下 ≤ 3.45e-04（因 `h_v` 本身是 1e-2 量级的浮点非结合噪声，见下一行）。
2. **节点级 h 是置换等变（多重集意义）**：Hausdorff 1.10e-02（相对 2.25e-03）——这是 `torch.bmm` / `index_add_` 求和顺序不同带来的浮点噪声，不是结构误差；它被 Readout 的全局求和**平均掉**，所以 z 又回到 1e-7。
3. **顺序敏感对照给出了 4 个数量级的差距**：同容量、同 sum Readout、只差"是否把可学习位置编码绑到序号上"的 `OrderSensitiveMLP`，+PE 的单位球漂移 1.61e-03，−PE（纯 DeepSets）2.59e-07 —— 漂移几乎全部来自 PE 项，证明方案A 的 `replace` 路径在**同一任务口径**下比序列式路径稳 3–4 个数量级。
4. **重要的反面发现（诚实边界）**：`Transformer + PE + sum 池化` 这条"序列式基线"**数值幅度**漂移巨大（rel_fro 0.35、max|Δ| 2.5e4），但 **L2 归一化后方向几乎不变**（cos_min 0.99999982，单位球距离 5.9e-04）；**现有 register 式 z_s 路径更稳**（单位球距离 8.9e-07，只比方案A 差 ~4.6 倍，远不是数量级）。原因：DINOv2 的注意力/FFN 对"给每个 patch 换一个位置编码"并不敏感（位置信息在 24 层里被高度混合），且 z_s 经 LayerNorm + 随机读出 MLP 后归一化。所以 —— **本实验不能证明"打乱 patch 顺序会让现有 z_s 表示显著漂移"**；方案A 的优势是**结构性保证**（对任意顺序、任意图同构都逐位不变），而不是"实测漂移更大所以更好"。
5. **正对照（不变性不是坍塌）**：跨图余弦 `schemeA_sum_z` mean 0.913 / min 0.824、节点级 h 0.483/0.366、`OrderSensitiveMLP`(±PE) 0.627/0.248、`order_mlp_nope` 0.627/0.248、`dino_registers_zs` 0.443/0.178（8 张图两两）⇒ 方案A 的 z **没有**退化成常数。相反 `schemeA_proto` 跨图余弦 **1.0000**（！）——随机初始化的 attention 原型对 576 个节点做加权平均后，各图的**方向**几乎相同（区分度≈0）；这解释了为什么 §4 里 proto 臂的重建改善全在"注入方式"而不是"图表示内容"。这是下一步必须修的（见 §7）。

---

## 4. 真实训练实验（BPTT 口径，K=35）

### 4.1 配置与口径

| 项 | 值 |
|---|---|
| 代码 | clone `sr-diffusion-v3-gnnA` @ `feat-gnn-schemeA` f24b863（基线 `bptt@d76f0c5`，**循环 carry 不 detach = BPTT**） |
| 数据 | `construction_site` 全量 7009 train / 3004 test；448×252（N=576，DINOv2-large 不冻结） |
| 切片/K | `--slice_start 0 --slice_end 5` ⇒ 采样步 `[1,4,9,16,25]`，K=35（`derive_num_specials` 自动） |
| 批量 | 单卡 `bs=16 × grad_accum 2`（等效 32）——**与锚点的单卡 bs32 等效批量一致**，但每步只算 16 张前向（显存保守） |
| 步数 | `--max_steps 2000`（**注意：锚点是 8760 步**，见 §4.3 口径警告） |
| 其它 | lr 1.5e-4、cosine、warmup 60、seed 42、fp32、无 bf16、`--num_workers 12` |
| 显存/速度 | 峰值 ~50 GB / 96 GB（bs16+ga2、fp32、BPTT）——无 OOM；~1.9 s/step（GNN）vs ~1.15 s/step（off） |
| GPU | off→GPU0、sum→GPU0、proto→GPU1（先 20 步 smoke 验证不 OOM 才放大） |

### 4.2 结果（全量 3004 test；`infer_v2_test.py`，bs16，非锚点的 bs32）

| 臂 | full_norm_l1 | 像素 L1 (0–255) | ±std | 每采样步像素 L1（步 1→5） |
|---|---|---|---|---|
| **off（register 基线，同代码同配置）** | 0.457595 | 26.2784 | 9.59 | 26.199 → 26.171 → 26.173 → 26.200 → **26.278** |
| **方案A sum**（1 个全局向量替换槽位 1） | **0.411752** | **23.6203** | 8.23 | 23.549 → 23.509 → 23.513 → 23.544 → **23.620** |
| **方案A proto**（K=35 原型占满槽位） | 0.418699 | 24.0418 | 8.24 | 23.961 → 23.942 → 23.958 → 23.991 → **24.042** |

- **相对 off 基线**：sum 臂 像素 L1 **−10.12%**（26.2784→23.6203）、norm L1 **−10.01%**；proto 臂 像素 L1 **−8.51%**、norm L1 **−8.50%**。同一 2000 步、同 seed、同数据、只差 `--gnn_mode`。
- **训练/eval 曲线**（每 500 步 eval，loss / recon，归一化空间）：

| 步 | off | sum | proto |
|---|---|---|---|
| 500 | 0.5364 / 0.5424 | 0.5312 / 0.5384 | 0.5297 / 0.5347 |
| 1000 | 0.5228 / 0.5242 | 0.4704 / 0.4720 | 0.4680 / 0.4711 |
| 1500 | 0.4681 / 0.4697 | 0.4346 / 0.4371 | 0.4299 / 0.4313 |
| 2000 | 0.4557 / 0.4575 | **0.4098 / 0.4116** | 0.4170 / 0.4186 |

  训练 loss（每 20 步记录）末段：off `0.4535`（步 2000）、sum `0.4066`、proto `0.4120`。三臂都在 2000 步处基本收敛到同一量级，proto 不再明显优于 sum（**说明 proto 的额外容量没有兑现**）。
- **每步增益形状没变**：三臂的 `步1 ≈ 步5`（差异 <0.4%），和文献里 2000 步时的"前重后轻"一致；方案A 没有改变"早期步基本定稿"这一现象。
- **推理耗时**：off 49.4 s / sum 73.6 s / proto 72.3 s（3004 图，GPU0/1）⇒ 方案A 让每图前向贵 ~50%（GNN 2 层 + 建图）。

### 4.3 口径警告（必须一起引用）

- 锚点 **K=35 单卡 bs32 14.1470 px / 0.246021** 是 **8760 步**的产物；本表是 **2000 步**（约 23% 训练量），所以绝对数字（26.28 / 0.4576）**不能**与 14.1470 直接比。
- 本表的 off 臂就是"同代码、同 2000 步、同 slice、同 seed、bs16+ga2"的**自建锚点**；三臂之间的差值（−10.1%）才是可比的。
- 尚未在本轮做：2000 步 checkpoint 上的 `bs32` 口径复测、8760 步满跑、K∈{48,99} 的 GNN 对照。

---

## 5. 复现命令

```bash
# ── 0) 隔离工作区（只读基线 → clone + 分支）──
cd /root/autodl-tmp
cp -a sr-diffusion-v3-bptt-ksweep sr-diffusion-v3-gnnA && cd sr-diffusion-v3-gnnA
git checkout -b feat-gnn-schemeA          # 基线 d76f0c5(bptt)
# 本报告对应的代码 = 该 clone 的 793e99a（= f24b863 功能提交 + 793e99a 工具提交；未 push）

# ── 1) 自检 ──
export PATH=/root/miniconda3/bin:$PATH; export HF_HUB_OFFLINE=1
python model_gnn_schemeA.py               # → ALL CHECKS PASSED
python model_v2.py                        # → ALL CHECKS PASSED（含 §7 方案A 集成）

# ── 2) E2 置换不变性（真实 DINO 特征, ~1 min/GPU）──
CUDA_VISIBLE_DEVICES=0 python tools/e2_permutation.py \
  --data_dir /root/autodl-tmp/construction_site \
  --dino_dir /root/autodl-tmp/models/dinov2-large \
  --n_images 8 --n_perm 8 --k 8 --num_proto 35 \
  --out /root/train_logs/e2_permutation_k8.json --device cuda:0
python tools/e2_table.py /root/train_logs/e2_permutation_k8.json

# ── 3) 2000 步三臂训练（单卡; off→GPU0, sum→GPU0, proto→GPU1）──
COMMON="--data_dir /root/autodl-tmp/construction_site \
  --dino_dir /root/autodl-tmp/models/dinov2-large \
  --slice_start 0 --slice_end 5 --batch_size 16 --grad_accum 2 \
  --num_workers 12 --max_steps 2000 --eval_every 500 --save_every 1000 --log_every 20"
CUDA_VISIBLE_DEVICES=0 python train_v2.py $COMMON --gnn_mode off \
  --output_dir /root/autodl-tmp/gnnA_off_K35_2000
CUDA_VISIBLE_DEVICES=0 python train_v2.py $COMMON --gnn_mode sum --gnn_inject replace \
  --gnn_hid 256 --gnn_layers 2 --gnn_conv gin --gnn_k 8 \
  --output_dir /root/autodl-tmp/gnnA_sum_K35_2000
CUDA_VISIBLE_DEVICES=1 python train_v2.py $COMMON --gnn_mode proto --gnn_inject replace \
  --gnn_hid 256 --gnn_layers 2 --gnn_conv gin --gnn_k 8 \
  --output_dir /root/autodl-tmp/gnnA_proto_K35_2000

# ── 4) 全量 test 指标（3004 图, bs16）──
CUDA_VISIBLE_DEVICES=0 python infer_v2_test.py \
  --data_dir /root/autodl-tmp/construction_site \
  --dino_dir /root/autodl-tmp/models/dinov2-large \
  --final_model /root/autodl-tmp/gnnA_{off,sum,proto}_K35_2000/final_model.pt \
  --output /root/train_logs/infer_gnnA_{off,sum,proto}.json \
  --model_input 448x252 --batch_size 16 --num_workers 12 --slice_start 0 --slice_end 5
python tools/compare_infer.py off sum proto

# 日志: /root/train_logs/gnnA_{off,sum,proto}_K35_2000.log
#       /root/train_logs/gnnA_e2_k8_v2.log, /root/train_logs/e2_permutation_k8.json
#       /root/train_logs/gnnA_infer.log, /root/train_logs/infer_gnnA_*.json
```

---

## 6. 诚实边界（没做成 / 可疑 / 与设计文档的差距）

1. **训练量不足**：只跑 2000/8760 步（锚点的 23%），绝对指标不能与 K-sweep 锚点比；且只做了 **K=35 一个点**，§10 E3 想让 sum 单向量与 k 原型"在重建任务上取舍"的完整结论需要至少 K∈{35,48,99} 的 GNN 臂。
2. **proto 臂的图表示"区分度≈0"**（跨图余弦 1.0000，未训练随机权重下）。它 2000 步的重建改善（−8.5%）与 sum（−10.1%）几乎同源，很可能主要来自"K 个可学习向量直接进 DINO 序列"这一注入方式，而不是 GNN 压出的图语义。**严格说：本轮没有证明"原型 Readout 比 sum 更好"**，甚至没有证明"图表示内容比随便一组可学习向量更好"——缺一个 **null 对照**：`--gnn_mode proto` 但把原型换成 `SpecialTokenBank`（即 register 式）在**同等槽位数 + 同等注入**下的对照，或 `gnn_layers=0`（只投影不传消息）的消融。
3. **"图片信息"仍经 DINO 的位置编码**：节点特征取 DINO 末层 patch token，其 `embeddings()` 里已经加了 patch 位置编码；`knn_adj` 只对**特征**做 kNN，所以严格讲方案A 的输入不是"纯内容多重集"。E2 测的是"同一个 patch 集合换编号顺序"下的不变性（这正是 §4 的声明），但**没有**测"换一个与位置无关的特征提取器"的更强口径。若要完全对齐 §4 的纯结构保证，应改用不含 PE 的 patch 特征（如 `embeddings.patch_embeddings(x)` 的卷积输出）。
4. **网格邻接（超像素/横纵亲和）实现了没测**：`--gnn_grid` / `--gnn_grid_weight` 已实现并入图，但本轮 `grid=None`（纯 kNN k=8）。E1 的"可选把坐标/超像素邻接并入"没做实验。
5. **GIN/GCN/SAGE 只跑了 GIN**：`--gnn_conv` 三选一都在自检里通过，训练只用 GIN（§5 说 GIN 可达 1-WL 上界，是默认首选）。
6. **§5 的多尺度读out、§7 的对比学习/边重建等自监督目标、§8 的采样式大图**都没做。
7. **一致性小瑕疵（日志卫生）**：`/root/train_logs/gnnA_proto_sum_K35_2000.log` 里混进了不止一条 run 的输出（多次 `方案A 开启` 段、eval 条目多于 4 条），其中包含一条**我没有完整追踪、也不清楚来源**的 proto 延伸 run（步数超出 2000，日志无 exit code；launcher wrapper 最终以 `EXIT_CODE=137` 被终止）。本报告里 proto 的**所有指标**都只取自 `gnnA_proto_K35_2000/final_model.pt`（run1 的 2000 步 checkpoint，mtime 16:18，对应 eval 0.4170/0.4186），**没有**采用那条延伸 run 的任何数字。复现时请按 §5 的命令各自重定向到独立日志，避免同名日志互相覆盖。

---

## 7. 下一步建议（按优先级）

1. **补 null 对照**（成本最低、信息量最大）：`register 同槽位 + 同注入` 与 `gnn_layers=0`/随机图（`--gnn_k` 换随机边）三档，回答"−10.1% 里有多少是图"。
2. **proto 诊断与修复**：跨图余弦 1.0 说明 attention 原型在随机初始化下坍塌；可加原型间的多样性/正交正则、把原型打分改成 `LayerNorm(h)` 上的 `w_m`、或对 `alpha` 做温度/熵约束；训练后重测跨图余弦。
3. **多尺度 Readout**（§5）：把各层 `h` 拼/加权后再 sum，看 E2 不变性是否仍成立（应当成立）+ 重建是否改善。
4. **跑满 8760 步 + bs32 口径**复现锚点口径，再报 K∈{48,99} 的 GNN 曲线（预计 sum 臂的 −10% 在长训后会收敛到更小，需要实测）。
5. **E1 文本侧**：把 `z`（或 K 原型）冻结后接 Qwen，比 Phase-2 文本质量——这才是验收口径。
6. **纯内容特征口径**：换成不含 DINO PE 的 patch 卷积特征重建 kNN 图，把 §4 的"无序"主张做到更严格。
