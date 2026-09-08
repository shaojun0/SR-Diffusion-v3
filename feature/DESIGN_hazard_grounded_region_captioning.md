# DESIGN 方案：文本定位隐患区域 + 特定部位看图说话（Grounded Region Captioning, GRC）

> 状态：**设计提案**（未在仓库落地；待评审后分期实施，本文 §6 为分期路线）。
> 需求来源（2026-09）："**我们可以通过文本去定位图片中的隐患位置，以精确地进行特定部位的看图说话，以泛化到没有标注隐患位置的图片进行看图说话。**"
> 性质：feature 级方案。与 [`DESIGN_graph_embedding_schemeA.md`](DESIGN_graph_embedding_schemeA.md)（图嵌入，无关正交）并列存放于 `feature/`。
> 数据前提：**方案同时覆盖 P0（训练集无任何隐患位置标注）与 P1（仅部分位置标注）**，实际按数据现状走数据引擎两档配置（§4.1），不阻塞。
> 目标口径（重要，防误读）：本文"看图说话"指 **Phase-2 文字生成**（权威定义见 [`doc/2026-08-28/GOAL_compression_for_nlp.md`](../doc/2026-08-28/GOAL_compression_for_nlp.md)：冻结编码器 → MLP → Qwen → 中文工地描述/隐患）。"特定部位的看图说话" = 对**被文本指出的隐患区域**单独生成描述（region/grounded captioning），是现有"整图描述"能力的新增形态；"整图看图说话"可视为"全图=一个大区域"的退化情形。
> **关键机制（2026-09 追加）：所有标注框（含训练伪框与推理出框）由 Agent Loop 给出**——propose→read→verify→refine 循环收敛后的产物，而非任何模型的一次性输出（协议见 §4.1）。

---

## 0. 一句话目标

**让"看图说话"从'整图一句话'升级为'指哪说哪'：用户用一句话（隐患描述/类别）提问 → 系统经 Agent Loop 在图中定位该隐患的像素区域（框由 loop 收敛给出，可自我校验）→ 再针对该区域精确描述（类型/数量/状态/相对位置），且整个过程在推理时不需要任何人工框标注——训练期伪标注与推理期定位共用同一条 Agent Loop，天然泛化到没有标注隐患位置的图片。**

产品侧最终输出建议格式（与研究计划"隐患类型+位置+严重度+处置建议"结构化输出对齐）：

```
隐患：未佩戴安全帽的作业人员
位置：(0.31, 0.42, 0.38, 0.55)（画面中上部脚手架二层）
描述：一名工人未戴安全帽站在脚手架上作业，帽色缺失，面部可见……
严重度：高    建议：立即纠正并复查该作业面
```

## 1. 需求拆解：把一句话需求变成 3 个可单独验收的能力

| 需求子句 | 能力（技术名） | 用户可见行为 | 现有差距 |
|---|---|---|---|
| **用文本定位图片中的隐患位置** | 指代定位 / 短语定位（referring expression localization / phrase grounding） | 输入"未盖板的井口"→ 返回框/区域 | 仓库无定位模块、无语言解码器 |
| **精确地进行特定部位的看图说话** | 区域级看图说话（region / grounded captioning） | 对定位到的区域给出**针对该区域**的细描述，不张冠李戴 | 仅规划过整图描述（Phase 2），无区域通道 |
| **泛化到没有标注隐患位置的图片** | 无框泛化（annotation-free grounding）：训练期 Agent Loop 伪标注自举 + 推理期同一 loop 文本驱动、开放词表 | 任意新图，用户直接问隐患，无需先给框 | 数据无框列；需数据引擎与开放词表定位 |

三者的依赖链：**定位是看图说话的"目光"，看图说话是定位的"读法"，无框泛化决定两者在部署时都不依赖人工框。** 割裂做任何一个都会回到"整图一句话、说不准位置"的老路。

> 适用边界（先声明再展开）：本 feature 负责**"可被文本定位、可被一个框/区域容纳"的实体类隐患**（裂纹/腐蚀/泄漏/异物/井口无盖/未戴安全帽等）。图级环境属性、全局关系/秩序、以及依赖规范知识推断的隐患（如光线昏暗、摆放杂乱、工人撸起袖子）**不满足该前提**，纯语义理解识别不了，需整图/规则通道兜底——详见 §9 局限性。

## 2. 现状盘点与差距（以 2026-09-07 的 main 为准）

### 2.1 代码现状（能复用什么）

| 资产 | 位置 | 与本 feature 的关系 |
|---|---|---|
| v2/v4 压缩编码器（DINOv2+register `z_s`），像素重建脚手架 | main `model_v2.py` / test `model_v4.py` | Phase-2 冻结编码器的**待接入对象**（GOAL 范式）；也提供"压缩是否保位置"的**待探针对象**（§4.4） |
| 解码器 `OutputQueryDecoder`：查询行 `k ↔ patch k` 逐 patch 对齐，`h (B,N,D)` 是逐 patch 的 | `model_v2.py` | 天然的空间坐标轴——"区域读出"可挂在 `h` 的行的子集上 |
| DINOv2 patch 特征 `P (B,N,D)`，N=576=32×18 网格（patch 14，输入 448×252） | 编码器出口 | Agent Loop 中 propose 步所需的**像素级对齐面**（文本↔patch 对齐 = 开放词表初始候选基座） |
| v1 范式 `SR-Qwen-VL v11`：SVD→DINO→MLP Projector→Qwen→Text | test 分支 `model.py` / `train.py` | Phase-2 接入范式的**现成参考**（MLP projector、Qwen 文本训练、中文描述目标）；其 Qwen 也可作 Agent Loop 的 read/verify 后端 |
| 文字 CE/TextDecoder 历史实现 | git `e22c42c`/`e7087f4`（后被移除） | "文字信号直接练联想"的可回收资产 |
| 探针方法论（`z_s`/`h` 线性解码、within-std、渐进曲线） | doc/ 全系列 | 本 feature 新增"区域可定位性探针"的**方法学基座**（§4.4），先探针后训练 |

### 2.2 数据现状（必须 M0 复核）

- `construction_site` parquet：train 7009 / test 3004；列 = `image` / `image_caption`（中文）/ `violations`（隐患文本），**截至文档口径无任何位置/框/掩码列**。
- 含义：**整图有文字（隐患类别与整图描述），但"哪句话指图中哪块"从未被标注**——这正是"定位看图说话"要补的对应关系，也是"泛化到无标注图片"的天然实验场。

### 2.3 结论：差距是"语言侧 + 对应关系"，不是"视觉信息源"

Phase-1 的证据（`z_s` 均值无信息但 token 间有分工、`h` 逐 patch 可线性解码、DINO 冻结特征可线性解码 L1≈9.8）说明**像素/语义信息在编码链路上是"在的"**；缺的是：① 把"区域"从视觉侧显式取出并送进语言模型的通道；② 文本与区域的对应监督（数据，由 Agent Loop 产出）；③ 语言模型消费区域 token 的训练（Phase 2 首次落地）。

## 3. 总体架构（推荐）

```
                        ┌─────────────── 推理/训练统一链路（文本驱动 + Agent Loop 出框）────────────┐
                        │                                                                           │
 隐患文本 query ───────▶│ L 定位 = Agent Loop 出框协议（§4.1，标注框的唯一来源）                      │
 (如"未戴安全帽")        │   ① propose   L1 开放词表 patch 对齐 / 粗检测 → 初始候选框集               │
                        │   ② read+verify 候选 crop → 看图说话 → 与 query 一致性校验                 │
                        │   ③ refine    不一致 → 平移/缩放/换候选/拆分 → 回 ①（≤R 轮, 置信早停）      │
                        │   ④ converge  一致 → 收敛框 = 标注框（训练伪框 / 推理输出同一条）            │
                        │   （L2 可学习定位=收敛后的一步出框快速路径；L3 register 读出=远期后端）      │
                        │           │                                                                 │
 图像 ────────────────▶│           ▼                                                                 │
 (任意新图，无框)       │ R 区域看图说话：                                                             │
                        │   R1 区域 crop 通道：每个收敛框的 crop → 冻结压缩编码器 → 区域 token z_r     │
                        │   R2（远期）区域掩码读出：区域掩码在 h 上读出 → 区域 token（省重复编码）      │
                        │   R3 全局缩略 token z_thumb（语境） + 区域 token 集 → MLP → Qwen 逐区域描述  │
                        │           │                                                                 │
                        │           ▼                                                                 │
                        │   输出：隐患短语 / 位置(box+方位词) / 区域描述 / 严重度 / 建议（§0 格式）     │
                        └────────────────────────────────────────────────────────────────────────────┘

数据引擎（仅训练期）：整图 caption/violations 文本 → 同一 Agent Loop 出框 + 区域描述（§4.1），产出
    (image, query_text, box, region_caption) 区域指令数据 → 训练 L2 与 R 通道的监督。
```

设计要点：

1. **两通道混合是默认推荐**：全图只留"缩略 token + 候选区域 token"，token 预算**内容自适应地集中在隐患区域**——这是研究计划"按内容复杂度/关键区域自适应分配 Token"在本 feature 的具体落地形态（简单图区域少→token 少；复杂图多隐患→token 给到候选区）。它同时服务"省 token"（主线价值）与"小隐患不挤丢"（区域细读不压缩到整图 K 个 token 里）。
2. **"文本驱动"同时解决两件事**：推理期它是定位信号（无框依赖）；训练期它把"整图 caption/violations"这类**弱对应**文本拆成"短语↔区域"的**强对应**（Agent Loop 数据引擎），从而**泛化到没有标注隐患位置的图片**——用户拿文字问，模型拿文字找、找到就读。
3. **开放词表**：L1 的文本编码器与 DINO patch 特征天然开放词表 → 未见过的隐患类别仍可被问、被粗定位（置信不足时输出"疑似，待人工确认"，与研究计划"模型初筛+人工确认"一致）。
4. **标注框由 Agent Loop 给出（协议层，训练/推理同一条）**：任何进入训练数据或输出的框都不是后端模型的一次性粗结果，而是 **propose→read→verify→refine→converge** 循环收敛后的产物（§4.1）；后端（GroundingDINO/SAM/Qwen2.5-VL/patch 对齐、Phase-2 自研模型）只是 loop 内可替换的工具，**不直接产出标注框**。这一设计让"无框泛化"带自校验闭环：框给错 → 区域描述与文本对不上 → loop 自行修正，而不是把错误框悄悄写进数据或答案。

## 4. 分模块设计

### 4.1 数据引擎 D1 + Agent Loop 出框：把"整图文字"变成"区域指令"（无框泛化的核心机制）

无论 P0/P1，训练数据都必须是 `(image, query_text, box, region_caption)` 四元组。差异只在 **box 从哪来**。**本方案约定：任何标注框都由 Agent Loop 给出（而不是某模型的一次性输出）**——协议如下（训练期与推理期共用同一条，见 §4.2）：

```
Agent Loop 出框协议（propose → read → verify → refine → converge）
  ① propose  依据隐患文本 query 生成初始候选框集 {b}。后端工具可并行出多候选：
             L1 开放词表 patch 对齐（冻结 DINO patch + 文本编码器，零训练热图 → 框）
             / GroundingDINO+SAM（开放词表检测+掩码）/ 教师 MLLM 粗框（Qwen2.5-VL 等）；
  ② read     对每个候选框 crop（扩边防截断）→ 看图说话：reader 输出该区域描述
             （D1 阶段 reader = 开源 MLLM 或整图 caption/violations 对齐；Phase-2
             自研区域看图说话模型就绪后替换为自研 reader，loop 协议不变）；
  ③ verify   校验"区域描述 ↔ query/整图 violations"一致性（文本蕴含 / LLM 判分），
             并做落物自检（SAM 掩码 IoU）→ 一致性分 s；
  ④ refine   s < 阈值 τ → 按差异反馈修正候选：平移 / 缩放 / 换候选 / 拆分多目标 → 回 ①；
  ⑤ converge 一致（s ≥ τ）→ 输出 (query, box, region_caption, conf=s, 轮数)；
             达轮次上限 R 仍不一致 → 判"疑似缺失/低置信"：仅进预训练或转人工复核。
```

> 为什么 loop 而非一次性出框：一次性框**无法自我校验**——框错时没有任何信号知道错了；loop 用"区域描述与文本是否对得上"做**自校验闭环**，框与文自洽后才成为标注框，伪标注噪声因此可被程序化压制（§8 风险 1 的对策）。轮次成本由"置信早停（s≥τ 即收敛）+ 难例才多轮 + 轮次上限 R"控制（§8 新增行）。

| 档 | 前提 | 标注框来源（均经 Agent Loop 收敛） | 说明 |
|---|---|---|---|
| D1-P0 | 训练集无任何框 | ① 初始候选：开放词表检测（GroundingDINO+SAM 给框/掩码）或教师 MLLM 粗框（Qwen2.5-VL 等，用 caption/violations 追问"图中有哪些 X 类隐患，位置在哪"）或 L1 短语↔patch 对齐自举；② loop 迭代 refine 至收敛 → `(短语, 框, 区域描述, conf, 轮数)` | 主导路线；多初始候选 + 收敛判据（描述↔文本一致性 + SAM 落物）双重把关，低置信记录不进评测 |
| D1-P1 | 仅部分人工框 | 人工框直用（并作为 loop verify 步的真值参照）；未覆盖短语走 P0 档 loop 补齐；人工框与 loop 收敛框 IoU 冲突时以人工框为准；难例（loop 收敛失败/类别少）进主动学习补标队列 | 半监督 + 主动学习 |

短语提取（query 与 region_caption 的来源）：

- **主源**：`violations` 结构化隐患短语（或解析出的"隐患类别/对象+状态"，如"未佩戴安全帽"）；
- **辅源**：整图 `image_caption` 中与视觉实体强相关的名词短语（分词 + 依存/实体抽取），作为 query 送 Agent Loop 出框；
- **类别兜底**：隐患类别词表（裂纹/腐蚀/泄漏/异物/违章等）全量枚举追问，保证**类别覆盖完备性**（负样本：loop 收敛失败或确认缺失 = 该图无此类隐患，顺手得负例）。

区域描述（region_caption）来源：

- Agent Loop 的 **read 步产物**：reader 对收敛框内 crop 单独看图说话（问句带类别，抑制答非所问）；
- 已有整图 caption/violations 按"短语↔框↔句子"对齐后做归一化改写（可作 read 步的提示/校验参照）；
- 质量闸：每条记录附 `conf`、`source`（人工 / agent_loop）、`rounds`、`区域与整图描述一致性`抽查标记；低置信记录**只进预训练不进评测**。

数据质检与自检判据：随机 200 条人工/程序化检查——框中心落在对应物体上（可用 SAM 掩码 IoU 自检）、region_caption 提到 query 类别、无同图重复框冲突、loop 收敛率（一致收敛占比）上报；**自检不过不发版**（对齐仓库"自检 ALL CHECKS PASSED 才提交"的纪律）。

### 4.2 推理协议（两阶段，一次问答 = Agent Loop 定位 + 区域看图说话）

```
输入 1：image（任意新图，无框）
输入 2：query_text ∈ { 单隐患问句："定位并描述：图中未盖板的井口"
                     多隐患枚举："找出图中所有安全隐患并逐一描述"
                     整图问句  ：描述整张图 }   ← 退化情形= 全图一个大区域
输出：见 §0 格式；多隐患枚举时输出区域列表，每项含短语/框/描述。
```

关键点：**推理路径与训练路径同构，且共用同一条 Agent Loop（§4.1）**——训练时监督就是"文本→框→区域描述"（框 = loop 收敛框），推理时同一 loop 以文本为信号迭代出框，无需任何标注；泛化不是靠"推理时无框"，而是靠**训练时就教模型从文本自己找框**。推理侧的性能旋钮：L2 定位头（用 loop 收敛框训练）就绪后走"**一步出框**"为主路径，仅在低置信（verify 分不足）时启用多轮 refine 兜底（§4.3）。

### 4.3 组件选型决策矩阵

| 组件 | 候选 | 优点 | 缺点/风险 | 建议 |
|---|---|---|---|---|
| **L 定位** | L1 冻结 patch+文本编码器对齐（零训练） | 开放词表、零训练成本、立即可用；给 loop 提供初始候选，也是 L2 伪标签与评测对照 | 精度有限、无语义级精修 | 作 Agent Loop 的**初始候选器**（propose 后端） |
| | L2 Qwen 出 `<box>`（Kosmos-2/Shikra/Qwen2.5-VL 式坐标 token；监督 = Agent Loop 收敛框） | 端到端、语义强、**一步出框省轮次** | 依赖 Phase-2 Qwen 训练落地 | 主线的**快速出框路径**（低置信才进多轮 refine） |
| | L3 文本条件 register 读出（query 作 key 进 OutputQueryDecoder） | 与压缩主线同构，最"自研" | 未验证；依赖 z_s 保位置 | 远期科研点（可作 loop 的 propose 后端）（§6 M4） |
| | **Agent Loop 出框协议（统一）** | propose→read→verify→refine→converge：标注框与推理出框走同一条，自带一致性自校验（§4.1） | 多轮带来推理时延（置信早停 + 难例多轮 + 轮次上限 R 控制） | **所有框的默认来源** |
| **R 区域看图说话** | R1 区域 crop→冻结编码器→区域 token | 不依赖"压缩保位置"前提（crop 天然带空间）；复用冻结编码器与 Phase-2 范式；crop 恰为 loop 的 read 步输入，链路复用 | 区域多时 token 总量上升（靠内容自适应预算/区域级压缩收敛） | **主线默认** |
| | R2 区域掩码在 h 上读出 | 省重复编码、直接吃 Phase-1 坐标轴 | 需要 h 空间语义成立（探针定） | 探针通过后再做 |
| | R3 整图缩略 token + 区域 token 混合 | 语境保真、防区域描述"断章取义" | — | 默认伴随 R1/R2 |

### 4.4 与压缩主线的衔接：**区域可定位性探针先行**（关键，防第 N 次"训完才发现机制不成立"）

沿仓库"先探针后训练"铁律，在投入 Phase-2 区域看图说话训练前，先回答一个前置问题：**K 个压缩 token（z_s）里，区域位置信息还剩多少？**

- **探针 RLP（Region-Localizability Probe）**：取一批已有人工框/Agent Loop 收敛框的图，从冻结编码器出口各层（`P`、`z_s`、`h`）分别线性/浅层回归"逐 patch 是否属于某隐患框"的标签，比较三者 AUC/Acc@0.5。
- **预期三态与行动**：
  1. `h`（甚至 `z_s`）定位精度 ≈ `P`（patch 级）：压缩保位置 ⇒ R2/L3 通道可做，token 最省；
  2. `z_s` 明显劣于 `P`、`h` 尚可：区域看图说话走 R1 crop + R2 从 h 读；定位走 L1/L2（patch 侧）；
  3. 全部劣化：位置信息在压缩中被压没 ⇒ **先回 Phase-1 补位置保真**（位置编码 A/B、区域重建损失、K 压缩×位置保真验收），再谈 Phase-2 区域能力——避免在错误地基上盖楼。
- 配套 Phase-1 口径新增（轻量）：在现有 `eval_recon` 之外加**区域重建 L1**（只对隐患框内 patch 计算），作为"小隐患信息不被压缩挤丢"的直接探针，与研究计划"布局/物体/边界保真"验收同一口径。

### 4.5 Phase-2 接入（冻结编码器 + Qwen，唯一验收链路）

沿用 GOAL 范式与 v1 `model.py` 的 MLP Projector→Qwen 接入，输入序列改为：

```
[区域指令模板] 用户：定位并描述"未戴安全帽的作业人员"  助手：<box>0.31,0.42,0.38,0.55</box> 一名工人……
     视觉侧：[cls; 整图缩略 z_thumb; 区域1 z_r1; 区域2 z_r2; ...]   ← 冻结编码器输出, MLP 投影
```

训练目标：区域看图说话 **CE**（唯一验收信号）+ 可选定位 CE（L2 的 `<box>` 监督，box = Agent Loop 收敛框）。文字 CE 让冻结编码器被迫产出"能支持区域描述"的表示——正是 GOAL §4.3"文字 CE 直接练联想"的落地。若 RLP 探针（§4.4）显示 h 强于 z_s，则视觉侧可混入 h 的区域行（R2）作为通道消融变量。Phase-2 就绪后，Agent Loop 的 **read/verify 后端从开源模型切换为自研模型**（loop 协议不变，只换 reader），实现"自产数据→自训模型→更优数据"的闭环迭代。

## 5. 三方案对比与推荐

| | 方案 A-0：试点基线（周级） | 方案 A-1：主线落地（月级，**推荐**） | 方案 B：远期科研形态 |
|---|---|---|---|
| 内容 | 冻结 v1 范式或直接用开源 grounding MLLM（如 Qwen2.5-VL）在 D1（Agent Loop 收敛标注）区域指令上微调 | A-0 + 冻结**自研压缩编码器**（v2/v4 产物），R1 区域 crop→区域 token→Qwen；Agent Loop 先以 L1 为初始候选、L2 快速出框跟进 | A-1 + 定位头进自研解码链（L3/R2）+ 内容自适应区域 token 预算 + 多隐患并发枚举 + 小目标专项 + 自研 reader 闭环 |
| 时间/算力 | ~1–2 周 + 已有 GPU | 1–2 月（含 Agent Loop 出框器、RLP 探针与 D1 质检） | 论文级，与 Phase-1 主线并行演进 |
| 对"压缩保位置"的依赖 | 无（天然上界对照） | 低（R1 crop 不依赖；z_s 通道由探针把关） | 高（L3/R2 全依赖） |
| 交付物 | 试点 demo + Agent Loop 数据引擎质检报告 | 仓库 Phase-2 区域看图说话最小可运行集 + 评测 | 研究内容 1/3/4 的创新点闭环 |
| 作用 | **尽早暴露数据/模板/loop 收敛问题，压低总风险**；同时是 A-1 的质量上界参照 | 本项目 feature 的主体交付 | 贡献研究计划"隐患识别、描述与定位端到端验证"的学术增量 |

**推荐路径：Agent Loop 数据引擎（D1）与 A-0 试点先行（并行启动），RLP 探针结果出来后定 A-1 的通道构成；A-1 通过区域看图说话验收后，B 才立项。** 理由：A-0 不需要等 Phase-1 收敛，能立刻产出"文本→框→区域描述"的端到端效果供评审；A-1 保证自研链路（本项目核心价值）；B 是增量。

## 6. 分期实施路线与验收判据（每期含自检闸门）

| 期 | 内容 | 主要代码/数据改动 | 验收判据（可证伪） |
|---|---|---|---|
| **M0** | 数据现状复核 + 评测集筹建 | 盘点 parquet 是否已有框列（复核 P0/P1）；人工框小评测集 ≥300 张（类型+框+区域描述），定标注规范 | 评测集可用性报告；P0/P1 判定文档化 |
| **M1** | Agent Loop 数据引擎 D1（P0/P1 两档）+ A-0 试点 | `tools/` 新增区域指令数据生产者：Agent Loop 出框器（propose/read/verify/refine 四步可插拔）、短语提取、质检、可视化 | ① 伪框抽检命中率达标（中心落物 IoU）+ **loop 收敛率上报**；② 区域指令 200 条人工抽检通过；③ A-0 试点 demo 可端到端"文本→框→描述" |
| **M2** | RLP 区域可定位性探针 | 复用 `trace_info_pixel` 思路新增 RLP 探针脚本（输入含 Agent Loop 收敛框） | 出 P/z_s/h 三层定位精度表 → 决策 R2/L3 是否可做（§4.4 三态） |
| **M3** | A-1 主线：Phase-2 区域看图说话最小可运行集 | main 新增 Phase-2 训练/推理（冻结编码器 + 区域 token 通道 + Qwen），对齐 `model_info.json` 记录字段；推理走"L2 一步出框 + 低置信 loop refine" | ① 区域看图说话：隐患类型/位置/描述在评测集上达标（研究计划口径：类型与位置识别准确率≥90% 为方向，基线对照 A-0 不降 >2%）；② 定位 Acc@0.5 ≥ A-0 对照的 ~90%；③ 无标注泛化：留出"训练伪标注置信低/新场景"子集不劣化 |
| **M4** | B 远期立项判定 | L3/R2/内容自适应预算/自研 reader 闭环设计稿 | 由 M2 探针与 M3 结果决定是否立项 |

> 所有"达标"数字为**目标草案**，需在 M0 评审对齐后固化为验收基线；每期闸门不过则回到上一期修（对齐仓库"机制没生效就不进下一步"的纪律）。

## 7. 评测与指标

- 评测集：**人工框小集合（M0，≥300 张）**是必须的——即使训练数据无框，评测必须有框才能测"定位准不准、描述对不对区域"，否则无法验收泛化。
- 指标表：

| 能力 | 指标 | 说明 |
|---|---|---|
| 定位 | Acc@0.5（IoU≥0.5 命中率）/ 分类准确率（框内类别对否）/ **Agent Loop 收敛率与平均轮数** | 对齐研究计划"隐患类型与位置识别准确率"；loop 指标单独上报（成本-质量旋钮） |
| 区域看图说话 | region-caption 人工评分（类型/对象数/状态/相对位置四项分）+ LLM-judge 一致性分；BLEU-4/CIDEr（中文）作辅助 | 关键防"张冠李戴"：**描述内容必须属于被指区域** |
| 无框泛化 | 留出子集（伪标注低置信图/未见位置布局/未见类别）上上述指标 | 类别开放用词表召回补测 |
| 回归 | Phase-1 `eval_recon`/区域重建 L1 不劣化；A-1 相对 A-0（未压缩上界）降幅 ≤2% | 保住压缩主线价值 |

## 8. 风险与对策

| 风险 | 对策 |
|---|---|
| 无框数据下伪标注噪声大（错框/漏框/类别串扰） | **Agent Loop 收敛判据把关**（区域描述↔文本一致性 + SAM 落物自检），框与文不自洽不落地；多初始候选交叉；低置信记录只进预训练；人工小评测集仲裁 |
| Agent Loop 推理轮次成本/时延 | 置信早停（s≥τ 即收敛）、轮次上限 R、仅难例多轮；L2 一步出框为主路径，loop refine 只兜底（§4.3）；多候选并行 propose |
| 压缩把"小隐患"压丢（裂纹/腐蚀细微目标） | 区域 token 预算集中在候选区（内容自适应）；区域重建 L1 探针把关；小目标专项增强 |
| 区域描述"张冠李戴"（读到邻区内容） | verify 步的一致性校验即为此设；描述问句带类别约束；区域 crop 扩边/掩码防截断；评测单列"归属正确率" |
| Phase-2 Qwen 训练成本/数据不足 | A-0 开源模型先行（上界+降险）；文字 CE 小批量路线（GOAL §4.3）；中文数据与已有 caption 复用 |
| 与 Phase-1 主线争抢 GPU/人力 | A-0/D1 不需要 Phase-1 收敛即可并行；M2 探针是纯推理零训练成本（~30 分钟） |
| 安全红线（核电场景误报/漏报） | 输出带置信与"疑似待人工确认"档位（loop 收敛失败即降档）；人机协同复核（与研究计划一致）；评测漏检率专项 |

## 9. 适用边界与局限性（哪些"隐患"识别不了——纯语义/整图级/规则依赖类）

GRC 的默认前提是"隐患 = 图中**可被文本定位、可被一个框/区域容纳**的实体或局部缺陷"。以下类别**不满足该前提**：纯视觉语义理解（无论是否经 token 压缩、是否走 Agent Loop）识别不了或不该由本 feature 负责：

| 类别 | 例子 | 为什么现有链路失效 | 处理方式（边界外的兜底通道） |
|---|---|---|---|
| **图级/环境级属性**（非实体） | **光线昏暗**（照明不足）、能见度低、雾/烟弥漫 | 无可定位实体 → propose 给不出有意义的框 → loop 无法收敛；"位置"语义不存在（是全图属性，不是某区域的内容） | 整图通道：图级评估器（亮度/照度估计）+ 整图看图说话（§4.2 的"整图问句"退化情形），不强制出框 |
| **全局关系/秩序类**（非局部） | **摆放杂乱**（物料堆放无序/占道/超高）、现场混放 | "是否构成隐患"需**整图比较 + 规范依据**（堆放高度/通道宽度/码放要求），不是某一块区域里的东西；区域 caption 无从谈起 | 关系/秩序评估通道 + 规范知识问答；数据侧打**图级标签**（不追求框） |
| **规则推断类**（视觉弱信号 + 规范知识） | **工人撸起袖子**（着装规范：袖口须放下）、防护配件佩戴不规范 | 违反的是"看不见的规则"：需要人体/着装细粒度解析 + 作业规范知识库推断；视觉特征不显著，判定本身可争议，"看图说话"容易漏判或误报 | 规则知识库 + 人体/着装细粒度模型做**整图级判定**；输出带规则依据与置信，低置信转人工复核（与研究计划"模型初筛+人工确认"一致） |

机制层面的三点说明（为什么不是"多训点数据"就能吸收进本 feature）：

1. **对 Agent Loop 的影响**：这类 query 的 propose 无实体候选或候选无意义 → 多轮 refine 也无法收敛；loop 的 verify 判据（"区域描述 ↔ 文本"一致性）对它们**天然失效**——要验证的内容是整图/关系/规则，不是单一区域的描述；
2. **对 D1 数据引擎的影响**：`(phrase, box)` 伪标注产不出来 → 训练数据天然覆盖不到；硬造只会得到"框=全图或随机位置"的噪声框，反过来污染可定位类的训练；
3. **分类学与评测建议**：隐患类别体系应显式区分 **A 类（可定位实体/局部缺陷，本 feature 负责）**与 **B 类（图级/关系/规则类，整图+规则通道负责）**；评测集按 A/B 分组报告，B 类不套用 Acc@0.5 / 区域描述口径。

> 交互降级约定：用户 query 属 B 类、或 loop 多轮仍无法收敛出框时，**自动降级为整图看图说话**并明示"该描述针对整图/需结合规范判断，未定位到单一区域"，不给假框、不硬报位置。

## 10. 参考实现（方案伪代码，非提交代码；风格对齐仓库"先自检后提交"）

### 10.1 数据引擎 D1（Agent Loop 出框的区域指令数据生产者）

```python
# tools/build_region_instructions.py（方案示意）
def agent_loop_locate(img, phrase, propose, reader, verifier, max_rounds=6, tau=0.5):
    """Agent Loop 出框协议（§4.1）: propose → read → verify → refine → converge。
    收敛（verify 分 ≥ tau）才返回 (框, 区域描述, conf, 轮数)；否则判疑似缺失/低置信。
    propose : L1 开放词表 patch 对齐 / GroundingDINO 粗候选 → 候选框列表（可多候选并行）
    reader  : crop 候选区 → 看图说话（区域描述; D1 用开源 MLLM, Phase-2 后换自研 reader）
    verifier: 描述与 phrase 的一致性分（文本蕴含 / LLM 判分）+ 落物自检（SAM IoU）
    refine  : 不一致 → 平移 / 缩放 / 换候选 / 拆分 → 更新候选框列表"""
    cands = propose(img, phrase)                     # ① propose
    for rnd in range(max_rounds):
        box = best(cands)
        cap = reader(img, box, hint=phrase)          # ② read：候选区看图说话
        s = verifier(cap, phrase, img, box)          # ③ verify：一致性 + 落物
        if s >= tau:                                 # ⑤ converge → 这就是标注框
            return norm(box), cap, s, rnd + 1
        cands = refine(img, phrase, box, cap, s)     # ④ refine：反馈修正后回 ①
    return None, None, 0.0, max_rounds               # 收敛失败 → 仅预训练 / 人工复核


def build_region_instructions(parquet_files, propose, reader, verifier, lexicon, conf_thr=0.3):
    """整图 caption/violations → (image, query_text, box, region_caption) 区域指令数据。
    标注框一律来自 agent_loop_locate 的收敛框（P1 档: 人工框直用 + 未覆盖短语走 loop 补齐）。"""
    for row in load_parquet(parquet_files):
        img = decode(row.image)
        phrases = extract_hazard_phrases(row.violations)          # 主源
        phrases += scan_lexicon(img, lexicon, propose)            # 类别兜底(含负例)
        for phrase in dedup(phrases):
            box, cap, conf, rounds = agent_loop_locate(img, phrase, propose, reader, verifier)
            if conf < conf_thr:                                   # 低置信 → 仅预训练
                continue
            emit({"image": img, "query": f"定位并描述：{phrase}", "box": box,
                  "region_caption": cap, "conf": conf, "rounds": rounds, "source": "agent_loop"})
    return write_region_dataset(...)   # 输出区域指令 parquet + 质检 JSON + 可视化图
```

### 10.2 RLP 区域可定位性探针（M2，复用 trace_info_pixel 思路）

```python
# probe_region_localizability.py（方案示意）—— 零训练，纯推理
def probe(model, images, region_masks, layers=("P", "z_s", "h")):
    """从各层特征线性回归 'patch 是否属于隐患框'，比较定位精度。
    region_masks 由人工框或 Agent Loop 收敛框转出。
    三态判定见 §4.4：决定 R2/L3 是否可做、区域看图说话走哪条视觉通道。"""
    for name in layers:
        F = collect(model, images, name)            # (M, N, D) 或 (M, K, D)
        auc = train_val_split(F, region_masks)      # logistic head, AUC/Acc@0.5
        log(f"{name}: AUC={auc:.3f}  vs patch-level P 上界")
```

### 10.3 Phase-2 区域看图说话（A-1 概念 forward）

```python
class Phase2GRC(nn.Module):   # 概念示意：冻结编码器 + 区域 token 通道 + Qwen
    def __init__(self, enc, projector, lm, loop=None):  # enc=冻结 v2/v4 压缩编码器; loop=§4.1 Agent Loop
        super().__init__(); self.enc, self.proj, self.lm = enc, projector, lm
        self.loop = loop                                # 含 L1 初始候选 / L2 快速出框 / verify
    def forward_train(self, image, query, box, caption):   # box = Agent Loop 收敛框（D1 产出）
        z_thumb = self.enc.compress(image)                      # (B,K,D) 整图缩略
        crops   = [crop(image, b) for b in box]                 # R1 区域 crop 通道
        z_r     = [self.enc.compress(c) for c in crops]         # 冻结编码器复用
        vis     = self.proj(cat([z_thumb, *z_r]))               # → Qwen 嵌入空间
        out = self.lm(inputs_embeds=vis, labels=text_ids(query, box, caption))
        return out["loss"]                                      # 区域看图说话 CE(+定位 CE)
    def generate(self, image, query_text):
        # §4.1 协议: L2 一步出框为主路径, verify 分不足才启用多轮 refine 兜底
        box = self.loop.locate(query_text, image, fast_path="L2")
        return self.forward_train(image, query_text, box, None, generate=True)
```

## 11. 待评审决策清单（决策入口）

1. **数据前提**：M0 复核当前/近期核电隐患图数据是否有框列；评测人工框小集（≥300 张）可否安排（预算/人力/脱敏）。
2. **Agent Loop 后端工具选型与许可**：propose（L1 patch 对齐 / GroundingDINO+SAM）/ read（Qwen2.5-VL 等看图说话）/ verify（判分模型）的自部署选型（国产算力可用性、数据不出域约束）。
3. **loop 超参与质量线**：轮次上限 R、收敛阈值 τ、低置信分流线 conf_thr、多候选并行数——M1 用一组默认值起跑并上报收敛率后调优。
4. **输出规范**：是否按 §0 "隐患/位置/描述/严重度/建议"结构化格式立项（与研究计划一致）。
5. **目标数值**：§6 里程碑的达标线（90%/2% 等）是否按研究计划 KPI 固化为验收基线。
6. **排期与算力**：A-0/D1 是否可与 Phase-1 主线并行；GPU 预算。
7. **优先级**：本 feature 与当前"后步归零/区域损失"未决项（README §4.1）的先后。

## 12. 参考文献（已验证 arXiv 编号）

- 定位/开放词表：Grounding DINO（Liu et al., 2023）https://arxiv.org/abs/2303.05499 ；GLIP（Li et al., 2022）https://arxiv.org/abs/2112.03857 ；MDETR（Kamath et al., 2021）https://arxiv.org/abs/2104.12763
- 掩码/区域：SAM（Kirillov et al., 2023）https://arxiv.org/abs/2304.02643 ；DenseCap（Johnson et al., 2016）https://arxiv.org/abs/1511.07571
- 指代看图说话 MLLM：Kosmos-2（Peng et al., 2023）https://arxiv.org/abs/2306.14824 ；Shikra（Chen et al., 2023）https://arxiv.org/abs/2306.15195 ；Ferret（You et al., 2023）https://arxiv.org/abs/2310.07704 ；LLaVA（Liu et al., 2023）https://arxiv.org/abs/2304.08485 ；Qwen2.5-VL（Bai et al., 2025，原生框/点定位，可作 Agent Loop 的 read/粗框后端）https://arxiv.org/abs/2502.13923
- 压缩-语言范式（仓库已引/已用）：BLIP-2 https://arxiv.org/abs/2301.12597 、Flamingo https://arxiv.org/abs/2204.14198 、InstructBLIP https://arxiv.org/abs/2305.06500 、Qwen-VL https://arxiv.org/abs/2308.12966
- 仓库内相关：`doc/2026-08-28/GOAL_compression_for_nlp.md`；test 分支 v1 `model.py`（SR-Qwen-VL 接入范式）；`feature/DESIGN_graph_embedding_schemeA.md`

> 说明：区域指令数据的"伪标注数据引擎"在本文的落地形态是 **Agent Loop 出框协议**（propose/read/verify/refine 四步后端可替换，方案不依赖单一开源实现）；GrIT 的"图文 → 短语↔框"构造思路（Kosmos-2, 2306.14824 §3）可作短语提取与初始候选的参考。
