# SR-Diffusion Phase 1 v2（dev 分支 · 最小可运行）

用**像素重建**当代理任务，训练视觉编码器的 **token 压缩 + 联想**能力（Phase 1 脚手架）；
训练完成后**冻结编码器**，接入 Qwen 做中文工地描述/隐患生成（Phase 2 NLP 解码，**最终验收标准**）。
像素重建 = 信息保持的直接探针（能还原像素 ⇒ z 携带整图信息 ⇒ 语义信息必然在）；
纹理级高频属"死信息"不追，压缩率 K 是练联想的真正杠杆（当前 k=N 全量零压缩，尚未验证）。

> 权威目标定义见 [`doc/2026-08-28/GOAL_compression_for_nlp.md`](doc/2026-08-28/GOAL_compression_for_nlp.md)。

## 架构（register 式，唯一路径）

```
448×252 输入 (1600:900 画布预处理)
    │
    ▼  [DINOv2-large, 不冻结]   specials 作为额外 token 拼进输入序列
                                 [cls; specials; patches] (2N+1 token, 全双向)
z_cls, z_s (B, 1+N, D)          ← 由 DINO 24 层直接算出（register token 式, Darcet et al.）
    │
    ▼  [OutputQueryDecoder]      采样时刻 = 分块起点（平方数 1,4,9,…，每块一步）
                                 每步只 attend 自己的 z_s 块；结果沿采样步累加
F_hat (B, N, D) 特征             ← Σ_t Y_t
    │
    ▼  [PixelHead]  Linear D→588
像素 patch (B, N, 588) → 重建 448×252
```

- 损失 = 每个采样步**累加结果**的 L1 平权全覆盖（梯度按步解耦：carry 整体 detach + 自己的预测，每步恰收 1 份梯度）。
- 无 ReEncoder、无 TextDecoder、无选择/预算机制。

## 快速开始

```bash
pip install -r requirements.txt

# 0) 准备 DINOv2-large 权重（HF hub 或本地目录）
python -c "from transformers import Dinov2Model; \
Dinov2Model.from_pretrained('facebook/dinov2-large').save_pretrained('models/dinov2-large')"

# 0b) 没有真实数据？生成合成样本（CPU 可跑冒烟）
python make_sample_data.py --out_dir sample_data --n_train 32 --n_test 8

# 1) 冒烟训练（小图 224x126，3 步，几分钟内可跑完）
python train_v2.py --data_dir sample_data \
    --dino_dir models/dinov2-large \
    --output_dir output/smoke --smoke --limit 16 --max_steps 3 \
    --eval_every 1 --batch_size 2 --num_workers 0 --model_input 224x126

# 真实训练（2 卡 DDP，fp32，448x252）:
NUM_GPUS=2 ./run_v2_train.sh \
    --data_dir /path/to/construction_site \
    --dino_dir /path/to/dinov2-large \
    --output_dir output/phase1_v2 --epochs 40

# 2) 推理测试（全量像素 L1 + 渐进曲线）
python infer_v2_test.py --data_dir sample_data --dino_dir models/dinov2-large \
    --final_model output/smoke/final_model.pt --output output/smoke/infer_test.json

# 3) 重建可视化（原图 vs 各采样步）
python visualize_recon_pixel.py --data_dir sample_data --dino_dir models/dinov2-large \
    --final_model output/smoke/final_model.pt --out output/smoke/recon_visual.png

# 自检
python model_v2.py      # 模型形状 / 分块掩码 / 梯度（含按步解耦）
python data_v2.py       # 数据管线
```

## 数据格式

`--data_dir` 下 `train-*.parquet` / `test-*.parquet` 分片，`image` 列为
`struct{bytes, path}`（HF parquet Image 格式），可选 `image_caption` / `violations` 文本列。
`data_v2.py` 处理链：最优旋转角 → 等比缩放 → 居中填充 1600:900 画布 → 缩放到模型输入
（必须 16:9，且宽高为 14 的倍数）。

## 文件

| 文件 | 说明 |
|---|---|
| `model_v2.py` | 核心模型（register 式）+ 自检 |
| `data_v2.py` | 数据管线（画布预处理 + parquet dataset）+ 自检 |
| `train_v2.py` | 训练（HF Trainer，fp32，平权 L1） |
| `run_v2_train.sh` | 训练入口（`NUM_GPUS=n` 多卡 DDP） |
| `infer_v2_test.py` | 推理测试（像素 L1 + 渐进曲线） |
| `visualize_recon_pixel.py` | 重建可视化 |
| `make_sample_data.py` | 合成样本生成（冒烟/试跑） |
| `doc/2026-08-28/GOAL_compression_for_nlp.md` | 权威项目目标 |

## 给新开发者

- **掩码约定**：torch 的 bool 注意力掩码 **True=屏蔽**（直觉相反）；`F.scaled_dot_product_attention`
  的 bool 掩码实测 True=允许。本项目解码器统一用加法浮点掩码（-inf=屏蔽）规避歧义。
- **register 式**：specials 无内容输入（`SpecialTokenBank` 共享 token + 逐位置位置编码），
  内容由 DINO 24 层全双向注意力路由；"前缀稳定性"不成立，渐进语义完全由解码器分块掩码提供。
- **梯度按步解耦**：`SRPhase1V2.decode` 里 carry 整体 detach + 自己的预测——数值上仍是
  cumsum（F_hat 不变），但每个 Y_t 只从自己那一步的损失收梯度，避免"t=0 收 |T| 份梯度"失衡。
- 本分支只保留 v2 最小可运行集；v1/v3/v4/v5 模型与历史实验归档在 `main` 分支
  （`git checkout main` 可见）。
- 推理/可视化脚本读取 `output_dir/model_info.json` 与训练配置对齐；推理端参数必须与训练
  一致（strict load 权重）。
- 推理/可视化脚本自动选择 `cuda`/`cpu`，无 GPU 也能跑。

## 已知

- 冒烟训练的 L1 数值无意义（随机权重 + 合成图），只验证链路可跑。
- 全量训练预期在 Linux + GPU 环境（`run_v2_train.sh` 面向该环境）。
