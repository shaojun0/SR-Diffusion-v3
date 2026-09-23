# 新服务器验收测试：AutoDL bjb1 / A800-80GB（2026-09-23）

> 连接：`ssh -p 54066 root@connect.bjb1.seetacloud.com`（本机 `~/.ssh/id_ed25519` 免密可用）
> 容器：`autodl-container-sq1jeg5hau-00ab253e`｜Ubuntu 22.04.5｜kernel 5.15.0-78

## 0. 结论

**可用，且很适合跑 SR-Diffusion 的分辨率扫描。** 单卡 A800-80GB、144 核、1TB 内存；
仓库自检 `ALL CHECKS PASSED`；`tools/sweep_res_train.py` 的 224² 端到端 GPU 冒烟 9 秒跑完。
**唯一注意：只有 1 张卡**（`run_v2_train.sh` 默认 `NUM_GPUS=2`，要改 1）。

---

## 1. 硬件

| 项 | 值 |
|---|---|
| GPU | **1× NVIDIA A800-SXM4-80GB**（sm_80，driver 580.126.09，空闲） |
| CPU / 内存 | 144 核 / **1007 GB** |
| `/root/autodl-tmp`（数据盘） | **50 GB**（已用 205 MB） |
| `/`（overlay） | 30 GB（已用 830 MB） |
| `/dev/shm` | 60 GB |
| 公共数据盘 `/autodl-pub/data` | 10 TB 只读，含 **DIV2K / Flickr2K / ImageNet / COCO / VOC** 等 43 个数据集 |

## 2. 软件环境（实测）

| 组件 | 状态 |
|---|---|
| Python | `/root/miniconda3/bin/python` **3.12.3**（base env；`python3` 不在 PATH） |
| torch | **2.12.1+cu130**，`cuda.is_available()=True`，A800 cap (8,0)，bf16 正常 |
| torchvision | 0.27.1+cu130 |
| numpy / pillow | 2.4.6 / 12.2.0 |
| **本轮新装** | transformers **5.17.0**、accelerate **1.15.0**、safetensors 0.8.0、datasets 5.0.1、sentencepiece 0.2.2、tqdm 4.70.1、bitsandbytes **0.50.2** |
| bitsandbytes | `AdamW8bit` 前反向实测通过（torch 2.12 + CUDA 13 下可用） |
| 其他工具 | git / screen / rsync / curl / wget / htop 有；**tmux 没有**（用 `screen` 或 nohup+setsid） |

> transformers 5.17.0 正好是分辨率扫描报告里写的那个版本（`sweep_res_train.py` 头部对
> `Dinov2Embeddings` 自动插值的注释在 5.x 下成立）。

## 3. 网络（重要）

| 目标 | 结果 |
|---|---|
| `pypi.tuna.tsinghua.edu.cn` | **200，~20 MB/s**（推荐） |
| `pypi.org` | 200，~10 MB/s |
| `mirrors.aliyun.com`（pip 默认源） | 200 但**只有 ~0.5 MB/s** |
| `github.com` | 直连 200（偶发超时）；`source /etc/network_turbo` 后 **2.2 s** |
| `huggingface.co` | **超时/被墙** |
| **`hf-mirror.com`** | **200，0.6 s** ✅（`facebook/dinov2-small` 9 秒下完） |

- 下模型：`HF_ENDPOINT=https://hf-mirror.com python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/dinov2-large', local_dir='/root/autodl-tmp/models/dinov2-large')"`
- ⚠️ **`source /etc/network_turbo` 会设 `http_proxy`，会让 pip 直接失败**
  （我第一次 probe 就是这个原因报 "No matching distribution"）。**pip 和 turbo 不要同壳**。
- 用 git 拉 GitHub 前先 `source /etc/network_turbo`；pip 前先 `unset http_proxy https_proxy`。

## 4. 已完成的部署与实测

| 动作 | 位置 | 结果 |
|---|---|---|
| 同步仓库工作树（不含 .git，避免带出 token） | `/root/sr-diffusion-v3` | 40 MB，rsync OK |
| DINOv2-small 权重 | `/root/autodl-tmp/models/dinov2-small`（88 MB） | OK |
| 合成小数据集（64 train / 16 val PNG） | `/root/autodl-tmp/srres/data/` | OK（仅冒烟用） |
| 代码自检 `python model_v2.py` | — | **ALL CHECKS PASSED**（含 layer_tap / 梯度解耦 / 损失口径） |
| GPU 端到端冒烟 `tools/sweep_res_train.py --size 224 --smoke --warm_steps 20` | `/root/autodl-tmp/srres/smoke_out/` | **9 秒**跑完，产出 `result.json` + `final_model.pt`（117 MB） |

**冒烟顺带复现了平凡解塌缩**（与根因分析一致）：
A 相只跑 20 步 → 单步全读 L1 **52.16 / 12.19 dB**（还在"预测均值"）；
B 相 60 步后仍停在 L1 52.3–53.0 / 12.0–12.2 dB，**16 步曲线几乎平的**。
⇒ 佐证 `ANALYSIS_res_sweep_rootcause.md` §3：224² 的 A 相需要 ~2000 步才逃出平凡解。

## 5. 吞吐基准（DINOv2-small + d2 + patch 读出 + 平方块读窗口，fp32/TF32）

| 配置 | it/s | 峰值显存 |
|---|---:|---:|
| 112²（N=64，\|T\|=8，bs16） | **16.3** | 1.9 GiB |
| 224²（N=256，\|T\|=16，bs16） | **3.29** | 9.2 GiB |
| 336²（N=576，\|T\|=24，bs8） | **2.04** | 14.0 GiB |
| 336²（N=576，\|T\|=24，bs16） | 1.08 | 27.4 GiB |
| 336²（N=576，\|T\|=24，bs8，**fp16 autocast**） | **6.56（3.2×）** | 13.4 GiB |

推算墙钟（不含数据读取；真实 DIV2K 建议先跑 `sweep_res_prep.py` 转 .npy）：
- `run_sweep.sh`（5 分辨率 × 2500 步等预算）≈ **45–60 分钟**
- `run_sweep2.sh`（放大预算到 7000 步）≈ **1.5 小时**
- **关键验证臂（224²，A=3000 + B=5000 ≈ 8000 步）≈ 40 分钟**

## 6. 跑真实实验前还差的两步

1. **DIV2K**：公共盘上是 zip（`/autodl-pub/data/DIV2K/HighResolution/DIV2K_{train,valid}_HR.zip`），
   需要解压到数据盘（约 8.5 GB，50 GB 够）：
   ```bash
   mkdir -p /root/autodl-tmp/srres/data
   cd /root/autodl-tmp/srres/data
   unzip -q /autodl-pub/data/DIV2K/HighResolution/DIV2K_train_HR.zip
   unzip -q /autodl-pub/data/DIV2K/HighResolution/DIV2K_valid_HR.zip
   ```
2. **主线的工地图数据 + DINOv2-large**：公共盘里**没有** `construction_site`，也没有 dinov2-large 权重。
   要跑 448×252 主线臂需要：上传 `construction_site` 数据集 + 从 hf-mirror 下 `facebook/dinov2-large`（~1.1 GB）。

多卡相关：本机只有 1 卡 ⇒ `NUM_GPUS=1 ./run_v2_train.sh ...`（原配方是 2×bs16 = 全局 32，
单卡可用 `--batch_size 32` 或 `--batch_size 16 --grad_accum 2` 保持全局 batch 不变）。

## 7. 建议的下一步（把根因实验跑掉）

```bash
# 224² 的判决实验：A 相给到 3000 步（逃出平凡解）再 B 相 5000 步
cd /root/sr-diffusion-v3
sed -n '1,40p' tools/sweep_res_train.py   # 已确认参数名
setsid nohup /root/miniconda3/bin/python -u tools/sweep_res_train.py \
  --size 224 --z_mode patch --batch 16 --accum 1 \
  --warm_steps 3000 --steps 5000 --warmup 100 \
  --lr 1.5e-4 --enc_lr_mult 1.0 --head_zero_init --eval_every 500 \
  --train_dir <DIV2K_train_HR> --val_dir <DIV2K_valid_HR> \
  --out_dir /root/autodl-tmp/srres/out_224_A3000 \
  > /root/logs/224_A3000.log 2>&1 &
```
判据：**若 16 步曲线开始下降（不再停在 ~38 的平凡解水平）⇒ 原报告"多步递归训不起来"的结论被推翻，
根因是 A 相预算 / A→B 两相切换**；若仍平 ⇒ 再查 carry 归一化（`model_v2.py:596`）与 register 读出。
