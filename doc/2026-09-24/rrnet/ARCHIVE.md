# 归档说明 — rrnet 12 组循环 ResNet 重建实验

- 归档时间：2026-09-25
- 来源机器：AutoDL `connect.westc.seetacloud.com:49687`（1× RTX 4080 SUPER 32GB），工作目录 `/root/autodl-tmp/rrnet/`
- 运行时间：2026-09-24 `15:03:31` → `18:11:17`（3 h 08 min），**12/12 组 `rc=0`**
- 完整性：`code/` 与 `results/` 共 15 个文件已与远程实例逐文件 **md5 比对，全部一致**（无修改）

## 目录

| 路径 | 内容 |
|---|---|
| `REPORT.md` | 完整实验报告（架构、协议、结果、结论、诚实边界） |
| `README.md` | 实验总览与复现入口 |
| `code/` | 模型 / 训练 / 数据 / 自检 / 编排 / 汇总脚本 |
| `results/` | 总表 `summary.{md,csv,json}`、`curves_val_l1.png`、`samples_{224,448}.png`、`mean_baseline.json` |
| `runs/<backbone>_<res>/` | 每组原始记录：`metrics.json`（最终指标）、`history.json`（40 epoch 逐轮）、`config.json`、`train.log` |
| `logs/` | 流程日志：`run_all.status`（12 组 BEGIN/END/rc）、`bench.json` 测速、`prep_data.log`、`hf_download.log` 等 |

## 未包含 / 注意

- **权重未入库**：`best.pt` / `last.pt` 合计 **5.0 G**（每组 best 与 last 同尺寸），被 `.gitignore` 的 `*.pt` 规则排除。
  仍保存在实例数据盘 `/root/autodl-tmp/rrnet/ckpt/`（跨关机保留），**本地未镜像**（工控机剩余空间不足）。
- `runs/*/train.log` 与 `logs/*.log` 原本被 `.gitignore` 的 `*.log` 排除，本次为保留实验证据用 `git add -f` 强制入库。
  其中 `train.log` 含每次运行的 `load_report` 行，是 REPORT.md 第 3 节「预训练权重 100% 命中」的直接证据。
- `code/` 与 `results/` 的文件内容与远程实例逐字节一致，未做任何修改。
