{
  "_WARNING": "本目录的探针脚本与产物是在数据切分 bug 修复**之前**跑的：`make_data(0)`/`make_data(1)` 会各自重抽投影矩阵，训练集与测试集是两个不同结构的世界。因此下面的 R² 与量级数字**不可采信**，仅作方法学留档。权威结果见 ../REPORT_cpu_capacity_vs_loss.md 与 cpu_repro_results.json。",
  "bug": "train/test 必须来自同一数据集切分（见 REPORT 第 4 节坑 #4）",
  "cum": {"path": "probe_residual_cum.json", "train_steps": 30, "trustworthy": false},
  "region": {"path": "probe_residual_region.json", "train_steps": 800, "trustworthy": false}
}
