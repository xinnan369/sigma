# R2h — Corrupt-Context Counterfactual State Repair

R2g PR #15 的验收 Head 为 `35e19f0f57a01dd2d02cb3f6ccb9ece900f54018`，CI 2/2 成功。已合并至 `main`，合并提交 `2bebdbf363a6cb66d208dd5cd13b08ac3fd3498b`。[R2g PR](https://github.com/xinnan369/sigma/pull/15)

R2h 固定 R2g 的验证集和 15 份 Epoch 5 权重，使用同一张 RGB 与中心遮挡的 Thermal 执行自然 Corrupt 和全部修复回放。主要干预为共享编码器 mask 6，仅将 h 换回同图 Clean 值。错配对照使用冻结 ID 顺序的下一张图的 Clean 编码器 h。自然 Corrupt 用于衡量修复引入和纠正的错误，Clean 用于识别遮挡新增的错误。

共享编码器 h-only 的主要指标为 ROI 外净错误率变化。报告新增错误恢复比例、修复引入错误数、分组指标和全图表现。一个干预可能同时纠正遮挡新增错误并引入更多错误。结果以净错误变化和置信区间共同判断。各干预使用固定模型及固定扫描边界，范围与交互比较涉及多重比较。

## 执行状态

全量和配对统计待完成。当前不据 Pilot 或部分样本判断修复有效性。未启动训练，未读取官方测试图像。

[冻结协议](../tasks/R2h_corrupt_context_repair.md)
