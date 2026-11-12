# R2f — Selective Scan 数值一致性修复

R2e 归档验收提交为 `f87cd509f265527ba36994745e4d9a512d718ba9`，合并提交为 `f9d5f6fc7a9cb619fc8d07c921d3d465f35cee9a`。R2e 的 pilot 状态为 FAIL，全量归因为 NOT RUN。

固定 R2d seed 3407、SSM、图 `00057D`，审查 `backbone.vssm.layers.2.blocks.2.op` 的 RGB 扫描。沿用 R2b 的混合误差阈值 `2e−5+2e−4|CUDA|` 及上限 1。原失败证据在 R2e 目录保留。逐时间步的 CUDA 状态由单位基向量 C、零直接项的扫描读出获得，并以块末端前缀交叉核对。比较原 CUDA、R2b Triton、独立 FP32 顺序递推。只记录统计和坐标。

候选修复使用 CUDA 原算术获得状态，读出按状态索引显式执行 FP32 FMA。用逐项 FP64 运算后舍入至 FP32 的独立读出核验八种组合。先覆盖合成的不同长度、分组和输入幅度，再执行 R2e 的原定 12 图三架构 pilot，核对四尺度、实际方向、R2d 两个端点预测哈希、Identity 与 Hook 清理。R2f 的证书使用独立的实现指纹。

本阶段执行范围止于 pilot 和数值验证。全量归因另行验收，权重、数据与原实验结果保持只读。
