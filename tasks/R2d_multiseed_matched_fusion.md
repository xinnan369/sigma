# R2d — Parameter-Matched Multi-Seed Fusion Robustness Study

R2c PR #11 已验收 Head `16bed87380feae7d1550d1c68c8751772460edb2` 经两项 CI 与冲突核对后合入 main `9c5b49fd227291f970771e582ae0ae26ecfe07b5`。R2d 基于该提交。

- 原 SSM 融合作参考。CNN 和 Attention 仅调整内部宽度，沿用 R2b 的双输入 BHWC 接口、残差和四头注意力。对所有尺度使用同一宽度倍率，按 8 对齐。静态搜索以全模型总参数误差最小为目标，验证数据和模型效果不参与搜索。合成输入审计实际参数、MAC、状态更新数、FP32 推理延迟、峰值显存和八样本训练更新。
- 目标 seed 为 3407–3411。每种子三臂均从官方 VMamba-T 分类权重重新构建。R1g 划分、R2c 训练协议、每轮 1,254 条记录和 157 次更新、六样本尾窗及预先固定 Epoch 5 比较终点保持一致。各种子分别核对公共参数、首窗实际输入和全程采样、增强及有效像素日志。
- 三臂先通过各自首轮和最终独立审计，才对同一组 Epoch 5 checkpoint 运行全部 14 项 R2a 预测条件。Best checkpoint 仅用于次要 clean 结果。中心遮挡的 ROI 外新增及净错误率为共同主指标。
- 10,000 次 Bootstrap 同时抽种子和视觉组。由于同一验证集用于所有种子，每次重复中对每个种子使用同一组场景抽样，架构之间共享种子和组权重。先在每种子汇总像素计数并求指标，再取等权种子均值。记录种子间样本标准差、逐种子效应方向和配对区间。三至四个完整种子只支持探索性结论。

冻结配置见 [R2d 配置](../configs/R2d_protocol.json)。运行先执行 `experiments.r2d.profile`，再以同样的资产路径执行 `experiments.r2d.run --seeds 3407 3408 3409 3410 3411`，两者均接收 `--upstream --init --root --evidence`，训练另接收外部 `--checkpoint-dir`。完成后执行 `experiments.r2d.analysis --evidence`。正式 test 图像及已有分割权重不进入训练或评估，训练权重留在仓库外。
