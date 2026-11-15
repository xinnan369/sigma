# R2i — 可靠性感知 RGB-T 融合

R2h PR #16 的验收 Head 与两项 CI 均匹配，main 仍处于原验收基线。已合并至 `78390978454698712720134f6c7e68df0289608c`。[R2h PR](https://github.com/xinnan369/sigma/pull/16)

R2i 假设、比较、Clean 非劣性边界、增强及 Pilot 预算在 GPU 运行前冻结。四尺度门控只使用真实 RGB/Thermal 特征。等容量对照改用 RGB 特征生成输入描述，保留同形状评分和两模态融合分支。两组各增加 16 个可训练参数，执行相同的门控运算。FLOPs 按门控运算约定统计，指数类算子调用单列，原模型 selective scan 的精确底层硬件 FLOPs 未测量。

每个训练策略包含三组，单组训练 24 条冻结训练记录。只使用内部验证像素，正式测试留待一次独立确认。历史使用过的 157 张内部图不能作为盲测。所有数值结果仅用于工程评估。

Windows 95 通过、4 跳过，WSL 99 通过。GPU 结果待完整归档。[预注册](../tasks/R2i_reliability_aware_fusion.md)
