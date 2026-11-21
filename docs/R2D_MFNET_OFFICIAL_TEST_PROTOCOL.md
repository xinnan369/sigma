# R2d-E1 — MFNet 官方 393 对独立留出测试协议

**任务**：不训练、不微调、不可选择新 checkpoint，不用官方测试结果改参数。

## 测试性质

- 数据集：MFNet，**同一数据集的官方独立 Test，不是跨数据集验证**。
- 官方测试：`reports/evidence/r1c/lists.json` 冻结的 393 个 ID；官方 `test.txt` 的 SHA256 为 `a236aa6ac7e55ec6b5e419b55c6934439611453939eee2c0bf7b0f0c34917c9d`。
- 三模态文件集合 `RGB/`,`Modal/`,`Label/` 共 1179 件，必须对照 `reports/evidence/r1b/data-manifest.json` 的 1179 个 SHA256 和长度审计。
- 训练集：627 个 MFNet 原始图像，内部验证：157 个，测试 393 个。测试 ID 不得与训练或验证 ID 重叠。禁止把测试用作选择 checkpoint、阈值、调参、映射或后续训练的依据。
- 本协议对官方 Test 的首次完整测评即作为最终结果；如果失败，应完整记录失败原因和重试，不能只公布有利结果。

## 模型和初始化

- 直接加载 R2d 五种子 `3407…3411` × `ssm/cnn/attention` 共 15 个 `last.pt`，即**冻结 Epoch 5**，不使用 best epoch 选择更好结果。
- 使用同一固定 Sigma upstream、原始 VMamba-T 初始化、R2d 固定宽度、同样输入预处理和 `models/backbone/decoder`。
- 每个 checkpoint SHA256 必须逐一吻合 `reports/evidence/r2d/seedNNNN/training/<arm>/checkpoint-ledger.json` 的末轮 SHA256 及最终审计。
- 一律 `.eval()`、`torch.inference_mode()`，推理输出和统计，不计算梯度、不调用 `optimizer.step()`、不修改 BatchNorm 统计值。
- 预处理调用原 `experiments.r1c.pilot.read_sample(..., training=False)`，不自行改通道、颜色空间、归一化、分辨率及标签。

## 指标与统计

- 官方 Test 九类 mIoU（包含背景）、前景八类 mIoU、九个类别的 IoU，以及逐类支持像素。
- 主指标使用固定 Epoch 5 的模型，不以本测试结果选择模型。
- 每个种子每臂汇总所有 393 对测试图像形成一组结果，对五种子计算均值与样本标准差。
- SSM–CNN、SSM–Attention、CNN–Attention 使用**种子配对差**；不得将 393 张图像当作 393 个独立随机种子，不能据 n=5 宣称无保留的显著性结论。
- 每个模型输出 393 个逐图混淆矩阵和预测哈希；仅保存元数据不保存受限图像。
- 输出测试结果必须明确标记 `cross_dataset_validation=false`。

## 验收

1. 15 个 checkpoint 字节校验全 PASS；CPU 单测通过；上游 commit、运行环境和模型加载一致。
2. 1179 个 test 文件哈希、官方 393 ID 顺序完全一致，且与训练、内部验证无重叠。
3. 15 × 393 次无梯度前向成功；每份逐图文件包含 393 个唯一 ID；每组 9×9 混淆矩阵与全部逐图之和一致。
4. 15 份模型权重推理前后 SHA256 和张量摘要不变；模型处于 eval 状态；官方 Test 不参与 checkpoint 选择。
5. 发布完整结果和每份 JSONL 的 SHA256；未经本机原始权重和数据验证之前只报告 `BLOCKED`，不得声称完成测试。

## 重要限制

本测试测量同源官方 holdout 泛化，不代表异源传感器/场景泛化。训练和测试可能仍存在相似场景或时间关联；ID 无交集不等于真实场景完全独立。更强的跨数据集泛化需要 FMB 等独立采集的数据及合格的标签映射。
