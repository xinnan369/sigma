# R1d 单 Epoch 独立训练协议

## 固定资产与训练

使用已验收 R1c 的 1410 条训练记录和 79 张验证原图。启动时重算两份清单和已审计 PNG 的 SHA256，仅打开训练与内部验证 triplets。393 条 test.txt 仅用于划分核对。代码从固定上游 `98ba3c6938d29169bbd27f91519084bb462561df` 导入模型和 TrainPre，使用已核验的官方 VMamba-T 分类初始化。新实验的分割参数独立初始化，续行只读取本次保存的 checkpoint。

[配置](../configs/R1d_epoch.json) 固定 seed 3407、FP32、microbatch 1、累积 8、AdamW lr 6e-5、betas (0.9, 0.999)、权重衰减 0.01（R1c 无衰减组仍为 0）。进程内单卡 cuda:0，关闭 TF32。驱动在开始时要求实际分组逐名称、形状、decay 与 R1c 的 668 张量清单相同。

每 epoch 以独立 generator 生成一次排列。每次累积取游标后的最多 8 条，取到末尾返回短窗口，下一轮必须显式开始。本任务固定 1 epoch，完整窗口 176 个，尾窗 2 条，更新共 177 次。每个微批的 mean CE 乘有效像素数后反传，更新前统一除以本窗口有效像素总数。类别 0 有效，增强后 padding 255 排除。沿用 R1c 的上游 BGR 读法、灰度热图复制、原始归一化以及镜像、缩放、裁剪与 padding。

## Checkpoint 与恢复

每 32 个完成的优化器窗口以及本次调用的停止点保存 last.pt。文件写入并 fsync 后在同目录原子替换。checkpoint 包含严格模型状态、AdamW 状态、恒定倍率 LambdaLR 状态、epoch、排列、游标、采样器 RNG、全局 Python/NumPy/torch CPU/CUDA RNG、累计训练损失和类别像素统计。签名固定配置、关键源码、数据摘要、环境与 CUDA 扩展哈希。锁防止两个进程同时写同一目录。

首次运行在第 96 次更新停止，再从新进程恢复。第 96 步只保存状态，验证仅在 epoch 完成时运行。异常中断后从最近保存的更新边界恢复，未保存的窗口重新计算。验证前已保存第 177 步状态，可在验证中断后恢复至只做内部验证。GPU 数值逐位一致性不在本轮验证范围内。

## 内部验证与指标

79 张图原尺寸、单尺度、无增强。混淆矩阵行是真值、列是预测，类别为 0–8，忽略 255。IoU = TP/(真值数+预测数−TP)，Precision = TP/预测数，Recall = TP/真值数。分母为零记 null。总体 mIoU 纳入类别 0，前景 mIoU 只平均类别 1–8，均排除 union 为零的类别。预测像素占比以全部有效像素为分母。对照来自已验收 R1c 第 13 步的原始混淆矩阵。训练中逐窗口记录增强后标签计数、损失、梯度 L2、实际 LR 和显存。

## 执行

在 bridge 根目录的 WSL 环境设置 `CUDA_VISIBLE_DEVICES=0`。先用既有 R1 venv 运行 `python -m pytest -q -s tests`，设置 `SIGMA_R1D_TEST_UPSTREAM` 和 `SIGMA_R1D_TEST_INIT` 以启用真实模型覆盖测试。

```sh
python -u -m experiments.r1d.train \
  --upstream <UPSTREAM> --init <VMAMBA_T> --root <ASSETS>/MFNet \
  --checkpoint-dir <NEW_EXTERNAL_RUN>/checkpoints \
  --output reports/evidence/r1d/first --stop-after 96
```

第二段使用同样资产和 checkpoint 目录，替换为 `--output reports/evidence/r1d/resume --resume`。每段输出目录需没有 events.jsonl。仅从可信本地 R1d checkpoint 恢复。配置硬限制单轮；官方测试评估与 500 epoch 作业不属于本次实验。
