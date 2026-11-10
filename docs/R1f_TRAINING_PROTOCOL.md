# R1f 视觉分组与重新训练协议

## 视觉审计

对象为冻结的 Sigma 作者 train.txt 中的 784 张原图、784 个作者翻转及 RGB、Modal、Label 三文件。先核验 R1c 文件摘要，并逐像素验证作者翻转。检索使用原图，原图和翻转以同一身份绑定。官方 test.txt 仅做 ID 核对；官方 test 图不进入本次图像筛查或评估。

对全部 306,936 个不同原图对计算 RGB 灰度及热图在 64×48 下的中心化相关性，分别比较原方向和相对镜像。联合分数取同方向下两个模态的较小值。精确解码 RGB/热图对相同、联合分数 ≥0.90，或同 D/N 且数字距离 ≤5 同时联合分数 ≥0.80，均构成分组边。

几何检索候选为逐图的 RGB、热图、联合分数各 Top-8 的并集，再加同 D/N 且数字距离 ≤5 的所有对。对未被上述规则合组的候选，在两种相对方向、任一模态做 320×240 SIFT 匹配与 RANSAC 单应性。Lowe ratio 0.72，至少 18 个内点、内点比例至少 0.55、两边内点凸包均覆盖至少 0.06 图像面积。要求合理投影、共同区域至少 0.60，另一模态配准后相关性至少 0.72。满足即加边。使用连通分量保守合并，包括传递连接。算法阈值于训练前确定并冻结。

## 划分和审核

固定 705 训练原图及其 705 flip，79 个内部验证原图；验证 D/N 各至少 25。训练与验证各类均至少 3 张原图、来自至少 2 个视觉组。SciPy HiGHS MILP 在整组约束下最小化昼夜和各类图像存在数相对目标的偏差，使用小的组序惩罚，目标为 41 张 D 及按 79/784 比例的类别图数（至少 3）。类别标签仅用于预先固定的分层分配，未使用模型指标。

冻结图、样本清单及内容 SHA256。训练启动前重建组，拒绝原图身份、增强、可见视觉边或官方测试 ID 交叉，核对标签支持和所有实际训练/验证 PNG 哈希。这是视觉场景分组代理。无法确认每个捕获时间和采集序列，也无法保证内容检索未漏掉低纹理、视角变化或日夜变化下的同一地点。

## 训练与验证

RTX 3090 上由固定上游 `98ba3c6938d29169bbd27f91519084bb462561df` 加载官方 VMamba-T 分类初始化。保持 R1e 的 Python/CUDA、预处理、FP32、seed 3407、microbatch 1、累积 8、AdamW 恒定 lr 6e-5 及全 668 个张量覆盖。其余分割参数独立初始化，AdamW 初始空状态。使用新输出目录。每轮 176 个 8 样本窗口和 1 个 2 样本窗口，按实际有效像素数归一化。训练总计 5 epoch，轮末在冻结 79 图上计算逐类 IoU、Precision、Recall、预测占比及混淆矩阵。最优快照以总体 mIoU 选择，同分保留先前轮次。独立 best.pt 与 last.pt 留在外部目录。

在既有 WSL CUDA venv 内，从 bridge 根目录运行。先通过隔离单元测试及真实场景审计。

```sh
OPENBLAS_NUM_THREADS=4 python -u -m experiments.r1f.scene \
  --root <ASSETS>/MFNet --output reports/evidence/r1f

CUDA_VISIBLE_DEVICES=0 python -u -m experiments.r1f.train \
  --upstream <UPSTREAM> --init <VMAMBA_T> --root <ASSETS>/MFNet \
  --checkpoint-dir <FRESH_EXTERNAL_RUN>/checkpoints --output reports/evidence/r1f/gpu

python -m experiments.r1f.audit_run --root reports/evidence/r1f \
  --checkpoint-dir <FRESH_EXTERNAL_RUN>/checkpoints
```

配置只允许 5 轮。若运行中断，驱动仅接受同签名的本地 R1f checkpoint，按最近保存的窗口恢复，输出至新的日志目录。已提交的审计复核本次单进程轨迹。官方 393 图不用于验证或选模，恒定学习率短程验证与官方 500 epoch warmup/poly 协议不同。
