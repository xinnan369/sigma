# R2d-E1 — MFNet 官方 393 对独立留出测试

状态 **COMPLETE**。在同一台 RTX 3090 上完成五个训练种子、三种融合架构、每组 393 对图像，共 5,895 次完整前向。CPU 独立复算全部通过。

## 五种子结果

表中数值为百分比，离散度为五个独立训练种子的样本标准差。每个种子先在完整官方 Test 上汇总混淆矩阵，再计算 IoU。总体 mIoU 包含类别 0，前景 mIoU 取类别 1–8 的均值。

| 融合架构 | 总体 mIoU | 前景 mIoU |
|---|---:|---:|
| SSM | 54.1466 ± 0.6962 | 48.6843 ± 0.7849 |
| CNN | 53.6704 ± 1.3855 | 48.1515 ± 1.5524 |
| Attention | 53.5182 ± 0.8916 | 47.9809 ± 1.0005 |

### 九类 IoU（%）

| ID / 类别 | GT 像素 | SSM | CNN | Attention |
|---|---:|---:|---:|---:|
| 0 unlabeled | 112,023,545 | 97.8448 ± 0.0928 | 97.8209 ± 0.0958 | 97.8169 ± 0.1136 |
| 1 car | 4,854,504 | 88.0005 ± 0.4866 | 87.4323 ± 0.6562 | 87.2476 ± 1.0471 |
| 2 person | 1,345,972 | 69.0235 ± 2.7111 | 69.2892 ± 2.3601 | 68.7615 ± 1.3827 |
| 3 bike | 1,261,567 | 62.3799 ± 1.1305 | 61.9881 ± 1.0822 | 62.0277 ± 1.0996 |
| 4 curve | 602,620 | 40.8243 ± 0.9742 | 40.9039 ± 1.4095 | 40.3601 ± 1.5165 |
| 5 car stop | 355,314 | 31.1728 ± 4.3758 | 30.7567 ± 5.1598 | 30.5530 ± 4.1911 |
| 6 guardrail | 17,794 | 7.3609 ± 2.9187 | 5.5736 ± 3.1485 | 5.6733 ± 2.6085 |
| 7 color cone | 85,246 | 44.4076 ± 1.8077 | 44.0831 ± 0.9855 | 42.4616 ± 0.7578 |
| 8 bump | 183,038 | 46.3047 ± 3.0037 | 45.1855 ± 8.7522 | 46.7623 ± 3.5995 |

### 逐种子结果

| Seed | 架构 | 总体 mIoU | 前景 mIoU | 用时（秒） | 证据 |
|---:|---|---:|---:|---:|---|
| 3407 | SSM | 54.9356% | 49.5734% | 195.27 | [九类 IoU 与混淆矩阵](evidence/r2d-e1/seed3407/ssm.json) |
| 3407 | CNN | 55.1601% | 49.8205% | 178.68 | [九类 IoU 与混淆矩阵](evidence/r2d-e1/seed3407/cnn.json) |
| 3407 | Attention | 54.2643% | 48.8093% | 188.72 | [九类 IoU 与混淆矩阵](evidence/r2d-e1/seed3407/attention.json) |
| 3408 | SSM | 54.1887% | 48.7351% | 192.78 | [九类 IoU 与混淆矩阵](evidence/r2d-e1/seed3408/ssm.json) |
| 3408 | CNN | 54.7139% | 49.3288% | 184.22 | [九类 IoU 与混淆矩阵](evidence/r2d-e1/seed3408/cnn.json) |
| 3408 | Attention | 53.7106% | 48.2161% | 187.96 | [九类 IoU 与混淆矩阵](evidence/r2d-e1/seed3408/attention.json) |
| 3409 | SSM | 54.4695% | 49.0338% | 187.09 | [九类 IoU 与混淆矩阵](evidence/r2d-e1/seed3409/ssm.json) |
| 3409 | CNN | 53.7802% | 48.2737% | 185.47 | [九类 IoU 与混淆矩阵](evidence/r2d-e1/seed3409/cnn.json) |
| 3409 | Attention | 54.2779% | 48.8226% | 188.40 | [九类 IoU 与混淆矩阵](evidence/r2d-e1/seed3409/attention.json) |
| 3410 | SSM | 54.0911% | 48.6384% | 192.76 | [九类 IoU 与混淆矩阵](evidence/r2d-e1/seed3410/ssm.json) |
| 3410 | CNN | 51.6912% | 45.9434% | 177.81 | [九类 IoU 与混淆矩阵](evidence/r2d-e1/seed3410/cnn.json) |
| 3410 | Attention | 53.2015% | 47.6358% | 190.31 | [九类 IoU 与混淆矩阵](evidence/r2d-e1/seed3410/attention.json) |
| 3411 | SSM | 53.0480% | 47.4406% | 192.40 | [九类 IoU 与混淆矩阵](evidence/r2d-e1/seed3411/ssm.json) |
| 3411 | CNN | 53.0064% | 47.3914% | 184.97 | [九类 IoU 与混淆矩阵](evidence/r2d-e1/seed3411/cnn.json) |
| 3411 | Attention | 52.1367% | 46.4205% | 184.34 | [九类 IoU 与混淆矩阵](evidence/r2d-e1/seed3411/attention.json) |

### 同种子架构配对差（百分点）

| 配对比较 | 指标 | 3407 | 3408 | 3409 | 3410 | 3411 | 均值 ± 样本 SD |
|---|---|---:|---:|---:|---:|---:|---:|
| CNN − Attention | 总体 mIoU | +0.8958 | +1.0032 | -0.4977 | -1.5102 | +0.8697 | +0.1522 ± 1.1156 |
| CNN − Attention | 前景 mIoU | +1.0111 | +1.1127 | -0.5489 | -1.6924 | +0.9709 | +0.1707 ± 1.2473 |
| SSM − Attention | 总体 mIoU | +0.6712 | +0.4780 | +0.1915 | +0.8896 | +0.9113 | +0.6283 ± 0.3015 |
| SSM − Attention | 前景 mIoU | +0.7641 | +0.5190 | +0.2112 | +1.0026 | +1.0201 | +0.7034 ± 0.3427 |
| SSM − CNN | 总体 mIoU | -0.2246 | -0.5252 | +0.6893 | +2.3998 | +0.0416 | +0.4762 ± 1.1650 |
| SSM − CNN | 前景 mIoU | -0.2471 | -0.5937 | +0.7601 | +2.6950 | +0.0492 | +0.5327 ± 1.3075 |

SSM 的总体和前景 mIoU 在五个种子上均高于 Attention，在三个种子上高于 CNN。SSM 与 CNN 的总体 mIoU 配对差在 −0.5252 到 +2.3998 个百分点之间，效果方向随种子改变。

五个训练种子共享同一官方测试集。上述标准差描述训练种子间波动，独立训练重复数为五。真实场景和采集时间的独立性未经核实。三组均保留共享编码器及解码器。评估范围为 MFNet 官方留出的干净输入。

## 输入、执行与独立审计

- 以项目 main `1478fab03d3dff8e760f11ff4f5f8f59c677c0a0` 为基线。已审阅用户提供的 ZIP，原 ZIP 和导入成员 SHA256 保存在 [包来源记录](evidence/r2d-e1/package.json)。
- 393 个 ID 按官方 `test.txt` 原顺序，逐一复核 1,179 个 RGB、Thermal、Label 文件的大小与 SHA256。官方 ID 与 R1g 训练和内部验证 ID 的交集均为空。每组共 120,729,600 个有效像素，忽略标签像素为 0。[输入契约](evidence/r2d-e1/input-contract.json)
- 15 份本机 `last.pt` 均对应固定的 Epoch 5、785 次更新，与 R2d 权重账本一致。分割权重严格加载，推理前后复核文件与模型张量 SHA256。GPU 日志中的预期 missing / unexpected keys 出现在构建骨干的 VMamba 分类初始化阶段，完整的 R2d 分割 state_dict 随后以 `strict=True` 加载。[检查点预检](evidence/r2d-e1/checkpoints.json)
- Sigma 源码固定为 `98ba3c6938d29169bbd27f91519084bb462561df`。使用 R2d 已有的单尺度 480×640、无翻转预处理，保持 RGB 读取和 Thermal 灰度复制规则。单次整图推理在 `.eval()` 和 `torch.inference_mode()` 下运行。测试过程没有训练、微调、重新选模或替换 checkpoint。
- 通过逐图预测哈希、固定 ID 顺序及 9×9 混淆矩阵，使用独立标准库实现复算 15 组指标、九类统计及同种子配对差。机器可读结果包含全精度值。[汇总 JSON](evidence/r2d-e1/summary.json) · [独立复算](evidence/r2d-e1/independent-audit.json) · [文件 SHA256](evidence/r2d-e1/artifact-sha256.json)
- 原始数据、权重、预测图像均留在本机。GitHub 归档逐图计数、预测哈希与日志。

## 运行记录

本机时间 2026-11-21 18:14:31 +0800 生成完整汇总。各模型加载、推理、落盘及校验耗时合计 **2811.19 秒**（46.85 分钟），不含前置数据和权重清单预检。运行时同一 GPU 上存在其他任务，耗时仅记录本次实际执行。启动脚本末行的回车进入输出目录名，进程成功退出后将该目录改为正常名称，文件内容照原样归档。

- WSL2 Ubuntu 22.04，RTX 3090，驱动 610.88，Python 3.11.11，PyTorch 2.2.2+cu121，CUDA 12.1，cuDNN 8902，NumPy 1.26.4，OpenCV 4.9.0。原 `selective_scan_cuda_core` 二进制 SHA256 为 `833021baeecf26b84d5ecba208417893a9eda202c38854fd9a9737c8ccd7513e`。[环境记录](evidence/r2d-e1/environment.log) · [GPU 日志](evidence/r2d-e1/gpu.log)
- WSL 固定推理环境的 R2d-E1 专项测试 **15 通过**，编译检查及独立复算的退出码均为 0。首次全套测试记录为 159 通过、3 跳过、1 失败，失败来自历史 LLVIP 扫描对 NumPy 2 的 `bitwise_count` 依赖。R2d 运行环境保留原 NumPy 1.26.4，仓库 CI 使用 `requirements-r0.txt` 的 NumPy 2.3.5。[最终专项日志](evidence/r2d-e1/cpu-final-targeted.log) · [首次全套日志](evidence/r2d-e1/cpu.log)

可复现命令（从仓库根目录运行，输出目录须为新目录）：

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
python -u -m experiments.r2d_official_test.run \
  --root /mnt/d/download/sigma-r1-assets/MFNet \
  --upstream /home/xinnan/sigma-r1-upstream \
  --init /mnt/d/download/sigma-r1-assets/pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth \
  --checkpoint-dir /home/xinnan/sigma-r2d-runs \
  --output /mnt/d/work/sigma-r2d-e1/output

python -m experiments.r2d_official_test.audit \
  --output /mnt/d/work/sigma-r2d-e1/output \
  --audit-file /mnt/d/work/sigma-r2d-e1/independent-audit.json
```
