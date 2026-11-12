# R2g — Full-Scale Causal Attribution of Cross-Modal Selective Scan Paths

## 基线与资产

R2f PR #14 的验收 Head 为 `f469f8af64a4c02c5c6112c09e4267bb9b07ca4d`，合并前两项 CI 成功。已在 `main` 核验合并提交 `a3428a2f85ff0671a6dec423e250a67d24e19a9e` 包含验收 Head，R2f 文件未变更。[R2f PR](https://github.com/xinnan369/sigma/pull/14)

GPU 预检验证了 R2d 的 15 份 Epoch 5 checkpoint、R1g 隔离证书、冻结验证资产、VMamba 初始化权重和上游源码。独立复核了 30 份 R2d 自然端点文件的 4,710 行样本与冻结视觉图的分组映射。实验保持 157 张图、54 个视觉代理组、中心 64×64 Thermal 遮挡。仅用官方测试 ID 文本核查数据隔离，读取官方测试像素数为 0。未训练或写入权重。[组映射审计](evidence/r2g/group-audit.txt)

## 干预与数值闸门

使用 R2f 的原生 CUDA 基向量状态恢复、显式 FP32 FMA 及原容差。每个样本都核查 CUDA 分块末状态、原生读出与八组合读出，验证自然端点的 R2d 预测哈希、Identity logits、坐标与 Hook 清理。使用独立 R2g Pilot 证书开启全量阶段。每图按指纹和结果 SHA256 事务落盘。恢复逐行复核，遇到已保存的失败记录时阻断后续执行。

自然 Corrupt 直接用遮挡图推理。其余非零 mask 都在 Clean 输入上下文中替换指定扫描的读出。即使三项联合替换也保留未干预的 Clean 计算路径。在联合范围中，下游被选扫描使用冻结读出，可绕过上游干预向该扫描的传播。可由此比较固定模型的边界干预效应，无法据此识别所有自然输入中的独立因果贡献，或证明能在真实受扰输入中恢复精度。

## R2g 独立 Pilot

固定 12 张 × 三架构均通过，GPU 为 RTX 3090。两端点重构 3,312 次均与 CUDA 完全一致。CUDA 逐步状态和分块末状态比较 3,312 次，最大混合容差比为 0.033714。八条件独立读出比较 13,248 次，最大比为 0.00414943。72 个自然端点预测与 R2d 一致，252 次 Identity 与 36 次 Hook 清理通过。原逐元素上限仍为 1。

| 架构 | 图像数 | 逐图耗时合计 | 峰值 PyTorch 分配 |
|---|---:|---:|---:|
| SSM | 12 | 278.66 s | 10.78 GiB |
| CNN | 12 | 115.46 s | 9.55 GiB |
| Attention | 12 | 119.95 s | 9.55 GiB |

按 R2g 各架构实测吞吐外推全量约 9.34 GPU 小时，另有加载、写盘与统计开销。全量进程在 Pilot 通过后启动。恢复验证使用独立复制的 Pilot 日志，重新核对 36 条记录，三份压缩结果均逐字节一致。[恢复日志](evidence/r2g/pilot-resume.txt)[Pilot 独立审计](evidence/r2g/pilot-audit.json) · [GPU 日志](evidence/r2g/pilot-gpu.txt) · [独立 Pilot 证书](evidence/r2g/pilot/certificate.json)

## 执行状态

R2f 的 36 次 Pilot 测量为 542.743 秒、峰值分配 10.79 GiB。按 2,355 次图像、种子、架构组合线性外推约 9.86 小时。R2g 新增共享、融合聚合范围及逐图检查，实际耗时以 R2g Pilot 与全量日志为准。

全量五种子结果、主效应和交互效应、10,000 次双层 Bootstrap、seed-level t 区间均等待完整审计后的分析输出。主要科学问题 A–F 在证据完成前保持开放。各尺度与交互项属于探索性分析，区间未校正多重比较。54 个分组为视觉代理分组。

[协议](../tasks/R2g_full_scan_attribution.md) · [Windows 测试](evidence/r2g/tests-windows.txt) · [WSL 测试](evidence/r2g/tests-wsl.txt)
