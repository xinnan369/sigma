# R1c 执行报告

2026-11-08 启动，2026-11-09 续行（Asia/Shanghai）。实验范围为从官方 VMamba-T 初始化开始的 Sigma-Tiny 独立重新训练 pilot。

## 前置版本与环境

核对已验收 PR 的原 Head、CI 成功及无冲突后，依次将 R1 #2 和 R1b #3 合入 main。合并提交为 `9fe51d16ebbb933df7bbedc40b53df86cd5d1901`、`83a8d6454a7bd6508e901a0aa13ac133923cb665`。R1c 基于后者，合并结果见 [GitHub 记录](evidence/r1c/prerequisite-merges.txt)。

沿用 Ubuntu 22.04/WSL2、RTX 3090 24 GiB、driver 610.88、Python 3.11、PyTorch 2.2.2+cu121、torchvision 0.17.2+cu121、CUDA toolkit 12.6、timm 0.4.12、einops 0.8.1、NumPy 1.26.4、OpenCV 4.9.0。上游固定 `98ba3c6938d29169bbd27f91519084bb462561df`，核对 tracked diff 干净。CUDA 扩展及包版本见 [指纹](evidence/r1c/environment.txt) 和 [构建检查](evidence/r1c/model-inspection.txt)。

VMamba-T 初始化文件 91,649,482 bytes，SHA256 `dbc0cc4f5ec0e45db5fba7c939d2c7d9b617e891ac10766912a8d604c37c5e47`。构造器加载后再显式检查 state_dict，8 个 outnorm weight/bias missing、4 个 classifier norm/head weight/bias unexpected，均符合分类预训练向分割 backbone 的预期迁移。[LFS 来源与摘要](evidence/r1b/vmamba-init.json)

## 数据与划分

从 Sigma 作者提供的 MFNet Drive 目录获取两张训练清单的并集。保留 PNG 原始字节，逐个核对格式、尺寸、标签和哈希。原始资产保存在本地外部目录。来源与原始顺序见 [列表核对](evidence/r1c/lists.json) 和 [公开文件 ID](evidence/r1c/source-index.json)。作者建议使用 train.txt，固定源码配置使用 train2.txt，两者须分别报告。[作者说明](https://github.com/zifuwan/Sigma/issues/37)

| 官方清单 | 样本 | SHA256 |
|---|---:|---|
| train.txt | 784 原图 + 784 `_flip` | `e14ef52501a5d0000361e79acb42734f391427ac5a4164ad7864b0d2718269f7` |
| train2.txt | 1,176 原图 | `503cdbc99aaab424db5b1f33c5be1e163e46ec9df90256919ad05f1d3f6530c1` |
| test.txt | 393 原图 | `a236aa6ac7e55ec6b5e419b55c6934439611453939eee2c0bf7b0f0c34917c9d` |

两训练表共有 784 个原图身份，并集含 1,960 个样本，预期 5,880 个文件。两表去掉 `_flip` 后均与 test 无交集。Pilot 采用 train.txt，按原图身份及 day/night 分层哈希划分，79 原图用于验证，其 flip 同时排除，剩余 705 原图与各自 flip 共 1,410 训练条目。验证划分冻结于模型运行前。[固定划分](evidence/r1c/split.json)

WSL 首次下载因网络不可达退出 1，改用 Windows。首个 Windows 进程在切换为可复用连接的下载器时主动停止，保留进度和退出记录。恢复下载验证已有文件，并完成全部 5,880 个任务，失败为 0。[WSL 首次日志](evidence/r1c/data-download-audit.txt) · [Windows 首次日志](evidence/r1c/data-download-windows.txt) · [恢复下载日志](evidence/r1c/data-download-resumed.txt)

WSL 严格审计退出 0，5,880/5,880 文件均为原始 640×480 uint8 PNG，累计 835,055,430 bytes。RGB 三通道，热图和标签单通道，PIL 标签与 OpenCV 灰度读法一致，类别范围 0–8，原始 ignore 像素为 0。逐文件路径、来源 ID、尺寸和 SHA256 均已记录，规范化 JSON 文件记录聚合 SHA256 为 `981e56ff3c783f4659e41ca235bd568ff9919a078661d8d929e3fdff0633591d`。[文件清单](evidence/r1c/manifest.json) · [执行日志](evidence/r1c/data-audit.txt)

内部验证含 41 张 day、38 张 night，划分 SHA256 `397a4fa60b692a59cc4747b72807f7514416ea304c2af556c8b5fc259d75f8e3`。训练并集与最终测试、内部训练与验证分别比对 RGB/热图文件哈希对，精确重复均为 0。最终测试只读取固定列表和 R1b 已生成的哈希元数据。该检查覆盖原图身份及相同文件对，无法判断近邻帧或场景的关联。[泄漏审计](evidence/r1c/leakage-audit.json)

## 驱动修正与协议差异

上游 group_weight 按 Module 类型选参数，漏掉直接注册的 SSM 投影、状态和时间步参数。R1c 保留原分组，把遗漏的张量按 `_no_weight_decay`、一维或 bias 规则补入对应组，再按参数对象身份断言覆盖恰好一次。[逐参数分组](evidence/r1c/prepilot-coverage.json)

| 分组 | 参数张量 | 参数元素 | 遗漏元素 |
|---|---:|---:|---:|
| 原 group_weight | 473 | 41,879,552 | 6,407,808 |
| R1c 完整分组 | 668 | 48,287,360 | 0 |

新增的 195 个张量中，133 个不施加权重衰减，62 个施加 0.01 权重衰减。重复加入的参数数为 0。

使用进程可见的唯一 `cuda:0`，移除上游验证设备 `[1,2,3]` 的依赖。训练预处理继续调用固定上游 TrainPre。采用 FP32 microbatch=1、8 次梯度累积，按有效像素数汇总 CE 梯度。AdamW 使用恒定 lr 6e-5，作为短 pilot 配置。每步核对有限梯度和实际参数变化，按内部验证 mIoU 选择最佳 checkpoint。保存和恢复包含严格模型键、优化器、采样顺序和游标、Python/NumPy/torch CPU/CUDA RNG 及配置/资产/驱动签名。[完整协议及命令](../docs/R1c_TRAINING_PROTOCOL.md)

## RTX 3090 真实 pilot

104 个不同训练条目，13 次 AdamW 更新，4 次 79 图内部验证。前段完成 64 微步并退出 0，第二个进程从第 8 步的 last.pt 继续至第 13 步，退出 0。恢复文件 SHA256 与第 8 步保存记录一致，模型键全部匹配，恢复 668 个优化器状态和采样游标 64。两段合计 146.39 秒训练循环墙钟，包含内部验证与保存。两段完整进程耗时另见命令日志。[前段](evidence/r1c/pilot-first.txt) · [恢复段](evidence/r1c/pilot-resume.txt) · [交叉核对](evidence/r1c/pilot-verification.txt)

每一次更新都观察到 668/668 参数张量具有有限、非零 L2 梯度，且每个张量至少有一个元素实际变化。13 个窗口的梯度 L2 为 1.5103–59.2759。首末窗口的有效像素加权训练 CE 分别为 3.24290 和 0.43922，来自不同样本和增强，不能据此估计收敛。两个逐张量审计记录包括非零梯度和变更元素数上界，张量通过不代表每个标量每步都变更。未观察到 OOM 或非有限数值。[优化器覆盖](evidence/r1c/pilot/parameter-coverage.json) · [梯度审计](evidence/r1c/pilot/gradient-audit.json) · [恢复后梯度审计](evidence/r1c/pilot/resume-gradient-audit.json)

| 优化器更新 | 验证 CE（逐有效像素） | 79 图内部 mIoU（含类别 0） | 验证墙钟 |
|---:|---:|---:|---:|
| 4 | 0.46812 | 10.2554% | 16.18 s |
| 8 | 0.40961 | 10.2572% | 19.92 s |
| 12 | 0.37609 | 10.2706% | 16.12 s |
| 13 | 0.37636 | 10.3142% | 16.26 s |

第 13 步是当前内部验证最佳。九类 IoU 按上游顺序为 unlabeled 92.29597%、car 0.53168%、person 0%、bike 0%、curve 0%、car stop 0%、guardrail 0%、color cone 0%、bump 0%。该次验证 24,268,800 像素中有 24,250,843 个预测为类别 0（99.926%）。这些分数用于核对训练和选模管线，前景分割效果尚未建立。完整混淆矩阵、逐微步 CE、LR、耗时及样本 ID 见 [初段事件](evidence/r1c/pilot/pilot-events.jsonl) · [恢复事件](evidence/r1c/pilot/resume-events.jsonl) · [汇总](evidence/r1c/pilot-summary.json)。

本地 best.pt 和 last.pt 均指向第 13 步内容，单个 580,308,439 bytes，SHA256 `d1c8110ab7acb712f2b9da2fad4fac527ddb8e2ab7f33fed1044566bf609ca8f`。仅提交哈希及日志。[实际文件哈希](evidence/r1c/checkpoint-hashes.txt)

## 资源、可行性与未完成项

- PyTorch 峰值 allocated 7,363,910,144 bytes（6.86 GiB），reserved 7,577,010,176 bytes（7.06 GiB）。为包含逐参数更新审计的进程内分配器计数，GPU 物理容量为 24 GiB。真实 microbatch=1 在本卡完成，物理 batch=8 本轮未运行。
- 微步中位 0.416 s，P95 0.612 s，含读图、预处理、传输及 forward/backward，约 2.40 样本/s 中位。13 个训练窗口含梯度/更新审计共 52.45 s，四次验证 68.48 s，checkpoint 写入与哈希 24.64 s。每次 checkpoint 原子写入另需一个临时文件的空间。
- 若后续批准同一 1,410 条训练划分、每轮完整遍历 500 次，则 705,000 微步。按本轮微步中位外推约 81.5 GPU 小时，按含更新审计的平均 0.504 s/样本外推约 98.8 GPU 小时。二者均未计入完整训练的验证、保存、重启或争用，是量级估算。短 pilot 支持单卡可运行性。后续需固定 epoch/累积尾部规则、调度与预算、降低验证和保存频率，审查场景关联和多 seed 复现。保留 last、best 和一个原子临时文件约需 1.74 GB checkpoint 空间，另计资产及日志。
- 完整训练、最终 393 图 clean test、官方分割权重复现均为 **NOT RUN**。官方分割 checkpoint 仍待获取。R1c 当前只支持有界 pilot，完整训练协议需另行审查。

## 测试与提交

Windows 与 WSL 各 38 项测试通过，退出 0。新增回归覆盖原 SSM 裸参数遗漏、非等量 padding 下的有效像素梯度累积、原图与 flip 分组、类别 0 的指标口径，以及 CPU 的 optimizer/RNG/sampler 断点恢复逐位对照。GPU 的实际恢复验证如上，未做 GPU 逐位连续对照。[Windows 测试](evidence/r1c/tests.txt) · [WSL 测试](evidence/r1c/wsl-tests-final.txt)

PR 等待验收。提交仅含源码、配置、文档、清单元数据和执行记录。
