# 独立重新训练方案（计划，未启动 MFNet 训练）

这不是作者 checkpoint 的复现，也不能用初始化权重或合成探针结果填充 clean baseline。

## 初始化与数据

- 固定 Sigma commit，沿用已经实际测试的 torch 2.2.2+cu121 / torchvision 0.17.2+cu121 / timm 0.4.12 / einops 0.8.1；保存 R1 包清单、扩展 fingerprint、Git tree 和所有随机种子。
- 官方 VMamba-T 初始化已从固定上游 Git LFS 原始文件恢复；SHA256 `dbc0cc4f5ec0e45db5fba7c939d2c7d9b617e891ac10766912a8d604c37c5e47`，91,649,482 bytes。只初始化共享 backbone，不是 MFNet 分割模型。
- 固定 config 使用 train2.txt：1176 唯一原始样本，无 `_flip`，SHA `503cdbc99aaab424db5b1f33c5be1e163e46ec9df90256919ad05f1d3f6530c1`。与 393 test ID 无交集。训练图像还未批量获取/审计。
- 作者在 issue #37 推荐 train.txt 做公平比较；本次取得的 train.txt 含 1568 条（784 原图 + 784 flip），SHA `e14ef52501a5d0000361e79acb42734f391427ac5a4164ad7864b0d2718269f7`，与 test 无交集。不得把两种训练 regime 混称同一复现。

## 配置与必须先解决的问题

固定配置的参照：480×640，batch 8，500 epochs，148 updates/epoch，AdamW lr 6e-5、betas (0.9,0.999)、weight_decay .01，poly power .9，warmup 10 epochs，CE ignore 255（类别 0 有效），原 TrainPre scale/flip/crop/normalization 不变。

实际审计发现上游 group_weight 没将 6,407,808 个参数加入 optimizer；完整遗漏名称已在资源日志中保留。必须先决定是严格复刻上游分组，还是审核后修正完整参数覆盖。修正会构成新的独立训练协议，不能静默修改并称作者复现。

上游单卡 train.py 的评估设备硬编码为 [1,2,3]，不适用于当前单张 3090；还在训练过程中反复用 test split 选 best checkpoint。独立训练必须另设训练内 validation、预注册 checkpoint 选择规则，最终 test 只作一次报告；不得把 test 重新划分或根据 test 调参。需要单独审核训练驱动修复，官方 eval.py 和测试协议保持原样。

## 资源可行性

- 真实合成 FP32 microbatch=1 一步 forward/backward/AdamW 通过，峰值 allocated 6,731,221,504 bytes（约 6.27 GiB），reserved 约 6.48 GiB；单次冷步 16.59s，不能当稳定吞吐。
- 实测 batch=8 OOM，并造成 WDDM/WSL 共享主机内存压力；PyTorch 记录的约 49 GiB allocator peak 不是物理 GPU 显存容量。不得在当前环境重复默认 batch=8。代码默认已改为只测 batch=1，并逐步即时输出日志。
- 候选：单卡 microbatch=1、梯度累积 8（有效 batch 8），或多卡每卡更小 batch。累积会改变随机序列、归一化/优化器更新行为，须明确写为独立训练变体，先验证梯度与完整参数覆盖。AMP 可另行试验，但不能无说明启用。
- 预计最低资源：一张 24GB GPU、至少 16GB 空闲主机 RAM（建议 32GB），独立 SSD 工作区预留 10–20GB（数据、last/best 模型与优化器状态、日志的预算，并非实测训练总占用）。microbatch=1 的显存证据支持开展小规模 pilot，不保证修正优化器/AMP/数据流水线后占用完全一致。
- 时间预算必须由不参与最终 test 的 100 个真实训练微步 pilot 给出。若稳定微步耗时为 t 秒，train2 regime 的 500 epochs、148 updates/epoch、累积 8 的纯计算量级为 `74000 × 8 × t / 3600` GPU-hours，另加 I/O、validation 和 checkpoint 开销。本轮只有受冷启动/资源压力影响的单步数据，因此不提供虚假的完整训练 ETA。

## 复现与验收

冻结 train/validation ID 和 hash、seed（3407，随后至少另两种种子）、全部配置、包清单、初始化键匹配；保存每步 lr、loss、峰值显存、失败/恢复日志、optimizer/scaler/RNG/sampler 状态。禁止只保存最佳 test 分数。Pilot、训练驱动修复与完整训练需要后续科研任务明确批准；本轮未启动 500 epochs，也没有训练所得分割分数或 checkpoint。
