# R2a — Sigma 跨模态局部扰动传播诊断

R1g PR #8 已验收 Head `4fccc907c70123d70a7f769bd1cdad6d9563998f` 经 CI、SHA 与无冲突核对后合入 main `33d41235f83f0cde2a0ea9e2d8222940a4f2c51b`。R2a 基于该版本。沿用 R0 Experiment 0 的局部干预、响应和错误定义；本轮评估对象按用户任务改为独立短程 Sigma-Tiny、R1g Epoch 5 和冻结的 157 图视觉分组验证集。

- Checkpoint SHA256 `9e1573f050bd0f8319b4e8cf22927d6c31c091f4d63cbfacf666a7f85c6b5159`，划分 SHA256 `4e6508427b401f055ce1878dd99cb2420c6e485a22f172e23d369448095d24c3`。FP32、eval、inference_mode。逐一核对 471 个验证输入文件的哈希。模型和 checkpoint 完整性前后检查。
- 先复跑全 157 张 clean，要求总体 mIoU 四舍五入 61.1498%，且每图混淆矩阵与 R1g 完全相同。任何不匹配停止。再运行固定 12 张 pilot，按 `SHA256("3407:"+id)` 对组首 ID 及组内候选排序，D/N 各取 6 张且 12 组不同。全量重复 pilot 的样本但只计每张图一次。
- 主实验为中心 64×64 Thermal 遮挡 fill 0。三位置各做 Gaussian σ20、seed 3407/3408/3409，及局部右移 dx8、dy0、fill0。高斯的实际 seed 由设定 seed、样本数字部分、N 指示和位置字典序编号组成 SeedSequence；不重复确定性条件。归一化前操作单通道 0–255，再复制至 3 通道并沿用原通道归一化。冻结实际 Boolean mask，记录行优先 true-runs 及 packed SHA256。
- 逐图汇总净错误、新增错误、纠正错误和分歧，以有效 ROI 外像素为分母；另保留按真值类别、全图与外部混淆及外部预测转移矩阵。距离带为至最近 ROI 像素中心的欧式距离 (0,16]、(16,32]、(32,64]、(64,128]、>128 px。Identity 支持为空，不划距离带。
- 每个扰动条件记录四尺度 CrossMamba 与 ConcatMamba 的 BHWC 输入和输出。归一化差异为差值 RMS / clean RMS，逐格响应为通道 L2 / clean RMS。最大池映射 ROI，报告外部响应和占比，以及内外平均幅值。零总响应比例记 null。每次仅保留一图的 clean 快照，corrupt 只存标量。
- 主遮挡条件逐一替换八个模块的输出，CrossMamba 同时替换 tuple 两分支。Pilot 对每图每模块验证 clean→clean 替换，额外复跑无 Hook clean 和主遮挡。全 157 张验证输入 identity 预测逐像素相同。只称模块输出替换。
- 固定 54 视觉组，seed 3407，10,000 次组级 percentile bootstrap。同组 clean/corrupt、替换及对照同权重抽样。率用累计计数的比值，IoU 先合并矩阵再计算。主遮挡为主要结果；距离、位置、种子和逐模块替换为探索对照，不作 SSM 特异性或隐藏状态推论。

运行依次使用 `python -m experiments.r2a.run clean|pilot|full --upstream <UPSTREAM> --init <VMAMBA_T> --root <ASSETS>/MFNet --checkpoint <R1G_EPOCH5> --output reports/evidence/r2a`，每阶段成功才进入下一阶段。正式测试集、训练、调参均不在范围内。PR 等待验收。
