# R2b — Sigma 跨模态状态传播机制归因

R2a PR #9 已验收 Head `a38920c2ce73b27c97468ea0bc1a938cac896952`，核对两项 CI 与合并状态后合入 main `ffae823878875fd33d309dff0e2f30cf16c3d53c`。R2b 基于该提交。只使用 R1g Epoch 5 固定分割状态、R2a 原位置主遮挡与原验证划分。该模型源于独立短程训练。

1. 核对真实 fusion 的 CUDA 入口及块末前缀状态。数值闸门要求合成四方向、多个 batch/长度、逐时间步的短序列参考、实际 clean 与 corrupt 输出和 CUDA 分块末端状态全部满足预定误差界。四方向用于覆盖上游布局和核数值测试。实际 Cross/Concat 路径按源码各自的方向数运行。
2. 主条件固定为 `concat.2` 逆向通道组，零基 `t=1199`，即逆序 Thermal 最后一个状态、RGB 第一项之前。将 corrupt 的更新后 `h[t]` 置为同图同时间的 clean 状态，保持本次输入、`Δ`、`A`、`B`、`C`、`D` 及其他方向固定。错误时间对照固定为同方向 `t=0`，即逆序 Thermal 最早一格。Thermal 编码器可将扰动传到该格，故此时间对照允许实测非零。另设 clean→clean 和 corrupt→corrupt 的精确 Identity。
3. 先完成 R2a 固定 12 图。所有观察输出必须逐图复现 R2a，实际状态与输出数值审计、Identity、无 Hook 复跑及独立显式重置校验均须通过。全量启动另要求至少一张 pilot 图完成从指定状态、后续状态、scan 读出、融合输出、logits 到 argmax 翻转的因果链。
4. 统计固定 54 个视觉代理组，seed 3407，10,000 次组级配对 Bootstrap。绝对率和相对未干预 corrupt 的差值均用同一组抽样。按 R2a 定义报告外部新增、纠正、净增、分歧和距离带。只保存时间索引、空间映射、标量统计、计数、混淆矩阵和预测哈希。
5. CNN、Attention 提供相同的双输入 BHWC 接口，Cross 返回两路、Concat 返回一路。A 在固定 R1g checkpoint 的其余模块上临时接入未训练替换，只验证接口与有限输出。B 从官方 VMamba-T 初始化，SSM、CNN、Attention 三臂分别跑一个相同的 8 microbatch 更新，作为梯度和资源可行性检查。正式比较需独立五轮匹配训练，未完成前只报告工程数据。

详见 [冻结配置](../configs/R2b_protocol.json) 和 [报告](../reports/R2b_REPORT.md)。正式 test 图像不在读取范围，checkpoint 只读，R2b PR 等待验收。
