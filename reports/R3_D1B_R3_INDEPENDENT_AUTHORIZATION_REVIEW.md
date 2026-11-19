# R3-D1B-R3 — Independent Authorization Audit

**结论：HOLD。** RTX 3090 数值、恢复、开发数据和回执签名均独立复验通过，但基线在固定 WSL CUDA 环境下的完整 CPU 测试有一项失败。暂不进入授权签发；没有登记候选，没有启动正式训练。

- 审计 Head：`3ce9a30c1521df6143cec8d371d5d0b6be579434`（PR #34）。
- 候选 SHA256：`eb002f9b3e4c82bdb6bbab6bb205f8ff5f88c0ddb13cc03b47d0796c9550effa`。
- 实际复跑输出目录：WSL `/tmp/sigma-r3-independent-ViERkJ`；使用本地 LLVIP 和固定 Sigma 目录，未下载数据。
- [独立结论 JSON](evidence/r3-d1b-r3/independent.json)记录每个原始文件摘要、退出码、GPU/软件信息与实测指标。[归档摘要](evidence/r3-d1b-r3/verification.json)另绑定提交内文件。

| 门禁 | 独立结果 |
|---|---|
| 回执及证据 | PASS：从指定 Git blobs 核验全部 R2R 归档 SHA256、六项证据路径与链接、候选 SHA256；核对实际运行源码原始字节、规范化 Git blobs、pinned checkout 源文件和安装的 wrapper/CUDA binary |
| 运行签名 / 批序 | PASS：独立按冻结字段构建并哈希 20 组签名，用 NumPy PCG64 逐轮重算 50 条摘要；核对各轮 2,482 批及 2 图尾批 |
| 协议 | PASS：唯一现行 JSON 合同与正文一致；旧 D 原生阻断条款已被确定性后端取代，跨实现容差与同版本逐位标准分离，独立授权仍必需 |
| CUDA 数值和重复 | PASS：16 个原生级用例复跑，du/ddelta 原生重放不变，原生 dA/dB/dC/dD/ddelta_bias 仍有非确定差异；确定性前/反向重放及前向缓存逐位通过 |
| 整模型 | PASS：batch 4/2，三个尺度前向、155 个可训练参数张量梯度与双输入梯度均覆盖并通过预设对照容差；确定性重复逐位一致 |
| A/B/C/D 恢复 | PASS：独立子进程复跑通过，模型、BN、AdamW、RNG、顺序与事件字节均通过；D 连续 18/72、恢复两进程各 9/36 原生前/反向子调用 |
| 数据和 FCOS | PASS：Fit 9,926 对、Validation 2,087 对；24,026 JPEG、12,013 XML 逐文件校验和解码。匹配 GT 分别 28,540/28,547 和 5,530/5,536，逐图账本逐字节一致，含所有未匹配记录 |
| 限制及信任 | PASS：官方 Test 像素读取、真实 LLVIP 优化器更新均为 0；训练未运行，冻结目录及 `TRUSTED_RECEIPTS=frozenset()` 未改动 |
| 完整 CPU / compileall | **HOLD**：固定 WSL 1 失败、202 通过、3 跳过；compileall 通过。Windows 对照 198 通过、7 跳过；独立证据校验专项 3 通过 |

独立证据读取器 `experiments/r3_d1_r3/independent_audit.py` 不调用签发器的验证函数或训练模块。数值与恢复实跑使用指定工具，独立读取器复核实际用例矩阵、每类梯度、阈值、调用次数及边界，不把工具返回的总状态当作唯一验证。读取器放在签名范围外，没有改变候选绑定的实验生产代码。

R2R 原生级最大混合比 0.0081623，本轮 0.0052231；整模型参数最大绝对差约 7.23e−7 → 8.57e−7，输入梯度约 2.79e−9 → 3.03e−9。浮点原生基准不是逐位稳定；这些差异仍在冻结容差内，确定性重放和恢复没有放宽。逐图 FCOS 账本与此前相同，其 SHA256 `6bef3d384796f98c8f2b8d5f406083b35d4d33202b34c9b2e158762319ba71cd`；避免重复提交该账本，原始副本保留在外部运行目录。数据报告的环境字段不同：R3 数据执行在 WSL，而 R2R 数据检查在 Windows；统计不变。GPU 训练签名环境与候选一致。

WSL 失败发生在历史 LLVIP 感知哈希扫描器的 CPU 单测：调用 NumPy 2 才有的 `np.bitwise_count`，而固定环境为 NumPy 1.26.4。独立 [修复 PR #35](https://github.com/xinnan369/sigma/pull/35) 的隔离工作树在同一 WSL venv 下 **204 通过、3 跳过**。修复预览不替代基线失败，不影响本候选的签名源码；需另行接纳修复并复核合并基线后再考虑签发。

复现：从基线根目录通过固定 venv 顺序或并行执行以下命令，输出及 stdout/stderr 写入全新仓库外目录（CPU 测试隐藏 CUDA；GPU 两项单设备且顺序运行）。

```sh
python -m pytest -q -rs tests
python -m compileall -q experiments/r3_d1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m experiments.r3_d1.deterministic_audit --upstream <PINNED_SIGMA> --output <FRESH>/numerics.json
python -m experiments.r3_d1.resume_audit --backend deterministic --upstream <PINNED_SIGMA> --output <FRESH>/resume.json
python -m experiments.r3_d1.authorization_preflight data --root <EXISTING_LLVIP> --output <FRESH>
python -m experiments.r3_d1_r3.independent_audit --fresh <FRESH> --output <FRESH>/independent.json
```
