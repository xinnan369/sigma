# R3-D1B-R5 — 最终独立训练授权审查

**结论为技术 PASS。** 四臂五种子训练的技术检查通过，可提交独立授权批准。候选回执仍未受信任，正式训练入口仍拒绝运行。

## 基线与审查范围

- [PR #37](https://github.com/xinnan369/sigma/pull/37) 的验收 Head 为 `f800f26ca15b4cafe9b4f38e5cdc031a93f0e5e1`。核对了九个文件的变更范围、目标分支、可合并状态和两项成功的 [Push CI](https://github.com/xinnan369/sigma/actions/runs/44484086318)、[PR CI](https://github.com/xinnan369/sigma/actions/runs/44484137161)。
- PR #37 合并提交为 `70456bf60630cdeb593115316c9997021bd067dc`，目标为 `codex/R3-D1B-R3-independent-authorization-audit`。Tree SHA `1c352ab618d5249f3eeded6d001260df76d69666` 与验收 Head 完全相同。R5 从该合并提交创建。合并后的 [Push CI](https://github.com/xinnan369/sigma/actions/runs/44514770708) 与 [PR CI](https://github.com/xinnan369/sigma/actions/runs/44514779462) 均通过。
- 候选授权原始字节 SHA256 为 `eb002f9b3e4c82bdb6bbab6bb205f8ff5f88c0ddb13cc03b47d0796c9550effa`，协议 SHA256 为 `ff9e193482eb0cc808a428a5c1c662516d43bd7d1d0e603d0b3b8edc3d9028db`，D0 split SHA256 为 `bd1090cb2ad4696db3ba2f8e984788204c433a7e0cab656073c5e2b67e53e3d4`。
- R5 从合并树重新计算证据哈希，并核对 WSL 读取的签名源码原始字节和规范化 Git blob。独立证据读取器依据合同、各条原始记录和数值阈值重新判定，复算签名和批序。生产签发程序未调用，现行运行器签名用于交叉核对。[独立审查结果](evidence/r3-d1b-r5/authorization-review.json) · [证据清单](evidence/r3-d1b-r5/verification.json)

## 六项授权门禁

| 门禁 | R5 结论 | 独立核验及证据来源 |
|---|---|---|
| A / RGB-only | PASS | 重验原 RTX 3090 独立恢复证据。两步连续与跨进程恢复在模型、BN、AdamW、RNG 和日志前缀上逐位一致。本轮以合成 RGB 输入重建模型并执行前向。 |
| B / Thermal-only | PASS | 同一标准复核逐位恢复原始记录。本轮以合成 Thermal 输入重建模型并执行前向。 |
| C / RGB-T CNN | PASS | 同一标准复核逐位恢复原始记录。本轮双模态合成前向的初始化、参数和算子计数与原证据一致。 |
| D / RGB-T Mamba | PASS | 同一标准复核原始恢复记录，D 连续两步计数为 18 次原生前向、72 次原生反向子调用。重验 16 组原生与确定性反向样例及 batch 4/2 整模型证据，原生级最大混合容差比 0.005223，上限 1。三尺度 CrossMamba 加 ConcatMamba 使用本轮加载的固定原生扩展，合成前向追踪到 9 次原生扫描。CUDA 反向数值与逐位恢复均为原始证据复核。 |
| 数据完整性 | PASS | 本轮重新校验并解码 Fit 9,926 与 Validation 2,087 对，共 24,026 个 JPEG、12,013 份 XML。Quarantine 12，分别对应 5 个异常标注和 7 个 UNKNOWN 跨划分候选。五个代理组仅覆盖 Fit 和 Validation，各组完整落入一个开发分区。FCOS 逐图账本 SHA256 为 `6bef3d384796f98c8f2b8d5f406083b35d4d33202b34c9b2e158762319ba71cd`。Fit 匹配 28,540／28,547，Validation 匹配 5,530／5,536。 |
| 协议与签名 | PASS | 本轮独立复算 A/B/C/D × 3407–3411 的 20 组完整签名和 50 个批次顺序摘要，并与实际运行器和存档候选逐组核对。每轮 2,481 个四图批和一个两图尾批。 |

本轮 CPU 全套为 **207 通过、3 跳过、0 失败**，用时 115.55 秒，另有一条依赖弃用警告，`compileall` 退出码 0。R4 同样为 207 通过、3 跳过，原始日志和测试覆盖的冻结生产代码均重新核对。三项跳过属于既有条件测试，分别依赖外部 Sigma 初始化或专用 CUDA 条件。[本轮 CPU 日志](evidence/r3-d1b-r5/cpu.log)

本轮官方 Test 像素读取、选模和真实 LLVIP optimizer update 均为 0。历史 D0 读取并统计了 3,463 份官方 Test XML，早期 Thermal Pilot 接触过其中 64 对图像，原暴露记录保留。训练和选模只允许 Fit 与 Validation。当前使用视觉代理分组，尚无真实场景泛化结论。

## 入口和 Checkpoint 审查

审查了 `authorization.py`、`review_registry.py`、`train.py`、`plan.py`、`checkpoint.py` 和所有开发数据调用点。唯一真实 LLVIP 训练入口在访问数据、设备及输出前要求显式运行标志和可信回执。可信摘要命中后还检查六项门禁、冻结 split 和全 20 个 arm/seed 键。加载运行环境后，它在读训练样本或打开 checkpoint 前比较当前 arm/seed 的完整签名。恢复路径检查 ledger 和嵌入签名、checkpoint 哈希、批次游标、日志前缀及 RNG。每轮只用 Validation 的 AP50:95、AP50 和最早 Epoch 规则选模。其余开发数据命令用于读取或审计，数值与恢复工具使用合成张量。

在实际空信任登记下，对 20 个配置各测试新开与恢复两种模式，以及缺显式标志、缺回执、传入未登记候选三种情形，共 120 次。全部在 CUDA、数据、输出、checkpoint 或 optimizer 调用之前被拒。此外，实际 CLI 拒绝了 D/3407 的未授权恢复命令。合成门禁与损坏恢复拒绝测试随全套 CPU 测试通过。

## 正式训练预检

矩阵为 20 次独立训练，每次 FP32、10 Epoch、24,820 次更新，总计 200 Epoch、496,400 次 AdamW 更新。使用本机 LLVIP 开发集和合并树中的 D0 split。RTX 3090 设备和软件指纹与候选一致。Sigma commit 为 `98ba3c6938d29169bbd27f91519084bb462561df`，实际加载的 Selective Scan 0.0.1 二进制 SHA256 为 `833021baeecf26b84d5ecba208417893a9eda202c38854fd9a9737c8ccd7513e`。前向保持原生，D 的正式签名绑定 `sigma-r3-exclusive-block-pairwise-v1` 反向。

| 臂 | 参数量 | 语义 MAC／单图 | 五种子计算估计 |
|---|---:|---:|---:|
| A | 135,062 | 666,777,600 | 3.70 小时 |
| B | 135,062 | 666,777,600 | 3.73 小时 |
| C | 489,938 | 1,771,118,400 | 4.93 小时 |
| D | 490,022 | 1,771,173,888 | 7.08 小时 |
| 合计 | | | 19.44 小时 |

C/D 参数相对差 **0.01714%**，语义 MAC 差 **0.00313%**，分别满足 1% 和 5% 的预算。估计来自原 R3 合成 batch 4/2 的完整更新中位数，含前向、FCOS loss、反向及 AdamW。数据读取、Validation、COCO 和 checkpoint 的耗时尚无全程实测。D 历史 batch 4 峰值 reserved 约 1.55 GiB。当前可用显存和计划输出盘空间为审查时点快照。[资源与启动矩阵](evidence/r3-d1b-r5/authorization-review.json)

计划使用仓库外目录 `/mnt/d/work/sigma-r3-d1-formal/<arm>/<seed>`。20 个路径均未占用。运行器接受 `--root`、`--split`、`--upstream`、`--output`、`--seed`、`--arm`、`--authorization` 和 `--authorize-formal-training`，轮数和 batch 由协议常量控制。每组使用独立进程锁，每 100 次全局更新以及每 Epoch 结束时原子提交 checkpoint。首次启动只能使用空输出目录。恢复使用同一配置、显式 `--resume` 和已有完整 ledger。本轮计划输出目录和 checkpoint 均未创建。

## 可复现命令

在 R5 审计 checkout 中运行。WSL 使用 `/home/xinnan/.venvs/sigma-r1-cu121/bin/python`。`$R5_OUTPUT` 指向新的仓库外审计目录。将各命令日志写入该目录，末行记录 `Exit code: <code>`。CPU 与开发集复验期间隐藏 CUDA。

~~~sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
python -m pytest -q -rs tests
python -m compileall -q experiments/r3_d1
python -m experiments.r3_d1.authorization_preflight data \
  --root /mnt/d/BaiduNetdiskDownload/LLVIP/LLVIP --output "$R5_OUTPUT/development"
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m experiments.r3_d1_r3.final_authorization_review \
  --upstream /home/xinnan/sigma-r1-upstream \
  --data /mnt/d/BaiduNetdiskDownload/LLVIP/LLVIP \
  --output "$R5_OUTPUT" --checkpoint-root /mnt/d/work/sigma-r3-d1-formal
~~~

R5 的技术结论可以提交 R6 独立批准。受信任回执登记、候选字节、协议、模型、损失和历史证据保持原样，正式训练未启动。
