# R3-D1B-R6 — Seed 3407 正式训练与授权审批

**状态为 BLOCKED，等待 R6 人工明确批准。** 候选的技术复核通过。生产信任登记保持为空，A/B/C/D 的真实训练均为 NOT RUN。

## 审批依据

[独立审查 PR #38](https://github.com/xinnan369/sigma/pull/38) 从验收 Head `e32c15d73c2d06ab384fb13f1f133f37a6d05835` 合入 `codex/R3-D1B-R3-independent-authorization-audit`。合并提交为 `758e4f42f7038a7cff446525ba92fe73bad3f242`，Tree SHA 为 `eafb72260a3488f070d0d7a7b834bf17b7a2ec1c`，与验收 Head 一致。合并差异只有 R5 审计程序、报告及证据。冻结模型、训练器、损失、评估器、训练协议和 D0 划分在合并前后保持一致。[Push CI](https://github.com/xinnan369/sigma/actions/runs/44517487337) · [PR CI](https://github.com/xinnan369/sigma/actions/runs/44517544216)。合并后的 [Push CI](https://github.com/xinnan369/sigma/actions/runs/44520214972) 和 [PR CI](https://github.com/xinnan369/sigma/actions/runs/44520223725) 均已通过。

本 PR 提请审批唯一拟登记摘要 `eb002f9b3e4c82bdb6bbab6bb205f8ff5f88c0ddb13cc03b47d0796c9550effa`。它对应原始 `reports/evidence/r3-d1b-r2r/authorization-candidate.json` 文件的字节 SHA256。审批记录应明确指向本 PR 和该摘要。审批后才能将该摘要写入 `experiments/r3_d1/review_registry.py`，并引用 PR #38 的审查依据。当前审批提案只包含报告和证据。[候选与门禁复核](evidence/r3-d1b-r6/approval-candidate-review.json) · [SHA256 清单](evidence/r3-d1b-r6/verification.json)

本轮在固定 RTX 3090 WSL2 环境重算候选及 R5 证据文件摘要，审查 A/B/C/D、数据完整性和协议签名六项证据并复算 **20 组运行签名、50 个 Epoch 批次顺序摘要**。运行器的签名与独立复算、原始候选及 R5 摘要逐项一致。本机载入固定 Sigma `98ba3c6938d29169bbd27f91519084bb462561df`、Selective Scan 0.0.1，CUDA 扩展 SHA256 为 `833021baeecf26b84d5ecba208417893a9eda202c38854fd9a9737c8ccd7513e`，冻结的确定性后端为 `sigma-r3-exclusive-block-pairwise-v1`。GPU 数值和跨进程恢复采用已有 R5 原始证据，本轮只重复了哈希、记录和指纹核验。

## 批准后的执行闸门

1. 仅登记已批准摘要，复核所有 20 个运行签名，确认错误 seed 或 arm、与臂或种子不匹配的签名、错误 split 和未登记回执均被拒绝。
2. 验证缺少 ledger、checkpoint 损坏、签名或日志前缀不匹配时恢复会停止。保留现有日志和失败证据。
3. 上述检查通过后，仅运行 Seed 3407 的四个实验。每组独立开始，采用同一固定 10 Epoch、batch 4、FP32、AdamW 配置，以及审查通过的 Sigma CUDA 扩展和确定性 backward。
4. 使用仓库外独立输出目录，按冻结协议保存事件日志、optimizer、RNG、checkpoint 和运行签名。中断后仅从校验通过的 checkpoint 续跑。只用 Fit 更新参数及用 Validation 选模。

## 首批实验记录

| 实验 | 计划 Epoch | 计划更新 | 实际更新 | Validation AP50 / AP50:95 | 最优 checkpoint、耗时、显存 |
|---|---:|---:|---:|---|---|
| A/3407 | 10 | 24,820 | 0 | NOT RUN | NOT RUN |
| B/3407 | 10 | 24,820 | 0 | NOT RUN | NOT RUN |
| C/3407 | 10 | 24,820 | 0 | NOT RUN | NOT RUN |
| D/3407 | 10 | 24,820 | 0 | NOT RUN | NOT RUN |

人工批准尚未记录，授权后负面测试和真实训练均为 NOT RUN。当前训练数据读取、官方 Test 像素读取和真实 LLVIP AdamW 更新均为 0。获批后按轮报告 loss、Validation AP50/AP50:95、实际读取和更新数、checkpoint 哈希、时长、显存及异常，不根据首批结果调整后续四个种子的设计。
