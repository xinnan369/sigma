# R3-D1B-R6 — Seed 3407 正式训练与授权审批

**R6 已获人工批准。** 获批回执已登记，首批执行状态见下方。

## 审批依据

[独立审查 PR #38](https://github.com/xinnan369/sigma/pull/38) 从验收 Head `e32c15d73c2d06ab384fb13f1f133f37a6d05835` 合入 `codex/R3-D1B-R3-independent-authorization-audit`。合并提交为 `758e4f42f7038a7cff446525ba92fe73bad3f242`，Tree SHA 为 `eafb72260a3488f070d0d7a7b834bf17b7a2ec1c`，与验收 Head 一致。合并差异只有 R5 审计程序、报告及证据。冻结模型、训练器、损失、评估器、训练协议和 D0 划分在合并前后保持一致。[Push CI](https://github.com/xinnan369/sigma/actions/runs/44517487337) · [PR CI](https://github.com/xinnan369/sigma/actions/runs/44517544216)。合并后的 [Push CI](https://github.com/xinnan369/sigma/actions/runs/44520214972) 和 [PR CI](https://github.com/xinnan369/sigma/actions/runs/44520223725) 均已通过。

本 PR 提请审批唯一拟登记摘要 `eb002f9b3e4c82bdb6bbab6bb205f8ff5f88c0ddb13cc03b47d0796c9550effa`。它对应原始 `reports/evidence/r3-d1b-r2r/authorization-candidate.json` 文件的字节 SHA256。审批记录应明确指向本 PR 和该摘要。审批后才能将该摘要写入 `experiments/r3_d1/review_registry.py`，并引用 PR #38 的审查依据。审批提交保留于 PR #39 历史。用户随后在 Codex 对话中明确批准登记这一摘要并启动四个 seed 3407 实验。生产登记只包含此摘要。[人工批准记录](evidence/r3-d1b-r6/human-approval.json)[候选与门禁复核](evidence/r3-d1b-r6/approval-candidate-review.json) · [审批前 SHA256 清单](evidence/r3-d1b-r6/verification.json)

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

登记后重新验证 20 组签名，并实测 10 个拒绝情形。全套测试覆盖损坏 checkpoint、恢复签名与事件前缀校验。原授权单测改为检查合成回执确实未登记，并继续验证所有臂拒绝该回执。模型、训练器、损失、评估和数据切分 SHA 保持冻结。

RTX 3090 固定 WSL 环境的全套 CPU 测试为 207 通过、3 跳过、0 失败。首批输出目录为仓库外的 `D:/work/sigma-r3-d1-formal`，四臂按 A、B、C、D 顺序执行。已审计的 launcher 对每臂的正式运行签名、最后 ledger、事件批序和 checkpoint 哈希进行核验。 [授权后检查清单](evidence/r3-d1b-r6/approved-verification.json)
