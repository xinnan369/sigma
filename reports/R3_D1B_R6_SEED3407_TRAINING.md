# R3-D1B-R6 — Seed 3407 正式训练与授权审批

**R6 首批训练已完成。** A、B、C、D 的 seed 3407 实验均完成 10 Epoch，共 99,280 次真实 AdamW 更新。训练报告与证据提交 PR #39，等待独立验收。

## 审批依据

[独立审查 PR #38](https://github.com/xinnan369/sigma/pull/38) 从验收 Head `e32c15d73c2d06ab384fb13f1f133f37a6d05835` 合入 `codex/R3-D1B-R3-independent-authorization-audit`。合并提交为 `758e4f42f7038a7cff446525ba92fe73bad3f242`，Tree SHA 为 `eafb72260a3488f070d0d7a7b834bf17b7a2ec1c`，与验收 Head 一致。合并差异只有 R5 审计程序、报告及证据。冻结模型、训练器、损失、评估器、训练协议和 D0 划分在合并前后保持一致。[Push CI](https://github.com/xinnan369/sigma/actions/runs/44517487337) · [PR CI](https://github.com/xinnan369/sigma/actions/runs/44517544216)。合并后的 [Push CI](https://github.com/xinnan369/sigma/actions/runs/44520214972) 和 [PR CI](https://github.com/xinnan369/sigma/actions/runs/44520223725) 均已通过。

本 PR 提请审批唯一拟登记摘要 `eb002f9b3e4c82bdb6bbab6bb205f8ff5f88c0ddb13cc03b47d0796c9550effa`。它对应原始 `reports/evidence/r3-d1b-r2r/authorization-candidate.json` 文件的字节 SHA256。审批记录应明确指向本 PR 和该摘要。审批后才能将该摘要写入 `experiments/r3_d1/review_registry.py`，并引用 PR #38 的审查依据。审批提交保留于 PR #39 历史。用户随后在 Codex 对话中明确批准登记这一摘要并启动四个 seed 3407 实验。生产登记只包含此摘要。[人工批准记录](evidence/r3-d1b-r6/human-approval.json)[候选与门禁复核](evidence/r3-d1b-r6/approval-candidate-review.json) · [审批前 SHA256 清单](evidence/r3-d1b-r6/verification.json)

本轮在固定 RTX 3090 WSL2 环境重算候选及 R5 证据文件摘要，审查 A/B/C/D、数据完整性和协议签名六项证据并复算 **20 组运行签名、50 个 Epoch 批次顺序摘要**。运行器的签名与独立复算、原始候选及 R5 摘要逐项一致。本机载入固定 Sigma `98ba3c6938d29169bbd27f91519084bb462561df`、Selective Scan 0.0.1，CUDA 扩展 SHA256 为 `833021baeecf26b84d5ecba208417893a9eda202c38854fd9a9737c8ccd7513e`，冻结的确定性后端为 `sigma-r3-exclusive-block-pairwise-v1`。GPU 数值和跨进程恢复采用已有 R5 原始证据，本轮只重复了哈希、记录和指纹核验。

## 批准后的执行闸门

1. 仅登记已批准摘要，复核所有 20 个运行签名，确认错误 seed 或 arm、与臂或种子不匹配的签名、错误 split 和未登记回执均被拒绝。
2. 验证缺少 ledger、checkpoint 损坏、签名或日志前缀不匹配时恢复会停止。保留现有日志和失败证据。
3. 上述检查通过后，仅运行 Seed 3407 的四个实验。每组独立开始，采用同一固定 10 Epoch、batch 4、FP32、AdamW 配置，以及审查通过的 Sigma CUDA 扩展和确定性 backward。
4. 使用仓库外独立输出目录，按冻结协议保存事件日志、optimizer、RNG、checkpoint 和运行签名。中断后仅从校验通过的 checkpoint 续跑。只用 Fit 更新参数及用 Validation 选模。

## 首批实验结果

批次于北京时间 2026 年 11 月 19 日 22:57:00 启动，11 月 20 日 09:38:08 完成，墙钟用时约 **10 小时 41 分**。四臂均在 RTX 3090 上独立完成 24,820 次更新，无恢复记录。各臂耗时包含数据读取、checkpoint 和 Validation。峰值显存是训练事件中 PyTorch 记录的最大已分配显存。

| 实验 | 最佳 Epoch | 最佳 AP50 | 最佳 AP50:95 | Epoch 10 AP50 | Epoch 10 AP50:95 | 墙钟耗时 | 显存峰值 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| A/3407 | 9 | 31.85% | 10.27% | 16.23% | 5.14% | 2:32:54 | 203.5 |
| B/3407 | 9 | 67.74% | 27.00% | 64.77% | 25.83% | 2:25:37 | 203.5 |
| C/3407 | 8 | 86.24% | 42.39% | 85.11% | 41.08% | 2:35:23 | 555.5 |
| D/3407 | 9 | 54.22% | 24.34% | 45.03% | 19.14% | 3:07:11 | 1226.5 |

最佳 checkpoint 按冻结规则选择，以 AP50:95 为首要排序指标，其次 AP50，再取较早 Epoch。轮次以 1 起算。四臂每轮均完成 2,482 次更新，最后一批为 2 张。独立复核逐事件重算 40 个训练损失与 40 个 Fit 批次顺序摘要，逐轮将机器记录的 AP 与 COCO 输出比对。每臂读取 Fit 99,260 对次和 Validation 20,870 对次，合计 397,040 和 83,480 对次。官方 Test 像素读取为 0，依据为冻结读取器的角色检查、签名批序和完整的 Validation 记录。

四臂的 best/last 文件及内容寻址 checkpoint 均与终态 ledger 的 SHA256 一致。终态事件日志长度和完整哈希与 ledger 一致，逐事件 loss 及梯度指标均为有限值。原始权重、训练事件日志和独立输出目录保存在本机。归档记录各轮损失、AP50、AP50:95、预测摘要、验证用时、完整 checkpoint 哈希与运行签名。[机器可读完整结果](evidence/r3-d1b-r6/completed/final-training-audit.json) · [证据 SHA256](evidence/r3-d1b-r6/completed/final-verification.json) · [A 日志](evidence/r3-d1b-r6/completed/A-3407-validation.log) · [B 日志](evidence/r3-d1b-r6/completed/B-3407-validation.log) · [C 日志](evidence/r3-d1b-r6/completed/C-3407-validation.log) · [D 日志](evidence/r3-d1b-r6/completed/D-3407-validation.log)

seed 3407 上，D 的最佳 AP50:95 比 C 低 **18.04 个百分点**，两者在第 10 轮的差距为 **21.93 个百分点**。比较限于这次训练和冻结内部 Validation，其余随机种子尚未运行。

## 执行证据

正式训练提交为 `460bb30d296470fb5d91e658a5a363abb2f88588`。WSL2 使用固定虚拟环境和单张 RTX 3090，FP32，batch 4，AdamW，seed 3407。终态环境指纹与审查记录一致，四个训练进程均退出 0。以下以 A 臂为例。

```sh
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/home/xinnan/.venvs/sigma-r1-cu121/bin/python -u -m experiments.r3_d1.train \
  --root /mnt/d/BaiduNetdiskDownload/LLVIP/LLVIP \
  --split reports/evidence/r3-d0/split.json --upstream /home/xinnan/sigma-r1-upstream \
  --output /mnt/d/work/sigma-r3-d1-formal --seed 3407 --arm A \
  --authorize-formal-training --authorization reports/evidence/r3-d1b-r2r/authorization-candidate.json
```

登记后重新验证 20 组签名，并实测 10 个拒绝情形。全套测试覆盖损坏 checkpoint、恢复签名与事件前缀校验。原授权单测改为检查合成回执确实未登记，并继续验证所有臂拒绝该回执。模型、训练器、损失、评估和数据切分 SHA 保持冻结。

RTX 3090 固定 WSL 环境的全套 CPU 测试为 207 通过、3 跳过、0 失败。首批输出目录为仓库外的 `D:/work/sigma-r3-d1-formal`，四臂按 A、B、C、D 顺序执行。已审计的 launcher 对每臂的正式运行签名、最后 ledger、事件批序和 checkpoint 哈希进行核验。 [授权后检查清单](evidence/r3-d1b-r6/approved-verification.json)

启动阶段的 WSL Git 索引扫描在训练工作进程创建前中止，随后改为读取被冻结的源码 SHA256 并从第一步启动 A/3407。启动快照记录已提交事件前缀、检查点 SHA256 和当时的更新次序。 [启动证据](evidence/r3-d1b-r6/startup-observation.json)
