# R3-D1B-R7 五随机种子正式实验

**3408、3409 阶段已完成。** 8 组均通过审计，共完成 198,560 次真实更新。3410、3411 暂缓，完整五种子统计待后续阶段。

## 授权与冻结基线

PR #39 的验收 Head 为 `c4ec764a12bb5b9fb1971934c47ec3c7ce6bcae7`，Push 和 PR CI 均通过。合并至 `codex/R3-D1B-R3-independent-authorization-audit`，合并 SHA 为 `e005dd1e9e4aaaa73eaaed14f2fef2edad465aaf`，Tree SHA 为 `152e32f958fdcdbe81a9655460426523dc565a8b`，和验收 Head 的树一致。[PR #39](https://github.com/xinnan369/sigma/pull/39) · [Push CI](https://github.com/xinnan369/sigma/actions/runs/44672375441) · [PR CI](https://github.com/xinnan369/sigma/actions/runs/44672380724)

用户在 Codex 对话中明确授权 A、B、C、D 的 seeds 3408–3411。以 R5 独立审查、原候选回执和现行源码重算 20 个运行签名，已有 seed 3407 的四项授权逐值匹配。候选文件、授权模块、模型、优化器、损失、训练和评估源码及冻结协议保持原样。候选 SHA256 为 `eb002f9b3e4c82bdb6bbab6bb205f8ff5f88c0ddb13cc03b47d0796c9550effa`，现行 `authorization.py` 的 SHA256 为 `d71821c150c64453358654eb3eacc79b0423a89437d71dd9bb09f959618c607c`。[授权记录与 20 组精确指纹](evidence/r3-d1b-r7/scope-extension-approval.json) · [R5 审查](https://github.com/xinnan369/sigma/pull/38) · [R6 历史结果](R3_D1B_R6_SEED3407_TRAINING.md)

## 执行与证据

运行顺序为 3408 的 A–D、3409 的 A–D、3410 的 A–D、3411 的 A–D。每臂 10 Epoch、24,820 次更新，RTX 3090、FP32、batch 4、冻结 AdamW 与确定性 Sigma selective scan。仅访问冻结 Fit 和 Validation。seed 3407 从不可变历史归档加入最终比较，R7 调度器不会训练该种子。

运行程序为 `experiments/r3_d1_r7/run_remaining.py`，发布程序为 `experiments/r3_d1_r7/publish_after_completion.ps1`。独立、固定的工作树为 `D:/work/sigma-r7-runtime`。权重、完整事件、进程日志、循环状态及每个种子的审计保存在 `D:/work/sigma-r3-d1-r7-formal`。正式训练和恢复调用原训练器，任何恢复尝试均须核对运行签名、checkpoint SHA256、终态或已提交事件前缀以及运行环境。进程退出异常时保留原日志，由签名和哈希复验后才能恢复。

每个种子的四臂通过独立审计后，才启动下一种子。审计包括每轮更新、Fit ID 覆盖和精确批序、翻转、有限梯度、加权损失、Validation 指标、最佳及末轮 checkpoint 和运行环境。完整五种子统计同时报告按 AP50:95 选出的最佳轮与第 10 轮，使用样本标准差并对同种子的 D 与 C 配对。原始权重、数据与完整事件日志留在本机。全部结果生成且证据哈希核对后，脚本发布 R7 PR，供独立验收。

启动前在固定 WSL 环境执行 `python -m pytest -q -rs tests` 与 `python -m compileall -q experiments/r3_d1 experiments/r3_d1_r7`。授权测试覆盖 20 个获批运行、错误臂与种子、错误签名与 split、未登记回执以及未授权恢复。CPU 和编译结果随证据归档。[测试与启动闸门](evidence/r3-d1b-r7/preflight-verification.json)

固定 WSL 的 CPU 套件实测 257 通过、3 跳过，编译检查退出 0。

## 本轮调度缩为两个新增种子

按用户指令，本阶段只调度 3408、3409，各执行 A、B、C、D，计划 198,560 次更新。3407 沿用既有结果，3410 与 3411 暂缓调度。原授权登记和实验协议保留。两种子的单独审计汇总写入本机 stage-through-3409-summary.json。阶段结束时暂停，完整五种子报告与 PR 仍等待剩余两个种子运行及独立核验。

## 3408、3409 阶段结果

本阶段于北京时间 2026 年 11 月 20 日 13:24:57 开始，11 月 21 日 13:14:20 完成，端到端墙钟约 23 小时 49 分。每组完成 10 Epoch、24,820 次更新。最高轮按 AP50:95、AP50、最早 Epoch 的顺序选取，所有分数均来自冻结内部 Validation。耗时含读取、Validation 和 checkpoint 开销，显存是事件中记录的峰值已分配显存。

| Seed | 实验臂 | 最佳轮 | 最佳 AP50 | 最佳 AP50:95 | 第 10 轮 AP50 | 第 10 轮 AP50:95 | 墙钟小时 | 峰值 MiB |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 3408 | A | 8 | 29.24% | 8.68% | 26.08% | 7.87% | 2.29 | 203.5 |
| 3408 | B | 10 | 74.26% | 28.32% | 74.26% | 28.32% | 2.49 | 203.5 |
| 3408 | C | 6 | 85.74% | 41.63% | 84.59% | 40.95% | 3.73 | 555.5 |
| 3408 | D | 9 | 67.85% | 31.37% | 52.85% | 25.51% | 3.13 | 1226.5 |
| 3409 | A | 9 | 32.38% | 10.68% | 29.86% | 9.88% | 2.63 | 203.5 |
| 3409 | B | 8 | 74.55% | 27.45% | 74.58% | 26.51% | 2.59 | 203.5 |
| 3409 | C | 10 | 87.77% | 40.41% | 87.77% | 40.41% | 2.95 | 555.5 |
| 3409 | D | 10 | 79.24% | 28.88% | 79.24% | 28.88% | 3.57 | 1226.5 |

每组累计读取 Fit 99,260 对次、Validation 20,870 对次。官方 Test 像素读取为 0。A/3408 因缩小调度范围而受控停止，从通过 SHA256 与运行签名核验的第 2,300 次更新恢复。其余七组一次完成，未发生恢复。两种子的原始审计分别记录 40 个逐轮 loss、AP50/AP50:95、验证用时、批序、最佳与末轮 checkpoint、完整事件日志及进程日志的 SHA256。

本轮上传前逐文件复核了 85 件本机运行工件和 9 份进程日志，终态账本、checkpoint、签名和原始两份种子审计均一致。权重、完整事件和数据集保留在本机。本节仅记录两种子结果，五种子比较留待完整实验。 [3408 审计](evidence/r3-d1b-r7/two-seed-stage/results/seed-3408-audit.json) · [3409 审计](evidence/r3-d1b-r7/two-seed-stage/results/seed-3409-audit.json) · [SHA256 与环境指纹](evidence/r3-d1b-r7/two-seed-stage/results/verification.json)
