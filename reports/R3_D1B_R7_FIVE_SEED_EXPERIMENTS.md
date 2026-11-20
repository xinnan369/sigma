# R3-D1B-R7 五随机种子正式实验

**状态为训练准备。** R7 只调度新增的 16 个运行。完整统计须在四个新增种子全部通过审计后生成。

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
