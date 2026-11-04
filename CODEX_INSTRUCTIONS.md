# 发给 Codex 的任务指令（粘贴全文即可）

你担任科研项目的工程执行者。唯一交接仓库：https://github.com/xinnan369/sigma。它不同于上游 Sigma 源仓库 https://github.com/zifuwan/Sigma。

**任务 R0：初始化仓库并落地 Experiment 0 可复现基础。**

本任务的交接包包含 `README.md`、`AGENTS.md`、`tasks/R0_bootstrap.md`、`docs/`、`experiments/exp0/`、`tests/` 和 `reports/R0_REPORT.md`。请先查看实际远端仓库状态。若确实为空，先只在 main 创建 README 初始化提交，然后从 main 创建 `codex/R0-bootstrap` 分支，将交接包除重复 README 外内容提交到该分支。若远端已有人提交，保留其内容并基于最新 main 工作。

严格遵循 `AGENTS.md`、`tasks/R0_bootstrap.md`、`docs/EXPERIMENT_0_PROTOCOL.md`。核查 `zifuwan/Sigma` 的 upstream commit `98ba3c6938d29169bbd27f91519084bb462561df` 和 MIT 许可，检查 Sigma-Tiny / MFNet 的模型与评估配置，但不要把上游模型代码未经必要许可说明直接拷进中转仓库。

运行基础测试与环境检查，记录实际测试命令、退出码和输出。如测试失败，先修复并增加回归测试。所有修改后提交 PR；不得在没有真实 GPU 数据/权重时伪造 mIoU 或宣称状态污染已被验证。不要上传数据、权重、凭据或大文件。

创建 PR：`[R0] Bootstrap research bridge and Experiment 0 protocol`，目标 main。PR 说明包括文件清单、上游 commit、测试实际结果、相关日志和 `reports/R0_REPORT.md`。**不要自动合并，等待 ChatGPT 发出 PASS 验收结论**。此后的合并由你执行，需核对 PR head SHA 与验收时一致，无需另问用户。

结束后请输出 PR URL、最新 head SHA、测试命令及退出码、尚未完成项目。下一阶段 R1 才开始真实 Sigma/MFNet 评估。
