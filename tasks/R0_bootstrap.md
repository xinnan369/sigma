# R0 — 科研仓库初始化与 Experiment 0 基础验证

**状态：本地实现与检查完成；远端 403 阻塞 main 初始化、分支推送及 PR；待 ChatGPT 验收** · 优先级 P0 · 日期 2026-11-04

## 本轮任务
1. 在空仓库 `xinnan369/sigma` 的 `main` 上只创建最小 README 初始化提交，再创建分支 `codex/R0-bootstrap`。保留本交接包的目录和文件。
2. 复核 `docs/UPSTREAM.md` 的固定上游 commit 和 MIT 许可证，记录 checkout 命令与上游本地 HEAD。
3. 实际执行 `python -m pytest -q tests` 与 `python experiments/exp0/env_check.py`，将**完整命令、环境、退出码、输出摘要**写入 `reports/R0_REPORT.md`；若失败先修复，不要虚报。
4. 对 `experiments/exp0/perturbations.py` 和 `metrics.py` 做代码审查：固定 seed、RGB 不变、支撑区域、ignore 255、错误指标归一化是否正确；发现问题要加回归测试。
5. 核查 `docs/EXPERIMENT_0_PROTOCOL.md` 与真实 Sigma 代码匹配，尤其 CrossMamba/ConcatMamba 观测点和官方 mIoU 配置。
6. 不下载或提交数据集/权重；不要求本轮真实 GPU 模型运行。写出 R1 所需设备、数据与依赖清单。
7. 在 `reports/R0_REPORT.md` 填写最终报告，并以 `[R0] Bootstrap research bridge and Experiment 0 protocol` 创建 PR 指向 main，等待 ChatGPT 审查。

## 验收条件
- 明确任务和报告；无机密、大文件和上游许可违规。
- 单测实际执行通过，复现命令明确，不依赖私有数据或个人目录。
- 修改的每个统计指标都可解释且有负对照，禁止把模型输出当作 SSM 隐状态。
- PR 提供 head commit SHA、完整测试证据、未完成项。
- **只有 ChatGPT 审查结论 PASS 才可由 Codex 合并**；审核后提交再变化需要重新审查。
