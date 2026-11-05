# R1b — Sigma-Tiny + MFNet 官方基线恢复与真实推理验证

用户批准日期 2026-11-04；续行与交付日期 2026-11-05（Asia/Shanghai）。

固定上游 `98ba3c6938d29169bbd27f91519084bb462561df`；沿用 RTX 3090 / WSL2 CUDA 环境。

本任务分支 `codex/R1b-baseline-recovery` 基于 R1 的 `bb45419670f7385c6aca590878b741b8ca9c1879`。R1 PR #2 尚未验收/合并，故 R1b PR 以 R1 分支为 base，单独审查新增工作；不自行合并任何 PR。

## 范围

- 实例化完整 Sigma-Tiny，合成 RGB-T 前向，记录 shape、参数量、模块调用、显存和耗时；明确无分割 checkpoint。
- 获取作者测试清单对应的 393 RGB/Modal/Label triplets，严格审计格式、类别映射、SHA256；不上传原始资产。
- 检查作者正式 checkpoint 发布渠道与联系信息，不擅自发送消息、不代用其他模型权重。
- 仅在官方分割 checkpoint 齐备后原样执行 eval.py。缺权重时九类 IoU/mIoU 为 NOT RUN。
- 评估官方 VMamba 初始化下独立重训练的配置与资源；仅允许明确标注的合成资源探针，不启动完整训练。
- 提交报告、必要代码、完整实际日志和数据 manifest；等待 ChatGPT 验收。

执行结果：完整合成前向 PASS；1179 文件最终审计通过；VMamba 初始化文件获取并核验；官方 MFNet 分割 checkpoint 仍缺失，clean mIoU NOT RUN。batch=8 合成训练探针 OOM，默认后续资源探针限制 microbatch=1；详情见 reports/R1b_REPORT.md。
