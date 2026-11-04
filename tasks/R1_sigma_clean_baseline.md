# R1 — Sigma-Tiny + MFNet 真实基线复现

日期：2026-11-04。用户已明确批准本任务；R1 不自行合并，提交 PR 后等待 ChatGPT 验收。

执行状态：GPU 扩展构建与 192+12 案例数值兼容性检查通过；取得官方 393 样本 test.txt 的 SHA256；完整图像/标注与官方分割 checkpoint 未齐备，clean baseline 阻塞。详见 reports/R1_REPORT.md。

前置 R0 已 PASS，验收 head `3d300624cc5c9022ba83b9b8e059881dc44793d8`；squash merge 到 main 的提交为 `68b7dacce44a995bc821fdbcfd51b64bdef3daec`。合并前确认两组 CI success、MERGEABLE/CLEAN、head 完全一致；采用带 expected-head 条件的 squash merge，未绕过验收。

## 范围与验收

1. 分支 `codex/R1-sigma-clean-baseline` 从上述 main 创建。
2. 固定上游 `98ba3c6938d29169bbd27f91519084bb462561df`；不改模型、测试划分、预处理或类别统计。
3. 建立 GPU 运行条件，真实检查 CUDA torch、timm、einops 和该上游 selective-scan 扩展；实际执行兼容性测试，记录环境差异和未运行项。
4. 记录 MFNet 官方 test.txt、所有所需图像/标注和 Sigma-Tiny/MFNet 分割 checkpoint 的来源及 SHA256；不存在的文件写 NOT AVAILABLE，不以 URL 文本 hash 或 VMamba 初始化权重代替。
5. 齐备后运行原始 eval.py，记录 9 类 IoU、含类别 0 的总体 mIoU、命令、设备、显存、耗时和完整日志。缺失条件时如实报告，不在随机权重或其他模型权重上生成所谓 baseline。
6. 提交 reports/R1_REPORT.md、工具和实际证据到 R1 PR；无数据、权重、密钥或编译二进制上传。

本任务仅 clean baseline；Experiment 0 扰动、模块干预、隐藏状态和非 SSM 对照保留后续阶段，不以本轮兼容性 smoke 代替科学证据。
