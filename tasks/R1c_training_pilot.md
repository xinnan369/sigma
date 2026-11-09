# R1c — Sigma-Tiny 独立训练基线准备与小规模验证

用户批准日期：2026-11-08（Asia/Shanghai）。这是独立重新训练，不是作者分割 checkpoint 复现。

前置 R1 #2 head `bb45419670f7385c6aca590878b741b8ca9c1879` 与 R1b #3 head `aa335e92f10515f5e3698e7b8e63355eda064b14` 均经本轮用户确认验收；核对 head 未变、CI 全成功且无冲突，依次保留历史合并。合并提交分别为 `9fe51d16ebbb933df7bbedc40b53df86cd5d1901`、`83a8d6454a7bd6508e901a0aa13ac133923cb665`。R1c 从完整 main 后者创建。

- 下载、严格审计作者 train.txt 与 train2.txt 全部 triplet，记录列表与逐文件 SHA256；检查规范化原图身份与 393 test 无交集。
- 使用 train.txt regime；按原图身份、day/night 分层 SHA256(seed:id) 排序，各保留 ceil(10%) 作为内部验证。原图及 flip 不跨训练/验证。冻结清单后再训练，不读取测试像素。
- 导入固定上游模型与 TrainPre，不复制修改模型；修复驱动的完整参数覆盖、单卡 cuda:0、梯度累积、内部验证选模、原子 checkpoint 与 RNG/sampler 恢复。披露所有协议变化。
- 官方 VMamba-T 初始化，FP32 microbatch=1、累积8、固定 seed3407、AdamW lr6e-5。最多13次优化器更新（104微步）；内部验证每4步及末步。先停止在第8步，再独立进程恢复至第13步，测试断点恢复。
- 不做 batch=8、不启动500 epoch，不把 pilot 验证指标称 clean test mIoU；不上传资产或 checkpoint。
- 提交代码、完整覆盖审计、真实日志、资源与可行性报告；R1c PR 等待验收，不合并。
