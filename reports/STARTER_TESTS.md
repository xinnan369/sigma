# R0 交接包作者自检（不是 Codex 执行报告）

执行日期：2026-11-04
执行位置：ChatGPT 隔离工作区，不是目标 GitHub 仓库，也不是 Sigma GPU 实验环境。

## 执行结果

```text
$ python -m pytest -q tests
.............                                                            [100%]
13 passed in 1.11s
Exit code: 0

$ python experiments/exp0/env_check.py
python: 3.13.5
system: Linux-6.18.57-x86_64-with-glibc2.41
numpy: installed
pytest: installed
torch: installed
cv2: installed
timm: missing
einops: installed
torch_version: 2.10.0+cpu
cuda_available: False
git: git version 2.47.3 exit: 0
Exit code: 0

$ python -m compileall -q experiments tests
(no output)
Exit code: 0
```

## 可得与不可得的结论

- **可以确认：** 本交接包的 13 个基础单元测试覆盖扰动确定性、局部支撑掩码、指标、忽略标签、输出捕获及替换等纯代码行为。
- **尚未确认：** 真实 Sigma 架构是否能成功加载权重、MFNet 验证集的 mIoU、实际 CUDA Selective Scan 数值/速度、真实跨模态传播、期刊创新性。
- **下一步：** Codex 在可提交 PR 的环境中复跑并提供 commit、Python/CUDA 版本与测试证据；R1 才需要 GPU + MFNet + Sigma-Tiny 权重。
