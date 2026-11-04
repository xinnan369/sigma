# R0 执行报告

执行日期：2026-11-04（Asia/Shanghai）。**本地实现、上游静态核查与测试完成；远端 main 初始化、分支推送和 PR 被实际 GitHub 写权限阻塞，R0 尚未完整交付。未合并，未作 PASS 验收结论。**

## 仓库与交接来源

| 字段 | 真实结果 |
|---|---|
| 目标仓库 | <https://github.com/xinnan369/sigma> |
| 远端初始状态 | API size 0；`git ls-remote` 无 refs；main 未初始化 |
| 本地 main 初始化 | `af463ab`，仅最小 README；完整 SHA 用 `git rev-parse main` 获取 |
| 本地 feature 分支 | `codex/R0-bootstrap`，基于上述 main |
| 已测试实现提交 | `fb8e315ab288302e0f34c5d874ece8b491d882a5`；后续仅补报告/发布证据 |
| PR URL / PR head SHA | **NOT CREATED / NOT AVAILABLE**，不能将本地 commit 冒充远端 PR head |
| 本地最终 head | 报告提交后由 `git rev-parse HEAD` 获取，并在最终交付和预备 PR 描述中提供；报告不嵌入自身 commit SHA |
| 固定上游 HEAD | `98ba3c6938d29169bbd27f91519084bb462561df`，实际 checkout 且最终工作树干净 |
| 交接包 | `sigma_R0_kickoff_20261104.zip` |
| ZIP SHA256 | `088dae9b9a9ecc81010c22305e326122bfb43d0ca8e48056f147d88cceef8f13` |

解压后优先阅读 AGENTS.md、CODEX_INSTRUCTIONS.md、tasks/R0_bootstrap.md、docs/EXPERIMENT_0_PROTOCOL.md，再阅读其余源码、测试和文档。交接包所有文件均已保留并按目录纳入；feature README 已扩充为实际复现说明，作者自检保留在 STARTER_TESTS.md，未冒充本机结果。

## 已完成的审查与修复

1. 扰动函数沿用局部 ROI、局部 seed（默认 3407）、非原地操作和 bool support。修复三通道 Gaussian 的独立噪声问题：共享空间噪声保持 MFNet 灰度 Thermal 的三通道复制关系。新增物理像素范围、有限数值、位移整数和 fill 检查。旧 shift 测试的 -1 sentinel 改为范围内的 7。
2. 指标审核确认 ignore=255、class 0 有效、所有区域外 rate 分母为全部有效区域外像素；new/corrected 分开，net=(new−corrected)/N，wrong-to-wrong 只计 disagreement，空区域 NaN。新增整数标签和布尔 mask 检查，防止 NaN mask 被静默转 True。没有改动原指标归一化公式。
3. 模块替换原来可能返回 replacement 自身，导致调用者或下游原地操作污染 clean 参考。现在注册时 snapshot、每次替换独立 clone，保持 tuple/list 双分支结构、shape guard 和 dtype/device 对齐；新增 context manager，异常退出清理 Hook。只支持 Tensor 及嵌套 tuple/list，其他结构显式拒绝。
4. 新回归测试首次运行真实复现 9 个失败、1 个通过，修复后再新增 5 个输入/位移/种子负对照，最终原始 13 + 新增 15 = **28 passed**。
5. 复核上游 MIT 许可并保存完整许可副本；逐文件 hash 对照 Git blob，无上游核心模型代码拷贝。核实共享编码器、Tiny 尺度、BHWC Hook 边界、CrossMamba 与 ConcatMamba 不同扫描路径、MFNet 灰度复制、官方 resize/滑窗和含类别 0 的 mIoU。详见 [UPSTREAM](../docs/UPSTREAM.md) 与 [源码证据](evidence/upstream_audit.txt)。
6. 更新实验协议，明确官方 clean gate 与扰动插入位置、区域外指标分母、模块输出替换与 hidden-state 干预的区别；预定义的特征响应公式及其实现仍属于 R1。
7. 添加 R0 版本清单、检查日志驱动和 CPU CI 工作流。CI 尚未远端执行，不能宣称通过；全新隔离环境安装也未验证，本机 venv 复用 host site-packages。

## 实际测试与环境

所有命令在目标本地 clone 根目录执行，使用 `.venv` 的 Python；表中以 `python` 表示已激活后的等价命令。最终完整 stdout、时间戳及退出码已存入 evidence。首次基础测试的宿主 anyio 插件有弃用警告，后续设置 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 排除无关自动插件，没有屏蔽测试失败。

| 命令 | 实际输出摘要 | 退出码 |
|---|---|---|
| `python -m pytest -q tests`（原交接包） | 13 passed, 1 warning in 6.40s | 0 |
| `python -m pytest -q tests/test_regressions.py`（修复前） | 9 failed, 1 passed in 3.19s | 1 |
| `python -m pytest -q tests`（最终） | 28 passed in 3.14s | 0 |
| `python experiments/exp0/env_check.py` | inventory 成功；CUDA torch 不可用 | 0 |
| `python -m compileall -q experiments tests tools` | 无输出 | 0 |
| `python tools/run_r0_checks.py` | 以上最终三项逐项记录，均 0 | 0 |
| `git diff --cached --check` | 无空白问题 | 0 |

证据：[pytest](evidence/pytest.txt)、[环境](evidence/environment.txt)、[compileall](evidence/compileall.txt)、[修复前摘要](evidence/review_before_fix.txt)、[执行命令台账](evidence/bootstrap_commands.txt)。修复前记录明确是实际输出摘要，未伪称完整 traceback。

| 环境项 | 实际值 |
|---|---|
| OS | Windows 11 / 10.0.26200 |
| Python | 3.13.2 |
| NumPy / pytest | 2.3.5 / 9.2.0 |
| PyTorch | 2.11.0+cpu |
| torch CUDA build / cuDNN | None / None |
| `torch.cuda.is_available()` | False |
| 硬件 | NVIDIA GeForce RTX 3090，24576 MiB |
| NVIDIA driver | 610.88 |
| nvcc / toolkit | 12.6，V12.6.85 |
| 关键缺失依赖 | timm、einops、selective_scan_cuda_core |

GPU 硬件和 toolkit 存在不等于当前 torch 支持 CUDA。环境检查的退出码 0 仅代表盘点完成。R0 单测在 CPU 上运行，不涉及真实模型特征、数据切分或语义分割分数；扰动 seed 用 3407，种子负对照用 3408。合成测试无训练/验证划分，也没有统计假设检验。

## 远端发布阻塞与恢复步骤

GitHub connector 和本机 `gh api` 均返回 `permissions.push=false`。connector 创建 main/README 返回 `403 Resource not accessible by integration`；本机 `git push -u origin main` 返回：

```text
remote: Permission to xinnan369/sigma.git denied to floomeer83felix-source.
fatal: unable to access 'https://github.com/xinnan369/sigma.git/': The requested URL returned error: 403
```

push 退出码 1。用户随后明确授权写入，本任务授权充分，但再次尝试和权限读取结果没有改变。此为服务端账号/集成权限问题，不能靠口头授权改变 token 权限。未向其他仓库推送、未新建 fork、未绕过权限，未启用 auto-merge。

恢复需要将本机 gh 或 GitHub 集成切换为对目标仓库实际可写的账号。恢复后先重新读取远端：如果仍无 refs，推送本地 `main` 最小初始化提交，再推送 feature；若已出现他人提交，保留其内容，基于最新 main 协调后重新检查，禁止 force push 覆盖。

```bash
git ls-remote origin
git push -u origin main
git push -u origin codex/R0-bootstrap
gh pr create --repo xinnan369/sigma --base main --head codex/R0-bootstrap --title "[R0] Bootstrap research bridge and Experiment 0 protocol" --body-file ../R0_PR_BODY.md
```

以上是**待执行发布步骤**；PR 创建依赖尚未成功的两次推送，本轮未执行可成功的 PR 创建。提交 PR 后仍需等待 ChatGPT PASS 并核对 head SHA 与 CI，本轮不合并。

## 安全检查与执行偏差

- 目标仓库仅提交文本、Python、配置、许可和测试日志，没有数据集、权重、密钥和大文件；上游独立 clone 未位于目标仓库内。提交文件检查另见 security_audit.txt（报告完成后生成）。
- 本次最初 checkout 意外触发上游 Git LFS 下载，外部 `.git/lfs` 留有约 461 MB 的对象/未完成缓存；未读取为模型、未评估、未复制到目标仓库、未上传。最终 skip-smudge 工作树中的三个权重文件只是 133/134 字节指针。此为执行偏差，不能宣称从未下载权重。
- 自动审批拒绝清理外部 `.git/lfs` 的命令，仅返回 `blocked by policy`，未给出更具体原因；未改用其他删除途径绕过。缓存清理仍未完成。
- 所有公开证据去除了个人机器路径；本地仓库位置只在聊天交付中提供。

## 未完成项目与 R1 gate

- **R0 阻塞项：**远端 main 初始化、feature 推送、PR URL/head、远端 CI 以及 ChatGPT 验收均未完成。
- **NOT RUN：**真实 Sigma 实例化/权重加载、selective-scan CUDA 编译与 kernel tests、MFNet clean/corrupt mIoU、真实模块替换效果、逐步隐藏状态观测、bootstrap CI、非 SSM 公平对照。无 GPU 实验结果，无状态污染或创新性结论。
- R1 设备/依赖/数据/checkpoint 清单和加载风险详见 UPSTREAM；先在兼容 CUDA torch 环境验证官方 clean gate，保存 MFNet split 和 checkpoint checksum，再实现完整成对推理、输入 Hook、响应统计和跨尺度坐标核对。当前工具还不能直接运行 Experiment 0 全流程。
- 未进入合并阶段；只有后续 ChatGPT 明确 PASS 且所验 head 未变化、CI/测试无退化时，才能按已授权规则合并。
