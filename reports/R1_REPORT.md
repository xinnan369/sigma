# R1 — Sigma-Tiny + MFNet 真实基线复现

日期：2026-11-04（Asia/Shanghai）。**GPU 依赖、源码扩展编译及选定兼容性测试已完成；参考分割 checkpoint 不可获取，完整测试图像/标注未就绪，clean baseline 为 NOT RUN。R1 未验收、未合并。**

## R0 合并与 R1 分支

- 用户正式给出 R0 PASS，并指定验收 head `3d300624cc5c9022ba83b9b8e059881dc44793d8`。
- 合并前重新读取 PR #1：head 完全相同，OPEN，MERGEABLE/CLEAN，push 和 pull_request 两个 cpu-tests 均 COMPLETED/SUCCESS，无新提交或冲突。
- 实际执行 `gh pr merge 1 --repo xinnan369/sigma --squash --match-head-commit 3d300624cc5c9022ba83b9b8e059881dc44793d8`，退出码 0。
- GitHub 确认 MERGED，合并时间 `2026-11-04T12:10:17Z`；**squash commit：`68b7dacce44a995bc821fdbcfd51b64bdef3daec`**。fetch 后 main fast-forward，核实 R0 的 33 个仓库文件均在 main；再创建 `codex/R1-sigma-clean-baseline`。
- 合并证据：[r0-merge.txt](evidence/r1/r0-merge.txt)。R1 最终 head SHA 和 PR 地址由最终交付与 PR 描述记录，避免报告引用自身 commit SHA。

## 环境与构建

| 项目 | 实际结果 |
|---|---|
| OS / Python | Ubuntu 22.04 / WSL2，Python 3.11.11 |
| GPU / driver | NVIDIA GeForce RTX 3090，24576 MiB，610.88 |
| torch / torchvision | 2.2.2+cu121 / 0.17.2+cu121 |
| CUDA torch build / nvcc | 12.1 / CUDA Toolkit 12.6 V12.6.85 |
| 编译器 | Ubuntu GCC 11.4.0；MAX_JOBS=2 |
| NumPy / OpenCV | 1.26.4 / 实际 cv2 4.9.0（opencv-python distribution 4.9.0.80） |
| timm / einops | 0.4.12 / 0.8.1 |
| 扩展 | 固定上游 selective_scan 0.0.1，从源码重新编译，两个接口均从新 overlay 加载 |
| 上游 commit | `98ba3c6938d29169bbd27f91519084bb462561df`，独立 Linux 文件系统 clone，工作树最终干净 |

环境是独立 **overlay venv**，通过 `--system-site-packages` 复用已有 CUDA torch；不宣称为完全 hermetic 安装。原 conda 环境未修改，timm 等新增/降级依赖写入 overlay。完整包盘点见 [environment-packages.txt](evidence/r1/environment-packages.txt)。继承环境同时有 opencv-python-headless distribution 元数据，实际导入版本以 probe 的 cv2 4.9.0 为准；后续完整重建应消除重复 OpenCV distribution 风险。

实际 torch 比上游 README 声明的 2.2.1 有 patch-version 差异，nvcc 12.6 与 torch CUDA 12.1 有 minor-version 差异。这些差异已披露；成功编译与数值测试仅证明本次覆盖的兼容性，不保证所有真实模型输入均正确。上游源码、评估配置和测试划分未修改。克隆时设置 GIT_LFS_SKIP_SMUDGE=1，没有在 R1 checkout 下载初始化权重。

构建命令（本机路径以变量表示）：

```bash
"$R1_BASE_PYTHON" -m venv --system-site-packages "$R1_ENV"
"$R1_ENV/bin/python" -m pip install -r "$BRIDGE/requirements-r1.txt"
export GIT_LFS_SKIP_SMUDGE=1
git clone --no-hardlinks --no-checkout "$LOCAL_FIXED_UPSTREAM" "$UPSTREAM"
git -C "$UPSTREAM" checkout --detach 98ba3c6938d29169bbd27f91519084bb462561df
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=2
cd "$UPSTREAM/models/encoders/selective_scan"
"$R1_ENV/bin/python" -m pip install --no-build-isolation .
```

依赖与扩展安装均退出 0；完整安装输出见 [dependency-install.txt](evidence/r1/dependency-install.txt)、[extension-build.txt](evidence/r1/extension-build.txt)。命令恢复说明见 [R1_RUNTIME.md](../docs/R1_RUNTIME.md)。

已安装核心扩展 `.so` 的 SHA256 为 `833021baeecf26b84d5ecba208417893a9eda202c38854fd9a9737c8ccd7513e`，确认来自新 overlay，见 [runtime-fingerprint.txt](evidence/r1/runtime-fingerprint.txt)。这不是分割模型权重的 SHA，未上传二进制。

### 真实诊断与中断记录

- 系统 Python 3.10 venv 缺 ensurepip，创建失败；改用已有 conda Python 创建 venv，未修改系统 Python。
- 首次尝试全新 torch 2.2.1+cu121：757MB wheel 只下载约 64.5MB，速度约 180kB/s、剩余预计一小时，主动中断该任务专属 installer。该安装不算成功；之后复用已有 torch 2.2.2+cu121。没有留下运行中的下载任务。
- 首版依赖计划强制下载新版 OpenCV，随后审查已有版本满足上游要求，停止该次 installer，最终固定 OpenCV 4.9.0.80 / SciPy 1.13.0 复用安装，依赖安装完成。未改变图像预处理代码。
- 临时 bootstrap shell 初次因 Windows CRLF 导致 pipefail 解析失败，转为 LF 后重跑成功。
- 扩展指纹工具首次直接 import 扩展，缺少预加载 torch 导致 libc10.so 加载失败；修正工具先 import torch 后，指纹验证退出 0。首次失败日志保留于 runtime-fingerprint-first-attempt.txt；不是数值测试失败。
- WSL 每次启动提示 localhost 代理/NAT 警告，网络访问与 GPU 测试实际可用；部分原始 stderr 因 WSL 混合编码保留替换字符，测试 stdout 与退出码完整。

## 实际测试

| 命令 / 覆盖 | 实际结果 | 退出码 |
|---|---|---|
| `python -m experiments.r1.gpu_probe` | CUDA True；所有依赖导入成功；确定性 CUDA matmul 与 CPU 精确对照 PASS | 0 |
| 上游 `python -m pytest -q test_selective_scan.py -k 64` | **192 passed, 1728 deselected in 10.54s** | 0 |
| `python -m experiments.r1.kernel_batch_check --upstream ...` | **12 个完整 batch 输出与所有输入梯度案例全部 PASS** | 0 |
| WSL overlay `python -m pytest -q tests` | **30 passed in 3.78s** | 0 |
| Windows R0 CPU venv `python -m pytest -q tests` | **30 passed in 5.44s** | 0 |
| 原始上游 `python eval.py --help` | 原始 CLI 导入/帮助可执行；不是模型构建或评估 | 0 |
| `python -m compileall -q experiments tests tools` | 无输出 | 0 |
| `python -m experiments.r1.preflight --upstream ...` | 固定源码匹配；完整数据与 checkpoint 缺失；明确阻止 baseline | **2（预期阻塞）** |
| `python -m experiments.r1.check_sources` | checkpoint 404；数据页面 200；未将网页当资产 | **2（预期阻塞）** |

GPU probe seed 3407。矩阵 smoke elapsed 0.420s，allocator peak allocated 8,521,728 bytes；两者仅代表小型 CUDA smoke，**不代表 Sigma 推理时间或显存**。上游数值测试及完整 batch 补充使用 seed 0。

上游 192 案例覆盖长度 64，float32/fp16/bf16、nrows 1–4、B/C groups、D/bias/softplus 开关，包含 backward；其余 1728 案例 **NOT RUN**。审查发现原始测试 `out, *rest = ...` 对 Tensor 解包，实际只比较 batch 0；因此增加 B=2、长度 257、三种 dtype × 四种 nrows 的 12 案例，直接比较完整输出与 u/delta/A/B/C/D/bias 梯度。补充案例的 forward 最大绝对误差分别约为 FP32 0.000298、FP16 0.015625、BF16 0.0625，均在预定 dtype 容差内，不能解释为模型精度。

完整日志：[gpu-probe](evidence/r1/gpu-probe.txt)、[selective-scan](evidence/r1/selective-scan.txt)、[完整 batch](evidence/r1/kernel-full-batch.txt)、[WSL 单测](evidence/r1/wsl-bridge-tests.txt)、[CPU 单测](evidence/r1/bridge-tests.txt)、[eval help](evidence/r1/eval-help.txt)、[preflight](evidence/r1/preflight.txt)、[sources](evidence/r1/sources.txt)。GitHub CI 只运行 bridge CPU 测试，不能代替上述本机 GPU 证据；最终 CI 状态在 PR checks 中核对。

## 官方数据与权重

| 资产 | 来源与本轮结果 | SHA256 |
|---|---|---|
| Sigma 作者提供的 MFNet `test.txt` | [作者 MFNet 文件夹](https://drive.google.com/drive/folders/1bD5b4xXkDBasOLhupUBDdw2rq9oogbu_)；文件 ID `1SlNee8aFtwJXyKtAPZ3umKtEFPfyLOOz`，HTTP 200，原字节 2751，393 行、393 唯一样本，未修改顺序/内容 | `a236aa6ac7e55ec6b5e419b55c6934439611453939eee2c0bf7b0f0c34917c9d` |
| 完整 MFNet 测试 RGB/Modal/Label | [MFNet 原始项目](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/) 与 [Sigma 处理后目录](https://drive.google.com/drive/folders/1GD4LYF208h9-mHJ_lxW11UM0TPlRmv0z) 均可访问，作者 MFNet 下三目录可列出；图像与标签未下载 | **NOT AVAILABLE** |
| Sigma-Tiny/MFNet 分割 checkpoint | 固定上游 README 的 [官方文件](https://drive.google.com/file/d/1N9UU9G5K8qxKsZOuEzSLiCGXC5XCaMaU/view)，查看页与 uc 下载入口均 HTTP 404；usercontent 下载入口另核查也 404 | **NOT AVAILABLE** |
| VMamba 初始化权重 | 与 MFNet 分割 checkpoint 不同；本轮未下载、未代用 | NOT AVAILABLE |

用户确认没有现成本地数据/权重。只获取了小型官方测试清单到仓库外；在参考分割权重 404、不能完成官方 baseline 的情况下，没有继续批量下载图像/标签。文件夹页面可访问不能证明全部图像已获得，也不能生成不存在的文件 SHA。test.txt 及任何原始数据、权重均未提交；仅提交来源、hash、计数元数据，见 [mfnet-split.json](evidence/r1/mfnet-split.json)。

预检查会记录所有测试 triplet 文件的实际 SHA、保持清单顺序、拒绝重复/缺文件和 LFS 指针；它不会反序列化 checkpoint，也不会仅凭 393 这个数量声称官方 split 身份。恢复后仍需核对图像通道、标签集合、来源 hash 和严格 state_dict 加载日志。

## clean baseline 结果：NOT RUN

拟执行且未执行的原始命令：

```bash
cd "$UPSTREAM"
CUDA_VISIBLE_DEVICES=0 "$R1_ENV/bin/python" eval.py -d 0 -n mfnet -e "$OFFICIAL_SIGMA_TINY_MFNET_CHECKPOINT"
```

| 官方类别 | IoU |
|---|---|
| 0 unlabeled | NOT RUN |
| 1 car | NOT RUN |
| 2 person | NOT RUN |
| 3 bike | NOT RUN |
| 4 curve | NOT RUN |
| 5 car stop | NOT RUN |
| 6 guardrail | NOT RUN |
| 7 color cone | NOT RUN |
| 8 bump | NOT RUN |
| 总体 mIoU（含 0） | **NOT RUN** |

真实样本执行数 0；模型 eval 耗时/显存为 NOT RUN，无 clean/corrupt 分数，也没有以随机初始化模型、第三方权重或上游论文表格数值填表。未实例化并加载完整分割 checkpoint，未证明完整 Sigma 480×640 前向成功；kernel 与 CLI smoke 不能替代这些 gate。

官方协议维持 test.txt、9 类（0 参与，255 排除）、全数据集 confusion matrix 后计算 nanmean IoU、`gt_transform=False`、灰度 Thermal 复制三通道、原始 normalization、scale `[1]`、flip False、crop 480×640、stride 2/3。不改变测试划分或使用其他 checkpoint。

## 阻塞与恢复

1. 需要作者恢复官方 Sigma-Tiny/MFNet checkpoint 链接，或提供经说明和校验的同一官方文件；记录真实 SHA256 和来源后才能加载，不能以镜像/第三方模型不说明地替代。
2. 需要取得官方清单对应的 393 个完整 RGB/Modal/Label triplet，逐文件 hash、图像/标注审计；保持已获取 test.txt 的 SHA 不变。
3. 两项齐备后运行原始 eval.py，保存严格权重加载日志、九类 IoU、主 mIoU、墙钟时间和注明采样频率的显存日志，再更新 R1 PR 等待验收。
4. 本轮新增文件安全检查在提交前执行；无 dataset/weights/build binaries/credentials 上传。先前 R0 外部 LFS 缓存状况没有改变，本轮不尝试绕过清理审批。

**结论：R0 已按 PASS 正确合并；R1 的 GPU 兼容性准备完成，真实基线复现因资产不齐而阻塞，不作科研成功结论。**
