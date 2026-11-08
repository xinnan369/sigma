# R1 运行与恢复说明

固定上游：`98ba3c6938d29169bbd27f91519084bb462561df`。本任务原始入口是其 `eval.py`，不修改或移植评估实现。

## 本次环境策略

Ubuntu 22.04 / WSL2，RTX 3090 24GB。系统 Python 3.10 的 venv 缺 ensurepip；已有 Linux conda Python 可创建 venv。本轮先尝试全新 Python 3.11 环境下载 torch 2.2.1+cu121，因下载速度约 180kB/s、预计超过一小时，主动中断该任务专属安装；未将该安装写为成功。

改用已存在的 CUDA torch 环境解释器建立独立 overlay venv（`--system-site-packages`）。依赖安装只写新 venv，原环境不修改。复用的 torch 2.2.2+cu121 是相对上游 README 2.2.1 的 patch-version 差异，必须通过真实兼容性测试，不声称是上游原测试环境。实际包版本和结果以 R1_REPORT、evidence/r1 为准。

示例 Linux 命令；变量由执行者设置为本机路径，不提交个人路径：

```bash
"$R1_BASE_PYTHON" -m venv --system-site-packages "$R1_ENV"
"$R1_ENV/bin/python" -m pip install -r "$BRIDGE/requirements-r1.txt"
export GIT_LFS_SKIP_SMUDGE=1
git clone --no-checkout https://github.com/zifuwan/Sigma.git "$UPSTREAM"
git -C "$UPSTREAM" checkout --detach 98ba3c6938d29169bbd27f91519084bb462561df
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=2
cd "$UPSTREAM/models/encoders/selective_scan"
"$R1_ENV/bin/python" -m pip install --no-build-isolation .
```

cuda toolkit 12.6 与 torch CUDA build 12.1 的 minor-version 差异必须记录；是否能用由实际编译和数值测试决定。扩展从固定上游源码重新构建，不复用来源不明的已安装 selective_scan_cuda_core。

## 兼容性测试

```bash
cd "$BRIDGE"
"$R1_ENV/bin/python" -m experiments.r1.gpu_probe
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$R1_ENV/bin/python" -m pytest -q tests
cd "$UPSTREAM/models/encoders/selective_scan"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$R1_ENV/bin/python" -m pytest -q test_selective_scan.py -k 64
```

上游测试一共预定义 1920 个组合。`-k 64` 是长度 64 的 192 个组合，覆盖 dtype float32/fp16/bf16、nrows 1–4、可变 B/C groups 1/2、D 和 delta 参数开关，以及 forward/backward 对照；不是完整 suite，也不是网络/数据集实验。未执行的长度组合明确标为 NOT RUN。

审查发现上游测试使用 `out, *rest = selective_scan_fn(...)`，但函数实际返回 Tensor，因此仅取 batch 0 比较，不能据此声称完整 batch 覆盖。R1 另提供 `python -m experiments.r1.kernel_batch_check --upstream "$UPSTREAM"`，直接调用固定版本的原始函数/参考实现，在 B=2、长度 257 下比较完整输出与所有输入梯度，覆盖三种 dtype × 四种 nrows，共 12 个补充案例。其结果也仅是合成 kernel 兼容性证据。

## 获取资产与 clean gate

- 数据原始来源：[MFNet 官方项目页](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/)，原始图像为四通道。不能直接以该四通道文件冒充 Sigma 的三目录布局；优先核对 [Sigma 作者提供的处理后数据](https://drive.google.com/drive/folders/1GD4LYF208h9-mHJ_lxW11UM0TPlRmv0z) 的 MFNet 目录和官方测试清单。
- 分割权重来源：固定 README 的 [Sigma-T-MFNet](https://drive.google.com/file/d/1N9UU9G5K8qxKsZOuEzSLiCGXC5XCaMaU/view)。本轮该链接返回 404，不能自行用其他模型或 VMamba 初始化权重替换。需获得作者可访问的新链接或经说明、核对的同一官方文件。
- 本地文件齐备后运行 `python -m experiments.r1.preflight --upstream "$UPSTREAM" --dataset "$MFNET_ROOT" --dataset-source "$DATA_SOURCE_URL" --checkpoint "$CHECKPOINT" --checkpoint-source "$CKPT_SOURCE_URL"`。该工具只审核固定源码与文件结构/hash，不反序列化权重，不能证明文件确为官方版本；需将 test.txt 和权重 hash 与实际官方来源记录对应起来。
- 原始 config 要求 `$UPSTREAM/datasets/MFNet/{RGB,Modal,Label,test.txt}`。可将外部数据目录链接到此处，保持 eval 配置不变。不要提交链接、数据或权重；上游 clone 必须位于交接仓库之外。

全部满足后，原始命令为：

```bash
cd "$UPSTREAM"
CUDA_VISIBLE_DEVICES=0 "$R1_ENV/bin/python" eval.py -d 0 -n mfnet -e "$CHECKPOINT"
```

同时采集 `/usr/bin/time -v` 的墙钟时间/进程内存和 nvidia-smi 的显存采样。显存采样最大值不是精确 allocator peak，必须注明采样频率。必须保存严格 state_dict 加载成功日志、393 张实际测试文件清单和 9 类结果。配置 9 类（含 unlabeled=0）、GT 255 排除、test.txt、不变的 RGB/Thermal 预处理、单尺度 `[1]`、不 flip、crop `[480,640]`、stride `2/3`，均不得变更。模型构造器的初始化权重加载失败会打印后继续，不能把它当作分割 checkpoint 已成功加载。

缺数据或官方分割 checkpoint 时，原始 eval 命令为 NOT RUN；不运行随机权重评估来填补指标。
