# Sigma 固定上游核查（2026-11-04）

- 源仓库：<https://github.com/zifuwan/Sigma>；交接仓库 `xinnan369/sigma` 是独立研究仓库。
- 实际本地 HEAD：`98ba3c6938d29169bbd27f91519084bb462561df`，detached checkout，最终 `git status --short` 无输出。
- [LICENSE](https://github.com/zifuwan/Sigma/blob/98ba3c6938d29169bbd27f91519084bb462561df/LICENSE)：MIT，Copyright (c) 2024 Zifu Wan。备份见 [Sigma-MIT.txt](licenses/Sigma-MIT.txt)。仅复制许可文本，没有复制模型实现；依赖包和外部数据的许可须另行核查。
- 逐文件 SHA256 与 checkout 证据见 [upstream_audit.txt](../reports/evidence/upstream_audit.txt)。这是静态源码审查，未构建上游模型或验证权重。

## 安全获取源码

PowerShell，克隆到交接仓库外：

```powershell
$env:GIT_LFS_SKIP_SMUDGE='1'
git clone --no-checkout https://github.com/zifuwan/Sigma.git upstream-Sigma
git -C upstream-Sigma checkout --detach 98ba3c6938d29169bbd27f91519084bb462561df
git -C upstream-Sigma rev-parse HEAD
git -C upstream-Sigma status --short
```

POSIX shell 用 `export GIT_LFS_SKIP_SMUDGE=1`。R0 禁止 `git lfs pull`。本次首次 checkout 未禁用 smudge，曾意外触发 LFS 下载，已中断并改为 skip-smudge checkout；工作树的三个 `.pth` 现为 133/134 字节的文本指针，没有用于模型运行。外部 clone 的 LFS 缓存清理被自动审批拒绝，未纳入交接仓库，见报告。

## 模型与 Hook 对应

以下文件链接均固定在上述 commit。

| 源码 | 实际行为与 R0 结论 |
|---|---|
| [models/builder.py](https://github.com/zifuwan/Sigma/blob/98ba3c6938d29169bbd27f91519084bb462561df/models/builder.py) | `sigma_tiny` → `dual_vmamba.vssm_tiny`，四尺度 `[96,192,384,768]`；MFNet 配置 `MambaDecoder`，输出插值回输入分辨率，`align_corners=False`。 |
| [dual_vmamba.py](https://github.com/zifuwan/Sigma/blob/98ba3c6938d29169bbd27f91519084bb462561df/models/encoders/dual_vmamba.py) | Tiny depths `[2,2,9,2]`；共享 `self.vssm` 分别编码 RGB 和 Thermal。每尺度先调用 `cross_mamba[i]` 再 `channel_attn_mamba[i]`；两个 Hook 边界都是 BHWC，后者返回后转 BCHW。 |
| [vmamba.py](https://github.com/zifuwan/Sigma/blob/98ba3c6938d29169bbd27f91519084bb462561df/models/encoders/vmamba.py) | CrossMambaFusionBlock → CrossMambaFusion_SS2D_SSM → Cross_Mamba_Attention_SSM；两分支扫描交换 C 参数，返回双分支特征并加残差。不能笼统称其为两条 2HW 拼接扫描。 |
| 同上 ConcatMambaFusionBlock | 使用 ConMB_SS2D；CrossScan_multimodal 构造 `(B,2,C,2HW)` 正向拼接与反向序列；cross_selective_scan_multimodal_k2 返回两模态特征，再经跨模态 SE 门控、通道拼接和投影；Block 输出 `x_rgb+x_e+op(...)` 的单一特征。 |
| 同上 Backbone_VSSM.load_pretrained | Tiny 构造器硬编码 VMamba 初始化权重路径；加载异常会被捕获并打印，不一定使构建失败。R1 必须审核初始化和最终 MFNet checkpoint 的加载日志、missing/unexpected keys，不能以成功构建视为成功加载。 |

这两个 Block 的 forward docstring 标注 BCHW，与实际调用矛盾。Hook 只能观测输入/模块输出，不能称为 SSM 隐状态 `h_t`。R0 通用输出工具未接入真实 Sigma，也没有实现内部扫描状态插桩。

## MFNet 官方评估

- [config_MFNet.py](https://github.com/zifuwan/Sigma/blob/98ba3c6938d29169bbd27f91519084bb462561df/configs/config_MFNet.py)：seed 3407，`test.txt`，配置声明 393 个测试样本（未下载数据核实）；9 类，`gt_transform=False`，`x_is_single_channel=True`，crop 480×640，scale `[1]`，flip False，stride 2/3。
- [RGBXDataset.py](https://github.com/zifuwan/Sigma/blob/98ba3c6938d29169bbd27f91519084bb462561df/dataloader/RGBXDataset.py) 将 Thermal 灰度图复制成三个通道；[dataloader.py](https://github.com/zifuwan/Sigma/blob/98ba3c6938d29169bbd27f91519084bb462561df/dataloader/dataloader.py) 的 ValPre 原样返回。
- [engine/evaluator.py](https://github.com/zifuwan/Sigma/blob/98ba3c6938d29169bbd27f91519084bb462561df/engine/evaluator.py) 执行比例 resize、归一化、padding/滑窗、eval/no_grad、输出 `exp(score)` 聚合和 argmax。MFNet 的三通道 Thermal 与 RGB 使用相同 ImageNet mean/std 和线性插值。官方并非任意 resize 到固定 480×640。
- [eval.py](https://github.com/zifuwan/Sigma/blob/98ba3c6938d29169bbd27f91519084bb462561df/eval.py) 将所有样本混淆矩阵相加；[utils/metric.py](https://github.com/zifuwan/Sigma/blob/98ba3c6938d29169bbd27f91519084bb462561df/utils/metric.py) 有效 GT 为 `[0,9)`，255 排除，IoU 为 TP/(GT+Pred−TP)，主 mIoU 为 `nanmean(iou)`（含类别 0），另计算 `nanmean(iou[1:])`；`show_no_back=False` 没有从主 mIoU 中排除 0。

## R1 设备、依赖、数据与权重清单

- 建议独立 Linux/CUDA 环境。上游 README 声明测试过 PyTorch 1.13.1 + CUDA 11.7、2.2.1 + CUDA 12.1，示例 Python 3.9；本次 Windows/Python 3.13/CPU torch 不属于经上游声明验证的组合。实际可用 3090 24GB，所需显存仍须 R1 实测。
- 锁定兼容 Python、CUDA 版 PyTorch、torchvision、CUDA toolkit/编译器；安装上游 requirements（其中 timm 固定 0.4.12），另核对实际 import 所需 einops、mamba_ssm 及 selective-scan 扩展。先编译 `models/encoders/selective_scan`，运行上游 kernel tests 并保存结果，不能只看 nvcc 或 nvidia-smi。
- 经授权本地准备 MFNet 的 RGB/Modal/Label 与官方 test.txt；记录每文件清单、划分 checksum、真实样本数与标签映射，不提交原始数据。
- 经授权准备对应 Sigma-Tiny/MFNet 分割 checkpoint，记录来源、SHA256、权重键加载结果；VMamba 初始化权重与 MFNet 分割 checkpoint 是两种不同对象。上游链接可获得性和最终模型文件本轮未验证。
- 官方 clean gate 通过后，才实现成对扰动前向、输入/输出 Hook、跨尺度 mask、特征响应统计、模块替换、置信区间和公平非 SSM 对照。真实 hidden-state 干预属于后续专项工作。
