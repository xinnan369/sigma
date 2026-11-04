# Experiment 0 — Sigma 跨模态扰动传播诊断（预注册协议草案）

**状态：计划，尚未完成真实模型实验。** 目标是尽早证伪 H1：仅局部破坏 Thermal 后，跨模态融合是否导致 RGB/融合输出的区域外新错误，以及是否与 SSM 机制具有特异关系。

## 评估条件

- 数据：MFNet 官方测试划分；主干：Sigma-Tiny，官方预训练 MFNet checkpoint，固定 SHA 和模型文件校验和。
- 对每个样本运行 clean `(RGB,T)` 和 corrupted `(RGB,T')`；除 Thermal 外保持一切不变，包括 RGB、类别标注、预处理、输入尺寸和 eval 模式。
- 首轮在经过**确定性 resize** 后、进入归一化之前施加局部干预；支持遮挡、ROI Gaussian 噪声、ROI 内位移。每个样本保留明确布尔 mask，表示被干预的像素集合。
- 对照：identity 输入、扰动 seed 重复、不同 ROI 位置、扰动强度；R1 增加非 SSM 融合模块或 CNN/attention 基线。
- 正常任务：记录 clean mIoU、corrupt mIoU、各类别 IoU（含/不含背景的协议要注明）。
- 局部问题：记录干净正确→受扰错误（new error），干净错误→受扰正确（corrected error），区域外净错误增量，两次预测的分歧率；必须排除 GT ignore index。
- 特征层：每尺度 `backbone.cross_mamba[i]` 和 `backbone.channel_attn_mamba[i]` 的输入/输出 Hook；包括 `B,H,W,C` 与 `B,C,H,W`，不能混淆维度。记录归一化 L2 响应图、区域外响应比。避免一次性缓存全测试集特征。
- 干预机制：用干净前向记录的**同一模块输出**替换受扰前向对应输出，并测量 GT 错误变化。此实验称“模块输出替换”，不是 SSM 隐状态因果干预；若模块存在双分支 tuple 输出，必须同时替换对应结构。
- 注意：模型卷积与普通注意力本身也能导致非局部变化；仅凭响应扩散不能支持“状态污染”。

## 禁止过度推断

1. 未获得 CUDA/参考 SSM 逐步状态之前，不报告“状态 h_t”变化图。
2. 单个样本的可视化不是统计证据：至少报告样本数、置信区间（或 bootstrap）、不同扰动种子。
3. 如果 shift 导致 mask 大范围覆盖，使用明确的位移支持区域评价，不得拿全局位移数据计算所谓“区域外传播”。
4. 用同协议对照 CNN/attention，及简单输入门控；若它们达到同等改善，就应重新评估选题。

## 推进 R1 的 Gate

- [ ] 对 Sigma-Tiny 真实 checkpoint 验证 clean mIoU 与官方评估协议一致。
- [ ] 局部干预在扰动区域外导致可重复的新增错误；区分模块间的响应。
- [ ] 模块输出替换对错误有明确贡献，但不能据此宣称 SSM 机制特异。
- [ ] 下一阶段计划包含非 SSM 公平对照及真正隐藏状态插桩。

若任一核心假设没有证据，要求在报告中给出否定或不确定结论，不强行写成创新成果。

## R0 源码复核后的实施约定

固定源码版本和逐文件证据见 [UPSTREAM.md](UPSTREAM.md)。以下是对实现细节的澄清，不构成模型实验结果，也不改变 H1。

- Sigma-Tiny 的双模态编码器是**同一个** `backbone.vssm` 实例被调用两次；四尺度深度为 `[2,2,9,2]`，通道为 `[96,192,384,768]`。主分割头为 `MambaDecoder`，模型输出不是单独 RGB 分割头。
- `backbone.cross_mamba[i]` 的两输入以及 tuple 中两输出均为 BHWC；`backbone.channel_attn_mamba[i]` 的两输入和单一输出也均为 BHWC。只有其返回后 `dual_vmamba.py` 才将融合输出转回 BCHW。上游这两个 Block 的 forward docstring 写 BCHW，与调用点不符，应以真实操作为准。
- MFNet 读取灰度 Thermal 后复制为三个相同通道；resize 后 Gaussian 使用同一空间噪声广播至三通道，避免人为引入伪彩色模态。RGB 输入不传入扰动函数，R1 成对前向还需独立断言 RGB、GT、样本顺序完全相同。
- 官方 `ValPre` 不进行固定尺寸 resize；评估入口使用 OpenCV `INTER_LINEAR`、比例 `[1]`，crop `[480,640]`、stride `2/3`，flip 关闭。复制后的 Thermal 是 HWC 三通道，因此也走 `INTER_LINEAR` 分支并使用 RGB 相同的 mean/std。R1 clean gate 必须先原样复现官方入口；不得默默把它替换成任意固定 resize。扰动插入在官方比例 resize 之后、归一化及 padding/crop 之前，mask 保留在该图像坐标系，若改动坐标须同步映射 GT 和预测。
- 官方 `gt_transform=False`，类别 0 (`unlabeled`) 参与计分，255 不参与。`eval.py` 汇总全测试集混淆矩阵，再用 `nanmean(IoU[0:9])` 返回 mIoU；不等于逐图 mIoU 均值。R1 主指标保持含 0，另列不含 0 的 `nanmean(IoU[1:9])`。无 union 的类别 IoU 为 NaN，不用 0 代替。
- `paired_error_stats` 的所有 rate 分母均为 `N = count((GT != 255) & ~support)`；new error 和 corrected error 的分子分别是 clean 对/corrupt 错、clean 错/corrupt 对；net delta = `(new-corrected)/N`。两次都错但类别不同仅计入 disagreement。N=0 时计数为 0、rate 为 NaN，汇总时报告被排除数量。
- occlusion/gaussian 的 support 是整个目标 ROI；`shift_local` 只在 ROI 内搬移并填充，support 也为整个 ROI，非位移目的区域和图像外部的并集。identity、sigma=0、dx=dy=0 的 support 全 False。support 表示被干预位置，不能以数值碰巧未改变来缩小它。
- R1 响应图预定义为 `d(p)=||F_corrupt(p)-F_clean(p)||_2 / max(RMS_p(||F_clean(p)||_2), 1e-8)`，按模块/尺度/分支分别计算；区域外响应比为 `sum_outside d / sum_all d`，总响应为 0 时记 NaN 并单列零响应数。mask 需以保守覆盖映射到特征网格，记录具体映射方式；该几何区域外不等于卷积感受野之外。此响应指标本轮**未实现、未运行**。
- 模块输出替换必须同样本、同尺度、同次调用；CrossMamba 两个输出同时替换，ConcatMamba 替换单输出。使用上下文管理器清理 Hook，捕获为 detached copy，每次替换返回独立 copy，避免下游原地操作污染 clean 对照。R0 只测试通用辅助工具；R1 仍需真实模块输入 Hook、样本/调用匹配和在线特征汇总。
- 统计阶段至少固定并记录扰动 seeds（首个 3407），对 identity/不同种子/ROI/强度进行对照；以样本为 bootstrap 单位，不能把像素作为独立样本。置信区间实现和样本数尚待 R1 冻结。
