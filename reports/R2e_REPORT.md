# R2e — Selective Scan 因果路径分解

R2d 的验收 Head 为 `d07aac9cd02e39925723b7c381f45fcce8e27e1a`。补充 seed 元数据说明的提交为 `d88b3c9a1824146ff6082838c028df4b261dc8cc`，该提交只修改 R2d 报告。Windows 81 项通过、2 项跳过，更新后的 GitHub Actions 两项通过。合并提交 `f8cdbfe4ed8a32373225c621cc97c983930cf92a` 已核对包含更新后的 Head。[合并记录](evidence/r2e/r2d-merge.txt)

## 文献对照

2026-11-12 核对 arXiv、CVF、IEEE、Elsevier 和 Taylor & Francis 的论文或出版页。表中的状态传播特指在固定权重下干预扫描内部路径，并追踪局部扰动的空间外影响。

| 工作 | 科学问题与融合 | 状态空间与可靠性 | 数据与实验 | 与 R2e 的重叠 |
|---|---|---|---|---|
| RSGMamba（Xu 等，2026，arXiv v2） | 辅助模态存在噪声、缺失与不一致。RSGMB 做全局跨模态融合，LCGM 用局部交叉门控补充细节。 | 两路选择性状态递推。模态不确定性与跨模态一致性门控共同调节有效读出 `C_eff`。 | NYUDepth V2、SUN-RGBD、MFNet、PST900 的分割比较与组件消融。 | 与 R2e 的跨模态读出及可靠性动机高度重叠。已查阅的实验采用门控、组件消融和可视化，未见固定模型中按 `h/C/Dx` 对同一局部扰动做完整反事实分解。[论文](https://arxiv.org/html/2604.12319v2) |
| SCRNet（Zhao 等，2025） | 高层 RGB 与 Thermal 语义冲突。SRM 用视觉 Transformer 的全局上下文校正冲突，CFM 用相似度与通道注意力增强互补信息。 | 以全局一致性筛选特征，出版页所述机制无选择性状态递推。 | MFNet、PST900。MFNet 沿用数据集划分政策。 | 对冲突和互补的科学问题重叠，公开方法与实验针对特征校正和分割性能。[出版社](https://www.sciencedirect.com/science/article/abs/pii/S0031320325000585) |
| RegSeg（Lai 等，IEEE TIP，2024 在线发表） | RGB 与 Thermal 错位。联合训练配准和分割，通过形变场对齐热特征后融合，配准用互信息与语义辅助监督。 | 编解码结构使用卷积特征与矩阵分解建模全局，公开方法重点为显式配准。 | IR SEG 为主要错位评估场景，比较配准与分割及消融。 | 空间扰动动机相近，形变对齐与 R2e 的 scan 内部路径干预对应不同操作。[IEEE](https://doi.org/10.1109/TIP.2024.3501077) |
| Sigma（Wan 等，WACV 2025） | 高效 RGB-X 分割。Siamese VMamba 提取特征，CroMB、ConMB 融合，CMD 解码。 | CroMB 通过跨模态系数进行状态读出。论文比较 A/B/C 交换位置和模块消融。 | MFNet、PST900、NYUDepth V2、SUN-RGBD，公开分割基准与架构消融。 | 已有 C 交换与融合模块贡献证据。R2e 的对象正是该扫描，差异在于固定权重与输入配对下的路径干预和空间响应。[CVF](https://openaccess.thecvf.com/content/WACV2025/html/Wan_Sigma_Siamese_Mamba_Network_for_Multi-Modal_Semantic_Segmentation_WACV_2025_paper.html) |
| DSMNet（2026） | RGB-T 全局依赖与纹理补偿。双路 VMamba，加全局和多方向相关性模块、频域补偿与融合。 | 使用方向扫描和频域特征增强。论文所列机制着眼相关性、全局监督与局部频域细节。 | MFNet、PST900 的原基准划分，组件消融及效率对比。 | 已覆盖扫描方向和长程依赖设计，空间外预测变化的路径干预在已查阅的实验中未见。[出版社](https://doi.org/10.1080/10095020.2026.2707066) |
| Mul-VMamba（2026，DOI 2025） | RGB-XS 分割的效率和空间一致性。MCS 在同位置选择特征，MCFM 用 Cross-S6 做跨模态融合。 | 线性复杂度 S6，具有输入选择机制。 | 出版页可核实 MCubes 分割和效率对比。其余协议细节未独立核实。 | 与跨模态选择性扫描重叠。已核实的实验针对融合设计与分割准确率。[出版社](https://doi.org/10.1016/j.knosys.2025.115119) |

RSGMamba 已将可靠性门控用于状态读出，Sigma 已比较跨模态参数交换。因此本项目可检验的新问题是，固定权重和局部输入扰动下，替换哪条内部路径会改变区域外预测，以及这种效应是否跨种子重复。一般可靠性门控属于已有设计。当前查阅范围支持这一研究定位，尚不足以主张优先权。

## 实验边界

选择性状态递推为 `h[t] = exp(Δ[t]A) h[t−1] + Δ[t]B[t]x[t]`。扫描读出为 `C[t]h[t] + D*x[t]`。在固定 checkpoint 下，`A`、`D` 与步长 bias 按调用核对相同，`x/Δ/B` 来自实际投影及进入内核的输入。混合调用用原 CUDA 内核实现指定状态轨迹，并选择独立的 `C` 及 `D*x`。参考实现只用于数值审计。普通模块输出单独记录，扫描状态通过递推和 CUDA 分块末端前缀核对。

原始 clean/corrupt 全图输出各自按 R2d 哈希核对。全 clean 或全 corrupt 的 scan 端点均采用各自原始输出。联合路径干预以 clean 输入为背景，替换目标范围内的 scan 读出，保留其他算子的自然前向；自然 corrupt 前向提供独立对照。逐元素数值容差采用 R2b 原值，任一失败均阻止全量。

## GPU 闸门结果

15 份 R2d Epoch 5 权重均通过 ledger、审计证书和实际文件 SHA256 的三方核对。首张 pilot 图 `00057D`、seed 3407、SSM 臂在共享编码器 `backbone.vssm.layers.2.blocks.2.op` 的 RGB 调用触发数值闸门。该调用的网格为 30×40，包含四方向、3,072 扫描通道、16 状态分量、1,200 步。此次复查先验证了该调用的 CUDA 块末端前缀，再在读出处失败。

| 原生读出与参考读出比较 | 最大绝对偏差 | 峰值归一化偏差 | 相对 RMS | 最大逐元素误差比 |
|---|---:|---:|---:|---:|
| R2b 原参考求和 | 1.22786e−4 | 1.31211e−6 | 4.80614e−7 | 2.46644 |
| 按 CUDA 状态索引累加 | 1.26839e−4 | 1.35543e−6 | 4.76578e−7 | 2.34286 |

逐元素误差比为 `|参考−CUDA| / (2e−5 + 2e−4×|CUDA|)`，上限为 1。复查调整读出加法顺序后仍超限。门槛保持 R2b 原值。两次均在首图完成反事实回放前退出，完整 pilot 结果为 **FAIL**。全量 GPU 因果实验、双层 Bootstrap、seed-level t 区间和路径效应均为 **NOT RUN**。[初次日志](evidence/r2e/pilot-gpu-attempt-1.txt) · [复查日志](evidence/r2e/pilot-gpu.txt) · [15 份权重指纹](evidence/r2e/pilot/assets.json)

扫描布局记录保留模块名、调用索引、网格、各方向与时间步映射。共享 SS2D 使用行、列及各自反向的四个方向。CrossMamba 两模态各自采用行优先，ConcatMamba 采用正向和逆向的模态拼接序列。坐标指特征网格索引。

独立的原生前向诊断在 RTX 3090 上完成了同一 12 图的三架构核对。72 次 clean/corrupt 的预测哈希均与 R2d 一致，移除观测 Hook 后的 logits 逐位一致。每个前向实际观测到 30 次共享编码器、12 次共享解码器调用，SSM 臂另外记录 8 次 CrossMamba 与 4 次 ConcatMamba 调用。首轮原生诊断因在同一观测上下文执行两次前向触发调用计数检查。改为每次前向使用独立上下文后通过，耗时 65.325 s。此诊断采用原生读出，混合输出和 Identity 替换仍待完整 pilot 验证。[诊断证书及布局](evidence/r2e/baseline) · [运行日志](evidence/r2e/baseline-gpu.txt) · [初次诊断日志](evidence/r2e/baseline-gpu-attempt-1.txt)

## 科研判断

当前 GPU 证据显示，将 R2b 的参考递推扩展到共享编码器时需要进一步做逐元素数值验证。R2e 尚无合格的全图反事实样本，因此状态轨迹是否介导扰动、C 路径是否更重要、直接输入项能否解释大部分变化、共享编码器或解码器是否主导，以及哪条路径跨种子可重复，均待检验。文献对照支持以固定权重、配对内部干预和空间外误差定位为研究问题。它和 RSGMamba 的读出可靠性门控存在机制对象上的交集，证据目标不同。

后续数值验证可直接追踪 CUDA 内核每一步状态分量，分别核对递推、读出累加及接近零处的差异，再重跑固定 pilot。两次失败及检查过的权重摘要随本 PR 保存。

Windows 测试 83 通过、2 跳过，WSL 85 通过。统计代码已实现配对种子和视觉组抽样、三因素交互以及 seed-level t 区间，输入须为通过闸门的 15 份全量记录。[Windows](evidence/r2e/tests-windows.txt) · [WSL](evidence/r2e/tests-wsl.txt)
