# R2k-Q1v2-P2 独立数据准入调查

**总状态 `BLOCKED_NO_ELIGIBLE_INDEPENDENT_DATA`。** LasHeR 值得继续索取元数据。现有证据无法将它纳入主验证集。TartanRGBT 有跨设备补充价值，当前版本同样未获准入。此次核对仅涉及官方文档、论文、目录和小型公开元数据。原始影像核验、真实质量推理、训练和 GPU 均为 **NOT RUN**。 [P0 冻结判据](../docs/R2k_Q1v2_INDEPENDENT_EVALUATION.md)

## LasHeR 发布清单与完整性

使用 LasHeR 官方仓库固定提交 `c7ca5dbc96eed90b2b0b7dccc5014d1e611b8399`。对 Protocol 2 的两个文本清单按空白切分，对压缩包内 `AttriSeqsTxt/*.txt` 按文件名对齐、按 `Attributes_order.txt` 读取。1224 条属性行各有 19 个字段。两套清单无内部重复、无交集，与属性 ID 集合完全一致。这些是按目标类别安排的跟踪划分。 [Protocol 2 清单](https://github.com/BUGPLEASEOUT/LasHeR/tree/c7ca5dbc96eed90b2b0b7dccc5014d1e611b8399/List_for_Protocol2) · [属性包](https://github.com/BUGPLEASEOUT/LasHeR/tree/c7ca5dbc96eed90b2b0b7dccc5014d1e611b8399/Attributes) · [发布说明](https://github.com/BUGPLEASEOUT/LasHeR/blob/c7ca5dbc96eed90b2b0b7dccc5014d1e611b8399/README.md)

| 公开序列标签 | Training | Testing | 合计 |
| --- | ---: | ---: | ---: |
| 序列 ID | 979 | 245 | 1224 |
| MB，目标运动模糊 | 298 | 88 | 386 |
| LI，目标区域低照度 | 151 | 51 | 202 |
| CM，移动相机拍摄 | 552 | 133 | 685 |
| FL，采集中丢帧属性 | 17 | 20 | 37 |

| 审计输入 | SHA256 |
| --- | --- |
| `trainingsetList.txt` | `ac02d9c820b445b3d7532ed62470ac8f01c09571ff0fb2f1643bdc4fe1f91dcb` |
| `testingsetList.txt` | `16eef326975aacccb94a7e133e8ef0aceb734ece4264caf6f82bc94442c20e46` |
| `Attributes&Order.zip` | `09e951376949bb0996c44ded66f4915621649c3aa1b4b54e5a46e9e682f24235` |

论文公布约 **734.8K 对**，平均约 **600 对/序列**，最长 **12,862 对**，涵盖 32 种目标。VTUAV 的公开基准表另报最短 57 对。逐序列 RGB/T 文件数、有效配对数、空序列数和缺失或损坏帧总数无法从官方轻量清单复核。FL 统计的是序列属性，作者说明某些丢帧时刻仍保留重复或空白 Thermal 文件。Issue #18 有不能解码的可见光文件报告，#23 是 unaligned 版本缺少注释文件报告。这些是待检查的发布线索，当前归档的完整性尚未核实。 [LasHeR 论文 Table I/§III-D](https://arxiv.org/pdf/2104.13202) · [VTUAV 表](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Visible-Thermal_UAV_Tracking_A_Large-Scale_Benchmark_and_New_Baseline_CVPR_2022_paper.pdf) · [Issue #18](https://github.com/BUGPLEASEOUT/LasHeR/issues/18) · [Issue #23](https://github.com/BUGPLEASEOUT/LasHeR/issues/23)

## LasHeR 独立性、昼夜与模糊

- 论文报告超过 20 个室内外场景及两个采集平台。固定杆可转平台使用 DLS-H37DM-A 与 SONY EXView HAD CC，手持平台使用 DLD-J18-161 与 DS-2ZCN3007。官方轻量清单没有提供可连接序列 ID 的站点、原始连续录制、时间戳、重复路线或设备清单。本轮**可证实准入的独立组为 0**，真实组数未知。论文列举场景的方式不足以证明 ≥40 组或证明不可能达到。 [论文 §III-A/D](https://arxiv.org/pdf/2104.13202)
- **昼夜 `INSUFFICIENT_METADATA`。** 作者确认采集覆盖白天和夜间，未在所审清单发布逐序列昼夜标签或带时区的采集时间。202 个 LI 标签描述目标局部低照度。可由独立双审查员复核作者记录及采集上下文。只有影像亮暗或含混室内证据时保留未知。昼间和夜间的真实独立支持均未知，分别达到 15 组尚未证明。 [论文 §III-D 与 Table II](https://arxiv.org/pdf/2104.13202) · [分组方案](../docs/R2k_Q1v2_P2_GROUPING_PROTOCOL.md)
- **合成模糊支持未知。** MB 的 386 个正例是序列级目标运动属性，没有 Thermal 专属、全局模糊、像素范围或逐帧标注。CM 指移动相机，单凭该标签也无法证明相机运动引起模糊。全局 Thermal 模糊和人工受控物理失焦都需独立证据。主终点所需的 15 个正例组来自经核验的完整健康帧、固定局部 Gaussian ROI 和几何支持，当前已核准为 0 组，真实可用数量未知。 [论文 §III-D 与 Table II](https://arxiv.org/pdf/2104.13202) · [冻结评价](../docs/R2k_Q1v2_INDEPENDENT_EVALUATION.md)

## LasHeR 输入兼容性

发布结构说明 visible 和 infrared 为 JPEG。原论文描述等速同步采集与人工尽量对准光轴，随后保持 Thermal 图像不变，用匹配点估计的单应性将 visible 映射至目标附近。论文明确保留径向畸变，目标局部对齐的证据不足以认证全图 ROI 或 RGB 对照。Aligned 与 unaligned 版来自同批采集。原生分辨率、实际输出尺寸、位深、通道数及每个平台的强度处理、AGC 或 radiometric 单位缺少可审核的逐序列资产头与采集记录。JPEG 扩展名不足以证明 Thermal 是单通道 uint8 或能够比较两平台的强度分布。 [官方结构](https://github.com/BUGPLEASEOUT/LasHeR/blob/c7ca5dbc96eed90b2b0b7dccc5014d1e611b8399/README.md) · [论文 §III-A/B/E](https://arxiv.org/pdf/2104.13202)

同行的 VTUAV 原始基准论文报告 LasHeR 为 **630×480**。这为预备独立协议提供依据，尚不能确认指定作者归档的逐文件形状。P0 要求 640×480、原生 8-bit 单通道 Thermal 和全图配准。若官方原始归档核实为 630×480，则为 **`PROTOCOL_CHANGE_REQUIRED`**。v2.1 草案保留原生像素与强度，明确更窄的采样网格，限制工程开发为原 Fit/Calibration 和合成夹具。若验证发现仅目标区域对齐、混合编码或未说明的强度变化，该方案仍不足以准入。 [630×480 比较表](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_Visible-Thermal_UAV_Tracking_A_Large-Scale_Benchmark_and_New_Baseline_CVPR_2022_paper.pdf) · [输入适配草案](../docs/R2k_Q1v2_P2_GROUPING_PROTOCOL.md)

## 授权调查

MMIC 官方目录列出 LasHeR，固定提交 `61f4fdc57e08ddccac296037b46c6dc8d10df6ea` 的 Research-Only Use License 明确覆盖数据/代码的学术研究使用、复制与修改，禁止商业使用，要求署名及引用。按其文字，本地科研合成模糊及本地完整性审计一般落在这些用途内。条款预期研究发表需引用，对汇总统计没有看到单独禁止或专项许可。条款没有明确原图/派生数据留存期、精确归档版本范围、第三方下载附加条件。此次没有取得对具体 aligned 归档和保留安排的书面确认，也未接受条款或下载影像。草稿请求将这些缺口与元数据一次核清。 [官方目录](https://github.com/mmic-lcl/Datasets-and-benchmark-code/blob/61f4fdc57e08ddccac296037b46c6dc8d10df6ea/README.md) · [数据/代码许可证](https://github.com/mmic-lcl/Datasets-and-benchmark-code/blob/61f4fdc57e08ddccac296037b46c6dc8d10df6ea/LICENSE) · [邮件草稿](../docs/R2k_Q1v2_P2_ACCESS_REQUEST.md)

## RGBT234 交叉来源

RGBT234 与 LasHeR 共享发布团队和部分作者。LasHeR 论文明确固定杆平台类似 RGBT234，另增加手持平台并自行采集。作者把 RGBT234 与 LasHeR 当作不同基准，论文没有给出逐序列溯源交叉表，无法判断是否存在完全相同视频、共享采集会话、重复背景或近重复帧。VOT-RGBTIR2019 则明确基于 RGBT234。当前跨源重复和场景关系均 **未核验**。先请作者提供 RGBT234/RGBT210 的录制来源和别名，再在许可范围内比较原始 SHA256 和跨模态近重复并审核路线与背景。重叠风险未消除前，两者的组数不得相加，亦不将 RGBT234 视作独立补充集。 [RGBT234 原论文](https://www.sciencedirect.com/science/article/pii/S0031320319302808) · [LasHeR 论文 §I/III-A](https://arxiv.org/pdf/2104.13202) · [官方发布目录](https://github.com/mmic-lcl/Datasets-and-benchmark-code/blob/61f4fdc57e08ddccac296037b46c6dc8d10df6ea/README.md)

## TartanRGBT 补充候选

- 官方数据卡及 API 声明 **BSD-3-Clause-Clear**。固定发布 `23475c0fa9190154e6f1fd78f6b82b29f9d508a0` 的公开 CSV 实测 **46 条轨迹**，采集批次 day1–day5 分布 7/12/8/16/3，46 个完整轨迹名称不同。标签有同一 NSH 建筑的多楼层与校园多条线路，名称唯一性不能证明独立。《AnyThermal》描述的四类环境亦非四个经核验场景组。无可靠逐轨昼夜列，dayX 是批次名，时间对齐侧文件亦不足以证明拍摄照明。独立组和昼夜支持未知，当前 **`INSUFFICIENT_METADATA`**。CSV SHA256 `c0f53287cd28c7fcef7d88a79ca50e05dae0162a23ba31676fb69ae8bc30b555`。 [官方数据卡](https://huggingface.co/datasets/theairlabcmu/TartanRGBT/blob/23475c0fa9190154e6f1fd78f6b82b29f9d508a0/README.md) · [公开 CSV](https://huggingface.co/datasets/theairlabcmu/TartanRGBT/blob/23475c0fa9190154e6f1fd78f6b82b29f9d508a0/AnyThermal_data_distribution.csv)
- 30 Hz 硬件同步的 ZED X 和 FLIR Boson，处理后数据以 10 Hz 对齐。FLIR 原生/标定规格 **640×512**；官方 RGB-in-Thermal 由 FoundationStereo 深度、外参和热相机内参重投影，带遮挡/视野缺口，并采用 z-buffer 与双线性 splatting。8-bit Thermal 使用 Min-Max、CLAHE、双边滤波，改变与本项目相关的频带和强度。当前卡片的标定段又将 RGB-in-Thermal 与 960×540 ZED 网格并列，文件段和论文则描述投影至 Thermal-left。实际归档输出尺寸、有效区域和独立性须复核。FFC 的 0 表示有效帧，1 为重校准期间应丢帧，标志按 bag time 对齐。FFC 不提供模糊、昼夜或完整健康标签。 [AnyThermal §IV–V](https://arxiv.org/html/2602.06203v1) · [官方数据卡](https://huggingface.co/datasets/theairlabcmu/TartanRGBT/blob/23475c0fa9190154e6f1fd78f6b82b29f9d508a0/README.md)
- 只有通过单独的跨设备输入、分组、覆盖和授权审查后才可作为补充。640×512、投影缺口及强度处理使当前发布不能直接作为 P0 输入。46 个轨迹没有证实达到 40 个独立组、15/15 昼夜及 15 个合成支持组，当前不得单独充当主集。

## 交接

等待用户书面确认可访问的归档版本及小型录制清单。收到授权资料后，仅按分组方案审核文件 SHA256、可用配对、来源交叉、独立分组、昼夜和输入格式，再交独立验收。未确认的问题保持阻塞。P0 的模型、参数、阈值、特征和科学判据均保留。WSL CPU 上只读冻结校验通过，参数 SHA256 为 `fbc78757c2711558623d4c11f7efdc3693c70bb1711e7b5e105de63b161b6e08`。元数据审计核对了二值属性域、清单唯一性和集合相等。
