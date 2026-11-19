# R3-D1A — 真正的 Sigma Mamba RGB-T 行人检测

**合成工程预检通过，正式 LLVIP 训练未启动。** R3-D1 早期设计中的单状态线性递推在本阶段被真正的 Sigma CrossMamba→ConcatMamba 和原生 `selective_scan_cuda_core` 取代。旧实现保留为稿件历史，不再是 D 臂。上一阶段的 [D0 划分](evidence/r3-d0/split.json) 逐字节不变，Fit 9,926、Validation 2,087、Quarantine 12。

| 臂 | 参与的输入 | 融合 |
|---|---|---|
| A | RGB | 单塔，无跨模态输入 |
| B | Thermal 灰度复制 RGB 形状 | 单塔，无跨模态输入 |
| C | RGB + Thermal | 两段 WideCNN 式 1×1 → depthwise 3×3 → GELU → 1×1，保留 Cross 两输出和 Concat 单输出的残差接口 |
| D | RGB + Thermal | 上游 CrossMambaFusionBlock 两输出送入 ConcatMambaFusionBlock；三个尺度各自实例化 |

保留五层卷积塔、80/40/20 三尺度 64 通道、同一个共享 FCOS 头。布局按上游 NCHW→BHWC→NCHW；hidden_dim=64、d_state=4、mlp_ratio=0、默认 ssm_ratio=2、dt_rank=4。未复制上游核心代码。两条包装路径分别为 CrossMamba 的已安装 selective_scan wrapper 和 ConcatMamba 的 vmamba 内部 wrapper，均指向同一个 native CUDA binary。加载前核对 Git commit、已追踪 encoder 与三个源文件，核对已安装 Python wrapper 内容，记录扩展版本和二进制哈希。上游 Sigma 为 MIT；本 PR 引用外部源码。

[指纹和实测 JSON](evidence/r3-d1/gpu.json)记录 Sigma commit `98ba3c6938d29169bbd27f91519084bb462561df`、三个上游源码 SHA256、加载路径、扩展版本 0.0.1 及二进制 SHA256。D 实测每尺度两个 Cross 扫描 `[B,128,HW]` 与一个 Concat 扫描 `[B,256,2HW]`（Concat 分组数 2），并非假定每条路径四方向。batch=1 共 25,804,800 个逻辑状态元素更新。RTX 3090 / WSL2 / PyTorch 2.2.2+cu121（CUDA 12.1、cuDNN 8902）上，三尺度输出均符合 6×80×80、6×40×40、6×20×20；batch 4/2 的损失及全部参数梯度有限，各 Cross/Concat 尺度都有非零梯度。10 次 D 前反向共观测 90 次原生 CUDA backward 调用。没有 optimizer step。

## 配平和资源

MAC 是可审计的语义算子计数。完整模型 Conv/Linear 用实际输出与收缩维度计数；Mamba 的函数式 dt 投影及两个 Concat einsum 单列；scan 按状态递推 FMA、读出乘加及 D skip 计入，exp、softplus、注入乘法、残差/门控、激活和归一化元素等另外列出。它不是 CUDA kernel 指令数量；padding 卷积按理论 dense tap、scan 并行前缀额外操作未算入。候选 CNN 的 cross 和 concat 在每个尺度共享同一内宽，遍历整数 16–512，在满足双预算后按参数差、MAC 差、宽度字典序最小化；最终 80/40/20 网格宽度为 **158 / 370 / 18**。各层通道都进入前向计算。五个种子的公共塔与头的初始张量逐元素验证一致。

| 臂 | 参数量 | MAC/单图 | 模型 median/P95 (ms) | 合成预处理+模型+CPU decode median/P95 (ms) | batch 4/2 梯度预检峰值 allocated (MiB) | 该臂五种子计算估计 (h) |
|---|---:|---:|---:|---:|---:|---:|
| A RGB | 135,062 | 666,777,600 | 0.670 / 2.009 | 32.848 / 37.657 | 210.62 | 5.08 |
| B Thermal | 135,062 | 666,777,600 | 0.704 / 1.745 | 32.501 / 36.605 | 210.62 | 3.38 |
| C CNN | 489,938 | 1,771,118,400 | 2.376 / 4.756 | 42.927 / 48.246 | 567.09 | 3.71 |
| D Sigma Mamba | 490,022 | 1,771,173,888 | 6.358 / 13.759 | 38.935 / 56.654 | 1,036.39 | 4.52 |

C/D 相对参数差 **0.01714%**，MAC 差 **0.00313%**。延迟 FP32 每臂、每模式 20 次预热与 100 次测量，每次同步；含预处理的合成路径采用内存中的随机数组，不含 JPEG I/O。正式报告须补真实开发集数据、延迟与硬件占用；本预检显示即便语义 MAC 相近，模型延迟也不同。

五种子 × 四组 × 十轮，以合成输入实测 batch 4/2 的前向、损失和反向中位时间乘固定批次数，合计 **16.68 小时**。这是短程计算估算，不是正式训练时间；不含 optimizer 更新、真实数据加载、验证/NMS/COCO、ckpt 和硬件竞争。每个参数的两个 Adam 浮点状态缓冲另算（D 约 3.74 MiB，不含临时缓冲/分配保留）。没有训练输出、AP 或权重。

## 复验与闸门

CPU 核对 D0 Fit 010001 与 Validation 020001 数据读取、坐标和输入形状。对官方 test 角色的拒绝只发生在路径打开之前，test 像素读取为 0。冻结 [协议](../docs/R3_D1_DETECTION_PROTOCOL.md)记录五种子、完整 FCOS 训练与推理、ImageNet 归一化、COCO 评测和选模。CPU 合成 COCO 测试覆盖完美、错类、漏检、重复预测、错位及空预测；重复且低分在召回已满之后可不降低插值 AP，不能把这一实例误读成 evaluator 忽略所有重复。测试及 GPU 输出见 [evidence](evidence/r3-d1/)。

```sh
python -m pytest -q -rs tests
python -m compileall -q experiments/r3_d1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python -u -m experiments.r3_d1.preflight --upstream <PINNED_SIGMA> --output reports/evidence/r3-d1/gpu.json
```

未生成 LLVIP 真实图像退化产物，未上传图像或权重。此工程预检不说明检测效果或跨真实场景泛化。正式训练、验证整集和官方 test 都未运行；等待独立审核。

## D1A 独立验收整改：缓存与 GT 覆盖

读取器现在每个 DevelopmentData 实例只打开、SHA256 校验并解析一次 D0 split。后续读样本继续验证 RGB JPEG、Thermal JPEG 和 XML 的文件 SHA256。缓存不跨实例，新建实例会拒绝被修改的清单。Fit 010001 和 Validation 020001 的[真实读取回执](evidence/r3-d1/review/read-receipt.json)保存文件、解码内容与归一化张量摘要、尺寸、dtype 和变换后的全部框。两张原始输入分别为 RGB 1024×1280×3、Thermal 1024×1280，uint8；归一化后均为 float32 3×640×640。Fit 有 6 框，Validation 有 2 框。文件未上传，单测验证 test 角色在文件读取前被拒绝。

独立覆盖命令只打开 Fit/Validation XML（逐份校验），使用与损失同一 FCOS 分配原语的获胜 GT 索引，不以“至少有一个候选点”代替真正获胜。对每个 Fit ID 校验了翻转前后逐 GT 获胜数与候选数一致；Validation 不翻转。测试图像读取 0。完整输出见[覆盖汇总](evidence/r3-d1/review/gt-coverage.json)和[逐图记录](evidence/r3-d1/review/gt-per-image.jsonl)。

| 分区 | 总 GT | 匹配 GT | 未匹配 GT | stride 8 / 16 / 32 正点数 | 零正点图 |
|---|---:|---:|---:|---:|---:|
| Fit | 28,547 | 28,540 | 7 | 90,027 / 150,737 / 27 | 2 |
| Validation | 5,536 | 5,530 | 6 | 5,272 / 41,060 / 0 | 0 |

按变换后 sqrt(area) 分箱，表中为匹配/总量，未显示的区间无支持：

| 区间 | Fit | Validation |
|---|---:|---:|
| [8,16) | 10/10 | 2/3 |
| [16,32) | 217/217 | 19/19 |
| [32,64) | 10,194/10,194 | 594/594 |
| [64,128) | 18,116/18,123 | 4,915/4,920 |
| [128,∞) | 3/3 | 0/0（无支持） |

12 个未匹配框输在重叠或并列竞争，全部位于 [64,128)，例如 Fit 010112 与 Validation 020127。Validation 150032 的一个变换后 3×21.5 px 框没有合格点。未匹配目标的宽、高、sqrt(area) 分位数和全部 ID/坐标见证据。两张零正点 Fit 图 100030/100033 原本无 GT。总体匹配率 99.96%，但极小框验证支持很少，stride 32 验证无正样本，不能据总体数字宣称这些分层可靠。不改变已冻结分配、尺度或数据；若未来小框支持增加后仍存在漏配，可另提跨尺度候选点或更细网格的独立对照。未匹配框全部保留在验证 COCO GT 中。

Windows CPU 全套 157 通过、7 跳过，compileall 退出 0。RTX 3090 按整改后源码重跑合成预检：四臂通过，D 的三尺度 Cross/Concat 扫描反向 90 次；全模型参数与 MAC 逐项与原证据一致，差异仍为 0.01714% 和 0.00313%。新测四臂五种子计算部分约 15.64 小时（同前述范围，不含数据加载、优化器、验证和保存；不是正式训练用时）。旧 GPU 证据保留，新[日志](evidence/r3-d1/review/gpu.log)、[计数与实测](evidence/r3-d1/review/gpu.json)和[校验回执](evidence/r3-d1/review/verification.json)单独归档。未执行 optimizer.step()。

## D1B 执行器与恢复闸门

执行器已实现并通过 CPU 测试；[D1B 独立报告](R3_D1B_TRAINING_RUNNER.md)记录恢复机制和 RTX 3090 复验。A/B/C 通过跨进程逐位等价，D 的 checkpoint 本身能逐位恢复，但固定的原生 Selective Scan 反向重放存在浮点差异，最终训练轨迹不逐位一致。D 正式入口因此阻断。D1A 架构、数据划分和合成前后向证据保留；未进行正式训练。
