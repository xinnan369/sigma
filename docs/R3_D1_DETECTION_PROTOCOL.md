# R3-D1A — LLVIP 行人检测正式训练协议（已冻结，未执行）

验收前禁止启动训练。D1A 使用已验收 D0 commit a7a28b519acb173e01aad00b9d55742885ec87bc 和 split SHA256 bd1090cb2ad4696db3ba2f8e984788204c433a7e0cab656073c5e2b67e53e3d4。只准读取 Fit 9,926、Validation 2,087，隔离 12。按 split 清单顺序取 ID，官方 test 禁止读像素、选型和定预算。后续获专门授权才开启官方 test；完整集与剔除历史 64 对的敏感性分别报告，不能视为盲测或跨真实场景验证。

输入：同帧的 RGB 与 Thermal 按 D0 INTER_AREA 缩到 640×512、上下各补 64，补充值 uint8 114。同步水平翻转仅用于 Fit（SHA256 串 seed/epoch/id/flip 首字节 <128）；坐标同变换。Thermal 先 BGR 转灰度，几何变换后复制至 3 通道。转 FP32、除 255，再按 ImageNet mean (.485,.456,.406)、std (.229,.224,.225) 逐通道归一化。单模态模型只接收对应模态，严禁另一模态特征或故障标签。

种子 3407–3411；每个种子每臂独立随机初始化，不用预训练权重，不共享训练过的权重。公共塔和共享 FCOS 头使用 SHA256(seed/模块名) 的前 8 个字节大端整数种子和 Kaiming normal fan_in/ReLU、BN 默认标度和偏置、卷积 bias=0。Sigma 融合按对应模块种子调用上游构造器，保留 Δ/A/D 原始初始化。CNN 使用同一模块命名策略，但保持其卷积初始化。所有臂接受同一 seed/epoch 自动排序及增强。模型、算子和预算见[设计及预检](../reports/R3_D1_DETECTION_DESIGN.md)。

每臂每种子固定 10 Epoch，单设备 FP32、关闭 TF32，batch 4，NumPy PCG64(SeedSequence([seed,epoch])) 对 Fit 遍历一次；每轮 2,481 个四图批次与 1 个两图尾批，尾批按实际图数平均，不跨轮。每批一次 AdamW 更新，恒定 lr=3e−4、betas (.9,.999)、eps 1e−8，Conv2d weight 的 decay=.01，其它参数 decay=0；全部可训练参数均覆盖。不早停。本轮只进行授权预检，不保存或更新正式训练权重。正式训练入口须通过下文独立授权。

FCOS：五层卷积塔 stride 2，通道 16/32/64/64/64；取 stride 8/16/32，三尺度共享同一 3×3 Conv 64→64 + BN + SiLU、1×1 Conv 64→6。每格中心为 (index+.5)×stride，剔除 padding。按变换后 sqrt(area) 分层 [0,64)、[64,128)、[128,∞)；格中心严格落入框及半径 1.5×stride 的中心区，竞争时取最小面积、再取 XML 顺序。无匹配的目标不设置虚假中心；正式报告未匹配 GT 数。回归距离为 stride×softplus(raw)，centerness GT 为 sqrt(min(l,r)/max(l,r) × min(t,b)/max(t,b))。每帧 focal BCE α=.25、γ=2 和 center BCE 除以 max(正样本数,1)，GIoU 按 centerness 加权除以权重和（无正样本则 0）；三项等权，跨实际批图平均。

推理分数 sqrt(sigmoid(person)×sigmoid(center))，按 stride 8/16/32、行列次序打破同分；逆变换、剪裁到 1280×1024，移除零面积、无效预测；原网格 NMS IoU>.65，最多 100 个，未加置信度阈值。pycocotools==2.0.10、单 person、COCO bbox、全部面积、maxDets=[1,10,100]、IoU .50:.05:.95、101 召回点；AP50 和 AP50:95 分别报告。每 Epoch 只用 Validation clean AP50:95 选 checkpoint，AP50 为同分次则，再选最早 Epoch。零检测是合法输入，GT 无支持不作为评分 0。

资源：分别记录完整模型参数量、语义 MAC（非 kernel FLOP），单图推理 median/P95、显存、真实训练耗时。合成预检时延不得写成 LLVIP 实测。后续经授权才可使用 D0 冻结退化协议，并与原始 clean 检测 AP 分开报告。未授权不对真实图像生成退化版本。

### D1A 整改后的读取与覆盖核查

每个 DevelopmentData 实例只在初始化时读取、校验并解析 D0 split；每次读样本仍核验两张 JPEG 与 XML。独立覆盖审计只读开发侧校验后的 XML，按训练同一获胜规则记录每张图与每个尺度的正点数、未匹配 GT 和原因。覆盖不改变标注或实验协议；验证评价仍使用全部有效 GT，不按目标分配结果过滤。官方 test 角色在文件打开前被拒绝。

### D1B 单臂单种子执行与恢复

执行器沿用上述实验规则。每 100 个全局更新和每轮 Validation 完成后，在清空梯度时提交一次。每个臂和种子使用独立仓库外输出及进程锁。日志先 fsync，内容寻址 checkpoint 写入并 fsync 后原子发布 ledger；last/best 通过独立文件投影，若进程停在 ledger 发布之后可恢复旧投影。恢复校验已提交前缀，仅截断其后的未提交字节。未提交更新可在同一游标重放；第一次提交前中断没有可恢复 checkpoint，不自动重置。仅用 Validation 预测和完整原坐标 GT 选模；AP 与预测摘要按轮提交。不得把库中入口或授权参数视作本轮训练批准。

## D1B-R1 确定性后端（现行修订）

原版固定 Sigma CUDA backward 在跨 block 原子累加 dA、dB、dC、dD 和 ddelta_bias 时，浮点累加的顺序不固定。相同输入反向重放出现差异，旧 D 臂逐位恢复闸门因此失败；该结果保留在 [D1B 历史报告](../reports/R3_D1B_TRAINING_RUNNER.md)。

现行 D 臂使用 `sigma-r3-exclusive-block-pairwise-v1`。保持原生 Sigma 前向及前向数学定义不变；CUDA backward 按样本拆分，并为每通道提供独立的 B/C 梯度槽，消除跨 block 对同一槽的竞争。再以固定配对二叉树归约通道和样本维度，不更换模型或冻结参数。旧 D 臂 BLOCKED 条款已由本修订正式取代，不再是现行规则。

只在**原版与确定性版**数值对照中使用 `atol=2e-5`、`rtol=2e-4`。同版本重复反向及跨进程断点恢复仍须逐位一致；不使用容差判定重放或恢复。

RTX 3090 已通过 FP32 三融合尺度和额外网格、batch 4/2 的原生级及整模型数值与重复检查，以及 A/B/C/D 跨进程逐位恢复；见 [R1 数值与恢复报告](../reports/R3_D1B_R1_DETERMINISTIC_SCAN.md)。授权预检还需重跑并绑定当前源码、协议与环境。合成验收通过不构成训练许可：四臂必须同时通过六项闸门，并持有独立验收登记的可信回执后才可启动正式训练。

以下 JSON 为本修订的机器可读验收条款，必须与正文一致：

<!-- BEGIN D1B-R1 ACTIVE CONTRACT -->
```json
{
  "revision": "D1B-R1",
  "status": "ACTIVE",
  "backend": "sigma-r3-exclusive-block-pairwise-v1",
  "supersedes": "D_NATIVE_BACKWARD_BLOCKED",
  "superseded_rule_active": false,
  "forward": "PINNED_NATIVE_UNCHANGED",
  "backward": "EXCLUSIVE_SAMPLE_CHANNEL_SLOTS_FIXED_PAIRWISE_REDUCTION",
  "comparison": {"scope": "NATIVE_VS_DETERMINISTIC_ONLY", "atol": 2e-5, "rtol": 2e-4},
  "repeat_backward": "BITWISE",
  "checkpoint_resume": "BITWISE",
  "formal_training": "INDEPENDENT_AUTHORIZATION_REQUIRED"
}
```
<!-- END D1B-R1 ACTIVE CONTRACT -->
