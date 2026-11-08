# R1b 执行报告

启动 2026-11-04，续行交付 2026-11-05（Asia/Shanghai）。**完整合成前向、真实测试集获取与严格审计已完成；官方分割 checkpoint 未恢复，clean mIoU 仍为 NOT RUN。** 不合并 PR。

## 来源、分支与环境

固定上游 `98ba3c6938d29169bbd27f91519084bb462561df`，未修改模型与评估源码。沿用 R1 的 Ubuntu 22.04/WSL2、RTX 3090 24GB、driver 610.88、torch 2.2.2+cu121、CUDA toolkit 12.6、timm 0.4.12、einops 0.8.1、OpenCV 4.9.0。环境盘点见 evidence/r1b/environment.txt。

R1 PR #2 仍 OPEN；R1b 分支从其 head `bb45419670f7385c6aca590878b741b8ca9c1879` 创建，PR base 使用 R1 分支，避免混淆新增变更与尚未验收的 R1 内容。R1b 最终 head 在 PR 描述及交付中提供，不嵌入报告自身 SHA。

## 完整模型前向：PASS

命令：`python -m experiments.r1.model_smoke --upstream <UPSTREAM>`，退出 0。[完整日志](evidence/r1b/model-smoke.txt)。

| 项目 | 实测 |
|---|---|
| 输入 RGB / Thermal | 各 1×3×480×640，FP32；Thermal 三通道由同一合成通道复制 |
| 输出 | 1×9×480×640，全部有限 |
| 参数量 | 48,287,360 |
| 完整路径 | CrossMamba 四尺度、ConcatMamba 四尺度、MambaDecoder 四级上采样及输出卷积均执行，每个目标模块 4 次调用（1 warmup + 3 measured） |
| seed / 模式 | 3407，eval + inference_mode，TF32 关闭，未启用 AMP |
| CUDA event 耗时 | 159.054 / 160.507 / 159.074 ms，中位 159.074 ms |
| warmup / 3 次循环墙钟 | 9.213s / 0.479s；不含数据读取/预处理 |
| peak allocated / reserved | 604,900,352 / 889,192,448 bytes（约 577 / 848 MiB） |
| 异常 | 完整前向无异常；构造器尝试读取 LFS 指针初始化失败的原始警告已保留 |

这是**未加载官方分割权重的合成 smoke**，不报告其像素类别或 IoU，不将其称作作者模型精度。MambaDecoder 在上游被直接调用 forward，故记录其内部 layers_up 和输出卷积，以确认完整路径；这些 shape 不是 SSM 隐状态。

## 393 个样本的真实数据恢复：PASS

从 Sigma 作者提供的 MFNet 三目录公开 embeddedfolderview 列出文件，仅按既定 test.txt 的 393 个 ID 取原始 PNG，没有下载 `_flip` 测试替代品、重排/修改清单或重映射标签。获取 1179 个文件，下载命令退出 0；原始图像和 test.txt 全部存于仓库外，不上传。

首轮严格审计发现 `RGB/01597D.png` 被截断（223,316 bytes），没有忽略错误或开启容忍截断模式。按同一作者 file ID 重下载得到 449,595 bytes，严格解码成功；新旧 hash 与来源记录在 [image-recheck.json](evidence/r1b/image-recheck.json)。下载器已补充完整 PNG 验证，并新增截断/HTML 回归检查。

最终 OpenCV 4.9 + PIL 全量审计：**1179/1179 有效，errors=[]**，每个文件均 640×480、uint8、PNG；RGB mode RGB 393，Modal mode L 393，Label mode L 393。PIL 标签数组与官方 cv2 grayscale 读法一致，全部标签值在 0..8，ignore=255 像素为 0；没有改变类别映射。

- test.txt SHA256：`a236aa6ac7e55ec6b5e419b55c6934439611453939eee2c0bf7b0f0c34917c9d`。
- 按清单顺序记录的每文件路径、SHA、字节数、shape、dtype 和标签值集合见 [data-manifest.json](evidence/r1b/data-manifest.json)。文件记录 canonical JSON 的聚合 SHA256：`8230b56c0b3ae7b1999be317feb973d1cc8b599fd811f82d0e3c2c09f2c1429c`；这是明确序列化定义的 manifest digest，不冒充某个压缩包 hash。
- 类别 0..8 像素计数依次为 112023545、4854504、1345972、1261567、602620、355314、17794、85246、183038，总计 393×480×640。
- 命令 `python -m experiments.r1.audit_mfnet --root <ASSETS>/MFNet --manifest reports/evidence/r1b/data-manifest.json` 最终退出 0；首次失败与最终成功日志均保留。

## checkpoint 恢复：分割权重仍阻塞

检查当前作者仓库 README、作者 Sigma-anonymous 仓库、官方项目页、GitHub releases、issues #8/#26/#43/#47/#48/#49/#50、作者名下 Hugging Face 模型列表。当前正式 MFNet 分割文件查看页返回 404；作者在 issues 给出的备用初始化文件夹也 404；GitHub releases 无可用资产，检查的作者 HF 列表为空，项目页没有新增 Drive 模型链接。未采用第三方重训/其他模型权重。

证据：[checkpoint-recovery.json](evidence/r1b/checkpoint-recovery.json)、[公开讨论摘要](evidence/r1b/author-issues.json)。这些检查仅说明本次正式渠道未找到可获取文件，不声称互联网上绝不存在备份。可联系作者 `zifuw@andrew.cmu.edu`，或在 [作者仓库 issues](https://github.com/zifuwan/Sigma/issues) 提交带 commit/文件 ID 的恢复请求；本轮没有发送邮件或发布留言。

成功恢复的是**官方 VMamba-T 初始化**：从固定 commit 的 GitHub LFS 原始对象下载 91,649,482 bytes，SHA `dbc0cc4f5ec0e45db5fba7c939d2c7d9b617e891ac10766912a8d604c37c5e47`，与 pinned LFS oid 完全一致，见 vmamba-init.json。该对象不等于 Sigma-Tiny/MFNet 分割 checkpoint；未用它运行所谓 clean baseline。

## 训练可行性与实际失败

用上述真实初始化文件在临时运行目录满足原始构造器相对路径，执行了合成 forward/backward/AdamW 资源探针（非 MFNet 训练，不保存模型，不报告训练精度），日志保留了完整初始化加载信息和 optimizer 遗漏名称。

初始化加载采用上游 strict=False：missing 为 outnorm0..3 的 weight/bias（8 项），unexpected 为 classifier norm/head 的 weight/bias（4 项）。这属于分类预训练到 backbone 的部分加载，不是完整分割 state_dict 严格加载成功。

- microbatch=1：有限 loss 的单个合成更新通过，16.593s（无 warmup，不是稳定吞吐），peak allocated 6,731,221,504 bytes，reserved 6,958,350,336 bytes。
- batch=8：**OOM**，并引发 Windows/WSL 主机内存不足。PyTorch 记录约 52.8GB allocator peak，属于共享内存/超额分配压力下的计数，不代表 24GB 物理显存容量。探针退出码 0 只表示至少 batch=1 完成，不能解释为 batch=8 通过。续行核实探针已退出、GPU 资源释放；没有重跑大 batch。
- 上游 group_weight 遗漏 6,407,808 个参数；这对重训练正确性有影响，不能静默修正后仍称原始作者训练。
- 后续探针默认仅 batch=1，大 batch 要显式指定，且每步即时打印结果，避免异常时丢失已完成证据。

详细的训练配置、train/train2 区分、单卡设备硬编码与 test-set checkpoint selection 风险、microbatch/累积候选、资源预算和计时公式见 [独立重训练方案](../docs/R1b_RETRAIN_PLAN.md)。本轮没有启动真实训练，也不将合成更新视为训练结果。

## 官方 clean baseline：NOT RUN

九类（unlabeled、car、person、bike、curve、car stop、guardrail、color cone、bump）的 IoU，以及包含类别 0 的总体 mIoU，全部 **NOT RUN**。原因现在仅剩官方分割 checkpoint 未恢复；数据已完整审计。官方分割模型的严格 state_dict 加载、真实评估显存/时间均 NOT RUN。

资产恢复后原样执行 `CUDA_VISIBLE_DEVICES=0 python eval.py -d 0 -n mfnet -e <OFFICIAL_CHECKPOINT>`，保持 393 清单、gt_transform=False、原 normalization、单尺度 [1]、无 flip、crop 480×640、stride 2/3、全数据集混淆矩阵与类别 0 参与计分。完整加载日志、9 类 IoU、时间和 peak memory 必须再补实测；本轮不能用前向 smoke 的时间/显存代替官方数据评估。

## 其他验证与提交边界

31 项 bridge 测试通过（退出 0），包括截断 PNG 回归。GPU 完整前向、数据下载/最终审计均退出 0；训练 batch=8 失败按原日志保留。日志和 manifest 位于 evidence/r1b。只提交代码、元数据、审计结果与日志，不提交 PNG、原始清单、权重、二进制或密钥。PR 保持未合并，等待 ChatGPT 验收。
