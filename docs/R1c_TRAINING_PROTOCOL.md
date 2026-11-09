# R1c 独立重新训练 pilot 协议

本配置验证训练驱动和资源，不是作者 checkpoint 复现，不是完整训练结果。固定上游 `98ba3c6938d29169bbd27f91519084bb462561df` 的模型源码保持不变，按其 MIT 许可从外部 checkout 导入。

## 数据隔离

作者渠道为 [Sigma README](https://github.com/zifuwan/Sigma/blob/98ba3c6938d29169bbd27f91519084bb462561df/README.md) 中的 [MFNet Drive](https://drive.google.com/drive/folders/1bD5b4xXkDBasOLhupUBDdw2rq9oogbu_)。作者在 [issue #37](https://github.com/zifuwan/Sigma/issues/37) 明确建议 train.txt；固定源码配置使用 train2.txt，二者不是相同训练 regime。

- train.txt：1568 条，784 原图及其784个 `_flip`；train2.txt：1176 原图。审计两表并集1960个样本/5880文件。
- Pilot 从 train.txt 的784个原图身份分组；day/night各按 `SHA256(UTF-8("3407:"+id))` 排序，前 ceil(10%) 原图作验证，共79个；其flip不纳入训练。余705原图及flip为1410训练条目。
- 清单与规则在训练前冻结，不按图像标签/模型表现选样本。读取 test.txt 及历史测试文件 hash 仅用于防交叉审计，训练/验证代码不打开测试图像。
- 不能保证视频相邻帧/相同场景独立；缺乏可靠场景分组 metadata 时只声称原图身份及精确文件对的隔离。后续论文实验应审查场景相关性。

## 模型、分组与梯度

VMamba-T 初始化 SHA256 `dbc0cc4f5ec0e45db5fba7c939d2c7d9b617e891ac10766912a8d604c37c5e47`；构造后再次 fail-closed 加载并核对8个outnorm missing和4个classifier unexpected键，避免上游吞掉异常后静默随机初始化。

原 group_weight 使用 modules() 类型枚举，直接注册的 nn.Parameter 不会作为 Module 返回。保留原有参数的 decay/no-decay 归属，再把遗漏参数按 `_no_weight_decay`、一维参数或 bias 后缀放入无衰减组，其余 decay=.01。按对象身份断言全部 trainable 参数恰好一次；逐名称记录分组、形状、numel。该修正是独立训练协议变化。

CUDA_VISIBLE_DEVICES=0，进程内仅 cuda:0。FP32，不开启AMP/TF32；microbatch=1，8次累积。每次 mean CE 乘本微批有效像素数后反传，累积结束将梯度除以窗口总有效像素数；padding 255不参与，类别0有效。这匹配相同logits下整体批量有效像素平均CE，但随机drop-path序列和归一化行为不保证与物理batch8相同。

每次更新记录所有参数张量的grad存在性/有限性、L2、非零元素数及实际变更元素数。不得把“张量有梯度”误写为“每个标量每步都非零更新”。遇缺失或非有限梯度即停止并记录。

## 预处理和验证

导入原 TrainPre：随机mirror、scale=[.5,.75,1,1.25,1.5,1.75]、原crop/pad480×640、原均值方差；保留作者预生成flip条目再在线mirror这一行为。Python3.11将collections.Iterable兼容别名指向collections.abc.Iterable，不更改算法。

原 RGBXDataset 把 cv2.COLOR_BGR2RGB 常量传给 imread，数值4实际仍产生BGR通道；本驱动沿用该原始读法，不静默改色序。灰度thermal复制3通道，标签原值0..8/255，不重映射。

内部验证单尺度、整幅480×640、无flip/增强，按原normalize；全验证集汇总9×9混淆矩阵，包含类别0，忽略255，union为0的类别记录null并从均值排除。验证结果仅代表这79个训练内部holdout，不能称官方393 test clean mIoU。原eval.py保持未修改。

## 有界训练与恢复

配置 [R1c_pilot.json](../configs/R1c_pilot.json)：AdamW lr6e-5、betas(.9,.999)、wd.01；13次更新/104微步，验证在4/8/12/13步，按最大内部验证mIoU选best，同分保留先前者。短pilot采用恒定lr；不套用500 epoch的warmup/poly曲线，不通过pilot调参。

先在第8次更新完成后停止，独立进程从last.pt恢复到第13步。每个update边界zero_grad之后原子写last；改善时写best。保存完整model/optimizer、update/恒定lr调度状态、best指标及步数、Python/NumPy/torch CPU/所有CUDA RNG、sampler permutation/cursor/epoch/generator、配置及代码/数据哈希。num_workers=0，无预取状态；AMP未启用、scaler=None。临时文件fsync后同目录replace；崩溃只损失当前未提交窗口，从上一完整update重跑。不支持窗口中途checkpoint。

加载严格检查signature和完整模型键，恢复optimizer及采样器后最后恢复RNG。仅加载本地可信checkpoint；不接受未知pickle。CPU测试要求恢复后参数/优化器逐位相同；自定义CUDA kernel不预先声称跨运行逐位确定性。

## 执行方式

Windows下载：`python -m experiments.r1c.data --root <ASSETS>/MFNet --output reports/evidence/r1c --download --download-only --workers 16`。网络中断后可用同一次保存的公开source-index加`--reuse-index`继续，已有PNG须严格验证。Linux审计：同模块同root/output，不加download。

在bridge根目录的WSL终端，设置`CUDA_VISIBLE_DEVICES=0`，使用既有R1 venv：

```sh
python -u -m experiments.r1c.pilot \
  --upstream <UPSTREAM> --init <ASSETS>/pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth \
  --root <ASSETS>/MFNet --config configs/R1c_pilot.json \
  --split reports/evidence/r1c/split.json --manifest reports/evidence/r1c/manifest.json \
  --output reports/evidence/r1c/pilot --checkpoints <EXTERNAL_RUN>/checkpoints --stop-after 8
```

恢复时使用相同参数，删除`--stop-after 8`，添加`--resume <EXTERNAL_RUN>/checkpoints/last.pt`。实际执行日志记录解析后的命令、版本、UTC、退出码；checkpoint只留本地，不上传。

完整训练仍须独立审查：确认train regime/holdout、固定预算与warmup/poly、验证频率、保存策略、多种随机种子、最终test一次性报告计划，再讨论500 epoch；当前驱动硬限制max_updates≤20，不能直接提交500 epoch作业。
