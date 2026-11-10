# R1e 短程续训协议

## 继承与边界

R1e 从已验收 R1d 的第 177 次更新续训，共至多 5 个 epoch（R1e 新增 epoch 2–5）。保持 R1d seed 3407、FP32、microbatch=1、梯度累积=8、AdamW lr=6e-5、原始预处理、1410 条训练记录及 79 张内部验证图。固定恒定学习率是短程收敛协议，与上游 500 epoch warmup/poly 配置不同。

导入前先校验 R1d 末次 checkpoint 的完整 SHA256，并核对版本 `r1d-epoch-v1`、完整运行签名、epoch 1/游标 1410/更新 177。当前 Python、PyTorch/CUDA/cuDNN、OpenCV、NumPy、设备、CUDA 扩展二进制、R1d 配置与实现哈希须与已验收证据完全一致。保留冻结分割及资产哈希，重算实际训练/验证文件哈希，test.txt 只读取 ID 做隔离核对。加载严格核对模型张量键/形状/dtype/有限性，AdamW 分组配置、每个参数的一二阶矩和步数，调度器计数，Python/NumPy/torch CPU/CUDA RNG 和 sampler 的排列、生成器与游标。原始 R1d 文件只读。R1e 产物使用新目录和独立版本。

每 epoch 重新生成排列，每条训练记录恰好一次。每轮 176×8+2 个微步，177 次 AdamW 更新。尾窗使用其两条记录实际有效像素总数，padding 255 排除。类别 0 计入 CE 和总体 mIoU。每轮结束仅验证冻结的 79 图，汇总九类混淆矩阵，统计前景 mIoU、逐类 Precision、Recall、IoU 及预测占比。前景均值排除 0，指标分母为零时记 null。

## 保存与停点

每轮第 32、64、96、128、160 次更新及验证后的第 177 次更新保存 last.pt。文件先写临时文件并 fsync，再同目录原子替换。保存包含模型、优化器、调度器、全局 RNG、Sampler、轮内统计及验证历史。模型在验证前如果中断，从最近窗口恢复。best.pt 是独立保存的完整状态，依据最高内部总体 mIoU，持平保留最早轮次，Epoch 1 的已验收结果亦参与比较。各轮 checkpoint SHA256 和当前 best 指针写入日志，模型文件留在本地。

第一次执行必须止于 Epoch 3。继续前重建两轮完整采样序列，核对训练/验证数据摘要、尾窗、学习率、梯度有限性、混淆矩阵、last/best 及恢复记录。阈值在跑前固化于配置和任务文件。Epoch 3 总体和前景 mIoU 允许分别较前两轮最高值下降至多 3 个百分点，验证 CE 允许较最低值升高 max(0.02, 20%)，类别 0 预测占比必须低于 98%。此门槛用于本次安全续跑，不是显著性检验。

## 执行命令

在 WSL 的 bridge 根目录设置 `CUDA_VISIBLE_DEVICES=0`，使用 R1 已核验的 CUDA venv。先运行 pytest（真实模型覆盖检查需配置 R1d 资产环境变量）。

```sh
python -u -m experiments.r1e.train \
  --upstream <UPSTREAM> --init <VMAMBA_T> --root <ASSETS>/MFNet \
  --checkpoint-dir <NEW_EXTERNAL_RUN>/checkpoints \
  --parent <R1D_EXTERNAL_RUN>/checkpoints/last.pt --through 3 \
  --output reports/evidence/r1e/phase3

python -m experiments.r1e.audit --root reports/evidence/r1e \
  --checkpoint-dir <NEW_EXTERNAL_RUN>/checkpoints --through 3
```

通过停点后，使用相同路径重新启动训练命令，替换 `--parent … --through 3 --output …/phase3` 为 `--resume --through 5 --output reports/evidence/r1e/phase5`。完成后重跑 audit 命令，将 `--through` 改为 5。断点重试使用新的日志目录。正式实验两段日志路径用于复核。

场景风险用 `python -m experiments.r1e.scene_audit --root <ASSETS>/MFNet --output reports/evidence/r1e/scene-audit.json` 检查。只使用 705 个训练原图和 79 个验证原图，匹配同 D/N 后缀的邻近数字 ID 并筛查双模态画面相关性，训练的翻转方向也纳入比较。候选关系用于披露划分限制，保留已冻结划分与所有训练参数。
