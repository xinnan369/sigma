# R3-D1B 正式训练执行器与断点恢复

基线：`ba0c399f865931473c22dc61af63f2d6b1bf4c3f`（PR #30）。本分支只新增执行与审计，未改 Detector、冻结 plan、FCOS loss 或 D0 划分。**四臂逐位恢复总闸门未通过，D 臂 BLOCKED，正式训练未执行。**

| 项目 | 实际结果 |
|---|---|
| 执行器与 CPU | 170 通过、7 跳过；compileall 退出 0 |
| 合成恢复 RTX 3090 | A/B/C 逐位一致；D 不一致，GPU 审计退出 1 |
| 真实读取 | Fit 010001（6 框）与 Validation 020001（2 框）通过，均只读 |
| 真实 LLVIP 更新 / 官方 test 像素读取 | 0 / 0 |
| 四臂五种子正式训练及完整真实 Validation | NOT RUN |

## 执行与事务

`train.py` 只执行一个 seed/arm，经显式授权才可进入；读取冻结 DevelopmentData、重用原 plan 函数、同步逐批读取。每轮 2,481 个四图批及一个两图尾批，不重复不遗漏，2,482 次更新；总计划 200 Epoch / 496,400 次更新。使用全部有效 Validation GT、D1A decode 和 pycocotools 2.0.10，固定 AP50:95、AP50、最早轮次比较。日志记 ID、翻转、游标、损失、梯度、LR、耗时与显存。每轮记录 AP 和原网格预测 SHA256。

`checkpoint.py` 用锁、fsync 和原子替换保存 Model/BN、AdamW、四种 RNG、ID 批次顺序及摘要、游标、历史、best、环境及本地执行/上游/Split 签名。每 100 更新与每轮结束提交。仓库外各 arm/seed 保留 initialization.json、runtime.json、events.jsonl、epochs.json、checkpoint-ledger.json、resume-audit.json、last.pt 和 best.pt。日志字节前缀和 checkpoint 的 SHA256 都在 ledger 中；不匹配即停，未提交日志尾可截断，验证前的末批游标可直接重做 Validation。原子 ledger 是权威，独立 best/last 投影可在中断后恢复；checkpoint 仅从本机自己生成的可信输出读取。没有将任何运行权重保存到仓库。

## 合成恢复与 D 臂定位

RTX 3090，WSL2，PyTorch 2.2.2+cu121、CUDA 12.1，单 GPU，关闭 TF32 并启用 PyTorch 确定性。每臂 3 个独立进程（连续 2 步、1 步后退出、恢复后第 2 步）；合成输入同时消耗 Python、NumPy、PyTorch CPU 和 CUDA RNG。A/B/C 最终参数、BN、AdamW、游标、所有 RNG 与已提交事件前缀逐位或逐字节一致。D 在恢复加载后、第二步之前，对刚加载 checkpoint 内模型、AdamW 与 RNG 的逐位校验都通过；第二步结束后参数、BN、AdamW 与事件摘要不同，游标和 RNG 一致。D 两条轨迹各计 18 次原生 fwd、18 次训练 bwd，诊断额外重放不计入训练调用。

原生二进制 SHA256 与 D1A 一致（详见 JSON）。对最先触发的 D 原生反向使用相同保存输入及上游梯度，原调用基准加三次重放；形状 [4,256,800]（20×20 Concat 的两组扫描）。在首次成功完成定位的重放（native-probe-first.json）中，最多有 22,083 个 dB 元素和 22,399 个 dC 元素不同，dD 最大绝对差约 2.33e−10；dA、ddelta_bias 也存在差异。上游固定提交中 [backward kernel](https://github.com/zifuwan/Sigma/blob/98ba3c6938d29169bbd27f91519084bb462561df/models/encoders/selective_scan/csrc/selective_scan/selective_scan_bwd_kernel.cuh) 使用共享梯度的 gpuAtomicAdd，重放行为与浮点累加顺序不固定一致。没有更换扩展或用容差掩盖非确定性。下一步需独立审批确定性实现或修改验收目标；本阶段不启动训练。

```sh
python -m pytest -q -rs tests
python -m compileall -q experiments/r3_d1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python -m experiments.r3_d1.resume_audit --upstream <PINNED_SIGMA> --output reports/evidence/r3-d1b/resume-audit.json
python -m experiments.r3_d1.coverage read --root <LOCAL_LLVIP> --split reports/evidence/r3-d0/split.json --output reports/evidence/r3-d1b
```

[审计证据](evidence/r3-d1b/)包含初次失败、诊断工具比较类型的修正前日志、正式诊断复验、真实读取回执和 SHA256；不含图像、模型权重或正式训练日志。CPU 的 D 优化器分组测试用结构替身只测分组，实际 D 参数覆盖由 RTX 3090 的 AdamW 状态核查。
