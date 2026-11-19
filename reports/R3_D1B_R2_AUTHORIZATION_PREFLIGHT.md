# R3-D1B-R2 — 正式训练授权候选预检

基线 PR #32 `e1d0d462f84791f0b888fdae8dab25629b2aee9f`。只产生候选证据，不签发训练许可；`review_registry.py` 不变，`TRUSTED_RECEIPTS=frozenset()`。任何实验臂均不能凭候选开训。

| 分区 | 配对 | JPEG | XML | 有效 GT | 匹配 GT | 未匹配 GT | stride 8 / 16 / 32 正点 |
|---|---:|---:|---:|---:|---:|---:|---|
| Fit | 9,926 | 19,852 | 9,926 | 28,547 | 28,540 | 7 | 90,027 / 150,737 / 27 |
| Validation | 2,087 | 4,174 | 2,087 | 5,536 | 5,530 | 6 | 5,272 / 41,060 / 0 |

使用正式读取器校验开发侧每对 RGB、Thermal 及 VOC XML 的冻结 SHA256，解码两模态并验证原网格、uint8 和配对，解析全部框。覆盖流程独立重读开发侧 XML，重放原 FCOS 分配（两张零正点 Fit 图），结果与 D1A 一致。所有有效 GT 保留。没有读取官方 Test 或 Quarantine；没有真实图像前向或优化器更新。

RTX 3090 四臂合成跨进程恢复均逐位通过。D 连续轨迹为 18 次原生 forward、72 次原生 backward 子调用，恢复的两个进程各 9/36。候选生成时逐项比较恢复证据的执行环境、真实加载上游及源码哈希。用正式入口的签名函数重算四臂 × 五种子（3407–3411）共 20 条签名；每条绑定 FP32、10 Epoch、batch 4、2,482 更新/轮、冻结模型/损失、AdamW、split、协议、确定性后端与 CUDA 扩展。重算每种子十轮的批序摘要共 50 条，同种子四臂共享批序。

`authorization-candidate.json` 沿用 `r3-d1b-r1-authorization-v1` schema，标注 `CANDIDATE_NOT_TRUSTED`，六项技术闸门各有证据路径和 SHA256。生成后四臂授权校验全部拒绝候选。候选只适用于记录的执行路径、环境与工件；可信登记须由独立验收处理。

Windows 完整 CPU **189 通过、7 跳过**、compileall 退出 0；RTX 3090 合成恢复退出 0。正式训练、真实 LLVIP optimizer update、官方 Test 读取均为零。[证据及日志](evidence/r3-d1b-r2/)。

```sh
python -m pytest -q -rs tests
python -m compileall -q experiments/r3_d1
python -m experiments.r3_d1.authorization_preflight data --root <LOCAL_LLVIP> --output <NEW_EVIDENCE_DIR>
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m experiments.r3_d1.resume_audit --backend deterministic --upstream <PINNED_SIGMA> --output <NEW_EVIDENCE_DIR>/resume.json
python -m experiments.r3_d1.authorization_preflight issue --upstream <PINNED_SIGMA> --output <NEW_EVIDENCE_DIR>
```
