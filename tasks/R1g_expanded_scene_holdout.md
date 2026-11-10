# R1g — 扩大场景独立验证与统计可靠性

已验收 R1f PR #7，Head `b31dda76c57c6cd6ff8193fa14298d7e95059658`，核对 CI 及无冲突后合入 main `d53b8f414a34e08e693ec8858b1371a7166142be`。R1g 分支 `codex/R1g-expanded-scene-holdout` 基于该提交。

1. 首先单独执行严格 MILP 可行性检查。沿用 R1f 场景图 `eeef7f60543b90410e81c04cdd4e1caf03cf4594b051e24a4def4f93338a8baf`。锁定原 79 张所在的组，新增整组 78 张。157 张验证，627 原图及对应 flip 共 1254 条训练记录。两侧每类 ≥6 张原图且 ≥6 个组；验证昼夜各 ≥50。不可行即停止，原门槛保留。
2. 可行后独立核对无原图、flip、组、边、test ID 交叉，冻结 split、图和选定文件 SHA256。启动前再次核验真实文件字节。标签仅参与上述支持约束，选择过程不接收模型预测或性能。
3. 从官方 VMamba-T 分类初始化独立训练 5 轮。固定上游、seed 3407、FP32、microbatch 1、累积 8、AdamW 恒定 LR 6e-5、原预处理、CE 和 668 参数张量分组。每轮 156×8+6 条，尾窗按实际有效像素归一化，共 785 次更新。保存独立 best/last。
4. 每轮记录 157 图的总体、前景、逐类指标，D/N 和保留 79/新增 78 各自的指标，逐图及逐视觉组混淆矩阵。Epoch 5 预先固定为主要报告。10,000 次视觉组级 percentile bootstrap，seed 3407，报告总体、前景与逐类 IoU 的 95% 区间。选择的 best Epoch 仅作次要描述。
5. 对照 R1f 时同时记录训练/验证构成和组数变化。新增 78 张曾在 R1f 训练集，保留 79 张已被重复观察。此划分是视觉场景代理，不能当作时间独立或盲测集。官方 393 图不选模，完整 500 epoch 不启动，数据与权重留在本地。

实际命令和退出码保存在 `reports/evidence/r1g/`。先依次运行 `python -m experiments.r1g.partition feasibility` 与 `python -m experiments.r1g.partition certify`，再运行通过测试的 `experiments.r1g.train`、`experiments.r1g.audit_run` 和 `experiments.r1g.statistics`。R1g PR 等待验收。
