# RGB-X × Mamba：科研协作仓库

这是 ChatGPT ↔ Codex 的研究交接仓库，关联上游 [zifuwan/Sigma](https://github.com/zifuwan/Sigma)。[R0](tasks/R0_bootstrap.md) 已通过验收并合并；当前任务为 [R1 — Sigma-Tiny + MFNet 真实基线复现](tasks/R1_sigma_clean_baseline.md)。GPU 兼容性验证已完成，真实 clean baseline 因参考资产缺失仍未运行，见 [R1 报告](reports/R1_REPORT.md) 和 [运行说明](docs/R1_RUNTIME.md)。

执行记录见 [R0 报告](reports/R0_REPORT.md)，模型与许可核查见 [UPSTREAM](docs/UPSTREAM.md)，实验约定见 [Experiment 0](docs/EXPERIMENT_0_PROTOCOL.md)。交接包作者自检保留在 [STARTER_TESTS](reports/STARTER_TESTS.md)，它与本次本机执行证据分开。

## R0 复现

建议 Python 3.13（本机实际 3.13.2）。新建虚拟环境并激活，在仓库根目录执行：

```bash
python -m pip install -r requirements-r0.txt
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pytest -q tests
python experiments/exp0/env_check.py
```

可用 `python tools/run_r0_checks.py` 一次执行并将输出和退出码保存至 `reports/evidence/`。它禁用宿主环境第三方 pytest 插件自动加载，避免无关插件影响结果。R0 测试不需要 GPU、数据或权重；CPU 环境检查成功不表示满足 R1 运行条件。

工作方式：任务 → feature PR → ChatGPT 验收 → PASS 后由 Codex 核对 head SHA 与 CI 再合并。本轮不合并。

禁止提交数据集、权重、密钥和个人路径。上游代码独立 checkout，使用 MIT 许可并保留版权声明；本仓库没有复制上游核心模型实现。
