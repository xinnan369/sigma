# R2i-S2 — Seed 3408

执行 Head b10edba391808bb028633370c8f35430e8c844a1。S1 PR #18 合并提交为 3e7e8864cfc9db25e946695fc9efc60872bda957。S1 输入仅用于只读复算。S2 训练只运行 3408。

## 训练与独立审计

RTX 3090，FP32。6 组各 5 轮、785 次更新通过。六样本尾窗按有效像素归一化。3408通过独立进程 GPU 恢复检查。两套统计路径核验冻结 GT、ROI、逐图预测哈希、checkpoint、配对计数和区间。官方 test 像素读取量为 0。

## 两种子 Epoch 5

各列 mIoU 以百分比报告。

| Seed | 轨道 | 组 | Clean | 前景 | 中心 | 随机 | Gaussian |
|---|---|---|---:|---:|---:|---:|---:|
| 3407 | clean | baseline | 60.0137 | 55.3277 | 59.6755 | 59.7028 | 59.8152 |
| 3407 | clean | gated | 60.3735 | 55.7225 | 60.0818 | 60.1099 | 60.1836 |
| 3407 | clean | control | 60.9633 | 56.3872 | 60.6940 | 60.6817 | 60.7844 |
| 3407 | augmented | baseline | 60.0562 | 55.3676 | 59.9003 | 59.9073 | 59.8409 |
| 3407 | augmented | gated | 59.8479 | 55.1344 | 59.7462 | 59.7456 | 59.6820 |
| 3407 | augmented | control | 59.5015 | 54.7482 | 59.3135 | 59.3623 | 59.2265 |
| 3408 | clean | baseline | 64.0592 | 59.8730 | 63.7959 | 63.8668 | 63.9058 |
| 3408 | clean | gated | 62.6948 | 58.3455 | 62.4585 | 62.5061 | 62.4730 |
| 3408 | clean | control | 62.1718 | 57.7567 | 61.9638 | 62.0190 | 62.0312 |
| 3408 | augmented | baseline | 61.9539 | 57.5201 | 61.5896 | 61.7660 | 61.6209 |
| 3408 | augmented | gated | 61.0075 | 56.4575 | 60.6843 | 60.8572 | 60.7126 |
| 3408 | augmented | control | 61.1567 | 56.6283 | 60.8016 | 60.9637 | 60.8896 |

## 逐种子中心遮挡比较

净错误率差为候选减参考。负值表示错误更少。区间是单种子视觉组配对 95% CI，率的单位为百分点。新错及纠正列为像素数。

| Seed | 轨道 | 比较 | 新错 | 纠正 | 净错误率差及区间 | Clean 总体/前景非劣性 |
|---|---|---|---:|---:|---:|---|
| 3407 | clean | gated-minus-baseline | 84608 | 119769 | -0.07389 [-0.17379, -0.00407] | True / True |
| 3407 | clean | gated-minus-control | 109163 | 104642 | 0.00950 [-0.07916, 0.08879] | False / False |
| 3407 | clean | control-minus-baseline | 64026 | 103708 | -0.08339 [-0.17226, -0.02105] | True / True |
| 3407 | augmented | gated-minus-baseline | 48556 | 51087 | -0.00532 [-0.04871, 0.02379] | False / False |
| 3407 | augmented | gated-minus-control | 38939 | 56631 | -0.03718 [-0.05995, -0.01513] | True / True |
| 3407 | augmented | control-minus-baseline | 59284 | 44123 | 0.03186 [-0.02064, 0.07380] | False / False |
| 3408 | clean | gated-minus-baseline | 139961 | 114043 | 0.05446 [-0.08949, 0.15452] | False / False |
| 3408 | clean | gated-minus-control | 142431 | 152055 | -0.02022 [-0.18136, 0.09356] | False / False |
| 3408 | clean | control-minus-baseline | 132290 | 96748 | 0.07469 [-0.00627, 0.16958] | False / False |
| 3408 | augmented | gated-minus-baseline | 98673 | 87370 | 0.02375 [-0.05234, 0.08170] | False / False |
| 3408 | augmented | gated-minus-control | 77399 | 92342 | -0.03140 [-0.08464, 0.01286] | False / False |
| 3408 | augmented | control-minus-baseline | 87985 | 61739 | 0.05515 [-0.01140, 0.12419] | False / False |

## 两种子累计

同一轮重采样为各架构、轨道和条件共享视觉组权重与种子权重。10,000 次重采样计算等种子权重的平均效应。两个训练种子限制区间精度。

| 轨道 | 比较 | 两种子均值及 95% CI (pp) | 种子 SD (pp) | 种子方向 |
|---|---|---:|---:|---|
| clean | gated-minus-baseline | -0.00971 [-0.14054, 0.12478] | 0.09076 | negative, positive |
| clean | gated-minus-control | -0.00536 [-0.12953, 0.07762] | 0.02102 | positive, negative |
| clean | control-minus-baseline | -0.00435 [-0.13769, 0.13595] | 0.11178 | negative, positive |
| augmented | gated-minus-baseline | 0.00922 [-0.04937, 0.06287] | 0.02056 | negative, positive |
| augmented | gated-minus-control | -0.03429 [-0.07043, 0.00085] | 0.00408 | negative, negative |
| augmented | control-minus-baseline | 0.04351 [-0.01483, 0.10412] | 0.01647 | positive, positive |

## 研究判断

- clean 轨道的两种子筛查 H1=False、H2=False、H3=False，对两种对照的全部种子方向一致=False。
- augmented 轨道的两种子筛查 H1=False、H2=False、H3=False，对两种对照的全部种子方向一致=False。

两轨未同时显示相对原始 SSM 和等容量 RGB-only 对照的一致优势并通过 Clean 非劣性。当前证据不足以把收益稳定归因于 Thermal 可靠性信息，建议暂缓追加种子并交独立验收。

差分中差分为探索指标，完整新错、纠正、净错、差分区间及九类 IoU 随报告归档。内部验证集已被历史研究使用，视觉代理组独立性仍有限，次要区间未做多重校正。

## 资源

| Seed | 训练单元 | 训练秒 | 峰值 GiB | 推理 ms | 参数量 | 附加门控标量操作 |
|---|---|---:|---:|---:|---:|---:|
| 3408 | clean-baseline | 2824.0 | 6.85 | 151.47 | 48287360 | 0 |
| 3408 | clean-gated | 2782.4 | 6.89 | 150.58 | 48287376 | 27724500 |
| 3408 | clean-control | 2782.7 | 6.89 | 151.40 | 48287376 | 27724500 |
| 3408 | augmented-baseline | 2761.4 | 6.85 | 156.45 | 48287360 | 0 |
| 3408 | augmented-gated | 2868.3 | 6.89 | 151.43 | 48287376 | 27724500 |
| 3408 | augmented-control | 2857.9 | 6.89 | 153.67 | 48287376 | 27724500 |

3408 累计 4.688 逐窗 GPU 小时，验证累计 709.8 秒。

[累计统计](evidence/r2i-s2/cumulative.json) · [执行指纹](evidence/r2i-s2/execution-sha256.txt) · [日志](evidence/r2i-s2/pipeline.log)
