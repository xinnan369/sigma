# R3-D1B-R1 — Selective Scan 确定性反向

**合成验收通过，正式训练仍未获准。** 固定外部 Sigma `98ba3c6938d29169bbd27f91519084bb462561df` 的前向和真实 CUDA 反向均保留；新版本 `sigma-r3-exclusive-block-pairwise-v1` 消除了反向中的跨 block 写入竞争。没有修改或重编译 pinned 上游，没有替换为简化 SSM，没有停止任何参数梯度。实验基线为 PR #31 Head `b7a44916cc0c8f07261261a5798ab58ff506483c`。

## 根因与覆盖

对固定提交的 `selective_scan/csrc` 全树搜索 `gpuAtomicAdd(` 和 `atomicAdd(`，共五处调用，均见[反向 kernel](https://github.com/zifuwan/Sigma/blob/98ba3c6938d29169bbd27f91519084bb462561df/models/encoders/selective_scan/csrc/selective_scan/selective_scan_bwd_kernel.cuh)。其 grid 为 (batch, channel)，group=floor(channel / channels_per_group)。

| 位置 | 写入 | 相同地址的竞争者 |
|---|---|---|
| 224/225 | dB/dC[batch, group, state, t] | 相同样本、相同组的不同 channel block |
| 264 | dD[channel] | 相同通道的不同样本 block |
| 269 | ddelta_bias[channel] | 相同通道的不同样本 block |
| 272 | dA[channel, state] | 相同通道的不同样本 block |
| 普通 store | du/ddelta[batch, channel, t] | 无跨 block 共享，无 atomic 调用 |

重复传入相同的前向缓存、u/delta/A/B/C/D/bias 和上游梯度：原生五个共享梯度均出现差异，du/ddelta 在全部检测用例中逐位一致。干预仅改变写入归属与共享量归约顺序，重放差异消失。这将观测到的非确定性定位到这些竞争归约路径；不是仅从源代码出现 atomic 推断，也不主张涵盖扩展的未测数据类型和其他结构。

## 实现与数值比较

第一阶段对每个样本调用 pinned native backward，把组输入 B/C 重复到逐通道组。此时一个 launch 中每个输出梯度槽只有一个 block 写，原函数虽仍执行 atomic，但没有跨 block 地址竞争。第二阶段用 CUDA 张量相邻两元素配对归约，奇数尾项逐层携带，先按通道还原 dB/dC、再按样本归约 dA/dD/ddelta_bias；du/ddelta 直接拼回。分样本计算限制同时存活的部分张量。前向完全不改，保留原 `saved`。只支持固定 FP32、四状态、softplus ABI，其他输入报错。补丁只安装在本进程的上下文中，退出恢复原调用，未将单线程上下文当作通用并发接口。

预先写入 `atol=2e−5, rtol=2e−4`，只用于**原版对新版**比较；同版本重放和断点恢复始终比较字节，无容差。合成矩阵包括 B=4/2、80/40/20 三个融合尺度以及额外 7×7 网格，Cross 原生形状单组 128 通道，Concat 两组 256 通道且序列双长。16 个原生案例各重放 4 次；输出和 native 前向内部缓存均逐位相同。D 整模型在 batch 4 和 2 上重复测试三个尺度，所有可训练参数和两个模态输入的梯度新版重放逐位相同；候选与原版 forward 逐位相同。

| 梯度 | 原版重放最大绝对差 | 原版重放最大相对差 | 原版 vs 新版最大绝对差 | 原版 vs 新版最大相对差 |
|---|---:|---:|---:|---:|
| du | 0 | 0 | 0 | 0 |
| ddelta | 0 | 0 | 0 | 0 |
| dA | 7.45e−9 | 1.02e−4 | 7.45e−9 | 5.10e−5 |
| dB | 5.22e−8 | .2242 | 5.96e−8 | .1518 |
| dC | 4.47e−8 | .2591 | 4.47e−8 | .2791 |
| dD | 1.91e−6 | 5.72e−5 | 9.54e−7 | 6.79e−5 |
| ddelta_bias | 7.45e−9 | 1.72e−5 | 7.45e−9 | 1.58e−5 |

相对差用参考幅值作分母，接近零时可放大。审计另存每个试次的稳定化相对误差、字节差异数与混合容差比。原生级新版对原版最大混合比 0.00817；D 整模型所有参数梯度最大绝对差 1.01e−6、最大混合比 0.0443，输入梯度最大绝对差 3.26e−9、最大混合比 0.000156。整模型未经稳定化的相对峰值在近零输入梯度处约 43.86，不代表绝对误差大。所有预设用例都纳入，未按结果选样。

环境：RTX 3090／SM 8.6、WSL2、PyTorch 2.2.2+cu121、CUDA runtime 12.1、cuDNN 8902、FP32、TF32 关闭、单线程、cuBLAS :4096:8。未生成新二进制；native selective_scan 0.0.1 SHA256 `833021baeecf26b84d5ecba208417893a9eda202c38854fd9a9737c8ccd7513e`。原始构建记录见 [R1 build](evidence/r1/extension-build.txt)，本机 nvcc/host compiler 与实际加载路径、adapter 源码 hash 见数值 JSON；已安装编译器不作为扩展重新编译的证据。

## 跨进程恢复及全局闸门

A/B/C/D 四臂均通过 2 步连续 vs 首步保存并退出、重开恢复再运行第 2 步的精确比较。比较模型/BN、AdamW 全状态、四套 RNG、更新数、按冻结采样函数生成的合成索引顺序、事件字节前缀与 SHA256。D 每两步 18 次原生前向、72 次原生反向子调用，原生内部 scan 缓存 SHA256 也纳入逐步事件比较。损坏、签名不符、未提交日志尾及原子投影恢复的测试保留。初次 harness 仍按旧调用次数断言而失败，日志已保留；修正为实际分样本调用数后重跑通过。原版 D 的失败证据保留在 D1B。

四臂入口都需要显式授权和独立验收登记的回执摘要。回执必须同时声明 A/B/C/D 恢复、数据完整性及协议签名均 PASS，并绑定所有 20 个 arm/seed 运行签名、冻结 split。验证在真实样本访问前进行。R1 的信任登记为空，**本轮任何臂均不能开训**；未声称完成真实数据完整性复验或签发正式协议回执。

## 合成性能

以下为同步实测中位数及峰值 allocated；完整记录还分别列 loss、AdamW 和 reserved。输入为随机张量，更新使用冻结 FCOS 与 AdamW。

| D batch | 版本 | Forward ms | Backward ms | 批更新 ms | 峰值 allocated MiB |
|---|---|---:|---:|---:|---:|
| 4 | 原生 | 15.78 | 84.56 | 161.27 | 1257.53 |
| 4 | 确定性 | 13.53 | 110.52 | 179.16 | 1386.36 |
| 2 | 原生 | 12.96 | 49.87 | 93.78 | 706.72 |
| 2 | 确定性 | 13.09 | 64.55 | 111.12 | 890.34 |

batch 4 增加约 129 MiB allocated、backward +31%、批更新 +11%；batch 2 增加约 184 MiB、backward +29%、批更新 +19%。前向与调度不变，短测的前向时间波动不能解读成加速。四组五种子十轮计算估计 **18.06 GPU 小时**（A 3.61、B 3.69、C 4.59、D 6.18），不含数据加载、验证、checkpoint 和现场竞争，不能当作正式用时。

## 证据与复验

Windows CPU 全套 **183 通过、7 跳过**、compileall 退出 0。GPU 数值及资源审计、四臂恢复审计最终都退出 0。历史实验、D0 数据划分、模型、FCOS 头／损失、正式种子保持不变。官方 Test 读取 **0**、真实 LLVIP optimizer update **0**，正式训练 **NOT RUN**。

```sh
python -m pytest -q -rs tests
python -m compileall -q experiments/r3_d1
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m experiments.r3_d1.deterministic_audit --upstream <PINNED_SIGMA> --output reports/evidence/r3-d1b-r1/numerics.json
python -m experiments.r3_d1.resume_audit --backend deterministic --upstream <PINNED_SIGMA> --output reports/evidence/r3-d1b-r1/resume.json
```

[完整数值／资源](evidence/r3-d1b-r1/numerics.json) · [恢复结果](evidence/r3-d1b-r1/resume.json) · [日志和摘要](evidence/r3-d1b-r1/)。不提交任何图像、权重或大体积训练日志。
