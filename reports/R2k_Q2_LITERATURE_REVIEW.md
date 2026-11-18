# R2k-Q2-P0 定向文献核查

范围：核查作者论文、出版社或正式会议条目。以下是有针对性的比较，不是系统综述；「可测试」栏属于本项目设计，不是原文结论。

| 路线与一手来源 | 原文处理的问题 | 局限与 Q2 取舍 |
|---|---|---|
| Shi、Xu、Jia，CVPR 2014，[Discriminative Blur Detection Features](https://openaccess.thecvf.com/content_cvpr_2014/papers/Shi_Discriminative_Blur_Detection_2014_CVPR_paper.pdf) | 对比梯度、频域及局部学习滤波特征，检测空间变化的图像模糊。 | 内容本身可偏平滑；可借鉴频带/边缘证据，但不能把 RGB 数据上的模糊标签或成绩迁移给 Thermal。建议与同预算幅度对照比较。 |
| Ferzli、Karam，IEEE TIP 2009，[Just Noticeable Blur](https://pubmed.ncbi.nlm.nih.gov/19278916/)（DOI 10.1109/TIP.2008.2011760） | 用局部对比与边缘宽度建模人眼可觉察的模糊，再做概率汇总。 | 人眼感知清晰度不是热传感器的光学状态或下游信息量；可检验局部边缘宽度代理，不直接移植人的可察觉阈值。 |
| Mittal、Moorthy、Bovik，IEEE TIP 2012，[BRISQUE](https://ieeexplore.ieee.org/document/6272356) | 用归一化亮度的自然场景统计衡量无参考的整体失真。 | 域偏移可制造统计异常，正常低纹理也可显得异常。适合作为区分 IQA 和故障证据的背景，不列为主方案。 |
| Geifman、El-Yaniv，NeurIPS 2017，[Selective Classification for Deep Neural Networks](https://papers.neurips.cc/paper_files/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html) | 用拒判交换覆盖率和错误风险，在其数据及假设下校准风险。 | 不替 Thermal 的设备/场景偏移提供保证。采用风险—覆盖审计和训练组外错误监督，不把论文风险保证套用于本项目。 |
| Wan 等，WACV 2025，[Sigma](https://openaccess.thecvf.com/content/WACV2025/html/Wan_Sigma_Siamese_Mamba_Network_for_Multi-Modal_Semantic_Segmentation_WACV_2025_paper.html)；Xu 等，2026，[RSGMamba](https://arxiv.org/abs/2604.12319)（预印本） | Sigma 比较多模态分割架构；RSGMamba 用自门控状态空间融合并在 RGB-T/RGB-D 分割基准评价。 | 分割性能不能直接验证某个局部 Thermal 分数的可辨识性、传感器状态识别或拒判可靠性。Q2 先固定独立质量验证；不宣称一般门控创新。 |

历史 Q1D 发现模糊降低梯度及动态范围，而健康低纹理也可具有相同统计；Platt 不会修复单调反向排序。Q1v2 多尺度头在开发侧已见退化改善，LLVIP 的合成模糊却未给出预期方向。这与上述内容歧义相容，但不足以归因到某个滤波器、设备或输入适配。项目的新假设是带独立参考标签的 O、D 与可审查拒判 U 的联合验证；不是一般无参考模糊检测或可靠性门控首创。
