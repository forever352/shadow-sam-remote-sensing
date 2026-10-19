# 基于物理先验提示与 LoRA 微调 SAM 的遥感影像阴影检测

## 大学生创新创业训练计划项目结题报告

| 项目 | 内容 |
|---|---|
| 申报项目名称 | 基于方向提示与 SAM 的遥感影像阴影检测方法研究 |
| 团队名称 | ShadowHunter |
| 项目周期 | 2025 年 12 月 — 2026 年 10 月 |
| 核心方法 | 增强物理粗提取 → 连通域框提示 → LoRA 微调 SAM → 阴影矢量 |
| 主结果口径 | 无泄漏重划 68/16/21 原图（262/58/92 张），92 张测试图 / 21 个独立场景 |

> **署名说明**：本仓库为公开代码仓库，团队负责人与指导教师姓名不在本文件中列出，正式结题材料一并另附。

---

## 摘要

遥感影像中的阴影会遮挡地物细节、扭曲光谱信息，直接影响土地覆盖分类、变化检测与目标识别。现有方法要么依赖人工阈值与特征工程，要么依赖大量像素级标注；而通用分割基础模型 SAM 虽然具备强大的边界分割能力，却缺少遥感阴影领域的先验知识，且全量微调成本高。

本项目构建了一条**全自动、无需人工点选**的阴影检测链路：先用阴影“暗、偏蓝”的物理先验做增强粗提取，自动生成连通域框提示；再以 LoRA 参数高效微调 SAM 的图像编码器，并联合训练提示编码器与掩膜解码器；最后将二值掩膜矢量化，输出 GeoJSON / Shapefile。项目同时重建了**无泄漏评估基座**：发现 AISD 自带 Test51 与训练集存在 42/43 原始大图基名重叠，并修复了 U-Net / SegNet 基线中图像与掩膜独立翻转导致的标签错位 bug。

在按原始大图重划的 68/16/21 无泄漏划分上，LoRA 微调 SAM 的 3 seed 均值为 **IoU 0.7990 ± 0.0101、BER 0.0612 ± 0.0027、F1 0.8869**；同划分、同评估脚本下，全监督 U-Net 为 0.7547 / 0.0860，SegNet 为 0.7382 / 0.1003，Mask R-CNN 为 0.7392 / 0.0955。3-seed 配对检验支持该数值优势，但测试集仅 21 个独立场景、未做多重比较校正、效应量随聚合口径变化，因此本项目只表述为“**数值上高于**”，不宣称显著超越或 SOTA。模型仅训练 **4.51M / 93.7M（约 4.8%）** 参数，全部实验在单张 RTX 4060 Laptop（8GB）上完成。

项目还系统探索了阴影方向先验的四种注入方式，得到的是一项**诊断性负结果**：在本项目框架与口径下，方向模块未取得可测增益；该结果不等同于“方向先验普遍无效”。此外，项目交付了可运行的软件原型，实现“输入影像 → 输出阴影矢量”，并完成端到端自检。

**关键词**：遥感影像；阴影检测；Segment Anything Model；LoRA；参数高效微调；物理先验；无泄漏评估

---

## 1 引言

### 1.1 研究背景

阴影是高分遥感影像中的常见现象。它一方面为三维重建、太阳方位估计提供线索，另一方面会遮挡地物细节、改变局部光谱统计，给土地覆盖分类、建筑物提取、变化检测和目标识别带来系统性干扰。随着高分辨率航空与卫星影像的普及，如何在复杂城市场景中稳定、自动地提取阴影，仍然是遥感视觉中的基础问题。

阴影检测的核心困难来自两方面。其一是**类间相似性**：阴影与水体、深色植被、深色路面、蓝色屋顶等低反射率地物在光谱上接近，容易出现“异物同谱”的混淆。其二是**边界复杂性**：阴影由建筑、树木等遮挡体投影形成，边界细碎、孔洞多，且随太阳方位与传感器视角变化。传统方法依赖人工设计的阴影指数、阈值与形态学后处理，泛化能力有限；深度学习方法虽然精度更高，但需要大量像素级标注，跨传感器、跨场景迁移时仍不稳定。

### 1.2 现有方法的瓶颈

传统阴影检测方法通常基于色彩空间变换、阴影指数或区域生长，配合 Otsu 等阈值分割。这类方法计算便宜、无需训练，但对光照、传感器和地物类型敏感，阈值需要逐场景调整。

全监督深度学习方法（如 U-Net、SegNet、Mask R-CNN）通过数据驱动学习阴影特征，在固定数据集上表现稳定，但存在两个现实瓶颈：一是像素级标注成本高；二是模型容量与训练算力要求较高，跨域泛化仍依赖大量目标域数据。

视觉基础模型 SAM（Segment Anything Model）[1] 提供了强大的通用分割能力，可通过点、框提示分割任意目标。然而 SAM 并未在遥感阴影任务上训练，直接零样本使用效果有限；全量微调又会带来显存与训练成本问题。如何以**低算力、少参数**把 SAM 适配到遥感阴影任务，是本项目要解决的核心问题。

### 1.3 本文思路与贡献

本项目的核心思路是：**让物理先验回答“阴影在哪里”，让微调后的 SAM 回答“边界切多准”**。具体而言：

1. 用 CLAHE 增强、Otsu 阈值与形态学开运算得到高召回率的阴影候选区域；
2. 从候选连通域自动生成框提示，不读取任何真值；
3. 在 SAM 图像编码器注意力层挂 LoRA，仅训练少量参数，同时训练提示编码器与掩膜解码器；
4. 将预测掩膜矢量化，输出可直接进入 GIS 的 GeoJSON / Shapefile。

项目的主要贡献包括：

- **一条全自动的物理先验—基础模型融合链路**：无需人工点选，也无需逐场景调阈值；在 8GB 笔记本 GPU 上可完成训练与推理。
- **一个可复核的无泄漏评估基座**：发现并量化 AISD 自带划分的同源泄漏（43 个测试基名中 42 个出现在训练集，51 张测试图中 50 张来自训练场景），并修复基线脚本的图像/掩膜增强错位 bug；所有主结果均在重划的 68/16/21 划分上报告。
- **一组受控的技术演进对照**：同一提示链路、同一划分下比较“零样本 SAM → 冻结编码器微调 → LoRA 微调”，并给出 3 seed 均值与逐图配对结果。
- **一项方向先验的系统诊断**：用四种注入方式、随机方向对照与标签一致性探针，量化“整图单一阴影方向”在当前数据与框架下的可辨识度；如实报告负/中性结果。
- **一个可运行的软件原型**：输入影像输出阴影矢量，含地理参考处理与端到端自检脚本。

### 1.4 论文组织

第 2 节回顾相关工作；第 3 节介绍方法；第 4 节给出实验设置；第 5 节报告结果与分析；第 6 节讨论有效性、边界与局限；第 7 节总结；第 8 节列出成果与交付。与原申报书计划的调整说明放在附录 A，探索性实验与复现入口放在附录 B/C。

---

## 2 相关工作

### 2.1 传统阴影检测

传统方法通常利用阴影在 HSV、LAB 等色彩空间中的低亮度、较高蓝色分量等特性构造阴影指数，再通过阈值分割与形态学后处理提取阴影。代表性做法包括基于色彩特征的阈值法、C3I 阴影指数、以及结合区域生长与边缘约束的方法。这类方法无需训练数据，计算效率高，但阈值对光照、季节、传感器和地物类型敏感，复杂城市场景中容易把水体、深色植被与深色路面误判为阴影。

### 2.2 深度学习阴影检测

随着标注数据增加，U-Net [3]、SegNet [4]、Mask R-CNN [5] 等全监督网络被广泛用于阴影检测。编码器—解码器结构能够学习多尺度上下文，实例分割网络还能区分不同阴影连通域。这类方法在固定数据集上通常优于传统阈值法，但依赖大量像素级标注；当训练与测试场景存在重叠时，指标会被同源泄漏显著抬高，跨场景泛化能力也常被高估。本项目因此把“无泄漏划分”作为评估基座的第一原则。

### 2.3 视觉基础模型与参数高效微调

SAM 以大规模掩膜数据预训练，具备强大的零样本分割能力。将 SAM 迁移到遥感任务的工作大致分为三类：一是提示工程（点、框、文本提示）；二是适配器微调（在 Transformer 层插入轻量模块）；三是全量或部分参数微调。SAM-Adapter [6]、ShadowSAM [7] 等工作表明，少量可训练参数即可显著改善下游任务。本项目采用 LoRA（低秩适配）[2] 在编码器注意力层注入低秩矩阵，同时训练提示编码器与掩膜解码器；相比全量微调，显存占用和训练成本更低，适合 8GB 级消费级 GPU。

### 2.4 阴影方向先验

阴影方向由太阳方位角与遮挡体几何共同决定，理论上可作为额外的物理先验。祁昆仑等（2025）[8] 提出基于阴影方向先验的高分辨率遥感影像地物阴影检测，并引入 DTICT 噪声抑制。本项目在申报阶段计划沿这一方向深入；执行中系统实现了翻转自监督、方向回归头、方向提示与方向 Adapter 四种注入方式，但实验显示：在 AISD 城市场景与当前探针口径下，整图单一阴影方向标签的可辨识度有限，方向模块未取得可测增益。该部分作为诊断性结果在第 3.7 节与第 6.3 节如实报告。

---

## 3 方法

### 3.1 任务定义与总体框架

给定一幅遥感影像 $I$，阴影检测的目标是输出逐像素二值掩膜 $M$，其中 $M(p)=1$ 表示像素 $p$ 属于阴影。项目进一步把掩膜转换为多边形矢量，输出 GeoJSON 或 Shapefile，使结果可直接进入 GIS 流程。

总体框架如图 1 所示，包含五个环节：增强物理粗提取、自动框提示、LoRA 微调 SAM、掩膜合并、矢量化。全流程不需要人工交互，也不需要读取测试真值。

![图 1 总体技术路线](results/figures/pipeline.png)

*图 1　物理先验提示 + LoRA 微调 SAM 的阴影检测流程。物理先验负责高召回候选区域，SAM 负责精细边界；全流程无需人工点选。*

### 3.2 增强物理粗提取

阴影在灰度图上表现为局部低亮度。直接对原图做 Otsu 阈值容易受大面积亮色地物影响；因此本项目先对灰度图做 CLAHE（限制对比度自适应直方图均衡化），增强暗部细节，再对反相图做 Otsu 阈值分割，最后用形态学开运算去除细小路面噪声。

该模块的目标不是直接得到精确阴影，而是获得**高召回率的候选区域**，为后续 SAM 提供框提示。其物理依据是阴影的“暗”特性；CLAHE 与形态学参数在训练/验证阶段固定，测试阶段不调整。

图 2 给出一个完整阶段示例：原图、灰度、CLAHE 增强、反相 + Otsu、形态学开运算、连通域框提示、SAM 合并结果、最终叠加与误差图。

![图 2 分阶段处理示例](results/figures/stage_montage.png)

*图 2　单张 AISD 影像的分阶段处理。黄色框为自动生成的框提示；绿色为最终预测阴影；误差图中 TP 为绿、FP 为红、FN 为蓝。*

### 3.3 自动框提示生成

对粗提取掩膜做连通域分析，过滤面积小于 300 像素的连通域，取每个连通域的轴对齐包围框作为 SAM 的框提示。框提示由输入影像自身生成，**不读取 GT**，因此不存在真值泄漏。

选择框提示而不是点提示的原因有三点：第一，框提示同时编码了位置与尺度，对阴影这种大范围连通区域更稳定；第二，框提示对连通域内部的孔洞与凹陷不敏感；第三，框提示可以直接复用 SAM 的提示编码器，无需额外的提示学习网络。项目也实现了点提示、方向点提示与射线投射等变体，用于消融与诊断（见 3.7）。

### 3.4 LoRA 微调 SAM

SAM 的图像编码器是 ViT-B，参数量约 93.7M。全量微调在 8GB 显存上难以承受，因此本项目采用 LoRA：在编码器每个注意力层的 qkv 与 proj 线性层旁挂低秩矩阵 $W + \Delta W = W + BA$，其中 $B \in \mathbb{R}^{d \times r}$、$A \in \mathbb{R}^{r \times k}$，秩 $r=8$。原始权重冻结，仅训练 $A$、$B$。

除 LoRA 外，项目还训练提示编码器与掩膜解码器。可训练参数合计 **4.51M**，占模型总参数 **93.7M** 的约 **4.8%**；其中 LoRA 参数约 0.44M。图 3 给出参数效率对比。

![图 3 参数效率](results/figures/param_efficiency.png)

*图 3　LoRA 微调仅训练约 4.8% 的参数；其余编码器权重保持冻结。*

这一设计的意义不只是省显存：它把“适配遥感阴影域”的更新限制在低秩子空间内，降低了对小规模训练集（262 张）的过拟合风险，也使模型可以在消费级 GPU 上完成训练。

### 3.5 训练目标与实现细节

训练损失为 **Focal Loss + Dice Loss**。Focal Loss 缓解阴影/背景像素不平衡，Dice Loss 直接优化区域重叠。训练时使用点提示与框提示混合、水平翻转增强、AdamW 优化器、余弦学习率与混合精度（AMP）。

主要超参数：输入按 SAM 的 `ResizeLongestSide(1024)` 处理；编码器冻结，LoRA rank=8；训练 40 轮；batch size 依据 8GB 显存设置；每个 seed 独立训练，best-val 仅按验证集 IoU 选择，测试集不参与选模型。完整命令与权重见附录 C。

### 3.6 从掩膜到矢量

得到二值掩膜后，项目用 `rasterio.features.shapes` 做多边形化，保留孔洞；再用 Douglas–Peucker 简化（默认容差 1.5 像素），按最小面积 64 像素² 过滤碎片，最后写出 GeoJSON 或 Shapefile。

若输入是带地理参考的栅格，输出几何使用该栅格坐标系下的地图坐标，并在 GeoJSON 顶层写入 `crs` 成员；若输入是无地理参考的 PNG/JPG，则输出像素坐标，并在要素属性中标记 `"coord_system": "pixel"`。这一区分避免把像素坐标误当成地图坐标使用。

### 3.7 方向先验的探索性模块

申报阶段计划引入“阴影方向先验”。执行中项目没有把方向模块并入主方法，而是把它作为一组受控诊断实验：

1. **几何方向过滤**：用阴影连通域的主轴或最小外接矩形估计方向，过滤方向杂乱的暗色地物；
2. **方向点提示**：用旋转框角点或主轴采样点替代轴对齐框；
3. **方向点训练**：把方向点提示用于 SAM 训练；
4. **方向 Adapter**：把方向向量条件注入提示编码器。

此外，项目实现了翻转判别自监督预训练、方向回归头、DTICT 噪声抑制与方向标签一致性探针。第 5.7 节与第 6.3 节给出结果：在本项目的数据、标签定义与探针口径下，四种注入方式均未取得可测的 IoU/BER 增益；方向回归头的误差接近常数预测器；方向 Adapter 的 BER 效应可以被随机方向对照复制。该结论是**诊断性负结果**，不等同于“方向先验普遍无效”。

---

## 4 实验设置

### 4.1 数据集

主实验使用 **AISD** [9]（Aerial Imagery Dataset for Shadow Detection）。AISD 提供逐像素二值阴影标注，影像为多尺度航空影像（约 256–1688 像素，均为方图）。AISD 自带 Train412 / Val51 / Test51 划分；项目另外使用 **RSISD** [10]（SARU 提出的阴影检测基准）做跨数据集验证，其掩膜经核验为二值（像素值仅 {0, 255}）。

### 4.2 无泄漏划分

AISD 自带 Test51 与 Train412 存在同源泄漏：Test51 的 43 个原始大图基名中有 **42 个**同时出现在 Train412（仅 `vienna12` 独立）；51 张测试图中 **50 张**来自训练场景。该划分下的指标会系统性偏高——同一 LoRA 方法在自带划分上 IoU 0.8393，在无泄漏划分上 seed-0 为 0.7878，相差约 0.052。

因此，本项目按**原始大图基名整组重划** Train412 为 **68 训练 / 16 验证 / 21 测试** 原图，对应 **262 / 58 / 92** 张子图；三集基名互不重叠。所有深度方法（LoRA-SAM、U-Net、SegNet、Mask R-CNN）均在**同一划分**上重训与评估。AISD 自带划分的数字只作历史对齐，禁与主表横比。

### 4.3 评估指标

项目使用六项指标：IoU、BER、F1、OA、Precision、Recall。BER 定义为

$$
\mathrm{BER} = \frac{1}{2}\left(\mathrm{FPR} + \mathrm{FNR}\right),
$$

其中 FPR、FNR 分别为假阳性率与假阴性率。主表采用 **macro（逐图计算再平均）** 口径；项目另做了 pooled 与 scene 口径的敏感性分析（见 6.4）。

### 4.4 对比方法

学习型基线包括：

- **LoRA 微调 SAM（本项目）**：3 seed；
- **全监督 U-Net**：3 seed；
- **全监督 SegNet**：3 seed；
- **全监督 Mask R-CNN（ResNet-50 FPN，res800）**：3 seed；
- **冻结编码器微调 SAM**：仅训练提示编码器与解码器，单 seed；
- **同提示零样本 SAM**：与主方法只差“是否微调”，单 seed。

非学习/传统基线包括 Otsu、C3I、增强粗提取（本项目自研模块）与 SAM-AMG 零样本自动掩膜。

### 4.5 实现与算力

全部实验在 **NVIDIA RTX 4060 Laptop（8GB）**、Python 3.13、PyTorch 2.11（CUDA 12.8）、SAM ViT-B 上完成。项目使用线性阻塞训练队列与断点续训应对本机 Modern Standby 导致的训练中断；所有结果均落盘为 CSV/日志，脚本与权重路径见附录 C。

### 4.6 统计口径与可复现性

主表的“±”是 **3 seed 的样本标准差（ddof=1，n=3，自由度仅 2）**，只作量级参考，不是 95% 置信区间。逐图/场景配对 t 检验仅作描述性统计，未做多重比较校正。测试集只有 **21 个独立原始场景**，同一场景的子图高度相关，因此不以子图数冒充独立样本量。

所有模型（主表模型；探索性变体除外，见 §5.7）均使用 **best-val** 选模型：只在验证集按 IoU 选择检查点，测试集只用于最终报告。项目保留逐图 CSV、训练日志、检查点哈希与验证脚本，主结果可复核。

---

## 5 结果与分析

### 5.1 主结果：与全监督基线对比

表 1 给出无泄漏划分下的主结果。LoRA 微调 SAM 的 3 seed 均值为 **IoU 0.798970 ± 0.010104、BER 0.061231 ± 0.002728、F1 0.886881 ± 0.006505**；同划分下 U-Net 为 0.7547 / 0.0860，SegNet 为 0.7382 / 0.1003，Mask R-CNN 为 0.7392 / 0.0955。

**表 1　无泄漏划分主结果（92 张 / 21 独立场景，best-val，macro）**

| 方法 | IoU | BER | F1 | OA | 种子 |
|---|---:|---:|---:|---:|---|
| **LoRA 微调 SAM（本项目）** | **0.7990 ± 0.0101** | **0.0612 ± 0.0027** | **0.8869** | **0.9551** | 3 seed |
| 全监督 U-Net | 0.7547 ± 0.0028 | 0.0860 ± 0.0091 | 0.8567 | 0.9453 | 3 seed |
| 全监督 SegNet | 0.7382 ± 0.0044 | 0.1003 ± 0.0020 | 0.8448 | 0.9434 | 3 seed |
| 全监督 Mask R-CNN † | 0.7392 ± 0.0039 | 0.0955 ± 0.0042 | 0.8486 | 0.9412 | 3 seed |
| 冻结编码器 SAM（对照） | 0.5621 | 0.1327 | 0.7105 | 0.8454 | 单 seed |

> † **Mask R-CNN 强制声明**：主行使用 `min800/max1333` 的复跑结果，但输入仍固定 512×512 再上采样到 800，主方法走 SAM 原图 1024；因此分辨率不对称**仅部分缓解**，不得写成“对比已公平/偏置已消除”。Mask R-CNN seed-0 的 `clean_run` 未验证（无 meta），seed-1/2 干净；LoRA seed-1 为 `--resume` 续训。3 seed 对 3 seed 的逐图 IoU 差 **+0.0598**（图级 t=−13.34；**Mask R-CNN 5 胜 / LoRA 87 胜**；场景级 t=−14.42；**Mask R-CNN 0 胜 / LoRA 21 胜**），BER 方向一致。
>
> **统计限定**：测试集仅 21 个独立场景；未做多重比较校正；效应量随聚合口径变化（以 seed-0 为例，LoRA 相对 U-Net 的 IoU 优势在 pooled / macro / scene 下为 +0.005 / +0.035 / +0.050；3-seed macro 差为 +0.0443）。因此本项目只写“数值上高于，且 3-seed 配对检验支持该数值优势”，**不宣称显著超越 / 超越全监督 / SOTA**。

![图 4 主结果指标对比](results/metrics_bar.png)

*图 4　无泄漏主结果（左）与 AISD 自带划分的泄漏存档（右）。两面板口径不同，禁止横比。*

### 5.2 消融：零样本 → 冻结编码器 → LoRA

表 2 给出同一提示链路、同一划分下的递进对照。零样本 SAM（不微调）IoU 0.3480；冻结编码器、仅训练提示编码器与解码器时 IoU 0.5621；LoRA 微调后 seed-0 IoU 0.7878。零样本 → LoRA 的逐图配对平均提升 **+0.4399 IoU（seed-0）**，**92/92 张全部变好**；3 seed 平均配对提升 **+0.4510**。冻结 → LoRA 的对照中，LoRA 在 **91/92 张**更高。

**表 2　技术演进消融（无泄漏划分，92 张）**

| 配置 | IoU | BER | F1 | 说明 |
|---|---:|---:|---:|---|
| 零样本 SAM（同框提示，不微调） | 0.3480 | 0.2923 | 0.4997 | 只差“是否微调” |
| 冻结编码器 + 训练提示/解码 | 0.5621 | 0.1327 | 0.7105 | 单 seed |
| **LoRA 微调 SAM（本项目）** | **0.7990**（3 seed 均值） | **0.0612** | **0.8869** | seed-0 = 0.7878 |

> **归因纪律**：被微调的参数包括 LoRA adapter、prompt encoder 与 mask decoder（合计 4.51M），**不是“只训 LoRA”**。因此表 2 的增益不能全部归因于 LoRA 本身；它反映的是“参数高效适配 + 提示/解码器联合训练”的整体效果。

### 5.3 跨数据集验证

在 RSISD test200（200 张）上做 AISD→RSISD 零样本迁移，LoRA-SAM 的 IoU 为 **0.7842**（seed-0）。需要强调：该跨域任务本身较易——传统 Otsu 在 RSISD 上即可达到 0.5711；因此不能把 0.7842 读作“泛化很好”。跨域仅单 seed，且绝对 IoU 不可与 AISD 主表直接横比。

### 5.4 误差结构

在 seed-0 的主方法上，macro FPR 为 **0.0405**、FNR 为 **0.0882**，漏检约为误检的 **2.2 倍**，说明主要误差在召回侧。按 GT 连通域面积分箱，`<256 px` 连通域的 macro recall 为 **0.538**；但若按绝对漏检像素量统计，`>=4096 px` 的连通域贡献了 **39.0%** 的漏检像素。边界带 r=1/3/5 的带内 F1 分别为 **0.745 / 0.819 / 0.849**。

> 口径提醒：小阴影召回率低 ≠ 小阴影是误差主项；误差主项取决于按连通域计数还是按漏检像素量统计。项目在报告中同时给出两种口径，不把其中一种当作唯一结论。

误检成分分解（seed-0、HSV 启发式判类）显示：水样 **8.3%**、植被样 **12.4%**、其它 **79.3%**；植被样的 lift 为 1.44，水体样为 0.895。该分析的水体判据要求 `V≥110`，会低估深色水体，因此 8.3% 是下界；79.3% 的“其它”也不代表与异物同谱无关。

### 5.5 单图多方法可视对比

为了直观比较各方法的输出，项目在两张影像上运行了同一套 10 种方法/对照：增强粗提取、Otsu、C3I、SAM-AMG、零样本框提示、冻结 SAM、LoRA-SAM、U-Net、SegNet、Mask R-CNN。图 5 为 AISD 测试集中的 `chicago22_sub9.tif`（植被与暗色地物丰富），图 6 为 RSISD 的 `GuiZhou_37.png`（水体 + 植被 + 建筑阴影，跨数据集）。

![图 5 AISD 单图多方法对比](results/figures/compare_aisd.png)

*图 5　AISD 域内单图案例（n=1）。LoRA-SAM 的 BER 0.0814 为该图 10 个方法中最低；U-Net/Mask R-CNN 的 IoU 0.695 略高于 LoRA 的 0.676。单图数值只作陈列，不构成排序或方法优劣结论。*

![图 6 RSISD 单图多方法对比](results/figures/compare_rsisd.png)

*图 6　RSISD 跨数据集单图案例（n=1）。该图水体、植被与建筑阴影同时存在；所有方法都将大片暗色地物并入阴影，反映跨域与异物同谱的困难。单图不作泛化结论。*

这两张图的价值不在“谁第一”，而在展示失败模式：在域内植被/暗色地物密集场景，LoRA 的 BER 最低，但 IoU 不一定最高；在跨数据集水体场景，所有方法都明显过检。它们说明后续工作应把“暗色地物判别”和“跨传感器域适应”作为重点，而不是只追求单张图的 IoU。

### 5.6 软件原型验证

软件原型 `shadow_extract.py` 实现“输入影像 → 输出阴影矢量”。端到端验证脚本 `verify_shadow_extract.py` 覆盖三条分支：

- **GeoJSON 分支**：AISD 512×512 子图，输出 42 个多边形；矢量↔栅格往返面积差 **−5 像素 / 46614（0.0107%）**；栅格化后 IoU **0.843834**，对照管线自身 **0.845274**（差 −0.00144，来自简化与最小面积过滤）。
- **Shapefile 分支**：`.shp/.shx/.dbf` 产出完整，读回 42 要素。
- **地理参考分支**：合成 EPSG:32650 样本，顶层正确写出 `crs`，`coord_system="map"`，坐标落在 transform 推算范围内。

> 验证规模限制：以上仅在一张 512×512 子图与一个合成地理参考样本上完成；合成 CRS 只证明“CRS/坐标写出路径”走通，不代表真实遥感产品的 CRS/轴序已验证。工具默认加载 seed-0 无泄漏权重（IoU 0.7878 / BER 0.0644），不是 3-seed 集成。

### 5.7 探索性补充结果

项目后期用 5 天时间做了四组探索性实验，完整数据与限定见 `results/sprint20261018/SPRINT_APPENDIX.md`。这些结果不改变主表/headline，只作附录。

**表 3　探索性结果摘要（均不进入主表）**

| 实验 | 结果 | 必须随附的限定 |
|---|---|---|
| BER<5% 变体 `T25_none_tta` | test BER **0.048175** / IoU 0.788718 | 3-seed 集成 + flip TTA + 阈值 0.25；110 候选择一、单次 test、test 已多次查看、无多重比较校正；仅低于 5% 约 0.0018；IoU 比主方法低 0.0103；是变体，不是主方法 |
| H3 rotation 自监督 | val 21.631±1.747° vs matched flip 24.369°，t=1.07/0.64 | 配对不显著；按 kill criterion 关闭、不评 test；不写“方向先验无效” |
| H5 模块② regress | project 0.8037/0.0667，random/none 也高于 axis_box | IoU 提升来自点/扇形提示，不是方向分量；BER 均劣于 axis_box；project 8/92 兜底 |
| F3 cross-attn 原型 | 3-seed val ΔBER −0.000863±0.000713 | val-only、未评 test；不写“方向流有效/Feature Aggregator 已兑现” |

BER<5% 变体值得单独说明：在“3-seed 集成 + flip TTA + 阈值 0.25”的明确变体口径下，test BER 可以降到 **4.82%**，首次低于申报书的 5% 线；但代价是 IoU 下降、FPR 升高，且该工作点是 110 个候选中经 val 选择、test 已多次查看。因此它只能作为“存在一条达到 5% 的技术路径”的探索性证据，不能替换主方法，也不能表述为“主方法已达标”。

---

## 6 讨论

### 6.1 为什么有效

本项目的方法有效，主要来自两个互补因素。第一，物理先验把搜索空间从“全图”缩小到“高召回候选区域”，降低了 SAM 在复杂背景中误分割的机会；第二，LoRA 微调让冻结的 ViT 编码器在低秩子空间内适配遥感阴影域，使 SAM 的边界分割能力真正落到阴影任务上。消融显示，零样本 → LoRA 的逐图提升为 +0.4399 IoU（seed-0，92/92 张），冻结 → LoRA 在 91/92 张更高，说明“让编码器适配”比“只换提示”更关键。

### 6.2 与现有方法的关系

与全监督 U-Net / SegNet / Mask R-CNN 相比，本方法在无泄漏划分上的 IoU 数值更高，同时只训练 4.8% 参数，训练与推理可在 8GB 笔记本 GPU 上完成。与零样本 SAM 相比，本方法通过 LoRA 与提示/解码器联合训练取得配对提升 **+0.4399 IoU（seed-0，92/92 张）**。与 SAM-Adapter、ShadowSAM 等适配器工作相比，本项目采用 LoRA 这一更轻量的实现，并把“物理粗提取 + 自动框提示”作为提示来源；差异主要在提示生成方式与训练策略，而非全新的网络结构。

需要强调：本项目**没有**复现真正的阴影检测 SOTA。ShadowFormer [11] 属于阴影**去除**，与检测指标不可比；SARU-Net 仓库缺少可运行的网络定义；WHU Building 数据集没有阴影标注，不能用作阴影泛化测试。因此本项目不做“优于 SOTA”的声明。

### 6.3 方向先验的诊断性发现

方向模块是本项目执行中最重要的“负结果”。四种注入方式在同族无方向对照（project vs project_nofilter / none；random vs none）下，IoU 差异 ≤0.0007；但方向点提示相对轴对齐框提示的 oracle IoU 为 **0.8037 vs 0.7878（+0.0158）**，这部分来自提示族差异（点/扇形 vs 框），不能归因于方向。方向 Adapter 的 ΔBER 为 −0.0038、ΔIoU 为 −0.0005，但随机方向训练的对照也能复制出同量级 ΔBER（跨 adapter t=−1.00/−0.11），说明该效应不依赖于方向内容。方向回归头的验证误差为 28.29/28.37°、测试 29.34/28.56°，对照常数预测器 29.83°、随机 45°，相对常数改善约 0.5–1.5°（取决于 run）；方向标签一致性 R 为 0.23–0.37，样本量匹配随机基线 0.212，指向光源的物理代理仅 +0.015。

这些结果说明：在 AISD 城市场景、当前“整图单一方向”标签定义与探针口径下，方向先验的可辨识度有限，未能转化为检测增益。但该结论受四重局限：线性探针、全局池化、样本量、单划分；最上游假设（翻转自监督是否让骨干编码方向）仍悬置。因此不能写成“方向先验无效”或“导师论文结论不成立”，只能写成“在本项目框架与口径下未观察到可测增益”。

### 6.4 局限与失效边界

本项目的主要局限如下：

1. **统计规模**：测试集仅 21 个独立原始场景；未做多重比较校正；效应量随 pooled/macro/scene 口径变化（以 seed-0 为例为 +0.005 / +0.035 / +0.050；3-seed macro 差为 +0.0443）。因此只写“数值上高于”，不写“显著超越”。
2. **硬指标**：主方法 BER 未达到 5% 以下（无泄漏 3-seed 0.0612 / seed-0 0.0644；AISD 自带划分 0.0501）。5 天冲刺的 `T25_none_tta` 变体可到 0.0482，但属于附录变体，不是主方法。
3. **对比公平性**：Mask R-CNN 分辨率不对称仅部分缓解；输入协议不统一（SAM 类多一个增强 Otsu 预检测器）；Mask R-CNN seed-0 clean_run 未验证；LoRA seed-1 为 resumed；冻结编码器版仅单 seed。
4. **跨域证据**：RSISD 仅单 seed，且任务较易（Otsu 0.5711）；不能据此宣称泛化能力强。
5. **方向模块**：四种注入方式均无可测增益；最上游假设仍悬置。
6. **软件原型**：只在单张 512×512 子图与合成 CRS 样本上验证；预处理参数固定；跨传感器、真实 CRS/轴序未经大规模验证；默认权重是 seed-0，不是 3-seed 集成。
7. **训练增强**：仅水平翻转 + 点/框提示混合；辐射/几何校正未做；申报书中的对比度调整、随机裁剪、色彩抖动未纳入训练。
8. **SOTA 对比**：未复现真正的阴影检测 SOTA；ShadowFormer 不可比，SARU 无网络定义，WHU 无阴影标注。
9. **成果状态**：学术论文尚未产出；软著说明书初稿为旧口径，需按 3-seed headline 重做后提交。

这些局限不改变主结果“在无泄漏划分上数值上高于三个全监督基线”的事实，但限定了结论的适用范围：它是**中等规模、单数据集为主的实证结论**，不是跨传感器、跨大洲的泛化证明。

### 6.5 启示

后续工作可以从四个方向推进：第一，扩大测试场景与传感器类型，尤其是跨域与跨季节验证；第二，把“暗色地物判别”作为显式任务（多类掩膜或辅助分类），缓解水体/植被/深色路面的混淆；第三，继续探索方向先验，但需要更可靠的整图方向标签与空间—非线性探针；第四，把软件原型从单图 CLI 扩展为可安装工具包，并补充真实 GeoTIFF 的批处理验证。

---

## 7 结论

本项目面向遥感影像阴影检测，提出并实现了一条“物理先验粗提取 → 自动框提示 → LoRA 微调 SAM → 阴影矢量”的全自动流程。在按原始大图重划的无泄漏划分上，LoRA 微调 SAM 的 3 seed 均值为 IoU 0.7990、BER 0.0612，数值上高于同划分的 U-Net、SegNet 与 Mask R-CNN；模型仅训练 4.8% 参数，全部实验在 8GB 笔记本 GPU 上完成。项目还主动发现并修复了 AISD 自带划分的同源泄漏与基线增强 bug，重建了可复核的评估基座；系统探索了方向先验，并如实报告了负/中性结果；交付了“输入影像 → 输出阴影矢量”的可运行软件原型。

结论的边界同样明确：测试集仅 21 个独立场景，主方法 BER 未达到 5%，未复现真正的 SOTA，方向模块未取得可测增益。本项目不宣称显著超越或 SOTA，只报告在明确口径下可复核的数值结果与方法学产出。

---

## 8 成果与交付

### 8.1 代码与权重

| 类别 | 内容 |
|---|---|
| 主方法训练 | `train_lora_leakfree.py` / `train_lora_leakfree_seed.py`（主结果入口）；`train_adapter_v4.py`（通用训练模块） |
| 对比基线 | `train_cnn_leakfree.py`、`train_maskrcnn_leakfree.py`、`run_baselines_leakfree.py`、`sam_box_zeroshot_eval.py` |
| 评估与统计 | `eval_shadow.py`、`multiseed_summary.py`、`scene_ci.py`、`error_stratify.py`、`fpr_decompose.py` |
| 方向模块 | `train_direction_net.py`、`train_direction_regress.py`、`direction_prompt.py`、`direction_adapter.py` 等 |
| 软件原型 | `shadow_extract.py`、`verify_shadow_extract.py`、`shadow_extract_stages.py` |
| 数据与权重 | AISD、RSISD；LoRA seed0/1/2、U-Net/SegNet/Mask R-CNN 3 seed、冻结 SAM、方向模块权重 |

### 8.2 软件原型

`shadow_extract.py` 支持：

```bash
python shadow_extract.py --image input.tif --out shadows.geojson
python shadow_extract.py --image input.tif --out shadows.shp --format shp
python shadow_extract.py --image input.tif --out shadows.geojson --overlay preview.png
```

输出为带孔洞的阴影多边形，支持 GeoJSON / Shapefile 与地理参考处理。端到端验证见 `logs/verify_shadow_extract_*.log`。

### 8.3 文档与可复现性

项目保留：审查日志 `review_log.md`、进度快照 `进度接力.md`、代码索引 `代码与文件索引.md`、自动化程序说明 `自动化程序_使用说明.md`、资产获取说明 `资产获取说明.md`。所有主结果均有逐图 CSV、训练日志与检查点支撑；关键脚本改动留有 patch。

### 8.4 学术论文与软著状态

学术论文尚未产出。软著说明书初稿数字为旧口径，需按 3-seed headline（0.7990 ± 0.0101）重做，并把产品名改为与实际方法一致的名称后提交。这两项是本项目尚未完成的成果，不作“推进中”的美化。

---

## 参考文献

[1] Kirillov A, Mintun E, Ravi N, et al. Segment Anything[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 2023.

[2] Hu E J, Shen Y, Wallis P, et al. LoRA: Low-Rank Adaptation of Large Language Models[C]//International Conference on Learning Representations (ICLR). 2022.

[3] Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation[C]//Medical Image Computing and Computer-Assisted Intervention (MICCAI). 2015: 234-241.

[4] Badrinarayanan V, Kendall A, Cipolla R. SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017, 39(12): 2481-2495.

[5] He K, Gkioxari G, Dollár P, et al. Mask R-CNN[C]//Proceedings of the IEEE International Conference on Computer Vision (ICCV). 2017: 2961-2969.

[6] Chen T, Zhu L, Ding C, et al. SAM-Adapter: Adapting Segment Anything in Underperformed Scenes[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW). 2023.

[7] Chen X D, Wu W, Yang W, et al. Make Segment Anything Model Perfect on Shadow Detection[J]. IEEE Transactions on Geoscience and Remote Sensing, 2023, 61: 4410713.

[8] 祁昆仑, 马欣悦, 金准, 等. 基于阴影方向先验的高分辨率遥感影像地物阴影检测[J]. 遥感学报, 2025, 29(3): 776-791. DOI: 10.11834/jrs.20243314.

[9] Hu X, et al. Aerial Imagery Dataset for Shadow Detection (AISD). https://github.com/xw-hu/UnShadowNet（访问日期：2026-10）.

[10] SARU. RSISD: Remote Sensing Image Shadow Detection Benchmark. 项目内路径 `datasets/RSISD/`；获取方式见 `资产获取说明.md`.

[11] Guo L, Huang S, Liu D, et al. ShadowFormer: Global Context Helps Image Shadow Removal[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2023.

> AISD/RSISD 的下载与放置方式见 `资产获取说明.md`；RSISD 的原始出处以数据集目录说明为准。

---

## 附录 A　与原申报书计划的调整说明

> 本节只作执行记录，不作为正文结论。评委若不关注申报书对照，可跳过本节。

| 原计划 | 实际执行 | 调整原因 |
|---|---|---|
| 双流特征提取 + Feature Aggregator | 未实现；实际为物理粗提取 + 框提示 + LoRA-SAM | 时间与算力限制；优先保证主链路可运行、可评估 |
| 射线投射采样 + 扇形搜索 | 代码已实现，仅在 oracle 口径跑过；主方法用框提示 | 同族无方向对照增量 ≤0.0007；相对框提示 +0.0158 来自提示族差异，与方向无关；未进入主链 |
| IoU Loss + Boundary Loss | 未采用；实际 Focal + Dice | 人类决定不做；Focal+Dice 已能稳定训练 |
| ShadowFormer 对比 | 未做 | ShadowFormer 属阴影去除，指标不可比 |
| WHU Building 辅助测试 | 未做；改用 RSISD | WHU 无阴影标注，用途错位 |
| 数据预处理/三种增强 | 辐射/几何校正未做；训练增强仅水平翻转 + 点/框提示混合 | AISD 为已处理数据；优先保证主结果 |
| 算力描述 | 全部实验在 RTX 4060 Laptop 8GB 上完成 | 实际配置，如实报告 |

---

## 附录 B　探索性实验与复现入口

- 5 天冲刺完整附录：`results/sprint20261018/SPRINT_APPENDIX.md`
- 测试台账：`results/sprint20261018/TEST_LEDGER.csv`
- 主结果逐图：`results/lora_leakfree_test.csv`、`results/lora_leakfree_s1_test.csv`、`results/lora_leakfree_s2_test.csv`
- 多 seed 汇总/配对：`results/cnn_multiseed_summary.csv`、`results/cnn_multiseed_paired.csv`、`results/scene_level_ci_multiseed.csv`
- 跨数据集：`results/cross_rsisd_test200.csv`
- 误差结构：`results/error_component_size_enhanced.csv`、`results/error_boundary_band_enhanced.csv`、`results/error_by_scene_enhanced.csv`
- 误检成分：`results/fpr_composition.csv`
- 方向模块：`results/direction_regress_ablation.csv`、`results/diradapter_eval_main.csv`、`results/dirprompt_*_oracle_test.csv`
- 单图多方法对比：`results/visual_compare/`、`results/figures/compare_aisd.png`、`results/figures/compare_rsisd.png`
- 软件原型验证：`logs/verify_shadow_extract_*.log`

---

## 附录 C　复现命令

```bash
# 环境（示例）
conda create -n shadow-sam python=3.13
conda activate shadow-sam
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e ./segment-anything-main

# 主方法训练（无泄漏划分；主结果实际为 40 轮，默认 150 轮，必须显式 --epochs 40）
python train_lora_leakfree.py --epochs 40

# 多 seed（同样 40 轮）
python train_lora_leakfree_seed.py --seed 1 --epochs 40
python train_lora_leakfree_seed.py --seed 2 --epochs 40

# 3-seed 汇总与主结果 CSV（训练脚本会写出 lora_leakfree_test.csv）
python multiseed_summary.py

# 注意：evaluate_lora.py 只评估 AISD 自带 Test51（含同源泄漏），不能用于复现主表
# python evaluate_lora.py   # 仅历史对齐，输出 lora_test51.csv

# 软件原型
python shadow_extract.py --image AISD/AISD/Train412/shadow/chicago22_sub9.tif --out shadows.geojson
python verify_shadow_extract.py
```

> 资产（AISD、RSISD、SAM 官方权重、本项目微调权重）不在仓库内，获取方式见 `资产获取说明.md`。
> 示例命令中的 `chicago22_sub9.tif` 物理上位于 `Train412/` 目录，但按原始大图基名重划后属于 **92 张测试集**；目录名不代表划分归属。
