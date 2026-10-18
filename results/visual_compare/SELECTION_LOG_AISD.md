# 单图可视对比（AISD 版）· 选图日志与预注册锁定

- 时间：2026-10-18（在任何模型推理之前写定）
- 目的：人类要求“换 AISD 输出”。本版把同一套多方法对比跑在 **AISD 无泄漏测试集**的一张图上，作为**域内**案例。
- 本文件写定后，跑模型前不再换图；跑完也不因结果好坏换图。

## 选图过程（只用影像内容，不用模型输出/逐图指标/GT 表现）

1. 在 AISD 无泄漏 92 张测试集（21 个独立场景）中，按 HSV 水体样/植被样像素占比排序并做缩略图目视复核。
2. 结论：**AISD 测试集中没有明显水体**（高“水样”比例主要来自蓝色屋顶/阴影/深色路面）。因此本版**不宣称有水体**，改选**植被与暗色地物最丰富**的测试图。
3. 候选（目视 + HSV 占比）：

| 文件 | HSV 水样 | HSV 植被 | 目视 |
|---|---:|---:|---|
| chicago22_sub9.tif | 7.69% | 37.87% | **大面积草地/树冠、建筑阴影清楚、暗色路面/屋顶多** |
| chicago16_sub9.tif | 1.31% | 38.23% | 植被多，但阴影较碎 |
| chicago16_sub14.tif | 1.73% | 47.88% | 植被最多，图较小（320×320） |
| chicago22_sub5.tif | 25.63% | 9.59% | 绿色草坪+深蓝屋顶，水体不典型 |

4. 最终选 **`AISD/AISD/Train412/shadow/chicago22_sub9.tif`**（616×616；GT 阴影占比 11.70%）。理由：植被/暗色地物丰富、建筑与树木阴影都清楚，适合看“阴影 vs 深色植被/暗色地物”。

## 锁定信息

- 影像路径：`AISD/AISD/Train412/shadow/chicago22_sub9.tif`
- GT 路径：`AISD/AISD/Train412/mask/chicago22_sub9.tif`（unique={0,255}，已核验）
- **目录说明**：文件物理上位于 `Train412/`，但按原始大图基名重划后属于 **68/16/21 中的测试集（92 张）**，不是训练图。
- 该图属于 AISD 无泄漏 92 张测试集（21 个独立场景）中的一张；**不是独立新测试**。

## 预注册限定（写死在输出中）

- **域内**：学习模型均在 AISD 训练，本图也在 AISD 测试集；**不是跨数据集**。
- **n=1、seed-0**：无 TTA、无 3-seed 集成、无显著性检验。
- **不排序、不写 best、不写“优于/提升/改进”**；不与 AISD 主表 3-seed 均值（0.7990 / BER 0.0612）、92 张聚合或文献数字横比。
- **Mask R-CNN**：输入固定 512×512，模型内部 `min_size=800 / max_size=1333`，score `tau=0.05`（seed-0 的 val 选中值），mask 软阈值 0.5。
- **C3I**：`invert_if_mostly_white=False` 的主表口径；启发式副行不列入。
- **Enhanced**：本项目自研粗提取模块（主方法内部模块），不是外部对照。
- **方法集合**：Enhanced、Otsu、C3I、SAM-AMG、零样本框提示、冻结编码器、LoRA-SAM seed0、U-Net seed0、SegNet seed0、Mask R-CNN seed0。C3I 启发式副行、`mask_ours`、LoRA seed1/2 不列，理由同 RSISD 版。
- **GT 隔离**：推理只读输入影像；GT 在所有预测掩膜落盘后才读取，仅用于算指标。
- **LoRA 复现门禁**：本图 LoRA-SAM seed0 的 IoU/BER/F1 必须与 `results/lora_leakfree_test_per_image.csv` 的 `chicago22_sub9.tif` 行在 1e-6 内一致；不一致即停止，不报告。
- **水体说明**：AISD 测试集无可见水体，本图代表“植被/暗色地物”而非“水体”。

## 预期输出

- `results/visual_compare/AISD_chicago22_sub9_compare.png`
- `results/visual_compare/AISD_chicago22_sub9_metrics.csv`
- `results/visual_compare/AISD_chicago22_sub9_masks/`
- `logs/visual_compare_aisd_*.log`
