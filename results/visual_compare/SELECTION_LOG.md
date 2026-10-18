# 单图可视对比 · 选图日志与预注册锁定

- 时间：2026-10-18（在任何模型推理之前写定）
- 目的：人类要求选一张“带水体和植被”的影像，运行项目已存各方法模型，做直观输出对比。**这是 n=1 的案例展示，不是 benchmark。**
- 本文件写定后，**跑模型前不再换图**；跑完也不因结果好坏换图。

## 候选筛选过程（只用影像内容，不用任何模型输出/逐图指标/GT 表现）

1. 先在 AISD 无泄漏 92 张测试集中按 HSV 水体样/植被样像素占比排序，并做缩略图目视复核。候选包括 `vienna8_sub2`、`tyrol-e17_sub1/2`、`tyrol-e33_sub1`、`chicago22_sub5`、`chicago29_sub5`、`austin24_sub6` 等。目视结果：**AISD 测试集中没有明显水面**，高“水样”比例主要来自蓝色屋顶/阴影；因此不选 AISD。
   - 备注：`results/visual_compare/water_candidates.jpg` 是 AISD 候选拼图，仅用于这次排除，不是最终选图依据。
2. 转到 RSISD `test200`，对全部 200 张按同一 HSV 指标排序，并做缩略图目视复核。候选前 6：

| 文件 | HSV 水样占比 | HSV 植被占比 | 目视 |
|---|---:|---:|---|
| GuiZhou_37.png | 16.00% | 13.76% | **右下有明显水体，周边植被，大楼阴影清楚** |
| LuAn_40.png | 18.87% | 13.23% | 左侧有河道/水渠，植被多，阴影较碎 |
| POLY_7.png | 11.77% | 29.40% | 圆形罐体/厂房，水体不典型 |
| HeNan_47.png | 14.67% | 8.56% | 水体不明显 |
| GuiZhou_115.png | 33.11% | 7.73% | 水体/蓝色施工区混杂，阴影对比弱 |
| TangShan_12.png | 25.36% | 6.02% | 有喷泉/水池，植被偏少 |

3. 最终选 **`datasets/RSISD/test200/shadow/GuiZhou_37.png`**（512×512）。理由：水体、植被、建筑阴影三者同时清楚，最适合看“阴影 vs 暗色地物”的差异。

## 锁定信息

- 影像路径：`datasets/RSISD/test200/shadow/GuiZhou_37.png`
- 影像 SHA256：`ff75f755dc4aef18a9ad52b43a21a74fff56b60fb92a4ac20a7c748b925c000d`
- GT 路径：`datasets/RSISD/test200/mask/GuiZhou_37.png`（已核验 unique={0,255}）
- HSV 指标（仅作选图依据）：水样 15.998%，植被 13.760%

## 预注册限定（写死在输出中）

- **跨数据集**：所有学习模型均在 **AISD** 上训练，本图来自 **RSISD test200**；跨数据集差异是预期现象，但本单图不作掉点/泛化结论。
- **单图、n=1**：不是随机抽样、不是独立新测试；该图已包含在 `results/cross_rsisd_test200_per_image.csv` 的 200 张聚合评估中。
- **seed-0**：学习模型只跑 seed-0；无 TTA、无 3-seed 集成、无显著性检验。
- **不排序、不写 best、不写“泛化/优于/提升/改进/跨域掉点”**；不与 AISD 主表 0.7878、RSISD 聚合 0.7842 或文献数字横比。
- **Mask R-CNN**：输入固定 512×512，模型内部 `min_size=800 / max_size=1333`，score `tau=0.05`（seed-0 的 val 选中值），mask 软阈值 0.5。
- **C3I**：使用 `invert_if_mostly_white=False` 的主表口径（不带本项目自加的反转启发式）；启发式副行不列入。
- **Enhanced**：本项目自研的粗提取模块（主方法内部模块），**不是外部对照**。
- **方法集合取舍**：列入 10 个方法/对照（Enhanced、Otsu、C3I、SAM-AMG、零样本框提示、冻结编码器、LoRA-SAM、U-Net、SegNet、Mask R-CNN）；不列 C3I 启发式副行（已知代码缺陷，仅口径对照）、`mask_ours`（方向过滤负结果，非主方法）、LoRA seed-1/2（同方法，只展示 seed-0）。
- **GT 隔离**：推理只读输入影像；GT 在所有预测掩膜落盘后才读取，仅用于算指标。
- **LoRA 复现门禁**：本图 LoRA-SAM seed-0 的 IoU/BER/F1 必须与 `results/cross_rsisd_test200_per_image.csv` 的 `GuiZhou_37.png` 行在 1e-6 内一致；不一致即停止，不报告。

## 预期输出

- `results/visual_compare/GuiZhou_37_compare.png`
- `results/visual_compare/GuiZhou_37_metrics.csv`
- `results/visual_compare/GuiZhou_37_masks/*.png`
- `logs/visual_compare_*.log`
