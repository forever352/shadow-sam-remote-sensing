# 5 天冲刺附录（2026-10-18）

> 内部实验附录，仅作探索性证据。**主表 / headline 不变**：主方法 3-seed 均值 **BER 0.061231 / IoU 0.798970**。
> 所有数字均有落盘 CSV/JSON/日志；未落盘的不写。测试集已被多次查看，本附录的新 test 结果不构成独立泛化证明。

## 1. F1：BER<5% 变体（`T25_none_tta`）

配置：3-seed 集成 + flip TTA + 阈值 0.25，无后处理。

| 方法/口径 | IoU | BER | F1 | Recall | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|
| **T25_none_tta（test）** | **0.788718** | **0.048175** | 0.880389 | 0.958219 | 0.054568 | 0.041781 |
| B30（3-seed 集成 + 阈值 0.30，参照） | 0.790188 | 0.053664 | 0.881361 | 0.941246 | 0.048574 | 0.058754 |
| 主方法 3-seed 均值 | 0.798970 | 0.061231 | 0.886881 | 0.913947 | — | — |

- 逐图配对 vs B30：BER gain **0.005490**、t **9.391**、83 胜 / 9 负（n=92）。
- 场景级：gain **0.005753**、t **9.690**、20 胜 / 1 负（n=21）；唯一负场景 **innsbruck18**（0.046409 → 0.046803，−0.000393）。
- **必须披露**：test BER 仅低于 5% 约 **0.0018**，刚过线；IoU 比 B30 低 0.001470、比主方法低 0.010252；工作点从漏检偏多移向误检偏多（FPR +0.005994、FNR −0.016973）；这是 **BER 变体**，不是主方法；110 候选择一、单次 test、21 场景、test 已多次查看、未做多重比较校正。

**组件分解（val-only）**：
- B30 0.051175 → 仅 TTA `T30_none_tta` **0.046230** → 仅阈值 `T25_none_no` **0.050805**（t≈1.06，不显著）→ TTA+阈值 `T25_none_tta` **0.045896**。
- **TTA 是主因**；阈值 0.25 单独仅约 **0.000334**。
- top-4 val 窄带：0.045896 / 0.046002 / 0.046230 / 0.046249；不写“唯一最优”。

**稳健性（val-only）**：
- 留一原图 16 折：`T25_none_tta` 15/16 折被选中，另 1 折选 `T20_none_tta`；**selection subset 16/16 折 <0.047**；**held-out 9/16 折 <0.047**；**held-out 16/16 折 gain>0**。
- 12-of-16 原图子抽样（1000 次）：`T25_none_tta` 720/1000；固定配置 BER 5–95 分位 **0.043943–0.047435**，86.8% <0.047。

## 2. H3：rotation 自监督（val-only，未评 test）

| 条件 | val 角误差（°） | 训练角误差（°） | 备注 |
|---|---:|---:|---|
| flip_ref | **24.369** | 22.230 | matched baseline（40 epochs/默认增强/seed 0） |
| imagenet_ref | **24.415** | 18.468 | 显式 `--init imagenet` |
| rot_s0 | **23.018** | 11.379 | 3 seed 之一 |
| rot_s1 | **22.207** | 12.717 | 3 seed 之一 |
| rot_s2 | **19.668** | 14.689 | 单 seed 观察，不取 max |
| **rot_mean** | **21.631 ± 1.747** | — | 3 seed 间 sd，不是 CI |

- flip − rot_mean：图级 mean **+2.738°**、sd 19.442、t **1.072**、30/28；原图级 n=16 mean **+1.328°**、sd 8.353、t **0.636**、7/9。
- flip − rot_s2：图级 +4.700°、t 1.537；原图级 +2.909°、t 1.184。**仍不显著**。
- flip − imagenet_ref：图级 −0.046°、t −0.017；原图级 −4.857°、t −1.625 → **flip 相对 ImageNet 无优势**。
- 结论：rotation 的 val 均值数值上低 2.74°，但**配对不显著**（图级 t=1.07、原图级 t=0.64），按预注册 kill criterion **关闭 H3，不评 test**。不写“方向先验无效”；方向模块维持“已实现但无可测显著改善/仍不确定”。
- 限定：matched flip_ref 24.369 是本次重跑，历史 28.37 只作背景、不用于配对；rotation 3 seed vs flip/ImageNet 1 seed；16 个 val 原图；训练/val gap 明显；rot_s2 是 3 选 1 的单 seed 观察。

## 3. H5：模块② regress + 射线投射诊断（探索性）

| 变体 | IoU | BER | F1 | 说明 |
|---|---:|---:|---:|---|
| project（扇形+方向过滤） | **0.803690 ± 0.061623** | 0.066741 | 0.889813 | 方向取自 regress |
| project_nofilter | 0.803443 | 0.066300 | 0.889706 | 有扇形、无过滤 |
| random | 0.803413 | 0.066326 | 0.889682 | 随机方向 |
| none | 0.802989 | 0.066272 | 0.889415 | 无扇形/无过滤 |
| axis_box（主方法框提示） | **0.787821** | **0.064359** | 0.879607 | seed-0 参照 |
| centroid_points | 0.802594 | 0.065570 | 0.889162 | 点提示参照 |

- 4 个方向变体差异很小（IoU 0.00025–0.00070、BER 0.00003–0.00047）；`project` 的 IoU 高于 axis_box，但 **random / none 也同样高于 axis_box** → **IoU 提升来自点/扇形提示，不是方向**。
- `project` 的 BER **0.066741 劣于** axis_box 0.064359；是 IoU/BER 权衡，不是方向增益。
- `project` 有 **8/92** 张触发无方向兜底；数值是混合结果。
- 限定：seed-0、92 张/21 场景、单次 test、test 已多次查看、± 为逐图 sd；不做选择、不宣称增益、不入主表；不与主方法 3-seed 均值 0.061231/0.798970 同口径比较。

## 4. F3：Cross-Attention 融合原型（Feature Aggregator 部分实现；val-only）

**每 seed val（best epoch）**

| seed | arm | best epoch | val IoU | val BER |
|---:|---|---:|---:|---:|
| 0 | regress | 3 | 0.809997 | 0.057200 |
| 0 | random | 1 | 0.808782 | 0.058885 |
| 1 | regress | 1 | 0.813567 | 0.056948 |
| 1 | random | 6 | 0.813125 | 0.057435 |
| 2 | regress | 1 | 0.808673 | 0.057460 |
| 2 | random | 4 | 0.810425 | 0.057878 |

**每 seed ΔBER（regress − random；负=regress 更低）**

| seed | ΔBER | ΔIoU |
|---:|---:|---:|
| 0 | −0.001685 | +0.001216 |
| 1 | −0.000487 | +0.000441 |
| 2 | −0.000417 | −0.001751 |
| **mean ± sd（n=3）** | **−0.000863 ± 0.000713** | **−0.0000315 ± 0.001539** |

**3-seed 逐图/原图级配对（描述性；同图跨 seed 平均，非独立验证）**
- ΔBER：图级 mean −0.000863、sd 0.001421、t **−4.625**、9/49；原图级 n=16 mean −0.000888、sd 0.000619、t **−5.737**、0/16。
- ΔIoU：图级 mean −0.0000315、sd 0.005390、t **−0.0445**、16/42；原图级 mean −0.000288、t −0.529、5/11。**无显著/一致差异**（s2 的 ΔIoU 为 −0.001751，s0/s1 为正）。

**相对基线与 random 的贡献**
- 基线 seed-0 LoRA（A50 val）：IoU 0.812812 / BER 0.060884。
- base−regress：IoU +0.002815（t 2.49）、BER +0.003685（t 9.43）。
- base−random：IoU +0.004031（t 4.99）、BER +0.001999（t 6.63）。
- random 约占 regress 总 BER 改善的 **54%**（0.0020/0.00368）；方向分量额外约 **0.0017/0.00049/0.00042**（s0/s1/s2）。
- 两臂相对基线都是 **IoU 降、BER 降**，是权衡，不是净改进。

**结论**
- cross-attn 融合层可训练；在 3-seed val 上，方向分量相对随机方向显示一个**小的 BER 降低**（约 0.00086，3 seeds 同号，|mean|>seed sd），但效应约为**主方法 seed 级 BER sd 0.002728 的 0.32 倍**。
- **不能写“方向流有效”，也不能写“方向流无效”**；准确表述是“**val 上相对随机方向对照的小信号，未经 test 确认**”。
- 限定：val-only、固定 0.5 阈值、best-val 早期点（ep1–6）、58 张/16 原图、单划分、未评 test；best-val 选择与配对同集；3-seed t 为描述性、非独立；± 为 seed 间 sd（n=3），不是 CI。
- 只进本附录；不替换主方法；不写“完整双流/完整 Feature Aggregator/已兑现”。

## 5. 未兑现 / 不可兑现项

| 项 | 状态 | 说明 |
|---|---|---|
| ShadowFormer 对比 | **不可兑现** | 属阴影**去除**、指标不可比；未找到可在我们 68/262 训练集重训的替代检测 SOTA 代码 → 写“不可比/未复现”，不报外部数字、不写“优于主流”。 |
| WHU 辅助测试 | **不可兑现** | WHU 无阴影标注、用途错位；RSISD 已作跨数据集替代（零样本、单 seed、任务较易）。 |
| SARU-Net 真 SOTA | **未复现** | 仓库无网络定义代码。 |
| IoU + Boundary Loss | **未实现** | 人类已决定不做；实际 Focal + Dice。 |
| 数据预处理/三种增强 | **未实现** | 辐射/几何校正未做；训练增强仅水平翻转 + 点/框提示混合。 |
| 主方法 BER<5% | **未达成** | 主方法 3-seed 均值 0.061231；仅 `T25_none_tta` 变体 test 0.048175（刚过线，有 IoU 代价）。 |
| 论文 / 软著 | **未做** | 人类本轮优先技术指标，论文/软著暂停。 |

## 6. 证据路径

- F1：`ber_tta_selection.json`、`ber_tta_val_grid.csv`、`ber_tta_val_per_image.csv`、`ber_tta_robustness.json`、`ber_tta_test.csv`、`ber_tta_test_per_image.csv`、`TEST_LEDGER.csv`、`logs/sprint20261018/ber_tta_*.log`。
- H3：`direction_regress_*_summary.csv`、`direction_regress_*_val_per_image.csv`、`direction_regress_compare.json`、`logs/direction_regress_20261018_*.log`。
- H5：`results/dirprompt_*_regress_test.csv`、`*_per_image.csv`、`logs/dirprompt_20261018_174141.log`。
- F3：`diradapter_h7_cross_*_val_summary.csv`、`f3_{regress,random}_s{0,1,2}_val_per_image.csv`、`f3_compare_val.json`、`f3_compare_seeds.json`、`logs/diradapter_20261018_*.log`、`logs/sprint20261018/direction_adapter_selftest_20261018_175540.log`。
- 预注册/审查：`logs/PREREG_5day_sprint_20261018.md`、`review_log.md` 第五十三~五十九轮。
