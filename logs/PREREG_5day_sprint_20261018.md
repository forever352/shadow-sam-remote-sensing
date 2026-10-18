# 5 天冲刺预注册（2026-10-18 ~ 2026-10-23）· 第 2 版

> 目的：在**不伪造、不在 test 上调参**的前提下，尽可能补齐申报书技术承诺。
> 本文件在启动任何 >5 分钟 GPU 任务前写定；任何偏离须先改本文件并重新送审。
> **第 1 版已被 supervisor 判“不通过”**；本版按 12 条必改重写。
> 已运行的两个冒烟（`ber_tta_val_20261018_171947.log` 2 图、`direction_rotation_20261018_172006.log` 1 epoch）
> **只是代码检查，不得作为任何实验证据**。

## 0. 硬约束

1. **所有超参/阈值/模型选择只在 val（58 张 / 16 原图）上做**；test（92 张 / 21 场景）**不是 virgin holdout**（历史上已被查看多轮）。
2. **单一主终点**：F1 的 BER<5%。F2/F3/F4 均为**次级/探索性**，其 test 结果只作探索性证据，不宣称独立泛化。
3. **test 台账**：每一次 test 评估（日期、配置、结果、是否作为选择依据）必须登记；**所有结果（含失败）强制披露**；禁止跨族挑最好结果。
4. 主结果仍以 **3-seed 均值 + 自建无泄漏 68/16/21** 为准；新实验**单列**，不覆盖旧 CSV。
5. 任何数字必须有**落盘 CSV/日志**；未落盘 = 未完成。
6. 对外只写“**数值上高于**”，禁“SOTA/显著超越”；方向模块禁用词不变。
7. 5 天做不完的项，如实写“**未完成/未复现**”，不补假数字。
8. 任何 >5 分钟 GPU 任务前先跑 `python preflight_check.py`，退出码非 0 不得启动。

## 1. Test 台账（启动前为空；每评一次追加一行）

| 日期 | 实验族 | 配置 | 选择依据 | 结果 | 是否作为选择依据 |
|---|---|---|---|---|---|
| （待填） | | | | | |

**历史 test 查看（必须披露）**：BER 两轮（4 + 11 候选，11 个候选全部算过 test）、方向提示 oracle、提示来源、误差结构、多 seed 基线、跨数据集。当前 `results/` 下已有 82 个 `*test*.csv`；新实验的 test 结果是**探索性证据**，不是独立泛化证明。

## 2. 实验族与判据

### F1 BER<5%（唯一主终点）

**基线**：B30 = `T30_none_no`（3-seed 集成 + 阈值 0.30 + 无 TTA + 无后处理）。
- val：BER 0.051175（精确 0.051174737）、IoU 0.814307。
- test：BER 0.053664、IoU 0.790188（历史值，仅作参照）。

**H1**：3-seed 集成 + flip TTA + 阈值网格 `0.10–0.60`（step 0.05）+ 后处理候选 `none/close3/close5/fill/close3_fill`。
- 候选数：11 阈值 × 5 后处理 × 2 TTA = **110**；全部在 val 上评估。
- **选择规则（恢复既有协议）**：
  1. val IoU ≥ baseline val IoU − 0.01；
  2. 相对 B30 的**逐图配对 BER**：mean_gain > 0 且 |t| > 2.0；
  3. **按原图基名聚合**（n=16）的配对：mean_gain > 0 且 t > 2.0；
  4. 在满足以上者中取 **val BER 最小**；并列取 val IoU 高者。
- **边界检查**：若选中阈值 ∈ {0.10, 0.60}，标记 `threshold_at_grid_boundary=true`，报告中注明“最优不稳定”。
- **多重比较披露**：110 个候选、未做多重比较校正；选择是探索性的。
- **val 门禁**：val BER < 0.047。**test 成功判据（v2.1，test 前保守修订）**：test BER < 0.05 **且**相对 B30 逐图配对 `mean_gain > 0` **且**场景级 `base_paired_mean_gain > 0`；场景级 t 一并报告。本修订不改变 `selection_sha256`，不需重跑 val。
- **TEST_SIGNOFF.json 字段**：`approved`、`approved_by`、`selection_sha256`、`command`（精确为 `ber_tta_eval.py --stage test`，不含 `python`）、`checkpoints`（`SEED_CKPTS` 绝对路径）、`script_sha256`（`ber_tta_eval.py`、`ber_sweep_eval.py`）。
- **技术锁（test 前必须全部满足）**：
  - `ber_tta_selection.json` 含 `selection_sha256`；
  - `logs/sprint20261018/TEST_SIGNOFF.json` 由 supervisor 预签核，含同一 hash；
  - test 阶段开头检查 `test_evaluated`，已评过则拒绝；
  - 92 个 test name 与 `lora_leakfree_test_per_image.csv` 集合断言；
  - selection JSON 原子写；test 日志记录 selection hash、命令、checkpoint 路径。
- **H2**：**只由 H1 的 val 失败触发**；若 H1 已评 test，不得再跑 H2（除非新预注册 + 人类放行）。
  - 损失：`Tversky(alpha=0.3, beta=0.7, gamma=1.0) + Dice`（替换 Focal+Dice 中的 Focal，Dice 保留）。
  - 训练：`--epochs 40 --tag tversky_s{seed} --loss tversky_dice`；**禁传** `--val-frac/--test-frac`。
  - 先 1 seed（seed 0），val 改善（同 H1 的配对规则）再补 seed 1/2；3 seed 中至少 2 个改善才算通过。
  - 成功后接同一 TTA/阈值/后处理选择，test 一次。
- **停止**：H1/H2 均失败 → 报告最优诚实值，不再扩网格。
- **报告要求（v2.1）**：
  - 组件消融表：B30 → 仅 TTA（`T30_none_tta`）→ 仅阈值（`T25_none_no`）→ TTA+阈值（`T25_none_tta`）；明确 **TTA 是主因**，阈值 0.25 的单独效应不显著。
  - top-4 表：`T25_none_tta` / `T25_close3_tta` / `T30_none_tta` / `T30_close3_tta`，列 BER、IoU、逐图 t、场景级 t、场景胜负；说明 0.25 vs 0.30 的差异在窄带内（约 0.000334），不写“唯一最优”。
  - `T25_none_tta` 是“3-seed 集成 + flip TTA + 阈值 0.25”的**变体**；主结果仍是 3-seed 均值 BER 0.061231 / IoU 0.798970。不得写成“主方法 BER<5% 达成/泛化成立”。
  - test 无论好坏都报；失败写“尝试未成功”，不回改主方法/阈值/重选。

### F2 方向 + 自监督（次级/探索性）

- **H3 rotation vs flip vs ImageNet**：
  - rotation：4 向旋转 pretext；**增强与 flip 对齐**（`RandomResizedCrop(scale=(0.4,1.0))` + `ColorJitter(0.4,0.4,0.4,0.1)`）；**15 epochs**（约 495 步，与 flip 30 epochs × 2 样本约 510 步匹配）；3 seeds。
  - flip：沿用既有 `direction_resnet18_flip_basis.pth`（30 epochs、2× 样本）。
  - ImageNet：**显式** `--init imagenet`，不是静默回退；仅作参照，不宣称“计算量匹配”。
  - 回归头：固定 40 epochs、默认增强、seed 0；**所有非选中条件 `--val-only`**。
  - 选择：val 角误差最小者；只有 val 最优者允许评一次 test。
  - **种子对比口径**：rotation 报 3 seed 的 **mean ± sd**；与 flip 单 seed 的 val 28.37 比时用 **rotation mean**，**不得取 3 个 rotation 的 max**；也不得把 3 seed 的 sd 当作与单 seed 的可比不确定性。
  - **若 val 赢家是 flip/ImageNet**：直接引用其**历史 test 值**（flip 28.56；ImageNet 无历史回归 test 则写“未评”），**不重评 test**；rotation 只在 val 输赢，不评 test。
  - **kill criterion**：rotation 相对 flip 的 val 角误差**改善幅度 < 2° 或配对不显著** → 停止 rotation 路线；不设“绝对 24°”一刀切。
- **H4 方向标签去噪（时间允许）**：
  - 先按 `base_name` 切分 train/val/test，**再在各自内部**做场景级圆形中值；preflight 断言三集基名交集为 0。
  - 离群剔除：连通域轴角偏离场景中值 > 60° 的丢弃；至少保留 2 个连通域，否则保留原标签。
  - 同骨干、同头、同 40 epochs、同 seed 0 的单变量对比；能跑则 3 seed。
- **H5 模块② regress + 射线投射**：
  - 用 regress 权重跑 `direction_prompt_eval.py --direction-source regress`，一次运行内报告 `project / project_nofilter / none / random` **4 个变体**。
  - **诊断/探索性**：全部报告，不做“哪个最好”的选择叙事，不宣称增益。
  - regress 权重必须存在；`direction_utils.load_direction_net` 缺失时抛异常，不得静默回退。
- **H6 方向 Adapter 重评（可选）**：
  - 若 H3/H4 产生新方向权重，**重训** Adapter（不是只换推理输入）；否则不做。
  - 明确披露：Adapter 单 seed / 92 张 / 21 场景 / 与主方法 0.7878 无关。
  - 若只做 inference-only 敏感性分析，必须写明“不是改进证明”。

### F3 双流 + Feature Aggregator（次级/探索性，降级表述）

- **H7**：**最小 Cross-Attention 融合原型**，是申报书 Feature Aggregator 的**部分实现**（不含多尺度 DownConv/FusionConv）。
  - 架构：方向 token（2→hidden→256）对 prompt tokens 做 cross-attention；零初始化输出投影保证初始恒等；`gamma=1`；fp32 计算（避免 autocast 下梯度下溢）。
  - 训练基线：从 seed-0 主权重增量加载；**冻结图像编码器（含 LoRA）与 prompt/decoder，只训融合层**（`--freeze-base --adapter-mode cross_attn --val-only`）。这是“只训融合层”的简化版，**不写“完整双流/完整 Feature Aggregator”**。
  - **随机方向对照**：同预算跑 `--direction-source random --adapter-mode cross_attn --val-only`；若与 `regress` 无差异，H7 **不宣称方向流有效**，只报“融合层可训练、无净增益”。
  - 对比：同 seed-0、同 92/21、同框提示；与 B30/主方法**单列**。
  - **单 seed 即使 val 改善，也不得替换主方法**；须 3 seed + test 后才谈替换。
  - 8GB 显存 fallback：若 OOM，减 `--batch-size` 或只在 512 输入上跑；仍 OOM 则砍掉，写“未实现”。
  - **Day 3 12:00 前未跑起来 → 砍掉**。

### F4 真实检测 SOTA（条件执行，严格限制）

- **H8**：只有当外部方法（omni-shadow / Make SAM Perfect 等）满足以下全部条件才执行：
  1. 任务是阴影**检测**（不是去除）；
  2. 指标同口径（IoU/BER/F1）；
  3. **能在我们的 68 原图 / 262 张训练集上重训**，再评我们的 92 张 test；
  4. 代码/许可证可用。
- 若不能在我们的训练集上重训：写“**不可比/未复现**”，**不得**报告任何外部 SOTA 数字，**不得**写“优于主流”。
- **Day 2 结束前无法跑通 → 砍掉**。
- ShadowFormer 仍注明“阴影**去除**、指标不可比”；WHU 仍注明“无阴影标注、用途错位”。

## 3. 计算预算与排期

- GPU：RTX 4060 Laptop 8GB，当前空闲。
- 参考耗时：LoRA 1 seed ≈5h；3 seed ≈15h；ResNet-18 15 epochs ≈3 min/seed；回归头 40 epochs ≈2–3 min/seed；Mask R-CNN ≈1h；推理 92 张 × 3 seed ≈10min。
- **Day 1**：协议 + 送审；BER H1 代码 + val（不跑 test）；rotation 3-seed 预训练；omni-shadow 检查。
- **Day 2**：rotation 回归（val-only）；H5 模块② regress；H2 仅当 H1 val 失败时启动 1 seed；H7 编码；**Day 2 结束 F4 取舍**。
- **Day 3**：H2 补 seed；H7 训练；**12:00 代码冻结**（不再加新实验）。
- **Day 4**：统一 val 选择；**test 前 supervisor 预签核**（selection JSON hash、命令、checkpoint）；test 一次；落盘。
- **Day 5**：supervisor 审结果；更新文档。**Day 5 不再跑新实验**。

## 4. Kill criteria

| 项 | 判据 | 动作 |
|---|---|---|
| F1 | Day 4 val BER ≥ 0.047 | 停止，报告最优诚实值 |
| F2-H3 | rotation 相对 flip val 改善 < 2° 或配对不显著 | 停止 rotation，保留 regress 跑通 |
| F3 | Day 3 12:00 未启动训练 | 砍掉，写“未实现（部分原型）” |
| F4 | Day 2 结束未跑通 | 砍掉，写“不可比/未复现” |
| 任意 | test 不达标 | 如实写“尝试未成功”，不回改主方法 |

## 5. 输出

- `results/sprint20261018/*.csv`（含 `ber_tta_val_per_image.csv`）、`logs/sprint20261018/*.log`
- `results/sprint20261018/TEST_LEDGER.csv`：每次 test 一行。
- 主表不覆盖；新增“5 天冲刺”附表。
- 仅在 supervisor 通过后更新 `结题报告.md` §五/§七、`进度接力.md`、`README.md`。

## 6. 强制披露清单（报告任何冲刺结果时一并给出）

- 主表：macro、best-val、自建无泄漏 68/16/21（262/58/92、21 场景）；± 为 seed 间样本 sd（ddof=1，n=3）；未做多重比较校正。
- Mask R-CNN：tau=0.05/0.05/0.2；seed-0 clean_run unknown / 无 meta；U-Net/SegNet seed-0 同样无 meta；分辨率不对称（512→800/1333 vs 1024）仍在。
- LoRA seed-1：epoch 11 中断后 `--resume`（不恢复优化器矩/GradScaler/RNG）。
- B30：IoU 0.7902 < 3-seed 均值 0.7990；`success_criterion_met=false`；11 候选全部算过 test；**B30 阈值 0.30 位于旧网格下界，新网格 0.10–0.60 只部分缓解该不稳定**。
- 每次 test 报告须写“本轮新增 N 次 test 查看”，并附 `TEST_LEDGER.csv` 的累计行数。
- 方向回归：28.29/28.37/28.56/29.34 未经单独送审；常数基线 29.83、随机 45；`<15–20°` 是项目内部判据，申报书无门槛。
- 方向 Adapter：单 seed / 92 张 / 21 场景；ΔBER −0.0038、ΔIoU −0.0005；随机方向对照复制同量级 ΔBER；与主方法 0.7878 无关。
- 方向提示：oracle 取自 GT、仅诊断/上界、框提示框架、单划分、21 场景；不得写“方向先验无效”。
- 软件原型：仅一张 512×512 子图 + 一个合成地理参考样本；地理参考分支不代表真实 CRS/轴序已验证。
- 外部 SOTA：未在我们的训练集重训则写“不可比/未复现”，不报数字。
- 本冲刺的 test 结果是探索性证据，不是独立泛化证明。

## 7. 预注册之外的禁止项

- 禁止在 test 上选阈值/选模型/选后处理。
- 禁止把 oracle 方向结果写入主表。
- 禁止把单 seed 结果与 3-seed 均值并列表述。
- 禁止把“跑通”写成“有增益”。
- 禁止把“最小 Cross-Attention 原型”写成完整 Feature Aggregator。
- 禁止在 H1 已评 test 后再跑 H2 并二次评 test。
