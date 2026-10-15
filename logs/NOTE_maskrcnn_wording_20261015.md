# 旁注：本目录部分日志中的「冒烟」字样系措辞 bug（2026-10-15）

**结论：日志文字有误，但数字有效，本批是正式跑。**

## 背景

`train_maskrcnn_leakfree.py` 的日志措辞逻辑曾把「冒烟」与「自定义 `--tag`」混用：
只要 `--tag` 非空就打「冒烟产物…不得作为结果引用」。而 `_res800` 是**正式跑**的标签
（用于让同一脚本的不同分辨率产物**互不覆盖**），不是冒烟。

## 受影响的日志与行

| 文件 | 含误导字样的位置 |
|---|---|
| `logs/maskrcnn_leakfree_20261015_190926.log` | `:14`、`:107`、`:115`（打印了「冒烟产物（_res800），不得作为结果引用」） |
| `logs/chain_queue.log` | `[3/6]` 前后的同名字样 |

## 为什么数字仍然有效

- 该字样**只影响一行日志文本**，不进入任何计算路径（不进 `compute_metrics`、不改权重、不改 tau 选择）。
- `results/maskrcnn_leakfree_res800_test.csv` 的 92 行逐图指标与 `_res800_test_per_image.csv`
  **逐位一致**（第十六轮审查独立复算确认）；划分断言 `[OK]`（92 张 / 21 原图）通过。
- 冒烟与正式跑的**文件名已隔离**：冒烟写 `*_smoke_res800_*`，正式跑写 `*_res800_*`，
  **未互相覆盖**。

## 已做的修复

- 脚本已改为**只有 tag 含 `_smoke` 时**才输出「冒烟」字样；自定义 tag 输出「正式跑」。
- patch 备份：`backups/patches/train_maskrcnn_leakfree_20261015_wording.diff`
  与改动前整份：`backups/patches/train_maskrcnn_leakfree.before_20261015`。
- **本目录的日志文件未被改动** —— 它们是证据，只在此处旁注。
