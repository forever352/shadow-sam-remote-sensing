# archive/ — 归档区（2026-10-14）

> **归档 ≠ 删除。** 本项目铁律是"找不到证据 = 未完成"，所以所有移入这里的文件都**完整保留**，只是从工作目录挪开，避免干扰判断。
> 若要引用归档文件，路径为 `archive/<子目录>/<文件名>`。

## 归档内容与理由

| 子目录 | 文件 | 为什么归档 |
|---|---|---|
| `old_logs/` | `train_log.txt`、`train_v2_log.txt`、`train_v3_quick.log`、`boxes_tta_result.txt` | 2026-10-12 的早期临时输出；已在 `logs/` 下有正式日志；**全库 grep 确认无任何文档引用** |
| `old_figs/` | `c3i_result.png` | 早期 C3I 演示图，已被 `results/figures/` 取代；无引用 |
| `old_ppt/` | `结题答辩PPT.pptx` | 最旧版本（10-11）；当前版本为根目录的 `结题答辩PPT_最终版.pptx`，旧泄漏口径版另有备份 |
| `old_sam_dir/` | `segment-anything/`（整个目录，5 个文件） | **不是 SAM 包**，只是早期实验残留（`GPU_text.py`/`test_sam.py`/一张图）；真正的 SAM 源码在 `segment-anything-main/`（editable 安装）。原 `.gitignore` 已把 `segment-anything/` 标为"早期实验残留" |

## ⚠️ 特意**没有**归档的文件（因为被引用）

| 文件 | 被谁引用 |
|---|---|
| `train_v4_quick.log`、`segnet_train.log`、`unet_train.log` | **`review_log.md`（审计证据）** —— 例如"LoRA IoU 0.839 有落盘日志证据（`train_v4_quick.log` 第35行）"。挪动会破坏证据链 |
| `boxes_tta_full.py`、`interactive_sam.py`、`demo_result.png` | `README.md` 的目录结构说明 |
| `test_image.jpg` | `demo.py` 的默认输入图 |

## 2026-10-16 追加归档（会话收尾清理）

| 子目录 | 文件 | 为什么归档 |
|---|---|---|
| `old_results/` | `cross_rsisd_test200_smoke*.csv` | G 阶段**冒烟**（10 张）产物；无结果引用 |
| `old_results/` | `maskrcnn_leakfree_smoke_res800*.csv`（3 个） | Mask R-CNN@800 的**冒烟**产物；无引用 |
| `old_results/` | `summary.csv`、`per_image.csv` | **2026-10-11 的旧 run**（`Ours=0.4708`），不含 LoRA。**留着危险**——文件名像结果，易被误引（记忆 `experiment-results` 早已标注其"陈旧、不能当证据"） |
| `old_results/` | `metrics_bar.png` | **陈旧图**：用的还是旧数字（U-Net 0.741 等）。由 `metrics_chart.py` 可重新生成；**现图不能进材料** |
| `f3_cut_20261016/` | F3（IoU+Boundary 损失）三个实现文件 + README | 方案过审（第十九轮，6 条必改）但**人类决定不做**。**含一个语法错误的脚本**（README 已注明不可运行、含重启须落实的 6 条） |
| `reference_docs/` | `_application_form.txt` | **原始 XML 转储**（不可读）；同样的内容已有干净抽取版在外层 |

## 2026-10-16 另外**删除**（不是归档）

- `__pycache__/`（可再生）
- `.idea/`（IDE 配置，零价值）
- `backups/unet_shadow_leakfree_epoch50interrupted.pth`（**119 MB**）—— 中断的 50 轮 U-Net 权重，已被完整 60 轮版取代。**其数字仍完整保留**在同目录的两个 `*_epoch50interrupted.csv` 里（那是"旧 0.7414"说法的证据来源）。
  ⚠️ **它正是"新均值配旧标准差"事故的来源**：报告用了它的 std(0.1069) 配新均值(0.7525)。

## 另删除

- `__pycache__/`（Python 字节码缓存，可再生）

## 判定方法（可复现）

1. **依赖分析**：AST 解析根目录所有 `.py`，找出每个模块被谁 `import` → 被依赖的一律不动
2. **引用检查**：对候选文件 `grep -rl <文件名> --include=*.md .` → 被任何文档引用的不动
3. 只有"既无依赖、又无文档引用"的文件才归档

*生成日期：2026-10-14*
