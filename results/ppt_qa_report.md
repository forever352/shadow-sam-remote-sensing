# 结题答辩 PPT 质量报告（QA）

- 交付物：`结题答辩PPT_最终版.pptx`
- 生成器：`make_ppt.py`（支持 `PPT_OUT` 环境变量指定输出文件名）
- 页数：20（16:9）
- 嵌入媒体：10
- 演讲者备注：20 页均有
- 结构审计：`results/ppt_audit.md`（目标 = `结题答辩PPT_最终版.pptx`），`high=0 / medium=0 / low=0`
- 视觉复核：用 PowerPoint COM 从**最终 canonical** 导出 20 页 PNG 到 `results/ppt_preview/`，检查了 contact sheet 与关键页（封面、主结果、方向、结论、第 5/6/14/15 页）

## 审计范围说明

`audit_pptx_quality.py` 只做 **PPTX XML 结构检查**（形状越界、文本量、图片尺寸、对齐近失等），**不替代**逐页视觉检查，也不验证科学数字。数字与口径由 `review_log.md` 的审查轮次单独核验。

## 视觉复核结果

- 未发现文字溢出、文字被框裁切、图注被图片压住、图片错位。
- 关键页：封面含官方申报项目名；主结果页含统计限定与证据强度披露；方向页明确“同族无方向对照 ≤0.0007 / 相对框提示 +0.0158 来自提示族差异”；结论页保留边界与验证规模。
- 已知版面取舍：第 4、16 页及部分表格页下半部留白偏多；属于版式平衡问题，不影响信息完整性。

## 复现方式

```bash
python make_report_figures.py     # 生成报告/PPT 图表（需先有 results/stages 与 results/visual_compare 产物）
python make_ppt.py                # 生成 PPT
python ~/.agents/skills/nature-paper2ppt/scripts/audit_pptx_quality.py \
    结题答辩PPT_最终版.pptx --report results/ppt_audit.md
```

## 已知限制

- 字体按 `Microsoft YaHei` 设置；若答辩机器缺字体，PowerPoint 会替换，建议提前在答辩机预览一次。
- 视觉复核在本机完成；未在第二台机器上做字体/版式回归。
