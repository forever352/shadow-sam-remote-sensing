# -*- coding: utf-8 -*-
"""生成结题答辩 PPT（16:9）。"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn

DARK = RGBColor(0x1F, 0x3B, 0x63)
ACCENT = RGBColor(0x2E, 0x74, 0xB5)
GRAY = RGBColor(0x55, 0x55, 0x55)


def set_font(run, size=18, bold=False, color=None, name="Microsoft YaHei"):
    f = run.font
    f.size = Pt(size)
    f.bold = bold
    f.name = name
    if color is not None:
        f.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find(qn('a:ea'))
    if ea is None:
        ea = rPr.makeelement(qn('a:ea'), {})
        rPr.append(ea)
    ea.set('typeface', name)


def add_bullets(tf, items, size=18):
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = it
        set_font(r, size=size, color=DARK)
        p.space_after = Pt(8)


prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def title_bar(slide, text, sub=None):
    tb = slide.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(12.1), Inches(1.0))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = text
    set_font(r, size=30, bold=True, color=DARK)
    if sub:
        p2 = tf.add_paragraph()
        r2 = p2.add_run(); r2.text = sub
        set_font(r2, size=14, color=GRAY)


def content_slide(title, bullets, sub=None):
    s = prs.slides.add_slide(BLANK)
    title_bar(s, title, sub)
    tb = s.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.6))
    tf = tb.text_frame
    tf.word_wrap = True
    add_bullets(tf, bullets, size=20)
    return s


# 1. 封面
s = prs.slides.add_slide(BLANK)
tb = s.shapes.add_textbox(Inches(1.0), Inches(2.2), Inches(11.3), Inches(1.8))
tf = tb.text_frame; tf.word_wrap = True
r = tf.paragraphs[0].add_run(); r.text = "基于方向提示与SAM的遥感影像阴影检测方法研究"
set_font(r, size=34, bold=True, color=DARK)
tb2 = s.shapes.add_textbox(Inches(1.0), Inches(4.0), Inches(11.3), Inches(1.6))
tf2 = tb2.text_frame
for i, t in enumerate(["大学生创新创业训练计划项目 · 结题答辩",
                       "团队：ShadowHunter    负责人：潘俊豪    指导教师：祁昆仑"]):
    p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
    rr = p.add_run(); rr.text = t
    set_font(rr, size=18 if i == 0 else 16, color=GRAY)

# 2. 背景
content_slide("研究背景与问题", [
    "阴影是遥感影像中的常见干扰，遮挡地物细节、扭曲光谱信息",
    "影响土地分类、变化检测、目标识别等下游任务精度",
    "传统方法：人工调阈值、特征工程，效率低、适应性差",
    "深度学习方法：依赖大量人工标注，成本高",
    "我们的思路：物理先验（暗、偏蓝、方向一致）+ 分割一切大模型 SAM",
])

# 3. 技术路线
content_slide("技术路线", [
    "① 增强物理粗提取：CLAHE 增强 + Otsu 阈值 + 形态学去噪 → 候选阴影区域",
    "② 自动提示生成：候选区域包围框作框提示（不读 GT）",
    "③ 参数高效微调 SAM：冻结编码器、仅训练约 4.8% 参数，适配遥感阴影",
    "整体：物理先验回答“在哪”，微调后的 SAM 回答“切多准”",
])

# 4. 方法一
content_slide("方法 ①：增强物理粗提取", [
    "灰度化 + CLAHE 自适应直方图均衡化，增强暗部细节",
    "反相 + Otsu 自动阈值，分离暗区与背景",
    "形态学开运算去除细小路面噪声",
    "得到高召回率的候选阴影区域",
])

# 5. 方法二：参数高效微调（核心）
content_slide("方法 ②：LoRA 参数高效微调 SAM（核心）", [
    "在 SAM 图像编码器注意力层挂 LoRA（低秩适配，rank=8）",
    "冻结原权重，只训练 LoRA（0.44M）+ 解码器 + 提示编码器，共 4.51M（约 4.8%）",
    "Focal Loss + Dice 损失，点/框双提示，水平翻转增强，混合精度加速",
    "结果：无泄漏划分上 IoU 0.788 / BER 0.064，数值上高于全监督 U-Net(0.741)/SegNet(0.740)",
])

# 6. 方法三
content_slide("方法 ③：自动提示生成", [
    "从增强粗提取结果取连通域质心作为前景点",
    "在候选区域外采样背景点，共同构成点提示",
    "实现全自动阴影检测，无需人工点选",
])

# 7. 实验设置
content_slide("实验设置", [
    "数据集：AISD；⚠️ 官方 Test51 与训练集同源泄漏（43 基名中 42 个重叠）",
    "主口径：按原始大图重划 68/16/21 原图（262/58/92 张），三集互不重叠（无泄漏）",
    "模型：SAM ViT-B（358MB），NVIDIA RTX 4060（8GB）",
    "选模型：仅在验证集按 IoU 选 best-val，测试集只用于最终报告",
    "指标：IoU / BER / F1 / OA / Precision / Recall（逐图计算再平均）",
])

# 8. 结果表
s = prs.slides.add_slide(BLANK)
title_bar(s, "定量对比结果（无泄漏划分，92 张 / 21 独立场景，best-val）",
          "所有方法同一划分重训；单 seed、未做显著性检验")
rows = [
    ["方法", "IoU", "BER", "F1"],
    ["LoRA 微调 SAM（本项目）", "0.788", "0.064", "0.880"],
    ["全监督 U-Net †", "0.741", "0.087", "0.846"],
    ["全监督 SegNet", "0.740", "0.102", "0.847"],
]
tbl_shape = s.shapes.add_table(len(rows), 4, Inches(1.6), Inches(1.9), Inches(10.1), Inches(2.0))
tbl = tbl_shape.table
for i, row in enumerate(rows):
    for j, val in enumerate(row):
        cell = tbl.cell(i, j)
        cell.text = val
        pr = cell.text_frame.paragraphs[0]
        pr.alignment = PP_ALIGN.CENTER
        is_best = (i == 1)
        set_font(pr.runs[0], size=16, bold=(i == 0 or is_best), color=ACCENT if is_best else DARK)
        if is_best:
            cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor(0xDE, 0xEA, 0xF6)
tb = s.shapes.add_textbox(Inches(1.0), Inches(4.4), Inches(11.3), Inches(2.4))
tf = tb.text_frame; tf.word_wrap = True
add_bullets(tf, [
    "† U-Net 为训练 58/60 轮中断后、第 50 轮 best-val 检查点（val IoU 0.7831）",
    "统计限定：单 seed、未做显著性检验，测试集仅 21 个独立场景，领先幅度小于逐图标准差",
    "→ 仅表述为“数值上高于”全监督基线，不宣称“显著超越”",
    "官方（含泄漏）口径：LoRA 0.839 / U-Net 0.805‡ / SegNet 0.780‡（‡ 旧 run：泄漏+增强 bug），因系统性偏高，不作为干净泛化结论",
], size=13)

# 9. 指标柱状图
s = prs.slides.add_slide(BLANK)
title_bar(s, "指标对比柱状图（无泄漏划分）")
s.shapes.add_picture("results/metrics_bar.png", Inches(1.5), Inches(1.4), width=Inches(10.3))

# 10. 可视化对比
s = prs.slides.add_slide(BLANK)
title_bar(s, "分割效果可视化对比（示例）")
s.shapes.add_picture("results/figures/austin13_sub2.png", Inches(0.5), Inches(1.5), width=Inches(12.3))

# 11. 创新点
content_slide("主要创新点", [
    "LoRA 参数高效微调：只训 4.8% 参数，无泄漏划分上数值上高于全监督 U-Net/SegNet",
    "物理先验 + 基础模型融合，实现全自动阴影检测（无需人工点选）",
    "「零样本 → 冻结微调 → LoRA 微调」的完整技术演进路径",
    "主动排查并修复官方划分同源泄漏 + 基线增强 bug，重建诚实评估基座",
])

# 12. 不足与后续
content_slide("问题与不足、后续计划", [
    "不足：自监督方向感知网络尚未完成（当前方向先验为手写规则版）；论文/软著推进中",
    "后续：实现学习式方向估计 + 编码器内嵌 Adapter（参考 RSPrompter）",
    "后续：更大数据集验证、与 SOTA 方法对比、封装为工具包/QGIS 插件",
])

# 13. 结束
s = prs.slides.add_slide(BLANK)
tb = s.shapes.add_textbox(Inches(1.0), Inches(3.0), Inches(11.3), Inches(1.5))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
r = p.add_run(); r.text = "谢谢各位老师，请批评指正！"
set_font(r, size=36, bold=True, color=DARK)

prs.save("结题答辩PPT_最终版.pptx")
print("PPT 已生成：结题答辩PPT_最终版.pptx")
