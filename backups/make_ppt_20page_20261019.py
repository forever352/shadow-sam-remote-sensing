# -*- coding: utf-8 -*-
"""生成结题答辩 PPT（16:9，20 页）。

叙事：问题 -> 思路 -> 方法管线 -> 分阶段证据 -> 主结果 -> 消融 ->
跨数据集/误差 -> 多方法可视对比 -> 方向诊断 -> 软件原型 -> 优劣边界 -> 结论。

口径纪律：
  - 主结果 = 无泄漏 68/16/21 划分，LoRA 报 3 seed 均值；基线均 3 seed
  - AISD 自带划分含同源泄漏，禁横比；不得用“官方”指代该划分
  - 只表述“数值上高于”，不宣称“显著超越/SOTA”
  - 负结果与限制如实列入，不隐藏、不捏造
"""
import os
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn

BASE = Path(__file__).resolve().parent
DARK = RGBColor(0x1F, 0x3B, 0x63)
ACCENT = RGBColor(0x2E, 0x74, 0xB5)
GRAY = RGBColor(0x55, 0x55, 0x55)
RED = RGBColor(0xA6, 0x3A, 0x2E)
LIGHT = RGBColor(0xDE, 0xEA, 0xF6)


def set_font(run, size=18, bold=False, color=None, name="Microsoft YaHei"):
    f = run.font
    f.size = Pt(size)
    f.bold = bold
    f.name = name
    if color is not None:
        f.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = rPr.makeelement(qn("a:ea"), {})
        rPr.append(ea)
    ea.set("typeface", name)


prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def title_bar(slide, text, sub=None):
    tb = slide.shapes.add_textbox(Inches(0.6), Inches(0.28), Inches(12.15), Inches(0.95))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = text
    set_font(r, size=28, bold=True, color=DARK)
    if sub:
        p2 = tf.add_paragraph()
        r2 = p2.add_run(); r2.text = sub
        set_font(r2, size=13, color=GRAY)


def add_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def add_footer(slide, text):
    tb = slide.shapes.add_textbox(Inches(0.6), Inches(7.12), Inches(12.15), Inches(0.3))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run(); r.text = text
    set_font(r, size=9, color=GRAY)


def add_bullets(tf, items, size=18, color=DARK):
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = it
        set_font(r, size=size, color=RED if it[:1] in ("!", "！") else color)
        p.space_after = Pt(9)


def content_slide(title, bullets, notes="", sub=None, size=18):
    s = prs.slides.add_slide(BLANK)
    title_bar(s, title, sub)
    tb = s.shapes.add_textbox(Inches(0.6), Inches(1.45), Inches(12.13), Inches(5.35))
    tf = tb.text_frame; tf.word_wrap = True
    add_bullets(tf, bullets, size=size)
    add_notes(s, notes)
    return s


def image_slide(title, image, bullets=None, notes="", sub=None, mode="full",
                caption=None, image_h=5.5):
    s = prs.slides.add_slide(BLANK)
    title_bar(s, title, sub)
    img_path = BASE / image
    if mode == "full":
        s.shapes.add_picture(str(img_path), Inches(0.6), Inches(1.35), width=Inches(12.13))
    else:
        if caption:
            image_h = min(image_h, 5.3)
        pic = s.shapes.add_picture(str(img_path), Inches(0.6), Inches(1.5), height=Inches(image_h))
        if pic.width > Inches(7.3):
            scale = Inches(7.3) / pic.width
            pic.width = Inches(7.3)
            pic.height = int(pic.height * scale)
        tb = s.shapes.add_textbox(Inches(8.05), Inches(1.5), Inches(4.7), Inches(5.3))
        tf = tb.text_frame; tf.word_wrap = True
        add_bullets(tf, bullets or [], size=16)
    if caption:
        cap_top = 6.95 if mode == "left" else 6.62
        tb = s.shapes.add_textbox(Inches(0.6), Inches(cap_top), Inches(12.1), Inches(0.42))
        tf = tb.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]
        r = p.add_run(); r.text = caption
        set_font(r, size=10, color=GRAY)
    add_notes(s, notes)
    return s


def table_slide(title, rows, widths=None, notes="", sub=None, note=None,
                row_h=0.5, top=1.55, best_row=1, size=13):
    s = prs.slides.add_slide(BLANK)
    title_bar(s, title, sub)
    n_r, n_c = len(rows), len(rows[0])
    w = Inches(12.13)
    left = Inches(0.6)
    shape = s.shapes.add_table(n_r, n_c, left, Inches(top), w, Inches(row_h * n_r))
    tbl = shape.table
    if widths:
        total = sum(widths)
        for j, wj in enumerate(widths):
            tbl.columns[j].width = Inches(12.13 * wj / total)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = str(val)
            pr = cell.text_frame.paragraphs[0]
            pr.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            is_head = (i == 0)
            is_best = (best_row is not None and i == best_row)
            set_font(pr.runs[0], size=size, bold=(is_head or is_best),
                     color=ACCENT if is_best else DARK)
            if is_best:
                cell.fill.solid(); cell.fill.fore_color.rgb = LIGHT
    if note:
        tb = s.shapes.add_textbox(Inches(0.6), Inches(top + row_h * n_r + 0.12),
                                  Inches(12.13), Inches(7.0 - top - row_h * n_r))
        tf = tb.text_frame; tf.word_wrap = True
        add_bullets(tf, note, size=10)
    add_notes(s, notes)
    return s


# 1. 封面
s = prs.slides.add_slide(BLANK)
tb = s.shapes.add_textbox(Inches(0.9), Inches(2.05), Inches(11.5), Inches(1.9))
tf = tb.text_frame; tf.word_wrap = True
r = tf.paragraphs[0].add_run()
r.text = "物理先验提示 + LoRA 微调 SAM：\n面向遥感影像的自动阴影检测"
set_font(r, size=34, bold=True, color=DARK)
tb2 = s.shapes.add_textbox(Inches(0.9), Inches(3.95), Inches(11.5), Inches(2.1))
tf2 = tb2.text_frame; tf2.word_wrap = True
cover_lines = [
    ("申报项目：基于方向提示与 SAM 的遥感影像阴影检测方法研究", 14),
    ("大学生创新创业训练计划项目 · 结题答辩", 18),
    ("团队：ShadowHunter　|　2026 年 10 月", 16),
]
for i, (t, sz) in enumerate(cover_lines):
    p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
    rr = p.add_run(); rr.text = t
    set_font(rr, size=sz, color=GRAY)
add_notes(s, "各位老师好，我是 ShadowHunter 团队代表。今天汇报的题目是：物理先验提示 + LoRA 微调 SAM 的遥感影像自动阴影检测。"
             "我会先讲问题和思路，再讲方法与证据，最后如实说明边界与不足。")

# 2. 背景
image_slide(
    "阴影检测：下游任务的“隐形干扰源”",
    "results/stages/chicago22_sub9/01_original.png",
    bullets=[
        "阴影遮挡地物细节、扭曲局部光谱，影响分类、变化检测、目标识别",
        "传统阈值法：逐场景调参，水体/深色植被/深色路面容易混淆",
        "全监督网络：像素级标注成本高；同源泄漏会让指标虚高",
        "SAM 边界分割强，但零样本并不懂“遥感阴影”",
    ],
    mode="left", image_h=5.1,
    caption="左：AISD 测试图 chicago22_sub9；建筑、树木与暗色地物同时存在。",
    notes="阴影是遥感影像中的常见干扰。传统方法靠阈值，深度方法靠标注，SAM 虽然强但零样本不懂阴影。"
          "本项目要解决的就是：如何用低成本把 SAM 变成可用的阴影检测器。",
)

# 3. 核心思路
content_slide(
    "核心思路：物理先验回答“在哪”，SAM 回答“切多准”",
    [
        "物理粗提取：CLAHE 增强 + Otsu 阈值 + 形态学 → 高召回候选阴影区",
        "自动提示：候选连通域包围框 → 框提示（不读取 GT，全自动）",
        "参数高效适配：冻结 SAM 编码器，LoRA 只训练约 4.8% 参数",
        "矢量交付：掩膜 → 多边形（含孔洞）→ GeoJSON / Shapefile",
        "一句话：用物理先验缩小搜索空间，用微调 SAM 精细边界",
    ],
    sub="不依赖人工点选，也不逐场景调阈值",
    notes="我们的思路是把任务拆成两步：物理先验负责高召回地找到阴影候选，SAM 负责精细边界。"
          "两者之间用自动框提示连接，全程不需要人工点选。",
)

# 4. 方法总览
s = prs.slides.add_slide(BLANK)
title_bar(s, "方法总览：五段式全自动流程")
s.shapes.add_picture(str(BASE / "results/figures/pipeline_trim.png"),
                     Inches(0.6), Inches(1.45), width=Inches(12.13))
tb = s.shapes.add_textbox(Inches(0.6), Inches(4.55), Inches(12.13), Inches(1.9))
tf = tb.text_frame; tf.word_wrap = True
add_bullets(tf, [
    "物理先验：CLAHE + Otsu + 形态学，得到高召回候选区域",
    "自动提示：连通域框提示，不读 GT；SAM 逐框分割后取并集",
    "参数高效：LoRA 仅训练 4.51M（4.8%），8GB 笔记本 GPU 可完成",
], size=16)
add_notes(s, "这是整体流程：输入影像，增强物理粗提取，连通域框提示，LoRA 微调 SAM，掩膜合并与矢量化。"
             "注意两个细节：提示不读 GT；可训练参数只有 4.51M，占总参数 4.8%。")
add_footer(s, "图：输入影像 → 增强粗提取 → 框提示 → LoRA-SAM → 阴影矢量。")

# 5. 阶段对比（粗提取）
image_slide(
    "从像素到候选：物理粗提取与 SAM 并集",
    "results/figures/stage_coarse.png",
    bullets=[
        "① 原图：输入影像",
        "② 灰度：保留结构",
        "③ CLAHE：增强暗部细节",
        "④ 反相 + Otsu：分离暗区",
        "⑤ Otsu 掩膜：二值候选",
        "⑥ 形态学开运算：去细小噪声",
        "⑦ 连通域框提示：把候选交给 SAM",
        "⑧ 逐框 SAM 并集：精细边界",
    ],
    mode="left", image_h=5.4,
    caption="图：AISD chicago22_sub9 的粗提取阶段。",
    notes="前六步都是传统图像处理：CLAHE 增强暗部，反相 Otsu 分离暗区，形态学去噪，最后取连通域包围框。"
          "粗提取的目标是高召回，边界交给 SAM。",
)

# 7. 阶段对比（SAM）
image_slide(
    "SAM 精修：把毛糙候选修成可用边界",
    "results/figures/stage_sam.png",
    bullets=[
        "逐框预测后取并集，得到精细边界",
        "最终叠加：绿色为预测阴影",
        "误差图：TP 绿 / FP 红 / FN 蓝",
        "边界比粗提取干净，但暗色地物仍会过检",
    ],
    mode="left", image_h=5.2,
    notes="这一页是 SAM 精修的结果：逐框预测取并集，再叠加到原图，最后用误差图看 TP/FP/FN。"
          "可以看到边界比粗提取干净很多，但植被和暗色地物仍会被并进来。",
    caption="图：最终叠加 → 误差图（TP 绿 / FP 红 / FN 蓝）。",
)

# 8. 框提示
image_slide(
    "自动框提示：把“候选”变成 SAM 能读的提示",
    "results/stages/chicago22_sub9/07_boxes.png",
    bullets=[
        "连通域面积 ≥ 300 像素才保留",
        "取轴对齐包围框，逐框送 SAM",
        "所有框的预测取并集",
        "提示只来自输入影像，不读 GT",
        "框提示同时编码位置与尺度，比点提示更稳",
    ],
    mode="left", image_h=5.2,
    caption="图：黄色框为自动生成的框提示；无需人工点选。",
    notes="框提示的关键是“不读 GT”。我们只用输入影像做增强粗提取，再取连通域包围框。"
          "这样既保证全自动，也避免真值泄漏。",
)

# 7. LoRA
image_slide(
    "LoRA 微调：4.8% 参数撬动 ViT-B",
    "results/figures/param_efficiency.png",
    bullets=[
        "编码器冻结，注意力层 qkv/proj 挂 rank=8 LoRA",
        "可训练：LoRA 0.44M + prompt encoder + mask decoder = 4.51M",
        "损失：Focal + Dice；点/框双提示；水平翻转；AMP",
        "全部训练在 RTX 4060 Laptop（8GB）完成",
        "低秩适配降低小样本过拟合风险",
    ],
    mode="left", image_h=4.6,
    caption="图：总参数 93.7M，可训练 4.51M（约 4.8%）。",
    notes="LoRA 是我们方法的核心工程点。全量微调在 8GB 卡上不现实，所以我们冻结编码器，只训练低秩矩阵、提示编码器和解码器。"
          "这样既省显存，也降低在 262 张训练图上的过拟合风险。",
)

# 8. 实验设置
table_slide(
    "实验设置：先修评估基座，再谈性能",
    [
        ["环节", "设置"],
        ["数据", "AISD（Train412/Val51/Test51）+ RSISD 跨数据集"],
        ["发现的泄漏", "Test51 的 43 个基名中 42 个在 Train412；51 张中 50 张同源"],
        ["主口径", "按原图基名重划 68/16/21 原图 = 262/58/92 张，三集不重叠"],
        ["指标", "IoU / BER / F1 / OA / P / R；macro；best-val 选模型"],
        ["基线", "U-Net / SegNet / Mask R-CNN / 冻结 SAM / 零样本 / 传统方法"],
    ],
    widths=[2.4, 8.8], best_row=None, row_h=0.62, top=1.5, size=14,
    sub="测试集只用于最终报告，不参与选模型",
    note=[
        "AISD 自带划分的数字只作历史对齐；同源泄漏会让 IoU 系统性偏高约 5 个点。",
        "所有深度方法在同一无泄漏划分上重训；逐图/场景配对 t 仅作描述性统计。",
    ],
    notes="实验设置这一页最重要的是“无泄漏划分”。我们发现 AISD 自带 Test51 和训练集大量重叠，"
          "所以按原始大图基名重划了 68/16/21，所有方法都在同一划分上重训。这是后面所有数字可信的前提。",
)

# 9. 主结果
table_slide(
    "主结果：无泄漏划分上数值上高于三个全监督基线",
    [
        ["方法", "IoU", "BER", "F1", "种子"],
        ["LoRA 微调 SAM（本项目）", "0.7990 ± 0.0101", "0.0612 ± 0.0027", "0.8869", "3 seed"],
        ["全监督 U-Net", "0.7547 ± 0.0028", "0.0860 ± 0.0091", "0.8567", "3 seed"],
        ["全监督 SegNet", "0.7382 ± 0.0044", "0.1003 ± 0.0020", "0.8448", "3 seed"],
        ["全监督 Mask R-CNN †", "0.7392 ± 0.0039", "0.0955 ± 0.0042", "0.8486", "3 seed"],
        ["冻结编码器 SAM（对照）", "0.5621", "0.1327", "0.7105", "单 seed"],
    ],
    widths=[3.7, 2.5, 2.4, 1.4, 1.2], best_row=1, row_h=0.55, top=1.45, size=13,
    sub="92 张 / 21 独立场景；best-val；macro",
    note=[
        "† Mask R-CNN 分辨率不对称仅“部分缓解”：输入仍 512×512 上采样到 800，主方法走 SAM 原图 1024 → 不得写“对比已公平”。",
        "统计限定：21 个独立场景；未做多重比较校正；seed-0 三口径 +0.005/+0.035/+0.050；3-seed macro +0.0443。",
        "证据强度：Mask R-CNN seed-0 clean_run 未验证；LoRA seed-1 为 --resume 续训；SAM 类多一个增强 Otsu 预检测器。",
        "表述：数值上高于，且 3-seed 配对检验支持该数值优势；不宣称“显著超越/SOTA”。",
    ],
    notes="这是主结果。LoRA 的 3-seed 均值是 IoU 0.799、BER 0.061；三个全监督基线分别是 0.755、0.738、0.739。"
          "我们只说“数值上高于”，不说显著超越，原因写在下面的限定里：21 个场景、没有多重比较校正、效应量随口径变化。",
)

# 10. 主结果图 + 配对
image_slide(
    "3-seed 配对检验：数值优势不是单个 seed 的偶然",
    "results/metrics_bar.png",
    bullets=[
        "LoRA vs U-Net：IoU +0.0443（图级 t=-5.39）",
        "LoRA vs SegNet：+0.0608（t=-6.71）",
        "LoRA vs Mask R-CNN：+0.0598（t=-13.34）",
        "场景级：+0.0498 / +0.0655 / +0.0602",
        "效应量随 pooled/macro/scene 口径变化，不升级为 SOTA",
    ],
    mode="left", image_h=4.9,
    caption="图：左＝无泄漏主结果；右＝AISD 自带划分（含泄漏，仅存档）；两面板禁止横比。",
    notes="为了说明不是单个 seed 的偶然，我们做了 3-seed 逐图配对。三个基线的 t 值分别是 -5.39、-6.71、-13.34。"
          "但效应量随口径变化，所以仍然只写“数值上高于”。",
)

# 11. 消融
table_slide(
    "消融：零样本 → 冻结编码器 → LoRA 的受控递进",
    [
        ["配置", "IoU", "BER", "F1", "说明"],
        ["零样本 SAM（同框提示）", "0.3480", "0.2923", "0.4997", "只差“是否微调”"],
        ["冻结编码器 + 训练提示/解码", "0.5621", "0.1327", "0.7105", "单 seed"],
        ["LoRA 微调 SAM（本项目）", "0.7990", "0.0612", "0.8869", "3 seed 均值"],
    ],
    widths=[3.6, 1.3, 1.3, 1.3, 2.5], best_row=3, row_h=0.62, top=1.55, size=14,
    sub="同一提示链路、同一划分、同一评估脚本",
    note=[
        "零样本 → LoRA：逐图配对 +0.4399 IoU（seed-0），92/92 张全部变好；3-seed 平均 +0.4510。",
        "冻结 → LoRA：LoRA 在 91/92 张更高。",
        "归因：4.51M 含 LoRA + prompt encoder + mask decoder，不是“只训 LoRA”。",
    ],
    notes="消融回答“增益从哪来”。零样本 0.348，冻结编码器 0.562，LoRA 0.799。"
          "零样本到 LoRA 是 92 张全部变好，冻结到 LoRA 是 91/92 张更高。说明关键不是换提示，而是让编码器适配遥感域。",
)

# 12. 跨数据集与误差
table_slide(
    "跨数据集与误差：哪里有提升，哪里仍会错",
    [
        ["项目", "结果", "必须说明"],
        ["RSISD test200 零样本迁移", "IoU 0.7842（n=200, seed-0）", "任务较易：Otsu 即 0.5711；禁读作“泛化好”"],
        ["误检 / 漏检", "FPR 0.0405 / FNR 0.0882", "漏检约为误检 2.2 倍，主误差在召回侧"],
        ["小连通域召回", "<256 px：0.538", "但按漏检像素量，>=4096 px 贡献 39.0%"],
        ["边界带 F1", "r=1/3/5：0.745 / 0.819 / 0.849", "边界是难点，但不是唯一主项"],
        ["误检成分（HSV）", "水样 8.3% / 植被 12.4% / 其它 79.3%", "启发式判类；V≥110 低估深色水体，8.3% 是下界"],
    ],
    widths=[2.8, 3.8, 5.0], best_row=None, row_h=0.62, top=1.5, size=12,
    sub="均为无泄漏口径下的附加分析；不同口径不可混读",
    notes="跨数据集 IoU 0.784，但这个任务本身容易，Otsu 都能到 0.571，所以不能读成泛化好。"
          "误差方面，漏检是误检的 2.2 倍；小阴影召回低，但按漏检像素量，大连通域贡献更多。误检成分只是 HSV 启发式，不能当真实地物。",
)

# 13. AISD 单图多方法对比
image_slide(
    "同一张 AISD 影像，6 个关键方法输出",
    "results/figures/compare_aisd_key.png",
    bullets=[
        "LoRA：IoU 0.676 / BER 0.081（该图 BER 最低）",
        "SegNet：IoU 0.677 / BER 0.123",
        "冻结编码器：0.345 / 0.167；零样本：0.107 / 0.479",
        "U-Net / Mask R-CNN：IoU 0.695，略高于 LoRA（见 CSV）",
        "n=1、seed-0、域内单图；不排序、不标 best",
        "这张图说明：植被/暗色地物仍会过检",
    ],
    mode="left", image_h=5.55,
    caption="图：chicago22_sub9（AISD 测试集）；单图数值只作陈列，不构成方法优劣结论。",
    notes="这是同一张 AISD 图上的 6 个关键方法。LoRA 的 BER 最低，但 IoU 不是最高；SegNet 在这张图上接近，U-Net 和 Mask R-CNN 更高。"
          "我们不做排序，因为 n=1。它主要用来看失败模式：植被和暗色地物仍会被并进阴影。",
)

# 14. RSISD 单图多方法对比
image_slide(
    "跨数据集水体场景：6 个关键方法都会过检",
    "results/figures/compare_rsisd_key.png",
    bullets=[
        "AISD 训练 → RSISD 测试；水体 + 植被 + 建筑阴影",
        "LoRA：IoU 0.604 / BER 0.086",
        "SegNet：IoU 0.647 / BER 0.087；冻结 0.384",
        "零样本：IoU 0.258 / BER 0.332",
        "所有方法都把大片暗色地物并入阴影",
        "n=1、跨数据集；不作泛化结论",
    ],
    mode="left", image_h=5.55,
    caption="图：GuiZhou_37（RSISD test200）；单图跨域案例，不作为泛化证据。",
    notes="这张是跨数据集的水体场景。可以看到所有方法都过检，把水体、植被并进阴影。"
          "这正是后续要解决的问题：暗色地物判别和跨传感器域适应。",
)

# 15. 方向模块
content_slide(
    "方向先验：系统诊断与负结果",
    [
        "四种注入：几何过滤 / 方向点提示 / 方向点训练 / 方向 Adapter",
        "同族无方向对照：IoU 差异 ≤0.0007",
        "相对框提示：+0.0158 来自提示族差异（点/扇形 vs 框），不是方向增益",
        "方向 Adapter：ΔBER −0.0038、ΔIoU −0.0005；随机方向对照复制同量级 ΔBER（t=−1.00/−0.11）",
        "方向回归头：28–29°，对照常数 29.83°、随机 45°；相对常数改善约 0.5–1.5°（取决于 run）",
        "标签一致性 R=0.23–0.37，随机基线 0.212；物理方位代理仅 +0.015",
        "结论：本框架下未观察到可测增益；不写“方向先验无效”，带四重局限",
    ],
    sub="方向模块未进入主结果：0.7990 的 headline 中不含任何方向通路",
    notes="方向模块是我们执行中最重要的一次“负结果”。四种注入方式都没有可测增益，随机方向对照也能复制 Adapter 的 BER 效应。"
          "我们如实报告，但不说“方向先验无效”，因为探针有线性的、全局池化的、样本量的、单划分的四重局限。",
)

# 16. 软件原型
image_slide(
    "从掩膜到矢量：一个可运行的软件原型",
    "results/stages/chicago22_sub9/09_final_overlay.png",
    bullets=[
        "输入影像 → 阴影多边形（GeoJSON / SHP，含孔洞）",
        "42 个多边形；往返面积差 0.0107%",
        "栅格化 IoU 0.8438，对照管线 0.8453",
        "地理参考：输出地图坐标 + CRS；无参考：像素坐标",
        "默认 seed-0 权重；验证限单图 + 合成 CRS",
    ],
    mode="left", image_h=4.8,
    caption="图：最终叠加示例；绿色为预测阴影。验证日志见 logs/verify_shadow_extract_*.log。",
    notes="工程交付方面，我们做了 shadow_extract.py：输入影像输出 GeoJSON 或 Shapefile。"
          "验证包括 GeoJSON、Shapefile 和地理参考三条分支；但目前只测了单张图和合成 CRS，真实多传感器还需要继续验证。",
)

# 17. 算法优劣
table_slide(
    "算法间优劣与适用边界",
    [
        ["方法", "优势", "局限 / 适用边界"],
        ["传统 Otsu / C3I", "无需训练、快", "阈值敏感，暗色地物易混淆"],
        ["U-Net / SegNet", "固定场景精度稳定", "需像素级标注，跨域弱"],
        ["Mask R-CNN", "实例级输出", "输入 512 → 内部 800；分辨率敏感"],
        ["零样本 SAM", "无需训练", "不懂阴影，IoU 0.348"],
        ["冻结编码器 SAM", "轻量", "编码器未适配，IoU 0.562"],
        ["LoRA-SAM（本项目）", "全自动、4.8% 参数、矢量输出", "21 场景、BER 6.12%、未复现 SOTA"],
        ["方向模块", "物理可解释", "本数据无可测增益"],
    ],
    widths=[2.8, 4.0, 4.8], best_row=6, row_h=0.56, top=1.4, size=12,
    sub="同一无泄漏划分下的定位；不是绝对排名",
    notes="这一页把各算法的优劣放在一起。我们的定位是：在中等规模、单数据集为主的条件下，用低成本实现全自动阴影检测和矢量交付；"
          "不是所有场景都最优，也不是 SOTA。",
)

# 18. 结论与产出
content_slide(
    "结论：一条可运行、主结果可复核的自动阴影检测链路",
    [
        "主结果：无泄漏 3-seed IoU 0.7990 / BER 0.0612；仅训练 4.8% 参数",
        "方法学：发现并修复同源泄漏与基线增强 bug；建立可复核评估基座",
        "完整对照：零样本 → 冻结 → LoRA；零样本→LoRA 92/92 张变好",
        "工程：输入影像 → 阴影矢量；端到端验证通过",
        "! 边界：BER<5% 主方法未达；21 场景；未复现 SOTA；方向模块负结果",
        "后续：论文/软著、跨传感器验证、暗色地物判别、工具包",
    ],
    sub="不宣称 SOTA；主结果数字可在 results/ 与 logs/ 中复核；软件验证限单图 + 合成 CRS",
    notes="最后总结：我们交付了一条可运行、主结果可复核的链路，主结果 0.799/0.061，只训 4.8% 参数；"
          "同时修复了评估基座的两个问题。边界也很清楚：BER 没到 5%，只有 21 个场景，没有 SOTA，方向模块是负结果。这些我们都如实写进了报告。",
)

# 19. 结束
s = prs.slides.add_slide(BLANK)
tb = s.shapes.add_textbox(Inches(1.0), Inches(2.9), Inches(11.3), Inches(1.5))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
r = p.add_run(); r.text = "谢谢各位老师，请批评指正！"
set_font(r, size=36, bold=True, color=DARK)
add_notes(s, "我的汇报到此结束，谢谢各位老师，请批评指正。")

out = BASE / os.environ.get("PPT_OUT", "结题答辩PPT_最终版.pptx")
prs.save(str(out))
print(f"PPT 已生成：{out}  共 {len(prs.slides)} 页")
