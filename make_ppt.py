# -*- coding: utf-8 -*-
"""生成结题答辩 PPT（16:9）。

口径纪律（与 结题报告.md / README.md 保持一致）：
  - 主结果 = 无泄漏划分，LoRA 报 3 seed 均值；三基线为单 seed
  - AISD 自带划分含同源泄漏，禁横比（注意：不得用"官方"指代 AISD 自带划分）
  - 只表述"数值上高于"，不宣称"显著超越 / SOTA"
  - 负结果（BER<5% 未达成、方向模块无可测增益）如实列入"不足"，不隐藏
运行：先跑 metrics_chart.py 生成 results/metrics_bar.png，再跑本脚本。
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn

DARK = RGBColor(0x1F, 0x3B, 0x63)
ACCENT = RGBColor(0x2E, 0x74, 0xB5)
GRAY = RGBColor(0x55, 0x55, 0x55)
RED = RGBColor(0xA6, 0x3A, 0x2E)


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
        color = RED if it[:1] in ("!", "！") else DARK
        set_font(r, size=size, color=color)
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


def table_slide(title, rows, widths=None, sub=None, note=None, row_h=0.55,
                top=1.9, best_row=1, note_size=13):
    """rows[0] 为表头；best_row 高亮行号（None 表示不高亮）。"""
    s = prs.slides.add_slide(BLANK)
    title_bar(s, title, sub)
    n_r, n_c = len(rows), len(rows[0])
    w = Inches(10.4)
    left = Inches((13.333 - 10.4) / 2)
    shape = s.shapes.add_table(n_r, n_c, left, Inches(top), w, Inches(row_h * n_r))
    tbl = shape.table
    if widths:
        total = sum(widths)
        for j, wj in enumerate(widths):
            tbl.columns[j].width = Inches(10.4 * wj / total)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = str(val)
            pr = cell.text_frame.paragraphs[0]
            pr.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            is_head = (i == 0)
            is_best = (best_row is not None and i == best_row)
            set_font(pr.runs[0], size=14 if j == 0 else 14,
                     bold=(is_head or is_best),
                     color=ACCENT if is_best else DARK)
            if is_best:
                cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor(0xDE, 0xEA, 0xF6)
    if note:
        tb = s.shapes.add_textbox(Inches(0.8), Inches(top + row_h * n_r + 0.18),
                                  Inches(11.7), Inches(7.35 - top - row_h * n_r - 0.18))
        tf = tb.text_frame; tf.word_wrap = True
        add_bullets(tf, note, size=note_size)
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
                       "团队：ShadowHunter"]):
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
    "结果：无泄漏划分上 IoU 0.799 / BER 0.061（3 seed 均值），"
    "数值上高于全监督 U-Net(0.752) / SegNet(0.740) / Mask R-CNN(0.738)",
])

# 6. 方法三
content_slide("方法 ③：自动提示生成", [
    "从增强粗提取结果取连通域质心作为前景点",
    "在候选区域外采样背景点，共同构成点提示",
    "实现全自动阴影检测，无需人工点选",
])

# 7. 实验设置
content_slide("实验设置", [
    "数据集：AISD；注意 AISD 自带 Test51 与训练集存在同源泄漏（43 基名中 42 个重叠）",
    "主口径：按原始大图重划 68/16/21 原图（262/58/92 张），三集互不重叠（无泄漏）",
    "模型：SAM ViT-B（358MB），NVIDIA RTX 4060（8GB）",
    "选模型：仅在验证集按 IoU 选 best-val，测试集只用于最终报告",
    "指标：IoU / BER / F1 / OA / Precision / Recall（逐图计算再平均）",
    "本项目主方法报 3 个随机种子的均值；对比基线均为单 seed（横比时须点明）",
])

# 8. 结果表
table_slide(
    "定量对比结果（无泄漏划分，92 张 / 21 独立场景，best-val）",
    [
        ["方法", "IoU", "BER", "F1", "性质"],
        ["LoRA 微调 SAM（本项目）", "0.7990 ± 0.0101", "0.0612", "0.8869", "3 seed 均值"],
        ["全监督 U-Net", "0.7525", "0.0771", "0.8552", "单 seed=0"],
        ["全监督 SegNet", "0.7403", "0.1021", "0.8466", "单 seed=0"],
        ["Mask R-CNN †", "0.7382", "0.0968", "0.8481", "单 seed=0"],
        ["冻结编码器版 SAM（对照）", "0.5621", "0.1327", "0.7105", "单 seed=0，未适配编码器"],
    ],
    widths=[3.2, 1.5, 1.0, 1.0, 1.6], row_h=0.46, top=1.7,
    sub="所有方法同一划分重训；未做显著性检验",
    note=[
        "† Mask R-CNN 分辨率不对称仅“部分缓解”：输入仍固定 512×512，主方法走 SAM 原图 1024 → 不得写“对比已公平”。"
        "差值 +0.0608 大于其逐图 std，旧论证失效，改用配对检验（t≈1.70、46 胜 46 负，未达显著）。",
        "统计限定：测试集仅 21 个独立场景；种子处理两侧不对称（本项目 3 seed vs 基线单 seed）。",
        "→ 仅表述为“数值上高于”，不宣称“显著超越 / SOTA”。",
        "AISD 自带（含泄漏）口径：LoRA 0.8393 / U-Net 0.8153 / SegNet 0.8089 —— 因含 42/43 同源泄漏，"
        "系统性偏高，禁与上表横比、不作干净泛化结论。",
    ],
)

# 9. 指标柱状图
s = prs.slides.add_slide(BLANK)
title_bar(s, "指标对比柱状图", "左＝无泄漏主结果（禁与右图横比）；右＝AISD 自带划分（含同源泄漏，仅存档）")
s.shapes.add_picture("results/metrics_bar.png", Inches(1.1), Inches(1.6), width=Inches(11.1))

# 10. 消融：零样本 → 冻结 → LoRA
table_slide(
    "消融：零样本 → 冻结编码器 → LoRA 微调（递进对照）",
    [
        ["配置", "IoU", "BER", "说明"],
        ["零样本 SAM（同提示，不微调）", "0.3480", "0.2923", "同一提示链路，仅去掉微调"],
        ["冻结编码器 + 训练提示/解码", "0.5621", "0.1327", "只训练提示编码器与解码器"],
        ["LoRA 微调 SAM（本项目）", "0.7990", "0.0612", "再解锁编码器注意力层低秩适配"],
    ],
    widths=[3.6, 1.2, 1.2, 3.4], best_row=3,
    sub="三层共用同一数据划分、同一提示链路、同一评估脚本 —— 干净单变量",
    note=[
        "零样本 → LoRA 的逐图配对比：+0.4399 IoU，92/92 张全部变好（无一张变差）。",
        "冻结 → LoRA：LoRA 在 91/92 张更高 —— 说明增益主要来自“让编码器适配遥感域”，而非只换提示。",
    ],
)

# 11. 跨数据集与误差构成
table_slide(
    "跨数据集验证与误差构成",
    [
        ["项目", "数值", "证据 / 限定"],
        ["AISD→RSISD 零样本迁移", "IoU 0.7842（n=200）", "该任务本身较易：传统 Otsu 即达 0.5711 → 禁读作“泛化好”"],
        ["误差构成（seed-0）", "FPR 0.0405 / FNR 0.0882", "漏检(FNR) 是误检(FPR) 的 2.2 倍 —— 主要误差是漏检"],
        ["误检成分（HSV 启发式判类）", "水样 8.3% / 植被样 12.4% / 其它 79.3%", "植被样有富集(lift 1.45)、水体样无(0.895)；79% 未被归类"],
    ],
    widths=[2.6, 2.6, 4.4], best_row=None,
    sub="均为无泄漏口径下、主方法的附加分析",
    note=[
        "误检成分基于 seed-0、且判类用的是本项目自写 HSV 启发式（水体判据 V≥110 漏深色水体，故 8.3% 是下界）；",
        "不使用“异物同谱”这一说法（现象真实存在，但现有数据不足以支撑该命名）。",
    ],
)

# 12. 申报书硬指标达成情况（如实）
table_slide(
    "申报书硬指标达成情况（如实报告）",
    [
        ["申报书承诺", "实际结果", "结论"],
        ["BER 降至 5% 以下", "无泄漏 6.12%（3 seed均值）/ 自带划分 5.01%", "未达成"],
        ["对比 SOTA 方法", "SARU-Net 未复现（仓库无网络定义代码）", "未完成，如实写明"],
        ["双流特征融合 / 射线投射采样", "双流未实现；射线投射仅在 oracle 口径跑过", "未完成，如实写明"],
    ],
    widths=[3.0, 4.2, 2.0], best_row=None,
    sub="申报书共 11 条落差，详见结题报告 §五",
    note=[
        "BER 已尝试补救两轮：① 提示框单变量网格（外扩 2%/5%/10%、min_area 300→100，4 候选）→ 无候选稳健优于基线，"
        "test 未评估任何候选；② 3-seed 集成 + 概率阈值扫描（val 选 B30）→ test BER 0.0644 → 0.0537，仍未达 5%。",
        "→ 如实报告未达成，不通过改口径、换测试集来“凑”指标。",
    ],
)

# 13. 方向模块探索（如实、克制）
content_slide("方向模块：探索过程与结论（如实）", [
    "已实现学习式方向的翻转自监督判别（ResNet-18，验证翻转准确率 0.99~1.00）；已实现方向 Adapter",
    "系统测试了 4 种注入方式：几何方向过滤 / 方向点提示 / 方向点训练 / 方向 Adapter 条件注入",
    "！四种方式在 oracle（给定真方向）上界口径下，与“不用方向”的 IoU 差异均 ≤ 0.0007（噪声内）",
    "！方向 Adapter：ΔBER −0.0038、ΔIoU −0.0005；随机方向训练的对照复制出同量级 ΔBER → 该效应不依赖于方向内容",
    "！方向回归头：测试 29.34° / 28.56°，对照常数预测器 29.83°、随机 45° —— 仅比“什么都不学”好约 1.5°",
    "！方向标签一致性：4 种估计量 R 均值 0.23~0.37（样本量匹配随机基线 0.212）；指向光源的物理代理仅 +0.015",
    "以上均为“本项目 AISD 实现 + 本套探针口径”下的诊断性结果，受线性 / 全局池化 / 样本量 / 单划分四重局限。",
], sub="方向模块未进入主结果：0.7990 的 headline 中不含任何方向通路")

# 14. 可视化对比
s = prs.slides.add_slide(BLANK)
title_bar(s, "分割效果可视化对比（示例）")
s.shapes.add_picture("results/figures/austin13_sub2.png", Inches(0.5), Inches(1.5), width=Inches(12.3))

# 15. 创新点与工程产出
content_slide("主要工作与产出", [
    "LoRA 参数高效微调：只训 4.8% 参数，无泄漏划分上数值上高于全监督 U-Net/SegNet/Mask R-CNN",
    "物理先验 + 基础模型融合，实现全自动阴影检测（无需人工点选）",
    "「零样本 → 冻结微调 → LoRA 微调」的完整递进对照（配对 +0.44，92/92 张）",
    "主动排查并修复 AISD 自带划分同源泄漏 + 基线增强 bug，重建可复核的评估基座",
    "交付可运行的软件原型：输入影像 → 输出阴影矢量（GeoJSON / Shapefile），含端到端自检脚本",
])

# 16. 不足与后续
content_slide("问题与不足、后续计划", [
    "! 申报书硬指标 BER<5% 未达成（无泄漏 6.12% / 自带划分 5.01%），已如实记录",
    "! 方向模块四种注入方式均无可测增益；最上游假设（翻转自监督是否编码方向）仍悬置",
    "! 对比基线为单 seed、测试集仅 21 个独立场景、未做显著性检验；Mask R-CNN 分辨率不对称仅部分缓解",
    "! 训练增强仅水平翻转；辐射/几何校正未做；双流融合、射线投射主链未实现（详见结题报告 §五 11 条）",
    "后续：学术论文撰写；软著申请；学习式方向估计的进一步验证；更大数据集与多传感器验证",
], sub="以上全部在结题报告 §五 逐条如实列出")

# 17. 结束
s = prs.slides.add_slide(BLANK)
tb = s.shapes.add_textbox(Inches(1.0), Inches(3.0), Inches(11.3), Inches(1.5))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
r = p.add_run(); r.text = "谢谢各位老师，请批评指正！"
set_font(r, size=36, bold=True, color=DARK)

prs.save("结题答辩PPT_最终版.pptx")
print("PPT 已生成：结题答辩PPT_最终版.pptx")
