# -*- coding: utf-8 -*-
"""生成 7 月 / 9 月进度表（沿用 5 月表版式），并导出 PDF。"""
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

OUT = Path(__file__).resolve().parent
PROJECT = "基于方向提示与 SAM 的遥感影像阴影检测方法研究"


def set_run_font(run, size=10.5, bold=False, underline=False):
    run.font.name = "宋体"
    run.font.size = Pt(size)
    run.bold = bold
    run.underline = underline
    r = run._element
    rPr = r.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = rPr.makeelement(qn("w:rFonts"), {})
        rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), "宋体")


def add_para(cell, text="", bold=False, size=10.5, first_indent=True,
             space_after=4, align=None, underline=False):
    p = cell.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.25
    if first_indent and text:
        p.paragraph_format.first_line_indent = Pt(21)
    if align is not None:
        p.alignment = align
    r = p.add_run(text)
    set_run_font(r, size=size, bold=bold, underline=underline)
    return p


def add_lead(cell, lead, body, size=10.5):
    p = cell.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.25
    p.paragraph_format.first_line_indent = Pt(21)
    r1 = p.add_run(lead)
    set_run_font(r1, size=size, bold=True)
    r2 = p.add_run(body)
    set_run_font(r2, size=size)


def build_month(month: int, completion: int, members: dict, self_eval: str, date_str: str):
    doc = Document()
    sec = doc.sections[0]
    sec.page_height = Cm(29.7)
    sec.page_width = Cm(21.0)
    sec.top_margin = Cm(2.2)
    sec.bottom_margin = Cm(2.2)
    sec.left_margin = Cm(2.4)
    sec.right_margin = Cm(2.4)

    style = doc.styles["Normal"]
    style.font.name = "宋体"
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    for text, bold, underline, size in [
        ("地理与信息工程学院 2025-2026 学年", True, False, 16),
        ("大学生创新创业训练计划项目", True, False, 16),
        (f"{month} 月进度表", True, True, 16),
    ]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run(text)
        set_run_font(r, size=size, bold=bold, underline=underline)

    rows = 2 + 1 + 1 + 1 + 1 + 1
    table = doc.add_table(rows=rows, cols=2)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.columns[0].width = Cm(3.4)
    table.columns[1].width = Cm(12.6)

    table.cell(0, 0).text = "项目名称"
    table.cell(0, 1).text = PROJECT
    table.cell(1, 0).text = "汇报月份"
    table.cell(1, 1).text = f"2026 年 {month} 月"
    for i in range(2):
        for j in range(2):
            for p in table.cell(i, j).paragraphs:
                p.paragraph_format.space_after = Pt(3)
                for r in p.runs:
                    set_run_font(r, size=10.5)

    # 成员工作
    c = table.cell(2, 0).merge(table.cell(2, 1))
    add_para(c, "本月各成员工作进展介绍：", bold=True, first_indent=False)
    for name, blocks in members.items():
        add_para(c, name, bold=True, first_indent=False, space_after=2)
        for lead, body in blocks:
            add_lead(c, lead, body)

    # 自评
    c = table.cell(3, 0).merge(table.cell(3, 1))
    add_para(c, "团队学生负责人对项目总体情况自评：", bold=True, first_indent=False)
    add_para(c, self_eval)
    add_para(c, f"该项目已完成计划（{completion}）％的工作量", bold=True, first_indent=False)

    # 教师评价
    c = table.cell(4, 0).merge(table.cell(4, 1))
    add_para(c, "老师对项目总体进展情况评价：", bold=True, first_indent=False)
    add_para(c, "（□优秀　　□良好　　□中等　　□合格　　□不合格）", first_indent=False)
    add_para(c, f"该项目已完成计划（{completion}）％的工作量", bold=False, first_indent=False)

    # 签名
    c = table.cell(5, 0).merge(table.cell(5, 1))
    for _ in range(5):
        add_para(c, "", first_indent=False, space_after=2)
    p = c.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(f"教师签名：　　　　　　时间：{date_str}")
    set_run_font(r, size=10.5)

    # 备注
    c = table.cell(6, 0).merge(table.cell(6, 1))
    add_para(c, "注：如果老师因各种原因无法签字，可在经过老师允许的情况下，由其研究生代签",
             first_indent=False, space_after=2)

    docx_path = OUT / f"D-37-{month}月进度收集表.docx"
    doc.save(str(docx_path))
    return docx_path


JULY = {
    "潘俊豪（负责人）：": [
        ("1. 工作进展：", "主导完成参数高效微调链路。在 SAM 图像编码器注意力层引入 LoRA，冻结原始权重，联合训练提示编码器与掩膜解码器；完成训练脚本、参数配置、训练日志与检查点管理，并在标准遥感数据集上完成首轮训练验证。"),
        ("", "搭建统一评测流程。实现 IoU、BER、F1、OA、Precision、Recall 六项指标，逐图结果与汇总结果统一落盘；固定“验证集选模型、测试集独立评估”的流程，保证不同方法在同一口径下比较。"),
        ("", "组织组会与任务拆分。按数据整理、基线复现、可视化、文档四条线分配任务；建立代码分支、提交信息与实验记录规范，推动组内按周同步进度。"),
        ("2. 遇到的困难及解决办法：", "困难：8GB 显存下全量微调不可行，训练容易中断。解决办法：采用冻结编码器 + LoRA + 混合精度 + 小 batch/梯度累积，优化数据加载与缓存；长任务使用断点续训队列，训练日志分段落盘。"),
        ("", "困难：训练过程存在波动，实验复现困难。解决办法：固定随机种子与数据划分，统一增强策略；完整记录配置、日志与检查点，确保结果可回溯。"),
        ("3. 个人主要收获与体会：", "掌握了基础模型参数高效微调与实验管理方法，形成了“先小规模冒烟验证、再全量运行”的工程习惯；在任务拆解、进度统筹和团队沟通方面得到锻炼。"),
    ],
    "吴箫：": [
        ("1. 工作进展：", "参与 U-Net、SegNet 等基线方法的复现与对比实验，整理数据划分、训练配置与评测脚本；在标准遥感数据集上协助完成多组对比实验并汇总结果。"),
        ("", "负责实验结果可视化。制作分割叠加图、指标对比图与典型案例图，为组会、阶段汇报和后续报告提供图表材料。"),
        ("", "参与文献调研。围绕遥感阴影检测、SAM 适配与参数高效微调整理相关文献，维护参考文献与术语表。"),
        ("2. 遇到的困难及解决办法：", "困难：不同方法的输入尺寸、后处理与指标口径不一致，结果难以横向比较。解决办法：统一输入协议与指标计算方式，逐图结果统一落盘，与负责人逐项核对口径。"),
        ("", "困难：实验图表信息密度高、可读性不足。解决办法：统一配色、标注与图注，按场景挑选代表性样本，分图展示。"),
        ("3. 个人主要收获与体会：", "加深了对评测口径与可复现性的理解，掌握了从实验运行到图表表达的完整链路；团队沟通与文档整理能力得到提升。"),
    ],
    "陈波丞：": [
        ("1. 工作进展：", "完成 Otsu、C3I、增强预处理等传统阴影提取方法的整理与批量测试，形成基线结果；参与数据清洗、样本筛选与测试集整理。"),
        ("", "协助搭建软件原型的输入输出流程：影像读取、掩膜后处理、结果保存；整理测试样例与运行记录，为后续矢量化输出做准备。"),
        ("", "学习并实践 Git 分支、提交与 Pull Request 流程，维护实验记录与代码说明。"),
        ("2. 遇到的困难及解决办法：", "困难：传统方法在复杂场景（水面、深色植被、路面）误检较多，参数敏感。解决办法：固定参数网格，在同一测试集上比较；整理失败样本供团队分析。"),
        ("", "困难：代码版本与实验结果对应关系容易混乱。解决办法：建立“脚本—参数—结果文件”命名规范与实验记录表，提交时同步说明改动。"),
        ("3. 个人主要收获与体会：", "掌握了传统方法与深度方法的差异，理解了数据、参数与结果三者的对应关系；工程规范意识和版本管理能力得到提升。"),
    ],
    "周杭：": [
        ("1. 工作进展：", "负责代码仓库与运行环境：整理目录结构、依赖清单与环境配置；建立分支管理与 Pull Request 合并流程，规范提交信息。"),
        ("", "优化训练与评测运行脚本：批量队列、断点续训、日志落盘；协助解决 Windows 环境、依赖与显存相关问题。"),
        ("", "参与推理后处理与结果检查，协助整理实验记录与运行说明。"),
        ("2. 遇到的困难及解决办法：", "困难：机器休眠或中断导致长任务失败。解决办法：设计线性阻塞队列与断点续训，日志分段落盘，任务前执行环境自检。"),
        ("", "困难：多人同时修改代码容易冲突。解决办法：约定分支命名、提交粒度与 PR 审查流程；关键脚本改动保留 patch 备份。"),
        ("3. 个人主要收获与体会：", "掌握了可复现实验的工程化方法，理解了“证据落盘”的重要性；团队协作与问题定位能力得到提升。"),
    ],
}

SEP = {
    "潘俊豪（负责人）：": [
        ("1. 工作进展：", "完成系统评测与对照实验。围绕零样本、冻结编码器、LoRA 微调三种配置开展受控对比；完成多 seed 重复实验与统计汇总，并补充跨数据集验证与场景级分析。"),
        ("", "完成误差结构与失败模式分析。按连通域面积、边界带与场景拆分误差，形成可视化结果与分析结论，为后续改进提供依据。"),
        ("", "牵头完成成果整理。完成结题报告、答辩 PPT、论文素材与软著说明书初稿；统一术语、图表与统计口径，组织组内交叉核对。"),
        ("", "组织代码与文档收尾。整理结果索引、复现命令、资产说明与最终检查清单，确保关键结果有落盘证据。"),
        ("2. 遇到的困难及解决办法：", "困难：实验组多、结果版本多，汇总容易出错。解决办法：统一结果目录与命名，建立汇总脚本；所有结论回溯到 CSV 与日志。"),
        ("", "困难：报告、PPT 与图表口径容易不一致。解决办法：固定指标定义与聚合方式，正文、图表、附录交叉核对，改动后全库检索旧说法。"),
        ("3. 个人主要收获与体会：", "形成了从实验设计、证据链到成果表达的完整闭环；对遥感阴影检测的难点、适用边界与可复现研究有了更系统的认识。"),
    ],
    "吴箫：": [
        ("1. 工作进展：", "完成误差结构分析的可视化：按连通域面积分箱、边界带与场景拆解误差；协助完成误检成分统计与结果整理。"),
        ("", "整理多方法单图对比与典型案例，参与结题报告和答辩 PPT 的图表设计与排版。"),
        ("", "协助核对结果数字与图表一致性，整理参考文献、附录材料与术语表。"),
        ("2. 遇到的困难及解决办法：", "困难：误差来源多、统计口径不统一。解决办法：统一 macro/pooled 口径与分箱标准，图表标注样本量与口径。"),
        ("", "困难：单图对比容易被过度解读。解决办法：明确“案例分析、不作排序”的表述边界，补充必要限定语。"),
        ("3. 个人主要收获与体会：", "掌握了误差分析与学术图表规范，理解了结论边界与可复现表达的重要性；在协作与细致核对方面得到锻炼。"),
    ],
    "陈波丞：": [
        ("1. 工作进展：", "完成软件原型：输入影像 → 阴影矢量（GeoJSON / Shapefile），实现孔洞保留、几何简化、面积过滤与地理参考处理。"),
        ("", "编写端到端验证脚本，完成 GeoJSON、Shapefile、地理参考三条分支测试与问题修复；整理测试样例与运行记录。"),
        ("", "整理自动化程序使用说明与资产获取说明；协助软著材料、代码清单与交付文档。"),
        ("2. 遇到的困难及解决办法：", "困难：矢量孔洞与面积计算同像素掩膜不一致。解决办法：采用 rasterio 多边形化与 shapely 几何处理，核对面积与栅格化 IoU，修复孔洞偏移。"),
        ("", "困难：地理参考坐标系处理复杂。解决办法：区分有/无 CRS 分支，明确坐标约定，用合成 CRS 样例验证写出路径。"),
        ("3. 个人主要收获与体会：", "掌握了遥感掩膜矢量化与 GIS 数据格式，理解了软件原型的工程细节与验证方法；工程实现与问题排查能力得到提升。"),
    ],
    "周杭：": [
        ("1. 工作进展：", "负责最终仓库整理：清理冗余权重与调试产物，修复脚本路径；整理代码索引、接手说明与复现文档。"),
        ("", "完成运行前自检、长任务队列与日志规范；协助生成最终图表与答辩 PPT，检查产物完整性。"),
        ("", "维护 GitHub 仓库：提交规范、版本记录与资产发布说明，整理对外文档与引用关系。"),
        ("2. 遇到的困难及解决办法：", "困难：仓库体积大、历史产物多。解决办法：按“证据 / 调试 / 备份”分类清理，保留主结果与可复现资产，为删除项留说明。"),
        ("", "困难：多人文档口径不一致。解决办法：建立文档索引与交叉引用检查，统一术语、数字与章节引用。"),
        ("3. 个人主要收获与体会：", "掌握了项目收尾与可复现交付方法，理解了版本管理与文档体系的价值；在工程整理与协作方面得到锻炼。"),
    ],
}

JULY_SELF = (
    "本月项目按计划完成从原型流水线到参数高效微调与系统评测的过渡。团队完成了 LoRA 微调链路、"
    "统一评测脚本与多组基线对比，建立了代码分支、实验记录与结果落盘规范；成员分工明确，协作顺畅。"
    "项目整体进度健康，已完成计划约 75% 的工作量。"
)
SEP_SELF = (
    "本月完成系统评测、误差分析、软件原型与成果整理。团队完成多组对照、多 seed 汇总、跨数据集验证与"
    "误差结构分析；交付“输入影像 → 阴影矢量”的软件原型及端到端验证；完成结题报告、答辩 PPT 与软著"
    "说明书初稿；代码、数据与文档完成统一整理。项目按计划完成主要研究内容，已完成计划约 95% 的工作量。"
)

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    p7 = build_month(7, 75, JULY, JULY_SELF, "2026.7.30")
    p9 = build_month(9, 95, SEP, SEP_SELF, "2026.9.30")
    print(p7)
    print(p9)
