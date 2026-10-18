"""生成结题报告/答辩 PPT 使用的图表（可复现）。

输入：
  - results/stages/chicago22_sub9/*.png（由 shadow_extract_stages.py 生成）
  - results/visual_compare/AISD_chicago22_sub9_masks/*.png（由 visual_compare_aisd.py 生成）
  - results/visual_compare/GuiZhou_37_masks/*.png（由 visual_compare_methods.py 生成）
  - results/AISD_chicago22_sub9_metrics.csv / GuiZhou_37_metrics.csv

输出：
  - results/figures/pipeline.png / pipeline_trim.png
  - results/figures/param_efficiency.png
  - results/figures/stage_coarse.png / stage_sam.png
  - results/figures/compare_aisd_key.png / compare_rsisd_key.png
  - logs/make_report_figures_<ts>.log

用法：
    python shadow_extract_stages.py --image AISD/AISD/Train412/shadow/chicago22_sub9.tif \
        --gt AISD/AISD/Train412/mask/chicago22_sub9.tif --out-dir results/stages/chicago22_sub9
    python visual_compare_aisd.py
    python visual_compare_methods.py
    python make_report_figures.py
"""
import csv
import hashlib
import os
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
FIG = BASE / "results" / "figures"
FONT = os.environ.get("REPORT_FONT", "C:/Windows/Fonts/msyh.ttc")
if not Path(FONT).exists():
    FONT = None


def get_font(size):
    if FONT:
        try:
            return ImageFont.truetype(FONT, size)
        except OSError:
            pass
    return ImageFont.load_default()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_pipeline():
    fig, ax = plt.subplots(figsize=(16, 5.2), dpi=220)
    ax.set_xlim(0, 16); ax.set_ylim(0, 5.2); ax.axis("off")
    boxes = [
        (0.3, 2.2, 2.1, 1.2, "输入影像\nGeoTIFF / PNG / JPG", "#E8F1FA"),
        (2.9, 2.2, 2.5, 1.2, "增强物理粗提取\nCLAHE → Otsu → 形态学", "#DCEBFA"),
        (5.9, 2.2, 2.2, 1.2, "连通域框提示\n（不读取 GT）", "#DCEBFA"),
        (8.6, 2.2, 2.7, 1.2, "LoRA 微调 SAM\nViT-B，仅训练 4.8% 参数", "#CFE0F5"),
        (11.8, 2.2, 1.9, 1.2, "阴影掩膜\n逐像素二分类", "#E8F1FA"),
        (14.0, 2.2, 1.7, 1.2, "阴影矢量\nGeoJSON / SHP", "#E8F1FA"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    linewidth=1.6, edgecolor="#2E74B5", facecolor=color))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=12.5, color="#1F3B63")
    for x in [2.4, 5.4, 8.1, 11.3, 13.5]:
        ax.add_patch(FancyArrowPatch((x, 2.8), (x + 0.45, 2.8), arrowstyle="-|>",
                                     mutation_scale=18, linewidth=1.8, color="#2E74B5"))
    ax.text(8.0, 0.75, "物理先验回答“阴影在哪里”，微调后的 SAM 回答“边界切多准”；全流程无需人工点选",
            ha="center", va="center", fontsize=13, color="#444444")
    ax.text(9.95, 3.75, "可训练参数 4.51M / 总参数 93.7M", ha="center", va="center", fontsize=11, color="#A63A2E")
    fig.savefig(FIG / "pipeline.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    im = Image.open(FIG / "pipeline.png").convert("RGB")
    arr = np.array(im)
    mask = (arr < 250).any(axis=2)
    ys, xs = np.where(mask)
    box = (max(xs.min() - 20, 0), max(ys.min() - 20, 0),
           min(xs.max() + 20, arr.shape[1]), min(ys.max() + 20, arr.shape[0]))
    im.crop(box).save(FIG / "pipeline_trim.png")


def make_param_chart():
    fig, ax = plt.subplots(figsize=(7.5, 3.4), dpi=240)
    labels = ["总参数", "可训练参数"]
    vals = [93.7, 4.51]
    colors = ["#9DB7D5", "#2E74B5"]
    bars = ax.barh(labels, vals, color=colors, height=0.55)
    for b, v in zip(bars, vals):
        ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v:.2f} M",
                va="center", fontsize=12, color="#1F3B63")
    ax.set_xlim(0, 110)
    ax.set_xlabel("参数量（百万）", fontsize=11)
    ax.set_title("LoRA 微调：约 4.8% 参数参与训练", fontsize=13, color="#1F3B63", pad=10)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(FIG / "param_efficiency.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def overlay(bgr, mask):
    ov = bgr.copy()
    ov[mask] = (0, 255, 0)
    return cv2.addWeighted(bgr, 0.6, ov, 0.4, 0)


def panel(img_bgr, label, sub="", cell=480):
    im = cv2.resize(img_bgr, (cell, cell), interpolation=cv2.INTER_AREA)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(im)
    bar = 62
    canvas = Image.new("RGB", (cell, cell + bar), "white")
    canvas.paste(pil, (0, bar))
    d = ImageDraw.Draw(canvas)
    f = get_font(20)
    fs = get_font(15)
    d.text((8, 6), label, fill=(31, 59, 99), font=f)
    if sub:
        d.text((8, 34), sub, fill=(90, 90, 90), font=fs)
    d.rectangle([0, 0, cell - 1, cell + bar - 1], outline=(180, 180, 180), width=2)
    return canvas


def montage(panels, cols, out, title):
    rows = (len(panels) + cols - 1) // cols
    cell_w, cell_h = panels[0].width, panels[0].height
    top = 56
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h + top), "white")
    d = ImageDraw.Draw(sheet)
    ft = get_font(26)
    d.text((12, 10), title, fill=(31, 59, 99), font=ft)
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        sheet.paste(p, (c * cell_w, top + r * cell_h))
    sheet.save(out)


def make_stage_montages():
    stage = BASE / "results" / "stages" / "chicago22_sub9"
    need = ["01_original", "02_gray", "03_clahe", "04_inverted", "05_otsu",
            "06_morph_open", "07_boxes", "08_sam_union", "09_final_overlay", "10_error_map"]
    missing = [n for n in need if not (stage / f"{n}.png").exists()]
    if missing:
        raise SystemExit(f"缺少阶段图 {missing}；先运行 shadow_extract_stages.py")
    imgs = {n: cv2.imread(str(stage / f"{n}.png")) for n in need}
    coarse = [
        panel(imgs["01_original"], "① 原图", "输入影像"),
        panel(imgs["02_gray"], "② 灰度", "保留结构"),
        panel(imgs["03_clahe"], "③ CLAHE", "增强暗部"),
        panel(imgs["04_inverted"], "④ 反相 + Otsu", "分离暗区"),
        panel(imgs["05_otsu"], "⑤ Otsu 掩膜", "二值候选"),
        panel(imgs["06_morph_open"], "⑥ 形态学开运算", "去细小噪声"),
        panel(imgs["07_boxes"], "⑦ 连通域框提示", "交给 SAM"),
        panel(imgs["08_sam_union"], "⑧ 逐框 SAM 并集", "精细边界"),
    ]
    montage(coarse, 4, FIG / "stage_coarse.png", "物理粗提取：从影像到 SAM 并集")
    sam = [
        panel(imgs["09_final_overlay"], "⑨ 最终叠加", "绿色为预测阴影"),
        panel(imgs["10_error_map"], "⑩ 误差图", "TP 绿 / FP 红 / FN 蓝"),
    ]
    montage(sam, 2, FIG / "stage_sam.png", "SAM 精修与误差定位")


def read_metrics(path):
    with open(path, encoding="utf-8-sig") as f:
        return {r["method"]: (float(r["IoU"]), float(r["BER"]))
                for r in csv.DictReader(f)}


def key_compare(img_path, gt_path, mask_dir, metrics_path, out, title):
    bgr = cv2.imread(str(img_path))
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    gt_rgb = cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB)
    metrics = read_metrics(metrics_path)
    files = {"SAM-zero": "SAM-zero", "Frozen-SAM": "Frozen-SAM",
             "LoRA-SAM": "LoRA-SAM", "U-Net": "U-Net",
             "SegNet": "SegNet", "Mask R-CNN": "Mask_R-CNN"}
    masks = {k: cv2.imread(str(Path(mask_dir) / f"{v}.png"), cv2.IMREAD_GRAYSCALE) > 0
             for k, v in files.items()}
    ps = [
        panel(bgr, "输入影像", ""),
        panel(gt_rgb, "真值", "参考"),
        panel(overlay(bgr, masks["SAM-zero"]), "零样本 SAM",
              f"IoU {metrics['SAM-zero'][0]:.3f} / BER {metrics['SAM-zero'][1]:.3f}"),
        panel(overlay(bgr, masks["Frozen-SAM"]), "冻结编码器 SAM",
              f"IoU {metrics['Frozen-SAM'][0]:.3f} / BER {metrics['Frozen-SAM'][1]:.3f}"),
        panel(overlay(bgr, masks["LoRA-SAM"]), "LoRA-SAM（本项目）",
              f"IoU {metrics['LoRA-SAM'][0]:.3f} / BER {metrics['LoRA-SAM'][1]:.3f}"),
        panel(overlay(bgr, masks["U-Net"]), "U-Net",
              f"IoU {metrics['U-Net'][0]:.3f} / BER {metrics['U-Net'][1]:.3f}"),
        panel(overlay(bgr, masks["SegNet"]), "SegNet",
              f"IoU {metrics['SegNet'][0]:.3f} / BER {metrics['SegNet'][1]:.3f}"),
        panel(overlay(bgr, masks["Mask R-CNN"]), "Mask R-CNN",
              f"IoU {metrics['Mask R-CNN'][0]:.3f} / BER {metrics['Mask R-CNN'][1]:.3f}"),
    ]
    montage(ps, 4, out, title)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = BASE / "logs" / f"make_report_figures_{ts}.log"
    log_path.parent.mkdir(exist_ok=True)
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log("# 生成结题报告/答辩 PPT 图表")
    log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"# 脚本: {Path(__file__).resolve()}  SHA256: {sha256(Path(__file__).resolve())}")
    inputs = [
        BASE / "AISD/AISD/Train412/shadow/chicago22_sub9.tif",
        BASE / "AISD/AISD/Train412/mask/chicago22_sub9.tif",
        BASE / "datasets/RSISD/test200/shadow/GuiZhou_37.png",
        BASE / "datasets/RSISD/test200/mask/GuiZhou_37.png",
    ]
    for p in inputs:
        log(f"# 输入: {p}  SHA256: {sha256(p)}")
    stage_files = sorted((BASE / "results/stages/chicago22_sub9").glob("*.png"))
    mask_files = sorted((BASE / "results/visual_compare/AISD_chicago22_sub9_masks").glob("*.png")) + \
        sorted((BASE / "results/visual_compare/GuiZhou_37_masks").glob("*.png"))
    metric_files = [
        BASE / "results/visual_compare/AISD_chicago22_sub9_metrics.csv",
        BASE / "results/visual_compare/GuiZhou_37_metrics.csv",
    ]
    for p in stage_files + mask_files + metric_files:
        log(f"# 中间输入: {p}  SHA256: {sha256(p)}")

    log("[1/4] pipeline + param chart")
    make_pipeline(); make_param_chart()
    log("[2/4] stage montages")
    make_stage_montages()
    log("[3/4] AISD key compare")
    key_compare(BASE / "AISD/AISD/Train412/shadow/chicago22_sub9.tif",
                BASE / "AISD/AISD/Train412/mask/chicago22_sub9.tif",
                BASE / "results/visual_compare/AISD_chicago22_sub9_masks",
                BASE / "results/visual_compare/AISD_chicago22_sub9_metrics.csv",
                FIG / "compare_aisd_key.png",
                "AISD 单图：6 个关键方法 + 输入/真值（n=1，不排序）")
    log("[4/4] RSISD key compare")
    key_compare(BASE / "datasets/RSISD/test200/shadow/GuiZhou_37.png",
                BASE / "datasets/RSISD/test200/mask/GuiZhou_37.png",
                BASE / "results/visual_compare/GuiZhou_37_masks",
                BASE / "results/visual_compare/GuiZhou_37_metrics.csv",
                FIG / "compare_rsisd_key.png",
                "RSISD 跨数据集单图：6 个关键方法 + 输入/真值（n=1，不作泛化结论）")
    log("\n# 输出")
    out_lines = []
    for name in ["pipeline.png", "pipeline_trim.png", "param_efficiency.png",
                 "stage_coarse.png", "stage_sam.png",
                 "compare_aisd_key.png", "compare_rsisd_key.png"]:
        p = FIG / name
        h = sha256(p)
        out_lines.append(f"| `{p.relative_to(BASE)}` | {h} |")
        log(f"  {p}  SHA256: {h}")
    prov = BASE / "results" / "figure_provenance.md"
    prov.write_text(
        "# 报告/PPT 图表 provenance\n\n"
        f"- 生成脚本：`make_report_figures.py`  \n  SHA256：`{sha256(Path(__file__).resolve())}`\n"
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        "## 输入（SHA256）\n\n"
        + "\n".join(f"- `{p.relative_to(BASE)}`：`{sha256(p)}`" for p in inputs + stage_files + mask_files + metric_files)
        + "\n\n## 输出（SHA256）\n\n| 文件 | SHA256 |\n|---|---|\n"
        + "\n".join(out_lines)
        + "\n\n> 说明：matplotlib 生成的 PNG 字节哈希可能随字体缓存/环境变化，但图的内容与数据不变；"
        "本文件记录的是本次生成时的哈希。stage/compare 图依赖 `shadow_extract_stages.py` 与 "
        "`visual_compare_*` 的中间产物，其输入哈希已列在上方。\n",
        encoding="utf-8",
    )
    log(f"# provenance: {prov}")
    log(f"# 日志: {log_path}")
    logf.close()


if __name__ == "__main__":
    main()
