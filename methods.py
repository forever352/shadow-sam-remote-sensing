"""
阴影检测方法，统一接口：输入 BGR 图像，输出 0/255 二值掩膜。

方法清单：
- mask_otsu        传统 Otsu 阈值法（基线）
- mask_c3i         LAB 空间 C3I 阴影指数（传统基线）
- mask_sam_amg     SAM 自动分割 + 暗度筛选（零样本基线）
- mask_enhanced    Otsu + CLAHE 增强 + 形态学清理（增强的物理粗提取）
- mask_ours        我们的方法：增强粗提取 + 方向一致性过滤（方向先验）
"""
import cv2
import numpy as np
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator


def _rm_small(mask, min_area):
    """去掉面积 <= min_area 的连通域，返回 0/255 掩膜。"""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            out[labels == i] = 255
    return out


def mask_otsu(img_bgr):
    """方法1：灰度反相 + Otsu 自动阈值。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(255 - gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _rm_small(mask, 150)


def mask_c3i(img_bgr, invert_if_mostly_white=True):
    """方法2：LAB 阴影指数 SI = (255-L) + (255-B)，编码"阴影又暗又偏蓝"。

    [!] `invert_if_mostly_white`（默认 True）是**本项目自加的启发式**，**不是 C3I 原方法的一部分**：
        当 Otsu 结果前景占比 >40% 时整体取反（假定"阴影不该过半"）。
        第十三轮审查实测（无泄漏测试集 92 张）：该启发式触发 **61 张**，
        去掉后 **61/61 全部变好、0 张变差** —— 在当前数据上 **100% 有害**。
        故它**不得**被当作 C3I 方法本身的表现；基线评估应传 `False`。
        默认仍保持 `True`，是为了让既有调用方（`run_experiment.py` 等）及其历史产物
        **逐位可复现** —— 不得为了让基线数字好看而静默改默认值。
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, _, b = cv2.split(lab)
    si = (255.0 - l) + (255.0 - b)
    si_norm = cv2.normalize(si, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    si_f = cv2.medianBlur(si_norm, 5)
    _, mask = cv2.threshold(si_f, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert_if_mostly_white and np.sum(mask == 255) > mask.size * 0.4:
        mask = cv2.bitwise_not(mask)
    return _rm_small(mask, 150)


def mask_sam_amg(img_bgr, amg, dark_ratio=0.7):
    """
    方法3：SAM 自动分割（零样本）+ 暗度筛选。
    只保留"暗"的区域作为阴影候选，模拟"不带物理先验的零样本 SAM"基线。
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    anns = amg.generate(rgb)
    h, w = img_bgr.shape[:2]
    combined = np.zeros((h, w), bool)
    img_mean = gray.mean()
    for a in anns:
        seg = a["segmentation"]
        if seg.sum() == 0:
            continue
        if gray[seg].mean() < img_mean * dark_ratio:
            combined |= seg
    return (combined * 255).astype(np.uint8)


def mask_enhanced(img_bgr):
    """方法4：Otsu + CLAHE 对比度增强 + 形态学开运算（增强的物理粗提取）。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, mask = cv2.threshold(255 - enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return mask


def _direction_filter(mask, dev_thresh=25, min_area=300):
    """
    方向一致性过滤：保留主轴方向与全局主导方向一致的连通域。
    阴影朝向由太阳方位决定、具有一致性；道路/水体/植被方向杂乱，据此剔除误检。
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    angs, comps = [], []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        ys, xs = np.where(labels == i)
        if len(xs) < 5:
            continue
        rect = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        angs.append(rect[2])
        comps.append(i)
    if not angs:
        return mask
    # 圆形中值：把 [0,90) 角度翻倍到 [0,180) 求中值，避免 0/90 周期性断裂
    a = np.deg2rad(angs) * 2
    med = np.rad2deg(np.arctan2(np.median(np.sin(a)), np.median(np.cos(a)))) / 2 % 90
    keep = np.zeros_like(mask)
    for i, ang in zip(comps, angs):
        dev = min(abs(ang - med), 90 - abs(ang - med))
        if dev <= dev_thresh:
            keep[labels == i] = 255
    return keep


def mask_ours(img_bgr):
    """方法5：我们的方法 = 增强粗提取 + 方向一致性过滤（方向先验）。"""
    return _direction_filter(mask_enhanced(img_bgr))
