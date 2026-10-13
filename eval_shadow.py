"""
阴影检测定量评估模块
计算预测掩膜与真值掩膜之间的 IoU / BER / F1 / OA / Precision / Recall

指标说明：
- IoU (交并比): TP / (TP+FP+FN)，衡量分割边界贴合度，越接近 1 越好
- BER (平衡误差率): 0.5*(FPR+FNR)，综合前景/背景两类误差，越接近 0 越好（申报书核心指标）
- F1: 精确率与召回率的调和平均
- OA (总体精度): 分类正确像素占总像素比例
"""
import numpy as np


def compute_metrics(pred_mask, gt_mask):
    """
    计算两个二值掩膜之间的分割指标。

    参数:
        pred_mask: 预测掩膜 (H, W)，可为 bool / 0-1 / 0-255
        gt_mask:   真值掩膜 (H, W)，同上
    返回:
        dict: IoU / BER / F1 / OA / Precision / Recall
    """
    pred = np.asarray(pred_mask) > 0
    gt = np.asarray(gt_mask) > 0

    # 混淆矩阵的四个量
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()

    eps = 1e-7  # 防止除零

    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    # 平衡误差率 BER = 0.5 * (假阳性率 + 假阴性率)
    fpr = fp / (fp + tn + eps)
    fnr = fn / (fn + tp + eps)
    ber = 0.5 * (fpr + fnr)

    total = tp + fp + fn + tn
    oa = (tp + tn) / (total + eps)

    return {
        "IoU": float(iou),
        "BER": float(ber),
        "F1": float(f1),
        "OA": float(oa),
        "Precision": float(precision),
        "Recall": float(recall),
    }


def format_metrics(m):
    """把指标字典格式化成一行易读字符串。"""
    return "  ".join(f"{k}={v:.4f}" for k, v in m.items())


if __name__ == "__main__":
    # ---- 自测：验证指标计算是否正确 ----
    h, w = 10, 10
    gt = np.zeros((h, w), dtype=np.uint8)
    gt[2:6, 2:6] = 255  # 真值块 4x4 = 16 个前景像素

    # 场景1：预测与真值完全一致 → IoU 应为 1.0，BER 应为 0.0
    m_perfect = compute_metrics(gt, gt)
    print("场景1（完全一致）：", format_metrics(m_perfect))
    assert abs(m_perfect["IoU"] - 1.0) < 1e-6, "IoU 应等于 1.0"
    assert abs(m_perfect["BER"] - 0.0) < 1e-6, "BER 应等于 0.0"

    # 场景2：完全一致 + 额外 1 个误检像素
    pred = gt.copy()
    pred[8, 8] = 255  # 多检 1 个像素
    m_extra = compute_metrics(pred, gt)
    print("场景2（多1个误检）：", format_metrics(m_extra))
    # TP=16, FP=1, FN=0, TN=83; IoU=16/17≈0.941
    assert abs(m_extra["IoU"] - 16 / 17) < 1e-6, "IoU 应等于 16/17"

    # 场景3：全空预测 → 精确率/召回率应为 0
    m_empty = compute_metrics(np.zeros((h, w), np.uint8), gt)
    print("场景3（全空预测）：", format_metrics(m_empty))
    assert m_empty["Recall"] == 0.0

    print("\n✅ 自测全部通过，指标计算正确。")
