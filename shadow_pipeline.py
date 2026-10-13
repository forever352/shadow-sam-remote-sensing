import env_setup  # 必须在 import torch 之前，修复 Windows 上 CUDA DLL 加载问题
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor

# --- 1. 配置区 ---
# 用脚本自身所在目录作为基准，避免"从哪个目录运行"导致路径找不到
BASE_DIR = Path(__file__).resolve().parent
SAM_CHECKPOINT = str(BASE_DIR / "checkpoints" / "sam_vit_b_01ec64.pth")
MODEL_TYPE = "vit_b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_PATH = str(BASE_DIR / "test_image.jpg")


# --- 2. 物理感知流：LAB粗取掩膜 ---
def get_rough_mask(img):
    # 1. 转换为灰度图并增强对比度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用自适应直方图均衡化，把暗部的细节抠出来
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # 2. 这里的逻辑反转：我们要找的是“极暗”区域
    # 我们用 255 减去亮度，这样阴影就变成了高亮值
    inverted_gray = 255 - enhanced_gray

    # 3. 使用 Otsu 自动阈值，让计算机自己找“暗”和“亮”的分界线
    _, binary_mask = cv2.threshold(inverted_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. 【核心改进】：形态学开运算，去掉细小的路面杂质，保留大块阴影
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # 5. 【面积筛选】：只要不是特别小的碎点，都保留
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
    new_mask = np.zeros_like(binary_mask)

    count = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        # 只要面积大于 150 像素的块都保留，不设上限（防止把大阴影滤掉）
        if area > 150:
            new_mask[labels == i] = 255
            count += 1

    print(f"系统分析：检测到 {count} 个候选阴影连通域。")
    return new_mask


# --- 3. 提示生成流：自动点采样 ---
def get_auto_prompts(mask):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    total_area = mask.shape[0] * mask.shape[1]
    points = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        # --- 策略升级：不仅看面积，还看形状 ---
        # 用包围盒长宽比过滤细长条：马路/标线通常又细又长，长宽比很大；
        # 阴影块通常比较"方正"，长宽比接近 1
        w_box = stats[i, cv2.CC_STAT_WIDTH]
        h_box = stats[i, cv2.CC_STAT_HEIGHT]
        aspect_ratio = max(w_box, h_box) / min(w_box, h_box)

        # 过滤条件：面积不能太大，且形状不能太细长（长宽比大于3的通常是马路或标线）
        if 200 < area < (total_area * 0.03) and aspect_ratio < 3.0:
            cx, cy = centroids[i]
            points.append([cx, cy])

    print(f"最终过滤后：保留了 {len(points)} 个高置信度阴影中心点。")
    return np.array(points)


# --- 4. 主流水线 ---
def run_pipeline():
    # A. 加载模型
    print("正在初始化模型...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)

    # B. 读取图片
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"错误：找不到图片 {IMAGE_PATH}，请确认文件已放到项目根目录。")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    # C. 物理提取与自动布点
    print("正在执行物理感知与自动布点...")
    rough_mask = get_rough_mask(image)
    auto_points = get_auto_prompts(rough_mask)

    if len(auto_points) == 0:
        print("未检测到显著阴影区域。")
        return

    # D. SAM 自动化预测
    print(f"正在进行 SAM 细化分割（共 {len(auto_points)} 个引导点）...")
    # 逐点预测：SAM 会把多个提示点当作同一个目标，逐点才能分割出多个独立阴影
    combined_mask = np.zeros(image_rgb.shape[:2], dtype=bool)
    for pt in auto_points:
        masks, scores, _ = predictor.predict(
            point_coords=np.array([pt]),
            point_labels=np.array([1]),
            multimask_output=False,
        )
        combined_mask |= masks[0]

    # E. 最终结果展示
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.title("Step 1: Physical Rough Extraction & Auto Prompts")
    plt.imshow(rough_mask, cmap='gray')
    plt.scatter(auto_points[:, 0], auto_points[:, 1], c='red', s=20, label='Auto Prompts')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Step 2: SAM Final Refinement")
    final_img = image_rgb.copy()
    mask_overlay = np.zeros_like(final_img)
    mask_overlay[combined_mask] = [0, 255, 0]  # 绿色代表最终识别
    plt.imshow(cv2.addWeighted(final_img, 0.7, mask_overlay, 0.3, 0))
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_pipeline()