import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_shadow_v3(image_path):
    img = cv2.imread(image_path)
    if img is None: return

    # 1. 转换到 LAB 空间 (L代表明度，A红绿，B蓝黄)
    # 阴影在 L 通道极低，在 B 通道通常也较低（偏蓝）
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)

    # 2. 计算阴影指数：SI = (255 - L) + (255 - B)
    # 物理逻辑：阴影既要暗（L小），又要带有天光反射的蓝色（B小）
    # 这样可以完美避开白色的房顶和灰色的路面
    shadow_index = (255.0 - l) + (255.0 - b)

    # 3. 增强指数分布
    si_norm = cv2.normalize(shadow_index, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 使用中值滤波去除“椒盐”噪声（路面上的小碎块）
    si_filtered = cv2.medianBlur(si_norm, 5)

    # 4. 重点：使用自适应阈值或大津法（Otsu）
    _, binary_mask = cv2.threshold(si_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. 面积约束逻辑：如果识别出的“白色”太多，说明判定逻辑反了
    # 在正常城市影像中，阴影占比通常小于 40%
    if np.sum(binary_mask == 255) > (binary_mask.size * 0.4):
        binary_mask = cv2.bitwise_not(binary_mask)

    # 可视化展示
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1);
    plt.title("Original Image");
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB));
    plt.axis('off')
    plt.subplot(1, 3, 2);
    plt.title("New Shadow Index (LAB)");
    plt.imshow(si_norm, cmap='jet');
    plt.colorbar();
    plt.axis('off')
    plt.subplot(1, 3, 3);
    plt.title("Corrected Mask");
    plt.imshow(binary_mask, cmap='gray');
    plt.axis('off')
    plt.tight_layout();
    plt.show()

    return binary_mask


if __name__ == "__main__":
    extract_shadow_v3("test_image.jpg")