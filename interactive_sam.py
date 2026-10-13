import env_setup  # 必须在 import torch 之前，修复 Windows 上 CUDA DLL 加载问题
import torch
import cv2
import numpy as np
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor

# --- 配置区 ---
# 用脚本自身所在目录作为基准，避免"从哪个目录运行"导致路径找不到
BASE_DIR = Path(__file__).resolve().parent
sam_checkpoint = str(BASE_DIR / "checkpoints" / "sam_vit_b_01ec64.pth")  # 确保文件名正确
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"
image_path = str(BASE_DIR / "test_image.jpg")  # 确保根目录下有这张图

# --- 初始化模型 ---
print(f"正在加载模型到 {device}...")
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# 读取并预处理图片
image = cv2.imread(image_path)
if image is None:
    print("错误：找不到图片，请检查文件名！")
    exit()
image_display = image.copy()
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)

# 存储点击的点
input_points = []
input_labels = []


def show_mask(mask, target_img):
    """将分割结果叠加到显示图像上"""
    color = np.array([0, 255, 0], dtype=np.uint8)  # 绿色
    mask_bool = mask.astype(bool)
    target_img[mask_bool] = target_img[mask_bool] * 0.5 + color * 0.5
    return target_img


def mouse_click(event, x, y, flags, param):
    """鼠标点击回调函数"""
    global image_display, input_points, input_labels

    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击：增加阴影点提示
        print(f"点击坐标: x={x}, y={y}")
        input_points.append([x, y])
        input_labels.append(1)  # 1 代表前景提示

        # 实时预测
        masks, scores, _ = predictor.predict(
            point_coords=np.array(input_points),
            point_labels=np.array(input_labels),
            multimask_output=False,
        )

        # 刷新显示
        temp_img = image.copy()
        image_display = show_mask(masks[0], temp_img)
        # 画出点击的红点
        for p in input_points:
            cv2.circle(image_display, tuple(p), 5, (0, 0, 255), -1)
        cv2.imshow("SAM Interactive - Press 'R' to reset, 'Q' to quit", image_display)


# --- 主循环 ---
cv2.namedWindow("SAM Interactive - Press 'R' to reset, 'Q' to quit")
cv2.setMouseCallback("SAM Interactive - Press 'R' to reset, 'Q' to quit", mouse_click)

print("\n操作说明:")
print("1. 鼠标左键点击图像中的阴影区域进行分割")
print("2. 按 'R' 键重置（清空点击的点）")
print("3. 按 'Q' 键退出")

while True:
    cv2.imshow("SAM Interactive - Press 'R' to reset, 'Q' to quit", image_display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):  # 退出
        break
    elif key == ord('r'):  # 重置
        input_points = []
        input_labels = []
        image_display = image.copy()
        print("重置成功！")

cv2.destroyAllWindows()