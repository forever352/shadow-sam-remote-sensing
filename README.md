# 基于方向提示与 SAM 的遥感影像阴影检测

> **一句话**：把阴影的物理先验（又暗、又偏蓝、方向一致）与「分割一切」大模型 SAM 结合，用 **LoRA 参数高效微调**实现**全自动、无需人工点选**的遥感影像阴影检测。

- 团队：ShadowHunter
- 负责人：潘俊豪 · 指导教师：祁昆仑（副教授）
- 数据集：[AISD](https://github.com/xw-hu/UnShadowNet)（航空影像阴影检测公开数据集，多尺度约 256~1688 像素）
- 核心成果：**LoRA 微调 SAM**，仅训练约 4.8% 参数，把阴影检测 IoU 从 0.234 提升到 **0.839**，超过全监督 U-Net（0.805）和 SegNet（0.780）

---

## 一、项目简介

阴影是遥感影像中的常见干扰，会遮挡地物细节、扭曲光谱信息，直接影响土地分类、变化检测、目标识别等下游任务的精度。传统阴影检测依赖人工调阈值、适应性差；纯深度学习方法又依赖大量人工标注、成本高。

本项目的思路是「**物理先验引导 + 基础模型适配**」：

1. **物理粗提取**先回答「阴影大概在哪」——利用阴影「暗、偏蓝」的光谱特性做粗分割；
2. **自动提示生成**把粗分割结果转成 SAM 能理解的框提示；
3. **LoRA 参数高效微调 SAM** 让它回答「切得多准」——在图像编码器注意力层挂 LoRA，冻结原权重、只训练少量参数，让通用大模型适配遥感阴影场景。

---

## 二、技术方案

```
输入遥感影像
   │
   ▼
① 增强物理粗提取：CLAHE 对比度增强 → Otsu 阈值 → 形态学去噪
   │  （得到高召回率的候选阴影区域）
   ▼
② 自动提示生成：候选区域连通域包围框 → 框提示
   │
   ▼
③ LoRA 参数高效微调 SAM：图像编码器注意力层挂 LoRA（低秩适配）
   │  + 训练掩膜解码器 + 提示编码器（共 4.51M，约 4.8% 参数）
   ▼
输出精细阴影掩膜
```

其中「方向先验」是项目标题的核心，项目系统探索了三种注入方式（几何过滤、方向点提示、方向点训练），诚实结论是方向信息在框提示框架下收益有限，需学习式估计进一步探索（详见「已知问题」）。

---

## 三、实验结果（AISD Test51 全量 51 张平均）

| 方法 | IoU ↑ | BER ↓ | F1 ↑ | OA | Precision | Recall |
|------|------:|------:|-----:|-----:|----------:|-------:|
| Otsu（传统阈值） | 0.424 | 0.204 | 0.581 | 0.693 | 0.429 | 0.979 |
| C3I（LAB 阴影指数） | 0.162 | 0.606 | 0.211 | 0.447 | 0.168 | 0.296 |
| SAM-AMG（零样本 SAM） | 0.302 | 0.343 | 0.442 | 0.815 | 0.705 | 0.353 |
| 增强预处理（Otsu+CLAHE+形态学） | 0.512 | 0.149 | 0.662 | 0.798 | 0.533 | 0.940 |
| 冻结编码器微调 SAM | 0.717 | 0.109 | 0.833 | — | 0.842 | 0.833 |
| 全监督 SegNet | 0.780 | 0.086 | 0.875 | 0.948 | 0.902 | 0.854 |
| 全监督 U-Net | 0.805 | 0.066 | 0.891 | 0.953 | 0.891 | 0.898 |
| **LoRA 微调 SAM（本项目方法）** | **0.839** | **0.050** | **0.912** | **0.961** | **0.895** | **0.932** |

**核心结论**：

1. **LoRA 参数高效微调是最核心的成果**：零样本 SAM 只有 0.234 → 冻结编码器微调 0.717（+0.48）→ LoRA 微调 **0.839**（累计 +0.605），并**超过全监督 U-Net 和 SegNet**。BER 降至 0.050，几乎精确命中申报书"BER<5%"目标。
2. **接近监督 SOTA**：当前监督 SOTA（SARU，2026 ISPRS）在 AISD 上 IoU 为 0.864，本项目 0.839 已接近，且本项目 BER（0.050）优于 SARU（其论文自认 AISD 上 BER 非最佳）。
3. **方向先验的诚实结论**：方向提示优于无方向提示（+0.024），但不如框提示，说明方向需学习式估计（后续工作）。

---

## 四、环境配置

### 4.1 依赖安装

```bash
# 1. 创建独立环境（推荐）
conda create -n shadow-sam python=3.13
conda activate shadow-sam

# 2. 按你的 CUDA 版本安装 PyTorch（本项目为 CUDA 12.8）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. 安装其余依赖
pip install numpy opencv-python matplotlib

# 4. 从项目自带源码安装 segment-anything（关键，SAM 不在 PyPI）
pip install -e ./segment-anything-main

# 5. （可选）生成 PPT 才需要
pip install python-pptx
```

### 4.2 Windows 注意事项（重要）

本项目在 Windows + Python 3.13 + torch 2.11 上会碰到两个已知问题：

1. CUDA DLL（`caffe2_nvrtc.dll`）依赖找不到；
2. numpy(MKL) 与 torch 的 OpenMP 冲突（`libiomp5md.dll` 重复加载）。

**所有需要加载 torch 的脚本第一行都 `import env_setup` 来修复这两个问题**，不要删掉这一行。

---

## 五、快速开始

> 所有脚本都用 `Path(__file__).resolve().parent` 定位项目根目录，因此**从任意目录运行都不会找不到权重**。下面统一假设你在项目根目录（内层 `RemoteSensing-Shadow-SAM-main`）下。

### 5.1 单张影像演示（LoRA 微调模型，默认）

```bash
python demo.py                       # 检测默认测试图，输出 demo_result.png
python demo.py --image 你的图片路径  # 检测任意图片
python demo.py --checkpoint checkpoints/sam_vit_b_shadow_v2.pth   # 回退到冻结编码器版
```

### 5.2 微调模型全量评估（Test51）

```bash
python evaluate_lora.py              # LoRA 微调模型，完整六项指标（IoU 0.839 / BER 0.050）
python final_eval.py                 # 冻结编码器版框提示（0.717）
```

### 5.3 传统 / 零样本方法批量对比

```bash
python run_experiment.py             # 多种方法全量对比，输出 results/summary.csv
```

### 5.4 重新训练

```bash
python train_adapter_v4.py           # LoRA 微调（核心方法，AMP + resume 续训）
python train_adapter_v2.py           # 冻结编码器微调（对比基线）
python train_unet.py                 # U-Net 基线
python train_segnet.py               # SegNet 基线
```

---

## 六、目录结构

```
RemoteSensing-Shadow-SAM-main/
├── methods.py                 # 5 种阴影检测方法（Otsu/C3I/SAM-AMG/Enhanced/Ours）
├── eval_shadow.py             # 指标计算（IoU/BER/F1/OA/Precision/Recall）
├── evaluate_lora.py           # LoRA 微调模型评估（最终结果 0.839）
├── evaluate_finetuned.py      # 冻结编码器版评估（点提示）
├── final_eval.py              # 冻结编码器版评估（框提示）
├── direction_experiment.py    # 方向提示消融实验
├── prompt_experiment.py       # 提示策略对比实验
├── run_experiment.py          # 批量方法对比，输出 summary.csv
├── train_adapter_v4.py        # LoRA 微调（核心，AMP + resume）
├── train_adapter_v2.py        # 冻结编码器微调（对比基线）
├── train_unet.py              # U-Net 基线训练
├── train_segnet.py            # SegNet 基线训练
├── demo.py                    # 单图演示（支持 --checkpoint 切换权重）
├── visualize.py               # 生成对比可视化图
├── interactive_sam.py         # 交互式分割 Demo（鼠标点选）
├── boxes_tta_full.py          # 框提示 + TTA 测试
├── metrics_chart.py           # 生成指标柱状图
├── make_ppt.py                # 生成结题答辩 PPT
├── env_setup.py               # Windows torch DLL 修复（勿删）
├── requirements.txt           # 依赖清单
├── legacy/                    # 早期/弃用脚本（v1、方向训练失败版等）
│
├── checkpoints/               # 模型权重
│   ├── sam_vit_b_01ec64.pth               # 官方 SAM ViT-B 权重
│   ├── sam_vit_b_shadow_v2.pth            # 冻结编码器微调权重（0.717）
│   ├── sam_vit_b_shadow_v4_lora_last.pth  # LoRA 微调权重（0.839，核心成果）
│   ├── unet_shadow.pth                    # U-Net 权重
│   └── segnet_shadow.pth                  # SegNet 权重
├── AISD/                      # 数据集（Train412/Val51/Test51）
├── results/                   # 实验结果（summary.csv / per_image.csv / figures/）
└── segment-anything-main/     # SAM 官方源码（pip install -e 安装这个目录）
```

> 注：内层还有一个 `segment-anything/` 目录，是早期实验残留，**不是** SAM 源码，可忽略（已被 git 忽略）。

---

## 七、已知问题与后续计划

1. **自监督方向感知网络未实现**：申报书规划的学习式方向估计（自监督网络）尚未完成。当前对方向先验的研究以手写几何规则 + 提示工程为主，结论是方向信息在框提示框架下收益有限，需学习式估计进一步探索。
2. **学术论文 / 软件著作权未产出**：软著说明书已成稿、申请推进中；论文撰写中。
3. **方向 Adapter 未实现**：LoRA 微调已兑现"Adapter 微调"的主体，但"方向特征 Adapter"（方向向量注入解码器）尚未实现。

后续计划：实现自监督方向估计、方向特征 Adapter、更大规模数据集验证、封装为工具包 / QGIS 插件。
