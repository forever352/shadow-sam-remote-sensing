# 基于方向提示与 SAM 的遥感影像阴影检测

> **一句话**：把阴影的物理先验（又暗、又偏蓝、方向一致）与「分割一切」大模型 SAM 结合，实现**全自动、无需人工点选**的遥感影像阴影检测。

- 团队：ShadowHunter
- 负责人：潘俊豪 · 指导教师：祁昆仑（副教授）
- 数据集：[AISD](https://github.com/xw-hu/UnShadowNet)（航空影像阴影检测公开数据集）
- 核心成果：**参数高效微调 SAM**，仅训练 4.34% 参数，把阴影检测 IoU 从 0.234 提升到 **0.717**

---

## 一、项目简介

阴影是遥感影像中的常见干扰，会遮挡地物细节、扭曲光谱信息，直接影响土地分类、变化检测、目标识别等下游任务的精度。传统阴影检测依赖人工调阈值、适应性差；纯深度学习方法又依赖大量人工标注、成本高。

本项目的思路是「**物理先验引导 + 基础模型适配**」：

1. **物理粗提取**先回答「阴影大概在哪」——利用阴影「暗、偏蓝」的光谱特性做粗分割；
2. **自动提示生成**把粗分割结果转成 SAM 能理解的点/框提示；
3. **参数高效微调 SAM** 让它回答「切得多准」——冻结图像编码器，只训练少量参数，让通用大模型适配遥感阴影场景。

---

## 二、技术方案

```
输入遥感影像
   │
   ▼
① 增强物理粗提取：CLAHE 对比度增强 → Otsu 阈值 → 形态学去噪
   │  （得到高召回率的候选阴影区域）
   ▼
② 自动提示生成：候选区域连通域质心/包围框 → 前景点 + 背景点
   │
   ▼
③ 参数高效微调 SAM：冻结图像编码器，仅训练掩膜解码器 + 提示编码器
   │  （4.06M 参数，仅占 93.7M 的 4.34%）
   ▼
输出精细阴影掩膜
```

其中「方向先验」是项目标题的核心，当前以**手写几何规则**（`methods.py` 的 `_direction_filter`，用连通域主轴方向做一致性过滤）近似实现，**学习式的自监督方向感知网络是后续计划**（详见「已知问题」）。

---

## 三、实验结果（AISD Test51 全量 51 张平均）

| 方法 | IoU ↑ | BER ↓ | F1 ↑ | Precision | Recall |
|------|------:|------:|-----:|----------:|-------:|
| Otsu（传统阈值） | 0.424 | 0.204 | 0.581 | 0.429 | 0.979 |
| C3I（LAB 阴影指数） | 0.162 | 0.606 | 0.211 | 0.168 | 0.296 |
| SAM-AMG（零样本 SAM） | 0.302 | 0.343 | 0.442 | 0.705 | 0.353 |
| 零样本 SAM + 自动提示 | 0.234 | 0.430 | 0.370 | 0.273 | 0.628 |
| 增强预处理（Otsu+CLAHE+形态学） | 0.512 | 0.149 | 0.662 | 0.533 | 0.940 |
| 增强 + 方向过滤（手写规则） | 0.471 | 0.201 | 0.619 | 0.550 | 0.782 |
| **微调 SAM + 框提示（本项目方法）** | **0.717** | **0.109** | **0.833** | **0.842** | 0.833 |

**核心结论**：

1. **参数高效微调是最有效的成果**：在完全相同的自动提示下，微调后的 SAM 把 IoU 从零样本的 0.234 提升到 0.717（+0.483），BER 从 0.430 降到 0.109，四项指标全部最优。
2. **增强物理预处理稳定有效**：CLAHE + 形态学让传统 Otsu 的 IoU 从 0.424 提升到 0.512。
3. **手写方向过滤存在权衡**：提升 Precision 但损失 Recall，净 IoU 略降，说明方向应当用学习式方法估计（后续工作）。

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

> 所有脚本都用 `Path(__file__).resolve().parent` 定位项目根目录，因此**从任意目录运行都不会找不到权重**（前提是脚本路径正确）。下面统一假设你在项目根目录（内层 `RemoteSensing-Shadow-SAM-main`）下。

### 5.1 单张影像演示（微调模型）

```bash
python demo.py                       # 检测默认测试图 austin13_sub2.tif
python demo.py 你的图片路径          # 检测任意图片，输出 demo_result.png
```

### 5.2 微调模型全量评估（Test51）

```bash
python evaluate_finetuned.py --checkpoint checkpoints/sam_vit_b_shadow_v2.pth   # 点提示
python final_eval.py                                                            # 框提示（0.717 结果）
```

### 5.3 传统 / 零样本方法批量对比

```bash
python run_experiment.py             # 5 种方法全量对比，输出 results/summary.csv
```

### 5.4 提示策略对比（点/框/TTA/迭代）

```bash
python prompt_experiment.py          # 对比不同推理策略的 IoU
```

### 5.5 重新训练（微调 SAM）

```bash
python train_adapter_v2.py           # 150 epoch，RTX 4060 约 6 小时
```

---

## 六、目录结构

```
RemoteSensing-Shadow-SAM-main/
├── methods.py                 # 5 种阴影检测方法（Otsu/C3I/SAM-AMG/Enhanced/Ours）
├── eval_shadow.py             # 指标计算（IoU/BER/F1/OA/Precision/Recall）
├── evaluate_finetuned.py      # 微调模型评估（点提示）
├── final_eval.py              # 微调模型评估（框提示，最终结果）
├── prompt_experiment.py       # 提示策略对比实验
├── run_experiment.py          # 批量方法对比，输出 summary.csv
├── train_adapter.py           # 微调脚本 v1（30 epoch）
├── train_adapter_v2.py        # 微调脚本 v2（150 epoch，最终版）
├── demo.py                    # 单图演示
├── visualize.py               # 生成对比可视化图
├── interactive_sam.py         # 交互式分割 Demo（鼠标点选）
├── shadow_pipeline.py         # 早期单图流水线 Demo
├── c3i_analyzer.py            # 早期 C3I 阴影指数实验
├── boxes_tta_full.py          # 框提示 + TTA 测试
├── metrics_chart.py           # 生成指标柱状图
├── make_ppt.py                # 生成结题答辩 PPT
├── env_setup.py               # Windows torch DLL 修复（勿删）
├── requirements.txt           # 依赖清单
│
├── checkpoints/               # 模型权重
│   ├── sam_vit_b_01ec64.pth           # 官方 SAM ViT-B 权重
│   └── sam_vit_b_shadow_v2.pth        # 微调后权重（核心成果）
├── AISD/                      # 数据集（Train412/Val51/Test51）
├── results/                   # 实验结果（summary.csv / per_image.csv / figures/）
│
└── segment-anything-main/     # SAM 官方源码（pip install -e 安装这个目录）
```

> 注：内层还有一个 `segment-anything/` 目录，是早期实验残留（`GPU_text.py`、`test_sam.py` 等），**不是** SAM 源码，可忽略。

---

## 七、已知问题与后续计划

1. **自监督方向感知网络未实现**：申报书规划的学习式方向估计（ResNet-18 自监督）尚未完成，当前方向先验为手写几何规则，效果有限。
2. **未与 SOTA 方法对比**：尚未对比 U-Net、ShadowFormer、Mask R-CNN 等。
3. **学术论文 / 软件著作权未产出**。

后续计划：实现学习式方向估计、在编码器内嵌 Adapter（参考 RSPrompter）、补充 SOTA 对比、封装为工具包 / QGIS 插件。
