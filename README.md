# 基于方向提示与 SAM 的遥感影像阴影检测

> **一句话**：把阴影的物理先验（又暗、又偏蓝、方向一致）与「分割一切」大模型 SAM 结合，用 **LoRA 参数高效微调**实现**全自动、无需人工点选**的遥感影像阴影检测。

- 团队：ShadowHunter
- 负责人：潘俊豪 · 指导教师：祁昆仑（副教授）
- 数据集：[AISD](https://github.com/xw-hu/UnShadowNet)（航空影像阴影检测公开数据集，多尺度约 256~1688 像素）
- 核心成果：**LoRA 微调 SAM**，仅训练约 4.8% 参数，在**按原始大图重划的无泄漏划分**上把阴影检测 IoU 提升到 **0.788**（BER 0.064），数值上高于同划分下的全监督 U-Net（0.752）/ SegNet（0.740）

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

## 三、实验结果

> **⚠️ 口径声明（重要）**：AISD **AISD 自带 Test51 与训练集存在同源泄漏**（Test51 的 43 个原图基名中 42 个也在 Train412），指标系统性偏高。本项目已重建评估基座：**按原始大图基名整组重划为 68/16/21 原图（262/58/92 张），三集互不重叠**，主结果以该无泄漏划分报告。

### 3.1 主表 —— 无泄漏划分（92 张 / 21 个独立场景，best-val 口径）

**(a) 学习方法（需训练）**

| 方法 | IoU ↑ | BER ↓ | F1 ↑ | OA | Precision | Recall |
|------|------:|------:|-----:|-----:|----------:|-------:|
| **LoRA 微调 SAM（本项目方法）** | **0.7878** | **0.0644** | **0.8796** | **0.9513** | 0.8546 | 0.9118 |
| 全监督 U-Net | 0.7525 | 0.0771 | 0.8552 | 0.9426 | 0.8345 | 0.8909 |
| 全监督 SegNet | 0.7403 | 0.1021 | 0.8466 | 0.9439 | 0.8855 | 0.8222 |
| 全监督 Mask R-CNN †（主行：min800/max1333） | **0.7382** | **0.0968** | 0.8481 | 0.9411 | 0.8571 | 0.8424 |
| 　└ 同模型 · 敏感性（min512/max800） | 0.7330 | 0.0965 | 0.8443 | 0.9392 | 0.8459 | 0.8460 |

**(b) 传统阈值 / 粗提取方法（无训练，同 92 张测试集）**

| 方法 | IoU ↑ | BER ↓ | 性质 |
|------|------:|------:|------|
| 增强粗提取（Otsu+CLAHE+形态学） | 0.4837 | 0.1537 | **本项目自研粗提取模块**（主方法内部模块），**非外部对照** |
| C3I（LAB 阴影指数） | 0.4564 | 0.1650 | 传统阴影指数基线（已去除本项目自加启发式） |
| Otsu（传统阈值） | 0.4051 | 0.1969 | 传统阈值法基线 |
| C3I（含本项目反转启发式） | 0.1765 | 0.5712 | **仅供对照**，非 C3I 固有性能 |

> U-Net 为**完整 60 轮** from-scratch 训练，best-val = Epoch 30（val IoU 0.7840）。
> 三行均为**单 seed**，**未做显著性检验**，测试集仅 **21 个独立场景**，领先幅度（U-Net/SegNet 约 0.046，Mask R-CNN 0.055）**小于逐图标准差** → 仅表述为"**数值上高于**"，不宣称显著。
> **† 分辨率不对称（必须一并声明）**：主行已改用 **`min800/max1333`**（torchvision 默认）的复跑结果。但**复跑只缓解了一半**——本次仅把**模型内部** min/max 提到默认，**输入仍固定 `cv2.resize(..., (512,512))`**（把已降采样的 512 图**再上采样**到 800，**不恢复原始细节**），未跑完整默认 pipeline；主方法走 SAM 的 `ResizeLongestSide(1024)`（作用于**原图**）。**相对主方法的 1024 不对称仍在**，属**部分缓解**，**不得写成"偏置已消除/对比已公平"**。差值由 0.0548 收到 **0.0496**，只解释约 9%。且该 +0.0052 **落在逐图噪声内**（t≈1.70、**46 胜 46 负**）→ 只能写"数值上略高"。
> **C3I 两行必须同列**：`methods.py` 内置一段**本项目自加**的">40% 则反转"启发式（非 C3I 原方法），实测在 92 张上触发 61 张且**去掉后 61/61 全部变好** → 主行用去除启发式的 0.4564，末行 0.1765 仅供对照。
> 本表用自建划分，**不能与文献 AISD 报数直接横比**。证据：`results/maskrcnn_leakfree_test.csv`、`results/{otsu,c3i,enhanced}_leakfree_test.csv`；审查见 `review_log.md` 第十三、十四轮。

**核心结论**：修复基线脚本的增强 bug 并去除同源泄漏后，LoRA 微调 SAM 在**统一口径**下数值上仍高于全监督 CNN 基线 → "优于全监督"不是泄漏造成的假象。

### 3.2 附表 —— AISD 自带划分（含同源泄漏，仅作历史对齐，禁横比）

| 方法 | IoU ↑ | BER ↓ | F1 ↑ | 标注 |
|------|------:|------:|-----:|------|
| **LoRA 微调 SAM（本项目）** | 0.8393 | 0.0501 | 0.9122 | ⚠️ 泄漏（共享 42/43 基名） |
| 全监督 U-Net ‡ | 0.8153 | 0.0640 | 0.8971 | ⚠️ 含泄漏；增强 bug 已修，完整 60 轮 |
| 全监督 SegNet ‡ | 0.8089 | 0.0706 | 0.8939 | ⚠️ 含泄漏；增强 bug 已修，完整 60 轮 |
| 冻结编码器微调 SAM | 0.717 | 0.109 | 0.833 | 旧 run |
| 增强预处理（Otsu+CLAHE+形态学） | 0.512 | 0.149 | 0.662 | 旧 run |
| SAM-AMG（零样本） | 0.302 | 0.343 | 0.442 | 旧 run |
| Otsu（传统阈值） | 0.424 | 0.204 | 0.581 | 旧 run |
| C3I（LAB 阴影指数） | 0.162 | 0.606 | 0.211 | 旧 run |

> ‡ 早期 U-Net/SegNet 脚本存在**图像与掩膜增强错位 bug**（各自独立抽签），系统性压低 CNN 基线；该 bug 已在无泄漏版脚本中修复。
> ⚠️ **口径混排警示（2026-10-15 起）**：本表 Otsu / C3I / 增强 / SAM-AMG 四行是**早期在自带划分的泄漏测试集上的跑数**；这些方法**已在无泄漏 92 张上重评**，新数字见 3.1 主表 (b)。**两套口径不可混引**——引用主方法对比一律用主表 (b)，本表仅作历史对齐。

**补充结论**：同一 LoRA 方法，AISD 自带划分（含泄漏）划分 0.8393 vs 无泄漏划分 0.7878，**相差 0.052** —— 实证了泄漏使指标偏高约 5 个 IoU 点。
**方向先验的探索结论（已更新为首轮审查后的口径）**：模块②「方向引导提示」的**全对照差异 ≤ 0.0007，落在噪声内**（`results/dirprompt_*_oracle_test.csv`）；方向提示**不如框提示**。**「方向未取得可测增益」不适用于"方向先验无效"这类表述**——未试过的路径（方向 Adapter 注入解码器、射线投射采样）至今悬空（详见 3.x / `进度接力.md` §四 禁用词清单）。

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
python evaluate_lora.py              # LoRA 自带 Test51 口径评估（含泄漏，IoU 0.839）
python train_lora_leakfree.py        # LoRA 无泄漏重训（主结果口径）
python train_cnn_leakfree.py --arch unet    # U-Net 无泄漏基线
python train_cnn_leakfree.py --arch segnet  # SegNet 无泄漏基线
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
├── evaluate_lora.py           # LoRA 评估（自带划分的泄漏口径 0.839）
├── train_lora_leakfree.py     # LoRA 无泄漏重训（主结果口径 0.788）
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
│   ├── sam_vit_b_shadow_v2.pth                  # 冻结编码器微调权重（0.717）
│   ├── sam_vit_b_shadow_v4_lora_leakfree.pth    # LoRA 无泄漏权重（0.788，主结果）
│   ├── sam_vit_b_shadow_v4_lora_last.pth        # LoRA 自带划分的泄漏口径权重（0.839）
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
