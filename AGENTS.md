# AGENTS.md — 主智能体执行协议（第 2 版 · 2026-10-14）

## 项目背景
- 任务：基于方向先验的遥感影像阴影检测（SAM + 阴影方向自监督 + 方向引导提示 + 方向特征 Adapter）
- 数据：AISD（Train412/Val51/Test51，**共 514 张子图**；其中 **Train412 = 412 张、来自 105 张原始大图**，多尺度但**均为方图** 256~1688）
  + **RSISD**（跨数据集，`datasets/RSISD/`，train1000/val86/test200，二值 mask）
- **主口径划分**：按【原始大图基名】整组重划 **68/16/21 原图 = 262/58/92 张**，三集基名交集必须为 0
  （AISD AISD 自带划分存在同源泄漏：Test51 与 Train412 共享 **42/43** 基名，**禁用其作干净泛化结论**）
- 参考论文：祁昆仑等，2025，遥感学报《基于阴影方向先验的高分辨率遥感影像地物阴影检测》
  （PDF 已存 `references/`；其 DTICT 是本项目方向1 的关键对照）
- 核心方法：竖直翻转自监督 → DTICT（简化重构）→ 方向回归头 → 方向引导提示 → 方向特征 Adapter；下游 LoRA 微调 SAM
- 评估指标：IoU、BER、F1、OA、Precision、Recall（BER 为申报书核心，目标 <5%）

---

## ★ 监督协议（人类 2026-10-14 明确要求："时刻保持警惕"）

### 铁则
> **监督不是"出问题才找"，而是"每个动作前都过一遍"。**
> 项目周期将尽、GPU 时间是唯一瓶颈 —— **一次白跑 = 半天**。
> 因此宁可多审，不可漏审。作为主智能体，**无权自行判定"这步不用审"**。

### 何时必须请求监督（在动作【之前】）

| # | 时机 | 强制动作 |
|---|---|---|
| 1 | **启动任何 >1h 的 GPU 任务前** | 先跑 `python preflight_check.py`（退出码非 0 禁止启动）→ 再送 supervisor |
| 2 | **报告任何实验数字前**（含写进 md/PPT/记忆） | 送 supervisor，等"通过"或人类放行 |
| 3 | 完成一个模块 / 阶段（B/C/D/E/F/G）后 | 送 supervisor |
| 4 | 涉及**数据划分、指标计算、对比公平性、消融设计**的改动 | 送 supervisor |
| 5 | **连续两次运行失败** | 送 supervisor |
| 6 | 自评置信度 < 4/5 | 送 supervisor |
| 7 | **发现任何"意外的好结果"** | 先怀疑再送审（好得不真实 = 大概率有泄漏/口径错） |
| 8 | 修改任何**已产出结果的脚本**后 | 送 supervisor（provenance 变更，旧结果可能需重跑） |

### 监督节奏（每个阶段的固定动作）
```
preflight 自检 → supervisor 审方案 → 执行 → supervisor 审结果 → 更新文档/记忆
```

### 收到监督意见后
- 高风险：**立即停止**，修正后重新送审
- 中风险：修正后再进入下一阶段
- 低风险：记录 `review_log.md`，不阻塞
- **禁止忽略监督意见**，除非人类明确同意

---

## ★ 本项目已知的高危失效模式（监督时**逐条盯**）

这些都是本项目**真实踩过**或真实存在的坑，不是假想：

| # | 失效模式 | 症状（都很隐蔽） | 盯防动作 |
|---|---|---|---|
| 1 | **基底不一致** | 某脚本划分分母来自命令行变量，被改后基底静默改变 → 结果不可比 | `preflight_check.py` 穿透 argparse 默认值核查；跑 `train_lora_leakfree.py` **禁传** `--val-frac/--test-frac` |
| 2 | **静默回退** | 缺 `direction_resnet18_flip_basis.pth` → `train_direction_regress` **悄悄**用 ImageNet 初始化 → 方向变随机而日志看不出 | 启动前 `ls` 确认；`load_direction_net` 已改为抛异常 |
| 3 | **零梯度死锁** | `gate=0` + 末层零初始化并存 → adapter 全部梯度恒 0 → 永不更新；"指标没变"被误读为"方向无效" | `direction_adapter.py --self-test` 的梯度自检；`train_direction_adapter.py` 首步再自检 |
| 4 | **静默失效（规则过狠）** | 自造的 `not-dark` 光度判据杀掉 74~97% 的候选点 → 方法"看起来实现了"实则近乎失效 | 小样本冒烟必看**中间统计量**（点数、剔除数），不能只看"跑通了" |
| 5 | **梯度排名错位** | 在全候选取 top-k，选中全是边界外点（边界梯度最高）→ 随后被 in-mask 判据全丢 | 同上：冒烟必看漏斗数字 |
| 6 | **口径混淆** | oracle（GT 方向）混入主表；自带划分的泄漏数当干净泛化 | oracle 方法名强制带 `(oracle-dir)`，仅"诊断/上界"节；主表一律 `(regress-dir)` |
| 7 | **指标口径** | 我们逐图平均(macro)，文献用全数据集汇总(micro) → 数字不可比 | 本轮**不改主口径**，只补落盘 TP/FP/FN/TN 作敏感性附录 |
| 8 | **"完成"误判** | 队列只看 `_last.pth` 的 epoch，**不查结果文件** → 训练完成但评估被杀，误判为完成（已实测发生） | 完成判据必须含"结果 CSV 存在"；补跑 `--evaluate` |
| 9 | **单 seed 当提升** | 两次 from-scratch 同 seed 不逐位复现（loss 从第1轮就不同）→ +0.011 落在 run 间噪声内 | **禁止**写"提升/改进"；只写"数值上高于"+ 统计限定 |
| 10 | **脚本版本漂移** | 改了产出主结果的脚本 → 旧 CSV 的 provenance 被削弱 | 改脚本必须留 diff/patch 备份 + 记录在案 |
| 11 | **翻转方向错位** | 水平镜像时未同步变换方向 `(cos2θ,sin2θ)→(cos2θ,−sin2θ)`；因恒等初始化，**等价性自检抓不到** | 单独单测；训练脚本里显式调用 `flip_direction_horizontal` |
| 12 | **环境性中断** | 本机 Modern Standby 不定期杀训练 | `queue.sh` 的 `--resume` 自愈循环；长训练用 nohup |

---

## 对外口径纪律（违反即审计不通过）

1. 主结果 = **无泄漏划分**；AISD 自带划分数字只作"含同源泄漏"的历史对齐，**禁横比、禁作干净泛化/SOTA**
2. 只说"**数值上高于**"；**禁止**"显著超越 / 超越全监督 / SOTA"
   须附限定：**单 seed、未做显著性检验、测试集 21 个独立场景、差值 < 逐图 std**
3. oracle 结果**禁入主表/PPT/SOTA**
4. **找不到证据 = 未完成**：任何数字必须有落盘日志/CSV/检查点支撑
5. **用语纪律：禁用"官方"代指 AISD 自带的划分**（2026-10-15 起）。
   - 该划分是**数据集随附**的，没有任何权威性，反而**天然带 42/43 同源泄漏**；
     称它"官方"会**给它不该有的正当性**，让人误以为那是"正确/标准"口径。
   - **正确叫法**：含泄漏的那套叫「**AISD 自带划分**（Train412/Val51/Test51）」或「**自带 Test51**」；
     无泄漏的那套叫「**重划 68/16/21**」。首次出现时把"含同源泄漏"写在名字旁边。
   - **例外**：指 Meta 正式发布的 SAM 权重/源码时，"官方"名副其实，可用（如"官方 SAM ViT-B 权重"）。
   - **`review_log.md` 不改**——那是历史审计记录，回改等于篡改留痕。

---

## 提交审查请求包格式
- 里程碑/目标：
- 变更文件与 diff：
- 运行命令：
- 关键日志/指标：
- 自评置信度（1-5）：
- 已知风险：
- 请重点审查的 1-3 个问题：

---

## 记忆文件维护
- 每次完成里程碑后更新记忆（`memory/` + `MEMORY.md` 索引）
- 必须维护下方"证据索引"
- **禁止虚报、漏报、美化**

## 证据索引（2026-10-15 更新）
| 内容 | 位置 |
|---|---|
| 划分函数 | `train_direction_net.py::split_by_base()` |
| **前置自检器** | `preflight_check.py`（>1h 任务前必跑） |
| 结果落盘规范 | `results/{method}_test.csv`（含 `_std` 行）+ `_test_per_image.csv` |
| 主结果（无泄漏） | `results/lora_leakfree_test.csv` = IoU 0.787821 / BER 0.064359（seed=0） |
| 多 seed | `results/lora_leakfree_s1_test.csv`（seed-1，**epoch 11 中断后 resume**）；seed-2 跑中 |
| CNN 无泄漏基线 | `results/{unet,segnet}_leakfree_test.csv` |
| **Mask R-CNN 基线** | 主行 `results/maskrcnn_leakfree_res800_test.csv`（0.7382）；子行 `..._leakfree_test.csv`（0.7330） |
| **传统/零样本基线（无泄漏）** | `results/{otsu,c3i,enhanced,sam_amg}_leakfree_test.csv` |
| 压 BER 实验 | `ber_reduce_eval.py` → `results/ber_reduce_{val_grid,test,selection.json}` |
| 自带 Test51 口径（含泄漏，存档） | `results/lora_test51.csv` |
| 方向模块代码 | `direction_utils.py` / `direction_adapter.py` / `direction_prompt.py` / `train_direction_dtict.py` |
| 跨数据集 | `eval_cross_dataset.py`、`datasets/RSISD/` |
| SAM 补丁 | `segment-anything-main/.../prompt_encoder.py`（diff 存 `backups/patches/`） |
| 审查日志 | `review_log.md`（4 轮） |
| 运行手册 | `GPU执行清单_队列跑完后.md` |
| 应急预案 | `执行顺序与应急预案.md` |
