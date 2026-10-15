# GPU 执行清单（队列跑完后照着跑）

> 生成：2026-10-14 · 配套 `执行顺序与应急预案.md`
> **本文是 runbook**：队列跑完后，按 A→H 顺序执行，每步给命令、预期、失败怎么办。
> 全部 GPU 任务**必须串行**（RTX 4060 8GB，并发必 OOM）。每步启动前先 `tasklist | grep python` 确认无残留。

---

## 前置：通用纪律（每步都适用）

### ⛔ 强制关卡：跑任何 >1h 的任务前，先过自检

```bash
python preflight_check.py      # 5 秒出结论，退出码非 0 就不许启动
```
它自动核查这些**会让人白跑几小时**的问题：

| 检查项 | 为什么 |
|---|---|
| **划分一致性**（基底） | 所有脚本必须用同一 68/16/21；**某个脚本分母不同 → 结果不可比 → 两边重跑**。会穿透 argparse 默认值核查 |
| 三集基名交集 = 0 | 有交集 = 数据泄漏，全部指标作废 |
| 已有 CSV 的文件名集合 == 复算 test 集 | 防止"评估用的不是同一个测试集" |
| 主结果 IoU == 0.787821 且逐图均值吻合 | 防止权重/CSV 被污染 |
| `direction_resnet18_flip_basis.pth` 存在 | 缺失 → `train_direction_regress` **静默回退 ImageNet** |
| SAM 补丁完整 + patch 备份在 | 补丁丢失 → 方向 Adapter 静默失效 |
| 新脚本可编译 | 低级语法错 |

**已知的 1 个 FAIL（预期）**：`direction_resnet18_flip_basis.pth` 在队列跑完前不存在。
**已知的 1 个 WARN（无害）**：`direction_utils.py` 只 re-export 不调用 `split_by_base`。

### 其余纪律

```bash
cd "D:/RemoteSensing-Shadow-SAM-main/RemoteSensing-Shadow-SAM-main"
tasklist //FI "IMAGENAME eq python.exe" | grep -c python   # 应为 0（或只剩队列）
```

- **报告任何数字前 → 先过 supervisor**（`Agent` 工具，`subagent_type=supervisor`），结论写 `review_log.md`
- **长训练（>1h）前 → 也要先过 supervisor**
- 口径纪律：只用"**数值上高于**"；**禁**"显著超越/超越全监督/SOTA"；oracle 结果**禁入主表**
- ⚠️ **跑 `train_lora_leakfree.py` 时不要传 `--val-frac/--test-frac`** —— 默认 0.2/0.2 才是基准划分，改了基底就变了

---

## 阶段 A：队列收尾确认（约 0.5h）

### A1 等队列跑完
```bash
tail -5 logs/queue_console.log      # 等到出现 "队列全部结束"
ls checkpoints/direction_resnet18_flip_basis.pth    # 必须存在！
```
**预期**：队列依次完成 `U-Net 官方 60轮 → SegNet 官方 60轮 → 方向网络 basis 重训`。

**失败应对**：队列死了就重启（`nohup bash queue.sh > /dev/null 2>&1 &`），`_last.pth` 保证不丢进度。
⚠️ `direction_resnet18_flip_basis.pth` 若不存在，**B 阶段全部阻塞**——`train_direction_regress.py` 会**静默回退 ImageNet**，方向变成随机输出而日志看不出来。

### A1.5 ⚠️ 补跑缺失的最终评估（**已确认会发生**）

**问题**：队列的完成判据只看 `_last.pth` 的 epoch，**不检查结果 CSV**。实测 U-Net 官方跑满 60 轮后，
进程在**最终评估那一步**被 Modern Standby 静默杀掉（日志无 traceback），于是 **`results/unet_official_test.csv` 没生成**，
而队列以为"完成了"就往下走了。**SegNet 官方大概率会重犯。**

**修复**（权重都在，只需重跑评估，各约 1~2 分钟）：
```bash
ls results/unet_official_test.csv   || python train_cnn_leakfree.py --arch unet  --split official --evaluate
ls results/segnet_official_test.csv || python train_cnn_leakfree.py --arch segnet --split official --evaluate
```
**核对**：生成的 CSV 里 `n_images` 应为 51（官方 Test51）。

**彻底修复（队列已停后可改）**：`queue.sh` 的 `done_epochs` 判据增加"结果 CSV 存在"这一条，
否则"训练完成但评估未跑"会被误判为完成。

### A2 收官方 U-Net/SegNet 新数字
```bash
cat results/unet_official_test.csv results/segnet_official_test.csv
```
**注意**：官方划分含 42/43 同源泄漏 → 只更新**附表**，标 `‡`，**禁与主表横比**。

### A3 【需审查】把新数字落进对外材料
- `结题报告.md` / `README.md` / `答辩稿.md` / `make_ppt.py` / `metrics_chart.py` / 软著说明书
- 主表 U-Net 用新值 **0.7525**，**删** 58/60 中断限定语，**但禁止写"提升"**（见 `执行顺序与应急预案.md` 的 A3）
- 重生成：
```bash
python metrics_chart.py && python make_ppt.py
```
**先过 supervisor 再审再改。**

---

## 阶段 B：T1.1 方向回归头（约 2h）【长训练，先审】

```bash
# B0 守卫：确认 basis 骨干存在（否则静默回退 ImageNet）
ls checkpoints/direction_resnet18_flip_basis.pth

# B1 正式跑
python train_direction_regress.py --epochs 40
```
**产出**：`checkpoints/direction_resnet18_regress.pth`、`logs/direction_regress_*.log`
**预期**：角度误差 **< 15~20°** 即可用于下游。

**失败应对（应急预案 B1，按序）**：
1. 过滤低质量样本（阴影占比达标 + 角度一致性高）
2. 检查 `gt_direction` 的二倍角圆形均值
3. 兜底：下游先用 `--direction-source oracle` 跑通（**必须标 oracle**）

**过了这一步才能进 C/D。**

---

## 阶段 C：T1.2 方向引导提示（约 3h）

```bash
# C1 冒烟（3 张，先确认不崩）
python direction_prompt_eval.py --limit 3 --direction-source oracle

# C2 链路校验：axis_box 参考行应复现 ≈0.7878（证明评估链路正确）
grep -A2 "LoRA-SAM(axis-box)" logs/dirprompt_*.log | tail -5

# C3 oracle 全量（诊断/上界，**禁入主表**）
python direction_prompt_eval.py --direction-source oracle

# C4 regress 全量（主方法口径）
python direction_prompt_eval.py --direction-source regress
```
**产出**：`results/dirprompt_{project,random,none}_{oracle,regress}_test.csv` + 逐图 CSV

**失败应对（应急预案 C1，**最可能发生**）**：方向提示**大概率打不过框提示**（此前已验 0.714 < 0.717）。
→ **如实报告为消融，绝不硬造提升**。叙事定位为"自监督方向网络 + 自动化提示 + LoRA 微调"。
→ **反面教材**：挑一张好看的图报数，一问就崩。

**调参纪律**：阈值已在 `direction_prompt.DirPromptConfig` **预注册**；调参**只在验证集**做，**主表只报默认参数**。

**C5 过后 → 过 supervisor。**

---

## 阶段 D：T1.3 方向特征 Adapter（约 3h）【长训练，先审】

> **⚠️ 第四轮审查的必须修正（已改代码，命令随之更新）**
> - **主方法方向源 = `regress`（预测方向）**，不是 oracle。用 oracle 训练却按 regress 汇报 = 隐性泄漏。
> - **每次运行必须给不同 `--tag`**，否则 D2/D3/D4/D5 会**互相覆盖权重与结果**。
> - **D4 必须 `--ckpt` 指定 D3 的产物**才能真正 warm-start（否则等于从头再来）。
> - D1 的 check-only 已改为在**测试集(92张)**评估并加**硬 assert**（此前错用验证集 58 张 → 会假报警）。

```bash
# D0 前置：T1.1 必须已完成（regress 口径依赖它）
ls checkpoints/direction_resnet18_regress.pth

# D1 守卫：只检查不训练。断言：在测试集上装 adapter 后 IoU 仍 = 0.787821（差 <1e-3）
python train_direction_adapter.py --check-only --tag main
#   若断失败会直接抛异常 —— 说明恒等初始化被破坏，立即停止

# D2 冒烟（用独立 tag，避免污染生产产物）
python train_direction_adapter.py --epochs 1 --limit-train 8 --limit-val 3 \
    --direction-source regress --tag smoke

# D3 主训练（默认就是 regress）
python train_direction_adapter.py --epochs 20 --direction-source regress --tag main
#   ↑ 若想省时间：先 --epochs 10 --tag pilot，看验证曲线是否还动

# D4 联合微调（warm-start 自 D3 —— 必须显式指定 --ckpt，否则等于重来）
python train_direction_adapter.py --epochs 20 --direction-source regress --tag main_joint \
    --ckpt checkpoints/sam_vit_b_shadow_v4_dir_adapter_main_leakfree.pth

# D5 同权重对比"装 / 不装"（出净效果）
python evaluate_direction_adapter.py --tag main_joint \
    --ckpt checkpoints/sam_vit_b_shadow_v4_dir_adapter_main_joint_leakfree.pth
```
**产出**：`checkpoints/sam_vit_b_shadow_v4_dir_adapter_{main,main_joint}_leakfree.pth`、
`results/diradapter_{main,main_joint}_test.csv` + 逐图 CSV

> **关于"分两步"的价值**：只训 adapter 时 SAM 主干仍要前向+反向，每 epoch ~510s **不会变快**。
> 两步走的价值在于**课程式**（先适应 adapter 再联合），不是省时。**时间紧可只跑 D3 一步。**

**失败应对（应急预案 D2，**铁律**）**：训练结果 **< 0.7878 → 保留 LoRA-only 为主方法**，方向 Adapter 作"尝试过的探索"如实报告。**绝不为了"做了模块3"而用更差的方法当主结果。**

**已知坑**（代码里已修，但要知道症状）：
- 零梯度死锁（gate + 末层双零）→ 脚本首步即做梯度自检
- 翻转必须同步变方向 `(cos2θ,sin2θ)→(cos2θ,−sin2θ)` → 漏做**自检抓不到**，只在训练里悄悄错

**D6 过后 → 过 supervisor。**

---

## 阶段 E：T1.4 DTICT（约 0.5h，GPU 占用极小）

```bash
# E1 对照（单网络、无噪声抑制）
python train_direction_dtict.py --no-dtict --epochs 30

# E2 DTICT（双网络互选 + 方向一致性门）
python train_direction_dtict.py --epochs 30
```
**产出**：`results/direction_dtict_ablation.csv`（两行直读对比）

**报告纪律**：论文未公开 DTICT 完整超参，本实现是**按原理的简化重构、非逐行复现** → 报告必须写明，**不得声称"复现了论文的 DTICT"**。

**对照基线**：论文 Table 2 的 DTICT 增益是 **IoU +1.63% / F1 +0.81%**（其口径与我们不同，只可参考，不可直接引用为我们的数字）。

---

## 阶段 F：对比与指标（约 8h）

| 步 | 命令/做法 | 说明 |
|---|---|---|
| F1 | `torchvision` 的 `maskrcnn_resnet50_fpn` 在 AISD 上训+评 | 申报书点名"IoU 优于 Mask R-CNN"。detectron2 装不上就用 torchvision；再不行就以 U-Net/SegNet 作全监督代表并**如实写明未复现** |
| F2 | 用 `checkpoints/saru/best_AISD_ckp.pth` 实跑 | **真 SOTA 对比**。⚠️**不用 ShadowFormer**——它是"阴影**去除**"，指标不可比 |
| F3 | 把 Focal+Dice 换成 IoU Loss + Boundary Loss 再训一次 | 申报书点名的损失函数 |
| F4 | 补落盘逐图 TP/FP/FN/TN（**独立函数，不动 `compute_metrics`**） | 出"全数据集汇总"版 BER 作**口径敏感性**附录。**主口径不动** |
| F5 | 只对**最终主方法**跑 3 seed | 基线保持单次并说明 |

---

## 阶段 G：跨数据集（约 4h）

数据已就位：`datasets/RSISD/{train1000,val86,test200}/{shadow,mask}`（二值 mask，PNG 512×512，与 AISD 同构）。

```bash
# G1 冒烟
python eval_cross_dataset.py --limit 10

# G2 全量：零样本迁移（设计 A，不重训不微调）
python eval_cross_dataset.py
```
**产出**：`results/cross_rsisd_test200.csv` + 逐图 CSV

**对照**：AISD 无泄漏划分上同为框提示的 IoU = **0.787821**。跨域下降是**预期现象**，如实报告，**不据此调参**。

**可选加固**：用 `checkpoints/saru/best_RSISD_ckp.pth` 跑同集作 SOTA 参照（需适配 SARU 的网络结构，工作量另计）。

**口径修正（应急预案 H1）**：申报书写"WHU 作辅助**测试集**验证阴影泛化"是**用途错位**——导师论文里 WHU 是**无标签预训练**数据。结题报告须改写措辞，**但不可否定导师论文**。

---

## 阶段 H：成果（最后）

`结题报告.md` → 软著说明书 → 论文初稿 → Python 工具包/QGIS 插件。

---

## 附：一页速查（按顺序抄）

```bash
cd "D:/RemoteSensing-Shadow-SAM-main/RemoteSensing-Shadow-SAM-main"

# A
tail -5 logs/queue_console.log && ls checkpoints/direction_resnet18_flip_basis.pth

# B  （先过 supervisor）
python train_direction_regress.py --epochs 40

# C  （先过 supervisor）
python direction_prompt_eval.py --limit 3 --direction-source oracle
python direction_prompt_eval.py --direction-source oracle
python direction_prompt_eval.py --direction-source regress

# D  （先过 supervisor）
python train_direction_adapter.py --check-only
python train_direction_adapter.py --epochs 20 --freeze-base --direction-source oracle
python train_direction_adapter.py --epochs 20 --direction-source oracle
python evaluate_direction_adapter.py

# E
python train_direction_dtict.py --no-dtict --epochs 30
python train_direction_dtict.py --epochs 30
```

---

*生成日期：2026-10-14*
