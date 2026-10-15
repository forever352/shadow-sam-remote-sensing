# 审查与审计日志

## 监督者视图
- 当前阶段：阶段1（方向判别自监督预训练）已完成，但审计不通过，需先补证据
- 本阶段高风险项：BER 0.050 无证据；主结果 0.839 存在同源泄漏；99.46% 无落盘日志
- 最近一次审查结论：不通过（见下方审计记录 2026-10-13）
- 待人工确认项：~~ds-cc-proxy 是否安装~~ → **已确认无需安装**

## 环境/配置结论（2026-10-13 核实）
- 本地代理：**CC Switch（cc-switch.exe）**，监听 127.0.0.1:15721，settings.json 的 ANTHROPIC_BASE_URL 指向它
- 模型路由：已由 settings.json 的 `ANTHROPIC_DEFAULT_*_MODEL*` 映射实现（SONNET/FABLE→deepseek-flash，OPUS/HAIKU→deepseek-v4-pro）
- **ds-cc-proxy：不安装**。它是另一套工具（默认端口 16889，当前无监听），与现有 CC Switch 冲突且不会被使用
- settings.json 保持不变（用户决定）

## 审查记录

<!-- 格式：
### YYYY-MM-DD 审查类型：标准审查
- 范围：
- 结论：
- 高风险：
- 中风险：
- 低风险：
- 修正建议：
- 是否通过：
-->

## 审计记录

### 2026-10-13 审计类型：进度真实性（首次，独立只读子代理执行）

- 审计范围：3 条进度声明（方向预训练 99.46% / AISD 子图泄漏 / LoRA IoU 0.839+BER 0.050）
- 属实：
  - 方向预训练脚本 `train_direction_net.py` 真实实现，`split_by_base()` 确按原始大图基名划分（非注释造假）；检查点 44.7MB 时间自洽
  - AISD Train412×Test51 共享 **42** 个原始大图基名，独立统计完全属实
  - LoRA IoU 0.839 有落盘日志证据（`train_v4_quick.log` 第35行 `=== Test51 框提示 IoU = 0.8393 ===`）
- 虚报：无
- 无法验证：
  - **翻转准确率 99.46%** —— 无任何落盘日志，脚本只 print 不写文件，项目无 `logs/` 目录
  - **BER 0.050** —— 只存在于 md/PPT 文档，无日志、无 CSV；`evaluate_lora.py` 计算后不落盘
- 漏报：
  - DTICT 在本阶段脚本中未实现（文档/协议中提到）
  - `results/summary.csv`/`per_image.csv` 为 10-11 旧 run（Ours=0.4708），**不含 LoRA 行**，不能作为 0.839 的证据
  - 评估口径不一致：选模型用 Val51 质心点提示（0.8192），报结果用 Test51 框提示（0.839）
- 高风险（影响论文结论）：
  1. BER 0.050 无原始证据（申报书核心指标）
  2. 主结果 IoU 0.839 在与训练集 42/43 基名重叠的泄漏测试集上取得，系统性偏高；项目仅对方向自监督做了重划分，未对下游 LoRA 结果做
  3. 99.46% 不可复现
- 建议：
  - P0 重跑 `evaluate_lora.py` 并落盘 `results/lora_test51.csv`
  - P0 给 `train_direction_net.py` 加文件日志并重跑
  - P1 补无泄漏版下游评估（或至少披露泄漏影响）
  - P2 修正证据索引；实现/声明 DTICT
- 本次审查结论：**不通过**（存在高风险项，需修正后再进入阶段2）

### 2026-10-13 审查类型：标准审查（P0 修复后，supervisor 子代理执行）

- 范围：results/lora_test51.csv、lora_test51_per_image.csv、logs/eval_lora_*.log、logs/direction_pretrain_*.log、checkpoints/direction_resnet18_flip.pth 及相关脚本
- 结论：**有条件通过**
- 独立复算确认（P0 目标达成）：
  - LoRA 六项指标与文件逐项一致；逐图 CSV 51 行属实；**BER=0.050066 确实落在文件里**（不再是仅存于 md/PPT）
  - 逐图 51 行独立重算均值，与汇总行**六位小数完全相等**；std 亦吻合
  - compute_metrics 口径正确，框提示来自 Otsu(输入图自身)、未用 GT，无 prompt 泄漏
  - 方向日志 30 个 epoch 属实，best=1.0000 出现在 **Epoch 26**；`if acc > best_acc` 逻辑正确，低轮次未被误判；同 seed 复跑 split_by_base 得 320(84)/92(21) 与日志一致；checkpoint 时间戳自洽
- 高风险：
  1. **IoU 0.8393 建立在泄漏测试集上**：Test51 的 51 张里 **50 张来自训练场景**（仅 vienna12 独立）；同源=同传感器/同太阳方位/相邻裁剪。不能按"干净泛化/SOTA"对外报告
- 中风险：
  1. 评估口径不一致：选模型用 Val51 质心点(0.8192)，报结果用 Test51 框提示(0.8393)，且加载的是末轮 `..._last.pth` 而非验证选出的 `...v4_lora.pth`
  2. U-Net/SegNet 无逐图 CSV / std 落盘（结题报告写 0.805±0.059、0.780±0.053 但仅单行日志），与 LoRA 格式不统一
  3. 单 seed，无重复实验
  4. results/summary.csv 仍是 10-11 旧 run（Ours=0.4708），未清理/标注
- 低风险：1.0000 是饱和的代理指标（184 样本，翻转判别 ≠ 下游效果）；文档/图表数字来源索引未统一
- 必须修正：对外报告 0.8393 必须显式标注同源泄漏；写清口径（选择集/报告集/权重）
- 建议修正：补无泄漏版下游评估；U-Net/SegNet 补逐图 CSV+std；多 seed；清理 summary.csv
- 是否允许进入下一阶段：**否**（泄漏影响披露或补齐无泄漏评估之前）

### 2026-10-14 审查类型：标准审查（进度与指标数字，supervisor 子代理执行）

- 范围：进度六条声明（后台队列当前阶段 / 阶段0 完成度 / 阶段1 状态）+ 四行指标数字（LoRA 官方 0.8393、LoRA 无泄漏 0.7878、U-Net 无泄漏 0.7414、SegNet 无泄漏 0.7403）
- 结论：**有条件通过**
- 独立复算确认：
  - 复现 `split_by_base`：train=262(68) / val=58(16) / test=92(21)，三集基名交集 **0** → 子图同源泄漏**确已修复**
  - `lora_leakfree_test_per_image.csv` 的 92 个文件名集合 == 独立算出的 test 文件集合，完全相等
  - 4 组 summary 行与逐图重算均值/标准差，六项 **全部 <5e-5 吻合**
  - 进度第 1、2、5、6 条属实（`unet_official_20261014_143400.log` 第57行 `Epoch 48/60`；T1.1~T1.5 待办；`train_direction_regress.py` 存在且无日志=未运行）
- 高风险：
  1. **`results/lora_test51.csv` 的 0.8393/0.0501 建立在泄漏测试集上**（Test51 的 43 基名中 42 个出现在 Train412，仅 `vienna12` 独立）→ 不得作为"干净泛化/超越全监督/SOTA"汇报
  2. **仓库文档仍写着旧的泄漏版结论**：`结题报告.md` 第63–83行仍为 SegNet 0.780 / U-Net 0.805 / LoRA 0.839 并称"超过全监督 U-Net(0.805)" → 未更新前任何进度汇报与仓库文档自相矛盾
- 中风险：
  1. **U-Net 无泄漏 0.7414 来自 58/60 中断训练的第50轮 best-val 检查点**（`unet_leakfree_20261014_131323.log` 第7行"评估模式：加载…不训练"），非完整 60 轮 → 必须加限定语
  2. 选模型口径不一致：官方 LoRA 用 `_last.pth`，无泄漏 LoRA/CNN 用 best-val → 同一方法两行不可比
  3. SAM 用框提示（Otsu）vs CNN 纯图输入，非同输入对比（文档局限②已承认）
  4. **T1.1 依赖缺失会静默降级**：`train_direction_regress.py` 第40行依赖的 `direction_resnet18_flip_basis.pth` **当前不存在**；第122–125行会静默退回 ImageNet 初始化
  5. 方向网络 0.99~1.0 的划分是旧的 84/21，**不是 basis 68/16/21**；其验证集 21 场景 == basis 测试集 21 场景
  6. 官方 U-Net（在跑）best-val 仅第10轮 0.7720，之后 20/30/40 轮更低 → 再中断将取第10轮模型
- 低风险：陈旧易混文件（`*_SMOKE1ep.csv`、旧 `results/summary.csv` Ours=0.4708）；文档/PPT 数字索引未统一；单 seed
- 必须修正（汇报前）：
  1. 0.8393 一行加"测试集与训练集共享 42/43 基名，含同源泄漏"限定语，且不与被 92 张无泄漏三行同表横比
  2. 更新 `结题报告.md`（及答辩稿/PPT）中的泄漏版数字与"超过全监督"表述
  3. U-Net 0.7414 标注"第50轮 best-val，训练 58/60 中断"
  4. 确认 T0.3 方向网络重训真正启动（现有 `logs/direction_basis_console.log` 是 Git Bash fork 崩溃，未起进程）；`direction_resnet18_flip_basis.pth` 生成前不启动 T1.1
- 建议修正：统一选模型口径；"优于全监督"补输入协议/单 seed 限定；清理 SMOKE 与旧 summary；方向回归标签报告失败样本
- 是否允许进入下一阶段并向人类汇报：**否**（须先完成"必须修正"1–4 项）
- 可汇报口径：LoRA 无泄漏 0.7878/0.0644、SegNet 无泄漏 0.7403/0.1021（可直报）；U-Net 无泄漏 0.7414/0.0873（须加"中断/epoch50"限定语）
- 本次审查 token 估算：约 84k


### 2026-10-14 审查类型：标准审查（文档口径切换 + 队列重启方案，supervisor 子代理执行）

- 范围：方案 A（文档修正：结题报告.md/答辩稿.md/README.md 主结果切到无泄漏划分）+ 方案 B（train_cnn_leakfree.py 加 resume、U-Net/SegNet 官方从零重跑、方向 basis 重训）
- 结论：**有条件通过**
- 独立复核（未采信主智能体转述）：
  - 4 行 CSV 逐字核对**完全一致**：LoRA 无泄漏 0.787821/0.064359、U-Net 无泄漏 0.741381/0.087331、SegNet 无泄漏 0.740308/0.102080、LoRA 泄漏 0.839301/0.050066
  - 独立 ls 基名统计：Test51 43 基名、Train412 105 基名、交集 **42**，仅 vienna12 独立 → "共享 42/43"属实
  - 进度核实属实：queue_console.log / unet_official_20261014_143400.log 均止于 Epoch 48/60 (132s)；tasklist 无 python；direction_resnet18_flip_basis.pth 不存在；direction_basis_console.log / cnn_official_console.log 均为 Git Bash fork 崩溃残留（无 epoch 行）
  - train_cnn_leakfree.py 确无 --resume、无 _last.pth 保存（第150–180行仅 val 提升时存 best）
  - unet_leakfree_20261014_122049.log 止于 Epoch 58/60，best 存于 Epoch 50 (val IoU=0.7831)；unet_leakfree_20261014_131323.log 第7行"评估模式：加载…不训练" → 0.7414 来自此第50轮检查点属实
  - unet_shadow_official.pth 时间戳 14:58 ≈ 官方 run 起(14:34)+10×~145s → 与"仅第10轮(val 0.7720)落盘"自洽
  - 口径一致性（新发现，利好）：train_lora_leakfree.py 第165–167行最终评估前**加载 best-val**（sam_vit_b_shadow_v4_lora_leakfree.pth，时间戳01:14≈epoch25 val0.8128）；train_cnn_leakfree.py 第182–183行同载 best-val → **无泄漏三行同为 best-val 口径**，横比口径一致，非旧版 _last 混用
  - 增强 bug 属实：train_unet.py 第88/97行对图与掩膜**各自独立** random.random() 抽签 → 旧 0.805/0.780 确为图文错位 bug 版
- 高风险：
  1. **对外答辩材料不在方案 A 修正范围内**：结题答辩PPT_最终版.pptx（Oct13，含 0.839/0.805/0.780、"全监督"×4）与生成器 make_ppt.py（第110、139–141、168 行硬编码旧数字）仍输出泄漏版结论。只改 3 个 md，PPT 一旦用于答辩即"对外报告泄漏数字"
  2. **"领先"结论无显著性支撑**：无泄漏主表 LoRA(0.7878) 仅高于 U-Net(0.7414) 0.046，小于两者逐图 std(0.0686/0.1069)；单 seed、无检验；且测试集 92 张子图来自**仅 21 个独立场景**，有效样本极小。据此写"领先/超过全监督"属过度声明
- 中风险：
  1. 主表切到**自建 68/16/21 非标准划分**，与文献 AISD 数字不可直接比；附表官方划分又含泄漏 → 仅表注不足，**须补一整段"与文献可比性"局限说明**（对应局限⑤）
  2. 附表把"旧 run(增强 bug 版 0.805/0.780)"与"新 run"并列，有被横向误读风险；须隔离并标注"泄漏+增强 bug"双重问题
  3. README.md 第62行"SOTA(SARU 0.864) vs 本项目 0.839"用的是**泄漏版**数字做文献对比，须删/改
  4. B1 改脚本后的版本 provenance：现有 leakfree CSV 均由**改前脚本**产出；改脚本后"同一脚本可复现"声明被削弱，须记录版本或重跑说明
  5. train_cnn_leakfree.py 第150–180行的选模型口径(best-val)在加 resume 时**必须保持并持久化 best**，否则续训会重置 best、改变入选模型
- 低风险：陈旧易混文件（*_SMOKE1ep.csv、旧 results/summary.csv Ours=0.4708）；"第48轮132s疑似休眠"仅为时间推断，无直接证据
- 必须修正（对外汇报/答辩前）：
  1. 方案 A 范围**并入 PPT 与 make_ppt.py**（或书面声明答辩不使用旧 PPT）
  2. "领先/超过"改为带统计与样本限定的措辞
  3. 补"与文献可比性"局限整段
  4. 附表旧 run 隔离并标注"泄漏+增强 bug、待重跑、禁横比"
  5. 改脚本后记录版本 / 或重跑并留痕
- 建议修正：U-Net 限定语放**表注**（表内保留纯数字），并补该检查点 val IoU=0.7831；清理 SMOKE1ep 与旧 summary；报告有效独立场景数 n=21；单 seed 显式声明
- 是否允许进入下一阶段：**否**（须先完成必须修正 1–5；且 B1 脚本改动后须**重新送审**再启动 B3 长训练）
- 可汇报口径（本次核可）：
  - LoRA 无泄漏 **0.7878/0.0644**、SegNet 无泄漏 **0.7403/0.1021**（best-val，可直报）
  - U-Net 无泄漏 **0.7414/0.0873** → 须写明"best-val(epoch50)、训练于 58/60 中断"
  - 三行仅可表述为"**数值上**高于 CNN 基线；单 seed、未做显著性检验、测试集为 21 个独立场景，不足以断言显著优于"
  - LoRA 官方 **0.8393/0.0501** 仅可作"含 42/43 同源泄漏的历史/对齐数字"，禁作干净泛化与 SOTA 对比
- B 方向评价：B2"从零重跑 60 轮"**优于**"第10轮续跑"（脚本无 resume、中断轨迹随机状态不可复原、混两段轨迹不可复现；LoRA 亦为从零跑满）；B1 只要 split_by_base(files,0.2,seed=0)/split_by_base(train_pool,0.2,seed=1) 调用**逐字不变**，即不破坏与 LoRA 无泄漏版"同一划分"的可比性（划分由固定 seed 决定，与 resume 无关）
- 本次审查 token 估算：约 40k

### 2026-10-14 审查类型：标准审查（第二轮：修正落实核验 + B1 resume 复审 + 队列放行，supervisor 子代理执行）

- 范围：① 上一轮"必须修正 1–5"落实情况（make_ppt.py / metrics_chart.py / 结题报告.md / README.md / 答辩稿.md + 重新生成的 PNG/PPT）；② B1 `train_cnn_leakfree.py` 的 `--resume` 与每轮 `_last.pth`；③ 拟启动的长训练队列（官方 U-Net/SegNet 从零、方向 basis、可选 U-Net leakfree）
- 结论：**有条件通过**

- 独立核验（未采信主智能体转述）：
  - 划分复算（独立重实现 base_name/split_by_base）：train=262(68)/val=58(16)/test=92(21)，三集基名交集 0；`results/{lora,unet,segnet}_leakfree_test_per_image.csv` 的 92 名集合 == 复算 test 集合（True×3）。`train_cnn_leakfree.py` L111-112 与 `train_lora_leakfree.py` L96-97 调用语义一致（0.2/0.2，seed=0/1；后者默认值亦为 0.2）。
  - 泄漏口径复算：Train412 105 基名 / Test51 43 基名 / 交集 **42** → "共享 42/43"属实。
  - 文档确已改：`结题报告.md` L70-81 主表无泄漏三行、L83-98 附表隔离+‡+统计限定、L134-145 局限①-⑥与"最终口径"；`README.md` 主/附表 + L59-60 + L62（旧"SOTA(SARU 0.864) vs 0.839"已删）；`答辩稿.md` L38/46-47/Q9；`make_ppt.py` L110/135-137/155-157/172；`metrics_chart.py` 双面板（左=无泄漏3、右=官方7，图注 †‡ + 限定）。
  - 产物确已重生成：`结题答辩PPT_最终版.pptx`(16:40) + 备份 `结题答辩PPT_最终版_旧泄漏口径_备份20261014.pptx`(16:39) + `results/metrics_bar.png`(16:40，已目视确认为双面板、无跨口径横比)。
  - B1：L134/L162-173 新增 resume 分支；L198-204 best-val 逻辑未变（仅 `epoch%10==0` 评估、仅 val 提升覆盖 `ckpt`）；L205-207 每轮存 `{model,optim,epoch,best}`。SegNet leakfree 为**完整 60 轮**日志（`segnet_leakfree_20261014_131333.log`，末行 0.7403），与其 CSV 值 0.740308 一致。
  - 队列停止证据：`unet_official_20261014_143400.log` 止于 `Epoch 48/60 (132s)`；`queue_console.log` 止于 15:47；无 python 进程；GPU 空闲；`direction_resnet18_flip_basis.pth` 仍不存在；`direction_basis_console.log`/`cnn_official_console.log` 为 Git Bash fork 崩溃残留。`train_direction_net.py` L162/L171-172 确认 `--split basis` 输出 `direction_resnet18_flip_basis.pth` 且与 CNN 同一 68/16/21 划分。

- 高风险：无（未发现新引入的泄漏/测试集参与训练/指标错误/不公平对比；对外文档口径已切到无泄漏）

- 中风险：
  1. `--resume` 完成态静默评估：`start_epoch=st["epoch"]+1` 若 > `args.epochs`，`for` 空转 → 落到 L209-226 用现有 `ckpt` 直接评估并写结果表，且无显式告警（日志无 Epoch 行）。`train_cnn_leakfree.py` L162-175/L209。
  2. `_last.pth` 非原子保存：L206-207 `torch.save` 直接覆盖目标文件；叠加"本机确会休眠"，写在途中被打断 → `_last.pth` 截断/损坏；而 L166 `torch.load` 无 try/except → 续训恰在最需要时崩溃，与"崩溃后续训"设计目标矛盾。
  3. 续训不恢复 RNG：L162-173 仅恢复 model/optim/epoch/best，未恢复 `random`/`numpy`/`torch` 状态 → 一旦发生 resume，其后轨迹与原不中断轨迹不同（洗牌/翻转抽签不同），精确不可复现。docstring"不影响任何数值计算"仅对 **from-scratch 运行**成立（该判断本身正确），对 resumed 运行不成立。
  4. 软著_软件说明书_初稿.md 未纳入修正范围：L48/L55 仍写"4.34%"、L55/L134 IoU 0.717、L135 BER 0.109、L138"显著优于"。属对外材料。
  5. 长训练休眠风险（操作性）：上一队列已在 48/60 被中断（该 epoch 132s vs 正常 ~75s），`进度接力.md` L24/L68 亦记录"疑因休眠崩溃"。电源"从不睡眠"未确认前重启，高概率重复失败。

- 低风险：
  1. `结题报告.md` L166 仍写"metrics_bar.png 仍为官方口径、重绘前不作对外材料" —— 与 16:40 已重绘的双面板图不符（陈旧注释）。
  2. `进度接力.md` L37"LoRA 仍领先 U-Net → '优于全监督'不是假象"无统计限定；L10/L18/L24 队列状态已过期。
  3. `demo.py` L42 help"默认 LoRA 0.8393"、`train_adapter_v4.py` L5 注释"U-Net 达 0.805"、`train_adapter_v2.py` L9"4.34%" —— 代码内未标口径的陈旧引用（非对外）。
  4. `results/` 仍存 `*_SMOKE1ep.csv`；README 目录说明仍列旧 `summary.csv`(Ours=0.4708)。
  5. `segnet_shadow_leakfree.pth` mtime(16:39) 晚于其训练结束(14:33)、无代码路径可解释（冒烟不触发 val 保存，不会写 ckpt）→ 疑为冒烟测试时手工备份/还原所致。该结果数字本身由未改动的完整日志(0.7403)背书，但 ckpt 完整性建议用 `--evaluate --split leakfree` 复核一次。

- 必须修正（放行前）：
  1. 启动前确认 Windows 电源"从不睡眠"（或备好"跑挂即续"操作）；**队列三条训练命令统一加 `--resume`**（无 `_last.pth` 时脚本自动告警并从 epoch 1 起 = 等价 from-scratch；中途休眠后重启即续）。
  2. 合入两类 resume 守卫（约 5 行）：(i) `start_epoch > args.epochs` 时打印明确告警并跳过最终评估（或要求显式 `--evaluate`）；(ii) `torch.load(last_ckpt)` 包 try/except，损坏时告警回退到 epoch 1（与既有"找不到则从头"一致）；推荐 `_last.pth` 先写临时文件再 `os.replace` 原子改名。

- 建议修正：
  1. 队列优先级：建议把"U-Net 无泄漏重跑（消除局限⑥）"提前或并列 —— 官方 U-Net/SegNet 重跑只改善一个**仍被标注"泄漏"**的附表；leakfree U-Net 重跑直接移除**主表（对外核心）**的第†号局限。电力/时间有限时优先主表。
  2. 队列写成 `queue.sh` 后 nohup（可复现），每次启动前 `tasklist | grep python` 确认无残留。
  3. 官方 U-Net/SegNet 跑完后需同步改 `结题报告.md`/`README.md`/`make_ppt.py`/`metrics_chart.py`（‡ 由"泄漏+增强 bug"→"泄漏"，数字更新）并重生成 PNG/PPT。
  4. 跑 leakfree U-Net 前注意项目自定规则"新结果≥旧结果才替换主方法"（`进度接力.md` L89）——若更差则保留旧值并说明。
  5. 更新 `结题报告.md` L166、`进度接力.md`（队列状态 + 统计限定）；下一轮把软著说明书纳入修正。

- 是否允许进入下一阶段 / 放行队列：**是（附条件）** —— 满足上"必须修正 1–2"即可启动；不满足（尤其电源未确认且未加 `--resume`）则**否**。
- 可汇报口径：
  - 主表可报：LoRA 无泄漏 0.7878/0.0644、SegNet 无泄漏 0.7403/0.1021（直报）；U-Net 无泄漏 0.7414/0.0873（**必须**带"best-val(epoch50)、58/60 中断"限定）。
  - 三行仅可表述为"**数值上**高于 CNN 基线；单 seed、未做显著性检验、测试集 21 个独立场景、差值 < 逐图 std"。
  - 附表官方 0.8393/0.0501、U-Net 0.805‡、SegNet 0.780‡ 仅作"含泄漏（‡ 另含增强 bug）历史对齐"，**禁横比、禁作干净泛化/SOTA**。
  - 不得出现"显著超越/超越全监督/SOTA"表述。
- 本次审查 token 估算：约 65k

### 2026-10-14 审查类型：标准审查（模块② 方向引导提示 + 模块③ 方向 Adapter 实现方案 + BER 口径核对，supervisor 子代理执行）

- 范围：待审方案 A–F（文件改动面 / direction_utils 定位 / 模块②算法 / 模块③ Adapter / 执行顺序 / 回退）+ BER 口径核对 §D
- 结论：**不通过**（模块③ 设计存在高风险缺陷，须修正后再审）

- 独立复核（未采信主智能体转述）：
  - `prompt_encoder.py:128-168` forward 签名 = (points, boxes, masks)，返回 `(sparse, dense)`；`sam.py:106` / `predictor.py:222` 均按位置调用、**不传 direction**；`utils/onnx.py` 不调用 forward → 加默认 `None` 关键字参数**不波及既有调用点**。`segment_anything` editable 安装属实（`segment-anything-main/segment_anything.egg-info` 存在）。
  - `mask_decoder.py:121-127`：`tokens=cat([output_tokens(5), sparse])`、`src=src+dense` → adapter 注入 prompt_encoder 出口**确为解码器输入**，注入点选择正确。
  - 主 ckpt 数字属实：`results/lora_leakfree_test.csv` = 0.787821/0.064359，n=92。
  - `checkpoints/direction_resnet18_flip_basis.pth`、`direction_resnet18_regress.pth` **均不存在**；GPU 当前 96% 占用（unet 官方 60 轮 queue 在跑，`queue_console.log` 止于 Epoch 41/60）→ 方案"需等 GPU 空闲"属实。
  - 六个拟新增文件**均不存在**；全项目 grep `direction_adapter` 无任何命中 → 模块②③确为零代码起步，非"已做未报"。

- 高风险：
  1. **模块③ Adapter 的"零初始化 net 末层 + gate=0"会造成零梯度死锁，adapter 永不更新（静默降级为 LoRA-only）**。证明：令 gate=g=0，net 末层零初始化 → scale=0、shift=0 → u=tanh(0)=0、v=0。逐项求偏导：∂out/∂v=g=0；∂out/∂u=g·sparse=0；∂out/∂g=sparse·u+v=0。**全部为 0**，AdamW 第一步 m=v=0 → 更新恒为 0。结果：③ 训练结束指标 == 0.7878，方案会把它误读为"方向 Adapter 无效"（与应急预案 D2 混同），实为**根本没训练**。方案 F 的"<0.7878 则作探索报告"在此情形下是**错误结论**。这是"声称完成模块③但实际未生效"的高风险，须先修。方案所坚持的"逐比特等价"仅对 **forward 输出**成立（严格说 x·1.0+0 与 x 的 allclose 成立，-0.0 边缘除外），**但对可训练性不成立**。

- 中风险：
  1. `dense + g*shift`：`shift` 若为 `[B,256]` 会在 H/W 上广播错误或报错，必须 reshape 到 `[B,256,1,1]`；`sparse` 侧 `[B,N,256]` 与 `[B,256]` 广播是"所有 token 共用同一 shift"，语义受限但可接受。
  2. AMP autocast dtype：训练在 `torch.cuda.amp.autocast()` 下跑（见 `train_adapter_v4.py:167`），adapter 的 `Linear(2→64)` 输入 `direction`(fp32) 与 `sparse/dense`(fp16) 混算需验证不报 dtype 错或隐式升精度；等价性自检**必须在 autocast 下做**，仅 fp32 通过不算。
  3. `axis_box` 复现 0.7878 的链路校验**必须复用同一函数**：`train_lora_leakfree.eval_box_full`→`direction_experiment.method_boxes`→`prompt_experiment.component_boxes(min_area=300)`。若新 ② 代码自带一套 min_area=150 的点/框逻辑，复现不出 0.7878 会产生**假"链路断裂"**或**假"通过"**。
  4. 若为落盘 TP/FP/FN/TN 去**改 `eval_shadow.compute_metrics` 返回键**，会**弄坏已审查链路**：`evaluate_lora.py:87-89` 与 `train_lora_leakfree.eval_box_full:177-180` 用 `DictWriter(fieldnames=[...]).writerows(dict含额外键)` → 多键会 `ValueError`。必须改用**独立函数/独立脚本**落盘混淆矩阵，不动 compute_metrics 契约。
  5. 模块② 的"CC 取面积≥150 连通域"**未指定粗提取 mask 来自哪一步**（HSV？C3I？`mask_enhanced`？）。项目里 `mask_c3i` 有 0.4 面积反转 hack + `_rm_small(150)`，与裸 HSV 阈值结果不同 → 不写死则不可复现、也不对应申报书措辞。
  6. 模块③新训练脚本**必须逐字复用同划分调用**（`split_by_base(files,0.2,seed=0)` 再 `(train_pool,0.2,seed=1)`）**与同 `eval_box_full` 口径（框提示）**，否则与 0.7878 不可横比。
  7. 即便把 gate 改活，`tanh(scale)` 在大 scale 下饱和 → 梯度消失，属次要隐患，建议改 ReZero/纯加性以规避。
  8. t5 的 "random 方向"若用无种子 RNG，则**不可复现**；须固定（见建议）。
  9. oracle 上界数字若无硬约束，存在"好看就报"诱惑（见必须修正 5）。

- 低风险：
  1. `--check-only` 断言 IoU `==0.787821` 建议改**容差 1e-4**（cuDNN 非严格确定性）。
  2. 新增 CSV 命名/schema 须对齐既有规范，并登记进 `AGENTS.md` 证据索引。
  3. `direction_utils.py` 大量 re-export 会在被 import 时**触发原模块顶层副作用**（如 `prompt_experiment/evaluate_finetuned/direction_experiment` 顶层建 `sam`/加载 ckpt；`evaluate_finetuned.py` 顶层 `transform=...`）→ 注意 import 顺序与 `env_setup` 先于 torch。

- 必须修正（放行前）：
  1. **修复③零梯度死锁**，三选一：(a) gate 初值 0，但 **net 末层非零小随机初始化**（ReZero 式，gate 提供梯度入口）；(b) **去掉 gate，用纯加性 `out=x+shift`，末层零初始化**（∂out/∂shift=1，梯度畅通，且初始逐比特恒等）；(c) FiLM **恒等初始化 γ=1、β=0**（无 gate）。并**强制新增自检**：一次 `train_step` 后 `adapter` 各参数 `.grad` 的 norm **严格 > 0** 且参数值发生变化（在 autocast 下）；仅 forward `allclose` 不足以证明可训练。
  2. 模块②粗提取 mask 来源**写死**（对应申报书"RGB→HSV + C3I"），并在日志/方法名体现。
  3. `axis_box` 参考实现**逐字复用** `component_boxes(min_area=300)`/`method_boxes`。
  4. TP/FP/FN/TN 落盘用**独立函数**，不改 `compute_metrics` 返回契约；**BER 主口径本轮不改**（见"可汇报口径"）。
  5. oracle/regress 硬约束：方法名强制带 `(oracle-dir)`/`(regress-dir)`；oracle 数字**只允许出现在"诊断/上界"专节**，CSV/图注/文件名带 `oracle`，**禁止进入主对比表、结题报告主表、PPT、SOTA/泛化表述**；主方法一律 `(regress-dir)`。
  6. 启动任何 GPU 任务前确认 `queue` 空闲（不得并发，见下）。

- 建议修正：
  1. `prompt_encoder.py` 的 3 行改动**提交 git 留 diff** 并备份 patch；guard 保持 `direction is not None and getattr(self,"direction_adapter",None) is not None`；若能**子类覆盖 PromptEncoder**（不碰 vendor 源码）则更易审计，权衡后二选一。默认 `None` 对既有 `sam.py/predictor.py` 调用点**逐比特无影响**（分支整段跳过），此论证成立。
  2. 模块② 全部阈值（H/S/V 区间、0.85×V、dev 45°、phi 30°、top30%）**预注册为默认值**，调参只允许在**验证集(58 张/16 场景)**，主表**只报默认参数**；none/random/project 三点除方向外**其余参数完全一致**。
  3. t5 "random" 用**单一固定全局方向**（如 0°）或"**打乱其它图方向**"的 shuffle 对照（更佳，保留方向分布）；若逐图随机，须固定种子 + 固定排序，禁止无种子。
  4. 模块③先 `--freeze-base` 只训 adapter 再联合微调（对应应急预案 D1）；`params` 选择按 `requires_grad` 过滤。
  5. 更新证据索引与 `进度接力.md`，把②③新 CSV/日志纳入。

- 是否允许进入下一阶段 / 放行实现：**否**
- 条件放行路径：完成"必须修正 1–6"并**重新送审**；其中"必须修正 1（含梯度自检）"与"3（axis_box 复用）"是模块③/②能否成立的前提。S1 `direction_utils.py`（纯工具，不涉该缺陷）可在修复方案确定后与其余一并启动。
- 可汇报口径（本轮无新数字可对外）：
  - 沿用上一轮核可：LoRA 无泄漏 **0.7878/0.0644**（best-val，可直报）；U-Net 无泄漏 **0.7414/0.0873**（须带"best-val(epoch50)、58/60 中断"）；SegNet 无泄漏 **0.7403/0.1021**（可直报）；官方 **0.8393/0.0501** 仅作"含 42/43 同源泄漏历史对齐"，禁横比/禁 SOTA。
  - **BER 口径**：维持现状"逐图算再取均值"（结题报告 L63 已声明、已被审计背书）。本轮只**补落盘 TP/FP/FN/TN + 附录给出"全数据集汇总"的 pooling 版 BER 作口径敏感性**，**不擅改主口径**——否则 0.7878/0.0644 及全部已审查 CSV 作废、需全量重跑重审，代价与结题冲刺不匹配。是否切换 pooled 为规范口径，需人类显式决定并单开"重跑+重审"轮次。
- 本次审查 token 估算：约 58k

### 2026-10-14 审查类型：标准审查（第三轮：U-Net 完整60轮新数字 + 模块③死锁修复复核 + 新代码回归，supervisor 子代理执行）

- 范围：① results/unet_leakfree_test.csv(+per_image) 与 logs/unet_leakfree_20261014_164624.log、checkpoints/unet_shadow_leakfree.pth(_last)、backups/*epoch50interrupted*；② direction_utils.py、direction_adapter.py、segment-anything prompt_encoder.py 改动；③ 执行顺序与应急预案.md（第2版）
- 结论：**有条件通过**

- 独立复算/复核（未采信主智能体转述）：
  - **U-Net 新数字属实且口径正确**：
    - 日志 runs 1→60/60 连续（单文件单头部），起 16:46:24、末 epoch60；best-val = **Epoch 30 (val IoU 0.7840)**；best ckpt mtime 17:11 ≈ 起+30 epoch，_last.pth 17:37=训练结束，时间自洽 → **完整 60 轮、无拼接轨迹**（该时间窗内仅此一个 log，无第二次 invocation 的"续训"行）。
    - 脚本 L232-233 最终评估前 load `ckpt`(best) → **口径 = best-val**，与 LoRA/SegNet 无泄漏两行一致。
    - 逐图 CSV 92 行独立重算均值/总体标准差，与汇总两行**六位小数完全相等**（IoU 0.752456/0.092760、BER 0.077060/0.038816…）。
    - 独立复算 split：train 262(68)/val 58(16)/test 92(21)，**三集基名交集 0**；`{unet,lora,segnet}_leakfree_test_per_image.csv` 文件名集合**三者完全相等**（92 行）。
    - 备份齐全：backups/unet_leakfree_test_epoch50interrupted.csv(=0.741381)、_per_image、unet_shadow_leakfree_epoch50interrupted.pth 均在。
  - **零梯度死锁已实质修复（上轮高风险 #1）**：实跑 `python direction_adapter.py --self-test` 复现声明值（恒等 max|Δ|=0.000e+00 autocast/fp32；net.0.weight step1=0→last=4.919e-04 ever>0=True；net.2.weight step1=3.922e-01；3 步参数变化 3.617e-04）。原理核对：改成**纯加性 FiLM**（去 gate），∂out/∂shift=1≠0，末层零初始化仅使**第 1 步**第一层梯度为 0（与 LoRA 的 B=0 同理），第 2 步起恢复；`ever_nonzero` 判据在旧"gate=0+末层零初始化"方案下**全部参数所有步恒 0 → __dead__ 命中 → assert 触发**，**确能抓住上一轮死锁**，非"把自检写松"。
  - **direction_utils.py 数学正确**：实跑自测复现（[1] 20 张 max|Δ|=0.000e+00；[2] 9.671e-07°；[3] 0.00°；[4] pass）。**独立以合成矩形（30/45/60/120/135°）证伪**：`flip_direction_horizontal=(d0,−d1)` 与"水平翻转掩膜再取 GT 方向"偏差全为 0.0000°，对照（错误恒等实现）会偏 60~90° → 变换正确且可证伪。`load_direction_net` 缺权重抛 FileNotFoundError 属实（L216-221）。
  - **无循环 import / 无顶层副作用**：全项目无任何文件 import direction_utils/adapter；被 re-export 的 prompt_experiment/direction_experiment/evaluate_finetuned 顶层 IF 均为 `if __name__=="__main__"` 守卫，且无模块级 build_sam/ckpt 加载 → 上轮"re-export 触发副作用"的低风险**已不成立**。
  - **prompt_encoder 改动无回归**：工作树 diff 与 backups/patches/*.patch **逐字一致**；forward 新增 `direction=None` 关键字参数；两个调用点 `sam.py:106`/`predictor.py:222` 均**位置传参不传 direction** → 分支整段跳过；`utils/onnx.py` **不调用 forward**（只取属性）→ 不受影响。`getattr(self,"direction_adapter",None)` 守卫正确。
  - **文档口径**：软著_软件说明书_初稿.md（16:44）**已切到无泄漏 0.788**（上轮中风险 #4 已消除）；§2.5 "ShadowFormer=阴影去除、指标不可比"**属实**（ShadowFormer 为阴影去除法）；§序8 H1 WHU 修正方向**恰当**（WHU Building 无阴影标注，不宜作阴影检测测试集）。
  - 队列实况：`queue_console.log` 18:19 在跑 **U-Net 官方 ~34/60**（unet_official_20261014_173713.log），GPU 97%/3069MiB，与"leakfree 完成→official 进行中"一致。

- 高风险：**无**（未发现新泄漏/测试集参与训练/指标算错/不公平对比；上轮高风险"死锁"已修复）

- 中风险：
  1. **"0.7525 > 0.7414" 不可作为稳健"提升/替换依据"**：两次均为 from-scratch、seed=0，但逐 epoch loss 从 **Epoch1 起即不同（0.4322 vs 0.4320）并持续发散**（Epoch2 0.3389 vs 0.3452…）；且 `train_cnn_leakfree.py` mtime 16:43 **晚于旧 run(12:20)**、两次日志头不同。→ 或为 GPU 非确定性，或含脚本版本差异；无论哪种，"新>旧"的差值 +0.011 **落在单 run 噪声内**（逐图 std 0.09~0.11，n=92 / 21 场景）。按"新≥旧才替换"机械替换 = **对单 seed 两次运行取优**，有轻微乐观偏置。**须在对外措辞中禁止称"改进/提升"**。
  2. **文档把 99.46% 归给 `direction_resnet18_flip.pth`，与该 ckpt 唯一落盘日志矛盾**：`logs/direction_pretrain_20261013_205735.log` 末行 "最佳验证翻转准确率=**1.0000**，模型已保存"（Epoch 26）；99.46% 仅出现在 **Epoch 15**（末轮 0.9891）。执行顺序与应急预案.md §1.1 L35 / §2.1 L72 的 "99.46%" 与证据不符。
  3. 待替换后的**全库一致性**：结题报告.md L145 与主表(L76-81)、README.md、答辩稿.md、make_ppt.py/metrics_chart.py、软著仍写 U-Net **0.7414**(†)；一旦采用 0.7525，需同步更新并重生成 PNG/PPT，否则文档自相矛盾。

- 低风险：
  1. `direction_utils._self_test` [3]**只打印不 assert、仅用 1 张真实掩膜**（该掩膜恰好非轴对齐 21.9°，故本次仍有效；若换轴对齐掩膜会假通过）→ 建议改合成角度 + assert。
  2. 执行顺序与应急预案.md 内部引用 `§4.7` / `T3.3` 与该文实际结构（§四·序8 H1；§三列 T3.1/T3.2）**对不上**，属编号漂移。
  3. 文档断言"导师论文用 WHU 做无标签预训练(论文3.2节)"——项目内**未见论文 PDF 可供独立核验**；引用保留即可，勿删。
  4. `results/` 仍存 `*_SMOKE1ep.csv`；旧 `summary.csv`(Ours=0.4708) 未清。
  5. 单 seed、无显著性检验（沿用既往）。

- 必须修正（对外汇报 U-Net 数字前）：
  1. **U-Net 限定语**：写明"无泄漏划分、best-val(Epoch30, val IoU 0.7840)、**完整 60 轮 from-scratch**、单 seed、21 个独立场景、未做显著性检验"；**删除**"58/60 中断"（已不成立）但**保留**统计限定。
  2. **禁止**把 0.7525 对 0.7414 表述为"提升/改进/替换更优"；如需说明替换，须注明"两次独立 from-scratch 运行，同 seed 不逐位复现，差值在 run 间噪声内"。
  3. **同步更新全部对外材料**中 U-Net 旧值 0.7414(†(58/60)) → 0.7525，并重生成 metrics_bar.png / 结题答辩PPT。
  4. **更正 99.46% vs 1.0000**：或把 §1.1/§2.1 的翻转准确率改为落盘 best **1.0000**（注明饱和、184 样本），或明确标注"99.46% 为 Epoch15 值/旧 run"。

- 建议修正：
  1. `direction_utils._self_test` [3] 加 `assert dev < 1e-3` + 合成 30/45/135° 用例。
  2. 若要严格复现：`torch.backends.cudnn.deterministic=True`、`benchmark=False`，或每 epoch 重设种子；否则主表注显式声明"GPU 非确定性使同 seed 不可逐位复现"。
  3. 队列跑完官方 U-Net/SegNet 后，同步改附表 ‡（"泄漏+增强bug"→"泄漏"、数字更新）并重生成 PNG/PPT。
  4. 清理 `*_SMOKE1ep.csv` 与旧 summary；更新 进度接力.md。

- 是否允许进入下一阶段 / 放行：**是（附条件）** —— 模块②③新代码无高风险、死锁已修、"新≥旧"数值成立，可**继续在 CPU 上编写 ②③ 脚本**（不占 GPU）；但**对外汇报 U-Net 新数字前须完成"必须修正 1–4"**。队列（GPU）继续跑官方 U-Net/SegNet，勿并发启动 GPU 任务。
- 可汇报口径：
  - **可直报**：LoRA 无泄漏 **0.7878/0.0644**、SegNet 无泄漏 **0.7403/0.1021**（best-val）。
  - **U-Net 无泄漏 0.7525/0.0771**：可报，但**必须**带"best-val(Epoch30)、完整60轮、单 seed、21 场景、未显著性检验"；**不得**称其相对 0.7414 "提升"。
  - 三行仅可表述为"**数值上**高于 CNN 基线；单 seed、未显著性检验、测试集 21 独立场景、差值 < 逐图 std"。
  - 官方 **0.8393/0.0501** 仅作"含 42/43 同源泄漏历史对齐"，禁横比/禁 SOTA。
- 本次审查 token 估算：约 70k

### 2026-10-14 审查类型：标准审查（第四轮：官方 U-Net/SegNet 补跑新数字 + D 阶段长训练放行，supervisor 子代理执行）

- 范围：① 官方划分（含泄漏）U-Net/SegNet 修复增强 bug 后的新数字（0.8153 / 0.8089）及其证据链；② 拟启动的 D 阶段（T1.3 方向特征 Adapter，>1h，约 6h）方案与 runbook
- 结论：**有条件通过**（Part① 数字属实、可对外汇报；Part② D 阶段**不予放行**，须先完成"必须修正 1–4"）

- 独立核验（未采信主智能体转述）：
  - **两个数字属实、口径为 best-val**：`results/{unet,segnet}_official_test_per_image.csv` 各 51 行，独立重算均值/总体标准差，与汇总两行**六项全部 <5e-7 吻合**（U-Net IoU 0.815320±0.058034、BER 0.063961±0.021938；SegNet IoU 0.808902±0.038255、BER 0.070620±0.019437）。
  - **加载的是 best-val 而非 `_last.pth`（核心疑问已查实）**：`train_cnn_leakfree.py` L139 `ckpt={arch}_shadow_{split}.pth` 仅在验证提升时保存（L221-224），L147 `--evaluate` 显式 `torch.load(ckpt)`；两处 evaluate 日志（`logs/unet_official_20261014_202930.log`、`segnet_official_20261014_202940.log`，均为 20:29 落盘）原文写明"评估模式：加载 …unet/segnet_shadow_official.pth（不训练）"。**佐证**：U-Net 训练日志 best 存于 **Epoch60(val 0.7900)**（与末轮重合），SegNet best 存于 **Epoch40(val 0.7742)**；`segnet_shadow_official.pth`(19:34=ep40) 与 `_last.pth`(19:55=ep60) 时间戳确非同一文件 → evaluate 取的是 ep40 的 best-val。三个文件时间戳（18:52 / 19:34 / 19:55）与各自训练起止+轮数**逐一对得上**。
  - **训练期最终评估确被休眠杀掉**：两个训练日志均止于 `Epoch 60/60`（含 `保存最佳`），**无 `=== 测试集 ===` 块** → 与"完成误判"失效模式一致；两 CSV（20:29）系 `--evaluate` 补跑产物，mtime 一致。
  - **口径一致**：训练期最终评估（L232-234）与 `--evaluate`（L147-149）**加载同一 best-val ckpt、调用同一 `eval_full`、评同一 test 集** → 无口径差异，两个数字非 `--evaluate` 引入的偏差。
  - **LoRA 不变可解释**：`train_adapter_v4.py::train_step`(L143-145) 图像与掩膜用**同一次** `random.random()` 判翻转（**从未有**该增强 bug），故修 bug 不动 LoRA；`results/lora_test51.csv` 仍 0.839301/0.050066。CNN 旧值来源为**另一脚本** `train_unet.py`(unet_train.log `Test51 IoU=0.8054`)/`train_segnet.py`(segnet_train.log `0.7799`)——即"旧→新"对比**混淆了脚本重写+bug 修复+run 噪声**，非单变量。
  - **新数字尚未写入任何文档/脚本**：全库 grep `0.8153|0.8089` 于 *.md/*.py **零命中** → 本轮无"文档已污染"风险。
  - preflight 复跑：**36 PASS / 0 FAIL / 1 WARN**（WARN=direction_utils 仅 re-export），与声明一致；`direction_resnet18_flip_basis.pth` 已存在(19:56)、`direction_resnet18_regress.pth` **仍不存在**。

- 高风险：
  1. **D4 联合微调不会 warm-start 自 D3，且 D2/D3/D4 与 D5 输出路径互相覆盖**。`build_sam(args.ckpt=None)` 恒加载 `BASE_CKPT`(LoRA-only)，主流程**无任何** `--ckpt <D3 产物>` 通路；`ckpt_out` 硬编码 `sam_vit_b_shadow_v4_dir_adapter_leakfree.pth`(L327)，CSV 硬编码 `diradapter_leakfree_test{,_per_image}.csv`(L394) 且与 `evaluate_direction_adapter.py` 输出（`diradapter_eval_*`）分离但 D2/D3/D4 三者**共用同一 ckpt/CSV**。后果：(a) 文档所述"先训 adapter 再联合微调"的**课程式策略实为 no-op**（D4 从零训 adapter，D3 训练被丢弃）；(b) D3 结果被 D4 覆盖、**无法留存对比**；(c) D2 冒烟（1 epoch/8 图）同样写生产路径，**污染** `sam_vit_b_shadow_v4_dir_adapter_leakfree.pth` 与结果 CSV，随后 D5 会加载"冒烟模型"。若照 runbook 原样连跑 = **约 3h GPU 白跑 + 产物不可辨**。
  2. **D1 `--check-only` 的守卫基线写错（验证集 vs 测试集口径错配）**。L288-291 在 **`va`（验证集 58 张）** 上评估，而 L293 与 `GPU执行清单_队列跑完后.md` L145、`执行顺序与应急预案.md` L214 均断言"两行 IoU 均 = **0.787821**"。0.787821 是**测试集 92 张**主结果；LoRA-only **验证集** best-val 实为 **0.8128**（`logs/lora_leakfree_20261013_214133.log` Epoch25）。两个 eval 管线（`train_lora_leakfree.eval_box_full`→`method_boxes` 与 `train_direction_adapter.eval_box_full`）逐字等价（同 `mask_enhanced`+`component_boxes(min_area=300)`+同 prompt/decoder 调用），故 check-only 会打印 ≈0.8128 ≠ 0.787821 → 按 runbook"若不等→立即停止"即**假报警阻断**；且该处**只有 log、无 `assert`**（与"断言"表述不符）。这是指定"关键守卫"，基线错会误导排查甚至诱发"乱改模型"。

- 中风险：
  1. **oracle 训练口径与主方法不一致**：D3/D4 计划用 `--direction-source oracle`（GT 方向）训练，而主方法口径应为 `regress`。若以 oracle 训练的 ckpt 去跑 `evaluate_direction_adapter.py`，其 `regress-dir` 行会被脚本标注为"主方法口径"(L9/L79) 实为**训练/推理不一致**；反之若在**评估**用 oracle 则属**测试时使用 GT 信息**（脚本已标"仅诊断/上界" L111，尚可）。主方法须**train+eval 同源=regress**。
  2. **`--freeze-base` 省时未经验证**：image_encoder 前向（冻结、无图）与 decoder 前向/反向（需回溯到 adapter）仍全程执行，仅省 encoder-LoRA 反向与优化器规模；"每 epoch 仍 ~510s"是**断言非实测**。且该项在"#高风险1"未修前无意义（D4 本就不消费 D3）。须用一次计时跑实测。
  3. **附表口径待重写**：`结题报告.md` L104-105 / `README.md` L69-70 现有 `U-Net 0.805‡ / SegNet 0.780‡`（标注"泄漏+增强bug，待重跑"）须替换为新值 0.8153/0.8089，且 `‡` 含义改为"**仅泄漏（增强bug已修复）**"；附表内选模型口径本就异质（LoRA 官方行来自 `_last.pth`，新 U-Net/SegNet 官方行来自 best-val）→"禁横比"须在表注强化。
  4. **无早停**：训练循环恒跑满 `args.epochs`（best-val 已保存但无 break），2×20=6h 无中途依据；"10 轮即收敛"无法在脚本内判定。
  5. `结题报告.md`/`README.md` 主表 U-Net 仍为 **0.7414(†58/60 中断)**，而第三轮已确证较新 leakfree U-Net=**0.7525**（完整60轮）尚未落文档——属既往遗留，仍待办。

- 低风险：
  1. 主智能体提交说明"**`--evaluate` 不写训练日志**"与事实不符：`train_cnn_leakfree.py` L95-97 无条件建日志、L148-164 写入；实测存在 `logs/{unet,segnet}_official_20261014_2029*.log`（评估模式）。不影响结论，但属描述错误。
  2. 旧→新 CNN 对比（0.805→0.8153、0.780→0.8089）**不得**称"提升/改进"（混淆脚本重写+bug 修复+单 seed run 噪声）。
  3. `results/` 仍存 `*_SMOKE1ep.csv`、旧 `summary.csv`(Ours=0.4708)；`results/{unet,segnet}_official_test*.csv` **未登记**入 `AGENTS.md` 证据索引。
  4. 任务书称 B（T1.1）"10 分钟"，而 `GPU执行清单` 阶段 B 自标"约 2h"——时长估计自相矛盾（不影响放行，仅影响排期）。

- 必须修正（D 阶段放行前）：
  1. **修 D1 check-only 口径**：改为在**测试集 `te`** 评估（使 0.787821 成立），或把期望值改为 LoRA-only **验证集** IoU（≈0.8128，注 best-val@Epoch25）；并在脚本内加**硬 `assert |IoU-基线|<1e-4`** 使守卫真正生效；同步改 `GPU执行清单_队列跑完后.md` L145 与 `执行顺序与应急预案.md` L214。
  2. **修 D2/D3/D4/D5 产物路径与 D4 冷启动**：明确 D3↔D4 关系——若为课程式，D4 须 `--ckpt <D3 产物>` 且 D3/D4 写**不同** ckpt/CSV 路径；若为两个独立实验，也必须分路径落盘；**冒烟 D2 必须写冒烟专用路径**，禁止覆盖生产 ckpt/CSV。
  3. **主方法训练口径改 `--direction-source regress`**（依赖 B/T1.1 产出 `direction_resnet18_regress.pth`，现不存在；`load_direction_net` 缺权重会抛异常）；oracle 仅作诊断/上界，**禁止**用 oracle 训练的 ckpt 的 "regress-dir" 行充当主方法。
  4. **先做 B（T1.1）再做 D**（runbook 顺序本就如此）；B 产出 regress 权重且角度误差达标后再启动 D。
  5. 对外材料：新官方数字写**附表**、标"含 42/43 同源泄漏（增强bug 已修复）"、替换 `‡` 旧值、强化禁横比；**禁止**与主表或文献横比、**禁止**作 SOTA/干净泛化。

- 建议修正：
  1. D 先跑 **10 epoch pilot** 看 val 曲线（或加 patience 早停），再决定是否补到 20；避免 6h 盲跑。
  2. D2 冒烟改用**一次计时跑**实测 `--freeze-base` 与联合两种模式 epoch 用时，验证"省时"假设；smoke 须打印中间统计量（得分/剔除数）。
  3. 把 `results/{unet,segnet}_official_test*.csv` 登记进 `AGENTS.md` 证据索引；顺带清理 SMOKE/旧 summary。
  4. 附表加"选模型口径（官方 LoRA=last；官方 U-Net/SegNet=best-val）"注，明示不可横比。

- 是否允许进入下一阶段 / 放行：
  - **Part①（官方新数字）可对外汇报**，须带口径（见"可汇报口径"）。
  - **Part②（D 阶段）不予放行**：须先完成"必须修正 1–4"并**重新送审**；其中 #2（路径/warm-start）与 #3（regress 口径）是 D 能否成立的前提，#1 是守卫能否生效的前提。
- 可汇报口径（本次核可）：
  - **可直报**：LoRA 无泄漏 **0.7878/0.0644**、SegNet 无泄漏 **0.7403/0.1021**（best-val）；U-Net 无泄漏 **0.7525/0.0771**（须带"best-val(Epoch30)、完整60轮、单 seed、21 场景"；不得称相对 0.7414 "提升"）。
  - **附表（官方，含泄漏）**：U-Net **0.8153/0.0640**、SegNet **0.8089/0.0706**（best-val、完整60轮）——仅作"含 42/43 同源泄漏历史对齐"，**禁横比/禁作干净泛化/SOTA**；官方 LoRA **0.8393/0.0501** 同。
  - 三行主表仅可表述为"**数值上**高于 CNN 基线；单 seed、未显著性检验、21 独立场景、差值 < 逐图 std"。
- 本次审查 token 估算：约 95k

### 2026-10-14 审查类型：深度审查（第五轮：T1.1 方向回归"负结果" + oracle 上界诊断方案，supervisor 子代理执行）

- 范围：① T1.1 方向回归头"负结果"及其 5 个实验 + 物理推断；② 拟采取的 oracle 上界诊断方案（C 阶段方向提示 + D 阶段方向 Adapter）
- 结论：**不通过** —— T1.1 "任务不可学"这一结论**证据缺失且逻辑不成立**，不得对外；oracle 方案**有条件放行**但须缩减规模并先修 2 处。

- 独立复核（未采信主智能体转述，逐项去文件找证据）：

  | 声明 | 落盘证据 | 实际发现 | 判定 |
  |---|---|---|---|
  | 实验1 验证角度误差 25.36° | `logs/direction_regress_20261014_203539.log` L44/L47；`logs/regress_console.log` L45/L48；ckpt `direction_resnet18_regress.pth`(20:37, 44.8MB) | val=25.36°(n=58)，**测试=21.07°(n=92)**，best-val 恰在末轮 Epoch40 | **属实**（但测试数被漏报） |
  | 实验2 常数基线 32.40°(n=58) | 无 | 全库（*.py/*.md/*.txt/*.log/*.csv）grep `32.4` 零命中 | **无法验证** |
  | 实验3 连通域中位20 / R中位0.341 / 79%<0.5 / 36%<0.3 / Rg=0.160 | 无 | grep `0.341`/`0.160`/`连通域`/`角度一致性` 零命中；无诊断脚本 | **无法验证** |
  | 实验4 长宽比≥2:29.45° / ≥3:35.96° | 无 | grep `29.45`/`35.96`/`长宽比` 零命中 | **无法验证** |
  | 实验5 "训练误差35~40° ≥ 验证误差"（自定为决定性判据） | 无 | `train_direction_regress.py` **根本没有训练集角度误差分支**：`evaluate_angle` 仅对 val(L199)/test(L210) 调用，训练循环只打印 loss(L197) | **无法验证 + 逻辑不成立** |
  | 物理推断（224 中心裁剪看不到投射物—阴影配对） | 无 | 纯假设，无对照实验 | **无证据** |

- 高风险：
  1. **实验2~5 四项诊断全部无落盘证据**，仅存在于会话声明。按项目铁律"找不到证据=未完成"，**不得写入报告/记忆/对外材料**。其中实验5被主智能体自定为"决定性判据"，却连脚本都不产出该量（`train_direction_regress.py` L197/L199/L210）——属"合理编造"高风险类。证据缺失本身即问题（无可引用文件）。
  2. **"训练误差不低于验证误差 ⇒ 任务不可学"的逻辑不成立**，且存在明确混杂变量未被排除：
     (a) **训练增强与标签口径错配**：`train_direction_regress.py` L80-86 训练变换用 `RandomResizedCrop(224, scale=(0.4,1.0))`（随机子区域裁剪）+ ColorJitter；而标签 d 是**整图**方向（L99 `gt_direction(整张 mask)`）。即"随机裁剪的输入"配"全图标签"，系统性抬高训练误差；评估用 `Resize(256)+CenterCrop(224)`（L88-91）更接近全图 → "训练误差≥验证误差"可由该错配直接解释，**不能**推出"任务学不出"。
     (b) 优化：lr=1e-3 恒定、无 schedule/warmup、40 epoch，训练 loss 仅降到 ~0.40（L44=0.4815）即平台；ResNet18 对 262 张的容量足以记忆，"记不住"更指向标签噪声/增强错配/优化未收，而非"输入无信息"。
     (c) 1-cos 损失在 cos≈0.5（≈60° 二倍角）处梯度正常（非饱和），不构成"梯度消失"解释。
  3. **对外"任务不可学"属过度声明**（同 AUDIT 定义的"合理编造不敏感"盲区）：现有证据只支持"**当前设置下未达 <15~20° 门槛**"，不支持"不可学"。

- 中风险：
  1. **oracle 不是真上界**：oracle=GT 掩膜聚合方向，与训练标签同一噪声源（若 R 中位≈0.34 属实，则标签自身病态）。故"oracle 无提升"只能推"**当前方向定义/信号在本框架下无可测收益**"，**不能**推"方向先验无效"。此区别对结论表述至关重要（主智能体已意识到，予以确认并要求写死措辞）。
  2. **eval 输出路径冲突**：`evaluate_direction_adapter.py` L92/L97 写**固定** `results/diradapter_eval_test{,_per_image}.csv`（**无 tag**）→ D-oracle_only 与后续 D-regress 评估会互相覆盖；且其 `regress-dir` 行会用 **oracle 训练的 ckpt** 评测（train=oracle / eval=regress **口径错配**），却被脚本标注为"主方法口径"(L9/L79) → **禁读作主方法数字**。
  3. **"0.714 < 0.717" 属口径混淆**：该两数来自**旧脚本 + 旧划分（非无泄漏）+ 冻结编码器**（`结题报告.md` L106/L125、`README.md` L71），非新模块② `direction_prompt_eval.py`（**该脚本从未运行**：`results/` 无任何 `dirprompt_*`、`logs/` 无 `dirprompt_*`）。且正确对照是"**方向点提示 0.714 vs 无方向 0.690（+0.024）**"——旧证据其实表明"方向在点提示内**小有益**"，与"方向早期已证无效"叙事**相反**。不能用它替 C/D 论证"早知无效"。
  4. **D 阶段未提 preflight**：D（20×2≈6h）>1h，AGENTS 要求启动前必跑 `preflight_check.py`。当前队列已结束（`queue_console.log` 20:00:08"队列全部结束"）、无 python 进程、GPU 空闲 → 可跑，但**须先 preflight**（缺 `direction_resnet18_flip_basis.pth` 曾会静默回退）。
  5. **样本口径**：val 仅 58 张/**16 个独立场景**，25.36 vs 32.40 的 7° 差落在噪声内（val 曲线 Epoch25→40 剧烈跳动 26.95→28.26→45.22→25.36）；且**测试 21.07° 未在声明中提及**（它才是无泄漏数、且优于 val）→ 只报 25.36 不完整。
  6. `train_direction_regress.py` L122-125 仍保留 **ImageNet 静默回退**分支，与 `direction_utils.load_direction_net`（改为抛 FileNotFoundError，L220-224）不一致；本次因 basis 权重存在（19:56）未触发，但建议同步收紧。

- 低风险：
  1. `logs/regress_console.log` 为 GBK 乱码（与 utf-8 的 `direction_regress_*.log` 内容一致，不影响）。
  2. 最佳模型恰落在**末轮**（Epoch40），val 曲线不稳 → 提示信号弱、不稳。
  3. 阶段1 basis 翻转 best=**1.0000**（饱和、n=58），与"方向回归不可学"并列时易被误读为矛盾（其实一个可学、一个难，属正常；论文 Table 1 亦为翻转判别）。
  4. 文档 99.46% vs 落盘 1.0000 仍不一致（第三轮遗留未修）。

- 对主智能体四问的裁定：
  - **Q2（是"不可学"还是"设置问题"）**：**是"设置问题 + 结论判据无效"，不是"任务不可学已证"**。三个替代解释（随机裁剪增强/全图标签错配、优化配置、标签本身病态）均未被排除，且决定性判据（实验5）无落盘、脚本无该量、逻辑混淆。
  - **Q3（物理推断）**：**无证据**，仅为假设。更简单且被现有线索支持的竞争解释是：**"整图单一全局方向"这一标签本身病态**（若 R 中位 0.34 属实，20 个连通域方向各异，圆形均值近退化）。验证设计（建议）：①关掉 `RandomResizedCrop`（输入与全图标签一致）重训对照；②加大输入/用全图；③**按原图基名汇总该基名下全部子图的连通域后统一取方向**（太阳方位角是**基名级常量**，不是子图级）——预期显著提高 R、降标签噪声，且与 split-by-base 设计自洽。此项主智能体未想到。
  - **Q4（是否与导师论文矛盾）**：**不冲突，辩解成立**。`references/祁昆仑2025_关键英文上下文_DTICT等.txt` 明确论文 Table 1 是"识别水平/垂直翻转的**二分类准确率**"（自监督代理任务），论文**不做角度回归、不预测全局太阳方位角**；我们阶段1 翻转判别同样达 99.46~100%，T1.1 是论文没有的、更难的连续量回归。二者任务不同。

- 对 oracle 方案四问：
  - **Q1**：**判断正确**。oracle 亦噪声，"oracle 无提升"只能否定"**当前方向定义**"，不能否定"方向先验"。
  - **Q2（是否值 3h / 是否可只跑 C+check-only）**：C（提示机制）与 D（特征条件机制）机制不同，**C 不能完全代替 D**；但 check-only **只验恒等初始化、不能**支撑"方向是否可用"的结论。**建议缩减**：先跑 C 全量（便宜、无训练混杂，且其 `axis_box` 行可复现 0.7878 作链路校验）；D 先跑 **`--freeze-base` 短 pilot（≤10 epoch）**看 val 是否移动，再决定是否补满 20+20。
  - **Q3（措辞）**：见"可汇报口径"。
  - **Q4（更值得试的方向定义）**：优先试**基名级方向标签**（见 Q3 验证③）；逐连通域方向意义有限（框提示已编码该几何）；用 SAM 自身特征估方向成本高、收益不明。**不认同**"现有结论已足够"——因"方向无效"当前证据（含 oracle）尚不牢，且被口径混淆污染。

- 必须修正（汇报/放行前）：
  1. **不得汇报"任务不可学"**；T1.1 只按"可汇报口径"表述（现状未达标 + 判据无效）。
  2. 若要保留实验2/3/5：**落盘为脚本产物**（constant-baseline、训练集角度误差、逐图 R），否则**不得作为结论引用**。
  3. 若要给"方向不可学/可行"级结论：**先消除"随机裁剪 vs 全图标签"错配**（关 RandomResizedCrop）重训一次。
  4. D 阶段跑前：`preflight_check.py`；`evaluate_direction_adapter.py` 输出路径**加 tag**；oracle 行不得标注/读作主方法。
  5. oracle 结论按 Q3 措辞。

- 建议修正：
  1. D 缩减为 freeze-base pilot；先出 C 全量。
  2. 试基名级方向标签（Q3③）。
  3. 汇报同时给 val 25.36 与 test 21.07，并声明 val 仅 16 独立场景、曲线不稳。
  4. 收紧 `train_direction_regress.py` L122-125 的 ImageNet 静默回退。

- 是否允许进入下一阶段：**否**（T1.1 现状不得以"不可学"对外）；**oracle 方案有条件放行**：满足"必须修正 4"（preflight + eval 路径 tag + oracle 标注），且按缩减规模（C 全量 + D freeze-base pilot）执行。

- 可汇报口径（本次核可）：
  - T1.1："无泄漏划分(basis 68/16/21)下，方向回归头验证角度误差 **25.36°**、测试 **21.07°**（n=92），**未达**应急预案 <15~20° 门槛；单 seed、验证集仅 16 个独立场景。是否可通过更优输入/标签设计改善**尚未定论**（诊断实验未落盘，且训练增强与标签口径存在错配）。"
  - **禁止**："任务不可学 / 方向回归在 AISD 上根本不可行 / 方向先验无效"。
  - oracle 数字（尚未跑）：仅作"**诊断/上界**"，方法名带 `(oracle-dir)`，**禁入主表/PPT/SOTA**；且须附"oracle 本身含标签噪声，负结果只否定当前方向定义"。

- 本次审查 token 估算：约 62k

### 2026-10-14 审查类型：深度审查（第六轮：T1.1 修正复核 —— 消错配重训 + 标签诊断落盘 + "过拟合"定性，supervisor 子代理执行）

- 范围：① `train_direction_regress.py` 新增 `--no-aug`/`--tag` 后重训（with-aug vs no-aug 两 run）；② `diagnose_direction_labels.py` 及 `results/direction_label_quality.csv`、`logs/direction_label_diag_20261014_210135.log`；③ `results/direction_regress_ablation.csv`；④ `evaluate_direction_adapter.py --tag`；⑤ 修正后的对外结论（"过拟合"）与四问裁定
- 结论：**有条件通过** —— 三处修正**确已落实、证据已落盘**；但把结论**定性为"过拟合而非任务不可学"缺乏证据、属新的过度声明**，且基名级诊断的"证伪"结论含几何/统计混淆。结论须按下方改写后方可对外。

- 独立复核（逐文件核验，未采信主智能体转述）：

  | 声明 | 落盘证据 | 实际发现 | 判定 |
  |---|---|---|---|
  | 消错配已实现 | `train_direction_regress.py` L78/L83-94/L162-165 | `train and not no_aug` → `no_aug=True` 时训练变换 = `Resize(256)+CenterCrop(224)`，与 eval(L137-140) **逐字相同**；错配**确已消除** | **属实** |
  | with-aug val 25.36 / test 21.07 | `logs/direction_regress_20261014_203539.log` L44/L47-48；ckpt(20:37, 44.8MB) | val=25.36(n=58)、test=21.07(n=92)，best-val 恰在末轮 Epoch40 | **属实** |
  | no-aug val 27.23 / test 26.06 | `logs/direction_regress_20261014_210227.log` L39/L47-48；ckpt(21:03) | val=27.23(n=58)、test=26.06(n=92)，best-val 在 Epoch35 | **属实** |
  | with-aug 训练 loss 平台 ~0.40 | 203539.log L5-44 | 平台 0.38~0.58、**末轮 0.4815**（"~0.40"略乐观，实为 ~0.45） | **基本属实** |
  | no-aug 训练 loss 降至 ~0.05 | 210227.log L5-44 | 末轮 0.0513、最低 0.0336，**确能拟合** | **属实** |
  | 汇总 CSV 一致 | `results/direction_regress_ablation.csv` | 两行与两份日志逐项一致（无 seed/epochs 列） | **属实** |
  | 标签诊断落盘 | `results/direction_label_quality.csv`(412 图行+105 基名行)、`logs/direction_label_diag_*.log` | 子图级 R 中位 0.333 / 基名级 0.256，与声明一致 | **属实** |
  | eval 路径加 tag | `evaluate_direction_adapter.py` L42-43/L94/L99 | 确已改为 `diradapter_eval_{tag}.csv / _per_image.csv` | **属实** |
  | 划分无泄漏 | 独立复算 split_by_base | train 262(68)/val 58(16)/test 92(21)，**基名交集 0、文件交集 0** | **属实** |

- 高风险：**无**（未发现新泄漏 / 测试集参与训练 / 角度误差算法错误 / 不公平对比。角度误差 `acos(p·d)/2`、二倍角处理正确；选模型用 val、test 独立；两 run 仅训练变换这一变量不同，seed/lr/epochs/划分/骨干一致）。

- 中风险：
  1. **"过拟合"这一定性证据不足，且把"过拟合"与"任务不可学"设成互斥二选一是假二分**。现有模式（no-aug 可把训练 loss 压到 0.05、val/test 却停在 26~27°）**同时兼容**三种成因：(a) 对标签噪声的过拟合；(b) **标签噪声下限**（训练/验证标签各自含噪 → 即便完美模型 val 也不降，此时"过拟合"与"按现有标签不可学"**同时成立**、非互斥）；(c) 数据量/信息量不足。**当前证据不足以在 (a)(b)(c) 间取舍**。判据缺口：训练循环**只打印 loss，不打印训练集角度误差**（L196-213 无 train `evaluate_angle`）→ "记住了 262 张图"是**由 loss 反推而非直接测量**。
  2. **口径不可比**：声明把"训练 loss"与"验证角度误差"直接对比。二者不同量纲：loss=`mean(1-cos2Δθ)`，误差=`mean(acos(·)/2)`。换算：loss 0.45→≈28°、loss 0.05→≈9°。据此：with-aug 训练误差≈28°≈val 25.36°（**无过拟合间隙**）；no-aug 训练误差≈9° vs val 27.23°（**有约 18° 间隙**）。故"能拟合却不泛化"仅在 no-aug 成立；且 with-aug（加了正则）**并未缩小该间隙到改善 val**（val 25.36 vs 27.23 几乎不动），更像"验证集存在**泛化上限**"而非"调正则即可修复的经典过拟合"。
  3. **缺基线，导致"学不到可泛化规律"这句站不住**。180° 周期角误差的下限/随机水平≈**45°**；val 27.23°、test 26.06° **显著优于 45°** → 模型**学到并泛化了部分方向信号**（只是不足 <20° 门槛）。报告**无常数/圆形均值/随机基线**，故"学不到可泛化的方向规律"是**过度声明**（与"差多少算差"无从判定）。
  4. **基名级"证伪"含混淆**：① R（合成向量长度）是**有偏估计**，样本越少越偏高；子图 n 中位 20、基名 n 中位 63 → 基名级 R 更低**部分只是偏差更小**，不必然等于"同一大图的子图更不一致"。② 更关键：标签 = 阴影连通域**最小外接矩形长轴**，它同时取决于**建筑自身朝向**与太阳方位角 → 跨子图/跨建筑汇总时 R 必然被**几何混杂**压低。故该诊断测的是"阴影轮廓长轴一致性"，**不能干净地证伪"同原图共享太阳方位角"**（它证伪的是"共享轮廓长轴"）。"你的假设被证伪"表述过强。
  5. **单 seed + val 仅 16 独立场景**（已自认）；且 with-aug 的 best-val 恰落末轮、no-aug val 曲线 27→31→27 抖动（L29/34/39）→ 选点不稳，25.36 vs 27.23 的 2° 差在噪声内。
  6. **消融实为两变量**：with-aug 训练同时含 `RandomResizedCrop` **和** `ColorJitter`（L85-86），no-aug 两者皆无 → 严格说"只改一个变量"未满足（针对"裁剪错配"的因果链仍成立，但颜色抖动贡献未分离）。

- 低风险：
  1. `direction_regress_ablation.csv` 无 `seed/epochs/backbone` 列，复现信息不全。
  2. with-aug 测试 21.07° 与门槛 20° 仅差 1.07°，"均未达"成立但属**贴线**；宜写明两 run 的 test 都贴/超门槛。
  3. `train_direction_regress.py` L122-125 的 ImageNet **静默回退**仍在（第五轮已提，未改）。

- 必须修正（对外汇报前）：
  1. **删除"过拟合而非任务不可学"的定性**，改为**只陈述可观测量**："可把训练 loss 压到 0.05（换算训练角度误差≈9°），但 val/test 停在 26~27°，未达 <20° 门槛；**当前证据不能区分**'对噪声过拟合 / 标签噪声下限 / 数据量不足'，亦**不足以判定任务不可学**。"
  2. **补基线**（常数/圆形均值预测器 + 随机）在 val 上的角度误差；否则不得出现"学不到/学不会"这类表述。若基线≈45°、模型≈27°，应如实写"**学到部分泛化信号但不足**"。
  3. **训练集角度误差落盘**（训练循环末加一次 `evaluate_angle(model, tr_files)`）——用以让"能拟合训练集"由**测量**而非由 loss 反推支撑；或至少在报告里把 loss 换算成角度并注明为近似。
  4. **基名级结论改写**：删除"证伪/同原图不共享方向"的强断言，改为"标签一致性在两级都低（子图 0.333 / 基名 0.256）；R 值含样本量偏差，且标签为轮廓长轴、受建筑几何混杂，**该诊断不足以判定太阳方位角是否为大图级常量**"。
  5. **Q3"当前方向定义在 AISD 上不成立"仍属过声明**：R=0.26~0.33 非零、模型优于 45° 随机水平 → 只能写"**该方向标签一致性低、回归头未达标**"，不得写"定义不成立/无效"。

- 建议修正：
  1. 消融报告注明"with-aug 含裁剪+色彩抖动两项 vs no-aug 均无"，或补一档"仅去裁剪、保留 ColourJitter"以分离变量。
  2. 两 run 各跑 3 seed 报均值±std（哪怕 3×40ep 在 GPU 上很快）。
  3. ablation CSV 增列 seed/epochs/lr/backbone 路径。

- 对四问裁定：
  - **Q1（能否对外）**：**暂不能按现稿对外**；完成"必须修正 1–4"后可对外，口径见下。
  - **Q2（"过拟合"是否成立）**：**不成立为"定性结论"**。推理链漏洞：(i) 训练误差非直接测量（只有 loss）且与 val 口径不同量纲；(ii) 缺基线，无法断言"学不到"；(iii)"过拟合"与"标签噪声下限/数据不足"**不互斥**，"过拟合而非不可学"是假二分；(iv) with-aug 加正则**未改善 val**，反而不支持"经典可修复过拟合"。**成立的说法**："no-aug 可拟合训练集而 val 不降"是**过拟合样签名之一**，但成因未定。
  - **Q3（"当前方向定义在 AISD 上不成立"是否够 hard）**：**仍过声明**。见"必须修正 5"。此外诊断本身受"轮廓长轴 ≠ 太阳方位角"的定义性混杂污染，连"定义不成立"都难干净支持。
  - **Q4（下一步）**：**先做"廉价门槛测试"，再决定是否花 6h**。建议顺序 **b → a → c**：
    - **(b') 先跑纯指标、不训练的"新方向标签一致性探针"**（分钟级、不占 GPU）：把方向改从"**阴影质心 → 投射物质心**连线/边界法线/主成分"等**更接近太阳方位角**的量估计，重算基名级 R。**若 R 显著抬升** → 根因是标签定义（不是任务），才值得投 a；**若 R 仍低** → 直接进 c（诚实收尾）。此步最便宜、最能决定后续 6h 是否值得。
    - **(a) C 全量（便宜、无训练）+ D freeze-base pilot（≤10ep）**：可行，但须注意 C/D 用的 **oracle=GT 掩膜聚合方向**与本次被质疑的标签**同源同混杂** → 即便"oracle 无提升"也只能否定"当前方向定义"，**不能**否定"方向先验"（第五轮已定，本轮重申）。故 a 的**信息量受同一标签问题限制**，不宜在 (b') 之前先烧 6h。
    - **(c) 收尾**：合理兜底。T1.1~T1.3 作为"系统探索 + 负/中性结果"的诚实记录、主贡献回归 LoRA，**是可接受且诚实的**，但**前提是结论按上面改写、不夹带"过拟合/不可学"的未证定性**。

- 是否允许进入下一阶段：**否**（"必须修正 1–4"完成前，T1.1 结论不得对外 / 不得进入依赖该结论的 D 阶段）。修正完成且 (b') 门槛测试给出方向后，可重新送审放行 a 或 c。

- 可汇报口径（本次核可，须逐字采用）：
  - "无泄漏划分(basis 68/16/21)下，方向回归头 **验证 25.36°/27.23°、测试 21.07°/26.06°**（有/无随机裁剪两 run，均未达 <20° 门槛）。**消除'随机裁剪 vs 整图标签'错配后**，训练 loss 可由 ~0.45 降到 0.05（换算训练角度误差≈9°），但验证/测试误差**不降**，说明**未达标非该错配所致**。诊断（落盘）：逐图方向一致性 R 中位 **0.333**、基名级 **0.256**——**两级都低**，回归头未达标。**成因（过拟合/标签噪声下限/数据量不足）当前未区分**；单 seed、验证集仅 16 独立场景；方向标签为阴影轮廓长轴、受建筑几何混杂，**其是否为太阳方位角的良好代理尚未定论**。"
  - **禁止**："任务不可学 / 方向回归不可行 / 方向先验无效 / **判定为过拟合（而非不可学）** / 当前方向定义不成立"。后两条为本轮新增的**禁止项**（未证定性）。

- 本次审查 token 估算：约 55k

### 2026-10-14 审查类型：深度审查（第七轮：b' 方向标签探针复核 + 是否跳过 D 的裁定 + T1.1 收尾口径，supervisor 子代理执行）

- 范围：① `probe_direction_labels.py`（含退化检测）与 `results/direction_label_probe.csv`、`logs/direction_label_probe_20261014_{210748,210917,211030,211136}.log`；② 修正后的 T1.1 最终结论；③ 第六轮"必须修正 1–5"落实情况复核；④ 是否同意跳过 D（6h）直接进 (c)；⑤ 顺带核验所提"主贡献 LoRA + DTICT"。
- 结论：**有条件通过** —— 探针是**真实改进**（样本量匹配基线直接回应第六轮质疑）、结论**已删除未证定性**；但 (i) 探针的**信号解释含未排除的混淆**（建筑/网格朝向 vs 太阳方位角），(ii) 第六轮"必须修正 2、3"**未落实**，(iii) 主智能体自述的"bug 症状/数字"与落盘日志**不符**，(iv) 拟作主贡献的 DTICT **无任何落盘证据**。按下方改写后方可对外。**同意跳过 D、进入 (c)**。

- 独立复核（逐文件核验，未采信转述）：

  | 声明 | 落盘证据 | 实际发现 | 判定 |
  |---|---|---|---|
  | 4 估计量观测 R 均值 0.3548/0.3654/0.3373/0.2269 | `results/direction_label_probe.csv` | 逐列与 CSV **逐字一致** | **属实** |
  | 匹配随机基线 0.2117/0.2122/0.2131/0.2123，超出 +0.143/+0.153/+0.124/+0.015 | CSV + `211030.log` L14-18 | 一致；算法正确（按**每图实际连通域数 n** 生成 n 个随机角、重复 200 次）→ **第六轮"样本量偏差"质疑已被实质回应** | **属实（方法学有效）** |
  | 换估计量救不了（最优 pca 0.365 vs 现状 rect 0.355） | CSV | 差 **0.0106** | **属实** |
  | 已加退化检测（前 5 图，同图所有域同角度即抛异常） | `probe_direction_labels.py` L160-171 | 确按声明实现（`comps[:8]`、`std(vals*2)<1e-6` → `RuntimeError`） | **属实（覆盖面窄，见低风险 2）** |
  | 初版 `to_bright`/`edge_grad` 误用整图掩膜 → "R 假性=1.000" | `210748.log` L7-8 + `git diff` | **仅 `edge_grad`=1.000；`to_bright`=0.200（非 1.0）**；diff 证实**两者确都**用过整图掩膜，但症状不同 | **部分不符** |
  | 修正后 R 降到 0.34/0.30 | `210917.log` L7-8 | `edge_grad` 1.000→**0.304**；`to_bright` 仅 0.200→0.205 | **数字对应不准**（0.34 非这两者任一） |
  | 模型 21~27° 优于随机 45° | `203539.log` L44-48 / `210227.log` L39-48 | val 25.36/27.23、test 21.07/26.06 **属实**；"45°"仅落盘于**旧版**探针日志 `210917` L17，**被引用的 `211030` 日志内无该行** | **属实（基线未在被引日志落盘）** |
  | 消错配后 loss 0.45→0.05、val/test 不降 | 两 log | 末轮 loss **0.4815→0.0513**；val 25.36→**27.23**、test 21.07→**26.06**（实则**略升**）；`regress_console.log` L5 记有"已加载翻转预训练骨干" | **属实**（"不降"为保守说法） |
  | 划分无泄漏 basis 68/16/21 | `train_direction_net.py` L55-73 `split_by_base` | 按 `base_name` 分组划分、seed 固定 | **属实** |
  | 主贡献含 DTICT | `train_direction_dtict.py`（有代码） | **全库无 DTICT 的 log/CSV/ckpt**；`执行顺序与应急预案.md` L44 "✅ 冒烟通过（剔除 8/40）"**无落盘证据** | **无法验证** |

- 高风险：**无**（探针与回归均用无泄漏 basis 划分；测试集未参与训练；角度误差 `acos(p·d)/2`、二倍角处理正确；两 run 仅训练变换这一变量不同）。

- 中风险：
  1. **探针"信号"解释含未排除混淆**：探针测的是"同图内各连通域方向估计的**一致性**"（AISD 无太阳方位角 GT 可比）。R 超出匹配随机基线 +0.12~0.15 **同样可由城市建筑/街道网格朝向一致**（阴影轮廓长轴 ≈ 建筑朝向）产生，**未必是太阳方位角**。**反证**：真正指向光源的物理代理 `to_bright` 超出仅 **+0.015（无信号）**——若确有太阳方位角信号，该代理本应最先响应。故"存在真实（方向）信号"应降级为"存在**与当前几何方向标签相关**的弱一致性"，**不得**直接写"太阳方位角信号弱/无"。此项**不改变"换估计量救不了 → 跳过 D"的决策**（两种解释都指向标签/框架限制，而非"多试就能救"）。
  2. **第六轮"必须修正 2"未落实**：结论称"显著优于随机水平 45°"，但**未提供**第六轮明确要求的"常数/圆形均值预测器"在 val 上的角度误差基线（全库 grep 零命中）。
  3. **第六轮"必须修正 3"未落实**：`train_direction_regress.py` 训练循环**仍无** `evaluate_angle(model, tr_files)`（L196-213 无、L215-226 只评估 val/test）→ "训练 loss 压到 0.05"仍**由 loss 反推**，且未按第六轮要求补"（换算训练角度误差≈9°，近似）"的量纲说明。
  4. **"DTICT 作主贡献"缺证据**：`train_direction_dtict.py` 仅代码存在，**无落盘 log/CSV/ckpt**，`执行顺序与应急预案.md` 的"冒烟通过（8/40）"无法验证 → 按 AUDIT"找不到证据=未完成"，**不得**将 DTICT 与 LoRA 并列为主贡献。

- 低风险：
  1. **自述 bug 症状不准**（复核表第 5、6 行）：仅 `edge_grad` 曾 R=1.000，`to_bright` 未曾；"0.34/0.30"对应不准。不影响结论。
  2. **退化检测覆盖面窄**：仅查前 5 图、且只在"整域同角"型 bug 触发（对 `to_bright` 型——逐域质心各异——不触发）。报告宜注明其为"针对整图掩膜型 bug 的护栏"。
  3. **引用日志 ≠ 产出 CSV 的日志**：`results/direction_label_probe.csv` mtime 21:12 = `211136` run 写出，主智能体引用的是 `211030` run。四次 run 覆盖同一 CSV；因固定 seed（`RandomState(0)`）内容逐字相同，**无实质影响**，但应写明。
  4. `train_direction_regress.py` L120-128 ImageNet **静默回退**仍在（第五/六轮已提）。**但实际两 run 无回退**：`checkpoints/direction_resnet18_flip_basis.pth`(19:56) 早于两次 run(20:35/21:02)，且 `regress_console.log` L5 stdout 有"已加载翻转预训练骨干" → 结果不受影响。

- 必须修正（对外汇报前）：
  1. **降低信号解释强度**：把"存在真实但很弱的方向信号"改为"存在**与当前几何方向标签相关**的弱一致性（R 超出匹配随机基线 +0.12~0.15）；**不可归因于太阳方位角**——指向光源的物理代理 `to_bright` 无超出（+0.015）"，并在限制中补"未区分'太阳方位角共享'与'建筑/网格朝向共享'"。
  2. **补第六轮"必须修正 2/3"之一**：(a) 加"常数预测器"在 val 的角度误差基线；**或** (b) 若暂不做，须在结论中**显式声明**"基线仅用理论随机水平 45°、未跑常数预测器基线"，并把 45° 的落盘引到实际含该值的日志（`210917`）。训练误差同理：要么补测、要么把 loss 换算角度并注明"近似、由 loss 反推"。
  3. **更正 bug 叙述**：写"初版 `to_bright`/`edge_grad` **均**误用整图掩膜，其中 **`edge_grad` 因整域同角使 R 假性=1.000**（`to_bright` 因保留逐域质心未致 1.000，仅 0.200→0.205）"，与落盘一致。
  4. **DTICT 不得作主贡献**：除非补出落盘冒烟日志/结果；否则只能写"已实现脚本、状态未验证"，主贡献仅 LoRA（0.7878/0.0644）。

- 建议修正：
  1. 为 `to_bright` 增一个设计有效性小验证（可视化环内最亮块是否为受光面），以判定其无信号是"无方位角信号"还是"该代理本身不可靠"。
  2. `direction_regress_ablation.csv` 补 seed/epochs/lr/backbone 列（第六轮遗留）。
  3. 将 regress 等脚本的 ImageNet 回退改为 assert/raise（对应 AGENTS.md 失效模式 #2）。

- 对四问裁定：
  - **Q1（能否对外）**：完成"必须修正 1–3"后可对外；口径见下。
  - **Q2（是否同意跳过 D、直接进 c）**：**同意**。理由：(i) 第六轮既定路径"b' → 若 R 仍低则直接进 c"，本轮 R 仍低（最优仅高 0.01）→ 触发该规则；(ii) D 的 oracle 与训练标签**同源同混杂**（第五轮已定）→ 即便 D 无提升也只能否定"当前方向定义"、不能否定"方向先验"；(iii) 省 6h GPU 的边际收益 > D 的受限信息。**唯一保留**：跳过 D 是"信息量受限下的经济决策"，报告中**不得**据此宣称"方向先验无效/任务不可学"。
  - **Q3（报告/答辩如何定位方向模块）**：定位为**方法学贡献**——"提出并落盘'方向标签一致性探针'，量化了方向先验在 AISD 上的可辨识度：几何/光度估计量的一致性仅略高于样本量匹配的随机基线（+0.12~0.15），物理方位代理 `to_bright` 无超出（+0.015）"，作为"负/中性结果 + 量化"如实呈现，**非失败**；主贡献仍为无泄漏划分下的 LoRA-SAM。
  - **Q4（自评置信度）**：本结论基于逐文件独立核验，置信 **4/5**（唯一不确定项：`to_bright` 无信号究竟源于"无方位角信号"还是"代理不可靠"，建议按建议 1 补验证）。

- 可汇报口径（逐字）：
  "无泄漏划分(basis 68/16/21)下，方向回归头**验证 25.36°/27.23°、测试 21.07°/26.06°**（有/无随机裁剪两 run），均未达 <20° 门槛。消除'随机裁剪 vs 整图标签'错配后，训练 loss 可由 ~0.45 降至 0.05（换算训练角度误差≈9°，**系由 loss 反推的近似**），但验证/测试误差**不降**，说明未达标非该错配所致。为定位成因，我们提出并落盘'方向标签一致性探针'（不训练）：4 种方向估计量的逐图一致性 R 均值 0.23~0.37，**样本量匹配的随机基线为 0.212**；其中几何类估计量（rect_axis/pca_axis/edge_grad）超出 +0.12~0.15，**物理方位代理 to_bright 仅超出 +0.015（无信号）** → 换方向估计量未能改善（最优仅高 0.01）；该一致性**不足以判定其来源为太阳方位角共享（亦可能来自建筑/网格朝向一致）**。模型角度误差 21~27° 优于理论随机水平 45°，说明学到了**与当前标签相关**的部分信号，但不足以支撑 <20° 精度。限制：单 seed；验证集仅 16 独立场景；成因（标签噪声下限/数据量/模型容量/优化）未区分；**未跑常数预测器基线、训练集角度误差未直接测量**。据此方向模块定位为'对方向先验在 AISD 上可辨识度的量化探索'，主贡献仍为无泄漏划分下 LoRA-SAM（IoU 0.7878 / BER 0.0644）。"
- 禁止用语："任务不可学 / 方向回归不可行 / 方向先验无效 / 当前方向定义不成立 / 判定为过拟合"；**本轮新增**："太阳方位角信号弱/无"（未与'建筑朝向一致'区分）、"方向先验在 AISD 上无效"、把 21~27° 称为"显著优于/大幅优于随机"、将 DTICT 与 LoRA 并列主贡献。

- 是否允许进入下一阶段：**是**（进入 (c) 收尾、主贡献回归 LoRA）；**但对外汇报 T1.1 结论前须先落实"必须修正 1–4"**。

- 本次审查 token 估算：约 62k

### 2026-10-14 审查类型：标准审查（第八轮：模块② 方向引导提示评估方案——oracle 扇形提示首跑，supervisor 子代理执行）

- 范围：拟执行方案 `direction_prompt_eval.py --direction-source oracle`（sector_source=global/component 两变体 + 5 方法一次遍历 ≈40min）；脚本 `direction_prompt.py`/`direction_prompt_eval.py`/`direction_utils.py`/`direction_experiment.py`/`evaluate_lora.py`/`prompt_experiment.py`/`eval_shadow.py`/`train_lora_leakfree.py`；`results/lora_leakfree_test.csv`、`lora_test51.csv`、`direction_label_probe.csv`；GPU/进程状态
- 结论：**有条件通过**（执行放行，须先落实"必须修正 1–4"再跑全量并判读）

- 独立复核（逐文件核验，未采信转述）：
  - **划分一致**：`direction_prompt_eval.leakfree_split()` = `split_by_base(files,0.2,seed=0)` 再 `split_by_base(train_pool,0.2,seed=1)`，与 `train_lora_leakfree.py`L96-97 同序同参 → 同一 test 92 张。**属实**
  - **脚本从未运行**：`results/` 与 `logs/` 均无任何 `dirprompt_*`（mtime 21:38 为最后改动）。**属实**
  - **依赖可解析**：`methods.mask_enhanced/mask_c3i/_rm_small` 均存在（L66/L33/L16）；`prompt_experiment`/`evaluate_finetuned` 无模块级建模型副作用 → 可 import，不会一启动即崩。
  - **指标口径**：`compute_metrics`（eval_shadow.py L14-55）TP/FP/FN/TN、`BER=0.5(FPR+FNR)` 正确。
  - **参考行链路**：`direction_prompt_eval` 的 `axis_box` = `mask_enhanced` + `component_boxes(em,300)` + 逐框 `predict_box_single`(与 `predict_box` 同逻辑) + OR，等价于 `train_lora_leakfree.eval_box_full→method_boxes→component_boxes(默认300)`；同 ckpt `sam_vit_b_shadow_v4_lora_leakfree.pth`、同 test 92。
  - **GPU 空闲**：nvidia-smi 29%/920MiB、无 python 进程；`queue_console.log` 止于 20:00。可跑（但仍须启动前再确认）。

- **Q4 裁定（axis_box 参考值）**：**写 0.7878 正确**。本脚本跑**无泄漏 test 92 + 无泄漏 best-val 权重**，对应 `results/lora_leakfree_test.csv` 的 0.787821（`AGENTS.md` 证据索引亦记此值）；`evaluate_lora.py` 的 0.8393 是 **Test51 官方（含 42/43 泄漏）+ `_last.pth`**，不适用。注意：须用**容差（≈1e-4，cuDNN 非严格确定性）**，不可要求逐位相等；若未复现 → 判为链路断裂，**立即停止、不得汇报 project 行**。

- 高风险：
  1. **判读规则含"对比不公平"（两变量同时变）**：`project(sector_source=global)` 相对 `no-dir` 同时改了 (i) 扇形布点 vs 均匀布点、(ii) **方向一致性过滤**（`dev_thresh_deg=45°`，`direction_prompt.py`L194 丢弃主轴偏离全局方向>45°的连通域；`no-dir`/`random` 均无此过滤）。故"project>no-dir ⇒ 方向提示有用"**不成立**——增益可能全部来自过滤而非方向；project(global) vs random 同样混杂。该过滤属项目**自加项**、不在申报书"HSV/梯度剔除伪阴影"描述内，进一步说明它把非方向效应混入。**补齐对照前，禁止据 project 行下"方向提示有用/项目核心主张成立"结论。**

- 中风险：
  1. **oracle-only，缺主方法口径**：方案仅 `--direction-source oracle`（GT 方向）→ 只能作"**诊断/上界**"，方法名须 `(oracle-dir)`，**禁入主表/PPT/SOTA**。要支撑项目方法须补 `--direction-source regress`（`direction_resnet18_regress.pth` 已存在，val≈25~28°）；oracle 先行顺序正确，但**不得把 oracle 结果当项目方法结论**。
  2. **冒烟不会暴露"近乎失效"实现**（AGENTS 失效模式 #4/#5）：脚本仅逐图打印 IoU，**不打印中间量**（每域候选点数、in-mask 丢弃数、dev 过滤丢弃域数、HSV 剔除数、兜底触发数）。`--limit 3` 只能验"跑通"，不能验"方法未退化"→ 须先补中间量再冒烟。
  3. **`n_fallback` 计数错误（静默降级未被记录）**：`project` 过滤后前景为 0 会**静默退用 `none`**（direction_prompt.py L250-254）；但计数器判据 `(l==1).sum()==0`（eval L216）在"已兜底"时为 False（兜底后的 `none` 通常有前景点）→ **统计不到真正兜底的图**。日志行"project 模式触发无方向兜底的图像数"**失实**；而兜底会**稀释方向效应**，必须正确落盘。
  4. **`component` 变体样本选择仍依赖 GT**：扇形角取自粗掩膜（不读 GT，**不构成泄漏**），但评估主循环对**所有**方法先算 `gt_direction(gt)`、为 None 则 `continue`（eval L189-193）→ component 的**评测图集由 GT 属性决定**（轻微 GT 依赖）。要作干净变体须绕过该门控。
  5. **`--direction-source random` 会崩**：src=random 时 `d=None`，`mode=project` 走 `axis_from_vector(None)`（direction_prompt.py L173）→ TypeError（本方案未用该分支；`random` 对照实际由内部 `mode=random` 提供，单固定角、全图同角、可复现）。
  6. **时间/占用未实测**：40min 为估计；每图 `mask_enhanced`/`mask_c3i`/`connectedComponents` 重复多次（em 1 次 + 每 build 各 1 次）。

- 低风险：
  1. `direction_prompt.py`L168 注释"由全局种子+图像尺寸派生"与实际不符（只用 seed）——注释失实。
  2. `direction_prompt_eval.py` docstring 称 `none`"同样点数"，实为"**每域 ≤ n_fg_max(5)** 上限相同、但连通域数不同（project 过滤后更少）"→ **总点数并不同**。
  3. `component` 运行时 `random/none/axis_box/centroid_points` 四行内容与 global 运行完全相同（仅 tag/文件名不同）——无妨，勿当独立证据。
  4. 文档仍引用旧 `direction_experiment` 的 0.714/0.690/0.717（旧脚本+旧划分+冻结编码器，无新实现落盘）：`结题报告.md`L127、`README.md`L71、`执行顺序与应急预案.md`L198（C1）、`GPU执行清单_队列跑完后.md`L128 → 本轮结果出来后须同步更新。

- 对六问裁定：
  1. **是否值得跑**：**值得**。第七轮已明确"先跑 C 全量（便宜、无训练混杂，其 `axis_box` 行作链路校验）"；旧实验（+0.024）来自不同脚本/划分/模型，**不足以**支撑对外结论，且现被 `结题报告.md`L127 引用 → 应用本实现替换。但须先补"必须修正"，否则跑完仍回答不了"方向是否有用"。
  2. **component 是否偏离申报书**：**是**。申报书"结合上述提取的**方向特征向量**"指模块①的**全局**方向；component 用的是**逐连通域 minAreaRect 主轴**（几何量，非模块①输出）→ 属**换概念**。诚实表述：命名"**逐域主轴扇形（几何、无方向网络）**"，定位**附加消融**，不得称"申报书的方向引导提示"。
  3. **对照是否完备**：**不完备**。缺"**global 方向 + 不过滤**"这一格以拆分"方向/扇形"与"过滤"。建议补为四格：(a) global+过滤、(b) global+不过滤、(c) random+不过滤、(d) none；component 作 (e) 附加。判读：**基于 (b) vs (c) 说"方向"**，**基于 (a) vs (b) 说"过滤"**。
  4. **axis_box 参考值**：**0.7878**（见 Q4 裁定），**容差 1e-4**。
  5. **component 是否可进主表**：**不构成泄漏**（扇形角与粗掩膜均来自输入图，不读 GT），**但不可作主方法进主表**——它不是申报书的方向先验方法；只可进**消融/诊断表**并标注"几何主轴、无模块①方向"。
  6. **其他泄漏/不公平**：见中风险 4（样本选择 GT 依赖）、高风险 1（两变量）；另 project 的粗掩膜（`mask_enhanced∩mask_c3i`）≠ 基线 axis_box 用的 `mask_enhanced` → project vs axis_box **非同掩膜**，禁作纯"提示类型"对比。

- 必须修正（跑全量前）：
  1. 加"**global 方向 + 不过滤**"对照（暴露 `--dev-thresh` 或等价开关），使"方向 vs 过滤"可拆；判读规则改为按 (b)vs(c)、(a)vs(b) 拆分表述。
  2. 修 `n_fallback`：在 `build_direction_points` 内显式返回"是否兜底"并**落盘**，日志须给出"兜底图数"。
  3. 冒烟前加**中间量统计**（每图：候选点数、in-mask 数、dev 丢弃域数、HSV 剔除数、最终前景点数），先 `--limit 3` 目视漏斗再全量。
  4. 命名/标签：oracle 行强制 `(oracle-dir)`；component 行标"component-axis（几何、无网络）"；**均禁入主表**。

- 建议修正：
  1. 先做一次**计时跑**（`--limit 3`）核 40min 估计；启动前 `nvidia-smi`/进程确认无并发。
  2. 补 `--direction-source regress` 一次（主方法口径），或结论中显式声明"本轮仅 oracle 诊断，主方法(regress)待补"。
  3. 补 `--dev-thresh`/关闭过滤的 CLI，避免改死代码重跑。
  4. 结果出来后同步更新 `结题报告.md`L127、`执行顺序与应急预案.md`L198、`GPU执行清单_队列跑完后.md`L128。
  5. 若 component 要进表，须让数据流**完全绕过 GT**（含样本选择）。

- 是否允许进入下一阶段 / 放行：**是（附条件）**——可先跑带中间量的诊断冒烟（`--limit 3`）；**全量运行与判读**须先完成"必须修正 1–4"。
- 可汇报口径（本轮无新数字）：一切 project 行**暂不得对外**；补齐对照后仅按 (b)vs(c)/(a)vs(b) 拆分表述，oracle 一律标注"诊断/上界、非主方法"。
- 本次审查 token 估算：约 60k

### 2026-10-14 审查类型：深度审查（第九轮：模块② 方向引导提示全量结果 + "点>框"发现 + 归档清理，supervisor 子代理执行）

- 范围：① `results/dirprompt_*_{oracle,oracle_component}_test.csv(+_per_image)` 与全部 `logs/dirprompt_20261014_2*.log`；② 脚本 `direction_prompt.py` / `direction_prompt_eval.py` / `direction_utils.py` / `prompt_experiment.py` / `eval_shadow.py`；③ 主结果链路 `results/lora_leakfree_test.csv`；④ 归档 `archive/` 与 `preflight_check.py`；⑤ 主智能体自述的"第一次全量异常→重跑"处置。
- 结论：**有条件通过**（模块② 的 oracle 诊断数字**属实且可复现**、链路校验**逐位通过**；但"第一次全量异常"的叙述**与落盘证据不符**，且"点>框"结论用错统计口径——均可修正后对外/继续）。

- 独立核验（逐文件核验，未采信转述）：

  | 声明 | 落盘证据 | 实际发现 | 判定 |
  |---|---|---|---|
  | 六方法 IoU 0.803656/0.802955/0.803413/0.802989/0.802594/0.787821 | `dirprompt_{project,project_nofilter,random,none,centroid_points,axis_box}_oracle_test.csv` | 六个 CSV **逐字一致**；与日志汇总行一致 | **属实** |
  | component 变体 project==nofilter==0.803004 | `dirprompt_project_*_oracle_component_test.csv(+per_image)` | 汇总与逐图**均完全相同**（逐图 diff 无差异） | **属实** |
  | 漏斗 1909 域/丢弃 887/兜底 9/前景 4312 vs 7412 | `logs/dirprompt_20261014_215345.log` L125-128 | 与日志**逐字一致**；887/1909=46.5% | **属实** |
  | `axis_box` 复现主结果 0.787821 | `dirprompt_axis_box_oracle_test.csv` vs `lora_leakfree_test.csv` | **汇总六位小数相同，且逐图 CSV `diff` = 0（逐字节相同）** → 链路校验**无容差即通过** | **属实（强）** |
  | 划分无泄漏 basis 68/16/21 | `direction_prompt_eval.leakfree_split()` L65-70 | 与 `train_lora_leakfree.py` L96-97 同序同参（0.2/seed0 再 0.2/seed1）；92 图 | **属实** |
  | oracle 方法带 `(oracle-dir)` 标注 | CSV method 列 / L148-149 日志 | 方法名含 `oracle-dir`；文件名含 `oracle`；日志有"仅作诊断/上界"告示 | **属实** |
  | 归档零引用、保留被引用文件 | `archive/README.md` + grep | 6 个归档名在 archive/ 外**零引用**；`boxes_tta_full.py`/`interactive_sam.py`/`demo_result.png` 仍被 `demo.py`/`README.md` 引用且**原位存在**；`train_v4_quick.log`/`segnet_train.log`/`unet_train.log` 原位存在 | **属实** |
  | preflight 38 PASS/0 FAIL/1 WARN | 实跑 `python preflight_check.py` | 复现 **38 PASS / 0 FAIL / 1 WARN**，退出码 0 | **属实** |
  | "第一次全量(214604)异常：project==nofilter、丢弃=0，与冒烟矛盾" | `logs/dirprompt_20261014_214604.log` L103-107/125-126 | **214604 实为 `--sector-source component` 运行**（输出 `_oracle_component_*.csv`、汇总名 `component-sector`）；component 下 project==nofilter 且丢弃=0 **正是预期行为**，非异常 | **不成立（叙述错误）** |
  | "改用独立 CPU 脚本复算漏斗" | 全库 | grep `887` 于 `*.py/*.sh/*.md` **零命中**；无任何 funnel/recompute/cpu 脚本 → "独立脚本"**无落盘证据**（887 本身有日志支撑） | **无法验证** |

- **关键发现：所谓"第一次全量异常"并不存在。** 逐条时间线（mtime 全部 ∈ [21:43:07, 21:55:08]，代码冻结于 `direction_prompt_eval.py` 21:44:03 / `direction_prompt.py` 21:43:15，**其后无任何代码改动**）：
  - `214322/214421`（21:43:22/21:44:21）= `--limit 3` 冒烟（global）：project 丢弃 20（`214322`）；
  - **`214445`（21:44:45）= 首次 global 全量 92 张**：project 丢弃 **887**、project≠project_nofilter、汇总 **0.8037**，**完全正确**；
  - **`214604`（21:46:04）= component 全量 92 张**：project==nofilter、丢弃 0，**属 component 预期**；
  - `215214/215226/215300` = `--limit 5` 冒烟（global）；`215345`=global 全量、`215508`=component 全量。
  - **独立 diff 证明可复现**：`214445` 与 `215345`（两次 global 全量）**除时间戳/路径外逐字相同**；`214604` 与 `215508`（两次 component 全量）同样**逐字相同**。
  - 结论：主智能体把 **component 运行（214604）误读为 global 全量**（因日志头**不打印 `sector_source`/`apply_dir_filter`**，仅打印 `sector=axis`），据此以为"global 过滤失效"→ 触发不必要的"重跑+独立脚本复核"。**不存在代码异常；首次 global 全量结果本就正确且可复现。** "数据已作废"之说无依据（该数据与重跑逐字等价）。

- 高风险：**无**（无同源泄漏/测试集参与训练/指标算法错误/不公平对比；oracle 已正确标注为诊断；主结果链路逐位复现；无越权改代码）。

- 中风险：
  1. **"点>框"的统计口径用错（会低估该差异的稳健性，也会误导"是否替换主方法"的判断）**：主智能体用"各方法内部逐图 IoU 的 std(≈0.061~0.069)"作对照，称 +0.015"落在噪声内"。**成对设计的正确分母是逐图差值 std**：实测 point−axis_box 逐图差值 std = **0.0393~0.0397**（配对 t≈3.6~3.8）；按**场景（基名）聚合**（21 场景）差值均值 **+0.0117、std 0.0121、t≈4.4**。→ 该 +0.015 其实是**跨 21 个场景都相当一致**的差距，**不是"噪声内"**。但这**不改变"不可据此替换主方法"的结论**，理由见下条。
  2. **"点>框"结论混杂、且非方向效应**：经查 `axis_box` 与 `centroid_points` **同用 `mask_enhanced`**（`prompt_experiment.component_boxes` 与 `evaluate_finetuned.auto_points` 均接收 `em`）→ 主智能体担心的"掩膜来源不同"对**这一对**不成立；但二者仍差 **min_area(300 vs 100)**、**聚合方式（逐框 OR vs 单一多点集）**、**是否含背景负点** 三处。更关键：**含方向的 `project/nonefilter` 与不含方向的 `none/random/centroid` 提升幅度相同（+0.0148~+0.0158）** → 该差距是**提示类型(点/框)机制效应、与方向无关**；且 **BER 反向**：`axis_box` BER=**0.0644 为全场最低**（点类 0.0656~0.0666）。→ **不得替换主方法**，不得作因果断言。
  3. **冒烟 `--limit` 写生产 CSV 路径（第四轮失效模式复发）**：`--limit 3/5` 运行直接覆盖 `results/dirprompt_project_oracle_test.csv`（曾为 n=3、n=5），后被全量运行还原为 n=92。当前落盘无误，但路径未隔离，存在"误读中间态/覆盖正式产物"风险。
  4. **"独立 CPU 复算脚本"无落盘证据**：见上表末行。876/887 有日志背书，但"独立脚本复核 correct"这一过程性声明无法验证。
  5. **模块② 仅 oracle 口径**：主方法口径 `regress`（`--direction-source regress`）**本轮未跑**，故所有 `DirPrompt(...)` 行均为**诊断/上界**，不得进主表/PPT/SOTA。结论只能表述为"方向机制（上界）无增益"，**不能**表述为"方向先验无效"（第七轮已定）。

- 低风险：
  1. 日志头缺 `sector_source`/`apply_dir_filter`（本轮混淆的**根因**）→ 建议补印。
  2. `archive/old_sam_dir/segment-anything/__pycache__` 仍随归档保留（"删除 __pycache__" 不彻底；无害、可再生）。
  3. 根目录 `__pycache__` 现存在（import 即再生；无害）。
  4. component 行里的 `random/none/axis_box/centroid_points` 与 global 运行数值相同，**非独立证据**（主智能体已知，勿重复引用）。
  5. 对照分解的结论（(1)−0.00003、(2)+0.0007）**正确**，但均 < 0.001 IoU，应显式标注"小于任何合理显著性门槛"。

- 必须修正（对外汇报/继续前）：
  1. **更正本轮叙述**：删除"第一次全量异常、数据作废"，改为"`214445`(21:44:45) 首次 global 全量结果正确、与 `215345` 重跑逐字一致；此前观察到的 `project==nofilter`/丢弃=0 来自 **component 变体（214604）**，属该变体预期，非缺陷"。**不追究"代码异常"（不存在）**；但须追究根因。
  2. **日志头补 `sector_source` 与 `apply_dir_filter`**（防止再次 global/component 混淆）。
  3. **对外口径**：oracle 行**严禁入主表/PPT/SOTA**；模块② 只作"**诊断：即便给出 oracle 方向上界，方向引导扇形提示相对无方向对照无可测 IoU 增益（(1)−0.00003、(2)+0.0007）；固定随机方向亦无差异**"，并附"单 seed、21 独立场景"。
  4. **"点>框"不得替换主方法**；若要主张，须补**同源同聚合的干净消融**（同 `mask_enhanced`、同组件集、同聚合方式，**仅改提示类型**点 vs 框，且分别报 IoU 与 BER）——低成本、无需训练。
  5. **冒烟输出路径与生产分离**（`--limit` 写 `results/_smoke/` 或加 tag）。

- 建议修正：
  1. 若保留"点提示更优"作为**改进线索**，请额外报告其 **BER**：框提示 BER 最低(0.0644)，故"更优"仅限 IoU，且与申报书核心指标 BER 方向相反——须并列呈现。
  2. 补跑 `--direction-source regress` 一次（主方法口径），使模块② 有可进主表的合法行（依赖 `direction_resnet18_regress.pth`）。
  3. 顺带核 `--limit` 产物是否已全部被全量覆盖（抽查 6 个 CSV 的 n_images 均为 92：**已确认**）。

- 对第二/三/四节请求的裁定：
  - **第二节（处置是否恰当）**：处置**操作上安全**（重跑无害、结果可复现），但**前提错误**——**不存在异常**；214604 的"全同/丢弃0"是 component 变体的正确行为；真正的首次 global 全量(214445)本就正确。**须追究根因（日志头缺 sector_source），更正叙述**，无需追责"代码异常"。
  - **第三节 Q1（点>框是否可信）**：**数值层面可信**（跨 21 场景一致，场景级 t≈4.4），但**其成因非方向、且混杂未除（min_area/聚合/负点）、BER 反向**。主智能体提出的"掩膜来源不同"对 `centroid_points vs axis_box` **不成立**（同为 `mask_enhanced`），反而该对照比其自陈**更干净**、更可信。
  - **第三节 Q2（是否 > 逐图 std）**：**该问法用错了数**——成对比较应比"逐图差值 std"(0.0397) 而非"各方法内部 std"(0.069)；按正确口径，+0.015 **≈ 0.4 个差值 std、场景级 t≈4.4**，**比"落在噪声内"更稳健**。
  - **第三节 Q3（换不换）**：**不换**。理由：混杂未除、非方向效应、BER 反向、单 seed、且现有主结果链路已逐位验证；机械套用"新≥旧才替换"属**对不干净对照取优**。
  - **第四节（归档副作用）**：**无证据链/可复现性破坏**——归档文件零引用、被引用文件原位保留、主结果 CSV 逐位复现、preflight 38 PASS。仅有的瑕疵是归档内残留一个 `__pycache__`（无害）。

- 是否允许进入下一阶段：**有条件通过** —— 模块② 诊断结果**可作诊断/负结果对外**（须带"仅 oracle 上界、单 seed、21 场景、方法名带 oracle-dir"），**但**：(a) 完成必须修正 1–2 前，**不得**把本轮"异常→重跑"叙述写进任何文档/记忆；(b) **不得**据"点>框"改主方法；(c) 模块② 结论**不得**读作"方向先验无效"。
- 可汇报口径（逐字，须含标签）：
  "无泄漏划分(basis 68/16/21，test 92 张/**21 个独立场景**，单 seed)下，在无泄漏 LoRA-SAM(框提示，IoU 0.787821) 基座上评估'方向引导扇形提示'（**方向取自 GT 掩膜，仅作诊断/上界**）：(i) '扇形(方向)' 净作用 = `project_nofilter − none` = **−0.00003 IoU**；(ii) '方向一致性过滤' 净作用 = `project − project_nofilter` = **+0.0007 IoU**；(iii) 固定随机方向与无方向亦无差异（`random − none` = +0.0004）。即**即便给出 oracle 方向上界，该方向引导提示相对无方向对照无可测 IoU 增益**。旁证：点提示类方法 IoU(0.8026~0.8037) 数值上高于框提示基线(0.787821)，但该差距**与方向无关**（无方向对照同样高出）、且 **BER 反向**（框提示 BER 0.0644 为最低），暂不作结论。"
- 禁止用语："方向引导提示有效/提升 IoU"；"点提示优于框提示"（作因果/替换依据）；把 oracle 行写入主表/PPT/SOTA；"方向先验无效/任务不可学"（沿用第七轮禁令）；"第一次全量异常/数据作废"。
- 本次审查范围：见首行。
- 本次审查 token 估算：约 90k

### 2026-10-14 审查类型：标准审查（第十轮：BER 口径敏感性 + 场景级配对检验，及"是否升级对外口径"的裁定，supervisor 子代理执行）

- 范围：`ber_sensitivity.py` / `results/ber_sensitivity.csv` / `logs/ber_sensitivity_20261014_221627.log`；`scene_ci.py` / `results/scene_level_ci.csv` / `logs/scene_ci_20261014_{221707,221713}.log`；三方法 leakfree 逐图 CSV 与官方逐图 CSV；"把对外口径由'不足以断言显著优于'升级为'显著优于'"的裁定；顺带复核 §四（日志头 / smoke 隔离 / preflight）
- 结论：**有条件通过**（两项分析**数值真实、可逐位复现**；但"升级为显著优于"**不予核准**——存在口径与统计两处过度声明；BER 敏感性可作附录）

- 独立复算（未采信转述，逐文件核对）：
  - **scene_ci.csv 逐位复现**：独立按【原始大图基名】聚合三方法（21 场景 / 92 图），重算场景级 IoU/BER 均值±std、配对差、bootstrap 95%CI(seed=0,20000)、配对 t —— 与 CSV/日志**六位小数全部一致**（LoRA 0.799392±0.044561、U-Net 0.749270±0.068718、SegNet 0.745889±0.067911；LoRA−U-Net +0.050122 [+0.026425,+0.075906] t=3.891）。`scene_ci.py` L42-44 `base_of` 正则 `(.+?)_sub\d+` 对全部 92 名解析正确（21 基名）。
  - **ber_sensitivity.csv 复现**：独立以 (N,OA,P,R) 反解 TP/FP/FN/TN 后汇总，六方法 pooled IoU/BER 与 CSV/日志**逐位一致**（LoRA leakfree 0.776167/0.065637 等）；总像素 26,015,616(=92 图)/17,015,808(=51 图) 与日志一致；反解无负计数（sanity 通过）。
  - **§四 复核**：`preflight_check.py` 复跑 **38 PASS / 0 FAIL**（与声明一致）；`direction_prompt_eval.py` L139-143 确已加 tag(`src[_component]`) 与 `_smoke{N}` 后缀（第九轮必须修正 1、5 已落实）；L155 确已打印 `sector_source`（必须修正 2 已落实）；6 个 `dirprompt_*_oracle_test.csv` 的 n_images 均=92（生产未被冒烟污染）。
  - **pairing 有效**：三方法同 test 92 图、同场景配对，独立单元=21 场景成立。

- 高风险：**无**（无同源泄漏 / 测试集参与训练 / 指标算法错误 / 不公平对比；两项分析均可逐位复现；未越权改代码）。

- 中风险：
  1. **拟把口径升级为"显著优于"属过度声明（不予核准）**，两处独立理由：
     (i) **效应量随口径 10 倍级摆动**：LoRA−U-Net 的 ΔIoU 在 pooled(文献口径)=**+0.005**、macro(现行主口径)=**+0.035**、scene(最有利口径)=**+0.050**（独立算自 `ber_sensitivity.csv`/`scene_level_ci.csv`）。`scene_ci` 恰采用**最有利口径**；其 +0.050 与主表 macro 数字(0.7878−0.7525=+0.035)并列时**无法自洽**。**尤其**：pooled 口径下 LoRA 相对 U-Net 仅 **+0.005 IoU（≈无差异）**——即"IoU 显著优于"高度依赖口径选择。
     (ii) **CI 未含 run-to-run(seed) 变异**：95%CI 仅刻画**场景抽样**不确定性(n=21)，以单次训练结果为固定效应；项目自身失效模式 #9 已记录"同 seed from-scratch 不逐位复现、差值 ~0.011 落在 run 间噪声内" → 该检验**条件于单 seed**，不能支撑"显著优于"的一般陈述。（依 t 分布 df=20 复核：p=0.0009 仍显著，verdict 不变；故问题不在"是否显著"，而在**不能据此取消 seed 口径限定**。）**建议保留**"不足以断言显著优于"的**保守口令**，改为**陈述检验事实**（见可汇报口径）；"显著优于"一词是否解禁须**人类明确放行**（现 `experiment-results` 记忆铁律明令禁用该词）。
  2. **场景级聚合改变估计量、含"等权小场景"偏置**（回应 Q3c）：21 场景规模 1~11 张不等（独立统计）；场景级均值对**小场景**(vienna28 n=1) 与**大场景**(chicago16 n=11) **等权**，与 macro(子图等权)、pooled(像素等权)是**三个不同 estimand**——并非"错误叠加"，而是**不可混用**；LoRA 场景级↑0.7994 / U-Net 场景级↓0.7493 的方向差**正是重加权的产物**，**不**证明"子图级比较有系统性错误"。结论：场景级**仅可用于推断(CI/检验)**，**不得替换**主表 macro 点估计。
  3. **配对检验未做多重比较校正**（3 组两两比较）；且 21 场景由固定 seed=0 划分决定、**非随机抽样**，t 检验是对"场景总体"的模型外推（标准做法但需声明）。

- 低风险：
  1. **BER 敏感性的"反解校验"偏弱**：以 (N,OA,P,R) 解 TP/FP/FN/TN 为 4 元 4 方程、**解唯一** → IoU/BER **由 OA/P/R 代数决定**，"失败数=0"基本是**代码自洽性检查**（能抓 `metrics_from_counts` 写错），**非对数据真值的独立验证**。pooled 计算本身有效，但表述宜降调。
  2. **`scene_ci_20261014_221707.log` 为未完成运行**（止于"=== 配对比较 ==="，无 [OK]、无 CSV），随后 221713 重跑覆盖；产物无影响，但遗留半截日志易误读。
  3. **记忆文件数字漂移**：`experiment-results.md` 主表仍写 U-Net **0.7414/0.0873**，而本轮两项分析均用 **0.7525**（完整 60 轮）→ 两处并引会自相矛盾（第三/四轮已要求更新 0.7414→0.7525，尚未落实）。同时该记忆的方向回归数字(28.29/28.37°、常数基线 29.83°)与 `review_log` 第六/七轮已核可数字(25.36/27.23°、21.07/26.06°)不一致 → 疑有**未经审查的重跑**（`logs/direction_regress_20261014_{212233,212458}.log`），**越出本轮范围，但须补审**。
  4. `results/` 仍存 `*_SMOKE1ep.csv`（旧冒烟）；`scene_ci` bootstrap 固定 seed=0（可复现，但未做 seed 敏感性）。

- 对五问裁定：
  - **Q1（"未做显著性检验"是否改）**：**是，必须改**（该句已成事实错误）。但**只改事实、不改结论强度**——替换为"已做场景级配对检验（独立单元=21 场景）：LoRA−U-Net ΔIoU +0.050，95%CI[+0.026,+0.076] 不跨 0；**但为单 seed、CI 未含 run-to-run 变异**"。**不同意**删该句而无替代。
  - **Q2（是否改"不足以断言显著优于"）**：**不同意**改为"显著优于"。理由见中风险 1（效应量口径依赖 0.005~0.050 + CI 未含 seed 变异 + 事后加检验）。可用**事实陈述**（"在场景级配对检验下差值 95%CI 不跨 0"）；是否启用"显著优于"这一强词需**人类放行**。主智能体自拟稿"在场景级配对检验下显著优于"**属新的过度声明**，不予核准（其"不能外推为换 seed 亦然"的补充**方向正确但不充分**，因问题不止 seed，还有口径）。
  - **Q3（方法可靠性）**：(a) 配对 t 用**正态近似**略反保守，**应报 t 分布(df=20)**；已复算 verdict 不变（LoRA−U-Net p_t=0.0009、U-Net−SegNet p_t=0.639）。(b) n=21 的 bootstrap 在 α=0.05 可用，与 t-CI 结论一致，属**可接受但样本小**。(c) **场景级聚合确引入"等权小场景"偏置、改变估计量**（见中风险 2）。
  - **Q4（遗漏陷阱）**：(i) **pooled 口径下 LoRA−U-Net IoU 仅 +0.005**——主智能体只写"IoU 对口径敏感"，**漏了这条最要命的后果**。(ii) **效应由少数场景驱动**：LoRA 对 U-Net **17 胜 4 负/21**，最大贡献来自 U-Net 崩掉的 tyrol-e17(−0.203)、tyrol-e33(−0.143)、austin2(−0.131)；中位差 +0.0448≈均值 +0.0501、leave-one-scene-out t∈[3.58,4.52] 恒显著 → **稳健但非普遍**，应报胜率/中位/LOO。(iii) U-Net−SegNet 场景级**中位差为负**(−0.0044)、胜 9 负 12 → 与"不显著"一致（子图级因均值驱动才"看似显著"，此点主智能体判断**正确**）。(iv) 方向相反变化(LoRA↑/U-Net↓)**可用重加权解释**，不证明子图比较"系统出错"。
  - **Q5（可否写入结题报告）**：**BER 敏感性可入附录**（标明"不改主口径、仍为 macro；pooled 为口径敏感性；反解校验为自洽性检查"）。**场景级 CI 可入附录，但仅作稳健性/敏感性**，**禁止**将其 +0.050 提升进主表/摘要/PPT 作"显著优于"依据，**禁止**替换 macro 点估计。

- 必须修正（若采纳上述两项写入报告前）：
  1. **删除/改写任何"显著优于"**：主表与摘要维持"**数值上高于**"+限定（单 seed、21 独立场景）；如引用场景级检验，须**并列**三口径 ΔIoU(macro +0.035 / scene +0.050 / pooled +0.005)并注明"效应量随口径显著变化、pooled 口径下近乎无差"。
  2. **补声明**：场景级 CI"条件于单 seed、未含 run-to-run 变异"；配对检验"未做多重比较校正"。
  3. **BER 敏感性附录**标注"反解=自洽性检查（非独立真值验证）""不改主口径"。
  4. **记忆/文档数字一致**：更新 `experiment-results.md` U-Net 7414→7525；澄清/补审方向回归新数字(28.29/28.37° 等)的来源。
- 建议修正：
  1. 场景级分析**增报**：胜率(17/21、20/21、9/21)、中位差、leave-one-scene-out、及"pooled/macro/scene 三口径 ΔIoU 对照表"。
  2. 至少补 **3 seed**（可缩 epoch 或复用）以给 run-to-run 方差的直接估计；否则报告显式标注该缺口。
  3. 配对 p 改用 **t 分布(df=20)** 或统一以 CI 为准（脚本已声明以 CI 为准，建议日志同步）。
  4. 清理半截日志与旧 `*_SMOKE1ep.csv`。
- 是否允许进入下一阶段：**是（附条件）** —— 两项分析可作**附录/敏感性**继续；但**对外口径改词须先落实"必须修正 1–2"并经人类对"是否解禁'显著'"放行**。
- 可汇报口径（本轮核可，须逐字）：
  - BER："macro(逐图平均，主口径) 与 pooled(全数据集汇总，文献口径) 两种 BER 差值 LoRA +0.0013 / U-Net −0.0016 / SegNet −0.0014 等，**均在 ±0.004 内** → **以 BER 为核心的结论不因口径改变**；IoU 则随口径在 −0.012~+0.019 摆动，**跨口径 IoU 不可比**。"
  - 场景级："以**原始大图为独立单元(21 场景)** 配对检验：LoRA−U-Net ΔIoU **+0.050**(95%CI [+0.026,+0.076])、LoRA−SegNet +0.054([+0.031,+0.078]) 不跨 0；U-Net−SegNet +0.003([−0.010,+0.017]) 不显著。**（单 seed；CI 未含 run-to-run 变异；场景级均值等权各场景，与 macro/pooled 为不同估计量。）**"
  - 三行主表：维持"**数值上高于** CNN 基线；单 seed、测试集 21 独立场景"。
- 禁止用语（本轮重申 + 新增）："显著优于 / 超越全监督 / SOTA / 提升 / 改进"（**沿用**）；**新增**：把场景级 +0.050 作"显著优于"的唯一依据、把场景级均值用于替换主表点估计、以 pooled 口径数字与 macro 混列而不标注。
- 本次审查范围：见首行。
- 本次审查 token 估算：约 55k

### 2026-10-14 审查类型：深度审查（第十一轮：方向编码线性探针 direction_probe_linear —— 质疑项目最上游假设"翻转自监督学到方向"，supervisor 子代理执行）

- 范围：`direction_probe_linear.py`；`results/direction_probe_linear.csv`；`logs/direction_probe_20261014_222735.log`；`train_direction_net.py`（翻转任务定义 + 增强）；`checkpoints/direction_resnet18_flip_basis.pth`；`logs/direction_pretrain_20261014_195557.log`；对照 `results/direction_regress_ablation.csv`；对"翻转预训练对方向无贡献"这一结论可否对外汇报的裁定。
- 结论：**不通过**（探针脚本本身可跑、证据已落盘、无泄漏；但由此得出的**结论"翻转预训练对方向无贡献 / 100% 翻转准确率来自方向无关捷径"超出证据、且触碰第七轮禁令"方向先验无效"，不得对外汇报**。补做"空间特征 + 非线性 + 多划分"探针后可重审。）

证据核验（逐文件，未采信转述）：
1. `results/direction_probe_linear.csv` 与 `logs/direction_probe_20261014_222735.log` 数值**逐位一致**（flip 11.93/26.94/32.75；imagenet 24.96/32.09/29.35；阴性对照 47.74；常数基线 28.21 / 随机 45）。**证据存在、非虚报、可复现**。
2. `checkpoints/direction_resnet18_flip_basis.pth` 存在（44,790,219 B，2026-10-14 19:56），与 `direction_pretrain_...195557.log:39` 末行保存路径一致；骨干加载自检 `120 键 / missing=0 / unexpected=0` 与 ResNet-18 去 fc 后键数吻合 → **骨干确为翻转预训练权重，非错载/非随机初始化**。
3. **无泄漏**：probe 用 `split_by_base(seed=0→1)` 复现 basis 划分；其 test=92 张 / 21 基名，与 `direction_pretrain_...195557.log:6`"21 个测试场景不参与训练"**同一划分** → 翻转骨干未见过 probe 的测试图。λ 仅在 val 上选 → **未用测试集做模型选择**。此三项合规。

**高风险（结论层面）**：

H1. **核心结论"翻转预训练对方向无贡献"不成立（过度声明）+ 触碰禁令**。
   - 探针实际只检验了"**GAP(512d) 特征经线性回归能否读出方向**"，负结果不能推出"骨干未编码方向"。至少三处替代解释未被排除：
     (i) **GAP 破坏空间信息**：阴影方向本质是**空间/朝向属性**（阴影往哪个方向延伸）；全局平均池化抹平空间布局，**无论骨干是否编码方向，GAP 特征都可能读不出**——探针设计本身可能无效（最要害）。
     (ii) **线性不可读 ≠ 无信息**：θ 在特征里可能以非线性/周期形式存在。
     (iii) **输入-标签空间错配**：标签由**整图 mask** 的 `gt_direction` 算出，特征取自 `Resize(256)+CenterCrop(224)`；两者空间范围不同，被裁掉的阴影可能正是定义方向的成分。
   - 故日志 L26 的自动判读"[X] 骨干几乎不含方向信息"是**脚本越权**，不是可下的结论。
   - "100% 翻转准确率来自与方向无关的捷径"**无直接证据**：翻转判别可由"上下亮度不对称"等解出属**推测**，本探针**不能**证实。

H2. **A vs B 比较不可靠（val/test 结论反转）**。
   - 独立核对日志：**验证**集 A(flip, λ*=1.0)=26.94° **优于** B(imagenet, λ*=1000)=32.09°，且**优于常数基线**(32.41°)；**测试**集 A=32.75° **差于** B=29.35°、且**差于常数基线**(28.21°)。**同一对比在 val 与 test 上符号相反** → 3.4° 之差是**划分噪声**，不是特征差异。
   - 且 A、B 的 λ 分别为 1.0 与 1000.0（差 3 个数量级）→ **不同有效模型，比较不受控**。B 选到最强正则本身≈"该特征无信息"，但这与 A 的对比因 λ 不同**不可直接并列**。
   - => "A 比 B 差""A 连常数基线都不如"**均不可作结论**。

H3. **绝对数值不可引用（同意主智能体）**，且**相对比较同样不可引用**（见 H2）。故"有意义的是相对比较 A vs B"这一立场**不成立**。

**中风险**：
M1. **单划分 + 小样本**：train/val/test = 262/58/92，特征 512 维；A 训练 11.93° / 测试 32.75° 严重过拟合；λ 由 58 样本的 val 选 → 选择噪声大。任何"显著性"缺 CI / 多 seed。
M2. **阴性对照协议与主协议不一致**：阴性对照固定 λ=1.0（未走 val 选 λ 流程），与 A/B 协议有出入；虽结论（47.74°>45°）稳健，严格性可统一。
M3. **常数基线 val/test 落差**（32.41 vs 28.21）本身提示标签分布不稳，进一步说明 58/92 样本不足以支撑精细比较。

**低风险**：
L1. 探针 eval 变换（Resize256+CenterCrop224）与骨干**训练**变换（`train_direction_net.py:82-94`：RandomResizedCrop 0.4–1.0 + ColorJitter）不同；训练期随机裁剪可能裁掉定义方向的阴影 → 翻转任务与方向标签的相关性在训练期即被削弱（是对论文"方向先验"论证的一个隐患，但**不属本轮探针能证实**）。
L2. 日志 L20-28 把"探针负结果"直接写成"骨干几乎不含方向信息"，建议改为描述性陈述，避免复用时被当结论。

对六问裁定：
- **Q1（A vs B 能否支撑"无贡献"）**：(a)(b)(c) **三项都成立**。(a) 线性 + GAP 不足；(b) A/B 过拟合不同且 λ 不同，比较不公平（且 val/test 反转）；(c) 应做**非线性探针**——但**仅换 MLP 不够**，须同时换**非 GAP 的空间特征**（feature map 展平 / mask 区域池化 / patch 级）。**该结论不成立**。
- **Q2（绝对 & 相对）**：绝对**不能**引用（同意）；**相对 A vs B 也不能**（见 H2）。"相对可信"立场**不成立**。
- **Q3（导师论文表述）**：见下"可汇报口径"。要点：**不否定导师论文**；只陈述我们在**本项目 AISD 实现**上的**诊断性负结果**，并明确列出探针局限。
- **Q4（其它解释 / 是否需重试）**：**是**。GAP、输入-标签错配、224 中心裁剪、RandomResizedCrop/ColorJitter、样本量——均未排除；**须用"空间特征 + 整图或 mask 区域"重试后**才有资格谈"骨干是否编码方向"。
- **Q5（下一步）**：(a) **必做**，但须"空间特征 + 非线性 + 多随机划分/多 seed"三管齐下，仅 MLP-on-GAP 不足以加固；(b) **暂不可**作结题核心（结论强度不够）；(c)(d) 可作备选，优先级低于 (a)。
- **Q6（泄漏/公平/证据）**：**无泄漏、无测试集参与选型、证据已落盘**（见证据核验 1–3）；**不公平点** = A/B 的 λ 不同 + 单划分（H2）；**无"找不到证据=未完成"违反**。

可汇报口径（逐字，须含标签；**须先落实必须修正 1–3**）：
"在本项目 AISD 复现上做了一个**诊断性探针**：冻结**翻转判别预训练**骨干（basis 划分 68/16/21，**测试集与骨干训练无同源基名**），取其**全局平均池化特征(512 维)**，用**岭回归线性**预测阴影方向（cos2θ,sin2θ）。测试角度误差 **32.75°**，**未优于**"常数方向"基线（28.21°）；阴性对照（随机目标 47.74°）正确地失败。**该负结果仅说明"在全局池化 + 线性可读的口径下未能读出方向"**，受**线性、全局平均池化、样本量(262/58/92)、单 seed/单划分**四重局限；**不足以判定翻转预训练未编码方向，更不构成对导师论文（祁昆仑等,2025）的否定**。"
禁止用语："翻转预训练对方向无贡献 / 无额外贡献"；"100% 翻转准确率来自与方向无关的捷径"；"方向先验无效 / 任务不可学"（**沿用第七轮禁令**）；"导师论文结论不成立 / 论文有误"；引用 A vs B 比较或以 32.75°/29.35° 作任何排名。

必须修正（对外前）：
1. **删除/改写**"翻转预训练对方向无贡献""A 比 B 还差""100% 翻转准确率来自捷径"等定性；只保留"GAP + 线性探针未读出方向"的**描述性**陈述。
2. **不得**引用 A vs B 与绝对数值（32.75°/29.35°）作任何结论或排名。
3. 报告中任何引用须带四重局限标签（线性 / 池化 / 样本量 / 单划分）。

建议修正：
1. 补做 **非 GAP 空间特征探针**（feature-map 展平 或 mask 区域池化）+ **非线性(kNN/MLP)** + **≥5 次随机划分 / 多 seed**，报均值±std 与 CI，再下"骨干是否编码方向"的硬结论。
2. 阴性对照与主协议统一 λ 选择流程。
3. 若重做仍为负，**方可**考虑把"自监督方向路线在本数据集上未兑现"作为**诚实的方法学负结果**写入结题（仍用"未兑现 / 未读出"而非"无贡献 / 无效"）。

是否允许进入下一阶段：**否** —— 该结论**不得对外**；先落实必须修正 1–3，并（建议）补做空间 / 非线性 / 多划分探针后重审。

- 本次审查范围：见首行。
- 本次审查 token 估算：约 70k

---

### 2026-10-14 审查类型：方案审查（第十二轮：**乙阶段 —— 对比与统计补齐**，supervisor 子代理执行）

- 范围：拟执行的 `train_maskrcnn_leakfree.py`(F1)、`run_baselines_leakfree.py`(F2)、`ber_pooled_appendix.py`(F3)、`train_lora_leakfree.py` 加 `--seed`(F4) 的**方案**（未写代码）；核对 `train_lora_leakfree.py`、`train_cnn_leakfree.py`、`eval_shadow.py`、`direction_experiment.py:106-116`+`methods.py`、`run_experiment.py`、`ber_sensitivity.py`、`results/*`、`queue.sh`、runbook、`执行顺序与应急预案.md`、申报书。
- 结论：**有条件通过**。划分/测试集隔离/指标算法/对比设计四维度方向正确、无原则性错误；但 **F4 的 provenance 前提被本项目自身文件证伪**，且 `--seed` 改法含**静默改划分**陷阱；F3 前提有误且与既有产物重复。

**高风险**
- **H1（F4）**：方案假定"seed=0 必须逐位复现"。该前提**不成立**——`执行顺序与应急预案.md:185-187` 与 `AGENTS.md` 失效模式 #9 已记录"同 seed from-scratch 不逐位复现"，旁证实测（同脚本同 seed 两次 U-Net 的 Epoch1 loss=0.4322 vs 0.4320）；全仓库无 cudnn 确定性设置。**且**：`train_lora_leakfree.py:96-97` 的**划分 seed 硬编码 0/1**，与 L73-75 的全局种子是两回事；若误改 L96-97，则新 seed 会在**不同的 92 张测试集**上评估 → 三 seed 不可比、结果全废。
- **H2（口径）**：申报书硬判据 **BER<5% 当前未达成**，且 F1~F7 均为"补对比/补统计"、**不改变主方法**，故不会使其成立。无泄漏 **6.44%**、官方含泄漏 **5.01%**，两口径均 >5%。**不得**把本阶段表述为"使验收判据达标"。

**中风险**
- **M1（F3）**：`results/*_leakfree_test_per_image.csv` **无 TP/FP/FN/TN 列**（表头仅 name+6 指标）；且 pooled 附录**已由 `ber_sensitivity.py` 完成**（`results/ber_sensitivity.csv` 已含六方法 pooled IoU/BER）。F3 须**取消或改述**为"显式落盘逐图计数（不动 `compute_metrics` 契约）"。
- **M2（F1）**：Mask R-CNN 在 262 张/8GB/多尺度图上易**欠拟合**；须**落盘验证曲线与收敛判据**、报选中 epoch，否则"优于"是对未收敛基线取优（申报书问题 3.2 同款陷阱）。
- **M3（F1 公平性）**：主方法提示框来自 `mask_enhanced`（Otsu+CLAHE 粗提取，**非 GT**）→ 主方法是"经典粗提取提议 + 预训练 SAM + LoRA"，**全自动无 GT 泄漏**；F1 文档须点明，否则答辩会被质疑"用了 GT"。

**低风险**
- L1：请求包引用的连通域统计（mean 23.9/median 22/min 6/max 110）**只对前 120 张成立**；全量 Train412 实测 **mean 22.0 / median 20 / min 2 / max 167**（主智能体已复核：两者均真，差在取样范围）。**F1 实现前须用代码重算并落盘**。结论（≈22 实例/图，Mask R-CNN 可行）不受影响。
- L2：逐图计数落盘的文件名/列名须沿用 `{method}_{split}_test.csv` 约定。
- L3：`--resume` 续跑不恢复 RNG，续跑后的 seed 运行与不中断运行不等价，报告须带此限定。

**必须修正（动工前）**
1. **F4 不改原脚本**：复制为 `train_lora_leakfree_seed.py`，**原文件一字不动**；副本 vs 原件 diff 存 `backups/patches/`。
2. **F4 的 `--seed` 只允许控制 L73-75 三个全局种子；L96-97 划分 seed 恒为 0/1**，并加断言（test 文件集合 == 主结果逐图 CSV 的 name 集合，不等即抛异常）。**不可妥协项**。
3. **F4 删除"seed=0 逐位一致"判据**：seed=0 **直接沿用现有产物**（`checkpoints/sam_vit_b_shadow_v4_lora_leakfree.pth` / `results/lora_leakfree_test.csv`），**不重跑**；仅新 seed 用副本跑，必须传 **`--epochs 40`**。
4. **H2 对外表述改写**：如实写"BER 6.44%（无泄漏）未达 5% 目标"，交人类决定是否接受。
5. **M1**：F3 取消或明确改述。
6. **M2**：F1 落盘验证曲线与收敛判据。

**建议修正**
1. 顺序（≈13h 预算）：**P0=F1** → **P1=F2** → **P2=F4**（预算紧则**只补 seed 1**，5.7h）→ **P3=F3**（近零成本/或取消）。**F5、F6 本轮缓**。
2. **F6（SARU-Net）**：仓库确无网络定义代码 → "实跑权重"不可行；走"**如实写明未复现**"或"引原文值并标注跨口径不可横比"，**绝不为基线卡主线**。
3. **F5（IoU+Boundary）**：既不为 BER<5% 也不为 IoU>MaskR-CNN 服务；若做须**单变量**、同划分同 40 轮，并做好"中性或负增益"准备。建议**缓**并说明为何未做。
4. **F2** 须用与主方法**逐字相同**的 `split_by_base(…,0.2,seed=0)`→`(…,0.2,seed=1)`；AMG 超参沿用 `run_experiment.load_amg` 以保持可比。
5. **F1** 细节：软掩膜并集阈值 `>0.5` 后**最近邻** resize 回 GT 尺寸；tau 与选模型**只用 val(58)**；报告 tau 与选中 epoch。
6. 新脚本纳入 `preflight_check.py` 的基底一致性检查。

**是否允许进入下一阶段**：**是（受门控）**——F4 必须先完成必须修正 1-3、F3 必须先澄清，未落实前不得开跑 F4/F3；**F1、F2 可即刻开始**（F1 须先落实 M2 收敛判据）。

- 本次审查范围：仅方案层（未写代码、未跑训练）。
- 本次审查 token 估算：约 40k。

---

### 2026-10-14 审查类型：**结果审查**（第十三轮：F2 阈值法基线在无泄漏划分上的评估，supervisor 子代理执行）

- 范围：`run_baselines_leakfree.py`；`results/{otsu,c3i,enhanced}_leakfree_test.csv` + `_per_image.csv`；`logs/baselines_leakfree_20261014_232556.log`；对照 `results/lora_leakfree_test.csv`、`methods.py`、`README.md`、`结题报告.md`、`软著_软件说明书_初稿.md`。
- 结论：**有条件通过**。数字逐位可核、无泄漏、口径正确、未调参；但 **c3i 行的归因错误**（高风险），另有两处定位/口径问题。

**核验结论（supervisor 独立复算）**
- 逐位复核：per-image 与 CSV **maxdiff = 0.00e+00**；macro 均值逐位吻合（IoU 0.4051/0.1765/0.4837，BER 0.1969/0.5712/0.1537，F1 0.5617/0.2337/0.6371，OA 0.6981/0.4812/0.7980）。
- 无泄漏复核：独立复现 68/16/21 = 262/58/92；train-val / train-test / val-test 基名交集**均为空**；测试集 name 集合与主结果一致（92）；三方法无参数、不读训练/验证集 → **无泄漏、未调参**。
- 声明核验：主智能体 claim 1-3 属实；claim 4（新数字未入任何报告/PPT/记忆）**属实**（全仓库与 memory 目录均无这些数字）。

**高风险**
- **H1（c3i 归因错误）**：`c3i` 的 IoU=0.1765 **不是 C3I 方法的性能**，而是 `methods.py:41-42` 自研启发式（`if mask>40% 则反转`）的产物。supervisor 实测同 92 张：
  原样 **0.1765/0.5712** ｜ **去掉反转 0.4564/0.1650（优于 otsu 0.4051）** ｜ 强制反转 0.0068。
  该分支触发 **61/92 张**，去掉后 **61/61 全部变好、0 张变差** → 在当前测试集上 **100% 有害**。
  主智能体"原样复用 methods.py、未调参"**字面属实**，但把 0.1765 当"C3I 基线"写入报告 = **用错误归因贬低对照方法**，并会误导出"我们远超传统法"的结论。

**中风险**
- **M1（enhanced 定位）**：`mask_enhanced`（`methods.py:8`）被标为"增强的物理粗提取"，且是主方法内部模块（`mask_ours = _direction_filter(mask_enhanced)`；主方法提示框亦来自它）。与 otsu/c3i 并列成"经典阈值法基线"会掩盖其"本项目自研粗提取"性质。项目既有表格已分列（`README.md:72/74`），应沿用。
- **M2（口径混排）**：现有报告基线数字仍是**含泄漏官方口径**（`结题报告.md:148-153`、`README.md:71-75`：Otsu 0.424 / C3I 0.162 / 增强 0.512 / SAM-AMG 0.302；`软著_软件说明书_初稿.md:142`）。新增无泄漏数字后若不同步改口径，会出现"主表无泄漏 vs 附表含泄漏"混排。
- **M3（缺项）**：申报书点名的"原始 SAM Zero-shot"仍缺——`results/sam_amg_leakfree_test.csv` **不存在**（GPU 被 F1 占用）。

**低风险**
- L1：启动断言是**集合相等**且以主结果 CSV 为基准；若全局 `split_by_base` 被改，两者同变、断言仍通过（对该风险不设防）。当前划分代码冻结，风险低。
- L2：断言包在 `if main_csv.exists()` 内，基准缺失时静默跳过（本次存在，未触发）。

**必须修正（写入报告/PPT 前）**
1. **c3i 行**：二选一——(a) 以"去掉 >40% 反转分支"的口径重跑并作为 C3I 基线；或 (b) 同时保留两个数并明确标注"C3I 低分由本项目 `methods.py` 的 0.4 反转启发式造成，非方法固有性能"。**不得只报 0.1765 而不加说明**。
2. **enhanced 行**：与 otsu/c3i **分列**，标签写"本项目增强粗提取（Otsu+CLAHE+形态学，方法内部粗提取模块）"，不计入"经典阈值法外部基线"。
3. **口径统一**：入主表时同步把 `结题报告.md` / `README.md` / 软著文档里的旧基线标注为"旧含泄漏口径"或一并更新，避免混排。
4. 申报书"零样本 SAM"项**如实标注"未做（SAM-AMG 待 GPU）"**。

**建议修正**：追查 `methods.py:41-42` 反转分支的来源（疑似移植自别的实现、极性约定不同）并在注释写清；建议 C3I 基线默认不带走反转分支；把本条结论追加 `review_log.md`；复跑时保留 otsu/enhanced 现产物不变，仅处理 c3i。

**是否允许进入下一阶段**：**是**
- otsu（0.4051/0.1969）与 enhanced（0.4837/0.1537）**可作为基线**写入报告与 PPT（enhanced 须按必须修正 2 分列标签）。
- c3i（0.1765/0.5712）**当前不得原样写入**，须按必须修正 1 处理。

- 本次审查范围：见上（未运行核验 F1 Mask R-CNN 训练本身；`sam_amg` 未跑）。
- 本次审查 token 估算：约 45k。

---

### 2026-10-15 审查类型：**结果审查**（第十四轮：F1 Mask R-CNN 基线，supervisor 子代理执行）

- 范围：`train_maskrcnn_leakfree.py`、`queue_maskrcnn.sh`、`results/maskrcnn_leakfree_{test,test_per_image,val_curve}.csv`、`logs/maskrcnn_leakfree_20261014_232208.log`、`logs/maskrcnn_queue.log`；对照 `results/lora_leakfree_test{,_per_image}.csv`、`evaluate_lora.py`、`direction_experiment.py`、`methods.py` + `backups/patches/`、`run_baselines_leakfree.py`。
- 结论：**有条件通过**。三个数字逐位可核、无泄漏、口径与主表一致；`methods.py` 改动 provenance 合规。但**对比公平性存在方向性偏置（利于我方）且未量化**，阈值选择稳健性偏弱。

**认证数字**：Mask R-CNN(leakfree) **IoU 0.733029 / BER 0.096548**（n=92，tau=0.05，macro）；
对照主方法 LoRA-SAM(leakfree) IoU 0.787821 / BER 0.064359。
（supervisor 独立复算 92 行逐图 CSV -> 均值与汇总 CSV 逐位吻合；name 集合与主结果完全相同。）

**高风险**：无（无泄漏、无测试集参与模型选择、无指标计算错误）。

**中风险**
- **M1（对比公平性，最高优先）**：脚本设 `min_size=512, max_size=800`，**非 torchvision 默认（800/1333）**；且输入已预缩放到 512×512。而主方法 LoRA-SAM 走 SAM 默认 `ResizeLongestSide(1024)`（`evaluate_lora.py:73`、`direction_experiment.py:112`）。即**基线在低于其自身默认、且低于主方法的分辨率下运行**，对小/细阴影不利 → **利我方**的系统性偏置，方向确定但**幅度未量化**。若不披露，该风险升级为"高"（误导性对比）。
- **M2（实例协议不对等）**：实例由 GT 连通域（area≥200）定义，主方法以 `mask_enhanced` 粗提取框为提示，两者输入表征不同，对 Mask R-CNN 上限的影响未量化。
- **M3（阈值稳健性）**：选中 tau=**0.05 落在网格下界**，且与 tau=0.1（val IoU 0.7418）几乎并列（0.7433）→ 阈值选择不稳，结论须带此限定。

**低风险**
- L1：`logs/maskrcnn_leakfree_20261014_232046.log`（**冒烟**）出现 `tau=0.1` 重复、缺 `0.05` —— 系脚本在两次冒烟之间修 `{:.1f}` 显示 bug 的**中间版本**所致。**生产运行（232208）输出干净**（已核对：`tau=0.05`/`tau=0.1` 显示正确），认证数字不受影响。
- L2：请求包"8 张图逐像素差=0"无独立日志佐证，但可由 diff 直接证明，非阻塞。
- L3：`methods.py` 改动**仅在工作区未提交**（`git status` 显示 ` M methods.py`），建议提交固化。

**必须修正（写入报告/PPT 前）**
1. **披露 M1**：基线行脚注须写明"Mask R-CNN 以 `min_size=512/max_size=800` 运行（非默认 800/1333，受 8GB 显存限制），输入 512×512；其真实上限可能被此设置压低，故本对比对我方偏有利"。
2. **披露 M2、M3**。
3. **对比措辞**：只可用"**数值上高于**"，且必须附 AGENTS.md 纪律 #2 的限定：**单 seed、未做显著性检验、测试集 21 个独立场景、差值 0.055 < 逐图 std（0.064~0.069）**。禁用"超越/优于/SOTA/显著"。
4. **F2/c3i 两行同列**（`c3i` 0.4564 主行 / `c3i_heuristic` 0.1765 副行），注明低分源于本项目自加启发式。

**建议修正**
1. **若显存允许，补一次 `min_size=800/max_size=1333` 敏感性** —— 这是申报书"IoU 优于 Mask R-CNN"判据**可信度的关键**；若 OOM 不可行，**如实标注为局限**。
2. F1 结论写入 `review_log.md` 与记忆。
3. 提交 `methods.py` 与 F1 新脚本，固化 provenance。

**四问裁定**
1. 可核验/无泄漏/口径：**通过**。
2. 公平性：**不一致、偏向我方**，幅度未量化 → 不构成"我们很强"的证据，必须标注为对基线的潜在不利设置。
3. 收敛判据：**成立**（最佳 val IoU 0.743266 在 epoch 30，末次评估 epoch 60，末段平台 0.7398~0.7433）→ 基线**确非欠训**，"LoRA 数值上高于 Mask R-CNN"（弱表述）成立，**不得**升级为"超越"。
4. F2 provenance：**合规**（改动仅为新增 `invert_if_mostly_white=True`，默认保持既有行为；patch 快照在 `backups/patches/`）。

**可否写入结题报告/PPT**：**可以，但必须带必须修正 1-4 的披露与限定；未带限定前不得写入。**

**是否允许进入下一阶段**：**是**（F1 结果可定稿入档；进入下一阶段前先把"必须修正"落到报告文本）。

- 本次审查范围：见上（未重跑训练与推理）。
- 本次审查 token 估算：约 87k。
