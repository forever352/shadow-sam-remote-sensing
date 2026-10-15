# F3（IoU + Boundary 损失）—— **已废弃，2026-10-16 由人类决定不做**

## 为什么在这里

申报书点名了"结合 IoU Loss 和 Boundary Loss"，与主方法实际用的 Focal+Dice 不符
（`申报书问题清单.md` 4.4 技术方案落差）。方案经第十九轮审查"有条件通过"（6 条必改），
但**人类评估后决定不做**：
- 对两条硬指标（BER<5%、IoU>Mask R-CNN）**均无贡献**
- 预期中性或负增益，且是**复合替换**（Dice→IoU + 新增 Boundary），归因不清
- 单 seed + 选模噪声大 → 结果大概率"无法区分"
- **成本最高（~4.8h GPU）、收益最低**

→ 结题时**如实写"未实现"**，不得含糊。

## ⚠️ 本目录文件不可直接运行

- `train_lora_leakfree_ioubound.py` —— **有语法错误**（生成脚本把 `ADD_DOC` 插到了
  docstring **外面**，导致裸文本出现在模块层）。若日后重启 F3，须先修此处。
- `losses_iou_boundary.py` —— 损失实现，本身可用（含 IoU + 有符号距离的 Boundary）
- `make_lora_leakfree_ioubound.py` —— 生成脚本，**含上述插入位置 bug**

## 重启 F3 时须落实的 6 条必改（审查）

1. 更正"只改一行"的表述：损失在 `train_adapter_v4.train_step` 内，副本须**重写 `train_step`**（逐行照抄、只换损失行），**不动 `train_adapter_v4.py`**
2. 距离图在 `train_step` 内对**已翻转+已 resize 的 gt256 现算**（不预计算/不缓存）；**严禁**新增 random 抽取
3. 划分硬断言（92/21 + 逐图 name == 主 CSV），`OUT`/`RESUME_CKPT` 加 `_ioubound` 后缀
4. 开跑前 preflight 退出码 0；**先冻结** diff+sha256
5. 损失在 autocast **之外**（fp32）；空 GT guard
6. 开跑前定死：λ 固定 / **保留 Focal** / 距离图**有符号** / **严格 40 轮**
