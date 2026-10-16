#!/bin/bash
# 方向 Adapter 的**随机方向对照**（补第三十二轮审查指出的缺失对照）
#
# 目的：判断方向 Adapter 带来的 ΔBER=-0.0038 是否来自**方向信息内容**。
#
# 两个对照回答不同问题：
#  (a) 评估期随机：在**已训好的 regress adapter** 上，同一权重、只把方向输入换成随机。
#      [!] 这是**分布外(OOD)辅助探针**：adapter 训练时见的是 regress 方向，喂随机属分布偏移。
#          仅在"复制出效应"时可比读；若结果不同，无法区分"用了方向内容"与"OOD 改变了输出"。
#  (b) 训练期随机：**另训一个** adapter，训练时即用随机方向（无信息输入）-> 这是**主对照**。
#          若"用无信息方向训出的 adapter"也产生同样的 ΔBER，则该下降与方向信息无关。
#  (c) 评估 (b) 的产物（四态：none/oracle/regress/random）。
#
# ⚠️【PYTHONHASHSEED 必须置 0 —— 本项目曾在此处写错】
#   `directions_for(source="random")` 与训练循环的随机分支都用
#   `np.random.RandomState(hash(Path(f).name) % (2**31))` 生成逐图确定性随机方向。
#   但 Python 对 **str 的 hash 是按进程随机化**的（未设 PYTHONHASHSEED 时
#   sys.flags.hash_randomization=1）：同名文件名在不同进程会得到**不同的 hash**。
#   后果：训练侧与评估侧拿到的随机方向**不一致**，且**每次重跑结果都不同**（不可复现）。
#   故本队列所有 python 调用一律前置 `PYTHONHASHSEED=0`。
#   （监督第三十三轮实测：同名文件名三次运行给出三个互异的 hash。）
#
# 【预注册判据】（跑之前定死，事后不得改）
#   主对照 = (c) 中 "random" 行 vs "none" 行（**同一权重、同图，仅方向开关**）：
#     从 per_image CSV 复算逐图配对 ΔBER 与配对 t；同样对 main_wrand 的 "regress" vs "none" 复算。
#     判定 A（效应与方向信息无关）：|t_random| > 2 且 ΔBER_random 与 ΔBER_regress 同号
#                                    且 |ΔBER_random − ΔBER_regress| <= 0.001
#     判定 B（未能复制）：random 侧效应量显著更小或不显著
#       [!] 判定 B **不得**升级为"方向信息有效"——均匀随机与 regress 方向的分布不同，
#           需另补"打乱 regress 方向"（保留边际分布、只破坏图↔方向配对）才能进一步判。
#
# 【tag 纪律】一律用新 tag（main_wrand / random_wrand），
# **绝不覆盖**第三十二轮已过审的 `diradapter_eval_main.*`。
#
# 用法：nohup bash queue_random_control.sh > /dev/null 2>&1 &
cd "D:/RemoteSensing-Shadow-SAM-main/RemoteSensing-Shadow-SAM-main" || exit 1
LOG=logs/queue_random_control_console.log
export PYTHONHASHSEED=0          # [!] 见上：保证随机方向跨进程一致且可复现
REGRESS_CKPT="checkpoints/sam_vit_b_shadow_v4_dir_adapter_main_leakfree.pth"
RANDOM_CKPT="checkpoints/sam_vit_b_shadow_v4_dir_adapter_random_leakfree.pth"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

ensure_awake () {
  if powershell -NoProfile -Command \
      "(Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe'\").CommandLine" \
      2>/dev/null | grep -qi "keep_awake"; then
    log "keep_awake 已在运行（不重复启动）"
  else
    log "keep_awake 未运行 -> 启动"
    nohup pythonw keep_awake.py > /dev/null 2>&1 &
    sleep 3
  fi
}

log ""
log "########## 随机方向对照队列启动 ##########"
log "PYTHONHASHSEED=$PYTHONHASHSEED （须为 0，否则随机方向不可复现）"
log "(a) 评估 regress-trained adapter + 随机行 -> tag=main_wrand （OOD 辅助探针）"
log "(b) 训练 random-source adapter（--freeze-base）-> tag=random （主对照）"
log "(c) 评估 random-trained adapter -> tag=random_wrand"
ensure_awake

# ---------- (a) 评估期随机 ----------
log "===== (a) 开始 ====="
if [ -f "$REGRESS_CKPT" ]; then
  python evaluate_direction_adapter.py --ckpt "$REGRESS_CKPT" --tag main_wrand --with-random >> "$LOG" 2>&1
  log "===== (a) 结束（退出码 $?）====="
else
  log "!! 跳过 (a)：缺少 $REGRESS_CKPT"
fi

# ---------- (b) 训练期随机 ----------
# 用无信息方向训 adapter；其余（划分/轮数/freeze-base/日程）与 B 步完全一致 -> 单变量
log "===== (b) 开始 ====="
python train_direction_adapter.py --epochs 20 --direction-source random --freeze-base --tag random >> "$LOG" 2>&1
log "===== (b) 结束（退出码 $?）====="

# ---------- (c) 评估 random-trained adapter ----------
log "===== (c) 开始 ====="
if [ -f "$RANDOM_CKPT" ]; then
  python evaluate_direction_adapter.py --ckpt "$RANDOM_CKPT" --tag random_wrand --with-random >> "$LOG" 2>&1
  log "===== (c) 结束（退出码 $?）====="
else
  log "!! 跳过 (c)：缺少 $RANDOM_CKPT（(b) 可能未产出成功权重）"
fi

# ---------- 完成判定（缺一即"未完成"，不得声称完成） ----------
# L1：除两个 eval CSV，还须查 (b) 自身的产物，否则 (b) 在"存权重后、写自身CSV前"被杀会被误判完成。
ok=1
for f in results/diradapter_random_test.csv \
         results/diradapter_eval_main_wrand.csv \
         results/diradapter_eval_random_wrand.csv; do
  if [ -f "$f" ]; then log "存在：$f"; else log "**缺失**：$f"; ok=0; fi
done
if [ "$ok" -eq 1 ]; then
  log "判定：随机对照队列已完成"
else
  log "判定：**未完成** -- 请如实上报，不得声称完成"
fi
log "########## 队列结束 ##########"
