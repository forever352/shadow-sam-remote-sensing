#!/bin/bash
# 线性阻塞队列：A 冻结编码器版无泄漏重训 -> B 方向 Adapter 训练 -> C 方向 Adapter 评估
#
# 【为什么线性阻塞】本机曾用"python 进程数 == 0"判断 GPU 空闲，但防睡眠守护 keep_awake.py
#   本身也是 python 进程 -> 计数永不为 0 -> F1 跑完后三个任务全没启动，GPU 空转 11.5 小时。
#   故此处**不做任何跨脚本进程数轮询**，只用 bash 顺序阻塞执行。
#
# 【为什么 A 用 while 重试】本机是 Modern Standby(S0)，长训练会被不定期杀掉。
#   A 支持 --resume，故按"结果 CSV 是否已生成"判断完成，未完成则 60s 后续训。
#   （判据用 CSV 而非 _last.pth 的 epoch：epoch 40 的续训点在训练循环内保存，
#     而最终评估与落盘在循环之后；若恰在两者之间被杀，只看 epoch 会误判"已完成"。）
#
# 【B 无 --resume】只能跑一次；但它每轮都验证并存最佳权重，中断不会全丢。
#   C（评估）是 B 的恢复路径：可在既有 best 权重上复算，不需重训。
#
# 用法：nohup bash queue_frozen_adapter.sh > /dev/null 2>&1 &
cd "D:/RemoteSensing-Shadow-SAM-main/RemoteSensing-Shadow-SAM-main" || exit 1
LOG=logs/queue_frozen_adapter_console.log
PY=python
FROZEN_LAST="checkpoints/sam_vit_b_shadow_frozen_leakfree_last.pth"
FROZEN_CSV="results/frozen_leakfree_test.csv"
FROZEN_EPOCHS=40
ADAPTER_CKPT="checkpoints/sam_vit_b_shadow_v4_dir_adapter_main_leakfree.pth"
ADAPTER_EVAL_CSV="results/diradapter_eval_main.csv"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

# --- 防睡眠守护：幂等检查，未运行才启动 ---
ensure_awake () {
  if powershell -NoProfile -Command \
      "(Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe'\").CommandLine" \
      2>/dev/null | grep -qi "keep_awake"; then
    log "keep_awake 已在运行（不重复启动）"
  else
    log "keep_awake 未运行 -> 启动"
    nohup pythonw keep_awake.py > /dev/null 2>&1 &
    sleep 3
    log "keep_awake 启动完成"
  fi
}

log ""
log "########## 队列启动（线性阻塞，不做进程数轮询）##########"
log "A = 冻结编码器版无泄漏重训（40 轮，可崩溃自愈）"
log "B = 方向 Adapter 训练（20 轮，regress，--freeze-base，无 resume）"
log "C = 方向 Adapter 评估（同一权重上 dir off/oracle/regress 三态；B 的恢复路径）"
ensure_awake

# ---------- A ----------
log "===== A 开始 ====="
while true; do
  if [ -f "$FROZEN_LAST" ]; then
    "$PY" train_frozen_leakfree.py --epochs "$FROZEN_EPOCHS" --resume "$FROZEN_LAST" >> "$LOG" 2>&1
  else
    "$PY" train_frozen_leakfree.py --epochs "$FROZEN_EPOCHS" >> "$LOG" 2>&1
  fi
  if [ -f "$FROZEN_CSV" ]; then
    log "===== A 完成：结果 CSV 已生成 ($FROZEN_CSV) ====="
    break
  fi
  log "!! A 未完成（$FROZEN_CSV 不存在），疑似休眠中断，60s 后续训"
  sleep 60
done

# ---------- B ----------
# 主方法口径必须用 regress（预测方向）；oracle 用 GT 方向，仅作诊断/上界。
# --freeze-base：只训方向 adapter，冻结 LoRA/prompt_encoder/mask_decoder
#   -> 结果才可归因于 adapter（否则相对主结果的差值混入了额外 20 轮联合微调）。
log "===== B 开始 ====="
"$PY" train_direction_adapter.py --epochs 20 --direction-source regress --freeze-base --tag main >> "$LOG" 2>&1
log "===== B 结束（退出码 $?）====="

# ---------- C ----------
# 同一权重下对比"方向关 / oracle / regress"三态，是 adapter 净效果最干净的口径；
# 也是 B 被截断时的恢复路径（不需重训）。
log "===== C 开始（方向 Adapter 评估）====="
if [ -f "$ADAPTER_CKPT" ]; then
  "$PY" evaluate_direction_adapter.py --ckpt "$ADAPTER_CKPT" --tag main >> "$LOG" 2>&1
  log "===== C 结束（退出码 $?）====="
else
  log "!! 跳过 C：找不到权重 $ADAPTER_CKPT（B 可能未产出成功权重）"
fi

if [ -f "$ADAPTER_EVAL_CSV" ]; then
  log "B/C 判定：已完成（$ADAPTER_EVAL_CSV 存在）"
else
  log "B/C 判定：**未完成**（$ADAPTER_EVAL_CSV 不存在）-- 请如实上报，不得声称完成"
fi

log "########## 队列全部结束 ##########"
