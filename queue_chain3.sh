#!/bin/bash
# 串联队列(4)：等 chain2（压 BER 实验）结束 -> 冒烟 -> G 阶段跨数据集 -> 任务B 同提示零样本
#
# 设计纪律：**不做任何"进程数"轮询**（防睡眠守护会污染计数）——只等**日志里的结束标记**。
#
# 第十九轮审查要求：
#   - G 阶段先冒烟再全量；冒烟与全量 tag 已分离（eval_cross_dataset.py 内 --limit 自动加 _smoke）
#   - 全量硬断言 n==200，不符拒绝落盘（防半截结果冒充完整）
#   - GPU **串行**；每步前不数进程
#
# 用法：nohup bash queue_chain3.sh > /dev/null 2>&1 &
cd "D:/RemoteSensing-Shadow-SAM-main/RemoteSensing-Shadow-SAM-main" || exit 1

LOG=logs/chain3_queue.log
PY=python
MARKER="串联队列(3) 结束"
MAX_WAIT_MIN=720

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

log ""
log "########## 串联队列(4) 启动：等 chain2 -> G阶段 -> 任务B ##########"

waited=0
while [ "$waited" -lt "$MAX_WAIT_MIN" ]; do
  if grep -q "$MARKER" logs/chain2_queue.log 2>/dev/null; then
    log "检测到 chain2 结束标记（等待 ${waited} 分钟）"
    break
  fi
  if [ $((waited % 30)) -eq 0 ]; then
    log "等待 chain2 中 ...（${waited} 分钟）"
  fi
  sleep 60
  waited=$((waited + 1))
done

if [ "$waited" -ge "$MAX_WAIT_MIN" ]; then
  log "!!!!! 等待超时（${MAX_WAIT_MIN} 分钟），放弃"
  log "########## 串联队列(4) 结束 ##########"
  exit 0
fi

sleep 60   # 让前一进程退出、显存回收

# ── G 阶段：先冒烟（走 _smoke tag，不污染正式产物）──
log "===== [G1] RSISD 跨数据集 冒烟（--limit 10）====="
"$PY" eval_cross_dataset.py --limit 10 >> "$LOG" 2>&1
log "===== [G1] 冒烟结束（退出码 $?）====="

# ── G 阶段：全量 test200 ──
log "===== [G2] RSISD 跨数据集 全量 test200 ====="
"$PY" eval_cross_dataset.py >> "$LOG" 2>&1
log "===== [G2] 全量结束（退出码 $?）====="

# ── 任务 B：同提示零样本对照 ──
log "===== [B] 同提示零样本对照（原生 SAM + 本项目框提示）====="
"$PY" sam_box_zeroshot_eval.py >> "$LOG" 2>&1
log "===== [B] 结束（退出码 $?）====="

log "########## 串联队列(4) 结束 ##########"
