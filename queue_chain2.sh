#!/bin/bash
# 串联队列(3)：等 queue_chain.sh 跑完（seed-2 结束）-> 自动接「压 BER」实验。
#
# 设计纪律（沿用 queue_chain.sh 的教训）：
#   **不做任何"进程数"轮询**（防睡眠守护会污染计数）——只等**日志里的结束标记**。
#
# 用法：nohup bash queue_chain2.sh > /dev/null 2>&1 &
cd "D:/RemoteSensing-Shadow-SAM-main/RemoteSensing-Shadow-SAM-main" || exit 1

LOG=logs/chain2_queue.log
PY=python
MARKER="线性串行队列 v2 结束"
MAX_WAIT_MIN=720          # 最多等 12 小时

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

log ""
log "########## 串联队列(3) 启动：等 chain v2 结束 -> 压 BER 实验 ##########"

waited=0
while [ "$waited" -lt "$MAX_WAIT_MIN" ]; do
  if grep -q "$MARKER" logs/chain_queue.log 2>/dev/null; then
    log "检测到 chain v2 结束标记（等待 ${waited} 分钟）"
    break
  fi
  if [ $((waited % 30)) -eq 0 ]; then
    log "等待 chain v2 中 ...（${waited} 分钟）"
  fi
  sleep 60
  waited=$((waited + 1))
done

if [ "$waited" -ge "$MAX_WAIT_MIN" ]; then
  log "!!!!! 等待超时（${MAX_WAIT_MIN} 分钟），放弃"
  log "########## 串联队列(3) 结束 ##########"
  exit 0
fi

# 给 GPU 一点释放时间（前一进程退出后显存回收）
sleep 60

log "===== 开始压 BER 实验（ber_reduce_eval.py）====="
"$PY" ber_reduce_eval.py >> "$LOG" 2>&1
RC=$?
log "===== 压 BER 实验结束（退出码 $RC）====="

log "########## 串联队列(3) 结束 ##########"
