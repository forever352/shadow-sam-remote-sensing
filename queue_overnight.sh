#!/bin/bash
# 过夜串联队列：等 Mask R-CNN（F1）队列结束后，自动接上 LoRA seed-1（F4）训练。
#
# 为什么串联：本机 RTX 4060 8GB，**GPU 任务必须串行**（并发必 OOM）。
# F1 占用 GPU 时不能起 F4；但夜里没人守着，故用一个等待循环自动衔接。
#
# 用法：nohup bash queue_overnight.sh > /dev/null 2>&1 &
cd "D:/RemoteSensing-Shadow-SAM-main/RemoteSensing-Shadow-SAM-main" || exit 1

LOG=logs/overnight_queue.log
SEED=${1:-1}
EPOCHS=${2:-40}
MAX_WAIT_MIN=240          # 最多等 F1 4 小时，超时则放弃衔接（防死等）

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

n_python () { tasklist //FI "IMAGENAME eq python.exe" 2>/dev/null | grep -c "python.exe"; }

log ""
log "########## 过夜串联队列启动（等 F1 -> 跑 F4 seed=${SEED}）##########"

waited=0
while [ "$waited" -lt "$MAX_WAIT_MIN" ]; do
  done_marker=0
  grep -q "Mask R-CNN 队列结束" logs/maskrcnn_queue.log 2>/dev/null && done_marker=1
  np=$(n_python | tr -d '[:space:]')
  [ -z "$np" ] && np=99

  if [ "$done_marker" -eq 1 ] && [ "$np" -eq 0 ]; then
    log "F1 队列已结束且无残留 python 进程（等待了 ${waited} 分钟）"
    break
  fi
  if [ $((waited % 10)) -eq 0 ]; then
    log "等待 F1 中 ...（${waited} 分钟；marker=${done_marker}，python 进程数=${np}）"
  fi
  sleep 60
  waited=$((waited + 1))
done

if [ "$waited" -ge "$MAX_WAIT_MIN" ]; then
  log "!!!!! 等待 F1 超过 ${MAX_WAIT_MIN} 分钟，放弃衔接 F4（请人工检查）"
  log "########## 过夜串联队列结束 ##########"
  exit 0
fi

# GPU 已空 —— 启动 F4
log "===== 启动 LoRA seed=${SEED}（${EPOCHS} 轮）====="
bash queue_lora_seed.sh "$SEED" "$EPOCHS"
log "===== LoRA seed=${SEED} 返回 ====="

log "########## 过夜串联队列结束 ##########"
