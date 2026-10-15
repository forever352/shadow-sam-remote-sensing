#!/bin/bash
# Mask R-CNN 训练队列（可崩溃自愈）—— 与 queue.sh 同思路，针对单个任务。
#
# 背景：本机启用 Modern Standby(S0)，长训练会不定期被杀（见 AGENTS.md 失效模式 #12）。
#       本脚本用 --resume 循环重试，直到「训练轮数达标」**且**「结果 CSV 已生成」。
#
# [!] 完成判据必须同时包含「结果 CSV 存在」——只查 _last.pth 的 epoch 会漏掉
#     "训练跑满但最终评估被杀" 的情形（AGENTS.md 失效模式 #8，本项目实测踩过）。
#
# 用法：nohup bash queue_maskrcnn.sh 60 > /dev/null 2>&1 &
cd "D:/RemoteSensing-Shadow-SAM-main/RemoteSensing-Shadow-SAM-main" || exit 1
LOG=logs/maskrcnn_queue.log
PY=python
EPOCHS=${1:-60}
MAX_RETRY=60

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

# 读取 _last.pth 里的已完成 epoch（先 import env_setup，否则 torch 因 DLL 问题导入失败）
done_epochs () {
  "$PY" - <<'PYEOF' 2>/dev/null
import sys
sys.path.insert(0, ".")
import env_setup
import torch
try:
    st = torch.load("checkpoints/maskrcnn_shadow_leakfree_last.pth",
                    map_location="cpu", weights_only=False)
    print(int(st["epoch"]))
except Exception:
    print(0)
PYEOF
}

log ""
log "########## Mask R-CNN 队列启动（目标 ${EPOCHS} 轮）##########"

i=0
while [ "$i" -lt "$MAX_RETRY" ]; do
  i=$((i + 1))
  ep=$(done_epochs)
  have_csv=0
  [ -f results/maskrcnn_leakfree_test.csv ] && have_csv=1

  if [ "$ep" -ge "$EPOCHS" ] && [ "$have_csv" -eq 1 ]; then
    log "===== 完成：_last.pth epoch=$ep，结果 CSV 已生成（第 $i 次检查）====="
    break
  fi

  if [ "$ep" -ge "$EPOCHS" ]; then
    log ">> 训练已到 ${EPOCHS} 轮但缺结果 CSV -> 补跑 --evaluate（第 $i 次）"
    "$PY" train_maskrcnn_leakfree.py --evaluate >> "$LOG" 2>&1
  else
    log ">> 开始/续训（当前 epoch=${ep}，目标 ${EPOCHS}，第 $i 次）"
    "$PY" train_maskrcnn_leakfree.py --epochs "$EPOCHS" --resume >> "$LOG" 2>&1
  fi

  sleep 45
done

if [ "$i" -ge "$MAX_RETRY" ]; then
  log "!!!!! 达到最大重试次数 $MAX_RETRY，放弃（请人工介入）"
fi
log "########## Mask R-CNN 队列结束 ##########"
