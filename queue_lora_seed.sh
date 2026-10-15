#!/bin/bash
# LoRA 多 seed 训练队列（可崩溃自愈）—— 跑 train_lora_leakfree_seed.py。
#
# 背景：本机 Modern Standby(S0) 会不定期杀长训练（AGENTS.md 失效模式 #12）。
#       单个 seed 约 5.7h（实测 ~510s/epoch × 40 轮）。
#
# 完成判据（两者都要满足，防"训练完但评估被杀"——AGENTS.md 失效模式 #8）：
#   (a) results/lora_leakfree_s{SEED}_test.csv 存在
#   (b) 续训点里的 epoch >= 目标轮数
#
# 用法：nohup bash queue_lora_seed.sh 1 40 > /dev/null 2>&1 &
cd "D:/RemoteSensing-Shadow-SAM-main/RemoteSensing-Shadow-SAM-main" || exit 1

SEED=${1:-1}
EPOCHS=${2:-40}
LOG=logs/lora_seed${SEED}_queue.log
PY=python
MAX_FAIL=5          # 连续无进展次数上限，超出即放弃（防死循环烧时间）
RCKPT="checkpoints/sam_vit_b_shadow_v4_lora_leakfree_s${SEED}_resume.pth"
RCSV="results/lora_leakfree_s${SEED}_test.csv"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

# 读续训点里已完成的 epoch（同时把是否可续训也打出来）
resume_info () {
  "$PY" - <<PYEOF 2>/dev/null
import sys
sys.path.insert(0, ".")
import env_setup
import torch
p = r"$RCKPT"
try:
    st = torch.load(p, map_location="cpu", weights_only=False)
    print(int(st.get("epoch", 0)))
except Exception:
    print(0)
PYEOF
}

log ""
log "########## LoRA seed=${SEED} 队列启动（目标 ${EPOCHS} 轮）##########"

fail=0
while true; do
  ep=$(resume_info | tr -d '[:space:]')
  [ -z "$ep" ] && ep=0
  have_csv=0; [ -f "$RCSV" ] && have_csv=1

  if [ "$have_csv" -eq 1 ] && [ "$ep" -ge "$EPOCHS" ]; then
    log "===== 完成：seed=${SEED}（续训点 epoch=$ep，结果 CSV 已生成）====="
    break
  fi

  if [ -f "$RCKPT" ]; then
    log ">> 续训（已完成 epoch=${ep}，目标 ${EPOCHS}）"
    "$PY" train_lora_leakfree_seed.py --seed "$SEED" --epochs "$EPOCHS" --resume "$RCKPT" >> "$LOG" 2>&1
  else
    log ">> 从头训练（无续训点，目标 ${EPOCHS}）"
    "$PY" train_lora_leakfree_seed.py --seed "$SEED" --epochs "$EPOCHS" >> "$LOG" 2>&1
  fi

  ep2=$(resume_info | tr -d '[:space:]')
  [ -z "$ep2" ] && ep2=0
  have_csv2=0; [ -f "$RCSV" ] && have_csv2=1

  if [ "$ep2" -le "$ep" ] && [ "$have_csv2" -eq 0 ]; then
    fail=$((fail + 1))
    log "!! 本次无进展（epoch ${ep} -> ${ep2}），连续第 ${fail}/${MAX_FAIL} 次"
    if [ "$fail" -ge "$MAX_FAIL" ]; then
      log "!!!!! 连续 ${MAX_FAIL} 次无进展，放弃（脚本可能有问题，请人工介入）"
      break
    fi
  else
    fail=0
  fi
  sleep 45
done

log "########## LoRA seed=${SEED} 队列结束 ##########"
