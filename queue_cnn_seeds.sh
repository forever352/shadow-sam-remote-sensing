#!/bin/bash
# CNN 基线多 seed 队列（U-Net/SegNet × seed 1/2，可崩溃自愈）
#
# 背景：本机 Modern Standby(S0) 会不定期杀长训练；单次 U-Net 约 45-60 min，
#       SegNet 约 75-90 min（60 epochs，RTX 4060 Laptop 8GB）。
#
# 完成判据（两者都要满足）：
#   (a) results/{arch}_leakfree_s{seed}_test.csv 存在
#   (b) 续训点里的 epoch >= 60
#
# 用法：nohup bash queue_cnn_seeds.sh > logs/queue_cnn_seeds_console.log 2>&1 &
cd "D:/RemoteSensing-Shadow-SAM-main/RemoteSensing-Shadow-SAM-main" || exit 1

LOG=logs/cnn_seeds_queue.log
PY=python
EPOCHS=60
MAX_FAIL=5

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

resume_epoch () {
  "$PY" - "$1" <<'PYEOF' 2>/dev/null
import sys
sys.path.insert(0, ".")
import env_setup
import torch
p = sys.argv[1]
try:
    st = torch.load(p, map_location="cpu", weights_only=False)
    print(int(st.get("epoch", 0)))
except Exception:
    print(0)
PYEOF
}

log ""
log "########## CNN 多 seed 队列启动（U-Net/SegNet × seed 1/2，${EPOCHS} 轮）##########"

for arch in unet segnet; do
  for seed in 1 2; do
    RCKPT="checkpoints/${arch}_shadow_leakfree_s${seed}_last.pth"
    RCSV="results/${arch}_leakfree_s${seed}_test.csv"
    log ""
    log "########## ${arch} seed=${seed} 开始 ##########"
    fail=0
    while true; do
      ep=$(resume_epoch "$RCKPT" | tr -d '[:space:]')
      [ -z "$ep" ] && ep=0
      have_csv=0; [ -f "$RCSV" ] && have_csv=1

      if [ "$have_csv" -eq 1 ] && [ "$ep" -ge "$EPOCHS" ]; then
        log "===== 完成：${arch} seed=${seed}（epoch=$ep，CSV 已生成）====="
        break
      fi

      if [ -f "$RCKPT" ]; then
        log ">> 续训 ${arch} seed=${seed}（已完成 epoch=${ep}，目标 ${EPOCHS}）"
        "$PY" train_cnn_leakfree_seed.py --arch "$arch" --seed "$seed" --epochs "$EPOCHS" --resume >> "$LOG" 2>&1
      else
        log ">> 从头训练 ${arch} seed=${seed}（目标 ${EPOCHS}）"
        "$PY" train_cnn_leakfree_seed.py --arch "$arch" --seed "$seed" --epochs "$EPOCHS" >> "$LOG" 2>&1
      fi

      ep2=$(resume_epoch "$RCKPT" | tr -d '[:space:]')
      [ -z "$ep2" ] && ep2=0
      have_csv2=0; [ -f "$RCSV" ] && have_csv2=1

      if [ "$ep2" -le "$ep" ] && [ "$have_csv2" -eq 0 ]; then
        fail=$((fail + 1))
        log "!! ${arch} seed=${seed} 本次无进展（epoch ${ep} -> ${ep2}），连续第 ${fail}/${MAX_FAIL} 次"
        if [ "$fail" -ge "$MAX_FAIL" ]; then
          log "!!!!! 连续 ${MAX_FAIL} 次无进展，跳过 ${arch} seed=${seed}，请人工介入"
          break
        fi
      else
        fail=0
      fi
      sleep 45
    done
  done
done

log ""
log "########## CNN 多 seed 队列结束 ##########"
