#!/bin/bash
# Mask R-CNN 多 seed 队列（seed 1/2，可崩溃自愈）
#
# 背景：本机 Modern Standby(S0) 会不定期杀长训练；单次约 60-90 min（60 epochs）。
# 完成判据（两者都要满足）：
#   (a) results/maskrcnn_leakfree_s{seed}_test.csv 存在
#   (b) 续训点里的 epoch >= 60
#
# 用法：nohup bash queue_maskrcnn_seeds.sh > logs/queue_maskrcnn_seeds_console.log 2>&1 &
cd "D:/RemoteSensing-Shadow-SAM-main/RemoteSensing-Shadow-SAM-main" || exit 1

LOG=logs/maskrcnn_seeds_queue.log
PY=python
EPOCHS=60
MIN=800
MAX=1333
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
log "########## Mask R-CNN 多 seed 队列启动（seed 1/2，${EPOCHS} 轮，${MIN}/${MAX} 与 seed-0 res800 一致）##########"

for seed in 1 2; do
  RCKPT="checkpoints/maskrcnn_shadow_leakfree_s${seed}_last.pth"
  RCSV="results/maskrcnn_leakfree_s${seed}_test.csv"
  RCSV_PER="results/maskrcnn_leakfree_s${seed}_test_per_image.csv"
  log ""
  log "########## Mask R-CNN seed=${seed} 开始 ##########"
  log "  protocol: min_size=${MIN} max_size=${MAX}  meta=checkpoints/maskrcnn_shadow_leakfree_s${seed}_meta.json"
  fail=0
  while true; do
    ep=$(resume_epoch "$RCKPT" | tr -d '[:space:]')
    [ -z "$ep" ] && ep=0
    have_csv=0; [ -f "$RCSV" ] && [ -f "$RCSV_PER" ] && have_csv=1

    if [ "$have_csv" -eq 1 ] && [ "$ep" -ge "$EPOCHS" ]; then
      log "===== 完成：Mask R-CNN seed=${seed}（CSV 与逐图 CSV 均已生成，epoch=$ep）====="
      break
    fi

    if [ "$ep" -ge "$EPOCHS" ]; then
      # 训练已完成但最终评估被杀：必须走 --evaluate，不能 --resume（否则脚本直接 return）
      log ">> 训练已完成但 CSV 缺失，调用 --evaluate（seed=${seed}, ${MIN}/${MAX}）"
      "$PY" train_maskrcnn_leakfree_seed.py --seed "$seed" --epochs "$EPOCHS" \
        --evaluate --min-size "$MIN" --max-size "$MAX" >> "$LOG" 2>&1
    elif [ -f "$RCKPT" ]; then
      log ">> 续训 Mask R-CNN seed=${seed}（已完成 epoch=${ep}，目标 ${EPOCHS}，${MIN}/${MAX}）"
      "$PY" train_maskrcnn_leakfree_seed.py --seed "$seed" --epochs "$EPOCHS" \
        --resume --min-size "$MIN" --max-size "$MAX" >> "$LOG" 2>&1
    else
      log ">> 从头训练 Mask R-CNN seed=${seed}（目标 ${EPOCHS}，${MIN}/${MAX}）"
      "$PY" train_maskrcnn_leakfree_seed.py --seed "$seed" --epochs "$EPOCHS" \
        --min-size "$MIN" --max-size "$MAX" >> "$LOG" 2>&1
    fi

    ep2=$(resume_epoch "$RCKPT" | tr -d '[:space:]')
    [ -z "$ep2" ] && ep2=0
    have_csv2=0; [ -f "$RCSV" ] && [ -f "$RCSV_PER" ] && have_csv2=1

    if [ "$ep2" -le "$ep" ] && [ "$have_csv2" -eq 0 ]; then
      fail=$((fail + 1))
      log "!! Mask R-CNN seed=${seed} 本次无进展（epoch ${ep} -> ${ep2}），连续第 ${fail}/${MAX_FAIL} 次"
      if [ "$fail" -ge "$MAX_FAIL" ]; then
        log "!!!!! 连续 ${MAX_FAIL} 次无进展，跳过 seed=${seed}，请人工介入"
        break
      fi
    else
      fail=0
    fi
    sleep 45
  done
done

log ""
log "########## Mask R-CNN 多 seed 队列结束 ##########"
