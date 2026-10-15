#!/bin/bash
# 过夜串联队列（第二段）：等 seed-1 结束后 -> SAM-AMG 基线 -> 健全性闸门 -> seed-2
#
# 为什么单独一段：第一段队列 queue_overnight.sh 已经在跑「等 F1 -> seed-1」。
# 本段接手它之后的活儿，**不打断、不 kill 任何已有进程**。
#
# GPU 串行：RTX 4060 8GB，任一时刻只允许一个 GPU 任务（并发必 OOM）。
#
# 用法：nohup bash queue_overnight2.sh > /dev/null 2>&1 &
cd "D:/RemoteSensing-Shadow-SAM-main/RemoteSensing-Shadow-SAM-main" || exit 1

LOG=logs/overnight2_queue.log
PY=python
S1_CSV="results/lora_leakfree_s1_test.csv"
MAX_WAIT_MIN=600          # 最多等 seed-1 十小时

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
n_python () { tasklist //FI "IMAGENAME eq python.exe" 2>/dev/null | grep -c "python.exe"; }

log ""
log "########## 过夜串联队列(2) 启动：等 seed-1 -> SAM-AMG -> 闸门 -> seed-2 ##########"

# ── 等 seed-1 完成 ──
waited=0
while [ "$waited" -lt "$MAX_WAIT_MIN" ]; do
  have=0; [ -f "$S1_CSV" ] && have=1
  np=$(n_python | tr -d '[:space:]'); [ -z "$np" ] && np=99
  if [ "$have" -eq 1 ] && [ "$np" -eq 0 ]; then
    log "seed-1 已完成（等待 ${waited} 分钟）"
    break
  fi
  if [ $((waited % 15)) -eq 0 ]; then
    log "等待 seed-1 中 ...（${waited} 分钟；CSV=${have}，python=${np}）"
  fi
  sleep 60
  waited=$((waited + 1))
done

if [ "$waited" -ge "$MAX_WAIT_MIN" ]; then
  log "!!!!! 等待 seed-1 超时（${MAX_WAIT_MIN} 分钟），放弃后续排程"
  log "########## 过夜串联队列(2) 结束 ##########"
  exit 0
fi

# ── SAM-AMG 基线（申报书点名的"原始 SAM Zero-shot"，约 10~20 分钟）──
log "===== 开始 SAM-AMG 基线（无泄漏划分）====="
"$PY" run_baselines_leakfree.py --methods sam_amg >> "$LOG" 2>&1
log "===== SAM-AMG 结束（退出码 $?）====="
sleep 30

# ── 健全性闸门：seed-1 结果不合理就不启动 seed-2 ──
# 目的：防"带着错误进入下一轮"——若 seed-1 本身跑崩/跑飞，seed-2 只是重复浪费 5.7h。
GATE=$("$PY" - <<'PYEOF' 2>/dev/null
import csv
try:
    with open("results/lora_leakfree_s1_test.csv", newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh)]
    r = next(x for x in rows if not x["method"].endswith("_std"))
    iou = float(r["IoU"]); ber = float(r["BER"])
    print("PASS" if (0.60 <= iou <= 0.95 and ber < 0.30) else f"FAIL iou={iou:.4f} ber={ber:.4f}")
except Exception as e:
    print(f"FAIL 读取失败 {type(e).__name__}")
PYEOF
)
log "健全性闸门(seed-1)：${GATE}（判据：0.60<=IoU<=0.95 且 BER<0.30；主结果 IoU 0.7878/BER 0.0644）"

case "$GATE" in
  PASS*)
    log "===== 闸门通过 -> 启动 LoRA seed=2（40 轮，约 5.7h）====="
    bash queue_lora_seed.sh 2 40
    log "===== LoRA seed=2 返回 ====="
    ;;
  *)
    log "!!!!! 闸门未通过 -> **不启动 seed-2**，等人工/监督审查后再决定"
    ;;
esac

log "########## 过夜串联队列(2) 结束 ##########"
