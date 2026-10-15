#!/bin/bash
# 线性串行队列 v2（2026-10-15）：seed-1 -> Mask R-CNN@默认分辨率 -> SAM-AMG -> 闸门 -> seed-2
#
# ⚠️ 设计纪律（血的教训）：
#   1. **纯线性阻塞**，不做任何跨脚本的"进程数"轮询 —— 上一版用 python.exe 计数判断
#      GPU 空闲，被 keep_awake.py（也是 python.exe）永久卡死，白等 11.5 小时。
#   2. 同一脚本跑不同设置**必须给不同 --tag**，否则会互相覆盖、静默改写旧结果。
#
# 为什么要跑 Mask R-CNN@默认分辨率（约 2h，插在 seed-2 之前）：
#   第十四轮审查裁定 M1——已完成的 F1 用的是 min_size=512/max_size=800（**非 torchvision
#   默认 800/1333**，为省 8GB 显存而调），而主方法 LoRA-SAM 走 SAM 默认 1024。
#   即**基线在低于其自身默认、且低于主方法的分辨率下运行**，是**利我方**的系统性偏置。
#   这是申报书"IoU 优于 Mask R-CNN"判据可信度的关键，故必须补做或如实标注局限。
#
# 用法：nohup bash queue_chain.sh > /dev/null 2>&1 &
cd "D:/RemoteSensing-Shadow-SAM-main/RemoteSensing-Shadow-SAM-main" || exit 1

LOG=logs/chain_queue.log
PY=python

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

log ""
log "########## 线性串行队列 v2 启动：seed-1 -> MaskRCNN@800 -> SAM-AMG -> 闸门 -> seed-2 ##########"

# ══ [1/6] LoRA seed-1 ══
log "===== [1/6] 开始 LoRA seed=1（40 轮，约 5.1h）====="
bash queue_lora_seed.sh 1 40
log "===== [1/6] LoRA seed=1 返回 ====="

# ══ [2/6] Mask R-CNN 默认分辨率：先冒烟试显存 ══
log "===== [2/6] Mask R-CNN 800/1333 可行性冒烟（6 训练图 / 4 验证图 / 1 轮）====="
"$PY" train_maskrcnn_leakfree.py --epochs 1 --limit-train 6 --limit-val 4 --eval-every 1 \
    --min-size 800 --max-size 1333 --tag _res800 >> "$LOG" 2>&1
SMOKE_RC=$?
log "冒烟退出码 = $SMOKE_RC"

if [ "$SMOKE_RC" -eq 0 ]; then
  # ══ [3/6] 正式跑 800/1333 ══
  log "===== [3/6] 冒烟通过 -> 正式跑 Mask R-CNN @800/1333（tag=_res800，60 轮）====="
  fail=0
  while true; do
    if [ -f results/maskrcnn_leakfree_res800_test.csv ]; then
      log "[3/6] res800 结果 CSV 已生成"; break
    fi
    "$PY" train_maskrcnn_leakfree.py --epochs 60 --min-size 800 --max-size 1333 \
        --tag _res800 --resume >> "$LOG" 2>&1
    if [ -f results/maskrcnn_leakfree_res800_test.csv ]; then
      log "[3/6] res800 完成"; break
    fi
    fail=$((fail + 1))
    if [ "$fail" -ge 5 ]; then
      log "!!!!! [3/6] res800 连续 ${fail} 次无进展，放弃（请人工检查是否 OOM）"
      break
    fi
    log "!! [3/6] res800 未完成，第 ${fail}/5 次未进展，45s 后重试"
    sleep 45
  done
else
  log "!!!!! [3/6] 冒烟失败（很可能显存不足 OOM）-> **不跑** 800/1333"
  log "!!!!! 报告须**如实标注为局限**：Mask R-CNN 以非默认的较低分辨率运行，其真实上限可能被压低"
fi

# ══ [4/6] SAM-AMG 零样本基线 ══
log "===== [4/6] 开始 SAM-AMG 基线（申报书点名的原始 SAM zero-shot）====="
"$PY" run_baselines_leakfree.py --methods sam_amg >> "$LOG" 2>&1
log "===== [4/6] SAM-AMG 结束（退出码 $?）====="

# ══ [5/6] 健全性闸门（seed-1）══
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
log "===== [5/6] 健全性闸门(seed-1)：${GATE} ====="

# ══ [6/6] seed-2（仅闸门通过）══
case "$GATE" in
  PASS*)
    log "===== [6/6] 闸门通过 -> 开始 LoRA seed=2 ====="
    bash queue_lora_seed.sh 2 40
    log "===== [6/6] LoRA seed=2 返回 ====="
    ;;
  *)
    log "!!!!! [6/6] 闸门未通过 -> **不启动 seed-2**，等审查后再定"
    ;;
esac

log "########## 线性串行队列 v2 结束 ##########"
