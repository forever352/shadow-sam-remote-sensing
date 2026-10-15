#!/bin/bash
# 训练队列（可崩溃自愈）：每个任务用 --resume 循环重试，直到对应 _last.pth 的 epoch >= 目标轮数。
# 背景：本机启用 Modern Standby(S0)，传统"睡眠超时=从不"无法阻止进入低功耗态，
#       长训练会不定期被杀（2026-10-14 15:47 曾中断于 48/60）。故用 --resume + 自动重启兜底。
# 用法：nohup bash queue.sh > /dev/null 2>&1 &
cd "D:/RemoteSensing-Shadow-SAM-main/RemoteSensing-Shadow-SAM-main" || exit 1
LOG=logs/queue_console.log
PY=python

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

# 判断某任务的 _last.pth 是否已达目标轮数
# 注意：必须先 import env_setup，否则本机 torch 因 DLL 问题导入失败 → 永远返回 0 → 死循环
done_epochs () { # $1=arch $2=split
  "$PY" - "$1" "$2" <<'PYEOF' 2>/dev/null
import sys
sys.path.insert(0, ".")
import env_setup
import torch
arch, split = sys.argv[1], sys.argv[2]
p = f"checkpoints/{arch}_shadow_{split}_last.pth"
try:
    st = torch.load(p, map_location="cpu", weights_only=False)
    print(int(st["epoch"]))
except Exception:
    print(0)
PYEOF
}

# CNN 任务：跑到 epochs 轮为止，崩溃自动续训
run_cnn () { # $1=arch $2=split $3=epochs
  local arch=$1 split=$2 epochs=$3
  log "===== 开始 ${arch}/${split}（目标 ${epochs} 轮）====="
  while true; do
    "$PY" train_cnn_leakfree.py --arch "$arch" --split "$split" --epochs "$epochs" --resume >> "$LOG" 2>&1
    local ep; ep=$(done_epochs "$arch" "$split")
    if [ "$ep" -ge "$epochs" ]; then
      log "===== 完成 ${arch}/${split}：_last.pth epoch=$ep ====="
      break
    fi
    log "!! ${arch}/${split} 未完成（_last epoch=$ep < $epochs），疑似休眠中断，60s 后续训"
    sleep 60
  done
}

log ""
log "########## 队列启动 ##########"

# 顺序：先修主表局限（U-Net 无泄漏重跑），再补官方划分，最后方向网络
run_cnn unet  leakfree 60     # 消除主表 † "U-Net 58/60 中断"
run_cnn unet  official 60     # 补官方划分 U-Net
run_cnn segnet official 60    # 补官方划分 SegNet

# 方向网络 basis 重训（T0.3，脚本无 resume，短任务，跑一次）
log "===== 开始 方向网络 basis 重训 ====="
"$PY" train_direction_net.py --split basis >> "$LOG" 2>&1
log "===== 方向网络 basis 结束（退出码 $?）====="

log "########## 队列全部结束 ##########"
