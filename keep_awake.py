"""防睡眠守护：长训练期间持续向 Windows 请求"不要睡眠"。

## 为什么需要

本机启用 **Modern Standby (S0 低电量待机)**（`powercfg /a` 实测：S1/S2 固件不支持）。
项目已知：把"睡眠超时"设为"从不"**挡不住** S0 进入（见 memory `modern-standby-crash`），
长训练会被不定期杀掉。本脚本用 Windows 官方的
`SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` 显式声明
"这个进程需要系统保持唤醒"，这比电源计划里的超时设置**更强**（会被 `powercfg /requests` 记录）。

## 用法
    nohup python keep_awake.py > /dev/null 2>&1 &          # 系统不睡（允许熄屏）
    nohup python keep_awake.py --keep-display > /dev/null 2>&1 &   # 连屏幕都不熄

## 边界（必须知道）
- 这是**尽力而为**，不是保证：S0 是否进入还受固件/驱动/合盖动作影响。
- 真正的兜底是训练队列的 `--resume` 自愈（见 `queue_maskrcnn.sh` / `queue_lora_seed.sh`）：
  即使被杀了，续训点还在，队列会自己接上。**两者要一起用。**
"""
import argparse
import sys
import time

ES_CONTINUOUS = 0x80000000       # 让设置持续生效，直到显式清除
ES_SYSTEM_REQUIRED = 0x00000001  # 系统保持唤醒
ES_DISPLAY_REQUIRED = 0x00000002 # 显示器也保持点亮


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-display", action="store_true",
                    help="连显示器也不熄（更强，但屏幕会一直亮着）")
    ap.add_argument("--interval", type=int, default=30, help="刷新间隔秒")
    args = ap.parse_args()

    if sys.platform != "win32":
        print("非 Windows 平台，无需本守护，退出。")
        return

    import ctypes
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    if args.keep_display:
        flags |= ES_DISPLAY_REQUIRED

    k32 = ctypes.windll.kernel32
    prev = None
    while True:
        rc = k32.SetThreadExecutionState(ctypes.c_uint(flags))
        if rc == 0 and prev != 0:
            print("SetThreadExecutionState 返回 0（请求被拒绝）", flush=True)
        prev = rc
        time.sleep(max(1, args.interval))


if __name__ == "__main__":
    main()
