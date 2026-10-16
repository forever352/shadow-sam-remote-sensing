"""诊断：方向 Adapter 动了多少？它的输出是否真的依赖方向？

## 这个脚本被审查判过两次"不通过"（教训记在这里，避免第三次）

**v1 的错误**：没固定随机种子就 build -> 参照是"另一次随机抽样"，不是训练起点。
**v2 的错误**：固定了种子，但**只复现了一条路径的 RNG 历史**。真相是：
  `train_direction_adapter.py` 的训练 RNG 序列是
    247: random.seed(0); np.random.seed(0); torch.manual_seed(0)
    270: **仅 regress 路径**调用 `load_direction_net()` -> 内部 `resnet18(weights=None)`
         （`direction_utils.py:225`）**消耗全局 torch RNG**
    272: `build_sam(...)` -> SAM + LoRA + adapter 初始化
  random 路径**跳过** 270。
  => 两条路径的 adapter 初始化**不共享 RNG 历史**。v2 只 seed 后直接 build，
     复现的只是 **random 路径**的起点，对 regress 路径仍是错参照。
  => v2 的"若参照错则两者都应 ≈sqrt(2)*||init||"论证**前提为假**，属无效论证。

**v3（本版）**：分别复现两条路径的 RNG 历史，各自用**自己的起点**做参照，
并做**交叉比对**（正确参照应使 Δ 明显小于交叉参照）。

## 三项检查
1. **路径专属参照**：`ref_random` = seed -> build；`ref_regress` = seed ->
   `load_direction_net()` -> build。对每个已训 adapter，同时报"对本路参照"与"对另一路参照"。
2. **零初始化末层**（`net.2`）：参照恒为 0、**与随机种子无关** -> 唯一的无争议测量。
3. **跨方向依赖性**：adapter 输出随方向变化的幅度占比（监督建议的功能检验）。
   注意：该比值大 ≠ 方向有意义（random 版的方向依赖来自**未训练的随机投影**）。

## 用法
    python diag_adapter_movement.py            # 会同时写 logs/diag_adapter_movement_{ts}.log
"""
import math
import time
from pathlib import Path

import env_setup  # 必须在 import torch 之前
import torch

from segment_anything import sam_model_registry
from train_adapter_v4 import apply_lora
from direction_adapter import install_direction_adapter, DirectionAdapter

BASE = Path(__file__).resolve().parent
CK = BASE / "checkpoints"
LOG_DIR = BASE / "logs"
OFFICIAL = str(CK / "sam_vit_b_01ec64.pth")
BASE_CKPT = CK / "sam_vit_b_shadow_v4_lora_leakfree.pth"
ADAPTERS = {
    "regress-trained": (CK / "sam_vit_b_shadow_v4_dir_adapter_main_leakfree.pth", "regress"),
    "random-trained":  (CK / "sam_vit_b_shadow_v4_dir_adapter_random_leakfree.pth", "random"),
}
FROZEN = CK / "sam_vit_b_shadow_frozen_leakfree.pth"

_logf = None


def log(s=""):
    print(s, flush=True)
    if _logf is not None:
        _logf.write(s + "\n")
        _logf.flush()


def _seed_all(seed=0):
    import random
    import numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def build_adapter_init(consume_direction_net: bool):
    """复现某条训练路径的 adapter 初始化。

    consume_direction_net=True  -> 复现 regress 路径（build 前先加载方向网，消耗 RNG）
    consume_direction_net=False -> 复现 random 路径（不调用）
    """
    _seed_all(0)
    if consume_direction_net:
        from direction_utils import load_direction_net
        load_direction_net(device="cpu")
    sam = sam_model_registry["vit_b"](checkpoint=OFFICIAL)
    apply_lora(sam)
    install_direction_adapter(sam, hidden=64)
    return dict(sam.prompt_encoder.direction_adapter.state_dict())


def load_sd(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    return sd["model"] if isinstance(sd, dict) and "model" in sd else sd


def ad_state(sd):
    return {k.replace("prompt_encoder.direction_adapter.", ""): v
            for k, v in sd.items() if k.startswith("prompt_encoder.direction_adapter.")}


def deltas(ref, trained):
    """返回 {tensor: (||ref||, ||trained||, ||delta||, rel)}。"""
    out = {}
    for k, v0 in ref.items():
        if k not in trained:
            continue
        v1 = trained[k]
        na = float(v0.float().norm()); nb = float(v1.float().norm())
        nd = float((v1.float() - v0.float()).norm())
        out[k] = (na, nb, nd, (nd / na) if na > 1e-12 else float("inf"))
    return out


def main():
    global _logf
    LOG_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    p = LOG_DIR / f"diag_adapter_movement_{ts}.log"
    _logf = open(p, "w", encoding="utf-8")

    log("# 方向 Adapter 位移 / 方向依赖性 诊断 (v3)")
    log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("# [!] v1 错在未固定种子；v2 错在只复现了一条路径的 RNG 历史。本版按路径分别复现。")

    # ---- (0) 两条路径的参照，及其交叉距离 ----
    ref_rnd = build_adapter_init(consume_direction_net=False)   # random 路径起点
    ref_reg = build_adapter_init(consume_direction_net=True)    # regress 路径起点
    log("")
    log("=== (0) 两条路径的参照（各自复现 RNG 历史）===")
    log("  这两份 init 若不同，即证明'两条路径不共享起点'（v2 的错误前提）")
    cross = deltas(ref_rnd, ref_reg)
    for k in sorted(cross):
        na, nb, nd, rel = cross[k]
        log(f"  {k:16s} ||ref_random||={na:10.6f}  ||ref_regress||={nb:10.6f}  "
            f"||diff||={nd:10.6f}  rel={rel:8.4f}")
    log("  参考: 若两向量相互独立，期望 ||diff|| ~= sqrt(2)*||ref|| -> 比值 ~1.41")

    # ---- (1) 每个 adapter 对「本路参照」与「交叉参照」 ----
    for label, (path, src) in ADAPTERS.items():
        if not path.exists():
            log(f"\n[{label}] 缺权重 {path.name} -- 跳过")
            continue
        trained = ad_state(load_sd(path))
        own = ref_reg if src == "regress" else ref_rnd
        other = ref_rnd if src == "regress" else ref_reg
        log("")
        log(f"=== (1) [{label}] {path.name}（训练路径方向源={src}）===")
        log(f"  {'tensor':16s} {'||init||':>11s} {'||trained||':>12s} "
            f"{'||d|本路':>11s} {'rel本路':>8s} {'||d|交叉':>11s} {'rel交叉':>8s}")
        d_own = deltas(own, trained); d_oth = deltas(other, trained)
        for k in sorted(d_own):
            na, nb, nd, rel = d_own[k]
            _, _, nd_o, rel_o = d_oth[k]
            tag = "  <- 零初始化(参照恒0)" if k.startswith("net.2") else ""
            log(f"  {k:16s} {na:11.6f} {nb:12.6f} {nd:11.6f} {rel:8.4f} "
                f"{nd_o:11.6f} {rel_o:8.4f}{tag}")
        log("  判读: 'rel本路' 明显小于 'rel交叉' -> 本路参照已复现（交叉那列是独立抽样的签名 ~1.41）。")
        log("  [!] 但 'rel本路' 只能读作『该层相对各自起点位移了百分之几』——")
        log("      **它不是一个标定过的『训练程度』判据**：本项目冻结编码器那轮 prompt_encoder/mask_decoder")
        log("      的相对位移只有 1.0%/3.9%，却是确实有效的训练。故不得据此断言某层『基本没训练』。")
        log("  [!] net.2 是零初始化，参照恒为 0、与种子无关 —— 唯一无争议的测量。")

        # 基座冻结校验（完整 state_dict）
        full = load_sd(path); base = load_sd(BASE_CKPT)
        common = [k for k in full if not k.startswith("prompt_encoder.direction_adapter.")
                  and k in base and full[k].shape == base[k].shape]
        worst = 0.0
        for k in common:
            worst = max(worst, float((full[k].float() - base[k].float()).abs().max()))
        log(f"  [基座冻结校验] 非 adapter 参数 {len(common)} 个，与 BASE_CKPT 最大逐元素差 = {worst:.3e}")

    # ---- (3) 参照量级：冻结编码器那轮（起点=官方 SAM，参照真实） ----
    if FROZEN.exists():
        fz = load_sd(FROZEN)
        ref = sam_model_registry["vit_b"](checkpoint=OFFICIAL).state_dict()
        log("")
        log(f"=== (3) [参照] {FROZEN.name} 相对官方 SAM（真实起点）===")
        for scope in ["prompt_encoder", "mask_decoder"]:
            na = nd = 0.0
            for k in ref:
                if k.startswith(scope + ".") and k in fz and fz[k].shape == ref[k].shape:
                    na += float(ref[k].float().norm()) ** 2
                    nd += float((fz[k].float() - ref[k].float()).norm()) ** 2
            na, nd = na ** 0.5, nd ** 0.5
            log(f"  {scope:18s} ||delta||={nd:12.6f}  ||init||={na:12.6f}  相对位移={(nd/na):.4f}")
        log("  注: 该轮 loss 0.449 -> 0.211（-53%），起点与参照均真实")

    # ---- (4) 跨方向依赖性（功能检验） ----
    log("")
    log("=== (4) 跨方向依赖性 ===")
    log("  [!] 比值大 ≠ 方向有意义：random 版的方向依赖来自**未训练的随机投影**。")
    for label, (path, _) in ADAPTERS.items():
        if not path.exists():
            continue
        ad = DirectionAdapter(hidden=64)
        ad.load_state_dict(ad_state(load_sd(path)))
        ad.eval()
        th = torch.arange(0.0, 180.0, 5.0) * math.pi / 180.0
        d = torch.stack([torch.cos(2 * th), torch.sin(2 * th)], dim=-1)
        with torch.no_grad():
            scale, shift = ad.net(d).chunk(2, dim=-1)
        for nm, t in (("scale", scale), ("shift", shift)):
            mag = float(t.norm(dim=-1).mean())
            std = float(t.std(dim=0).norm())
            log(f"  [{label}] ||{nm}|| 均值={mag:.4f}  跨方向std范数={std:.4f}  比值={std/mag:.4f}")

    log("")
    log(f"日志 -> {p}")
    _logf.close()


if __name__ == "__main__":
    main()
