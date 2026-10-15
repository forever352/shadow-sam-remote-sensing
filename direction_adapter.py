"""方向特征 Adapter（模块③）-- 把方向向量作为条件注入 SAM 的解码入口。

对应申报书模块3："冻结 SAM 图像编码器，改进掩膜解码器；在解码器中嵌入
方向特征适配器（Direction Adapter），将方向向量作为条件信息注入网络。"

## 注入点选择

注入在 `PromptEncoder.forward` **内部**（而非 `MaskDecoder`）。理由：
`PromptEncoder` 的输出（sparse/dense prompt embeddings）**就是解码器的输入**
（`mask_decoder.py:123/127` 直接消费），故等价于"在解码器入口注入"，且：
  - 既有 6 处调用点全部不用改（新参数默认 None）；
  - adapter 作为 `sam.prompt_encoder` 的子模块，参数**自动**进入
    `list(sam.prompt_encoder.parameters())`，从而被现有优化器 / `state_dict()` /
    `.to(device)` 体系覆盖，无需单独接线。

## [!] 为什么不能用"拼接一个零初始化 token"

`mask_decoder.py:123` 把 sparse token 拼进序列后，`TwoWayTransformer` 会对**全部**
token 做 softmax 归一化；即便新 token 数值为 0，只要它的 Q/K 非零（Linear 带 bias）
就会改变注意力分布 -> **初始状态不再等价于 LoRA-only**。

## [!] 为什么不能同时用 "gate=0" 和 "末层零初始化"（本项目踩过的坑）

曾设计为 `out = sparse*(1 + gate*tanh(scale)) + gate*shift` 且 `gate` 初值 0、
`net` 末层零初始化。二者相乘后，初始点对**全部** adapter 参数的偏导严格为 0：
    ∂out/∂shift = gate = 0
    ∂out/∂scale = gate * sparse = 0
    ∂out/∂gate  = sparse*tanh(0) + shift = 0
-> AdamW 更新恒为 0 -> **adapter 永远不更新**，训练完等价于没装。
更糟的是，"指标没变"会被误读成"方向信息无效"，实为"根本没训练"。

**正确做法（本实现）**：去掉 gate，改用 **FiLM 恒等初始化**：
    sparse_out = sparse * (1 + scale) + shift       # scale, shift 由零初始化末层产出
    dense_out  = dense  + shift
初始 ⇒ `scale=shift=0` ⇒ 输出 = 输入（恒等），且梯度正常流动（∂/∂shift = 1 != 0）。
这一点由 `--self-test` 的梯度自检强制保证（见 `grad_self_check`）。
"""
import argparse
import math
from typing import List

import env_setup  # 必须在 import torch 之前
import torch
import torch.nn as nn

EMBED_DIM = 256  # SAM ViT-B 的 transformer_dim（build_sam.py:87-98）


class DirectionAdapter(nn.Module):
    """把 (cos2θ, sin2θ) 映射为 FiLM 条件，作用于 sparse/dense prompt embeddings。

    恒等初始化：训练前 `forward(sparse, dense, d) == (sparse, dense)`（逐比特）。
    """

    def __init__(self, in_dim: int = 2, hidden: int = 64, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * embed_dim),
        )
        # 末层零初始化 ⇒ 初始 scale=shift=0 ⇒ FiLM γ=1, β=0（恒等）
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, sparse: torch.Tensor, dense: torch.Tensor,
                direction: torch.Tensor):
        b = sparse.shape[0]
        d = direction if isinstance(direction, torch.Tensor) else torch.as_tensor(direction)
        d = d.reshape(-1, 2).to(device=sparse.device, dtype=sparse.dtype)
        if d.shape[0] == 1 and b > 1:
            d = d.expand(b, 2)                     # 同一张图的多个 token 共享一个方向

        z = self.net(d)                            # [B, 2*embed_dim]
        scale, shift = z.chunk(2, dim=-1)          # 各 [B, embed_dim]

        sparse = sparse * (1.0 + scale).unsqueeze(1) + shift.unsqueeze(1)
        dense = dense + shift.view(-1, shift.shape[-1], 1, 1)
        return sparse, dense


# ══════════════════════════════════════════════════════════════════
# 安装 / 取参
# ══════════════════════════════════════════════════════════════════

def install_direction_adapter(sam, hidden: int = 64):
    """把 DirectionAdapter 挂到 `sam.prompt_encoder` 上（子模块，自动进参数体系）。

    必须在 `sam.to(device)` **之前**调用（与 LoRA 注入同约定）。
    """
    sam.prompt_encoder.direction_adapter = DirectionAdapter(hidden=hidden)
    return sam.prompt_encoder.direction_adapter


def direction_adapter_params(sam) -> List[nn.Parameter]:
    """只收集 adapter 自己的参数（用于 --freeze-base 分组与审计打印）。"""
    ad = getattr(sam.prompt_encoder, "direction_adapter", None)
    return list(ad.parameters()) if ad is not None else []


def trainable_params_with_adapter(sam, lora_params_fn):
    """可训练参数 = LoRA + prompt_encoder(含 adapter) + mask_decoder。

    [!] 标准写法 `lora + prompt_encoder + mask_decoder` 已自动含 adapter；
    本函数只是为了**显式断言** adapter 确实在里面，防止将来重构漏掉。
    """
    params = lora_params_fn(sam) + list(sam.prompt_encoder.parameters()) + \
        list(sam.mask_decoder.parameters())
    ids = {id(p) for p in params}
    missing = [p for p in direction_adapter_params(sam) if id(p) not in ids]
    assert not missing, "adapter 参数未进入可训练列表！"
    return params


# ══════════════════════════════════════════════════════════════════
# 自检
# ══════════════════════════════════════════════════════════════════

def equivalence_check(adapter: DirectionAdapter, b: int = 2, n: int = 7,
                      h: int = 8, w: int = 8, use_autocast: bool = True) -> float:
    """验证"注入方向"与"不注入"在初始化时逐比特等价，返回 max|Δ|。

    必须在 **autocast** 下也通过（训练就带 AMP）。
    """
    dev = next(adapter.parameters()).device
    dtype = torch.float32
    sparse = torch.randn(b, n, EMBED_DIM, device=dev, dtype=dtype)
    dense = torch.randn(b, EMBED_DIM, h, w, device=dev, dtype=dtype)
    d = torch.randn(b, 2, device=dev, dtype=dtype)
    d = d / d.norm(dim=1, keepdim=True)

    ctx = torch.amp.autocast("cuda", enabled=use_autocast and dev.type == "cuda")
    with torch.no_grad(), ctx:
        s2, d2 = adapter(sparse, dense, d)
    s2 = s2.float(); d2 = d2.float()
    return max(float((s2 - sparse).abs().max()), float((d2 - dense).abs().max()))


def grad_self_check(adapter: DirectionAdapter, b: int = 2, n: int = 7,
                    h: int = 8, w: int = 8, use_autocast: bool = True,
                    n_steps: int = 3) -> dict:
    """**关键自检**：跑 n_steps 次前向+反向，确认**每个参数都曾拿到非零梯度**。

    判据说明（重要，别把正常现象误判为 bug）：
      - 末层零初始化方案下，**第一步**反向时第一层（`net.0.*`）梯度必然是 0 --
        因为 `∂scale/∂h = W_last = 0` 阻断了路径。这与 LoRA 的 `B=0` 初始化同理，
        **不是死锁**；跑一步优化器后 `W_last != 0`，第二步起第一层就有梯度。
      - 真正的"零梯度死锁"特征是**所有参数、所有步都恒为 0**（gate 与末层同时零初始化）。
    因此判据取"N 步内每个参数至少一次非零梯度"，并对第 1 步与第 N 步分别记录。
    """
    dev = next(adapter.parameters()).device
    use_amp = use_autocast and dev.type == "cuda"
    opt = torch.optim.SGD(adapter.parameters(), lr=1e-2)

    ever_nonzero = {}
    step1 = {}
    step_last = {}
    names = [k for k, _ in adapter.named_parameters()]

    for step in range(n_steps):
        adapter.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            s2, d2 = adapter(
                torch.randn(b, n, EMBED_DIM, device=dev),
                torch.randn(b, EMBED_DIM, h, w, device=dev),
                torch.nn.functional.normalize(torch.randn(b, 2, device=dev), dim=1),
            )
            loss = s2.float().pow(2).mean() + d2.float().pow(2).mean()
        loss.backward()
        for k, p in adapter.named_parameters():
            gn = float(p.grad.norm()) if p.grad is not None else 0.0
            ever_nonzero[k] = ever_nonzero.get(k, False) or (gn > 0.0)
            if step == 0:
                step1[k] = gn
            step_last[k] = gn
        opt.step()
    adapter.zero_grad(set_to_none=True)

    return {"names": names, "step1": step1, "step_last": step_last,
            "ever_nonzero": ever_nonzero,
            "__dead__": [k for k in names if not ever_nonzero.get(k, False)]}


def _self_test():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {dev}")

    for amp in ([True, False] if dev == "cuda" else [False]):
        ad = DirectionAdapter().to(dev)
        delta = equivalence_check(ad, use_autocast=amp)
        tag = "autocast" if amp else "fp32"
        print(f"[1] 恒等性 ({tag})：max|Δ| = {delta:.3e}  (要求 < 1e-5)")
        assert delta < 1e-5, f"{tag} 下注入改变了输出，恒等初始化失败！"

    ad = DirectionAdapter().to(dev)
    before = [p.detach().clone() for p in ad.parameters()]
    rep = grad_self_check(ad)
    print(f"[2] 梯度自检（{3} 步 SGD，autocast 下）：")
    for k in rep["names"]:
        print(f"      {k:<28} step1={rep['step1'][k]:.3e}  "
              f"last={rep['step_last'][k]:.3e}  ever>0={rep['ever_nonzero'][k]}")
    assert not rep["__dead__"], f"零梯度死锁（参数永远拿不到梯度）：{rep['__dead__']}"
    print("      -> 每个参数都曾拿到非零梯度，adapter 可正常训练")

    moved = max(float((p.detach() - q).abs().max()) for p, q in zip(ad.parameters(), before))
    print(f"[3] {3} 步 SGD 后参数最大变化量：{moved:.3e}  (要求 > 0)")
    assert moved > 0, "参数未更新！"
    print("\n[OK] direction_adapter 自测全部通过")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    _self_test()
