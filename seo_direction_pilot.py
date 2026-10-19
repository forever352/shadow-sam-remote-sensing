# -*- coding: utf-8 -*-
"""S-EO 方向可学性试点（v2）。

协议见 results/seo_direction_pilot/PREREGISTRATION.md。
只做方向回归，不接 AISD 下游，不碰 AISD 主表。

用法：
    python seo_direction_pilot.py --dry-run
    python seo_direction_pilot.py --mode smoke --epochs 2 --seeds 0
    python seo_direction_pilot.py --mode overfit --epochs 100 --seeds 0
    python seo_direction_pilot.py --mode all --epochs 30 --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import env_setup  # 必须在 torch 之前
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import ResNet18_Weights, resnet18

BASE = Path(__file__).resolve().parent
DATA = BASE / "datasets" / "S-EO_subset"
MANIFEST = DATA / "manifest_all.csv"
OUT_DIR = BASE / "results" / "seo_direction_pilot"
LOG_DIR = BASE / "logs"
CKPT_DIR = BASE / "checkpoints" / "seo_pilot"
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGENET_CKPT = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "resnet18-f37072fd.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Item:
    domain: str
    image_id: str
    aoi: str
    path: Path
    sun_az: float
    sun_el: float
    label: np.ndarray
    split: str
    date: str


def load_manifest() -> list[Item]:
    items: list[Item] = []
    with open(MANIFEST, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            path = DATA / r["image_path"].replace("\\", "/")
            items.append(Item(
                domain=r["domain"],
                image_id=r["image_id"],
                aoi=r["aoi"],
                path=path,
                sun_az=float(r["sun_azimuth_deg"]),
                sun_el=float(r["sun_elevation_deg"]),
                label=np.array([float(r["cos2theta"]), float(r["sin2theta"])], dtype=np.float32),
                split=r["split"],
                date=r["acquisition_date"],
            ))
    return items


def read_rgb(item: Item) -> Image.Image:
    if item.domain == "OMA":
        return Image.open(item.path).convert("RGB")
    with rasterio.open(item.path) as ds:
        arr = ds.read().astype(np.float32)  # (4,H,W), 前 3 波段 = R,G,B
    rgb = arr[:3].transpose(1, 2, 0)
    lo, hi = np.percentile(rgb, [2, 98], axis=(0, 1))
    rgb8 = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1) * 255.0
    return Image.fromarray(rgb8.astype(np.uint8), mode="RGB")


def preload(items: list[Item]) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for i, it in enumerate(items, 1):
        img = read_rgb(it).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - MEAN) / STD
        xs.append(torch.from_numpy(arr.transpose(2, 0, 1)))
        ys.append(torch.from_numpy(it.label))
        if i % 25 == 0 or i == len(items):
            print(f"  preload {i}/{len(items)}", flush=True)
    return torch.stack(xs), torch.stack(ys)


def build_model() -> torch.nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    return model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def angle_error_deg(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    cos = torch.clamp((pred * target).sum(dim=1), -1.0, 1.0)
    return (torch.acos(cos) * 180.0 / math.pi / 2.0).cpu().numpy()


def circular_mean_np(labels: np.ndarray) -> np.ndarray:
    m = labels.mean(axis=0)
    n = float(np.linalg.norm(m))
    return (m / n).astype(np.float32) if n > 0 else np.array([1.0, 0.0], dtype=np.float32)


def elevation_oracle(train_items: list[Item], eval_items: list[Item]) -> np.ndarray:
    """真值高度角 -> 方位角的二次多项式 oracle；返回 eval 的绝对角误差。"""
    el = np.array([it.sun_el for it in train_items], dtype=np.float64)
    az = np.array([it.sun_az for it in train_items], dtype=np.float64)
    X = np.column_stack([np.ones_like(el), el, el ** 2])
    c = np.linalg.lstsq(X, np.cos(np.deg2rad(2 * az)), rcond=None)[0]
    s = np.linalg.lstsq(X, np.sin(np.deg2rad(2 * az)), rcond=None)[0]
    el_e = np.array([it.sun_el for it in eval_items], dtype=np.float64)
    Xe = np.column_stack([np.ones_like(el_e), el_e, el_e ** 2])
    pred = (np.degrees(np.arctan2(Xe @ s, Xe @ c)) / 2.0) % 180.0
    true = np.array([it.sun_az for it in eval_items], dtype=np.float64)
    return np.abs(((pred - true + 90.0) % 180.0) - 90.0)


def target_elevation_oracle_cv(items: list[Item]) -> np.ndarray:
    """目标域高程 oracle 的交叉拟合：单 AOI 用 LOO，多 AOI 用 LOAO。

    与主实验的折结构一致，避免 in-sample oracle 过于乐观。
    """
    errs = np.zeros(len(items), dtype=np.float64)
    aois = sorted({it.aoi for it in items})
    if len(aois) == 1:
        for i, it in enumerate(items):
            train = [x for j, x in enumerate(items) if j != i]
            errs[i] = elevation_oracle(train, [it])[0]
    else:
        for aoi in aois:
            train = [it for it in items if it.aoi != aoi]
            idx = [i for i, it in enumerate(items) if it.aoi == aoi]
            errs[idx] = elevation_oracle(train, [items[i] for i in idx])
    return errs


def train_and_eval(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    shuffle_labels: bool = False,
) -> tuple[np.ndarray, float, float]:
    """返回 (eval 角误差, 最终训练 loss 均值, 训练集角误差)。"""
    set_seed(seed)
    y_fit = y_train
    if shuffle_labels:
        gen = torch.Generator().manual_seed(seed + 90001)
        perm = torch.randperm(len(y_train), generator=gen)
        y_fit = y_train[perm]

    model = build_model().to(device)
    ds = TensorDataset(x_train, y_fit)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    model.train()
    losses: list[float] = []
    for _ in range(epochs):
        epoch_losses = []
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = F.normalize(model(xb), dim=1)
            loss = (1.0 - (pred * yb).sum(dim=1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.item()))
        sched.step()
        losses.append(float(np.mean(epoch_losses)))

    model.eval()
    with torch.no_grad():
        pred_eval = F.normalize(model(x_eval.to(device)), dim=1)
        eval_err = angle_error_deg(pred_eval, y_eval.to(device))
        pred_train = F.normalize(model(x_train.to(device)), dim=1)
        train_err = angle_error_deg(pred_train, y_fit.to(device))
    return eval_err, losses, float(np.mean(train_err))


def paired_signflip(diffs: np.ndarray, n_perm: int = 10000, seed: int = 0) -> float:
    """配对 sign-flip 置换检验；折间相关，p 只作描述性证据。"""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=np.float64)
    obs = abs(float(diffs.mean()))
    cnt = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(diffs))
        if abs(float((diffs * signs).mean())) >= obs:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def date_blocks(dates: list[str], n_blocks: int) -> np.ndarray:
    order = sorted(range(len(dates)), key=lambda i: dates[i])
    block_of = np.empty(len(dates), dtype=np.int64)
    for rank, idx in enumerate(order):
        block_of[idx] = min(rank * n_blocks // len(dates), n_blocks - 1)
    return block_of


def block_signflip(diffs: np.ndarray, blocks: np.ndarray, n_perm: int = 10000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=np.float64)
    uniq = np.unique(blocks)
    block_means = np.array([diffs[blocks == b].mean() for b in uniq])
    obs = abs(float(block_means.mean()))
    cnt = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(block_means))
        if abs(float((block_means * signs).mean())) >= obs:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diffs)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        means[i] = diffs[rng.integers(0, n, n)].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def summary_stats(name: str, model_errs: np.ndarray, const_errs: np.ndarray,
                  elev_errs: np.ndarray, shuffle_errs: np.ndarray | None,
                  blocks: np.ndarray | None, log) -> dict:
    diffs_c = model_errs - const_errs
    diffs_e = model_errs - elev_errs
    lo_c, hi_c = bootstrap_ci(diffs_c)
    lo_e, hi_e = bootstrap_ci(diffs_e)
    out = {
        "experiment": name,
        "n_folds": int(len(model_errs)),
        "model_mean_deg": float(model_errs.mean()),
        "model_median_deg": float(np.median(model_errs)),
        "model_std_deg": float(model_errs.std(ddof=1)) if len(model_errs) > 1 else 0.0,
        "model_iqr_deg": float(np.percentile(model_errs, 75) - np.percentile(model_errs, 25)),
        "model_worst_fold_deg": float(model_errs.max()),
        "constant_mean_deg": float(const_errs.mean()),
        "constant_median_deg": float(np.median(const_errs)),
        "elevation_oracle_mean_deg": float(elev_errs.mean()),
        "elevation_oracle_median_deg": float(np.median(elev_errs)),
        "model_minus_constant_deg": float(diffs_c.mean()),
        "model_minus_constant_ci_lo": lo_c,
        "model_minus_constant_ci_hi": hi_c,
        "relative_reduction_vs_constant": float(1.0 - model_errs.mean() / const_errs.mean()),
        "model_minus_elevation_deg": float(diffs_e.mean()),
        "model_minus_elevation_ci_lo": lo_e,
        "model_minus_elevation_ci_hi": hi_e,
        "relative_reduction_vs_elevation": float(1.0 - model_errs.mean() / elev_errs.mean()),
        "signflip_p_vs_constant": paired_signflip(diffs_c),
        "signflip_p_vs_elevation": paired_signflip(diffs_e),
        "random_deg": 45.0,
    }
    if blocks is not None:
        out["block_signflip_p_vs_constant"] = block_signflip(diffs_c, blocks)
        out["block_signflip_p_vs_elevation"] = block_signflip(diffs_e, blocks)
    if shuffle_errs is not None:
        diffs_s = model_errs - shuffle_errs
        out.update({
            "shuffle_mean_deg": float(shuffle_errs.mean()),
            "model_minus_shuffle_deg": float(diffs_s.mean()),
            "relative_reduction_vs_shuffle": float(1.0 - model_errs.mean() / shuffle_errs.mean()),
            "signflip_p_vs_shuffle": paired_signflip(diffs_s),
        })
        if blocks is not None:
            out["block_signflip_p_vs_shuffle"] = block_signflip(diffs_s, blocks)
    log(json.dumps(out, ensure_ascii=False))
    return out


def seed_means(rows: list[dict], seeds: list[int], prefix: str = "model_err") -> dict:
    out = {}
    for s in seeds:
        vals = [r[f"{prefix}_s{s}_deg"] for r in rows]
        out[f"{prefix}_mean_s{s}_deg"] = float(np.mean(vals))
    return out


def run_loocv(items: list[Item], x: torch.Tensor, y: torch.Tensor,
              epochs: int, lr: float, batch_size: int, seeds: list[int],
              with_shuffle: bool, log, fold_limit: int = 0) -> tuple[list[dict], dict]:
    assert len({it.date for it in items}) == len(items), "OMA 日期必须唯一，才能按日期留一"
    rows = []
    loss_rows = []
    n = len(items)
    n_folds = n if fold_limit <= 0 else min(fold_limit, n)
    for fold in range(n_folds):
        t0 = time.time()
        held = items[fold]
        train_items = [it for it in items if it.date != held.date]
        train_idx = [i for i, it in enumerate(items) if it.date != held.date]
        train_idx_t = torch.as_tensor(train_idx, dtype=torch.long)
        x_tr, y_tr = x[train_idx_t], y[train_idx_t]
        x_ev, y_ev = x[fold:fold + 1], y[fold:fold + 1]
        const = circular_mean_np(y_tr.numpy())
        const_err = float(angle_error_deg(torch.from_numpy(const)[None], y_ev)[0])
        elev_err = float(elevation_oracle(train_items, [held])[0])

        model_errs, shuffle_errs, train_errs, train_losses = [], [], [], []
        for seed in seeds:
            ev, losses, tr = train_and_eval(
                x_tr, y_tr, x_ev, y_ev, epochs, lr, batch_size, seed, False)
            model_errs.append(float(ev[0]))
            train_errs.append(tr)
            train_losses.append(float(np.mean(losses)))
            for ep, lv in enumerate(losses, 1):
                loss_rows.append({"fold": fold, "heldout_id": held.image_id,
                                  "seed": seed, "shuffle": 0, "epoch": ep, "loss": lv})
            if with_shuffle:
                ev_s, losses_s, tr_s = train_and_eval(
                    x_tr, y_tr, x_ev, y_ev, epochs, lr, batch_size, seed, True)
                shuffle_errs.append(float(ev_s[0]))
                for ep, lv in enumerate(losses_s, 1):
                    loss_rows.append({"fold": fold, "heldout_id": held.image_id,
                                      "seed": seed, "shuffle": 1, "epoch": ep, "loss": lv})

        row = {
            "fold": fold,
            "heldout_id": held.image_id,
            "heldout_date": held.date,
            "sun_azimuth_deg": held.sun_az,
            "sun_elevation_deg": held.sun_el,
            "constant_pred_azimuth_deg": float(np.degrees(np.arctan2(const[1], const[0])) / 2.0) % 180.0,
            "constant_err_deg": const_err,
            "elevation_oracle_err_deg": elev_err,
            "model_err_mean_deg": float(np.mean(model_errs)),
            "model_err_std_deg": float(np.std(model_errs, ddof=1)) if len(model_errs) > 1 else 0.0,
            "train_err_mean_deg": float(np.mean(train_errs)),
            "train_loss_mean": float(np.mean(train_losses)),
        }
        for seed, e, te, tl in zip(seeds, model_errs, train_errs, train_losses):
            row[f"model_err_s{seed}_deg"] = e
            row[f"train_err_s{seed}_deg"] = te
            row[f"train_loss_s{seed}"] = tl
        if with_shuffle:
            row["shuffle_err_mean_deg"] = float(np.mean(shuffle_errs))
            for seed, e in zip(seeds, shuffle_errs):
                row[f"shuffle_err_s{seed}_deg"] = e
        rows.append(row)
        log(f"[LOOCV {fold + 1:02d}/{n}] {held.image_id}  "
            f"model={row['model_err_mean_deg']:.2f}°  const={const_err:.2f}°  "
            f"elev={elev_err:.2f}°"
            + (f"  shuffle={row['shuffle_err_mean_deg']:.2f}°" if with_shuffle else "")
            + f"  train={row['train_err_mean_deg']:.2f}°  ({time.time() - t0:.1f}s)")

    model_errs = np.array([r["model_err_mean_deg"] for r in rows], dtype=np.float64)
    const_errs = np.array([r["constant_err_deg"] for r in rows], dtype=np.float64)
    elev_errs = np.array([r["elevation_oracle_err_deg"] for r in rows], dtype=np.float64)
    shuffle_errs = (np.array([r["shuffle_err_mean_deg"] for r in rows], dtype=np.float64)
                    if with_shuffle else None)
    blocks = date_blocks([r["heldout_date"] for r in rows], 6)
    summary = summary_stats("OMA_LOOCV", model_errs, const_errs, elev_errs,
                            shuffle_errs, blocks, log)
    summary.update(seed_means(rows, seeds))
    summary["effective_unit"] = "date (42 folds share training data; correlated)"
    write_csv(OUT_DIR / "oma_loocv_training_losses.csv", loss_rows)
    return rows, summary


def run_jax_aoi(items: list[Item], x: torch.Tensor, y: torch.Tensor,
                epochs: int, lr: float, batch_size: int, seeds: list[int],
                with_shuffle: bool, log) -> tuple[list[dict], dict]:
    aois = sorted({it.aoi for it in items})
    rows = []
    per_image_rows = []
    for k, aoi in enumerate(aois):
        t0 = time.time()
        train_idx = [i for i, it in enumerate(items) if it.aoi != aoi]
        eval_idx = [i for i, it in enumerate(items) if it.aoi == aoi]
        train_items = [items[i] for i in train_idx]
        eval_items = [items[i] for i in eval_idx]
        x_tr, y_tr = x[torch.as_tensor(train_idx)], y[torch.as_tensor(train_idx)]
        x_ev, y_ev = x[torch.as_tensor(eval_idx)], y[torch.as_tensor(eval_idx)]
        const = circular_mean_np(y_tr.numpy())
        const_err = angle_error_deg(torch.from_numpy(const)[None].repeat(len(eval_items), 1), y_ev)
        elev_err = elevation_oracle(train_items, eval_items)

        model_errs, shuffle_errs, train_losses = [], [], []
        per_seed_images = {s: [] for s in seeds}
        for seed in seeds:
            ev, losses, tr = train_and_eval(
                x_tr, y_tr, x_ev, y_ev, epochs, lr, batch_size, seed, False)
            model_errs.append(float(ev.mean()))
            train_losses.append(float(np.mean(losses)))
            per_seed_images[seed] = ev
            if with_shuffle:
                ev_s, _, _ = train_and_eval(
                    x_tr, y_tr, x_ev, y_ev, epochs, lr, batch_size, seed, True)
                shuffle_errs.append(float(ev_s.mean()))
        row = {
            "fold": k,
            "heldout_aoi": aoi,
            "n_eval": len(eval_items),
            "model_err_mean_deg": float(np.mean(model_errs)),
            "model_err_std_deg": float(np.std(model_errs, ddof=1)) if len(model_errs) > 1 else 0.0,
            "constant_mean_deg": float(const_err.mean()),
            "elevation_oracle_mean_deg": float(elev_err.mean()),
            "train_loss_mean": float(np.mean(train_losses)),
        }
        for seed in seeds:
            row[f"model_err_s{seed}_deg"] = float(per_seed_images[seed].mean())
        if with_shuffle:
            row["shuffle_err_mean_deg"] = float(np.mean(shuffle_errs))
        rows.append(row)
        for j, gi in enumerate(eval_idx):
            it = items[gi]
            image_row = {
                "fold": k,
                "heldout_aoi": aoi,
                "image_id": it.image_id,
                "date": it.date,
                "sun_azimuth_deg": it.sun_az,
                "sun_elevation_deg": it.sun_el,
                "constant_err_deg": float(const_err[j]),
                "elevation_oracle_err_deg": float(elev_err[j]),
                "model_err_mean_deg": float(np.mean([per_seed_images[s][j] for s in seeds])),
            }
            for seed in seeds:
                image_row[f"model_err_s{seed}_deg"] = float(per_seed_images[seed][j])
            per_image_rows.append(image_row)
        log(f"[JAX-AOI {k + 1}/{len(aois)}] heldout={aoi} n={len(eval_items)}  "
            f"model={row['model_err_mean_deg']:.2f}°  const={row['constant_mean_deg']:.2f}°  "
            f"elev={row['elevation_oracle_mean_deg']:.2f}°  ({time.time() - t0:.1f}s)")

    model_errs = np.array([r["model_err_mean_deg"] for r in rows], dtype=np.float64)
    const_errs = np.array([r["constant_mean_deg"] for r in rows], dtype=np.float64)
    elev_errs = np.array([r["elevation_oracle_mean_deg"] for r in rows], dtype=np.float64)
    shuffle_errs = (np.array([r["shuffle_err_mean_deg"] for r in rows], dtype=np.float64)
                    if with_shuffle else None)
    summary = summary_stats("JAX_LOAO", model_errs, const_errs, elev_errs,
                            shuffle_errs, None, log)
    summary.update(seed_means(rows, seeds))
    summary["effective_unit"] = "AOI (4 folds only; descriptive, no population significance)"
    summary["signflip_p_note"] = "4-fold LOAO: signflip p is not inferential"
    write_csv(OUT_DIR / "jax_loao_per_image.csv", per_image_rows)
    return rows, summary


def run_cross(src_items: list[Item], dst_items: list[Item],
              x_src: torch.Tensor, y_src: torch.Tensor,
              x_dst: torch.Tensor, y_dst: torch.Tensor,
              name: str, epochs: int, lr: float, batch_size: int,
              seeds: list[int], log, save_ckpt: bool = False) -> tuple[list[dict], dict]:
    src_az = np.array([it.sun_az for it in src_items], dtype=np.float64)
    src_min, src_max = float(src_az.min()), float(src_az.max())
    in_support = np.array([src_min <= it.sun_az <= src_max for it in dst_items])

    src_const = circular_mean_np(y_src.numpy())
    dst_const = circular_mean_np(y_dst.numpy())
    src_const_err = angle_error_deg(
        torch.from_numpy(src_const)[None].repeat(len(dst_items), 1), y_dst)
    dst_const_err = angle_error_deg(
        torch.from_numpy(dst_const)[None].repeat(len(dst_items), 1), y_dst)
    dst_elev_err = target_elevation_oracle_cv(dst_items)

    per_seed: dict[int, np.ndarray] = {}
    for seed in seeds:
        t0 = time.time()
        set_seed(seed)
        model = build_model().to(device)
        ds = TensorDataset(x_src, y_src)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
        model.train()
        for _ in range(epochs):
            for xb, yb in dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = F.normalize(model(xb), dim=1)
                loss = (1.0 - (pred * yb).sum(dim=1)).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            pred = F.normalize(model(x_dst.to(device)), dim=1)
            per_seed[seed] = angle_error_deg(pred, y_dst.to(device))
        if save_ckpt:
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CKPT_DIR / f"{name}_s{seed}.pth")
        log(f"[{name} s{seed}] all={per_seed[seed].mean():.2f}°  "
            f"in={per_seed[seed][in_support].mean():.2f}°  out={per_seed[seed][~in_support].mean():.2f}°  "
            f"src_const={src_const_err.mean():.2f}°  dst_elev={dst_elev_err.mean():.2f}°  "
            f"({time.time() - t0:.1f}s)")

    rows = []
    for i, it in enumerate(dst_items):
        row = {
            "image_id": it.image_id,
            "aoi": it.aoi,
            "date": it.date,
            "sun_azimuth_deg": it.sun_az,
            "sun_elevation_deg": it.sun_el,
            "in_source_support": int(in_support[i]),
            "source_const_err_deg": float(src_const_err[i]),
            "target_oracle_const_err_deg": float(dst_const_err[i]),
            "target_elevation_oracle_err_deg": float(dst_elev_err[i]),
            "model_err_mean_deg": float(np.mean([per_seed[s][i] for s in seeds])),
        }
        for seed in seeds:
            row[f"model_err_s{seed}_deg"] = float(per_seed[seed][i])
        rows.append(row)

    model_all = np.array([r["model_err_mean_deg"] for r in rows])
    const_all = np.array([r["source_const_err_deg"] for r in rows])
    elev_all = np.array([r["target_elevation_oracle_err_deg"] for r in rows])
    summary_all = summary_stats(f"{name}_all", model_all, const_all, elev_all, None, None, log)
    summary_in = summary_stats(f"{name}_in_support", model_all[in_support],
                               const_all[in_support], elev_all[in_support], None, None, log)
    summary_out = summary_stats(f"{name}_out_support", model_all[~in_support],
                                const_all[~in_support], elev_all[~in_support], None, None, log) \
        if (~in_support).any() else {"experiment": f"{name}_out_support", "n_folds": 0}
    summary_all["n_in_support"] = int(in_support.sum())
    summary_all["n_out_support"] = int((~in_support).sum())
    summary_all["source_az_min"] = src_min
    summary_all["source_az_max"] = src_max
    summary_all["target_oracle_const_mean_deg"] = float(dst_const_err.mean())
    summary_all["target_elevation_oracle_mean_deg"] = float(dst_elev_err.mean())
    summary_in["target_oracle_const_mean_deg"] = float(dst_const_err[in_support].mean())
    if (~in_support).any():
        summary_out["target_oracle_const_mean_deg"] = float(dst_const_err[~in_support].mean())
    summary_all.update(seed_means(rows, seeds))
    return rows, {
        "all": summary_all,
        "in": summary_in,
        "out": summary_out,
        "by_aoi": aggregate_rows(rows, "aoi"),
        "by_date": aggregate_rows(rows, "date"),
    }


def aggregate_rows(rows: list[dict], key: str) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(str(r[key]), []).append(r)
    out = []
    for k, rs in sorted(groups.items()):
        out.append({
            key: k,
            "n": len(rs),
            "model_mean_deg": float(np.mean([r["model_err_mean_deg"] for r in rs])),
            "source_const_mean_deg": float(np.mean([r["source_const_err_deg"] for r in rs])),
            "target_oracle_const_mean_deg": float(np.mean([r["target_oracle_const_err_deg"] for r in rs])),
            "target_elevation_oracle_mean_deg": float(np.mean([r["target_elevation_oracle_err_deg"] for r in rs])),
        })
    return out


def run_overfit(items: list[Item], x: torch.Tensor, y: torch.Tensor,
                epochs: int, lr: float, batch_size: int, seed: int, log) -> dict:
    idx = torch.arange(min(10, len(items)))
    xo, yo = x[idx], y[idx]
    ev, losses, train_err = train_and_eval(xo, yo, xo, yo, epochs, lr, batch_size, seed, False)
    out = {
        "experiment": "positive_control_overfit10",
        "n_images": int(len(idx)),
        "epochs": epochs,
        "train_loss_mean": float(np.mean(losses)),
        "train_err_deg": train_err,
        "eval_err_deg": float(ev.mean()),
        "pass": bool(train_err < 2.0),
    }
    log("positive control: " + json.dumps(out, ensure_ascii=False))
    return out


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"


def reproducibility_info() -> dict:
    info = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "rasterio": rasterio.__version__,
        "device": device,
        "script_sha256": sha256_file(BASE / "seo_direction_pilot.py"),
        "prereg_sha256": sha256_file(OUT_DIR / "PREREGISTRATION.md"),
        "diagnose_script_sha256": sha256_file(BASE / "diagnose_seo_labels.py"),
    }
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(BASE), text=True).strip()
        info["git_status_porcelain"] = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(BASE), text=True).strip()[:4000]
    except Exception:
        info["git_commit"] = "unknown"
        info["git_status_porcelain"] = "unknown"
    info["manifest_sha256"] = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    if IMAGENET_CKPT.exists():
        info["imagenet_weight_sha256"] = hashlib.sha256(IMAGENET_CKPT.read_bytes()).hexdigest()
    return info


def write_summary_md(loocv: dict | None, jax: dict | None, cross: dict | None,
                     overfit: dict | None, path: Path) -> None:
    lines = ["# S-EO 方向可学性试点 · 结果摘要", ""]
    if overfit is not None:
        lines += ["## 正对照（10 图过拟合）", "",
                  f"- 训练误差：{overfit['train_err_deg']:.2f}°；通过（<2°）：{overfit['pass']}", ""]
    if loocv is not None:
        lines += ["## OMA 留一日期", "",
                  f"- 模型：{loocv['model_mean_deg']:.2f}°（中位 {loocv['model_median_deg']:.2f}°）",
                  f"- 常数基线：{loocv['constant_mean_deg']:.2f}°",
                  f"- 高程-only oracle：{loocv['elevation_oracle_mean_deg']:.2f}°",
                  f"- 相对常数下降：{loocv['relative_reduction_vs_constant'] * 100:.1f}%",
                  f"- 相对高程 oracle 下降：{loocv['relative_reduction_vs_elevation'] * 100:.1f}%",
                  f"- 日期块 p（vs 常数）：{loocv.get('block_signflip_p_vs_constant', float('nan')):.4f}",
                  f"- 日期块 p（vs 高程）：{loocv.get('block_signflip_p_vs_elevation', float('nan')):.4f}",
                  ""]
    if jax is not None:
        lines += ["## JAX 留一 AOI", "",
                  f"- 模型：{jax['model_mean_deg']:.2f}°",
                  f"- 常数基线：{jax['constant_mean_deg']:.2f}°",
                  f"- 高程-only oracle：{jax['elevation_oracle_mean_deg']:.2f}°",
                  ""]
    if cross is not None:
        lines += ["## 跨域零样本", ""]
        for key, title in [("oma_to_jax", "OMA→JAX"), ("jax_to_oma", "JAX→OMA")]:
            if key in cross:
                s = cross[key]["all"]
                si = cross[key]["in"]
                so = cross[key]["out"]
                lines += [f"### {title}", "",
                          f"- 支撑内 {si['n_folds']} 张：模型 {si['model_mean_deg']:.2f}°，"
                          f"源常数 {si['constant_mean_deg']:.2f}°，"
                          f"目标高程 oracle {si['elevation_oracle_mean_deg']:.2f}°，"
                          f"目标均值 oracle {si.get('target_oracle_const_mean_deg', float('nan')):.2f}°",
                          f"- 支撑外 {so.get('n_folds', 0)} 张：模型 {so.get('model_mean_deg', float('nan')):.2f}°，"
                          f"源常数 {so.get('constant_mean_deg', float('nan')):.2f}°，"
                          f"目标均值 oracle {so.get('target_oracle_const_mean_deg', float('nan')):.2f}°",
                          ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dry-run", "smoke", "overfit", "loocv", "shuffle", "cross",
                                       "jax_aoi", "all"], default="dry-run")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--save-ckpt", action="store_true")
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--fold-limit", type=int, default=0,
                    help="仅用于冒烟：限制 LOOCV 折数；0=全部")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    OUT_DIR = Path(args.out_dir)

    items = load_manifest()
    oma = [it for it in items if it.domain == "OMA"]
    jax = [it for it in items if it.domain == "JAX"]
    print(f"manifest: OMA={len(oma)} JAX={len(jax)} device={device}", flush=True)

    if args.mode == "dry-run":
        print("dry-run: preloading OMA...", flush=True)
        xo, yo = preload(oma)
        print("dry-run: preloading JAX...", flush=True)
        xj, yj = preload(jax)
        print(f"OMA tensor {tuple(xo.shape)} labels {tuple(yo.shape)}; "
              f"JAX tensor {tuple(xj.shape)} labels {tuple(yj.shape)}", flush=True)
        print("label norms OMA/JAX:", float(yo.norm(dim=1).mean()), float(yj.norm(dim=1).mean()))
        print("dry-run ok; no training started.", flush=True)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"seo_direction_pilot_{ts}.log"
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg: str = "") -> None:
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log("# S-EO direction pilot v2")
    log(f"# time={time.strftime('%Y-%m-%d %H:%M:%S')} mode={args.mode} epochs={args.epochs} "
        f"lr={args.lr} batch={args.batch_size} seeds={seeds}")
    log("# reproducibility: " + json.dumps(reproducibility_info(), ensure_ascii=False))
    log(f"# data: OMA={len(oma)} JAX={len(jax)} manifest={MANIFEST}")

    log("preloading OMA...")
    xo, yo = preload(oma)
    log("preloading JAX...")
    xj, yj = preload(jax)

    if args.mode == "smoke":
        log("smoke: main 1 fold/2 epochs; shuffle 1 fold/1 epoch; cross 1 epoch each")
        tr_idx = torch.arange(1, len(oma))
        main_err = train_and_eval(xo[tr_idx], yo[tr_idx], xo[0:1], yo[0:1],
                                  min(args.epochs, 2), args.lr, args.batch_size, seeds[0], False)
        log(f"smoke main: heldout={oma[0].image_id} eval={main_err[0]} train_err={main_err[2]:.2f}")
        shuf_err = train_and_eval(xo[tr_idx], yo[tr_idx], xo[0:1], yo[0:1],
                                  min(args.epochs, 1), args.lr, args.batch_size, seeds[0], True)
        log(f"smoke shuffle: eval={shuf_err[0]} train_err={shuf_err[2]:.2f}")
        _, sm1 = run_cross(oma, jax, xo, yo, xj, yj, "smoke_oma_to_jax",
                           min(args.epochs, 1), args.lr, args.batch_size, [seeds[0]],
                           log, save_ckpt=False)
        _, sm2 = run_cross(jax, oma, xj, yj, xo, yo, "smoke_jax_to_oma",
                           min(args.epochs, 1), args.lr, args.batch_size, [seeds[0]],
                           log, save_ckpt=False)
        log(f"smoke cross summaries: OMA->JAX={sm1['all']['model_mean_deg']:.2f}° "
            f"JAX->OMA={sm2['all']['model_mean_deg']:.2f}°")
        log("smoke done; no scientific conclusions.")
        log(f"log -> {log_path}")
        logf.close()
        return

    overfit_result = None

    if args.mode in ("overfit", "all"):
        log("positive control: overfit 10 OMA images (100 epochs)")
        overfit_result = run_overfit(oma, xo, yo, max(args.epochs, 100),
                                     args.lr, args.batch_size, seeds[0], log)
        if args.mode == "overfit":
            log(f"log -> {log_path}")
            logf.close()
            return
        if not overfit_result["pass"]:
            log("positive control FAILED; aborting main experiments.")
            log(f"log -> {log_path}")
            logf.close()
            raise SystemExit(2)

    loocv_summary = None
    if args.mode in ("loocv", "shuffle", "all"):
        with_shuffle = args.mode in ("shuffle", "all")
        log(f"OMA LOOCV start: folds={len(oma)} seeds={seeds} shuffle={with_shuffle}")
        rows, loocv_summary = run_loocv(oma, xo, yo, args.epochs, args.lr, args.batch_size,
                                        seeds, with_shuffle, log, fold_limit=args.fold_limit)
        write_csv(OUT_DIR / "oma_loocv_per_fold.csv", rows)
        write_csv(OUT_DIR / "oma_loocv_summary.csv", [loocv_summary])
        log("LOOCV summary: " + json.dumps(loocv_summary, ensure_ascii=False))

    jax_summary = None
    if args.mode in ("jax_aoi", "all"):
        log("JAX leave-one-AOI-out start")
        rows, jax_summary = run_jax_aoi(jax, xj, yj, args.epochs, args.lr, args.batch_size,
                                        seeds, args.mode == "all", log)
        write_csv(OUT_DIR / "jax_loao_per_fold.csv", rows)
        write_csv(OUT_DIR / "jax_loao_summary.csv", [jax_summary])
        log("JAX LOAO summary: " + json.dumps(jax_summary, ensure_ascii=False))

    cross_all = None
    if args.mode in ("cross", "all"):
        cross_all = {}
        log("cross-domain OMA -> JAX")
        rows1, sums1 = run_cross(oma, jax, xo, yo, xj, yj, "oma_to_jax",
                                 args.epochs, args.lr, args.batch_size, seeds, log,
                                 save_ckpt=args.save_ckpt)
        write_csv(OUT_DIR / "cross_oma_to_jax_per_image.csv", rows1)
        write_csv(OUT_DIR / "cross_oma_to_jax_summary.csv",
                  [sums1["all"], sums1["in"], sums1["out"]])
        write_csv(OUT_DIR / "cross_oma_to_jax_by_aoi.csv", sums1["by_aoi"])
        write_csv(OUT_DIR / "cross_oma_to_jax_by_date.csv", sums1["by_date"])
        cross_all["oma_to_jax"] = sums1
        log("cross-domain JAX -> OMA")
        rows2, sums2 = run_cross(jax, oma, xj, yj, xo, yo, "jax_to_oma",
                                 args.epochs, args.lr, args.batch_size, seeds, log,
                                 save_ckpt=args.save_ckpt)
        write_csv(OUT_DIR / "cross_jax_to_oma_per_image.csv", rows2)
        write_csv(OUT_DIR / "cross_jax_to_oma_summary.csv",
                  [sums2["all"], sums2["in"], sums2["out"]])
        write_csv(OUT_DIR / "cross_jax_to_oma_by_aoi.csv", sums2["by_aoi"])
        write_csv(OUT_DIR / "cross_jax_to_oma_by_date.csv", sums2["by_date"])
        cross_all["jax_to_oma"] = sums2

    if args.mode == "all":
        write_summary_md(loocv_summary, jax_summary, cross_all, overfit_result,
                         OUT_DIR / "SUMMARY.md")
        log(f"summary -> {OUT_DIR / 'SUMMARY.md'}")

    log(f"done. log -> {log_path}")
    logf.close()


if __name__ == "__main__":
    main()
