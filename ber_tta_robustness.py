"""Val-only robustness for the 110-candidate BER selection (no test access).

Reads results/sprint20261018/ber_tta_val_per_image.csv and runs:
  1. leave-one-base-out selection (16 folds)
  2. bootstrap over the 16 base images (1000 draws)
  3. bootstrap BER distribution of the fixed chosen config

Outputs:
  results/sprint20261018/ber_tta_robustness.json
  results/sprint20261018/ber_tta_robustness_folds.csv
"""
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "results" / "sprint20261018"
BASELINE = "T30_none_no"
VAL_CRITERION = 0.047


def base_name(name):
    m = re.match(r"^(.+?)_sub\d+", name)
    return m.group(1) if m else Path(name).stem


def load():
    by_cfg = defaultdict(list)
    with open(OUT / "ber_tta_val_per_image.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by_cfg[r["config"]].append({
                "name": r["name"], "base": base_name(r["name"]),
                "IoU": float(r["IoU"]), "BER": float(r["BER"]),
            })
    return by_cfg


def paired_stats(base_rows, cand_rows):
    """Return (mean_gain, t) with d = base.BER - cand.BER (positive = cand better)."""
    b = {r["name"]: r for r in base_rows}
    d = np.array([b[r["name"]]["BER"] - r["BER"] for r in cand_rows
                  if r["name"] in b], dtype=float)
    if len(d) < 2:
        return 0.0, 0.0
    sd = d.std(ddof=1)
    t = float(d.mean() / (sd / math.sqrt(len(d)))) if sd > 0 else 0.0
    return float(d.mean()), t


def base_stats(base_rows, cand_rows):
    b = defaultdict(list)
    c = defaultdict(list)
    for r in base_rows:
        b[r["base"]].append(r["BER"])
    for r in cand_rows:
        c[r["base"]].append(r["BER"])
    common = sorted(set(b) & set(c))
    d = np.array([np.mean(b[k]) - np.mean(c[k]) for k in common], dtype=float)
    if len(d) < 2:
        return 0.0, 0.0
    sd = d.std(ddof=1)
    t = float(d.mean() / (sd / math.sqrt(len(d)))) if sd > 0 else 0.0
    return float(d.mean()), t


def select(by_cfg, bases):
    bases = set(bases)
    sub = {c: [r for r in rows if r["base"] in bases]
           for c, rows in by_cfg.items()}
    base_rows = sub[BASELINE]
    base_iou = float(np.mean([r["IoU"] for r in base_rows]))
    base_ber = float(np.mean([r["BER"] for r in base_rows]))
    cands = []
    for c, rows in sub.items():
        if not rows:
            continue
        mean_gain, t = paired_stats(base_rows, rows)
        bg, bt = base_stats(base_rows, rows)
        cands.append({
            "config": c,
            "BER": float(np.mean([r["BER"] for r in rows])),
            "IoU": float(np.mean([r["IoU"] for r in rows])),
            "mean_gain": mean_gain, "t": t,
            "base_gain": bg, "base_t": bt,
        })
    eligible = [x for x in cands
                if x["IoU"] >= base_iou - 0.01
                and x["mean_gain"] > 0 and abs(x["t"]) > 2.0
                and x["base_gain"] > 0 and x["base_t"] > 2.0]
    eligible.sort(key=lambda x: (x["BER"], -x["IoU"]))
    chosen = eligible[0] if eligible else next(
        x for x in cands if x["config"] == BASELINE)
    return chosen, base_ber, len(eligible)


def main():
    by_cfg = load()
    bases = sorted({r["base"] for r in by_cfg[BASELINE]})
    print(f"# configs={len(by_cfg)} bases={len(bases)}", flush=True)

    folds = []
    for b in bases:
        chosen, base_ber, n_elig = select(by_cfg, [x for x in bases if x != b])
        # held-out metrics on the excluded base (NOT the selection subset)
        ho_rows = [r for r in by_cfg[chosen["config"]] if r["base"] == b]
        ho_base = [r for r in by_cfg[BASELINE] if r["base"] == b]
        heldout_ber = float(np.mean([r["BER"] for r in ho_rows])) if ho_rows else float("nan")
        heldout_base_ber = float(np.mean([r["BER"] for r in ho_base])) if ho_base else float("nan")
        folds.append({
            "held_out": b, "chosen": chosen["config"],
            "selection_subset_BER": chosen["BER"],
            "selection_subset_IoU": chosen["IoU"],
            "selection_subset_below_criterion": chosen["BER"] < VAL_CRITERION,
            "heldout_BER": heldout_ber,
            "heldout_baseline_BER": heldout_base_ber,
            "heldout_gain": heldout_base_ber - heldout_ber,
            "eligible": n_elig,
        })
    chosen_all, base_ber_all, n_all = select(by_cfg, bases)

    # base-level subsampling: draw 12 of 16 bases without replacement.
    rng = np.random.default_rng(20261018)
    sub_size = max(2, len(bases) - 4)
    boot = []
    for _ in range(1000):
        draw = list(rng.choice(bases, size=sub_size, replace=False))
        uniq = sorted(set(draw))
        ch, _, _ = select(by_cfg, uniq)
        fixed_rows = [r for r in by_cfg["T25_none_tta"] if r["base"] in uniq]
        fixed_ber = float(np.mean([r["BER"] for r in fixed_rows])) if fixed_rows else float("nan")
        boot.append({"chosen": ch["config"], "fixed_BER": fixed_ber,
                     "chosen_BER": ch["BER"]})

    freq = defaultdict(int)
    for x in boot:
        freq[x["chosen"]] += 1
    fixed_bers = np.array([x["fixed_BER"] for x in boot], dtype=float)
    chosen_bers = np.array([x["chosen_BER"] for x in boot], dtype=float)
    out = {
        "n_configs": len(by_cfg),
        "n_bases": len(bases),
        "full_val_chosen": chosen_all["config"],
        "full_val_chosen_BER": chosen_all["BER"],
        "full_val_chosen_IoU": chosen_all["IoU"],
        "full_val_eligible": n_all,
        "val_criterion": VAL_CRITERION,
        "leave_one_base_out": {
            "n_folds": len(folds),
            "selection_subset_chosen_freq": {
                k: sum(1 for f in folds if f["chosen"] == k)
                for k in sorted({f["chosen"] for f in folds})},
            "selection_subset_ber_below_criterion": sum(
                1 for f in folds if f["selection_subset_below_criterion"]),
            "heldout_ber_below_criterion": sum(
                1 for f in folds if f["heldout_BER"] < VAL_CRITERION),
            "heldout_gain_positive": sum(
                1 for f in folds if f["heldout_gain"] > 0),
            "folds": folds,
        },
        "base_subsampling": {
            "n_draws": len(boot),
            "note": f"{sub_size}-of-{len(bases)} base draws without replacement; not a weighted bootstrap",
            "chosen_freq_top": sorted(freq.items(), key=lambda x: -x[1])[:10],
            "fixed_T25_none_tta_BER_mean": float(np.nanmean(fixed_bers)),
            "fixed_T25_none_tta_BER_p05": float(np.nanpercentile(fixed_bers, 5)),
            "fixed_T25_none_tta_BER_p95": float(np.nanpercentile(fixed_bers, 95)),
            "fixed_T25_none_tta_BER_below_criterion_frac": float(
                np.mean(fixed_bers < VAL_CRITERION)),
        },
    }
    with open(OUT / "ber_tta_robustness.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    with open(OUT / "ber_tta_robustness_folds.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(folds[0].keys()))
        w.writeheader()
        w.writerows(folds)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
