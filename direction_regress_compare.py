"""Paired comparison of direction-regression val angle errors (val-only, no test).

Reads results/sprint20261018/direction_regress_*_val_per_image.csv and reports
paired t (flip - candidate; positive = candidate better) at image and base level.
"""
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "results" / "sprint20261018"
TAGS = ["flip_ref", "imagenet_ref", "rot_s0", "rot_s1", "rot_s2"]


def base_name(name):
    m = re.match(r"^(.+?)_sub\d+", name)
    return m.group(1) if m else Path(name).stem


def load(tag):
    path = OUT / f"direction_regress_{tag}_val_per_image.csv"
    rows = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[r["name"]] = float(r["angle_err_deg"])
    return rows


def paired(a, b):
    names = sorted(set(a) & set(b))
    d = np.array([a[n] - b[n] for n in names], dtype=float)  # a=flip, b=cand
    if len(d) < 2:
        return {"mean_gain_deg": 0.0, "sd": 0.0, "t": 0.0,
                "win": 0, "lose": 0, "n": len(d)}
    sd = d.std(ddof=1)
    t = float(d.mean() / (sd / math.sqrt(len(d)))) if sd > 0 else 0.0
    return {"mean_gain_deg": float(d.mean()), "sd": float(sd), "t": t,
            "win": int((d > 0).sum()), "lose": int((d < 0).sum()), "n": len(d)}


def paired_base(a, b):
    ba, bb = {}, {}
    for n, v in a.items():
        ba.setdefault(base_name(n), []).append(v)
    for n, v in b.items():
        bb.setdefault(base_name(n), []).append(v)
    common = sorted(set(ba) & set(bb))
    d = np.array([np.mean(ba[k]) - np.mean(bb[k]) for k in common], dtype=float)
    if len(d) < 2:
        return {"mean_gain_deg": 0.0, "sd": 0.0, "t": 0.0,
                "win": 0, "lose": 0, "n": len(d)}
    sd = d.std(ddof=1)
    t = float(d.mean() / (sd / math.sqrt(len(d)))) if sd > 0 else 0.0
    return {"mean_gain_deg": float(d.mean()), "sd": float(sd), "t": t,
            "win": int((d > 0).sum()), "lose": int((d < 0).sum()), "n": len(d)}


def main():
    data = {t: load(t) for t in TAGS}
    names = sorted(set.intersection(*(set(v) for v in data.values())))
    rot_mean = {n: float(np.mean([data[t][n] for t in ["rot_s0", "rot_s1", "rot_s2"]]))
                for n in names}
    out = {"n_images": len(names), "val_tags": TAGS,
           "mean_deg": {t: float(np.mean(list(v.values()))) for t, v in data.items()}}
    out["mean_deg"]["rot_mean"] = float(np.mean(list(rot_mean.values())))
    for cand in ["imagenet_ref", "rot_s0", "rot_s1", "rot_s2"]:
        out[f"flip_minus_{cand}"] = {
            "image": paired(data["flip_ref"], data[cand]),
            "base": paired_base(data["flip_ref"], data[cand]),
        }
    out["flip_minus_rot_mean"] = {
        "image": paired(data["flip_ref"], rot_mean),
        "base": paired_base(data["flip_ref"], rot_mean),
    }
    out["imagenet_minus_rot_mean"] = {
        "image": paired(data["imagenet_ref"], rot_mean),
        "base": paired_base(data["imagenet_ref"], rot_mean),
    }
    with open(OUT / "direction_regress_compare.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
