"""5-day sprint F1: BER with flip TTA + lower thresholds + post-processing.

Pre-registered in logs/PREREG_5day_sprint_20261018.md (v2).
- All selection on val only (110 candidates: 11 thresholds x 5 posts x 2 TTA).
- Selection rule mirrors ber_sweep_eval.py: val IoU floor + paired |t|>2 +
  base-name aggregated paired t>2; then min val BER.
- Test is technically locked: requires TEST_SIGNOFF.json, selection hash match,
  92-name assertion, and refuses if test_evaluated is already true.
- Test is evaluated at most once, and only if val BER < 0.047.

Usage:
    python ber_tta_eval.py --smoke 3          # quick code check on 3 val images
    python ber_tta_eval.py --stage val        # full val grid + selection JSON
    python ber_tta_eval.py --stage test       # run test once for the selected config
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import env_setup  # noqa: F401  (must be imported before torch)

import cv2
import numpy as np
import torch
from scipy import ndimage

import ber_sweep_eval
from eval_shadow import compute_metrics
from ber_sweep_eval import (
    SEED_CKPTS, SEEDS, DEVICE, AISD, RES_DIR, LOG_DIR,
    load_sam, image_prob_maps, build_split, paired_t, paired_t_base,
)

OUT_DIR = RES_DIR / "sprint20261018"
SLOG_DIR = LOG_DIR / "sprint20261018"
THRESHOLDS = [round(0.10 + 0.05 * i, 2) for i in range(11)]
POSTS = ["none", "close3", "close5", "fill", "close3_fill"]
METRIC_KEYS = ["IoU", "BER", "F1", "OA", "Precision", "Recall"]
KEYS2 = METRIC_KEYS + ["FPR", "FNR"]
VAL_CRITERION = 0.047
TEST_CRITERION = 0.05
BASELINE_CODE = "T30_none_no"
SIGNOFF = SLOG_DIR / "TEST_SIGNOFF.json"
LEDGER = OUT_DIR / "TEST_LEDGER.csv"


def cfg_code(t, post, tta):
    return f"T{int(round(t * 100)):02d}_{post}_{'tta' if tta else 'no'}"


def parse_code(code):
    parts = code.split("_")
    t = int(parts[0][1:]) / 100.0
    tta = parts[-1] == "tta"
    post = "_".join(parts[1:-1])
    return t, post, tta


def postprocess(binary, kind):
    b = binary.astype(np.uint8)
    if kind == "none":
        return b
    if kind == "close3":
        return cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    if kind == "close5":
        return cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    if kind == "fill":
        return ndimage.binary_fill_holes(b).astype(np.uint8)
    if kind == "close3_fill":
        c = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        return ndimage.binary_fill_holes(c).astype(np.uint8)
    raise ValueError(kind)


def metrics_for(prob, gt, t, post):
    pred = (prob > t).astype(np.uint8)
    pred = postprocess(pred, post)
    m = compute_metrics((pred * 255).astype(np.uint8), gt)
    m["FNR"] = 1.0 - m["Recall"]
    m["FPR"] = 2.0 * m["BER"] - m["FNR"]
    return m


def prob_maps_tta(sams, img_bgr, seeds):
    """Return (pm_no, pm_tta), both {seed: prob map}."""
    pm_no = image_prob_maps(sams, img_bgr, seeds)
    pm_tta = {}
    flipped = img_bgr[:, ::-1].copy()
    pm_f = image_prob_maps(sams, flipped, seeds)
    for s in seeds:
        pm_tta[s] = 0.5 * (pm_no[s] + pm_f[s][:, ::-1])
    return pm_no, pm_tta


def ensemble(pm, seeds):
    return sum(pm[s] for s in seeds) / float(len(seeds))


def eval_files(sams, files, seeds, limit=0, only=None):
    """Return {code: [per-image rows]} for all configs, or only one code."""
    codes = [cfg_code(t, p, tta) for tta in (False, True)
             for t in THRESHOLDS for p in POSTS]
    if only is not None:
        codes = [only]
    out = {c: [] for c in codes}
    only_t, only_post, only_tta = parse_code(only) if only else (None, None, None)
    use = files[:limit] if limit else files
    for i, f in enumerate(use, 1):
        img = cv2.imread(f)
        if img is None:
            continue
        gt_path = AISD / "Train412" / "mask" / Path(f).name
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        pm_no, pm_tta = prob_maps_tta(sams, img, seeds)
        probs = {False: ensemble(pm_no, seeds), True: ensemble(pm_tta, seeds)}
        if only is not None:
            m = metrics_for(probs[only_tta], gt, only_t, only_post)
            m["name"] = Path(f).name
            out[only].append(m)
        else:
            for tta in (False, True):
                for t in THRESHOLDS:
                    for p in POSTS:
                        m = metrics_for(probs[tta], gt, t, p)
                        m["name"] = Path(f).name
                        out[cfg_code(t, p, tta)].append(m)
        if i % 10 == 0 or i == len(use):
            print(f"  [{i}/{len(use)}] {Path(f).name}", flush=True)
    return out


def agg(rows, keys=KEYS2):
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def read_per_image(path, config):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["config"] != config:
                continue
            row = {k: float(r[k]) for k in METRIC_KEYS}
            row["FNR"] = 1.0 - row["Recall"]
            row["FPR"] = 2.0 * row["BER"] - row["FNR"]
            row["name"] = r["name"]
            rows.append(row)
    return rows


def read_names(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return {r["name"] for r in csv.DictReader(fh) if r.get("name")}


def write_csv(path, rows, keys):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def sel_hash(sel):
    body = {k: v for k, v in sel.items() if k != "selection_sha256"}
    raw = json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def append_ledger(row):
    new = not LEDGER.exists()
    with open(LEDGER, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "date", "family", "config", "selection_basis", "result",
            "used_for_selection", "selection_sha256"])
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["val", "test"], default="val")
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SLOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = SLOG_DIR / f"ber_tta_{args.stage}_{ts}.log"
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    tr_files, val_files, test_files, (n_tr, n_va, n_te) = build_split()
    log("# BER TTA sprint (pre-registered v2)")
    log(f"# time={time.strftime('%Y-%m-%d %H:%M:%S')} device={DEVICE}")
    log(f"# split: train={n_tr} val={n_va} test={n_te} base images")
    log(f"# val criterion: BER < {VAL_CRITERION}; test criterion: BER < {TEST_CRITERION} + paired gain>0")
    log(f"# thresholds={THRESHOLDS}")
    log(f"# posts={POSTS}")

    log("# loading 3 seed checkpoints")
    sams = {s: load_sam(SEED_CKPTS[s]) for s in SEEDS}
    log(f"# allocated={torch.cuda.memory_allocated()/1e6:.1f} MB" if DEVICE == "cuda" else "# CPU")

    if args.stage == "val":
        limit = args.smoke if args.smoke else 0
        rows = eval_files(sams, val_files, SEEDS, limit=limit)
        if args.smoke:
            log(f"# smoke rows per config: {len(rows[BASELINE_CODE])}")
            for code in [cfg_code(0.30, "none", False), cfg_code(0.30, "none", True),
                         cfg_code(0.20, "close3_fill", True)]:
                if rows.get(code):
                    log(f"{code}: {agg(rows[code])}")
            log(f"# log -> {log_path}")
            return 0

        per_rows = []
        for code, rr in rows.items():
            for r in rr:
                per_rows.append({"config": code,
                                 **{k: r[k] for k in KEYS2},
                                 "name": r["name"]})
        write_csv(OUT_DIR / "ber_tta_val_per_image.csv", per_rows,
                  ["config"] + KEYS2 + ["name"])

        base_rows = rows[BASELINE_CODE]
        base = agg(base_rows)
        grid = []
        for code, rr in rows.items():
            if not rr:
                continue
            m, tt, win, lose = paired_t(base_rows, rr)
            mb = paired_t_base(base_rows, rr)
            grid.append({"config": code, **agg(rr), "n": len(rr),
                         "paired_mean_gain": m, "paired_t": tt,
                         "win": win, "lose": lose, **mb})
        grid.sort(key=lambda x: x["BER"])
        write_csv(OUT_DIR / "ber_tta_val_grid.csv", grid,
                  ["config"] + KEYS2 + ["n", "paired_mean_gain", "paired_t",
                                        "win", "lose", "base_paired_mean_gain",
                                        "base_paired_t", "base_win", "base_lose",
                                        "base_n"])

        log(f"# baseline {BASELINE_CODE}: BER={base['BER']:.6f} IoU={base['IoU']:.6f}")
        log("# known B30 val: BER=0.051175 IoU=0.814307 (ber_sweep_selection.json)")
        gate_dev = max(abs(base["BER"] - 0.051175), abs(base["IoU"] - 0.814307))
        log(f"# gate deviation={gate_dev:.3e}")
        if gate_dev > 1e-6:
            log("# [FAIL] baseline does not reproduce; stop")
            log(f"# log -> {log_path}")
            return 2

        eligible = [g for g in grid
                    if g["IoU"] >= base["IoU"] - 0.01
                    and g["paired_mean_gain"] > 0 and abs(g["paired_t"]) > 2.0
                    and g["base_paired_mean_gain"] > 0
                    and g["base_paired_t"] > 2.0]
        eligible.sort(key=lambda x: (x["BER"], -x["IoU"]))
        if eligible:
            chosen = eligible[0]
        else:
            chosen = next(g for g in grid if g["config"] == BASELINE_CODE)
        met = bool(chosen["BER"] < VAL_CRITERION) and chosen["config"] != BASELINE_CODE
        boundary = chosen["config"] in {cfg_code(0.10, p, tta) for p in POSTS for tta in (False, True)} \
            or chosen["config"] in {cfg_code(0.60, p, tta) for p in POSTS for tta in (False, True)}
        sel = {
            "ts": ts,
            "version": 2,
            "baseline": BASELINE_CODE,
            "baseline_val": {k: base[k] for k in KEYS2 + ["IoU"]},
            "known_b30_val": {"BER": 0.051175, "IoU": 0.814307},
            "gate_deviation": gate_dev,
            "thresholds": THRESHOLDS,
            "posts": POSTS,
            "n_candidates": len(grid),
            "selection_rule": ("val IoU >= baseline-0.01; paired |t|>2; "
                               "base-name paired t>2; then min val BER; tie-break higher IoU"),
            "multiple_comparison": "110 candidates, no correction; exploratory selection",
            "chosen": chosen["config"],
            "chosen_val": {k: chosen[k] for k in KEYS2 + ["IoU"]},
            "chosen_stats": {k: chosen[k] for k in
                             ["paired_mean_gain", "paired_t", "win", "lose",
                              "base_paired_mean_gain", "base_paired_t",
                              "base_win", "base_lose", "base_n"]},
            "threshold_at_grid_boundary": bool(boundary),
            "success_criterion_met": met,
            "val_criterion": VAL_CRITERION,
            "test_criterion": TEST_CRITERION,
            "test_evaluated": False,
        }
        sel["selection_sha256"] = sel_hash(sel)
        atomic_json(OUT_DIR / "ber_tta_selection.json", sel)
        log(f"# chosen={chosen['config']} val_BER={chosen['BER']:.6f} val_IoU={chosen['IoU']:.6f}")
        log(f"# boundary={boundary} success_criterion_met={met}")
        log(f"# selection_sha256={sel['selection_sha256']}")
        log("# test NOT evaluated in this stage")
        log(f"# log -> {log_path}")
        return 0

    # stage == test
    sel_path = OUT_DIR / "ber_tta_selection.json"
    with open(sel_path, encoding="utf-8") as fh:
        sel = json.load(fh)
    if sel.get("test_evaluated"):
        log("# [REFUSE] test_evaluated is already true; test must not be repeated.")
        log(f"# log -> {log_path}")
        return 4
    if sel.get("selection_sha256") != sel_hash(sel):
        log("# [REFUSE] selection JSON hash mismatch; file was edited.")
        log(f"# log -> {log_path}")
        return 5
    if not sel.get("success_criterion_met"):
        log("# [STOP] val criterion not met; test must not be evaluated.")
        log(f"# log -> {log_path}")
        return 3
    if not SIGNOFF.exists():
        log("# [REFUSE] missing TEST_SIGNOFF.json (supervisor pre-signoff required).")
        log(f"# log -> {log_path}")
        return 6
    with open(SIGNOFF, encoding="utf-8") as fh:
        signoff = json.load(fh)
    script_sha = {
        "ber_tta_eval.py": file_sha(Path(__file__)),
        "ber_sweep_eval.py": file_sha(Path(ber_sweep_eval.__file__)),
    }
    cmd = " ".join([Path(sys.argv[0]).name] + sys.argv[1:])
    ckpts = {str(k): v for k, v in SEED_CKPTS.items()}
    if (not signoff.get("approved")
            or signoff.get("selection_sha256") != sel["selection_sha256"]):
        log("# [REFUSE] TEST_SIGNOFF.json not approved or hash mismatch.")
        log(f"# log -> {log_path}")
        return 7
    if signoff.get("command") != cmd:
        log(f"# [REFUSE] signoff command mismatch: {signoff.get('command')} != {cmd}")
        return 7
    if signoff.get("checkpoints") != ckpts:
        log("# [REFUSE] signoff checkpoint paths mismatch")
        return 7
    if signoff.get("script_sha256") != script_sha:
        log("# [REFUSE] signoff script hash mismatch")
        return 7
    log(f"# command={cmd}")
    log(f"# checkpoints={ckpts}")
    log(f"# script_sha256={script_sha}")

    # 92-name assertion vs main result
    ref_names = read_names(RES_DIR / "lora_leakfree_test_per_image.csv")
    test_names = {Path(f).name for f in test_files}
    if len(test_files) != 92 or len(test_names) != 92:
        log(f"# [REFUSE] test split size {len(test_files)} != 92")
        return 8
    if ref_names and ref_names != test_names:
        log("# [REFUSE] test name set differs from main result CSV")
        return 8

    code = sel["chosen"]
    t, post, tta = parse_code(code)
    log(f"# chosen={code} threshold={t} tta={tta} post={post}")
    log(f"# selection_sha256={sel['selection_sha256']} signoff_by={signoff.get('approved_by')}")
    rows = eval_files(sams, test_files, SEEDS, only=code)
    chosen_rows = rows[code]
    a = agg(chosen_rows)
    b30_rows = read_per_image(RES_DIR / "ber_sweep_test_per_image.csv", "B30")
    m, tt, win, lose = paired_t(b30_rows, chosen_rows)
    mb = paired_t_base(b30_rows, chosen_rows)
    ok = bool(a["BER"] < TEST_CRITERION and m > 0
              and mb["base_paired_mean_gain"] > 0)
    log(f"# test {code}: BER={a['BER']:.6f} IoU={a['IoU']:.6f}")
    log(f"# vs B30: mean_BER_gain={m:.6f} t={tt:.3f} win={win} lose={lose}")
    log(f"# scene-level: gain={mb['base_paired_mean_gain']:.6f} t={mb['base_paired_t']:.3f} "
        f"win={mb['base_win']} lose={mb['base_lose']} n={mb['base_n']}")
    log(f"# test criterion met={ok}")
    summary = [{"config": "B30", "reference": True, **agg(b30_rows)},
               {"config": code, "reference": False, **a}]
    write_csv(OUT_DIR / "ber_tta_test.csv", summary, ["config", "reference"] + KEYS2)
    per = [{"config": code, **{k: r[k] for k in KEYS2}, "name": r["name"]}
           for r in chosen_rows]
    write_csv(OUT_DIR / "ber_tta_test_per_image.csv", per,
              ["config"] + KEYS2 + ["name"])
    sel["test_evaluated"] = True
    sel["pretest_selection_sha256"] = signoff["selection_sha256"]
    sel["test_result"] = a
    sel["test_vs_b30"] = {"mean_gain": m, "t": tt, "win": win, "lose": lose,
                          **mb, "criterion_met": ok}
    sel["test_log"] = str(log_path)
    sel["test_command"] = cmd
    sel["test_checkpoints"] = ckpts
    sel["test_script_sha256"] = script_sha
    per_sha = file_sha(OUT_DIR / "ber_tta_test_per_image.csv")
    sel["test_per_image_sha256"] = per_sha
    sel["test_sha256"] = hashlib.sha256(
        json.dumps({"result": a, "vs_b30": sel["test_vs_b30"],
                    "per_image_sha256": per_sha,
                    "command": cmd, "checkpoints": ckpts,
                    "script_sha256": script_sha},
                   sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    sel["selection_sha256"] = sel_hash(sel)
    atomic_json(sel_path, sel)
    append_ledger({
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "family": "F1_H1",
        "config": code,
        "selection_basis": "val BER + paired t + base-name t (see selection JSON)",
        "result": f"BER={a['BER']:.6f};IoU={a['IoU']:.6f};vsB30_gain={m:.6f};t={tt:.3f}",
        "used_for_selection": "yes",
        "selection_sha256": sel["pretest_selection_sha256"],
    })
    log(f"# log -> {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
