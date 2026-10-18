"""CNN/LoRA 多 seed 汇总、配对统计与场景级 CI（3-seed mean 口径）。

口径
----
- 每个方法先算每个 seed 的逐图指标；主表 headline = 各 seed 均值的再平均
  （等价于逐图 3-seed 均值的再平均）；`_seed_sd` = 3 个 seed 均值的样本 sd（ddof=1）。
- 配对比较：先对每个图取 3 seed 均值，再配对；场景级再按原图基名聚合。
- 95%CI 用 bootstrap（20000 次，固定 seed=0）；p 用正态近似（与 scene_ci.py 一致）。
- **不覆盖** `results/scene_level_ci.csv`（seed-0 敏感性）。

输出
----
results/cnn_multiseed_summary.csv
results/cnn_multiseed_paired.csv
results/scene_level_ci_multiseed.csv
logs/multiseed_summary_{ts}.log

用法
----
python multiseed_summary.py
"""
import csv
import hashlib
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
R = BASE / "results"
LOG = BASE / "logs"
CKPT = BASE / "checkpoints"
METRICS = ["IoU", "BER", "F1", "OA", "Precision", "Recall"]

METHODS = {
    "LoRA-SAM": ["lora_leakfree_test_per_image.csv",
                 "lora_leakfree_s1_test_per_image.csv",
                 "lora_leakfree_s2_test_per_image.csv"],
    "U-Net": ["unet_leakfree_test_per_image.csv",
              "unet_leakfree_s1_test_per_image.csv",
              "unet_leakfree_s2_test_per_image.csv"],
    "SegNet": ["segnet_leakfree_test_per_image.csv",
               "segnet_leakfree_s1_test_per_image.csv",
               "segnet_leakfree_s2_test_per_image.csv"],
    "Mask R-CNN": ["maskrcnn_leakfree_res800_test_per_image.csv",
                   "maskrcnn_leakfree_s1_test_per_image.csv",
                   "maskrcnn_leakfree_s2_test_per_image.csv"],
}
BASELINES = ["U-Net", "SegNet", "Mask R-CNN"]
NOTE = {
    "U-Net": "positive=baseline higher; BER positive=baseline worse; no multiple-comparison correction",
    "SegNet": "positive=baseline higher; BER positive=baseline worse; no multiple-comparison correction",
    "Mask R-CNN": "positive=baseline higher; BER positive=baseline worse; no multiple-comparison correction; "
                  "resolution asymmetry remains (input 512 upsampled to 800; main method 1024)",
}
SEED_LABEL = {0: "seed0", 1: "seed1", 2: "seed2"}


def base_of(name):
    m = re.match(r"^(.*?)_sub\d+", name)
    return m.group(1) if m else Path(name).stem


def load(fn):
    with open(R / fn, newline="", encoding="utf-8-sig") as f:
        return {r["name"]: {k: float(r[k]) for k in METRICS} for r in csv.DictReader(f)}


def boot_ci(x, n_boot=20000, alpha=0.05, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.RandomState(seed)
    means = np.array([x[rng.randint(0, len(x), len(x))].mean() for _ in range(n_boot)])
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def p_normal(t):
    return 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))


def paired(a, b, key, level):
    """a/b: {name: {metric: val}}；level=image|scene。返回统计。"""
    def agg(src):
        if level == "image":
            return {n: v[key] for n, v in src.items()}
        g = defaultdict(list)
        for n, v in src.items():
            g[base_of(n)].append(v[key])
        return {k: float(np.mean(v)) for k, v in g.items()}
    A, B = agg(a), agg(b)
    ks = sorted(set(A) & set(B))
    d = np.array([A[k] - B[k] for k in ks], dtype=float)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    t = float(mean / (sd / math.sqrt(n))) if sd > 0 else 0.0
    lo, hi = boot_ci(d, seed=0)
    return {
        "level": level, "n": n, "mean_diff": mean, "paired_sd": sd,
        "ci_lo": lo, "ci_hi": hi, "t": t, "p": p_normal(t),
        "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
    }


def main():
    LOG.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG / f"multiseed_summary_{ts}.log"
    logf = open(log_path, "w", encoding="utf-8")

    def log(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log("# 多 seed 汇总（3-seed mean 口径）")
    log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    per_seed = {}
    per_mean = {}
    names_ref = None
    for method, fns in METHODS.items():
        ds = [load(fn) for fn in fns]
        ns = [set(d) for d in ds]
        names = sorted(set.intersection(*ns))
        if names_ref is None:
            names_ref = set(names)
        if set(names) != names_ref:
            raise RuntimeError(f"{method} 的 test 名称集合与其他方法不一致")
        per_seed[method] = ds
        per_mean[method] = {n: {k: float(np.mean([d[n][k] for d in ds])) for k in METRICS}
                            for n in names}
        seed_ious = [float(np.mean([x["IoU"] for x in d.values()])) for d in ds]
        log(f"{method}: {len(ds)} seeds, {len(names)} images, "
            f"seed IoU = {[round(v, 6) for v in seed_ious]}")

    names_sorted = sorted(names_ref)
    name_hash = hashlib.sha256("\n".join(names_sorted).encode("utf-8")).hexdigest()[:16]
    n_scenes = len({base_of(n) for n in names_sorted})
    log(f"test 名称 sha256[:16]: {name_hash}；场景数: {n_scenes}")

    # 1) summary
    out1 = R / "cnn_multiseed_summary.csv"
    fields = ["method", "n_seeds", "n_images", "n_scenes",
              "IoU_mean", "IoU_seed_sd", "BER_mean", "BER_seed_sd",
              "F1_mean", "F1_seed_sd", "OA_mean", "OA_seed_sd",
              "Precision_mean", "Precision_seed_sd", "Recall_mean", "Recall_seed_sd",
              "IoU_seed0", "IoU_seed1", "IoU_seed2", "BER_seed0", "BER_seed1", "BER_seed2",
              "test_names_sha256"]
    with open(out1, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for method in METHODS:
            ds = per_seed[method]
            row = {"method": method, "n_seeds": len(ds), "n_images": len(names_sorted),
                   "n_scenes": n_scenes, "test_names_sha256": name_hash}
            for k in METRICS:
                vals = [float(np.mean([x[k] for x in d.values()])) for d in ds]
                row[f"{k}_mean"] = f"{np.mean(vals):.6f}"
                row[f"{k}_seed_sd"] = f"{np.std(vals, ddof=1):.6f}"
            for i, key in SEED_LABEL.items():
                row[f"IoU_{key}"] = f"{np.mean([x['IoU'] for x in ds[i].values()]):.6f}"
                row[f"BER_{key}"] = f"{np.mean([x['BER'] for x in ds[i].values()]):.6f}"
            w.writerow(row)
            log(f"summary {method}: IoU {row['IoU_mean']} +/- {row['IoU_seed_sd']}  "
                f"BER {row['BER_mean']} +/- {row['BER_seed_sd']}")

    # 1b) Mask R-CNN seed metadata：tau / clean_run / protocol
    def read_tau(fn):
        with open(R / fn, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                return float(r["tau"])
        return float("nan")

    def read_meta(path):
        if path is None or not Path(path).exists():
            return None
        return json.loads(Path(path).read_text(encoding="utf-8"))

    mask_meta = [
        (0, "maskrcnn_leakfree_res800_test.csv", None,
         "seed-0 meta absent; clean_run unverified. Log 20261015_190926 shows 60/60, no resume. "
         "Its '[冒烟产物]' wording is an old write_results() bug for non-empty tag, not a real smoke run "
         "(review_log 第十三/十四轮); numbers valid."),
        (1, "maskrcnn_leakfree_s1_test.csv",
         CKPT / "maskrcnn_shadow_leakfree_s1_meta.json", "clean 60 epochs, no resume"),
        (2, "maskrcnn_leakfree_s2_test.csv",
         CKPT / "maskrcnn_shadow_leakfree_s2_meta.json", "clean 60 epochs, no resume"),
    ]
    out_meta = R / "maskrcnn_multiseed_meta.csv"
    with open(out_meta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "clean_run", "resumed",
                                          "min_size", "max_size", "tau", "source", "note"])
        w.writeheader()
        for seed, fn, meta_path, note in mask_meta:
            meta = read_meta(meta_path)
            if meta is None:
                row = {"seed": seed, "clean_run": "unknown", "resumed": "unknown",
                       "min_size": 800, "max_size": 1333, "tau": read_tau(fn),
                       "source": fn, "note": note}
            else:
                row = {"seed": seed, "clean_run": bool(meta.get("clean_run")),
                       "resumed": bool(meta.get("resumed")),
                       "min_size": meta.get("min_size"), "max_size": meta.get("max_size"),
                       "tau": read_tau(fn), "source": fn, "note": note}
            w.writerow(row)
            log(f"mask-meta seed={seed} clean_run={row['clean_run']} "
                f"resumed={row['resumed']} min/max={row['min_size']}/{row['max_size']} "
                f"tau={row['tau']}")

    # 2) paired
    out2 = R / "cnn_multiseed_paired.csv"
    pfields = ["comparison", "level", "metric", "n", "mean_diff", "paired_sd",
               "ci_lo", "ci_hi", "t", "p", "wins", "losses", "pairing", "note"]
    with open(out2, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pfields)
        w.writeheader()
        for baseline in BASELINES:
            for metric in ["IoU", "BER"]:
                for level in ["image", "scene"]:
                    st = paired(per_mean[baseline], per_mean["LoRA-SAM"], metric, level)
                    row = {"comparison": f"{baseline} vs LoRA-SAM", "level": level, "metric": metric,
                           "n": st["n"], "mean_diff": f"{st['mean_diff']:+.6f}",
                           "paired_sd": f"{st['paired_sd']:.6f}",
                           "ci_lo": f"{st['ci_lo']:+.6f}", "ci_hi": f"{st['ci_hi']:+.6f}",
                           "t": f"{st['t']:+.3f}", "p": f"{st['p']:.4g}",
                           "wins": st["wins"], "losses": st["losses"],
                           "pairing": "3-seed mean per image",
                           "note": NOTE[baseline]}
                    w.writerow(row)
                    log(f"paired {baseline} vs LoRA {level} {metric}: "
                        f"mean={st['mean_diff']:+.6f} t={st['t']:+.3f} wins/losses={st['wins']}/{st['losses']}")

    # 3) scene-level CI (multiseed), do not overwrite scene_level_ci.csv
    out3 = R / "scene_level_ci_multiseed.csv"
    sfields = ["level", "method_or_comparison", "metric", "n",
               "mean", "std", "mean_diff", "paired_sd", "ci_lo", "ci_hi", "t", "p",
               "wins", "losses", "pairing"]
    with open(out3, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sfields)
        w.writeheader()
        for method in METHODS:
            for level in ["image", "scene"]:
                if level == "image":
                    vals = {n: v for n, v in per_mean[method].items()}
                else:
                    g = defaultdict(list)
                    for n, v in per_mean[method].items():
                        g[base_of(n)].append(v)
                    vals = {k: {m: float(np.mean([x[m] for x in v])) for m in METRICS}
                            for k, v in g.items()}
                for metric in ["IoU", "BER"]:
                    x = np.array([v[metric] for v in vals.values()], dtype=float)
                    w.writerow({"level": level, "method_or_comparison": method, "metric": metric,
                                "n": len(x), "mean": f"{x.mean():.6f}", "std": f"{x.std(ddof=1):.6f}",
                                "mean_diff": "", "paired_sd": "", "ci_lo": "", "ci_hi": "",
                                "t": "", "p": "", "wins": "", "losses": "",
                                "pairing": "3-seed mean per image"})
        for baseline in BASELINES:
            for metric in ["IoU", "BER"]:
                for level in ["image", "scene"]:
                    st = paired(per_mean[baseline], per_mean["LoRA-SAM"], metric, level)
                    w.writerow({"level": f"paired_{level}",
                                "method_or_comparison": f"{baseline} vs LoRA-SAM",
                                "metric": metric, "n": st["n"], "mean": "", "std": "",
                                "mean_diff": f"{st['mean_diff']:+.6f}",
                                "paired_sd": f"{st['paired_sd']:.6f}",
                                "ci_lo": f"{st['ci_lo']:+.6f}", "ci_hi": f"{st['ci_hi']:+.6f}",
                                "t": f"{st['t']:+.3f}", "p": f"{st['p']:.4g}",
                                "wins": st["wins"], "losses": st["losses"],
                                "pairing": "3-seed mean per image"})

    log(f"汇总 -> {out1}")
    log(f"Mask R-CNN meta -> {out_meta}")
    log(f"配对 -> {out2}")
    log(f"场景CI(3-seed) -> {out3}")
    log(f"日志 -> {log_path}")
    logf.close()


if __name__ == "__main__":
    main()
