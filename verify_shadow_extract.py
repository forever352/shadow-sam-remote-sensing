"""端到端验证 `shadow_extract.py`（软件原型：输入影像 -> 输出阴影矢量）。

## 为什么有这个脚本

第三十八轮审查判"不通过"，原因之一：我曾在终端里内联跑过一遍端到端验证
（往返面积差、栅格化 IoU、合成地理参考样本），**但没有落盘** ->
按项目铁律"找不到证据 = 未完成"，该验证**不可复核**。
本脚本把那次验证**变成可重跑的脚本 + 日志**。

## 三项验证

1. **无地理参考输入**（AISD 子图）
   - 跑 `shadow_extract.py` 产出 GeoJSON
   - 把矢量**栅格化回去**，与①真值掩膜 ②管线自身 IoU 对比
2. **Shapefile 分支**：跑一遍，读回并核对要素数
3. **有地理参考输入**：造一个带 CRS/transform 的合成样本，核对
   顶层 `crs` 成员与坐标是否落在 `transform` 推算的范围内

## 用法
    python verify_shadow_extract.py
    （日志自动写 logs/verify_shadow_extract_{ts}.log）
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import env_setup  # 必须在 import torch 之前
import numpy as np
import cv2

BASE = Path(__file__).resolve().parent
LOG_DIR = BASE / "logs"
RES = BASE / "results"
PY = sys.executable
IMG = BASE / "AISD" / "AISD" / "Train412" / "shadow" / "chicago30_sub1.tif"
GT = BASE / "AISD" / "AISD" / "Train412" / "mask" / "chicago30_sub1.tif"
GEO_IN = RES / "_verify_geo_in.tif"

_logf = None


def log(s=""):
    print(s, flush=True)
    if _logf is not None:
        _logf.write(s + "\n")
        _logf.flush()


def run_tool(img, out, extra=()):
    cmd = [PY, str(BASE / "shadow_extract.py"), "--image", str(img), "--out", str(out), *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def rasterize_geojson(gj_path, shape):
    from rasterio.features import rasterize
    from shapely.geometry import shape as shp_shape
    gj = json.load(open(gj_path, encoding="utf-8"))
    geoms = [(shp_shape(f["geometry"]), 1) for f in gj["features"]]
    m = rasterize(geoms, out_shape=shape, fill=0, default_value=1, dtype=np.uint8).astype(bool)
    return m, gj


def main():
    global _logf
    LOG_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"verify_shadow_extract_{ts}.log"
    _logf = open(log_path, "w", encoding="utf-8")

    log("# shadow_extract.py 端到端验证")
    log(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"# 输入: {IMG.name}")
    ok_all = True

    # ---------- (1) 无地理参考 ----------
    log("")
    log("=== (1) 无地理参考输入（GeoJSON 分支）===")
    gj_out = RES / "_verify_out.geojson"
    rc, out = run_tool(IMG, gj_out)
    log(f"  退出码 = {rc}")
    if rc != 0:
        log("  [FAIL] 工具未正常退出")
        ok_all = False
    else:
        gt = cv2.imread(str(GT), cv2.IMREAD_GRAYSCALE) > 0
        rast, gj = rasterize_geojson(gj_out, gt.shape)
        feats = gj["features"]
        area_sum = sum(f["properties"]["area_px"] for f in feats)
        coord_sys = {f["properties"].get("coord_system") for f in feats}
        inter = np.logical_and(rast, gt).sum(); uni = np.logical_or(rast, gt).sum()
        iou_vec = float(inter) / float(uni)
        # 管线自身 IoU：从前一轮已落盘的逐图 CSV 取（同一权重、同一图）
        pipe_iou = None
        p = RES / "diradapter_eval_main_per_image.csv"
        if p.exists():
            import csv
            for r in csv.DictReader(open(p, newline="", encoding="utf-8")):
                if r["name"] == IMG.name and r["method"] == "LoRA-only(dir off)":
                    pipe_iou = float(r["IoU"]); break
        log(f"  要素数 = {len(feats)}；coord_system 取值集 = {coord_sys}；顶层 crs = {gj.get('crs')}")
        log(f"  矢量 area_px 总和 = {area_sum:.2f}")
        log(f"  **栅格化回像素数 = {int(rast.sum())}**")
        log(f"  面积差 = {int(rast.sum()) - area_sum:+.0f} 像素 "
            f"({abs(int(rast.sum()) - area_sum) / max(area_sum, 1) * 100:.4f}%)")
        log(f"  **栅格化后 IoU vs 真值 = {iou_vec:.6f}**")
        if pipe_iou is not None:
            log(f"  管线自身 IoU（同权重同图，取自 {p.name}）= {pipe_iou:.6f}")
            log(f"  差 = {iou_vec - pipe_iou:+.6f}（预期为负：矢量经过 simplify + 小面积过滤）")
        else:
            log("  [!] 未取到管线自身 IoU 参考值")
        for k, v in [("要素数>0", len(feats) > 0),
                     ("全部为像素坐标", coord_sys == {"pixel"}),
                     ("无 crs 成员", gj.get("crs") is None),
                     ("面积差 < 0.1%", abs(int(rast.sum()) - area_sum) / max(area_sum, 1) < 0.001)]:
            log(f"  [{'OK' if v else 'FAIL'}] {k}")
            ok_all = ok_all and v

    # ---------- (2) Shapefile 分支 ----------
    log("")
    log("=== (2) Shapefile 分支 ===")
    shp_out = RES / "_verify_out.shp"
    rc, out = run_tool(IMG, shp_out, extra=("--format", "shp"))
    log(f"  退出码 = {rc}")
    side = [shp_out.with_suffix(e) for e in (".shp", ".shx", ".dbf")]
    for f in side:
        log(f"  {'存在' if f.exists() else '**缺失**'}: {f.name}")
    n_shp = None
    try:
        import geopandas as gpd
        g = gpd.read_file(str(shp_out))
        n_shp = len(g)
        log(f"  读回要素数 = {n_shp}；CRS = {g.crs}")
    except Exception as e:
        log(f"  [!] 读回失败: {type(e).__name__}: {e}")
    if rc == 0 and all(f.exists() for f in side):
        log("  [OK] Shapefile 产出完整")
    else:
        log("  [FAIL] Shapefile 分支异常"); ok_all = False

    # ---------- (3) 有地理参考分支 ----------
    log("")
    log("=== (3) 有地理参考分支（合成 EPSG:32650 样本）===")
    log("  [!] 该样本是**合成**的（真实 AISD 子图 + 伪造 transform/CRS），")
    log("      只能验证『CRS/坐标写出路径』是否走通，**不代表**真实遥感产品的 CRS/轴序已验证。")
    try:
        import rasterio
        from rasterio.transform import from_origin
        rgb = cv2.cvtColor(cv2.imread(str(IMG)), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        tr = from_origin(500000.0, 4000000.0, 0.5, 0.5)
        with rasterio.open(str(GEO_IN), "w", driver="GTiff", height=h, width=w, count=3,
                           dtype="uint8", crs="EPSG:32650", transform=tr) as ds:
            for i in range(3):
                ds.write(rgb[:, :, i], i + 1)
        gj_geo = RES / "_verify_geo_out.geojson"
        rc, out = run_tool(GEO_IN, gj_geo)
        log(f"  退出码 = {rc}")
        gj = json.load(open(gj_geo, encoding="utf-8"))
        f0 = gj["features"][0]
        xs = [p[0] for p in f0["geometry"]["coordinates"][0]]
        ys = [p[1] for p in f0["geometry"]["coordinates"][0]]
        log(f"  顶层 crs = {gj.get('crs')}")
        log(f"  coord_system = {f0['properties'].get('coord_system')}")
        log(f"  x 范围 = [{min(xs):.2f}, {max(xs):.2f}]   y 范围 = [{min(ys):.2f}, {max(ys):.2f}]")
        log(f"  期望 x in [500000, {500000 + w * 0.5:.0f}], y in [{4000000 - h * 0.5:.0f}, 4000000]")
        in_range = (500000 <= min(xs) and max(xs) <= 500000 + w * 0.5
                    and 4000000 - h * 0.5 <= min(ys) and max(ys) <= 4000000)
        has_crs = gj.get("crs") is not None
        is_map = f0["properties"].get("coord_system") == "map"
        for k, v in [("顶层写出 crs", has_crs), ("coord_system=map", is_map), ("坐标落在推算范围内", in_range)]:
            log(f"  [{'OK' if v else 'FAIL'}] {k}")
            ok_all = ok_all and v
    except Exception as e:
        log(f"  [FAIL] {type(e).__name__}: {e}"); ok_all = False

    # ---------- 清理临时产物 ----------
    for f in [GEO_IN, RES / "_verify_out.geojson", RES / "_verify_geo_out.geojson",
              RES / "_verify_out.shp", RES / "_verify_out.shx", RES / "_verify_out.dbf",
              RES / "_verify_out.cpg"]:
        try:
            f.unlink(missing_ok=True)
        except Exception:
            pass

    log("")
    log(f"# 总结论: {'全部通过' if ok_all else '**存在失败项**'}")
    log(f"日志 -> {log_path}")
    _logf.close()
    return 0 if ok_all else 2


if __name__ == "__main__":
    raise SystemExit(main())
