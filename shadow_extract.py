"""遥感影像阴影提取工具：**输入影像 -> 输出阴影矢量**（软件原型）。

申报书「研究计划」「预期研究成果」要求交付一个"输入影像即输出阴影矢量"的工具
（Python 工具包 / QGIS 插件）。本脚本是其中的**命令行版**。

## 用法

    # 默认：无泄漏口径的 LoRA 权重，输出 GeoJSON
    python shadow_extract.py --image 你的影像.tif --out shadows.geojson

    # 输出 shapefile（需要 geopandas + pyogrio）
    python shadow_extract.py --image 你的影像.tif --out shadows.shp --format shp

    # 同时输出绿色叠加预览图
    python shadow_extract.py --image 你的影像.tif --out shadows.geojson --overlay preview.png

    # 指定权重（例如自带划分口径的旧权重）
    python shadow_extract.py --image x.tif --checkpoint checkpoints/sam_vit_b_shadow_v4_lora_last.pth

## 管线（与项目主方法一致）

    增强物理粗提取（CLAHE + Otsu + 形态学） -> 连通域框提示 -> LoRA 微调 SAM 分割
    -> 二值掩膜 -> 轮廓矢量化（含孔洞） -> 简化 -> 按面积过滤 -> 写矢量文件

## 坐标

- 若输入是**带地理参考的栅格**（rasterio 能读出 CRS + transform），输出的几何坐标
  为**该栅格坐标系下的地图坐标**，并在 GeoJSON 顶层写入 `crs` 成员
  （注：GeoJSON RFC 7946 规定坐标为 WGS84，`crs` 成员是 2008 旧规范的用法，
  此处显式写出以免误用；如需严格 WGS84 请自行重投影）。
- 若输入**无地理参考**（如 PNG/JPG），则输出**像素坐标**（x=列, y=行），
  并在要素属性里标 `"coord_system": "pixel"`。

## 与 demo.py 的区别

- `demo.py`：单图演示，输出**绿色叠加 PNG**，不产出矢量。
- 本脚本：**产出矢量**（GeoJSON / Shapefile），是申报书要求的"输入影像 -> 输出阴影矢量"。

## 重要：默认权重

默认使用 **无泄漏划分** 训练的 LoRA 权重 `sam_vit_b_shadow_v4_lora_leakfree.pth`
（对应主结果口径）。**不要**默认使用 `sam_vit_b_shadow_v4_lora_last.pth`
—— 那是 AISD 自带划分（**含同源泄漏**）口径的权重，其数字系统性偏高，仅作历史对齐。
"""
import argparse
import json
import time
from pathlib import Path

# noinspection PyUnusedImports
import env_setup  # noqa: F401  必须在 import torch 之前；仅用于 DLL/OpenMP 副作用

import torch
import cv2
import numpy as np

from segment_anything import sam_model_registry
from prompt_experiment import component_boxes, predict_box
from methods import mask_enhanced

BASE = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OFFICIAL_CKPT = str(BASE / "checkpoints" / "sam_vit_b_01ec64.pth")

# 默认 = 无泄漏口径主结果权重（**不是** 含泄漏的 _last）
DEFAULT_CKPT = str(BASE / "checkpoints" / "sam_vit_b_shadow_v4_lora_leakfree.pth")


def load_sam(ckpt_path):
    """加载 SAM；文件名含 lora 时先挂 LoRA 结构再载权重。"""
    sam = sam_model_registry["vit_b"](checkpoint=OFFICIAL_CKPT)
    if "lora" in Path(ckpt_path).name.lower():
        from train_adapter_v4 import apply_lora
        apply_lora(sam)
    sam.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    sam.to(DEVICE)
    sam.eval()
    return sam


@torch.no_grad()
def predict_mask(sam, img_bgr, min_area):
    """增强粗提取 -> 框提示 -> 微调 SAM -> 布尔掩膜。"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    em = mask_enhanced(img_bgr)
    boxes = component_boxes(em, min_area=min_area)
    pred = np.zeros(img_bgr.shape[:2], bool)
    for b in boxes:
        pred |= predict_box(sam, img_rgb, b)
    return pred, len(boxes)


def mask_to_polygons(mask, simplify_px, min_poly_area):
    """布尔掩膜 -> shapely Polygon 列表（孔洞正确、几何精确）。

    用 `rasterio.features.shapes`（GDAL polygonize），坐标为**像素角点**约定
    （每个像素是一个单位正方形），因此面积是精确的。

    **为什么不用 cv2.findContours**：实测该函数返回的**孔洞轮廓会向外多算 1 像素**
    （真实孔洞 60..99x40..69 会得到 59..100x39..70，面积 1269 而非 1200），
    导致净面积系统性偏小；外轮廓本身是精确的，但孔洞偏差会污染面积统计。
    """
    from rasterio.features import shapes as rio_shapes
    from shapely.geometry import shape as shp_shape

    u8 = mask.astype(np.uint8)
    polys = []
    for geom, val in rio_shapes(u8, connectivity=4):
        if int(val) != 1:
            continue
        g = shp_shape(geom)
        if g.is_empty:
            continue
        if simplify_px and simplify_px > 0:
            g = g.simplify(simplify_px, preserve_topology=True)
        if g.is_empty:
            continue
        parts = [g] if g.geom_type == "Polygon" else list(getattr(g, "geoms", []))
        for p in parts:
            if p.geom_type == "Polygon" and p.area >= min_poly_area:
                polys.append(p)
    return polys


def read_geo(meta_path):
    """尝试用 rasterio 读出 CRS/transform；失败则返回 (None, None)。"""
    try:
        import rasterio
    except (ImportError, OSError):
        return None, None
    # noinspection PyBroadException
    try:
        with rasterio.open(str(meta_path)) as ds:
            return ds.crs, ds.transform
    except Exception:
        return None, None


def geojson_feature(poly, idx, transform):
    """构造一个 GeoJSON Feature。"""
    coords = []

    def ring(coords_seq):
        out = []
        for x, y in coords_seq:
            if transform is not None:
                mx, my = transform * (x, y)
                out.append([round(float(mx), 4), round(float(my), 4)])
            else:
                out.append([round(float(x), 2), round(float(y), 2)])
        if out and out[0] != out[-1]:
            out.append(out[0])          # GeoJSON 要求环闭合
        return out

    coords.append(ring(poly.exterior.coords))
    for interior in poly.interiors:
        coords.append(ring(interior.coords))

    props = {"id": idx, "area_px": round(float(poly.area), 2)}
    if transform is None:
        props["coord_system"] = "pixel"
    else:
        props["coord_system"] = "map"
    return {
        "type": "Feature",
        "properties": props,
        "geometry": {"type": "Polygon", "coordinates": coords},
    }


def write_geojson(out_path, features, crs):
    """用标准库写 GeoJSON（零额外依赖）。"""
    fc = {"type": "FeatureCollection", "features": features}
    if crs is not None:
        try:
            fc["crs"] = {"type": "name",
                         "properties": {"name": crs.to_string()}}
        except (AttributeError, TypeError, ValueError):
            pass
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False)


def write_shapefile(out_path, features, crs):
    """用 geopandas + pyogrio 写 shapefile；不可用时抛出。"""
    import geopandas as gpd
    from shapely.geometry import shape

    geoms = [shape(f["geometry"]) for f in features]
    props = [f["properties"] for f in features]
    gdf = gpd.GeoDataFrame(props, geometry=geoms, crs=crs if crs is not None else None)
    gdf.to_file(out_path, driver="ESRI Shapefile", engine="pyogrio")


def main():
    ap = argparse.ArgumentParser(description="遥感影像阴影提取：输入影像 -> 输出阴影矢量")
    ap.add_argument("--image", required=True, help="输入影像路径（tif/png/jpg）")
    ap.add_argument("--out", default=None, help="输出矢量路径（默认与输入同目录同名 .geojson）")
    ap.add_argument("--format", choices=["geojson", "shp"], default="geojson",
                    help="输出矢量格式（默认 geojson）")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT,
                    help="模型权重（默认=无泄漏主结果权重）")
    ap.add_argument("--min-area", type=int, default=300,
                    help="提示框的最小连通域面积（像素），默认 300（与主结果一致）")
    ap.add_argument("--min-poly-area", type=float, default=64.0,
                    help="输出多边形的最小面积（像素），默认 64")
    ap.add_argument("--simplify", type=float, default=1.5,
                    help="Douglas-Peucker 简化容差（像素），默认 1.5；0=不简化")
    ap.add_argument("--overlay", default=None, help="可选：同时输出绿色叠加预览 PNG")
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[ERR] 找不到影像: {img_path}")
        raise SystemExit(2)

    if args.out:
        out_path = Path(args.out)
    elif args.format == "geojson":
        out_path = img_path.with_suffix(".geojson")
    else:
        out_path = img_path.parent / (img_path.stem + "_shadows.shp")

    print("# 阴影提取工具（输入影像 -> 输出阴影矢量）")
    print(f"# 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}  设备: {DEVICE}")
    print(f"# 影像: {img_path}")
    print(f"# 权重: {Path(args.checkpoint).name}")
    if "leakfree" in Path(args.checkpoint).name:
        print("# 口径: 无泄漏划分（主结果口径）")
    else:
        print("# [!] 警告: 该权重不是无泄漏口径，数字系统性偏高，仅供历史对齐")

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"[ERR] cv2 无法读取影像: {img_path}")
        raise SystemExit(2)
    h, w = img_bgr.shape[:2]
    print(f"# 尺寸: {w} x {h}")

    sam = load_sam(args.checkpoint)
    mask, n_box = predict_mask(sam, img_bgr, args.min_area)
    print(f"# 提示框数: {n_box}   阴影像素占比: {100.0*mask.sum()/mask.size:.2f}%")

    polys = mask_to_polygons(mask, args.simplify, args.min_poly_area)
    print(f"# 矢量化多边形数: {len(polys)}")

    crs, transform = read_geo(img_path)
    if crs is not None and transform is not None:
        print(f"# 地理参考: 有（CRS = {crs.to_string()}）-> 输出地图坐标")
    else:
        print("# 地理参考: 无 -> 输出像素坐标（x=列, y=行）")
        transform = None

    features = [geojson_feature(p, i, transform) for i, p in enumerate(polys)]

    if args.format == "geojson":
        write_geojson(out_path, features, crs)
        print(f"[OK] 已写出 GeoJSON: {out_path}")
    else:
        try:
            write_shapefile(out_path, features, crs)
            print(f"[OK] 已写出 Shapefile: {out_path}")
        except Exception as e:
            print(f"[ERR] Shapefile 写出失败（{type(e).__name__}: {e}）")
            print("      可改用 --format geojson（零额外依赖）")
            raise SystemExit(3)

    if args.overlay:
        overlay = img_bgr.copy()
        overlay[mask] = [0, 255, 0]
        blended = cv2.addWeighted(img_bgr, 0.6, overlay, 0.4, 0)
        cv2.imwrite(str(args.overlay), blended)
        print(f"[OK] 已写出叠加预览: {args.overlay}")

    print(f"# 完成: {len(features)} 个阴影多边形")


if __name__ == "__main__":
    main()
