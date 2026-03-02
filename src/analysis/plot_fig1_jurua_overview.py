from __future__ import annotations

import argparse
import hashlib
import math
import sys
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from pyproj import Transformer

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.analysis.plot_preset import get_paper_figsize, setup_preset
from src.preprocessing.prepared_imagery import get_geotiffs_dir, get_prepared_imagery_dir


def _get_naturalearth_lowres_path() -> Path:
    cache_dir = _PROJECT_ROOT / "results" / "cache" / "naturalearth"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ne_zip = cache_dir / "ne_110m_admin_0_countries.zip"
    ne_shp = cache_dir / "ne_110m_admin_0_countries.shp"

    if not ne_shp.exists():
        if not ne_zip.exists():
            url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                ne_zip.write_bytes(resp.read())
        with zipfile.ZipFile(str(ne_zip), "r") as zf:
            zf.extractall(str(cache_dir))

    if not ne_shp.exists():
        raise FileNotFoundError(ne_shp)

    return ne_shp


def _get_naturalearth_rivers_path() -> Path:
    """下载 Natural Earth 50m 河流数据（包含黄河）"""
    cache_dir = _PROJECT_ROOT / "results" / "cache" / "naturalearth"
    cache_dir.mkdir(parents=True, exist_ok=True)

    ne_zip = cache_dir / "ne_50m_rivers_lake_centerlines.zip"
    ne_shp = cache_dir / "ne_50m_rivers_lake_centerlines.shp"

    if not ne_shp.exists():
        if not ne_zip.exists():
            url = "https://naturalearth.s3.amazonaws.com/50m_physical/ne_50m_rivers_lake_centerlines.zip"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                ne_zip.write_bytes(resp.read())
        with zipfile.ZipFile(str(ne_zip), "r") as zf:
            zf.extractall(str(cache_dir))

    if not ne_shp.exists():
        raise FileNotFoundError(ne_shp)

    return ne_shp


def _plot_yellow_river(ax: plt.Axes, *, extent: tuple[float, float, float, float]) -> bool:
    """在 ax 上绘制黄河干流线条"""
    try:
        import fiona
        from shapely.geometry import shape, LineString, MultiLineString
    except Exception:
        return False

    shp = _get_naturalearth_rivers_path()
    if not shp.exists():
        return False

    xmin, xmax, ymin, ymax = extent
    drew = False

    with fiona.open(str(shp)) as src:
        for feat in src:
            props = feat.get("properties", {})
            name = props.get("name", "") or ""
            # 筛选黄河（Huang He 或 Yellow）
            if "Huang" not in name and "Yellow" not in name:
                continue

            geom = feat.get("geometry")
            if geom is None:
                continue

            g = shape(geom)
            if isinstance(g, LineString):
                lines = [g]
            elif isinstance(g, MultiLineString):
                lines = list(g.geoms)
            else:
                continue

            for line in lines:
                coords = np.asarray(line.coords, dtype=float)
                if coords.ndim == 2 and coords.shape[0] >= 2:
                    # 检查是否在视图范围内
                    bx0, by0, bx1, by1 = line.bounds
                    if bx1 < xmin or bx0 > xmax or by1 < ymin or by0 > ymax:
                        continue
                    ax.plot(
                        coords[:, 0],
                        coords[:, 1],
                        color="#d4a017",  # 黄河用金黄色
                        linewidth=1.8,
                        zorder=2,
                        solid_capstyle="round",
                    )
                    drew = True

    return drew


def _try_plot_naturalearth_lowres(
    ax: plt.Axes,
    *,
    extent: tuple[float, float, float, float],
) -> bool:
    try:
        import fiona
        from shapely.geometry import shape, Polygon, MultiPolygon
    except Exception:
        return False

    shp: Path | None = None
    try:
        shp = _get_naturalearth_lowres_path()
    except Exception:
        shp = None

    if shp is None or (not shp.exists()):
        try:
            import geopandas as gpd
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                shp = Path(gpd.datasets.get_path("naturalearth_lowres"))
        except Exception:
            return False

    xmin, xmax, ymin, ymax = (float(extent[0]), float(extent[1]), float(extent[2]), float(extent[3]))
    
    # 先绘制海洋背景（浅蓝色）
    ax.add_patch(
        Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            facecolor="#d4e6f1",  # 浅蓝色海洋
            edgecolor="none",
            zorder=0,
        )
    )
    
    drew = False
    with fiona.open(str(shp)) as src:
        for feat in src:
            geom = feat.get("geometry")
            if geom is None:
                continue
            g = shape(geom)
            if isinstance(g, Polygon):
                geoms = [g]
            elif isinstance(g, MultiPolygon):
                geoms = list(g.geoms)
            else:
                continue

            for pg in geoms:
                bx0, by0, bx1, by1 = pg.bounds
                if float(bx1) < xmin or float(bx0) > xmax or float(by1) < ymin or float(by0) > ymax:
                    continue
                coords = np.asarray(pg.exterior.coords, dtype=float)
                if coords.ndim == 2 and coords.shape[0] >= 3:
                    ax.fill(
                        coords[:, 0],
                        coords[:, 1],
                        facecolor="#f5f5dc",  # 米色陆地
                        edgecolor="#666666",  # 深灰色边界
                        linewidth=0.6,
                        zorder=1,
                    )
                    drew = True

    return bool(drew)


def _default_background_path(site: str, year: int) -> Path | None:
    img = get_geotiffs_dir(site) / "image" / f"{site}_{int(year)}_01-01_12-31_full_image.tif"
    if img.exists():
        return img

    color = get_prepared_imagery_dir(site) / "Color" / f"{site}_{int(year)}_01-01_12-31_full_image_color.tif"
    if color.exists():
        return color

    return None


def _apply_bg_black_threshold(img: np.ndarray, valid: np.ndarray, black_threshold: float) -> np.ndarray:
    if float(black_threshold) < 0:
        return valid

    if img.ndim != 3 or img.shape[2] != 3:
        return valid

    brightness = np.nanmean(img, axis=2)
    return valid & np.isfinite(brightness) & (brightness > float(black_threshold))


def _apply_bg_style(img: np.ndarray, style: str, sat: float) -> np.ndarray:
    if img.ndim != 3 or img.shape[2] != 3:
        return img

    if style == "rgb":
        return img

    gray = (0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]).astype(float)
    gray3 = np.stack([gray, gray, gray], axis=2)

    if style == "gray":
        return gray3

    if style == "desat":
        a = float(np.clip(sat, 0.0, 1.0))
        return a * img + (1.0 - a) * gray3

    return img


def _read_rgb_geotiff_downsample(
    path: Path,
    *,
    max_dim: int = 2400,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    with rasterio.open(path) as src:
        if src.transform.is_identity and src.crs is None and not src.gcps[0]:
            raise ValueError(f"背景影像缺少地理参考: {path}")

        b = src.bounds
        extent = (float(b.left), float(b.right), float(b.bottom), float(b.top))

        scale = max(float(src.width), float(src.height)) / max(1.0, float(max_dim))
        if scale > 1.0:
            out_w = max(1, int(round(float(src.width) / scale)))
            out_h = max(1, int(round(float(src.height) / scale)))
        else:
            out_w = int(src.width)
            out_h = int(src.height)

        if int(src.count) >= 3:
            data = src.read(
                (1, 2, 3),
                out_shape=(3, out_h, out_w),
                resampling=Resampling.bilinear,
            )
            img = np.transpose(data, (1, 2, 0)).astype(float)
        else:
            data = src.read(
                1,
                out_shape=(out_h, out_w),
                resampling=Resampling.bilinear,
            )
            img = np.repeat(data[:, :, None].astype(float), 3, axis=2)

        m = src.read_masks(
            1,
            out_shape=(out_h, out_w),
            resampling=Resampling.nearest,
        )

    q2 = float(np.nanpercentile(img, 2))
    q98 = float(np.nanpercentile(img, 98))
    if np.isfinite(q2) and np.isfinite(q98) and q98 > q2:
        img = (img - q2) / (q98 - q2)

    img = np.clip(img, 0.0, 1.0)
    valid = np.asarray(m) > 0
    return img, valid, extent


def _read_mask_geotiff_downsample(
    path: Path,
    *,
    max_dim: int = 2400,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    with rasterio.open(path) as src:
        b = src.bounds
        extent = (float(b.left), float(b.right), float(b.bottom), float(b.top))

        scale = max(float(src.width), float(src.height)) / max(1.0, float(max_dim))
        if scale > 1.0:
            out_w = max(1, int(round(float(src.width) / scale)))
            out_h = max(1, int(round(float(src.height) / scale)))
        else:
            out_w = int(src.width)
            out_h = int(src.height)

        data = src.read(
            1,
            out_shape=(out_h, out_w),
            resampling=Resampling.nearest,
        )

    m = np.asarray(data)
    if m.dtype.kind in {"f"}:
        m = np.isfinite(m) & (m > 0)
    else:
        m = m > 0
    return m.astype(bool), extent


def _tight_extent_from_mask(
    mask: np.ndarray,
    extent: tuple[float, float, float, float],
    *,
    margin: float,
) -> tuple[float, float, float, float] | None:
    if mask is None or mask.size == 0:
        return None

    idx = np.argwhere(mask)
    if idx.size == 0:
        return None

    nrows, ncols = mask.shape
    # extent is (xmin, xmax, ymin, ymax)
    xmin, xmax, ymin, ymax = (float(extent[0]), float(extent[1]), float(extent[2]), float(extent[3]))
    dx = float(xmax - xmin)
    dy = float(ymax - ymin)
    
    # row 0 is top (ymax), row N is bottom (ymin)
    # col 0 is left (xmin), col N is right (xmax)

    r_min = int(idx[:, 0].min())
    r_max = int(idx[:, 0].max())
    c_min = int(idx[:, 1].min())
    c_max = int(idx[:, 1].max())

    x0 = xmin + (float(c_min) / float(ncols)) * dx
    x1 = xmin + (float(c_max + 1) / float(ncols)) * dx
    y1 = ymax - (float(r_min) / float(nrows)) * dy
    y0 = ymax - (float(r_max + 1) / float(nrows)) * dy
    
    # Apply margin
    x0 -= float(margin)
    x1 += float(margin)
    y0 -= float(margin)
    y1 += float(margin)

    return float(x0), float(x1), float(y0), float(y1)


def _get_raster_crs_str(path: Path) -> str | None:
    try:
        with rasterio.open(path) as src:
            if src.crs is None:
                return None
            return str(src.crs.to_string())
    except Exception:
        return None


def _read_site_polygon(shp_path: Path) -> tuple[list[np.ndarray], tuple[float, float, float, float]]:
    try:
        import fiona
        from shapely.geometry import shape, Polygon, MultiPolygon
    except Exception as e:
        raise RuntimeError("读取站点 Shapefile 需要 fiona+shapely。") from e

    polys: list[np.ndarray] = []
    xmin, ymin, xmax, ymax = np.inf, np.inf, -np.inf, -np.inf
    with fiona.open(str(shp_path)) as src:
        for feat in src:
            geom = feat.get("geometry")
            if geom is None:
                continue
            g = shape(geom)
            if isinstance(g, Polygon):
                geoms = [g]
            elif isinstance(g, MultiPolygon):
                geoms = list(g.geoms)
            else:
                continue

            for pg in geoms:
                coords = np.asarray(pg.exterior.coords, dtype=float)
                if coords.ndim == 2 and coords.shape[0] >= 3:
                    polys.append(coords)
                    bx0, by0, bx1, by1 = pg.bounds
                    xmin = min(xmin, float(bx0))
                    ymin = min(ymin, float(by0))
                    xmax = max(xmax, float(bx1))
                    ymax = max(ymax, float(by1))

    if not polys or not np.isfinite(xmin):
        raise ValueError(f"站点 shp 中未找到可用多边形: {shp_path}")
    return polys, (float(xmin), float(xmax), float(ymin), float(ymax))


def _add_panel_letter(ax: plt.Axes, letter: str) -> None:
    t = ax.text(
        0.015,
        0.985,
        str(letter),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="k",
        zorder=20,
    )
    t.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.2))
    t.set_path_effects([pe.Stroke(linewidth=2.0, foreground="w"), pe.Normal()])


def _try_add_online_basemap(
    ax: plt.Axes,
    *,
    crs: str,
    basemap: str,
    zoom: int | None,
    add_labels: bool = False,
) -> bool:
    b = str(basemap)
    if b in {"", "none", "off"}:
        return False

    if b in {"satellite", "imagery", "esri"}:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        if not (np.isfinite(xmin) and np.isfinite(xmax) and np.isfinite(ymin) and np.isfinite(ymax)):
            return False
        if float(xmax) <= float(xmin) or float(ymax) <= float(ymin):
            return False

        width_px = 1800
        if zoom is not None:
            try:
                z = float(zoom)
                width_px = int(np.clip(900.0 * (1.25 ** (z - 8.0)), 900.0, 3200.0))
            except Exception:
                width_px = 1800

        aspect = (float(xmax) - float(xmin)) / (float(ymax) - float(ymin))
        height_px = int(round(float(width_px) / max(float(aspect), 1e-6)))
        height_px = int(np.clip(height_px, 600, 3200))

        wkid = 4326
        try:
            s = str(crs).upper().strip()
            if s.startswith("EPSG:"):
                wkid = int(s.split(":", 1)[1])
        except Exception:
            wkid = 4326

        url_img = (
            "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
            f"?bbox={float(xmin)},{float(ymin)},{float(xmax)},{float(ymax)}"
            f"&bboxSR={int(wkid)}&imageSR={int(wkid)}"
            f"&size={int(width_px)},{int(height_px)}"
            "&format=png32&f=image"
        )

        url_lbl = (
            "https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/export"
            f"?bbox={float(xmin)},{float(ymin)},{float(xmax)},{float(ymax)}"
            f"&bboxSR={int(wkid)}&imageSR={int(wkid)}"
            f"&size={int(width_px)},{int(height_px)}"
            "&format=png32&transparent=true&f=image"
        )

        cache_dir = _PROJECT_ROOT / "results" / "cache" / "esri_world_imagery"
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.md5(url_img.encode("utf-8")).hexdigest()
        cache_path = cache_dir / f"{key}.png"

        try:
            if not cache_path.exists():
                req = urllib.request.Request(url_img, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    cache_path.write_bytes(resp.read())
            img = mpimg.imread(str(cache_path))
            ax.imshow(img, extent=(float(xmin), float(xmax), float(ymin), float(ymax)), origin="upper", zorder=1)

            if bool(add_labels):
                key2 = hashlib.md5(url_lbl.encode("utf-8")).hexdigest()
                cache_path2 = cache_dir / f"{key2}.png"
                if not cache_path2.exists():
                    req2 = urllib.request.Request(url_lbl, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(req2, timeout=30) as resp2:
                        cache_path2.write_bytes(resp2.read())
                img2 = mpimg.imread(str(cache_path2))
                ax.imshow(
                    img2,
                    extent=(float(xmin), float(xmax), float(ymin), float(ymax)),
                    origin="upper",
                    alpha=0.88,
                    zorder=2,
                )

            return True
        except Exception:
            return False

    return False


def _add_north_arrow(ax: plt.Axes, *, x: float = 0.08, y: float = 0.86, color: str = "black") -> None:
    """添加指北针。color 参数控制箭头和文字颜色。"""
    try:
        from matplotlib.patches import FancyArrow

        arrow = FancyArrow(
            float(x),
            float(y),
            0.0,
            0.06,
            transform=ax.transAxes,
            color=str(color),
            width=0.008,
            head_width=0.03,
            head_length=0.04,
            zorder=10,
        )
        ax.add_patch(arrow)
        ax.text(
            float(x),
            float(y) + 0.095,
            "N",
            transform=ax.transAxes,
            fontsize=12,
            ha="center",
            va="bottom",
            color=str(color),
            zorder=10,
        )
    except Exception:
        return


def _add_scalebar_lonlat(ax: plt.Axes, *, target_frac: float = 0.25, color: str = "black") -> None:
    """添加比例尺。color 参数控制边框和文字颜色。"""
    try:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        lat_c = 0.5 * (float(ymin) + float(ymax))
        meters_per_degree_lon = 111320.0 * math.cos(math.radians(float(lat_c)))
        map_width_m = (float(xmax) - float(xmin)) * float(meters_per_degree_lon)
        if not np.isfinite(map_width_m) or map_width_m <= 0:
            return

        target = float(map_width_m) * float(target_frac)
        mag = 10 ** int(math.floor(math.log10(max(target, 1.0))))
        scale_m = mag
        for s in [1, 2, 5]:
            cand = float(s) * float(mag)
            if cand <= target:
                scale_m = cand

        scale_deg = float(scale_m) / float(meters_per_degree_lon)
        h_deg = (float(ymax) - float(ymin)) * 0.010
        margin_x = (float(xmax) - float(xmin)) * 0.05
        margin_y = (float(ymax) - float(ymin)) * 0.04
        rect = Rectangle(
            (float(xmax) - float(margin_x) - float(scale_deg), float(ymin) + float(margin_y)),
            float(scale_deg),
            float(h_deg),
            facecolor=str(color),
            edgecolor=str(color),
            linewidth=1.0,
            transform=ax.transData,
            zorder=9,
        )
        ax.add_patch(rect)
        label_km = int(round(float(scale_m) / 1000.0))
        ax.text(
            float(xmax) - float(margin_x) - 0.5 * float(scale_deg),
            float(ymin) + float(margin_y) + 1.8 * float(h_deg),
            f"{label_km} km",
            fontsize=10,
            ha="center",
            va="bottom",
            color=str(color),
            zorder=10,
        )
    except Exception:
        return


def _add_flow_arrow(
    ax: plt.Axes,
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    label: str = "Flow",
    color: str = "black",
    outline: str | None = None,
) -> None:
    try:
        from matplotlib.patches import FancyArrowPatch

        if outline is None:
            outline = "black" if str(color).lower() in {"white", "w"} else "white"

        # 先画一条“箭身”线，避免出现只有箭头没有箭身的情况
        ln = ax.plot(
            [float(x0), float(x1)],
            [float(y0), float(y1)],
            color=str(color),
            linewidth=2.2,
            solid_capstyle="round",
            zorder=12,
        )[0]
        ln.set_path_effects([pe.Stroke(linewidth=4.4, foreground=str(outline)), pe.Normal()])

        # 再画箭头头部（matplotlib 有时会把短箭的箭身挤没，这里双保险）
        arr = FancyArrowPatch(
            (float(x0), float(y0)),
            (float(x1), float(y1)),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.2,
            color=str(color),
            zorder=13,
        )
        arr.set_path_effects([pe.Stroke(linewidth=4.4, foreground=str(outline)), pe.Normal()])
        ax.add_patch(arr)

        if str(label).strip():
            xm = 0.5 * (float(x0) + float(x1))
            ym = 0.5 * (float(y0) + float(y1))
            t = ax.text(
                xm,
                ym,
                str(label),
                fontsize=9,
                color=str(color),
                ha="center",
                va="bottom",
                zorder=13,
            )
            t.set_path_effects([pe.Stroke(linewidth=3.0, foreground=str(outline)), pe.Normal()])
    except Exception:
        return


def _yellow_river_flow_segment(
    *,
    extent: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    try:
        import fiona
        from shapely.geometry import shape, LineString, MultiLineString
    except Exception:
        return None

    try:
        shp = _get_naturalearth_rivers_path()
    except Exception:
        return None

    if not shp.exists():
        return None

    xmin, xmax, ymin, ymax = (float(extent[0]), float(extent[1]), float(extent[2]), float(extent[3]))

    best_coords: np.ndarray | None = None
    best_n = 0
    with fiona.open(str(shp)) as src:
        for feat in src:
            props = feat.get("properties", {})
            name = props.get("name", "") or ""
            if "Huang" not in name and "Yellow" not in name:
                continue

            geom = feat.get("geometry")
            if geom is None:
                continue

            g = shape(geom)
            if isinstance(g, LineString):
                lines = [g]
            elif isinstance(g, MultiLineString):
                lines = list(g.geoms)
            else:
                continue

            for line in lines:
                bx0, by0, bx1, by1 = line.bounds
                if float(bx1) < xmin or float(bx0) > xmax or float(by1) < ymin or float(by0) > ymax:
                    continue
                coords = np.asarray(line.coords, dtype=float)
                if coords.ndim != 2 or coords.shape[0] < 6:
                    continue
                if int(coords.shape[0]) > int(best_n):
                    best_coords = coords
                    best_n = int(coords.shape[0])

    if best_coords is None or int(best_coords.shape[0]) < 6:
        return None

    # 入口处：优先选取靠近图框左侧（xmin）的河段
    n = int(best_coords.shape[0])

    inside = (
        (best_coords[:, 0] >= xmin)
        & (best_coords[:, 0] <= xmax)
        & (best_coords[:, 1] >= ymin)
        & (best_coords[:, 1] <= ymax)
    )
    idx = np.where(inside)[0]
    if idx.size < 6:
        return None

    # 在“靠左侧”的范围内挑一个点（用于代表河流进入图框的位置）
    left_x = float(xmin) + 0.20 * float(xmax - xmin)
    cand = idx[best_coords[idx, 0] <= left_x]
    if cand.size == 0:
        cand = idx

    i = int(cand[np.argmin(best_coords[cand, 0])])
    i0 = max(0, i - 2)
    i1 = min(n - 1, i + 4)

    x0, y0 = float(best_coords[i0, 0]), float(best_coords[i0, 1])
    x1, y1 = float(best_coords[i1, 0]), float(best_coords[i1, 1])

    # 经验先验：黄河总体西→东（下游向右）。若方向相反则翻转。
    if float(x1 - x0) < 0:
        x0, y0, x1, y1 = x1, y1, x0, y0

    if (abs(x1 - x0) + abs(y1 - y0)) <= 0.10:
        return None

    return (x0, y0, x1, y1)


def _add_scalebar_xy_m(ax: plt.Axes, *, length_m: float) -> None:
    try:
        if float(length_m) <= 0:
            return
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        dx = float(xmax - xmin)
        dy = float(ymax - ymin)
        if dx <= 0 or dy <= 0:
            return

        x1 = float(xmax) - 0.06 * dx
        x0 = float(x1) - float(length_m)
        y0 = float(ymin) + 0.08 * dy
        h = 0.010 * dy

        rect = Rectangle(
            (float(x0), float(y0)),
            float(length_m),
            float(h),
            facecolor="white",
            edgecolor="black",
            linewidth=1.0,
            transform=ax.transData,
            zorder=9,
        )
        ax.add_patch(rect)
        ax.text(
            0.5 * (float(x0) + float(x1)),
            float(y0) + 1.8 * float(h),
            f"{float(length_m) / 1000.0:g} km",
            fontsize=10,
            ha="center",
            va="bottom",
            zorder=10,
        )
    except Exception:
        return


def _water_fraction_in_box(
    water: np.ndarray,
    extent: tuple[float, float, float, float],
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> float:
    xmin, xmax, ymin, ymax = extent
    if water.size == 0:
        return 0.0
    h, w = water.shape
    if float(xmax) <= float(xmin) or float(ymax) <= float(ymin):
        return 0.0

    xx0 = float(min(x0, x1))
    xx1 = float(max(x0, x1))
    yy0 = float(min(y0, y1))
    yy1 = float(max(y0, y1))

    c0 = int(np.floor((xx0 - float(xmin)) / (float(xmax) - float(xmin)) * (w - 1)))
    c1 = int(np.ceil((xx1 - float(xmin)) / (float(xmax) - float(xmin)) * (w - 1)))
    r0 = int(np.floor((float(ymax) - yy1) / (float(ymax) - float(ymin)) * (h - 1)))
    r1 = int(np.ceil((float(ymax) - yy0) / (float(ymax) - float(ymin)) * (h - 1)))

    c0 = int(np.clip(c0, 0, w - 1))
    c1 = int(np.clip(c1, 0, w - 1))
    r0 = int(np.clip(r0, 0, h - 1))
    r1 = int(np.clip(r1, 0, h - 1))
    if c1 <= c0 or r1 <= r0:
        return 0.0

    patch = water[r0:r1, c0:c1]
    if patch.size == 0:
        return 0.0
    return float(np.mean(patch.astype(float)))


def plot_fig1_jurua_overview(
    *,
    site: str,
    year: int,
    background_path: Path,
    rois: list[tuple[float, float, float, float]],
    out_path: Path,
    preset: str,
    dpi: int,
    bg_style: str,
    bg_sat: float,
    bg_alpha: float,
    bg_black_threshold: float,
    roi_color: str,
    roi_lw: float,
    roi_labels: str,
    roi_label_pos: str,
) -> None:
    setup_preset(preset, dpi)

    # --- Publication Grade Style ---
    import matplotlib as mpl
    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["font.size"] = 10 if preset == "paper" else 12
    # -----------------------------

    if not background_path.exists():
        raise FileNotFoundError(background_path)

    img, valid, extent = _read_rgb_geotiff_downsample(background_path, max_dim=2400)
    img = _apply_bg_style(img, style=bg_style, sat=bg_sat)
    valid = _apply_bg_black_threshold(img, valid, bg_black_threshold)

    xmin, xmax, ymin, ymax = extent
    dx = float(xmax - xmin)
    dy = float(ymax - ymin)
    aspect = dx / dy if dy > 0 else 1.0

    if preset == "paper":
        width_mm = 190
        height_mm = float(np.clip(float(width_mm) / max(aspect, 1e-6), 70.0, 150.0))
        figsize = get_paper_figsize(width_mm, height_mm)
    else:
        figsize = (10, 10 / max(aspect, 1e-6))

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

    ax.set_facecolor("white")
    ax.imshow(
        img,
        extent=extent,
        origin="upper",
        alpha=(valid.astype(float) * float(bg_alpha)),
    )

    show_roi_labels: bool
    if str(roi_labels) == "on":
        show_roi_labels = True
    elif str(roi_labels) == "off":
        show_roi_labels = False
    else:
        show_roi_labels = preset == "paper"

    for i, (rx0, rx1, ry0, ry1) in enumerate(rois, start=1):
        w = float(rx1 - rx0)
        h = float(ry1 - ry0)

        r0 = Rectangle(
            (float(rx0), float(ry0)),
            w,
            h,
            fill=False,
            edgecolor="k",
            linewidth=float(roi_lw) + 1.6,
            alpha=0.95,
            zorder=4,
        )
        ax.add_patch(r0)

        r1 = Rectangle(
            (float(rx0), float(ry0)),
            w,
            h,
            fill=False,
            edgecolor=str(roi_color),
            linewidth=float(roi_lw),
            alpha=0.98,
            zorder=5,
        )
        ax.add_patch(r1)

        if show_roi_labels:
            if str(roi_label_pos) == "right-outside":
                tx, ty, ha, va = float(rx1) + 0.01 * dx, float(ry0) - 0.012 * dy, "left", "top"
            elif str(roi_label_pos) == "upper-left-outside":
                tx, ty, ha, va = float(rx0) - 0.01 * dx, float(ry1) + 0.01 * dy, "left", "bottom"
            elif str(roi_label_pos) == "upper-left":
                tx, ty, ha, va = float(rx0) + 0.01 * dx, float(ry1) - 0.01 * dy, "left", "top"
            elif str(roi_label_pos) == "lower-left":
                tx, ty, ha, va = float(rx0) + 0.01 * dx, float(ry0) + 0.01 * dy, "left", "bottom"
            elif str(roi_label_pos) == "lower-right":
                tx, ty, ha, va = float(rx1) - 0.01 * dx, float(ry0) + 0.01 * dy, "right", "bottom"
            else:
                tx, ty, ha, va = float(rx1) - 0.01 * dx, float(ry1) - 0.01 * dy, "right", "top"

            t = ax.text(
                tx,
                ty,
                f"ROI-{i}",
                ha=str(ha),
                va=str(va),
                fontsize=9,
                color="k",
                zorder=6,
            )
            t.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.80, pad=0.25))
            t.set_path_effects([pe.Stroke(linewidth=2.0, foreground="w"), pe.Normal()])

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_fig1_study_areas(
    *,
    jurua_site: str,
    jurua_year: int,
    jurua_background: Path | None,
    jurua_rois: list[tuple[float, float, float, float]],
    jurua_roi_labels: str = "auto",
    jurua_roi_label_pos: str = "upper-right",
    huanghe_year: int,
    huanghe_mask_level: int,
    huanghe_a_shp: Path,
    huanghe_b_shp: Path,
    huanghe_overview_basemap: str,
    huanghe_overview_basemap_zoom: int,
    huanghe_zoom_basemap: str,
    huanghe_zoom_basemap_zoom: int,
    huanghe_overview_margin_lon: float,
    huanghe_overview_margin_lat: float,
    huanghe_zoom_margin_deg: float,
    show_axes: str,
    show_grid: str,
    jurua_mask_level: int,
    jurua_water_mask: Path | None,
    jurua_scalebar_km: float,
    out_path: Path,
    preset: str,
    dpi: int,
    bg_alpha: float,
    bg_black_threshold: float,
    bg_style: str,
    bg_sat: float,
) -> None:
    setup_preset(preset, dpi)

    # --- Publication Grade Style ---
    import matplotlib as mpl
    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["font.size"] = 10 if preset == "paper" else 12
    # -----------------------------

    polys_a, bounds_a = _read_site_polygon(huanghe_a_shp)
    polys_b, bounds_b = _read_site_polygon(huanghe_b_shp)

    axmin = min(bounds_a[0], bounds_b[0])
    axmax = max(bounds_a[1], bounds_b[1])
    aymin = min(bounds_a[2], bounds_b[2])
    aymax = max(bounds_a[3], bounds_b[3])

    ov_xmin = float(axmin) - float(huanghe_overview_margin_lon)
    ov_xmax = float(axmax) + float(huanghe_overview_margin_lon)
    # 纬度范围收窄到 33°~41°
    ov_ymin = 33.0
    ov_ymax = 41.0

    if preset == "paper":
        figsize = get_paper_figsize(190, 210)
    else:
        figsize = (12, 13)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.65, 1.0, 1.2])
    ax_over = fig.add_subplot(gs[0, :])
    ax_a = fig.add_subplot(gs[1, 0])
    ax_b = fig.add_subplot(gs[1, 1])

    # 第三行：Jurua 子图横跨两列（与 b/c 同宽）
    ax_j = fig.add_subplot(gs[2, :])

    # (a) Overview of Yellow River basin (location-map style)
    ax_over.set_facecolor("white")
    ax_over.set_xlim(ov_xmin, ov_xmax)
    ax_over.set_ylim(ov_ymin, ov_ymax)
    ok_basemap = _try_add_online_basemap(
        ax_over,
        crs="EPSG:4326",
        basemap=str(huanghe_overview_basemap),
        zoom=int(huanghe_overview_basemap_zoom) if int(huanghe_overview_basemap_zoom) > 0 else None,
    )

    did_plot_world = False
    if (not ok_basemap) and str(huanghe_overview_basemap) == "naturalearth":
        try:
            did_plot_world = _try_plot_naturalearth_lowres(ax_over, extent=(ov_xmin, ov_xmax, ov_ymin, ov_ymax))
        except Exception:
            did_plot_world = False

        ax_over.set_xlim(ov_xmin, ov_xmax)
        ax_over.set_ylim(ov_ymin, ov_ymax)

    if (not ok_basemap) and (not did_plot_world):
        ax_over.add_patch(
            Rectangle(
                (float(ov_xmin), float(ov_ymin)),
                float(ov_xmax - ov_xmin),
                float(ov_ymax - ov_ymin),
                facecolor="0.92",
                edgecolor="none",
                zorder=0,
            )
        )

    # 绘制黄河干流线条
    try:
        _plot_yellow_river(ax_over, extent=(ov_xmin, ov_xmax, ov_ymin, ov_ymax))
    except Exception:
        pass

    # 绘制 YR-A 研究区矩形框（更醒目）
    rect_a = Rectangle(
        (bounds_a[0], bounds_a[2]),
        bounds_a[1] - bounds_a[0],
        bounds_a[3] - bounds_a[2],
        fill=False,
        edgecolor="#ff1744",
        linewidth=2.5,
        zorder=4,
    )
    ax_over.add_patch(rect_a)
    # 标注 YR-A
    cx_a_ov = 0.5 * (bounds_a[0] + bounds_a[1])
    cy_a_ov = bounds_a[3]  # 标注在框上方
    ax_over.text(cx_a_ov, cy_a_ov + 0.5, "YR-A", color="#ff1744", fontsize=10, ha="center", va="bottom", weight="bold", zorder=6, path_effects=[pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()])

    # 绘制 YR-B 研究区矩形框（更醒目）
    rect_b = Rectangle(
        (bounds_b[0], bounds_b[2]),
        bounds_b[1] - bounds_b[0],
        bounds_b[3] - bounds_b[2],
        fill=False,
        edgecolor="#2979ff",
        linewidth=2.5,
        zorder=4,
    )
    ax_over.add_patch(rect_b)
    # 标注 YR-B
    cx_b_ov = 0.5 * (bounds_b[0] + bounds_b[1])
    cy_b_ov = bounds_b[3]  # 标注在框上方
    ax_over.text(cx_b_ov, cy_b_ov + 0.5, "YR-B", color="#2979ff", fontsize=10, ha="center", va="bottom", weight="bold", zorder=6, path_effects=[pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()])

    ax_over.text(
        0.5,
        0.08,
        "Yellow River",
        transform=ax_over.transAxes,
        fontsize=9,
        color="white",
        ha="center",
        va="center",
        bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1),
        zorder=5,
    )

    ax_over.set_aspect("equal", adjustable="box")
    _add_north_arrow(ax_over, x=0.08, y=0.76)

    # Flow direction (downstream) indicator for Yellow River
    if False:
        try:
            seg = _yellow_river_flow_segment(extent=(ov_xmin, ov_xmax, ov_ymin, ov_ymax))
            if seg is not None:
                dx = float(ov_xmax - ov_xmin)
                dy = float(ov_ymax - ov_ymin)
                vx = float(seg[2] - seg[0])
                vy = float(seg[3] - seg[1])
                mag = float((vx * vx + vy * vy) ** 0.5)
                if mag > 0:
                    vx /= mag
                    vy /= mag
                if float(vx) < 0:
                    vx *= -1.0
                    vy *= -1.0

                x0 = float(ov_xmin) + 0.010 * float(dx)
                y0 = 0.5 * (float(seg[1]) + float(seg[3]))
                y0 = float(np.clip(y0, float(ov_ymin) + 0.06 * float(dy), float(ov_ymax) - 0.06 * float(dy)))
                L = max(0.60, 0.055 * float(dx))
                x1 = float(x0) + float(L) * float(vx)
                y1 = float(y0) + float(L) * float(vy)

                _add_flow_arrow(
                    ax_over,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    label="",
                    color="black",
                )
        except Exception:
            pass
    if str(show_grid) == "on":
        ax_over.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    if str(show_axes) == "on":
        ax_over.set_xlabel("Longitude (°)", fontsize=10)
        ax_over.set_ylabel("Latitude (°)", fontsize=10)
    else:
        ax_over.set_xticks([])
        ax_over.set_yticks([])
    _add_panel_letter(ax_over, "(a)")


    # (b) Reach A zoom-in (study-area-map style)
    mask_a_path = get_geotiffs_dir("HuangHe-A") / f"mask{int(huanghe_mask_level)}" / f"HuangHe-A_{int(huanghe_year)}_01-01_12-31_mask{int(huanghe_mask_level)}.tif"
    mask_b_path = get_geotiffs_dir("HuangHe-B") / f"mask{int(huanghe_mask_level)}" / f"HuangHe-B_{int(huanghe_year)}_01-01_12-31_mask{int(huanghe_mask_level)}.tif"

    margin = float(huanghe_zoom_margin_deg)

    a_x0, a_x1, a_y0, a_y1 = (
        float(bounds_a[0]) - margin,
        float(bounds_a[1]) + margin,
        float(bounds_a[2]) - margin,
        float(bounds_a[3]) + margin,
    )
    b_x0, b_x1, b_y0, b_y1 = (
        float(bounds_b[0]) - margin,
        float(bounds_b[1]) + margin,
        float(bounds_b[2]) - margin,
        float(bounds_b[3]) + margin,
    )

    if mask_a_path.exists():
        try:
            m_a, ext_a = _read_mask_geotiff_downsample(mask_a_path, max_dim=2200)
            tight_a = _tight_extent_from_mask(m_a, ext_a, margin=margin)
            if tight_a is not None:
                a_x0, a_x1, a_y0, a_y1 = (float(tight_a[0]), float(tight_a[1]), float(tight_a[2]), float(tight_a[3]))
        except Exception:
            pass

    if mask_b_path.exists():
        try:
            m_b, ext_b = _read_mask_geotiff_downsample(mask_b_path, max_dim=2200)
            tight_b = _tight_extent_from_mask(m_b, ext_b, margin=margin)
            if tight_b is not None:
                b_x0, b_x1, b_y0, b_y1 = (float(tight_b[0]), float(tight_b[1]), float(tight_b[2]), float(tight_b[3]))
        except Exception:
            pass

    dx_a = float(a_x1 - a_x0)
    dy_a = float(a_y1 - a_y0)
    dx_b = float(b_x1 - b_x0)
    dy_b = float(b_y1 - b_y0)
    
    # 让 c 的宽高比与 b 一致：扩展 c 的经度范围
    aspect_a = dx_a / dy_a if dy_a > 0 else 1.0
    # c 的纬度范围保持紧贴，经度范围扩展以匹配 b 的宽高比
    target_dx_b = dy_b * aspect_a
    if target_dx_b > dx_b:
        # 需要扩展 c 的经度范围
        cx_b = 0.5 * (b_x0 + b_x1)
        b_x0 = cx_b - 0.5 * target_dx_b
        b_x1 = cx_b + 0.5 * target_dx_b
        dx_b = target_dx_b

    ax_a.set_facecolor("white")
    ax_a.set_xlim(a_x0, a_x1)
    ax_a.set_ylim(a_y0, a_y1)

    ok_a = _try_add_online_basemap(
        ax_a,
        crs="EPSG:4326",
        basemap=str(huanghe_zoom_basemap),
        zoom=int(huanghe_zoom_basemap_zoom) if int(huanghe_zoom_basemap_zoom) > 0 else None,
        add_labels=True,
    )
    if (not ok_a) and mask_a_path.exists():
        m, ext = _read_mask_geotiff_downsample(mask_a_path, max_dim=2200)
        water = np.where(m, 1.0, np.nan)
        cmap = ListedColormap(["#1e88e5"])
        cmap.set_bad(alpha=0.0)
        ax_a.imshow(water, extent=ext, origin="upper", cmap=cmap, vmin=0, vmax=1, alpha=0.55, zorder=1)

    for poly in polys_a:
        ax_a.plot(poly[:, 0], poly[:, 1], color="black", linewidth=1.6, zorder=6)

    ax_a.set_aspect("equal", adjustable="box")
    _add_north_arrow(ax_a, x=0.12, y=0.78, color="white")
    _add_scalebar_lonlat(ax_a, target_frac=0.22, color="white")
    if str(show_grid) == "on":
        ax_a.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax_a.text(
        0.5,
        0.98,
        "YR-A",
        transform=ax_a.transAxes,
        fontsize=9,
        color="white",
        ha="center",
        va="top",
        bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1),
        zorder=10,
    )

    if str(show_axes) == "on":
        ax_a.set_xlabel("Longitude (°)", fontsize=10)
        ax_a.set_ylabel("Latitude (°)", fontsize=10)
        # 纬度刻度两位小数
        from matplotlib.ticker import FormatStrFormatter
        ax_a.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    else:
        ax_a.set_xticks([])
        ax_a.set_yticks([])

    _add_panel_letter(ax_a, "(b)")

    # (c) Reach B zoom-in (study-area-map style)
    mask_b_path = get_geotiffs_dir("HuangHe-B") / f"mask{int(huanghe_mask_level)}" / f"HuangHe-B_{int(huanghe_year)}_01-01_12-31_mask{int(huanghe_mask_level)}.tif"

    ax_b.set_facecolor("white")
    ax_b.set_xlim(b_x0, b_x1)
    ax_b.set_ylim(b_y0, b_y1)

    ok_b = _try_add_online_basemap(
        ax_b,
        crs="EPSG:4326",
        basemap=str(huanghe_zoom_basemap),
        zoom=int(huanghe_zoom_basemap_zoom) if int(huanghe_zoom_basemap_zoom) > 0 else None,
        add_labels=True,
    )
    if (not ok_b) and mask_b_path.exists():
        m, ext = _read_mask_geotiff_downsample(mask_b_path, max_dim=2200)
        water = np.where(m, 1.0, np.nan)
        cmap = ListedColormap(["#1e88e5"])
        cmap.set_bad(alpha=0.0)
        ax_b.imshow(water, extent=ext, origin="upper", cmap=cmap, vmin=0, vmax=1, alpha=0.55, zorder=1)

    for poly in polys_b:
        ax_b.plot(poly[:, 0], poly[:, 1], color="black", linewidth=1.6, zorder=6)

    ax_b.set_aspect("equal", adjustable="box")
    _add_north_arrow(ax_b, x=0.12, y=0.78, color="white")
    _add_scalebar_lonlat(ax_b, target_frac=0.22, color="white")
    if str(show_grid) == "on":
        ax_b.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax_b.text(
        0.5,
        0.98,
        "YR-B",
        transform=ax_b.transAxes,
        fontsize=9,
        color="white",
        ha="center",
        va="top",
        bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1),
        zorder=10,
    )

    if str(show_axes) == "on":
        ax_b.set_xlabel("Longitude (°)", fontsize=10)
        ax_b.set_ylabel("Latitude (°)", fontsize=10)
        # 经度刻度两位小数
        from matplotlib.ticker import FormatStrFormatter
        ax_b.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    else:
        ax_b.set_xticks([])
        ax_b.set_yticks([])

    _add_panel_letter(ax_b, "(c)")

    # (d) Jurua validation site (Esri WorldImagery)
    # Convert Jurua ROIs (UTM) to WGS84 for consistent Lon/Lat display
    # Jurua-A 位于 UTM Zone 19S
    jurua_crs_source = "EPSG:32719"
    
    # Transform ROIs to WGS84
    jurua_rois_wgs84 = []
    try:
        from pyproj import Transformer
        transformer = Transformer.from_crs(jurua_crs_source, "EPSG:4326", always_xy=True)
        for r in jurua_rois:
            # r is (xmin, xmax, ymin, ymax)
            # transform points (xmin, ymin) and (xmax, ymax) is not enough for rotation, but for bbox it's approx
            # Better to transform the polygon. Here we just want the bounds.
            lons, lats = transformer.transform([r[0], r[1]], [r[2], r[3]])
            # lons = [lon0, lon1], lats = [lat0, lat1]
            jurua_rois_wgs84.append((min(lons), max(lons), min(lats), max(lats)))
    except Exception:
        # Fallback if pyproj fails or CRS unknown (keep as is, but it will look wrong if mixed)
        jurua_rois_wgs84 = jurua_rois

    ax_j.set_facecolor("white")

    # Calculate tight extent in WGS84
    j_x0 = min([r[0] for r in jurua_rois_wgs84])
    j_x1 = max([r[1] for r in jurua_rois_wgs84])
    j_y0 = min([r[2] for r in jurua_rois_wgs84])
    j_y1 = max([r[3] for r in jurua_rois_wgs84])
    
    # Add margin (in degrees)
    m_deg = 0.04 # Approx 4-5 km
    ax_j.set_xlim(j_x0 - m_deg, j_x1 + m_deg)
    ax_j.set_ylim(j_y0 - m_deg, j_y1 + m_deg)

    # Plot Esri Basemap (WGS84)
    _try_add_online_basemap(
        ax_j,
        crs="EPSG:4326",
        basemap="esri",
        zoom=int(huanghe_zoom_basemap_zoom) if int(huanghe_zoom_basemap_zoom) > 0 else 11,
        add_labels=True,
    )

    # Plot Water Mask (Transform to WGS84 if needed)
    # Since warping raster on the fly is heavy, we might skip showing the water mask overlay 
    # or just rely on the Esri map which shows water. 
    # The user asked for "ROI标识也依然跟河道覆盖、需解决".
    # We can transform the ROI label positions check using the original UTM mask.
    
    roi_color = "#00e5ff"
    roi_lw = 1.05

    show_roi_labels: bool
    if str(jurua_roi_labels) == "on":
        show_roi_labels = True
    elif str(jurua_roi_labels) == "off":
        show_roi_labels = False
    else:
        show_roi_labels = preset == "paper"

    for i, (rx0_w, rx1_w, ry0_w, ry1_w) in enumerate(jurua_rois_wgs84, start=1):
        w_w = rx1_w - rx0_w
        h_w = ry1_w - ry0_w
        
        # Original UTM coordinates for this ROI
        rx0_u, rx1_u, ry0_u, ry1_u = jurua_rois[i-1]
        w_u = rx1_u - rx0_u
        h_u = ry1_u - ry0_u

        ax_j.add_patch(
            Rectangle(
                (rx0_w, ry0_w),
                w_w,
                h_w,
                fill=False,
                edgecolor="k",
                linewidth=float(roi_lw) + 0.85,
                alpha=0.95,
                zorder=4,
            )
        )
        ax_j.add_patch(
            Rectangle(
                (rx0_w, ry0_w),
                w_w,
                h_w,
                fill=False,
                edgecolor=str(roi_color),
                linewidth=float(roi_lw),
                alpha=0.98,
                zorder=5,
            )
        )

        # Smart Label Placement (checking water coverage in UTM)
        # We define candidates in WGS84 relative to the box
        # But verify them by converting back to UTM and checking the mask? 
        # Or simpler: Just place them to the Right/Left with an arrow if needed.
        # User said: "ROI标识怎么还是覆盖到矩形框？统一往旁边挪一下啊"
        # So we move them OUTSIDE the box.
        
        if show_roi_labels:
            pad_x = 0.02 * float(w_w)
            pad_y = 0.02 * float(h_w)

            pos = str(jurua_roi_label_pos)
            if pos == "right-outside":
                tx, ty, ha, va = rx1_w + 0.35 * pad_x, ry0_w - 0.35 * pad_y, "left", "top"
            elif pos == "upper-left-outside":
                tx, ty, ha, va = rx0_w - 0.35 * pad_x, ry1_w + 0.35 * pad_y, "left", "bottom"
            elif pos == "upper-left":
                tx, ty, ha, va = rx0_w + pad_x, ry1_w - pad_y, "left", "top"
            elif pos == "lower-left":
                tx, ty, ha, va = rx0_w + pad_x, ry0_w + pad_y, "left", "bottom"
            elif pos == "lower-right":
                tx, ty, ha, va = rx1_w - pad_x, ry0_w + pad_y, "right", "bottom"
            else:
                tx, ty, ha, va = rx1_w - pad_x, ry1_w - pad_y, "right", "top"

            t = ax_j.text(
                tx,
                ty,
                f"ROI-{i}",
                ha=ha,
                va=va,
                fontsize=9,
                color="k",
                zorder=6,
                fontweight="bold",
            )
            t.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.2))
            t.set_path_effects([pe.Stroke(linewidth=2.0, foreground="w"), pe.Normal()])

    ax_j.set_aspect("equal", adjustable="box")
    _add_north_arrow(ax_j, x=0.12, y=0.82, color="white")
    _add_scalebar_lonlat(ax_j, target_frac=0.2, color="white")

    # Flow direction (downstream) indicator for Juruá
    if False:
        # 以 ROI-1 -> ROI-4 的整体走向近似流向，并在图框左下入口处绘制箭头
        try:
            xmin_j, xmax_j = ax_j.get_xlim()
            ymin_j, ymax_j = ax_j.get_ylim()

            if len(jurua_rois_wgs84) >= 4:
                r1 = jurua_rois_wgs84[0]
                r4 = jurua_rois_wgs84[3]
                cx1 = 0.5 * (float(r1[0]) + float(r1[1]))
                cy1 = 0.5 * (float(r1[2]) + float(r1[3]))
                cx4 = 0.5 * (float(r4[0]) + float(r4[1]))
                cy4 = 0.5 * (float(r4[2]) + float(r4[3]))
                vx = float(cx4 - cx1)
                vy = float(cy4 - cy1)
            else:
                vx, vy = 1.0, 0.5

            mag = float((vx * vx + vy * vy) ** 0.5)
            if mag <= 0:
                vx, vy = 1.0, 0.5
                mag = float((vx * vx + vy * vy) ** 0.5)

            vx /= mag
            vy /= mag

            # 入口处锚点：更靠边（左下角内缩更小）
            x0f = float(xmin_j) + 0.05 * float(xmax_j - xmin_j)
            y0f = float(ymin_j) + 0.06 * float(ymax_j - ymin_j)

            # 箭头长度：占短边约 16%
            L = 0.16 * min(float(xmax_j - xmin_j), float(ymax_j - ymin_j))
            x1f = float(x0f) + float(L) * float(vx)
            y1f = float(y0f) + float(L) * float(vy)

            _add_flow_arrow(
                ax_j,
                x0=x0f,
                y0=y0f,
                x1=x1f,
                y1=y1f,
                label="",
                color="white",
            )
        except Exception:
            pass
    
    if str(show_grid) == "on":
        ax_j.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    
    if str(show_axes) == "on":
        ax_j.set_xlabel("Longitude (°)", fontsize=10)
        ax_j.set_ylabel("Latitude (°)", fontsize=10)
    else:
        ax_j.set_xticks([])
        ax_j.set_yticks([])
    
    # Jurua 站点标签
    ax_j.text(
        0.5,
        0.98,
        "Juruá River",
        transform=ax_j.transAxes,
        fontsize=9,
        color="white",
        ha="center",
        va="top",
        bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=1),
        zorder=10,
    )
        
    _add_panel_letter(ax_j, "(d)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fig.1: Study areas (Yellow River + Jurua) with ROI boxes")
    parser.add_argument(
        "--mode",
        type=str,
        default="fig1_3row",
        choices=["jurua_only", "fig1_3row", "fig1_2x2"],
    )

    parser.add_argument("--jurua-site", type=str, default="Jurua-A")
    parser.add_argument("--jurua-year", type=int, default=2021)
    parser.add_argument("--jurua-background", type=str, default="")
    parser.add_argument("--roi", type=float, nargs=4, action="append", default=[])
    parser.add_argument("--roi-labels", type=str, default="auto", choices=["auto", "on", "off"], help="Show Jurua ROI labels")
    parser.add_argument(
        "--roi-label-pos",
        type=str,
        default="upper-right",
        choices=["upper-left", "upper-right", "lower-left", "lower-right", "upper-left-outside", "right-outside"],
        help="Jurua ROI label position",
    )

    parser.add_argument("--huanghe-year", type=int, default=2024)
    parser.add_argument("--huanghe-mask-level", type=int, default=4)
    parser.add_argument("--huanghe-a-shp", type=str, default="")
    parser.add_argument("--huanghe-b-shp", type=str, default="")
    parser.add_argument(
        "--huanghe-overview-basemap",
        type=str,
        default="naturalearth",
        choices=["naturalearth", "none"],
    )
    parser.add_argument("--huanghe-overview-basemap-zoom", type=int, default=5)
    parser.add_argument(
        "--huanghe-zoom-basemap",
        type=str,
        default="esri",
        choices=["esri", "none"],
    )
    parser.add_argument("--huanghe-zoom-basemap-zoom", type=int, default=11)
    parser.add_argument("--huanghe-overview-margin-lon", type=float, default=10.0)
    parser.add_argument("--huanghe-overview-margin-lat", type=float, default=5.0)
    parser.add_argument("--huanghe-zoom-margin-deg", type=float, default=0.002)

    parser.add_argument("--show-axes", type=str, default="on", choices=["on", "off"])
    parser.add_argument("--show-grid", type=str, default="on", choices=["on", "off"])

    parser.add_argument("--jurua-mask-level", type=int, default=1)
    parser.add_argument("--jurua-water-mask", type=str, default="")
    parser.add_argument("--jurua-scalebar-km", type=float, default=20.0)

    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--preset", type=str, default="paper", choices=["", "paper"])
    parser.add_argument("--dpi", type=int, default=600)

    parser.add_argument("--bg-style", type=str, default="gray", choices=["rgb", "gray", "desat"])
    parser.add_argument("--bg-sat", type=float, default=0.20)
    parser.add_argument("--bg-alpha", type=float, default=0.95)
    parser.add_argument("--bg-black-threshold", type=float, default=0.04)

    args = parser.parse_args()

    jurua_site = str(args.jurua_site)
    jurua_year = int(args.jurua_year)

    jurua_bg: Path | None
    if str(args.mode) == "jurua_only":
        if args.jurua_background:
            jurua_bg = Path(args.jurua_background)
        else:
            bg = _default_background_path(jurua_site, jurua_year)
            if bg is None:
                raise FileNotFoundError("未找到 Jurua 可用背景影像。请用 --jurua-background 指定。")
            jurua_bg = Path(bg)
        if not jurua_bg.exists():
            raise FileNotFoundError(jurua_bg)
    else:
        jurua_bg = Path(args.jurua_background) if args.jurua_background else None

    jurua_rois = [tuple(float(v) for v in r) for r in args.roi] if args.roi else []

    if args.huanghe_a_shp:
        shp_a = Path(args.huanghe_a_shp)
    else:
        shp_a = _PROJECT_ROOT / "data" / "GIS" / "HuangHe-A.shp"
    if args.huanghe_b_shp:
        shp_b = Path(args.huanghe_b_shp)
    else:
        shp_b = _PROJECT_ROOT / "data" / "GIS" / "HuangHe-B.shp"

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = _PROJECT_ROOT / "results" / "figures" / "paper" / "Fig1_Study_Areas.png"

    if str(args.mode) == "jurua_only":
        if not jurua_rois:
            raise ValueError("请提供 Jurua 的 4 个 ROI：重复使用 --roi xmin xmax ymin ymax")
        plot_fig1_jurua_overview(
            site=jurua_site,
            year=jurua_year,
            background_path=jurua_bg,
            rois=jurua_rois,
            out_path=out_path,
            preset=str(args.preset),
            dpi=int(args.dpi),
            bg_style=str(args.bg_style),
            bg_sat=float(args.bg_sat),
            bg_alpha=float(args.bg_alpha),
            bg_black_threshold=float(args.bg_black_threshold),
            roi_color="#00e5ff",
            roi_lw=1.8,
            roi_labels=str(args.roi_labels),
            roi_label_pos=str(args.roi_label_pos),
        )
        return

    if len(jurua_rois) != 4:
        raise ValueError("Fig.1 需要提供 Jurua 的 4 个 ROI：重复使用 --roi xmin xmax ymin ymax")

    plot_fig1_study_areas(
        jurua_site=jurua_site,
        jurua_year=jurua_year,
        jurua_background=jurua_bg,
        jurua_rois=jurua_rois,
        jurua_roi_labels=str(args.roi_labels),
        jurua_roi_label_pos=str(args.roi_label_pos),
        huanghe_year=int(args.huanghe_year),
        huanghe_mask_level=int(args.huanghe_mask_level),
        huanghe_a_shp=Path(shp_a),
        huanghe_b_shp=Path(shp_b),
        huanghe_overview_basemap=str(args.huanghe_overview_basemap),
        huanghe_overview_basemap_zoom=int(args.huanghe_overview_basemap_zoom),
        huanghe_zoom_basemap=str(args.huanghe_zoom_basemap),
        huanghe_zoom_basemap_zoom=int(args.huanghe_zoom_basemap_zoom),
        huanghe_overview_margin_lon=float(args.huanghe_overview_margin_lon),
        huanghe_overview_margin_lat=float(args.huanghe_overview_margin_lat),
        huanghe_zoom_margin_deg=float(args.huanghe_zoom_margin_deg),
        show_axes=str(args.show_axes),
        show_grid=str(args.show_grid),
        jurua_mask_level=int(args.jurua_mask_level),
        jurua_water_mask=(Path(str(args.jurua_water_mask)) if str(args.jurua_water_mask) else None),
        jurua_scalebar_km=float(args.jurua_scalebar_km),
        out_path=out_path,
        preset=str(args.preset),
        dpi=int(args.dpi),
        bg_alpha=float(args.bg_alpha),
        bg_black_threshold=float(args.bg_black_threshold),
        bg_style=str(args.bg_style),
        bg_sat=float(args.bg_sat),
    )


if __name__ == "__main__":
    main()
