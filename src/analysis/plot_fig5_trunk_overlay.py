from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.analysis.quantitative_relationships import analyze_trunk_level_relationships
from src.analysis.plot_preset import setup_preset, get_paper_figsize


def _iter_lines_from_vector(path: Path) -> Iterable[Tuple[str, np.ndarray]]:
    try:
        import fiona
    except Exception as e:
        raise ImportError("Missing dependency fiona: please install before running.") from e

    try:
        from shapely.geometry import LineString, MultiLineString, shape
    except Exception as e:
        raise ImportError("Missing dependency shapely: please install before running.") from e

    with fiona.open(path) as src:
        for idx, feat in enumerate(src):
            geom = feat.get("geometry")
            if geom is None:
                continue
            shp = shape(geom)
            lines: List[LineString] = []
            if isinstance(shp, LineString):
                lines = [shp]
            elif isinstance(shp, MultiLineString):
                lines = list(shp.geoms)
            if not lines:
                continue

            props = feat.get("properties", {}) or feat
            link_id = (
                str(props.get("id"))
                if props.get("id") is not None
                else (str(props.get("link_id")) if props.get("link_id") is not None else str(idx))
            )

            for li, line in enumerate(lines):
                lid = f"{link_id}_{li}" if li > 0 else link_id
                coords = np.asarray(line.coords, dtype=float)
                if coords.ndim != 2 or coords.shape[0] < 2:
                    continue
                yield lid, coords


def _sorted_trunk_ids(trunk_links: Dict[str, List[str]]) -> List[str]:
    def key(tid: str) -> int:
        try:
            return int(str(tid).split("_")[-1])
        except Exception:
            return 10**9

    return sorted(list(trunk_links.keys()), key=key)


def _read_basemap(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    try:
        import rasterio
    except Exception as e:
        raise ImportError("Missing dependency rasterio: please install before running.") from e

    with rasterio.open(path) as src:
        arr = src.read()
        b = src.bounds
        extent = (float(b.left), float(b.right), float(b.bottom), float(b.top))

    if arr.ndim == 2:
        img = arr
    else:
        if arr.shape[0] >= 3:
            rgb = np.transpose(arr[:3, :, :], (1, 2, 0)).astype(float)
            q2 = np.nanpercentile(rgb, 2)
            q98 = np.nanpercentile(rgb, 98)
            if np.isfinite(q2) and np.isfinite(q98) and q98 > q2:
                rgb = (rgb - q2) / (q98 - q2)
            img = np.clip(rgb, 0.0, 1.0)
        else:
            img = arr[0]

    return img, extent


def _plot_segments(ax, segments: List[np.ndarray], color: str, lw: float, alpha: float) -> None:
    try:
        from matplotlib.collections import LineCollection
    except Exception as e:
        raise ImportError("Missing dependency matplotlib: please install before running.") from e

    if not segments:
        return
    lc = LineCollection(segments, colors=color, linewidths=lw, alpha=alpha)
    ax.add_collection(lc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=str, default="HuangHe-B")
    parser.add_argument("--mask-level", type=int, default=4)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--npz", type=str, default="")
    parser.add_argument("--links-shp", type=str, default="")
    parser.add_argument("--basemap-tif", type=str, default="")
    parser.add_argument("--k-trunks", type=int, default=4)
    parser.add_argument("--endpoint-tol-m", type=float, default=80.0)
    parser.add_argument("--weight-by", type=str, default="length_B")
    parser.add_argument(
        "--min-trunk-lengths-m",
        type=float,
        nargs="+",
        default=[2000.0, 3000.0, 5000.0],
    )
    parser.add_argument("--out", type=str, default="")
    
    # Preset args
    parser.add_argument("--preset", type=str, default="", choices=["", "paper"], help="Style preset")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figure")
    
    args = parser.parse_args()
    
    # Apply preset
    setup_preset(args.preset, args.dpi)

    root = Path(__file__).resolve().parents[2]

    site = str(args.site)
    mask = int(args.mask_level)
    year = int(args.year)

    npz_path = Path(args.npz) if args.npz else root / "results" / "PostprocessedPIV" / site / f"{site}_mask{mask}_link_sBCMn_flat_step20_metric_v2.npz"
    links_shp = Path(args.links_shp) if args.links_shp else root / "results" / "RivGraph" / site / f"mask{mask}" / f"{site}_mask{mask}_links.shp"

    if args.basemap_tif:
        basemap_tif = Path(args.basemap_tif)
    else:
        basemap_tif = root / "data" / "GEOTIFFS" / site / f"mask{mask}" / f"{site}_{year}_01-01_12-31_mask{mask}.tif"

    if args.out:
        out_path = Path(args.out)
    else:
        out_dir = root / "results" / "figures" / "trunk_overlay"
        out_dir.mkdir(parents=True, exist_ok=True)
        thr_tag = "_".join([str(int(x)) for x in args.min_trunk_lengths_m])
        out_path = out_dir / f"Fig5_{site}_mask{mask}_year{year}_trunk_overlay_{thr_tag}.png"

    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    if not links_shp.exists():
        raise FileNotFoundError(links_shp)

    all_segments: Dict[str, np.ndarray] = {}
    for lid, coords in _iter_lines_from_vector(links_shp):
        all_segments[str(lid)] = coords

    img = None
    extent = None
    if basemap_tif.exists():
        img, extent = _read_basemap(basemap_tif)

    thresholds = [float(x) for x in args.min_trunk_lengths_m]
    ncol = len(thresholds)

    # Determine figsize based on preset
    if args.preset == "paper":
        if ncol == 1:
            # Single column width (90mm), height square-ish
            figsize = get_paper_figsize(90, 90)
        else:
            # Full width (190mm), height proportional to aspect ratio of subplots
            # Assuming 3 subplots row -> 190mm width, each ~60mm wide. 
            # Height maybe 60mm?
            figsize = get_paper_figsize(190, 190 / ncol * 1.0) # aspect ratio 1:1 per subplot
    else:
        figsize = (6.0 * ncol, 6.0)

    fig, axes = plt.subplots(1, ncol, figsize=figsize, constrained_layout=True)
    if ncol == 1:
        axes = [axes]

    base_segs = list(all_segments.values())

    for ax, thr in zip(axes, thresholds):
        res = analyze_trunk_level_relationships(
            npz_path,
            k_trunks=int(args.k_trunks),
            endpoint_tol_m=float(args.endpoint_tol_m),
            weight_by=str(args.weight_by),
            min_trunk_length_m=float(thr),
        )

        if img is not None and extent is not None:
            if img.ndim == 3:
                ax.imshow(img, extent=extent, origin="upper")
            else:
                ax.imshow(img, extent=extent, origin="upper", cmap="gray")

            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[3], extent[2])

        _plot_segments(ax, base_segs, color="0.6", lw=0.6, alpha=0.6)

        trunk_ids = _sorted_trunk_ids(res.trunk_links)
        colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple"]

        for i, tid in enumerate(trunk_ids):
            lids = set([str(x) for x in res.trunk_links.get(tid, [])])
            segs = [all_segments[l] for l in lids if l in all_segments]
            _plot_segments(ax, segs, color=colors[i % len(colors)], lw=2.5, alpha=0.95)

        k_eff = int(res.diagnostics.get("k_trunks", 0))
        ax.set_title(f"min_trunk_length={int(thr)} m | k_trunks={k_eff}")
        ax.set_aspect("equal", adjustable="box")

        if img is None or extent is None:
            ax.autoscale(enable=True)

    if args.preset != "paper":
        fig.suptitle(f"{site} mask{mask} trunk overlay", fontsize=14)
        
    fig.savefig(out_path, dpi=args.dpi)

    print(f"saved: {out_path}")
    print(f"npz: {npz_path}")
    print(f"links: {links_shp}")
    print(f"basemap: {basemap_tif if basemap_tif.exists() else '(none)'}")


if __name__ == "__main__":
    main()
