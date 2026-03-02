from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.analysis.quantitative_relationships import analyze_link_level_relationships
from src.analysis.plot_preset import setup_preset, get_paper_figsize


def _binned_stats(
    x: np.ndarray,
    y: np.ndarray,
    bin_edges: np.ndarray,
    min_count: int = 5,
) -> dict[str, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.asarray(bin_edges, dtype=float)

    centers = 0.5 * (edges[:-1] + edges[1:])
    mean = np.full(centers.shape, np.nan, dtype=float)
    p25 = np.full(centers.shape, np.nan, dtype=float)
    p75 = np.full(centers.shape, np.nan, dtype=float)
    count = np.zeros(centers.shape, dtype=int)

    for i in range(centers.size):
        lo, hi = edges[i], edges[i + 1]
        m = (x >= lo) & (x < hi) if i < centers.size - 1 else (x >= lo) & (x <= hi)
        v = np.isfinite(x) & np.isfinite(y) & m
        yi = y[v]
        count[i] = int(yi.size)
        if yi.size < int(min_count):
            continue
        mean[i] = float(np.mean(yi))
        p25[i] = float(np.percentile(yi, 25))
        p75[i] = float(np.percentile(yi, 75))

    return {"centers": centers, "mean": mean, "p25": p25, "p75": p75, "count": count}


def _default_npz_path(site: str, mask_level: int) -> Path:
    return (
        _PROJECT_ROOT
        / "results"
        / "PostprocessedPIV"
        / site
        / f"{site}_mask{mask_level}_link_sBCMn_flat_step20_metric_v2.npz"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot link-level |Mn| vs |C| and |Mn| vs B scatter with binned means."
    )
    parser.add_argument("--site", type=str, required=True)
    parser.add_argument("--mask-level", type=int, required=True)
    parser.add_argument("--npz", type=str, default="", help="Override input NPZ path")
    parser.add_argument("--out", type=str, default="", help="Override output PNG path")

    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--min-arc-length", type=float, default=200.0)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--min-count-per-bin", type=int, default=5)
    parser.add_argument("--xscale-curv", type=str, default="log", choices=["linear", "log"])
    parser.add_argument("--log-clip-q-low", type=float, default=5.0)
    parser.add_argument("--log-clip-q-high", type=float, default=99.0)

    parser.add_argument("--scatter-alpha", type=float, default=0.15)
    parser.add_argument("--scatter-size", type=float, default=8.0)

    parser.add_argument("--preset", type=str, default="", choices=["", "paper"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--titles", type=str, default="auto", choices=["auto", "on", "off"])

    args = parser.parse_args()

    setup_preset(args.preset, args.dpi)

    if args.titles == "on":
        show_titles = True
    elif args.titles == "off":
        show_titles = False
    else:
        show_titles = args.preset != "paper"

    site = args.site
    mask = int(args.mask_level)

    npz_path = Path(args.npz) if args.npz else _default_npz_path(site, mask)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = _PROJECT_ROOT / "results" / "figures" / "scatter_binned"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"Fig8_{site}_mask{mask}_link_scatter_binned.png"

    res = analyze_link_level_relationships(
        npz_path,
        use_abs_mn=True,
        min_samples=int(args.min_samples),
        min_arc_length=float(args.min_arc_length) if args.min_arc_length is not None else None,
    )

    mtr = res.metrics
    xC_all = np.asarray(mtr["mean_abs_C"], dtype=float)
    yM_all = np.asarray(mtr["mean_abs_Mn"], dtype=float)
    xB_all = np.asarray(mtr["mean_B"], dtype=float)
    n_samples = np.asarray(mtr["n_samples"], dtype=float)
    arc_len = np.asarray(mtr["arc_length"], dtype=float)

    m_base = np.isfinite(xC_all) & np.isfinite(yM_all)
    m_base &= n_samples >= float(args.min_samples)
    if args.min_arc_length is not None:
        m_base &= np.isfinite(arc_len) & (arc_len >= float(args.min_arc_length))

    mC = m_base
    mB = m_base & np.isfinite(xB_all)

    xC = xC_all[mC]
    yC = yM_all[mC]
    xB = xB_all[mB]
    yB = yM_all[mB]

    if args.xscale_curv == "log":
        mpos = np.isfinite(xC) & (xC > 0)
        xC = xC[mpos]
        yC = yC[mpos]

    print(
        "LINK_FILTER_SUMMARY=",
        {
            "path": str(npz_path),
            "n_links_total": int(np.asarray(mtr["link_id"]).size),
            "n_links_used_C": int(np.sum(mC)),
            "n_links_used_B": int(np.sum(mB)),
            "min_samples": int(args.min_samples),
            "min_arc_length": float(args.min_arc_length) if args.min_arc_length is not None else None,
            "xscale_curv": str(args.xscale_curv),
        },
    )
    print("FITS=", {k: {"r2": v.r2, "n": v.n, "params": v.params} for k, v in res.fits.items()})

    if xC.size < 10 or xB.size < 10:
        print("Warning: too few points after filtering; check thresholds.")

    if args.preset == "paper":
        figsize = get_paper_figsize(190, 80)
    else:
        figsize = (14, 6)

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    ax = axes[0]
    ax.scatter(
        xC,
        yC,
        s=float(args.scatter_size),
        alpha=float(args.scatter_alpha),
        color="0.2",
        edgecolors="none",
    )
    if xC.size > 0:
        if args.xscale_curv == "log":
            # Use robust range to avoid a few near-zero curvature values stretching the axis.
            # We already filtered xC > 0 above when xscale_curv == 'log'.
            qlo = float(args.log_clip_q_low)
            qhi = float(args.log_clip_q_high)
            qlo = max(0.0, min(qlo, 50.0))
            qhi = max(50.0, min(qhi, 100.0))
            if qhi <= qlo:
                qhi = min(100.0, qlo + 50.0)
            lo = float(np.nanpercentile(xC, qlo))
            hi = float(np.nanpercentile(xC, qhi))
            lo = max(lo, float(np.nanmin(xC)))
            hi = min(hi, float(np.nanmax(xC)))
            if not np.isfinite(lo) or lo <= 0:
                lo = float(np.nanmin(xC[xC > 0]))
            if not np.isfinite(hi) or hi <= lo:
                hi = float(np.nanmax(xC))
            if not np.isfinite(hi) or hi <= lo:
                hi = lo * 1.01
            edges = np.logspace(np.log10(lo), np.log10(hi), int(args.n_bins) + 1)
            ax.set_xscale("log")
            ax.set_xlim(float(edges[0]), float(edges[-1]))
        else:
            edges = np.linspace(float(np.min(xC)), float(np.max(xC)), int(args.n_bins) + 1)

        st = _binned_stats(xC, yC, edges, min_count=int(args.min_count_per_bin))
        ax.plot(st["centers"], st["mean"], color="tab:blue", linewidth=2.0)
        ax.fill_between(st["centers"], st["p25"], st["p75"], color="tab:blue", alpha=0.2, linewidth=0)

    ax.set_xlabel(r"Mean $|C|$ (m$^{-1}$)")
    ax.set_ylabel(r"Mean $|M_{\mathrm{n}}|$ (m/yr)")
    ax.grid(True, linestyle=":", alpha=0.6)
    if show_titles:
        ax.set_title(r"$|M_{\mathrm{n}}|$ vs $|C|$")
    ax.text(0.02, 0.98, f"n={int(xC.size)}", transform=ax.transAxes, va="top")

    ax = axes[1]
    ax.scatter(
        xB,
        yB,
        s=float(args.scatter_size),
        alpha=float(args.scatter_alpha),
        color="0.2",
        edgecolors="none",
    )
    if xB.size > 0:
        edges = np.linspace(float(np.min(xB)), float(np.max(xB)), int(args.n_bins) + 1)
        st = _binned_stats(xB, yB, edges, min_count=int(args.min_count_per_bin))
        ax.plot(st["centers"], st["mean"], color="tab:orange", linewidth=2.0)
        ax.fill_between(st["centers"], st["p25"], st["p75"], color="tab:orange", alpha=0.2, linewidth=0)

    ax.set_xlabel(r"Mean $B$ (m)")
    ax.set_ylabel(r"Mean $|M_{\mathrm{n}}|$ (m/yr)")
    ax.grid(True, linestyle=":", alpha=0.6)
    if show_titles:
        ax.set_title(r"$|M_{\mathrm{n}}|$ vs $B$")
    ax.text(0.02, 0.98, f"n={int(xB.size)}", transform=ax.transAxes, va="top")

    if show_titles:
        fig.suptitle(
            f"{site} Mask {mask} | min_samples={int(args.min_samples)} | min_arc_length={float(args.min_arc_length)}",
            fontsize=12,
        )

    fig.savefig(out_path, dpi=int(args.dpi))
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
