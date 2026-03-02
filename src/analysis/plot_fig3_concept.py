"""
plot_fig3_concept.py
====================
Generate **Fig. 3**: 2-panel *conceptual* illustration of the
trunk-aggregation algorithm for Computers & Geosciences.

Panel (a): Union-Find spatial clustering — overview + zoomed inset
           showing raw endpoints merged into clustered nodes.
Panel (b): PCA-based DAG construction — directed edges with PCA
           principal axis; nodes colored by downstream position.

Data source: YR-A Mask 4 real skeleton data.

JoH-style-rule compliant:
  - Times New Roman, >= 7 pt
  - 600 dpi, PNG output
  - 190 mm full width
  - Colorblind-friendly palette

Usage:
    conda activate riverpiv
    python src\\analysis\\plot_fig3_concept.py
    python src\\analysis\\plot_fig3_concept.py --preset paper
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np

from src.config import PROJECT_ROOT as CFG_ROOT
from src.analysis.plot_preset import setup_preset, get_paper_figsize
from src.analysis.quantitative_relationships import (
    _as_1d_float,
    _cluster_points_xy_m,
    _meters_per_degree,
)

# ── paths ───────────────────────────────────────────────────────────────
SITE = "HuangHe-A"
MASK = 4
NPZ = CFG_ROOT / "results" / "PostprocessedPIV" / SITE / (
    f"{SITE}_mask{MASK}_link_sBCMn_flat_step20_metric_v2.npz"
)
OUT = CFG_ROOT / "results" / "figures" / "paper" / "Fig3_Concept.png"

def main():
    preset = args.preset
    setup_preset(preset)

    data = dict(np.load(str(NPZ), allow_pickle=True))

    link_ids = np.asarray(data["link_ids"], dtype=object).astype(str)
    link_index = np.asarray(data["link_index"], dtype=int)
    x_all = _as_1d_float(data["x"])
    y_all = _as_1d_float(data["y"])
    s_all = _as_1d_float(data["s"])

    n_links = int(link_ids.size)
    x0 = np.full(n_links, np.nan)
    y0 = np.full(n_links, np.nan)
    x1 = np.full(n_links, np.nan)
    y1 = np.full(n_links, np.nan)
    link_len = np.full(n_links, np.nan)

    for i in range(n_links):
        m = link_index == i
        if not np.any(m):
            continue
        xs, ys = x_all[m], y_all[m]
        x0[i], y0[i] = float(xs[0]), float(ys[0])
        x1[i], y1[i] = float(xs[-1]), float(ys[-1])
        ss = s_all[m]
        if np.isfinite(ss).any():
            link_len[i] = float(np.nanmax(ss) - np.nanmin(ss))

    # ── Union-Find clustering ───────────────────────────────────────────
    lat_ref = float(np.nanmean(
        np.concatenate([y0[np.isfinite(y0)], y1[np.isfinite(y1)]])
    ))
    m_lon, m_lat = _meters_per_degree(lat_ref)

    pts_x_m = np.concatenate([x0, x1]) * m_lon
    pts_y_m = np.concatenate([y0, y1]) * m_lat

    ids = _cluster_points_xy_m(
        np.nan_to_num(pts_x_m, nan=1e30),
        np.nan_to_num(pts_y_m, nan=1e30),
        tol_m=80.0,
    )
    start_node = ids[:n_links]
    end_node = ids[n_links:]
    n_nodes = int(np.max(ids)) + 1

    # Raw endpoint coords in degrees
    ep_x_raw = np.concatenate([x0, x1])
    ep_y_raw = np.concatenate([y0, y1])

    # Clustered node centroids in degrees
    node_x = np.full(n_nodes, np.nan)
    node_y = np.full(n_nodes, np.nan)
    node_count = np.zeros(n_nodes, dtype=int)  # endpoints per cluster
    for nid in range(n_nodes):
        mm = ids == nid
        if np.any(mm):
            valid = np.isfinite(ep_x_raw[mm]) & np.isfinite(ep_y_raw[mm])
            node_x[nid] = float(np.nanmean(ep_x_raw[mm]))
            node_y[nid] = float(np.nanmean(ep_y_raw[mm]))
            node_count[nid] = int(np.sum(valid))

    # PCA axis
    finite = np.isfinite(node_x) & np.isfinite(node_y)
    XY_m = np.vstack([node_x[finite] * m_lon, node_y[finite] * m_lat]).T
    mu = np.mean(XY_m, axis=0)
    C_cov = np.cov((XY_m - mu).T)
    evals, evecs = np.linalg.eigh(C_cov)
    axis = evecs[:, int(np.argmax(evals))]
    node_p = (np.vstack([
        node_x * m_lon - mu[0],
        node_y * m_lat - mu[1],
    ]).T @ axis).astype(float)

    # ── Figure ──────────────────────────────────────────────────────────
    width_in, fig_h = get_paper_figsize(190, aspect_ratio=2.2)
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(width_in, fig_h),
        gridspec_kw={"wspace": 0.25},
    )

    # ════════════════════════════════════════════════════════════════════
    # Panel (a): Union-Find Spatial Clustering
    # ════════════════════════════════════════════════════════════════════
    ax_a.set_title("(a)", fontweight="bold", loc="left")

    # Draw link centerlines (thin grey)
    for i in range(n_links):
        m = link_index == i
        if not np.any(m):
            continue
        ax_a.plot(x_all[m], y_all[m], color="#c0c0c0", lw=0.3, alpha=0.6)

    # Draw raw endpoints (small, grey)
    valid_ep = np.isfinite(ep_x_raw) & np.isfinite(ep_y_raw)
    ax_a.scatter(ep_x_raw[valid_ep], ep_y_raw[valid_ep],
                 s=3, c="#aaaaaa", zorder=3, alpha=0.4)

    # Draw clustered nodes — sized by cluster membership count
    # Use Tableau-10 colorblind-friendly palette
    cmap = plt.cm.tab10
    multi_mask = (node_count > 1) & finite
    single_mask = (node_count == 1) & finite

    # Single-endpoint clusters (small grey diamonds)
    ax_a.scatter(node_x[single_mask], node_y[single_mask],
                 s=6, c="#888888", marker="D", zorder=4,
                 edgecolors="white", linewidths=0.2, alpha=0.6)

    # Multi-endpoint clusters (colored, sized by count)
    multi_ids = np.where(multi_mask)[0]
    for idx, nid in enumerate(multi_ids):
        col = cmap(idx % 10)
        sz = max(15, min(60, node_count[nid] * 8))
        ax_a.scatter(node_x[nid], node_y[nid], s=sz, c=[col],
                     marker="D", zorder=5, edgecolors="black",
                     linewidths=0.4)

    ax_a.set_xlabel("Longitude (°)")
    ax_a.set_ylabel("Latitude (°)")
    ax_a.set_aspect("equal")

    # ── Zoomed inset for panel (a) ──────────────────────────────────────
    # Pick a junction-rich area (near the main bend, ~113.20-113.24°E)
    zoom_xlim = (113.18, 113.24)
    zoom_ylim = (34.90, 34.94)

    axins = ax_a.inset_axes([0.05, 0.5, 0.4, 0.4])  # upper-left inset
    # Draw links in inset
    for i in range(n_links):
        m = link_index == i
        if not np.any(m):
            continue
        xi, yi = x_all[m], y_all[m]
        # Only draw if overlapping zoom box
        if (np.nanmax(xi) < zoom_xlim[0] or np.nanmin(xi) > zoom_xlim[1] or
                np.nanmax(yi) < zoom_ylim[0] or np.nanmin(yi) > zoom_ylim[1]):
            continue
        axins.plot(xi, yi, color="#c0c0c0", lw=0.5, alpha=0.8)

    # Raw endpoints in zoom
    in_zoom = (valid_ep &
               (ep_x_raw >= zoom_xlim[0]) & (ep_x_raw <= zoom_xlim[1]) &
               (ep_y_raw >= zoom_ylim[0]) & (ep_y_raw <= zoom_ylim[1]))
    axins.scatter(ep_x_raw[in_zoom], ep_y_raw[in_zoom],
                  s=8, c="#aaaaaa", zorder=3, alpha=0.5)

    # Multi-endpoint clusters in zoom — draw connecting lines from
    # raw endpoints to cluster centroid
    for idx, nid in enumerate(multi_ids):
        cx, cy = node_x[nid], node_y[nid]
        if not (zoom_xlim[0] <= cx <= zoom_xlim[1] and
                zoom_ylim[0] <= cy <= zoom_ylim[1]):
            continue
        col = cmap(idx % 10)
        # Find raw endpoints belonging to this cluster
        mm = ids == nid
        ex = ep_x_raw[mm]
        ey = ep_y_raw[mm]
        for j in range(len(ex)):
            if np.isfinite(ex[j]) and np.isfinite(ey[j]):
                axins.plot([ex[j], cx], [ey[j], cy],
                           color=col, lw=0.8, alpha=0.6, zorder=4)
                axins.scatter(ex[j], ey[j], s=12, c=[col], zorder=5,
                              edgecolors="none", alpha=0.7)
        sz = max(20, min(80, node_count[nid] * 10))
        axins.scatter(cx, cy, s=sz, c=[col], marker="D", zorder=6,
                      edgecolors="black", linewidths=0.5)

    axins.set_xlim(zoom_xlim)
    axins.set_ylim(zoom_ylim)
    axins.set_aspect("equal")
    axins.tick_params(labelsize=6)
    axins.set_xlabel("")
    axins.set_ylabel("")

    # Draw zoom indicator rectangle on main axes
    rect = mpatches.Rectangle(
        (zoom_xlim[0], zoom_ylim[0]),
        zoom_xlim[1] - zoom_xlim[0],
        zoom_ylim[1] - zoom_ylim[0],
        linewidth=0.8, edgecolor="black", facecolor="none",
        linestyle="--", zorder=10,
    )
    ax_a.add_patch(rect)

    # Legend for panel (a)
    raw_h = mlines.Line2D([], [], color="#aaaaaa", marker="o",
                          linestyle="None", markersize=3,
                          label="Raw endpoints")
    merge_h = mlines.Line2D([], [], color="tab:blue", marker="D",
                            linestyle="None", markersize=5,
                            markeredgecolor="black", markeredgewidth=0.4,
                            label="Clustered node ($\\delta$ = 80 m)")
    ax_a.legend(handles=[raw_h, merge_h], loc="lower right",
                framealpha=0.9, edgecolor="none")

    # ════════════════════════════════════════════════════════════════════
    # Panel (b): PCA-based DAG & Trunk Extraction
    # ════════════════════════════════════════════════════════════════════
    ax_b.set_title("(b)", fontweight="bold", loc="left")

    # Draw link centerlines (thin, very light grey)
    for i in range(n_links):
        m = link_index == i
        if not np.any(m):
            continue
        ax_b.plot(x_all[m], y_all[m], color="#e0e0e0", lw=0.3, alpha=0.5)

    # Draw a representative subset of DAG directed arrows (every 3rd link)
    # to avoid overcrowding
    arrow_indices = list(range(0, n_links, 3))
    for i in arrow_indices:
        u, v = int(start_node[i]), int(end_node[i])
        if not (np.isfinite(node_x[u]) and np.isfinite(node_x[v])):
            continue
        if not (np.isfinite(node_p[u]) and np.isfinite(node_p[v])):
            continue
        pu, pv = float(node_p[u]), float(node_p[v])
        if abs(pv - pu) < 1e-9:
            continue
        # Orient edge along PCA direction
        uu, vv = (u, v) if pu < pv else (v, u)
        dx = node_x[vv] - node_x[uu]
        dy = node_y[vv] - node_y[uu]
        ax_b.annotate(
            "", xy=(node_x[vv], node_y[vv]),
            xytext=(node_x[uu], node_y[uu]),
            arrowprops=dict(arrowstyle="-|>", color="#999999", lw=0.5,
                            mutation_scale=7, shrinkA=1, shrinkB=1),
            zorder=3,
        )

    # Clustered nodes colored by PCA position (colorblind-friendly: viridis)
    valid_p = np.isfinite(node_p) & finite
    sc = ax_b.scatter(
        node_x[valid_p], node_y[valid_p],
        c=node_p[valid_p], cmap="viridis", s=16, zorder=5,
        edgecolors="white", linewidths=0.3,
    )
    cb = fig.colorbar(sc, ax=ax_b, shrink=0.4, pad=0.02, aspect=15)
    cb.set_label("PCA position $p_j$ (m)")

    # Draw PCA principal axis as a dashed line
    cx_deg = float(np.nanmean(node_x[finite]))
    cy_deg = float(np.nanmean(node_y[finite]))
    axis_deg = axis.copy()
    axis_deg[0] /= m_lon
    axis_deg[1] /= m_lat
    axis_deg /= np.linalg.norm(axis_deg)
    span = 0.18  # degrees
    ax_b.plot(
        [cx_deg - span * axis_deg[0], cx_deg + span * axis_deg[0]],
        [cy_deg - span * axis_deg[1], cy_deg + span * axis_deg[1]],
        color="black", lw=1.5, ls="--", zorder=7,
    )
    # Arrow tip for downstream direction
    ax_b.annotate(
        "$\\mathbf{e}_1$",
        xy=(cx_deg + span * axis_deg[0],
            cy_deg + span * axis_deg[1]),
        xytext=(cx_deg + 0.7 * span * axis_deg[0],
                cy_deg + 0.7 * span * axis_deg[1]),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5),
        fontsize=plt.rcParams["axes.labelsize"], fontweight="bold",
        zorder=8,
    )

    # Legend for panel (b)
    dag_h = mlines.Line2D([], [], color="#999999", marker=">",
                          linestyle="-", markersize=4, lw=0.5,
                          label="DAG directed edge")
    pca_h = mlines.Line2D([], [], color="black", linestyle="--", lw=1.5,
                          label="PCA axis ($\\mathbf{e}_1$)")
    ax_b.legend(handles=[dag_h, pca_h], loc="lower right",
                framealpha=0.9, edgecolor="none")
    ax_b.set_xlabel("Longitude (°)")
    ax_b.set_ylabel("Latitude (°)")
    ax_b.set_aspect("equal")

    # ── Save ────────────────────────────────────────────────────────────
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT), dpi=600, bbox_inches="tight",
                pad_inches=0.05, facecolor="white")
    print(f"[OK] Saved Fig. 3 concept figure → {OUT}")
    print(f"     Size: {width_in:.2f} × {fig_h:.2f} in @ 600 dpi")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="paper",
                        help="Preset: 'paper' (default)")
    args = parser.parse_args()
    main()
