import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Circle

from src.analysis.plot_preset import setup_preset, get_paper_figsize


def _add_panel_label(ax: plt.Axes, text: str) -> None:
    ax.text(
        0.02,
        0.98,
        str(text),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.25),
        zorder=50,
    )


def _river_polygon(x: np.ndarray, w: np.ndarray, y0: float = 0.5) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    y_top = y0 + 0.5 * w
    y_bot = y0 - 0.5 * w
    poly = np.vstack(
        [
            np.column_stack([x, y_top]),
            np.column_stack([x[::-1], y_bot[::-1]]),
        ]
    )
    return poly


def _draw_width_driven(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    x = np.linspace(0.05, 0.95, 400)
    w = 0.24 + 0.10 * np.sin(2 * np.pi * (x - 0.05) / 0.45) + 0.06 * np.sin(2 * np.pi * (x - 0.05) / 0.22)
    w = np.clip(w, 0.14, 0.40)
    poly = _river_polygon(x, w, y0=0.50)
    ax.add_patch(Polygon(poly, closed=True, facecolor="0.92", edgecolor="0.25", linewidth=1.0, zorder=1))

    y_mid = 0.50 + 0.05 * np.sin(2 * np.pi * x / 0.6)
    ax.plot(x, y_mid, color="0.15", linewidth=1.0, alpha=0.65, zorder=2)

    bar_x = np.array([0.22, 0.38, 0.58, 0.76])
    bar_y = np.interp(bar_x, x, y_mid) + np.array([0.03, -0.02, 0.02, -0.03])
    for bx, by in zip(bar_x, bar_y):
        ax.add_patch(Circle((float(bx), float(by)), 0.030, facecolor="0.70", edgecolor="0.35", linewidth=0.8, zorder=3))

        ax.add_patch(
            FancyArrowPatch(
                (float(bx), float(by)),
                (float(bx) + 0.07, float(by) + 0.09),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=0.9,
                color="tab:blue",
                alpha=0.85,
                zorder=4,
            )
        )

        ax.add_patch(
            FancyArrowPatch(
                (float(bx) + 0.02, float(by) + 0.07),
                (float(bx) + 0.12, float(by) + 0.07),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=0.9,
                color="tab:red",
                alpha=0.85,
                zorder=4,
            )
        )

    ax.text(
        0.06,
        0.92,
        "Width-driven regime\n(bar-push; km-scale width oscillations)",
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2),
        zorder=10,
    )

    ax.text(
        0.06,
        0.12,
        "Corridor-scale multi-thread\norganization possible ($N_{\\mathrm{eff}}$ > 1)",
        ha="left",
        va="bottom",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=1.0),
        zorder=10,
    )


def _draw_curvature_responsive(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    x = np.linspace(0.05, 0.95, 400)
    y_mid = 0.50 + 0.12 * np.sin(2 * np.pi * (x - 0.05) / 0.75)
    w = np.full_like(x, 0.18)
    y_top = y_mid + 0.5 * w
    y_bot = y_mid - 0.5 * w
    poly = np.vstack(
        [
            np.column_stack([x, y_top]),
            np.column_stack([x[::-1], y_bot[::-1]]),
        ]
    )

    ax.add_patch(Polygon(poly, closed=True, facecolor="0.92", edgecolor="0.25", linewidth=1.0, zorder=1))
    ax.plot(x, y_mid, color="0.15", linewidth=1.0, alpha=0.65, zorder=2)

    wall_off = 0.06
    ax.plot(x, y_top + wall_off, color="0.2", linewidth=1.2, alpha=0.55, zorder=0)
    ax.plot(x, y_bot - wall_off, color="0.2", linewidth=1.2, alpha=0.55, zorder=0)

    hs_x = np.array([0.25, 0.55, 0.82])
    hs_y = np.interp(hs_x, x, y_mid)
    for bx, by in zip(hs_x, hs_y):
        ax.add_patch(
            FancyArrowPatch(
                (float(bx), float(by)),
                (float(bx), float(by) + 0.12),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.0,
                color="tab:red",
                alpha=0.9,
                zorder=4,
            )
        )

    ax.text(
        0.06,
        0.92,
        "Curvature-responsive regime\n(confined corridor; localized hotspots)",
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2),
        zorder=10,
    )

    ax.text(
        0.06,
        0.12,
        "Width-oscillation variability muted;\n$N_{\\mathrm{eff}}$ ≈ 1 at corridor scale",
        ha="left",
        va="bottom",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.70, pad=1.0),
        zorder=10,
    )


def plot_fig9(out_path: Path, *, preset: str = "paper", dpi: int = 600) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    setup_preset(str(preset), int(dpi))

    fig = plt.figure(figsize=get_paper_figsize(190, 92), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, wspace=0.02)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    _draw_width_driven(ax1)
    _draw_curvature_responsive(ax2)

    _add_panel_label(ax1, "(a)")
    _add_panel_label(ax2, "(b)")

    fig.savefig(out_path, dpi=int(dpi))
    fig.savefig(out_path.with_suffix(".pdf"), format="pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=False, default="")
    parser.add_argument("--preset", type=str, default="paper")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    if args.out:
        out_path = Path(args.out)
    else:
        root = Path(__file__).resolve().parents[2]
        out_path = root / "results" / "figures" / "paper" / "Fig9_Conceptual.png"

    plot_fig9(out_path, preset=str(args.preset), dpi=int(args.dpi))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
