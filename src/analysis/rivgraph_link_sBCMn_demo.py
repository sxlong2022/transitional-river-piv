"""Demo: Build sub-channel s-B-C-Mn profiles from PIV grid and RivGraph link geometry profiles and visualize.

Usage example (from project root):

    python -m src.analysis.rivgraph_link_sBCMn_demo \
        --site Jurua-A \
        --mask-level 1 \
        --mask-raster path/to/water_mask.tif \
        --links-vector path/to/rivgraph_links.gpkg

Note:
- Requires Step 4A strict affine result jurua_mask{mask}_multitilt_georef_step4a_strict.npz to exist;
- Mask raster and RivGraph links should be in the same projection as the reference mask used in Step 4A.
"""

from __future__ import annotations

from pathlib import Path

import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.config import PROJECT_ROOT
from src.postprocessing.postprocess import get_postprocessed_dir
from src.morphodynamics.rivgraph_link_profiles import compute_link_profiles
from src.morphodynamics.coupling import add_Mn_to_link_profiles
from src.morphodynamics.jurua_georef_multitilt import _choose_reference_mask
from src.analysis.link_sBCMn_io import export_link_sBCMn_to_npz


def _load_piv_georef_step4a(site: str, mask_level: int) -> dict:
    """Load Step 4A strict affine PIV results and return grid coordinates and velocity field."""

    out_dir = get_postprocessed_dir(PROJECT_ROOT, site)
    npz_path = out_dir / f"jurua_mask{mask_level}_multitilt_georef_step4a_strict.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Step 4A strict output not found: {npz_path}\n"
            "Please run first: python -m src.morphodynamics.jurua_georef_multitilt"
        )

    data = np.load(npz_path)
    X_grid = data["X_geo"]
    Y_grid = data["Y_geo"]
    U_grid = data["u_m_per_year"]
    V_grid = data["v_m_per_year"]

    return {
        "X_grid": X_grid,
        "Y_grid": Y_grid,
        "U_grid": U_grid,
        "V_grid": V_grid,
    }


def _plot_link_profile(
    site: str,
    mask_level: int,
    link_id: str,
    s: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    Mn: np.ndarray,
    out_dir: Path,
) -> None:
    """Generate s-B-C-Mn profile plot for a single link."""

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 6))

    axes[0].plot(s, B, "b-", lw=1.0)
    axes[0].set_ylabel("B (m)")
    axes[0].set_title("Width B(s)")

    axes[1].plot(s, C, "g-", lw=1.0)
    axes[1].set_ylabel("C (1/m)")
    axes[1].set_title("Curvature C(s)")

    axes[2].plot(s, Mn, "k-", lw=1.0)
    axes[2].set_xlabel("s along link (m)")
    axes[2].set_ylabel("Mn (m/yr)")
    axes[2].set_title("Normal migration rate Mn(s)")

    fig.suptitle(f"{site} Mask{mask_level} link {link_id}: s–B–C–Mn")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_png = out_dir / f"{site}_mask{mask_level}_link{link_id}_sBCMn.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print("Saved link profile plot to:", out_png)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Demo: build s–B–C–Mn profiles along RivGraph links using georeferenced PIV "
            "(Step 4A) and link geometries."
        ),
    )
    parser.add_argument("--site", default="Jurua-A", help="Site name, e.g., 'Jurua-A'")
    parser.add_argument("--mask-level", type=int, default=1, help="Mask level, e.g., 1 for Mask1")
    parser.add_argument(
        "--ref-year",
        type=int,
        default=1987,
        help="Preferred year for selecting reference mask (falls back to first .tif in directory if not found)",
    )
    parser.add_argument(
        "--mask-raster",
        default=None,
        help=(
            "Binary water mask raster path, should be in the same projection as the PIV grid. "
            "If omitted and site='Jurua-A', automatically calls _choose_reference_mask to use the same mask as Step 4A."
        ),
    )
    parser.add_argument(
        "--links-vector",
        required=True,
        help="RivGraph link 矢量文件路径（如 .gpkg / .shp）。",
    )
    parser.add_argument(
        "--step-m",
        type=float,
        default=100.0,
        help="沿 link 的加密采样间距 (m)，需与其他剖面分析保持一致。",
    )
    parser.add_argument(
        "--n-links",
        type=int,
        default=5,
        help="最多绘制的 link 数量（按 link_id 排序后取前 n 条）。",
    )
    parser.add_argument(
        "--export-npz",
        action="store_true",
        help=(
            "若提供该开关，则将所有 link 的 s–B–C–Mn 扁平导出为 .npz，"
            "输出到 PostprocessedPIV/<site>/ 下。",
        ),
    )

    args = parser.parse_args()

    # 0. 若未显式提供掩膜，对 Jurua-A 自动选择 Step 4A 参考掩膜
    if args.mask_raster is None:
        if args.site == "Jurua-A":
            mask_raster_path = _choose_reference_mask(
                site=args.site,
                mask_level=args.mask_level,
                year=args.ref_year,
            )
            print("未指定 --mask-raster，自动使用 Step 4A 参考掩膜:", mask_raster_path)
        else:
            raise ValueError(
                "未提供 --mask-raster，且当前站点不是 Jurua-A，"
                "无法自动推断掩膜路径，请显式指定。",
            )
    else:
        mask_raster_path = args.mask_raster

    # 1. 加载 Step 4A 严格仿射 PIV 结果
    piv = _load_piv_georef_step4a(site=args.site, mask_level=args.mask_level)

    # 2. 计算 RivGraph link 的几何剖面（s, x, y, B, C）
    link_geom = compute_link_profiles(
        mask_raster_path=mask_raster_path,
        links_vector_path=args.links_vector,
        step_m=args.step_m,
    )

    if not link_geom:
        raise RuntimeError("从 RivGraph 矢量中未获得任何 link 剖面，请检查输入数据。")

    # 3. 在几何剖面上叠加 PIV 法向迁移率 Mn(s)
    link_sBCMn = add_Mn_to_link_profiles(
        link_profiles=link_geom,
        X_grid=piv["X_grid"],
        Y_grid=piv["Y_grid"],
        U_grid=piv["U_grid"],
        V_grid=piv["V_grid"],
    )

    # 4. 为若干条 link 生成 s–B–C–Mn 剖面图
    out_dir = get_postprocessed_dir(PROJECT_ROOT, args.site)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 若开启导出开关，则扁平化导出 .npz
    if args.export_npz:
        flat_npz = out_dir / f"{args.site}_mask{args.mask_level}_link_sBCMn_flat.npz"
        export_link_sBCMn_to_npz(
            link_profiles=link_sBCMn,
            site=args.site,
            mask_level=args.mask_level,
            step_m=args.step_m,
            out_path=flat_npz,
        )
        print("已导出扁平 s–B–C–Mn .npz 到:", flat_npz)

    link_ids = sorted(link_sBCMn.keys())
    if args.n_links > 0:
        link_ids = link_ids[: args.n_links]

    for link_id in link_ids:
        prof = link_sBCMn[link_id]
        s = prof["s"]
        B = prof["B"]
        C = prof["C"]
        Mn = prof["Mn"]

        _plot_link_profile(
            site=args.site,
            mask_level=args.mask_level,
            link_id=link_id,
            s=s,
            B=B,
            C=C,
            Mn=Mn,
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()
