"""Universal s–B–C–Mn profile calculation tool: supports arbitrary site/mask_level configurations.

This module provides:
- `compute_link_sBCMn_for_site`: high-level function that couples RivGraph link geometry profiles with PIV normal migration projection.
- CLI entry point: can be called directly from the command line.

Typical usage (from project root):

    python -m src.morphodynamics.link_sBCMn_pipeline \
        --site Jurua-A \
        --mask-level 1 \
        --links-vector results/RivGraph/Jurua-A/mask1/Jurua-A_mask1_links.shp \
        --piv-npz results/PostprocessedPIV/Jurua-A/jurua_mask1_multitilt_georef_step4a_strict.npz \
        --step-m 100 \
        --export-npz results/PostprocessedPIV/Jurua-A/Jurua-A_mask1_link_sBCMn_flat.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np

from src.morphodynamics.rivgraph_link_profiles import compute_link_profiles
from src.morphodynamics.coupling import add_Mn_to_link_profiles
from src.analysis.link_sBCMn_io import export_link_sBCMn_to_npz


def compute_link_sBCMn_for_site(
    site: str,
    mask_level: int,
    links_vector_path: str,
    piv_npz_path: str,
    mask_raster_path: str | None = None,
    step_m: float = 100.0,
    export_npz_path: str | None = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Computes s–B–C–Mn profiles for all RivGraph links for a specified site/mask level.

    Parameters
    ------
    site : str
        Site name, e.g., "Jurua-A".
    mask_level : int
        Mask level, e.g., 1 for Mask1.
    links_vector_path : str
        Path to RivGraph links vector file (.shp / .gpkg).
    piv_npz_path : str
        Path to Step 4A strict affine PIV results .npz, containing X_geo, Y_geo, u_m_per_year, v_m_per_year.
    mask_raster_path : str, optional
        Path to binary water mask raster for B(s) calculation. If None, auto-selects the same reference mask as Step 4A.
    step_m : float
        Densified sampling interval along links (meters).
    export_npz_path : str, optional
        If provided, exports all link s–B–C–Mn profiles as a flattened .npz.

    Returns
    ------
    link_sBCMn : {link_id: {"s","x","y","B","C","Mn"}}
    """

    # 1. Load Step 4A PIV results
    piv_npz = Path(piv_npz_path)
    if not piv_npz.exists():
        raise FileNotFoundError(f"PIV npz not found: {piv_npz}")

    piv = np.load(piv_npz)
    X_grid = piv["X_geo"]
    Y_grid = piv["Y_geo"]
    U_grid = piv["u_m_per_year"]
    V_grid = piv["v_m_per_year"]

    # 2. Auto-select mask path if not provided
    if mask_raster_path is None:
        from src.morphodynamics.jurua_georef_multitilt import _choose_reference_mask

        # Don't specify year, let it select the first available .tif in the directory
        mask_raster_path = str(
            _choose_reference_mask(site=site, mask_level=mask_level, year=None)
        )
        print(f"Auto-selected reference mask: {mask_raster_path}")

    # 3. Compute geometric profiles (s, x, y, B, C) for RivGraph links
    link_geom = compute_link_profiles(
        mask_raster_path=mask_raster_path,
        links_vector_path=links_vector_path,
        step_m=step_m,
    )

    if not link_geom:
        raise RuntimeError("No link profiles obtained from RivGraph vector, please check input data.")

    # 4. Overlay PIV normal migration Mn(s) onto geometric profiles
    link_sBCMn = add_Mn_to_link_profiles(
        link_profiles=link_geom,
        X_grid=X_grid,
        Y_grid=Y_grid,
        U_grid=U_grid,
        V_grid=V_grid,
    )

    # 5. Optional export
    if export_npz_path is not None:
        export_link_sBCMn_to_npz(
            link_profiles=link_sBCMn,
            site=site,
            mask_level=mask_level,
            step_m=step_m,
            out_path=export_npz_path,
        )
        print(f"Exported flattened s–B–C–Mn .npz to: {export_npz_path}")

    return link_sBCMn


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Universal s–B–C–Mn profile calculation tool: supports arbitrary site/mask_level configurations.",
    )
    parser.add_argument("--site", required=True, help="Site name, e.g., 'Jurua-A'")
    parser.add_argument("--mask-level", type=int, required=True, help="Mask level, e.g., 1")
    parser.add_argument(
        "--links-vector",
        required=True,
        help="Path to RivGraph links vector file (.shp / .gpkg)",
    )
    parser.add_argument(
        "--piv-npz",
        required=True,
        help="Path to Step 4A strict affine PIV result .npz",
    )
    parser.add_argument(
        "--mask-raster",
        default=None,
        help="Path to binary water mask raster (optional, auto-selected if omitted)",
    )
    parser.add_argument(
        "--step-m",
        type=float,
        default=100.0,
        help="Densified sampling interval along links (meters)",
    )
    parser.add_argument(
        "--export-npz",
        default=None,
        help="If provided, exports all link s–B–C–Mn profiles as a flattened .npz",
    )

    args = parser.parse_args()

    link_sBCMn = compute_link_sBCMn_for_site(
        site=args.site,
        mask_level=args.mask_level,
        links_vector_path=args.links_vector,
        piv_npz_path=args.piv_npz,
        mask_raster_path=args.mask_raster,
        step_m=args.step_m,
        export_npz_path=args.export_npz,
    )

    print(f"Computed s–B–C–Mn profiles for {len(link_sBCMn)} links.")


if __name__ == "__main__":
    main()
