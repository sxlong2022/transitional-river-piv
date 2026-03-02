"""Minimal PIV example script running on Jurua-A site.

Dependencies:
- Data_and_code/Data/PreparedImagery/Jurua-A is extracted
- Dependencies such as openpiv, rasterio, matplotlib are installed

Usage (from project root):
    python -m src.piv_analysis.minimal_example_jurua
"""

from pathlib import Path

import numpy as np
import rasterio
from openpiv import pyprocess, validation, filters

from src.config import PROJECT_ROOT
from src.preprocessing.prepared_imagery import get_prepared_imagery_dir
from src.visualization.quicklook import describe_output_root


def run_minimal_jurua(site: str = "Jurua-A", mask_level: int = 1, tilt_deg: int = 0):
    """Runs a minimal PIV analysis for the given site and mask/tilt configuration.

    Returns (x, y, u, v, input directory, list of two images used).
    """
    base = get_prepared_imagery_dir(site)
    tilt_str = f"Mask{mask_level}_Tilt{abs(tilt_deg):02d}"
    in_dir = base / tilt_str

    tifs = sorted(in_dir.glob("*.tif"))
    if len(tifs) < 2:
        raise RuntimeError(f"Not enough tif files in {in_dir}")

    # Only take the first two images for minimal demonstration
    with rasterio.open(tifs[0]) as src:
        frame_a = src.read(1)
    with rasterio.open(tifs[1]) as src:
        frame_b = src.read(1)

    # PIV analysis (basic single-pass window version)
    u, v, sig2noise = pyprocess.extended_search_area_piv(
        frame_a.astype(np.int32),
        frame_b.astype(np.int32),
        window_size=64,
        overlap=32,
        dt=1.0,
        search_area_size=64,
        sig2noise_method="peak2peak",
    )

    x, y = pyprocess.get_coordinates(
        image_size=frame_a.shape,
        search_area_size=64,
        overlap=32,
    )

    # Simple quality control
    u, v, mask = validation.sig2noise_val(u, v, sig2noise, threshold=1.3)
    u, v = filters.replace_outliers(u, v, method="localmean")

    return x, y, u, v, in_dir, tifs[:2]


def main() -> None:
    import matplotlib.pyplot as plt

    x, y, u, v, in_dir, used = run_minimal_jurua()
    print("PIV input directory:", in_dir)
    print("Two images used:", [p.name for p in used])
    print("u/v shape:", u.shape)
    print("Number of valid vectors:", int(np.isfinite(u).sum()))

    out_root = describe_output_root(PROJECT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)
    out_png = out_root / "jurua_minimal_piv.png"

    fig, ax = plt.subplots(figsize=(8, 6))
    mag = np.hypot(u, v)
    q = ax.quiver(x, y, u, v, mag, cmap="viridis", scale=50)
    plt.colorbar(q, ax=ax, label="|v| (pixel/yr)")
    ax.set_aspect("equal")
    ax.set_title("Jurua-A minimal PIV example")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print("Saved image to:", out_png)


if __name__ == "__main__":
    main()
