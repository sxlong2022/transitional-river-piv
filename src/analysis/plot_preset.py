
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Tuple, Optional

def get_paper_figsize(width_mm: float, height_mm: float = None, aspect_ratio: float = None) -> Tuple[float, float]:
    """
    Calculate figure size in inches from millimeters.
    
    Args:
        width_mm: Target width in mm (90 for single column, 190 for full width)
        height_mm: Target height in mm. If None, uses aspect_ratio.
        aspect_ratio: Width/Height ratio. Default is 4/3 if height_mm is None.
        
    Returns:
        Tuple of (width_inch, height_inch)
    """
    mm_to_inch = 1 / 25.4
    width_inch = width_mm * mm_to_inch
    
    if height_mm is not None:
        height_inch = height_mm * mm_to_inch
    elif aspect_ratio is not None:
        height_inch = width_inch / aspect_ratio
    else:
        # Default goldenish ratio or 4:3
        height_inch = width_inch * 0.75
        
    return (width_inch, height_inch)

def apply_paper_style(dpi: int = 600, font_family: str = "Times New Roman"):
    """
    Apply Journal of Hydrology style presets to matplotlib.
    """
    # Reset to defaults first to avoid pollution
    mpl.rcdefaults()
    
    style_params = {
        # Font configuration
        "font.family": "serif",
        "font.serif": [font_family, "Times", "DejaVu Serif"],
        "font.size": 7,           # Base font size
        "axes.labelsize": 7,      # Axis labels
        "axes.titlesize": 8,      # Subplot titles
        "xtick.labelsize": 6,     # Tick labels
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "figure.titlesize": 9,
        
        # Line widths and markers
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "axes.linewidth": 0.8,    # Spines
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        
        # Grid
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
        "grid.linestyle": ":",
        
        # Layout
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.format": "png",  # User requested PNG only

        # Math text: make math consistent with Times New Roman (no external LaTeX required)
        "mathtext.fontset": "custom",
        "mathtext.rm": font_family,
        "mathtext.it": f"{font_family}:italic",
        "mathtext.bf": f"{font_family}:bold",
        "mathtext.cal": font_family,
        "mathtext.default": "it",
    
        # Color cycle (Color-blind friendly: Tableau 10 is mpl default v2.0+)
        # We can enforce a specific cycle if needed, but default is usually fine.
    }
    
    mpl.rcParams.update(style_params)

def setup_preset(preset: str, dpi: int = 600):
    """
    Main entry point to configure plots.
    """
    if preset == "paper":
        apply_paper_style(dpi=dpi, font_family="Times New Roman")
    else:
        # Default / Screen preset
        pass
