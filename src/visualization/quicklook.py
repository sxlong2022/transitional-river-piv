"""Quicklook placeholder module: Simple vector field visualization can be added here."""

from pathlib import Path


def describe_output_root(project_root: Path) -> Path:
    """Convention for the root directory of all figure outputs."""
    return project_root / "results" / "figures"
