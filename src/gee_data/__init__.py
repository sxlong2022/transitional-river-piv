"""GEE data acquisition module: For acquiring annual water masks for Yellow River and other sites from Google Earth Engine.

This module is used as a CLI entry point and does not pre-import in __init__ to avoid RuntimeWarning when running python -m.
If needed in other code, you can explicitly import:
    from src.gee_data.pull_huanghe_masks import pull_huanghe_masks
"""

__all__ = ["pull_huanghe_masks"]


def __getattr__(name):
    """Lazy import to avoid RuntimeWarning when running python -m."""
    if name == "pull_huanghe_masks":
        from .pull_huanghe_masks import pull_huanghe_masks
        return pull_huanghe_masks
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
