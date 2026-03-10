"""
ClimateVision: Open-source ML Platform for Deforestation Detection

An end-to-end machine learning platform for detecting deforestation from
satellite imagery using deep learning and computer vision techniques.
"""

__version__ = "0.1.0"
__author__ = "ClimateVision Contributors"
__license__ = "MIT"

# Lazy imports to avoid loading torch/heavy deps when only API is used
__all__ = ["__version__", "UNet", "AttentionUNet", "SiameseNetwork"]


def __getattr__(name):
    if name in ("UNet", "AttentionUNet", "SiameseNetwork"):
        from .models import UNet, AttentionUNet, SiameseNetwork
        return {"UNet": UNet, "AttentionUNet": AttentionUNet, "SiameseNetwork": SiameseNetwork}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
