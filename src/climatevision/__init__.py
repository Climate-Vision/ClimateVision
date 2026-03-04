"""
ClimateVision: Open-source ML Platform for Deforestation Detection

An end-to-end machine learning platform for detecting deforestation from
satellite imagery using deep learning and computer vision techniques.
"""

__version__ = "0.1.0"
__author__ = "ClimateVision Contributors"
__license__ = "MIT"

# Core imports will be added as modules are developed
from .models import *  # noqa
from .data import *  # noqa
from .inference import *  # noqa

__all__ = [
    "__version__",
]
