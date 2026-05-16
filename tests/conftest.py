"""Pytest fixtures for ClimateVision."""

import sys
from unittest import mock

import pytest
from fastapi.testclient import TestClient

# Conditionally stub heavy ML dependencies when they are not installed.
# This allows the import graph (api → inference → models → torch) to
# resolve in CI / torch-free environments.  Tests that exercise actual
# model logic will still fail — only the import chain is unblocked.
_MISSING_MODULES: list[str] = []

try:
    import torch  # noqa: F401
except ImportError:
    _MISSING_MODULES.extend([
        "torch", "torch.utils", "torch.utils.data",
        "torch.nn", "torch.nn.functional",
    ])

try:
    import torchvision  # noqa: F401
except ImportError:
    _MISSING_MODULES.append("torchvision")

try:
    import cv2  # noqa: F401
except ImportError:
    _MISSING_MODULES.append("cv2")

try:
    import rasterio  # noqa: F401
except ImportError:
    _MISSING_MODULES.append("rasterio")

try:
    import shapely  # noqa: F401
except ImportError:
    _MISSING_MODULES.extend(["shapely", "shapely.geometry"])

try:
    import geopandas  # noqa: F401
except ImportError:
    _MISSING_MODULES.append("geopandas")

try:
    import sklearn  # noqa: F401
except ImportError:
    _MISSING_MODULES.extend(["sklearn", "sklearn.metrics"])

try:
    import albumentations  # noqa: F401
except ImportError:
    _MISSING_MODULES.append("albumentations")

for mod_name in _MISSING_MODULES:
    sys.modules[mod_name] = mock.MagicMock()

from climatevision.api.main import create_app


@pytest.fixture
def client() -> TestClient:
    """FastAPI test client."""
    app = create_app()
    return TestClient(app)
