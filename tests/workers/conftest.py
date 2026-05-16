"""Pytest fixtures for alert delivery worker tests."""

import sys
from unittest import mock

import pytest
from fastapi.testclient import TestClient

# Stub heavy ML dependencies so tests can import the API layer
# without installing torch / rasterio / opencv.
sys.modules["torch"] = mock.MagicMock()
sys.modules["torch.utils"] = mock.MagicMock()
sys.modules["torch.utils.data"] = mock.MagicMock()
sys.modules["torch.nn"] = mock.MagicMock()
sys.modules["torch.nn.functional"] = mock.MagicMock()
sys.modules["torchvision"] = mock.MagicMock()
sys.modules["torchvision.transforms"] = mock.MagicMock()
sys.modules["rasterio"] = mock.MagicMock()
sys.modules["cv2"] = mock.MagicMock()
sys.modules["sklearn"] = mock.MagicMock()
sys.modules["sklearn.metrics"] = mock.MagicMock()
sys.modules["albumentations"] = mock.MagicMock()
sys.modules["geopandas"] = mock.MagicMock()
sys.modules["shapely"] = mock.MagicMock()
sys.modules["shapely.geometry"] = mock.MagicMock()

from climatevision.api.main import create_app


@pytest.fixture
def client() -> TestClient:
    """FastAPI test client."""
    app = create_app()
    return TestClient(app)
