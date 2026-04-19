"""Pytest fixtures for ClimateVision."""

import pytest
from fastapi.testclient import TestClient

from climatevision.api.main import create_app


@pytest.fixture
def client() -> TestClient:
    """FastAPI test client."""
    app = create_app()
    return TestClient(app)
