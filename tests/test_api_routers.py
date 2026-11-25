"""Tests for how the API routers are assembled into the application."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

import climatevision.api.main as api_main
from climatevision.api.main import create_app
from climatevision.api.routes import ROUTERS
from climatevision.api.shared import PROJECT_ROOT


def _http_routes(router) -> set[tuple[str, str]]:
    return {
        (route.path, method.lower())
        for route in router.routes
        if isinstance(route, APIRoute)
        for method in route.methods
    }


def test_every_router_route_is_served_by_the_app() -> None:
    """A router left out of include_routers would silently drop its endpoints."""
    paths = create_app().openapi()["paths"]
    served = {(path, method) for path, item in paths.items() for method in item}

    for router in ROUTERS:
        assert router.routes, "a router without routes is probably wired wrongly"
        assert _http_routes(router) <= served


def test_no_route_is_defined_by_two_routers() -> None:
    """The same path and method in two routers would make one of them unreachable."""
    seen: set[tuple[str, str]] = set()
    for router in ROUTERS:
        for route in _http_routes(router):
            assert route not in seen, f"{route} is defined twice"
            seen.add(route)


def test_api_routes_are_matched_before_the_frontend_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With a built frontend, API paths must still reach the API, not index.html."""
    dist = tmp_path / "frontend" / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>frontend</html>", encoding="utf-8")
    monkeypatch.setattr(api_main, "PROJECT_ROOT", tmp_path)

    client = TestClient(create_app())

    analysis_types = client.get("/api/analysis-types")
    assert analysis_types.status_code == 200
    assert isinstance(analysis_types.json(), list)

    deep_link = client.get("/runs")
    assert deep_link.status_code == 200
    assert deep_link.text == "<html>frontend</html>"


def test_project_root_is_the_repository_root() -> None:
    """Paths to outputs/, models/ and frontend/ are resolved from PROJECT_ROOT."""
    assert (PROJECT_ROOT / "src" / "climatevision" / "api" / "shared.py").is_file()
