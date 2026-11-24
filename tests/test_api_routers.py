"""Tests for how the API routers are assembled into the application."""

from __future__ import annotations

from fastapi.routing import APIRoute, APIWebSocketRoute

from climatevision.api.main import create_app
from climatevision.api.routes import ROUTERS
from climatevision.api.shared import PROJECT_ROOT


def _served_routes(routes) -> set[tuple[str, frozenset[str]]]:
    served = set()
    for route in routes:
        if isinstance(route, APIRoute):
            served.add((route.path, frozenset(route.methods)))
        elif isinstance(route, APIWebSocketRoute):
            served.add((route.path, frozenset({"WEBSOCKET"})))
    return served


def test_every_router_route_is_served_by_the_app() -> None:
    """A router left out of include_routers would silently drop its endpoints."""
    app_routes = _served_routes(create_app().routes)

    for router in ROUTERS:
        router_routes = _served_routes(router.routes)
        assert router_routes, "a router without routes is probably wired wrongly"
        assert router_routes <= app_routes


def test_no_route_is_defined_by_two_routers() -> None:
    """The same path and method in two routers would make one of them unreachable."""
    seen: set[tuple[str, str]] = set()
    for router in ROUTERS:
        for path, methods in _served_routes(router.routes):
            for method in methods:
                assert (path, method) not in seen, f"{method} {path} is defined twice"
                seen.add((path, method))


def test_api_routes_are_matched_before_the_frontend_fallback() -> None:
    """The frontend catch-all, when present, must come after every API route."""
    paths = [getattr(route, "path", None) for route in create_app().routes]

    if "/{full_path:path}" in paths:
        fallback = paths.index("/{full_path:path}")
        assert "/api/health" in paths[:fallback]
        assert "/api/predict" in paths[:fallback]


def test_project_root_is_the_repository_root() -> None:
    """Paths to outputs/, models/ and frontend/ are resolved from PROJECT_ROOT."""
    assert (PROJECT_ROOT / "src" / "climatevision" / "api" / "shared.py").is_file()
