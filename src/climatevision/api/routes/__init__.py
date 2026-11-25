"""
Route modules of the ClimateVision API, one ``APIRouter`` per resource.
"""

from __future__ import annotations

from fastapi import FastAPI

from climatevision.api.routes import (
    alerts,
    analysis,
    explain,
    health,
    organizations,
    predict,
    runs,
    subscriptions,
)

# Included in this order, which is the order the routes are matched in.
ROUTERS = (
    health.router,
    analysis.router,
    runs.router,
    predict.router,
    explain.router,
    organizations.router,
    subscriptions.router,
    alerts.router,
)


def include_routers(app: FastAPI) -> None:
    """Include every API router in ``app``."""
    for router in ROUTERS:
        app.include_router(router)
