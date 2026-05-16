"""
Background workers for ClimateVision.

Workers handle async operations that should not block the API:
- Alert delivery (email + webhook)
- Report generation
- Scheduled analysis runs
"""

from climatevision.workers.alert_delivery import AlertDeliveryWorker

__all__ = ["AlertDeliveryWorker"]
