"""Data coordinator for the energy planner integration."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Callable

from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import DOMAIN, LOGGER

try:
    from energy_planner.reporting import PlanReport, compute_plan_report, read_plan_from_db  # type: ignore
except ImportError as import_error:  # pragma: no cover - surfaced during runtime
    PlanReport = Any  # type: ignore[misc, assignment]

    def compute_plan_report() -> Any:  # type: ignore[override]
        raise UpdateFailed(
            "energy_planner package is not available. Ensure it is deployed with the integration."
        ) from import_error

    def read_plan_from_db() -> Any:  # type: ignore[override]
        raise UpdateFailed(
            "energy_planner package is not available. Ensure it is deployed with the integration."
        ) from import_error


ComputePlanReportFn = Callable[[], Any]


class EnergyPlanCoordinator(DataUpdateCoordinator):
    """Coordinate energy plan retrieval."""

    def __init__(
        self,
        hass: HomeAssistant,
        update_interval: timedelta,
        compute_fn: ComputePlanReportFn = read_plan_from_db,  # Default to reading from DB
    ) -> None:
        super().__init__(
            hass,
            LOGGER,
            name=f"{DOMAIN}_coordinator",
            update_interval=update_interval,
        )
        self._compute_fn = compute_fn

    async def _async_update_data(self) -> Any:
        """Fetch data via the blocking optimisation pipeline."""

        try:
            report = await self.hass.async_add_executor_job(self._compute_fn)
        except Exception as exc:  # pragma: no cover - defensive conversion
            raise UpdateFailed(f"Energy planner update failed: {exc}") from exc
        return report