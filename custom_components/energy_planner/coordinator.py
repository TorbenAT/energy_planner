"""Data coordinator for the energy planner integration."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Callable, Optional

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
        self._last_successful_plan: Optional[Any] = None

    async def _async_update_data(self) -> Any:
        """Fetch data via the blocking optimisation pipeline.
        
        If optimization fails due to missing sensor data or validation errors,
        return the last successful plan to avoid breaking the current execution.
        This prevents incorrect energy calculations during 15-min windows.
        """

        try:
            report = await self.hass.async_add_executor_job(self._compute_fn)
            # Cache the successful plan for use during failures
            self._last_successful_plan = report
            return report
        except UpdateFailed:
            # Re-raise update failed errors from the transport layer
            raise
        except Exception as exc:
            error_msg = str(exc)
            # Log the failure with context
            LOGGER.warning(
                "⚠️ Energy plan optimization failed: %s. "
                "Using last successful plan to maintain stable operation.",
                error_msg
            )
            
            # If we have a previous plan, return it instead of crashing
            if self._last_successful_plan is not None:
                LOGGER.info(
                    "Returning cached plan from previous optimization cycle. "
                    "Next successful calculation will update the plan."
                )
                return self._last_successful_plan
            
            # If no cached plan, raise the error
            raise UpdateFailed(f"Energy planner update failed: {error_msg}") from exc