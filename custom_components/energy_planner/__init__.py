"""Home Assistant integration for the energy planner."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import List

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import HomeAssistantError

from .const import (
    DOMAIN,
    LOGGER,
    SERVICE_UPDATE_PLAN,
    SERVICE_RUN_OPTIMIZER,
    DATA_COORDINATORS,
    DATA_SERVICE_REGISTERED,
    DATA_OPTIMIZER_SERVICE_REGISTERED,
)


def _ensure_library_path() -> None:
    component_dir = Path(__file__).resolve().parent
    candidates = [
        component_dir / "vendor",
        component_dir.parent / "energy_planner_vendor",
        component_dir.parent / "energy_planner",
        component_dir.parent.parent / "energy_planner",
    ]

    for base in candidates:
        package_dir = base / "energy_planner"
        if package_dir.exists() and package_dir.is_dir():
            base_str = str(base)
            if base_str not in sys.path:
                sys.path.insert(0, base_str)
                LOGGER.debug("Added energy_planner library path: %s", base_str)
            return


_ensure_library_path()


async def _async_refresh_coordinators(coordinators: List) -> None:
    """Refresh all coordinators."""
    for coord in coordinators:
        await coord.async_request_refresh()


async def async_setup(hass: HomeAssistant, config: dict) -> bool:
    """Set up the integration from configuration.yaml."""
    hass.data.setdefault(DOMAIN, {})
    return True


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up Energy Planner from a config entry."""
    hass.data.setdefault(DOMAIN, {})

    # Register services at domain level
    if not hass.data[DOMAIN].get(DATA_SERVICE_REGISTERED):
        async def _handle_update_service(call: ServiceCall) -> None:
            """Handle the update_plan service call."""
            LOGGER.debug("Received manual update request: data=%s", call.data)
            coordinators = hass.data[DOMAIN].get(DATA_COORDINATORS, [])
            if not coordinators:
                raise HomeAssistantError("No energy planner coordinators are active")
            await _async_refresh_coordinators(coordinators)

        hass.services.async_register(DOMAIN, SERVICE_UPDATE_PLAN, _handle_update_service)
        hass.data[DOMAIN][DATA_SERVICE_REGISTERED] = True
        LOGGER.info("Registered service: %s.%s", DOMAIN, SERVICE_UPDATE_PLAN)

    if not hass.data[DOMAIN].get(DATA_OPTIMIZER_SERVICE_REGISTERED):
        async def _handle_run_optimizer(call: ServiceCall) -> None:
            """Run the full optimizer pipeline and update database."""
            LOGGER.info("Running energy planner optimizer...")

            try:
                # Import run_once from scheduler
                from energy_planner.scheduler import run_once  # type: ignore

                # Run optimizer in executor (blocking operation)
                await hass.async_add_executor_job(run_once)

                LOGGER.info("Optimizer completed successfully")

                # Refresh all coordinators to load new data
                coordinators = hass.data[DOMAIN].get(DATA_COORDINATORS, [])
                if coordinators:
                    await _async_refresh_coordinators(coordinators)
                    LOGGER.info("Sensors refreshed with new plan data")
                else:
                    LOGGER.warning("No coordinators found to refresh")

            except Exception as exc:
                LOGGER.error("Optimizer failed: %s", exc, exc_info=True)
                raise HomeAssistantError(f"Failed to run optimizer: {exc}") from exc

        hass.services.async_register(DOMAIN, SERVICE_RUN_OPTIMIZER, _handle_run_optimizer)
        hass.data[DOMAIN][DATA_OPTIMIZER_SERVICE_REGISTERED] = True
        LOGGER.info("Registered service: %s.%s", DOMAIN, SERVICE_RUN_OPTIMIZER)

    # Forward the config entry to the sensor platform
    await hass.config_entries.async_forward_entry_setups(entry, ["sensor"])

    # Register update listener for options changes
    entry.async_on_unload(entry.add_update_listener(async_reload_entry))

    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    return await hass.config_entries.async_unload_platforms(entry, ["sensor"])


async def async_reload_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload config entry when options change."""
    await async_unload_entry(hass, entry)
    await async_setup_entry(hass, entry)
