"""Config flow for Energy Planner integration."""

from __future__ import annotations

from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.data_entry_flow import FlowResult
import homeassistant.helpers.config_validation as cv

from .const import (
    CONF_PLAN_LIMIT,
    CONF_MARKDOWN_LIMIT,
    CONF_MARKDOWN_MAX_LENGTH,
    DEFAULT_NAME,
    DEFAULT_PLAN_LIMIT,
    DEFAULT_MARKDOWN_LIMIT,
    DEFAULT_MARKDOWN_MAX_LENGTH,
    DEFAULT_SCAN_INTERVAL,
    DOMAIN,
)


class EnergyPlannerConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Energy Planner."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Handle the initial step."""
        if user_input is not None:
            # Check if already configured
            await self.async_set_unique_id("energy_planner_instance")
            self._abort_if_unique_id_configured()

            return self.async_create_entry(
                title=DEFAULT_NAME,
                data={},
                options={
                    CONF_PLAN_LIMIT: DEFAULT_PLAN_LIMIT,
                    CONF_MARKDOWN_LIMIT: DEFAULT_MARKDOWN_LIMIT,
                    CONF_MARKDOWN_MAX_LENGTH: DEFAULT_MARKDOWN_MAX_LENGTH,
                    "scan_interval_minutes": int(DEFAULT_SCAN_INTERVAL.total_seconds() / 60),
                },
            )

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({}),
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> EnergyPlannerOptionsFlowHandler:
        """Get the options flow for this handler."""
        return EnergyPlannerOptionsFlowHandler(config_entry)


class EnergyPlannerOptionsFlowHandler(config_entries.OptionsFlow):
    """Handle options flow for Energy Planner."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self.config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> FlowResult:
        """Manage the options."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_PLAN_LIMIT,
                        default=self.config_entry.options.get(
                            CONF_PLAN_LIMIT, DEFAULT_PLAN_LIMIT
                        ),
                    ): vol.All(vol.Coerce(int), vol.Range(min=0, max=1000)),
                    vol.Optional(
                        CONF_MARKDOWN_LIMIT,
                        default=self.config_entry.options.get(
                            CONF_MARKDOWN_LIMIT, DEFAULT_MARKDOWN_LIMIT
                        ),
                    ): vol.All(vol.Coerce(int), vol.Range(min=0, max=500)),
                    vol.Optional(
                        CONF_MARKDOWN_MAX_LENGTH,
                        default=self.config_entry.options.get(
                            CONF_MARKDOWN_MAX_LENGTH, DEFAULT_MARKDOWN_MAX_LENGTH
                        ),
                    ): vol.All(vol.Coerce(int), vol.Range(min=0, max=10000)),
                    vol.Optional(
                        "scan_interval_minutes",
                        default=self.config_entry.options.get(
                            "scan_interval_minutes",
                            int(DEFAULT_SCAN_INTERVAL.total_seconds() / 60),
                        ),
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=1440)),
                }
            ),
        )
