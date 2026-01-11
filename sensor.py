"""Sensor platform for the energy planner integration."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any, Dict, List, Optional
import math

import voluptuous as vol

from homeassistant.components.sensor import PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import CONF_NAME, CONF_SCAN_INTERVAL
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import (
    CONF_MARKDOWN_LIMIT,
    CONF_MARKDOWN_MAX_LENGTH,
    CONF_PLAN_LIMIT,
    DATA_COORDINATORS,
    DEFAULT_MARKDOWN_LIMIT,
    DEFAULT_MARKDOWN_MAX_LENGTH,
    DEFAULT_NAME,
    DEFAULT_PLAN_LIMIT,
    DEFAULT_SCAN_INTERVAL,
    DOMAIN,
    LOGGER,
)
from .coordinator import EnergyPlanCoordinator

try:
    from energy_planner.reporting import PlanReport  # type: ignore
    from energy_planner.plan_schema import PLAN_FIELDS_HA  # type: ignore
except ImportError:  # pragma: no cover - surfaced during runtime
    PlanReport = Any  # type: ignore[misc, assignment]
    # Fallback if plan_schema not available
    PLAN_FIELDS_HA = []

ICON = "mdi:solar-power"

# Use centrally-defined schema
PLAN_FIELDS = PLAN_FIELDS_HA if PLAN_FIELDS_HA else [
    # Fallback list (same as before, for backwards compatibility)
    "timestamp", "timestamp_local", "activity",
    "g_buy", "g_sell", "grid_to_batt", "grid_to_house", "grid_to_ev",
    "pv_to_batt", "pv_to_house", "pv_to_ev",
    "batt_to_house", "batt_to_sell", "batt_to_ev",
    "battery_in", "battery_out", "battery_soc", "battery_soc_pct",
    "battery_reserve_target", "battery_reserve_shortfall",
    "ev_charge", "ev_soc_kwh", "ev_soc_pct",
    "price_buy", "price_sell", "effective_sell_price",
    "grid_cost", "grid_revenue_effective", "grid_revenue", "cash_cost_dkk",
    "ev_bonus", "battery_cycle_cost", "battery_value_dkk", "objective_component",
    "consumption_estimate_kw", "pv_forecast_kw",
    "house_from_grid", "house_from_battery", "house_from_pv",
    "cheap24", "expensive24",
    "arb_gate", "arb_reason", "arb_basis", "arb_eta_rt", "arb_c_cycle",
    "price_buy_now", "future_max_sell_eff", "arb_margin",
    "policy_wait_flag", "policy_wait_reason", "policy_price_basis",
    "policy_price_now_dkk", "policy_future_min_12h_dkk",
    "policy_grid_charge_allowed", "policy_hold_value_dkk",
]

HOURLY_FIELDS = [
    "hour",
    "g_buy",
    "g_sell",
    "battery_in",
    "battery_out",
    "ev_charge",
    "battery_soc_start",
    "battery_soc_end",
    "battery_soc_min",
    "battery_soc_max",
    "grid_cost",
    "grid_revenue",
]

_MARKDOWN_CHUNK_SIZE = 7000

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
    vol.Optional(CONF_PLAN_LIMIT, default=DEFAULT_PLAN_LIMIT): vol.All(vol.Coerce(int), vol.Range(min=0)),
    vol.Optional(CONF_MARKDOWN_LIMIT, default=DEFAULT_MARKDOWN_LIMIT): vol.All(vol.Coerce(int), vol.Range(min=0)),
    vol.Optional(CONF_MARKDOWN_MAX_LENGTH, default=DEFAULT_MARKDOWN_MAX_LENGTH): vol.All(vol.Coerce(int), vol.Range(min=0)),
        vol.Optional(CONF_SCAN_INTERVAL, default=DEFAULT_SCAN_INTERVAL): cv.time_period,
    }
)


async def async_setup_platform(
    hass: HomeAssistant,
    config: dict,
    async_add_entities: AddEntitiesCallback,
    discovery_info: Optional[dict] = None,
) -> None:
    """Set up the energy plan sensor from configuration.yaml."""

    name: str = config.get(CONF_NAME, DEFAULT_NAME)
    plan_limit: int = config.get(CONF_PLAN_LIMIT, DEFAULT_PLAN_LIMIT)
    markdown_limit: int = config.get(CONF_MARKDOWN_LIMIT, DEFAULT_MARKDOWN_LIMIT)
    markdown_max_length: int = config.get(CONF_MARKDOWN_MAX_LENGTH, DEFAULT_MARKDOWN_MAX_LENGTH)
    scan_interval: timedelta = config.get(CONF_SCAN_INTERVAL, DEFAULT_SCAN_INTERVAL)

    # Import compute_plan_report so the coordinator runs the full optimizer instead of just reading DB
    try:
        from energy_planner.reporting import compute_plan_report  # type: ignore
        compute_fn = compute_plan_report
    except ImportError:
        from energy_planner.reporting import read_plan_from_db  # type: ignore
        compute_fn = read_plan_from_db
        LOGGER.warning("compute_plan_report not available, falling back to read_plan_from_db")

    coordinator = EnergyPlanCoordinator(hass, update_interval=scan_interval, compute_fn=compute_fn)
    # The coordinator will now run the full optimizer on each interval (every scan_interval minutes)

    hass.data.setdefault(DOMAIN, {})
    hass.data[DOMAIN].setdefault(DATA_COORDINATORS, []).append(coordinator)

    # Services are registered in __init__.py async_setup() so they're available immediately

    entity = EnergyPlanSensor(
        coordinator=coordinator,
        name=name,
        plan_limit=plan_limit,
        markdown_limit=markdown_limit,
        markdown_max_length=markdown_max_length,
    )

    async_add_entities([entity])


async def _async_refresh_coordinators(
    coordinators: List[EnergyPlanCoordinator],
) -> None:
    await asyncio.gather(*(coordinator.async_request_refresh() for coordinator in coordinators))


class EnergyPlanSensor(SensorEntity):
    """Sensor exposing the energy plan status and details."""

    _attr_should_poll = False
    _attr_icon = ICON

    def __init__(
        self,
        coordinator: EnergyPlanCoordinator,
        name: str,
        plan_limit: int,
        markdown_limit: int,
        markdown_max_length: int,
    ) -> None:
        self._coordinator = coordinator
        self._attr_name = name
        self._plan_limit = plan_limit
        self._markdown_limit = markdown_limit
        self._markdown_max_length = markdown_max_length
        self._attr_unique_id = f"{DOMAIN}_{name.replace(' ', '_').lower()}"

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        self.async_on_remove(
            self._coordinator.async_add_listener(self.async_write_ha_state)
        )
        # Trigger an immediate refresh in the background so startup isn't blocked
        self.hass.async_create_task(self._coordinator.async_request_refresh())

    @property
    def available(self) -> bool:
        return self._coordinator.last_update_success

    @property
    def native_value(self) -> Optional[str]:
        report = self._coordinator.data
        if not report:
            return None
        status = getattr(report, "status", None)
        return str(status) if status is not None else None

    @property
    def extra_state_attributes(self) -> Optional[Dict[str, Any]]:
        report = self._coordinator.data
        if not report:
            return None
        return _build_attributes(report, self._plan_limit, self._markdown_limit, self._markdown_max_length)

    @property
    def device_info(self) -> Dict[str, Any]:
        return {
            "identifiers": {(DOMAIN, DOMAIN)},
            "name": self.name,
            "manufacturer": "Energy Planner",
            "model": "Optimisation",
        }


def _build_attributes(
    report: Any,
    plan_limit: int,
    markdown_limit: int,
    markdown_max_length: int,
) -> Dict[str, Any]:
    attributes: Dict[str, Any] = {
        "objective": round(getattr(report, "objective_value", 0.0), 3),
        "notes": list(getattr(report, "notes", [])),
        "summary": getattr(report, "summary", {}),
        "day_summary": getattr(report, "day_summary", []),
        "generated_at": getattr(report, "generated_at", None).isoformat()
        if getattr(report, "generated_at", None)
        else None,
        "updated_at": getattr(report, "updated_at", None).isoformat()
        if getattr(report, "updated_at", None)
        else None,
        "timezone": getattr(report, "timezone", None),
        "ev_window": {
            "start": getattr(report, "ev_window_start", None).isoformat()
            if getattr(report, "ev_window_start", None)
            else None,
            "end": getattr(report, "ev_window_end", None).isoformat()
            if getattr(report, "ev_window_end", None)
            else None,
            "mode": getattr(report, "ev_planning_mode", None),
            "switch_state": getattr(report, "ev_switch_state", None),
        },
    }

    plan_limit_arg = None if plan_limit <= 0 else plan_limit
    markdown_limit_arg = None if markdown_limit <= 0 else markdown_limit

    # Get all plan records for splitting into daily attributes
    all_plan_records = report.plan_records(limit=None) if hasattr(report, "plan_records") else []
    plan_records = all_plan_records[:plan_limit_arg] if plan_limit_arg else all_plan_records
    hourly_records = report.hourly_records() if hasattr(report, "hourly_records") else []

    attributes["plan_fields"] = PLAN_FIELDS
    
    # Split plan data into three daily attributes to stay under HA's 16KB limit
    slots_per_day = 96  # 24 hours * 4 slots per hour
    attributes["plan_today"] = [
        [_extract_field(record, field) for field in PLAN_FIELDS]
        for record in all_plan_records[:slots_per_day]
    ]
    attributes["plan_tomorrow"] = [
        [_extract_field(record, field) for field in PLAN_FIELDS]
        for record in all_plan_records[slots_per_day:slots_per_day*2]
    ]
    attributes["plan_day3"] = [
        [_extract_field(record, field) for field in PLAN_FIELDS]
        for record in all_plan_records[slots_per_day*2:slots_per_day*3]
    ]
    
    # Keep the original "plan" attribute for backward compatibility (limited to plan_limit)
    attributes["plan"] = [
        [_extract_field(record, field) for field in PLAN_FIELDS]
        for record in plan_records
    ]

    attributes["hourly_fields"] = HOURLY_FIELDS
    attributes["hourly"] = [
        [_extract_field(record, field) for field in HOURLY_FIELDS]
        for record in hourly_records
    ]

    markdown = report.to_markdown(limit=markdown_limit_arg) if hasattr(report, "to_markdown") else ""
    if markdown_max_length and len(markdown) > markdown_max_length:
        markdown = markdown[: markdown_max_length - 3] + "..."
    if markdown:
        local_display = None
        # Prefer updated_at (from DB) over generated_at (when freshly computed)
        timestamp_to_show = getattr(report, "updated_at", None) or getattr(report, "generated_at", None)
        if timestamp_to_show and getattr(report, "timezone", None):
            try:
                import pytz

                tz = pytz.timezone(report.timezone)  # type: ignore[arg-type]
                local_ts = timestamp_to_show.astimezone(tz)
                local_display = local_ts.strftime("%Y-%m-%d %H:%M")
                if getattr(report, "updated_at", None):
                    attributes["updated_at_local"] = local_ts.isoformat()
                    attributes["updated_at_local_display"] = local_display
                else:
                    attributes["generated_at_local"] = local_ts.isoformat()
                    attributes["generated_at_local_display"] = local_display
            except Exception:  # pragma: no cover - fall back silently
                local_display = None
        status_line = f"\n\n**Status:** {getattr(report, 'status', 'unknown')}"
        if local_display and getattr(report, "timezone", None):
            label = "Opdateret fra DB" if getattr(report, "updated_at", None) else "Genereret"
            status_line += f"\n\n**{label}:** {local_display} ({report.timezone})"
        markdown += status_line
        markdown_chunks = _chunk_text(markdown, _MARKDOWN_CHUNK_SIZE)
        attributes["markdown_html"] = markdown_chunks[0]
        if len(markdown_chunks) > 1:
            attributes["markdown_html_chunks"] = markdown_chunks

    if hasattr(report, "debug_inputs_summary"):
        attributes["debug_inputs"] = report.debug_inputs_summary()
    if hasattr(report, "debug_house_balance_records"):
        attributes["debug_house_balance"] = report.debug_house_balance_records(limit=plan_limit_arg)

    # Mirror reconciliation KPIs to top-level attributes if available in summary
    try:
        summary: Dict[str, Any] = attributes.get("summary", {})
        rec_today = summary.get("reconciliation_today")
        rec_yday = summary.get("reconciliation_yesterday")
        if isinstance(rec_today, dict):
            attributes["reconciliation_today"] = rec_today
            # Simple KPIs for dashboards
            attributes["house_consumption_actual_kwh"] = rec_today.get("house_total_kwh")
            attributes["house_consumption_no_ev_kwh"] = rec_today.get("house_no_ev_kwh")
            attributes["ev_energy_hist_kwh"] = rec_today.get("ev_kwh")
        if isinstance(rec_yday, dict):
            attributes["reconciliation_yesterday"] = rec_yday
    except Exception:  # pragma: no cover - defensive only
        pass

    # Compute KPIs server-side to avoid heavy/fragile HA templates
    try:
        kpis: Dict[str, Any] = {}

        def _to_float(v: Any) -> float:
            try:
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    return 0.0
                return float(str(v).replace(",", "."))
            except Exception:
                return 0.0

        recs: List[Dict[str, Any]] = []
        try:
            # Prefer direct DataFrame iteration if available for performance
            if hasattr(report, "plan_records"):
                recs = list(report.plan_records(limit=None))  # type: ignore
            else:
                recs = []
        except Exception:
            recs = []

        def _sum(field: str) -> float:
            return round(sum(_to_float(r.get(field)) for r in recs), 3)

        if recs:
            kpis["grid_import_kwh"] = round(_sum("g_buy"), 2)
            kpis["grid_export_kwh"] = round(_sum("g_sell"), 2)
            # Prefer explicit house_from_* if present, else fall back to flow names
            house_grid = _sum("house_from_grid") or _sum("grid_to_house")
            house_batt = _sum("house_from_battery") or _sum("batt_to_house")
            house_pv = _sum("house_from_pv") or _sum("prod_to_house")
            kpis["house_consumption_kwh"] = round(house_grid + house_batt + house_pv, 2)

            kpis["grid_cost_dkk"] = round(_sum("grid_cost"), 2)
            kpis["grid_revenue_eff_dkk"] = round(_sum("grid_revenue_effective"), 2)
            kpis["battery_cycle_cost_dkk"] = round(_sum("battery_cycle_cost"), 2)
            kpis["ev_bonus_dkk"] = round(_sum("ev_bonus"), 2)
            kpis["net_dkk"] = round(
                kpis.get("grid_cost_dkk", 0.0)
                - kpis.get("grid_revenue_eff_dkk", 0.0)
                + kpis.get("battery_cycle_cost_dkk", 0.0)
                - kpis.get("ev_bonus_dkk", 0.0),
                2,
            )

            # EV energy
            ev_kwh = _sum("ev_charge")
            if ev_kwh <= 0.0:
                ev_kwh = _sum("grid_to_ev") + _sum("batt_to_ev") + _sum("prod_to_ev")
            kpis["ev_energy_kwh"] = round(ev_kwh, 2)

            # Wait/arbitrage counts
            try:
                kpis["wait_slots"] = int(sum(1 for r in recs if r.get("policy_wait_flag")))
            except Exception:
                kpis["wait_slots"] = 0
            try:
                kpis["arb_ok"] = int(sum(1 for r in recs if r.get("arb_gate") in (True, "true", "True", 1)))
                kpis["arb_blocked"] = int(sum(1 for r in recs if r.get("arb_gate") in (False, "false", "False", 0)))
            except Exception:
                kpis["arb_ok"] = 0
                kpis["arb_blocked"] = 0

            # Cheapest block (6 slots ~ 1.5h) by average price_buy
            try:
                w = 6
                price_list: List[float] = []
                time_list: List[str] = []
                for r in recs:
                    price_list.append(_to_float(r.get("price_buy")))
                    ts = r.get("timestamp_local") or r.get("timestamp")
                    time_list.append(str(ts) if ts is not None else "")
                best_avg: Optional[float] = None
                best_s: Optional[int] = None
                if len(price_list) >= w:
                    for s in range(0, len(price_list) - w + 1):
                        window = price_list[s : s + w]
                        if any(v is None for v in window):
                            continue
                        avg = sum(window) / w
                        if best_avg is None or avg < best_avg:
                            best_avg = avg
                            best_s = s
                if best_avg is not None and best_s is not None:
                    start = time_list[best_s]
                    end = time_list[best_s + w - 1]
                    kpis["cheapest_block"] = f"{start} → {end} @ {best_avg:.2f} DKK/kWh"
                else:
                    kpis["cheapest_block"] = "-"
            except Exception:
                kpis["cheapest_block"] = "-"

        attributes["kpis"] = kpis
        # Back-compat: mirror common KPIs at top-level so existing templates can read them directly
        for key, value in kpis.items():
            attributes.setdefault(key, value)

        # Always provide an advisory EV charge recommendation within the available EV window
        # independent of charger status, so dashboards can display it.
        try:
            evw: Dict[str, Any] = attributes.get("ev_window", {}) if isinstance(attributes.get("ev_window"), dict) else {}
            w_start = evw.get("start")
            w_end = evw.get("end")
            best_rec: Optional[Dict[str, Any]] = None
            best_avg: Optional[float] = None
            start_str: Optional[str] = None
            end_str: Optional[str] = None
            if recs and w_start and w_end:
                try:
                    from datetime import datetime
                    ws = datetime.fromisoformat(str(w_start))
                    we = datetime.fromisoformat(str(w_end))
                except Exception:
                    ws = None  # type: ignore
                    we = None  # type: ignore
                # Build filtered list within the EV window (using timestamp_local when available)
                w = 6
                filtered_prices: List[float] = []
                filtered_times: List[str] = []
                for r in recs:
                    ts_val = r.get("timestamp_local") or r.get("timestamp")
                    try:
                        t = datetime.fromisoformat(str(ts_val)) if ts_val else None
                    except Exception:
                        t = None
                    if ws is not None and we is not None and t is not None and ws <= t <= we:
                        filtered_prices.append(_to_float(r.get("price_buy")))
                        filtered_times.append(str(ts_val))
                if len(filtered_prices) >= 1:
                    if len(filtered_prices) >= w:
                        for s in range(0, len(filtered_prices) - w + 1):
                            window = filtered_prices[s : s + w]
                            if not window or any(v is None for v in window):
                                continue
                            avg = sum(window) / float(w)
                            if best_avg is None or avg < best_avg:
                                best_avg = avg
                                start_str = filtered_times[s]
                                end_str = filtered_times[s + w - 1]
                    else:
                        # Not enough slots for a full window; use average over what's available
                        best_avg = sum(filtered_prices) / float(len(filtered_prices))
                        start_str = filtered_times[0]
                        end_str = filtered_times[-1]
            # Compose recommendation payload
            if best_avg is not None and start_str and end_str:
                energy_kwh = kpis.get("ev_energy_kwh") or 0.0
                # If EV energy is zero, try to use policy buffer target
                policy_dict: Dict[str, Any] = attributes.get("summary", {}).get("policy", {}) if isinstance(attributes.get("summary"), dict) else {}
                if not energy_kwh:
                    try:
                        energy_kwh = float(policy_dict.get("ev_buffer_target_kwh", 0.0))
                    except Exception:
                        energy_kwh = 0.0
                charge_rec = {
                    "start": start_str,
                    "end": end_str,
                    "energy_kwh": round(float(energy_kwh), 2),
                    "average_price_dkk": round(float(best_avg), 2),
                    "advisory": True,
                }
                # Insert into summary.policy so templates can pick it up
                if isinstance(attributes.get("summary"), dict):
                    summary_dict: Dict[str, Any] = attributes["summary"]
                    policy_out: Dict[str, Any] = summary_dict.get("policy", {}) if isinstance(summary_dict.get("policy"), dict) else {}
                    policy_out["charge_recommendation"] = charge_rec
                    summary_dict["policy"] = policy_out
                    attributes["summary"] = summary_dict
        except Exception:  # pragma: no cover - non critical
            pass
    except Exception:  # pragma: no cover - non-fatal
        pass

    return attributes


def _extract_field(record: Any, field: str) -> Any:
    return record.get(field) if isinstance(record, dict) else getattr(record, field, None)


def _chunk_text(text: str, chunk_size: int) -> List[str]:
    if chunk_size <= 0:
        return [text]
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]
