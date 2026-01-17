"""Configuration and secrets loading utilities for the energy planner."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .constants import DEFAULT_LOOKAHEAD_HOURS, DEFAULT_RESOLUTION_MINUTES

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore


logger = logging.getLogger(__name__)

_LOADED_ENV_FILES: List[str] = []

def _strip_inline_comment(value: str) -> str:
    if value.startswith(("'", '"')):
        return value
    marker = value.find(" #")
    if marker != -1:
        return value[:marker]
    marker = value.find("#")
    if marker != -1:
        return value[:marker]
    return value


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _apply_env_file(path: Path) -> None:
    try:
        if not path.exists():
            return
        if load_dotenv:
            load_dotenv(dotenv_path=path, override=False)
            _LOADED_ENV_FILES.append(str(path))
            logger.debug("Loaded environment variables from %s", path)
            return
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, value = line.partition("=")
            if not sep:
                continue
            key = key.strip()
            if not key:
                continue
            value = _strip_inline_comment(value).strip()
            value = _unquote(value)
            os.environ.setdefault(key, value)
        _LOADED_ENV_FILES.append(str(path))
        logger.debug("Loaded environment variables from %s (manual parser)", path)
    except OSError:
        return


def _load_dotenv_candidates() -> None:
    candidates = []

    env_hint = os.environ.get("ENERGY_PLANNER_DOTENV_PATH")
    if env_hint:
        candidates.append(Path(env_hint))

    module_root = Path(__file__).resolve().parent
    cwd = Path.cwd()

    candidates.extend(
        [
            cwd / ".env",
            module_root / ".env",
            module_root.parent / ".env",
            Path("/config/custom_components/energy_planner/.env"),
            Path("/config/custom_components/energy_planner/vendor/.env"),
            Path("/config/energy_planner/.env"),
            Path("/config/.env"),
            Path("/homeassistant/.env"),
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.is_absolute() else candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        _apply_env_file(candidate)


def get_loaded_env_files() -> List[str]:
    return list(_LOADED_ENV_FILES)


if load_dotenv:
    load_dotenv()

_load_dotenv_candidates()


def _looks_like_token(value: str) -> bool:
    return value.count(".") >= 2 and len(value) > 40


def _read_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to read secrets.yaml. Install it or set HA_API_KEY via environment."  # noqa: E501
        )
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_secret_from_file(secret_name: str) -> Optional[str]:
    secrets_path_env = os.environ.get("HA_SECRETS_PATH")
    if not secrets_path_env:
        return None

    path = Path(secrets_path_env).expanduser()
    if not path.exists():
        if _looks_like_token(secret_name):
            return secret_name
        raise FileNotFoundError(f"Secrets file not found at {path}")

    secrets = _read_yaml(path)
    value = secrets.get(secret_name)
    if value is None:
        if _looks_like_token(secret_name):
            return secret_name
        raise KeyError(f"Secret '{secret_name}' not found in {path}")
    return str(value)


@dataclass(slots=True)
class Settings:
    ha_base_url: str
    ha_api_key: str
    mariadb_dsn: str
    # Optional DB controls for plan slot persistence
    db_enabled: bool = False
    db_url: Optional[str] = None
    timezone: str = "Europe/Copenhagen"
    lookahead_hours: int = DEFAULT_LOOKAHEAD_HOURS
    resolution_minutes: int = DEFAULT_RESOLUTION_MINUTES
    consumption_history_hours: int = 96
    learning_history_days: int = 30
    battery_reserve_bias: float = 0.1  # Changed from 0.3 to 0.1 for more aggressive battery usage
    grid_sell_price_multiplier: float = 0.6
    grid_sell_penalty_dkk_per_kwh: float = 0.0
    ev_default_daily_kwh: float = 16.0
    ev_planned_kwh_sensor: Optional[str] = "sensor.energy_planner_ev_planned_kwh"
    ev_arrival_soc_sensor: Optional[str] = "input_number.energy_planner_ev_arrival_soc_pct"
    ev_departure_soc_sensor: Optional[str] = "input_number.energy_planner_ev_departure_soc_pct"
    house_expected_daily_kwh_sensor: Optional[str] = "input_number.energy_planner_house_expected_daily_kwh"
    ev_energy_sensor: Optional[str] = "sensor.carport_lifetime_energy"
    ev_energy_history_days: int = 14
    future_extra_load_sensor: Optional[str] = None
    ev_future_daily_buffer_kwh: float = 0.0
    ev_window_local_buffer_pct: float = 0.2
    # Optional cumulative (increasing) sensors for direct quarter-hour deltas
    house_total_sensor: Optional[str] = "sensor.deye12_sun12k_total_consumption"
    pv_total_sensor: Optional[str] = "sensor.deye12_sun12k_total_pv_production"
    battery_charge_total_sensor: Optional[str] = "sensor.deye12_sun12k_total_charge_of_the_battery"
    battery_discharge_total_sensor: Optional[str] = "sensor.deye12_sun12k_total_discharge_of_the_battery"
    grid_buy_total_sensor: Optional[str] = "sensor.deye12_sun12k_total_energy_bought"
    grid_sell_total_sensor: Optional[str] = "sensor.deye12_sun12k_total_energy_sold"
    price_forecast_url: Optional[str] = "https://raw.githubusercontent.com/solmoller/Spotprisprognose/main/prognose.json"
    price_forecast_zone: str = "DK1"
    price_forecast_eur_to_dkk: float = 7.45
    low_sell_price_threshold_dkk: float = 0.05
    price_future_margin_dkk: float = 0.10
    low_price_reserve_pct: float = 0.35
    flat_price_reserve_pct: float = 0.20
    forecast_sensors: List[str] = field(default_factory=lambda: [
        "sensor.solcast_pv_forecast_forecast_today",
        "sensor.solcast_pv_forecast_forecast_tomorrow",
        "sensor.solcast_pv_forecast_forecast_day_3",
        "sensor.solcast_pv_forecast_forecast_day_4",
        "sensor.solcast_pv_forecast_forecast_day_5",
        "sensor.solcast_pv_forecast_forecast_day_6",
        "sensor.solcast_pv_forecast_forecast_day_7",
    ])
    price_buy_sensor: str = "sensor.energi_data_service"
    price_sell_sensor: str = "sensor.energi_data_service_salg"
    house_consumption_sensor: Optional[str] = "sensor.house_load_quarter_hour"
    battery_soc_sensor: str = "sensor.deye12_sun12k_battery_capacity"
    ev_start_sensor: str = "select.ev_smart_charging_charge_start_time"
    ev_end_sensor: str = "select.ev_smart_charging_charge_completion_time"
    ev_soc_sensor: str = "sensor.tessa_battery"
    ev_target_soc_sensor: str = "input_number.tesla_charge_procent_limit"
    ev_default_departure_sensor: Optional[str] = "input_number.energy_planner_ev_default_pct"
    ev_status_sensor: str = "sensor.easee_status"
    ev_planning_switch: Optional[str] = "switch.ev_smart_charging_smart_charging_activated"
    # Optional recorder/utility quarter-hour sensors for fallback (kWh per slot)
    ev_qh_sensor: Optional[str] = None
    pv_qh_sensor: Optional[str] = None
    grid_buy_qh_sensor: Optional[str] = None
    grid_sell_qh_sensor: Optional[str] = None
    battery_charge_qh_sensor: Optional[str] = None
    battery_discharge_qh_sensor: Optional[str] = None
    # Optional user controls (HA helpers)
    optimistic_charging_pct_sensor: Optional[str] = "input_number.energy_planner_optimistic_charging_pct"
    cheap_price_threshold_sensor: Optional[str] = "input_number.energy_planner_cheap_price_threshold_dkk"
    planning_profile_sensor: Optional[str] = "input_select.energy_planner_profile"
    allow_multi_windows_switch: Optional[str] = "input_boolean.energy_planner_multi_windows"
    # New hardware sensors
    battery_capacity_sensor: Optional[str] = "input_number.deye12_sun12k_battery_capacity_watts"
    battery_charge_power_sensor: Optional[str] = "input_number.deye12_sun12k_battery_charge_watts"
    battery_discharge_power_sensor: Optional[str] = "input_number.deye12_sun12k_battery_discharge_watts"
    battery_min_soc_sensor: Optional[str] = "input_number.battery_minimum"
    battery_max_soc_sensor: Optional[str] = "input_number.battery_maximum"
    # Feature flags
    use_linear_solver: bool = False
    use_ml_forecast: bool = False


DEFAULT_SECRET_NAME = "homeassistant_api_token"


def load_settings() -> Settings:
    """Populate settings from environment variables and HA secrets."""

    base_url = os.environ.get("HA_BASE_URL")
    if not base_url:
        raise EnvironmentError("HA_BASE_URL must be set to the Home Assistant API base URL")

    api_key = os.environ.get("HA_API_KEY")
    if not api_key:
        secret_name = os.environ.get("HA_API_KEY_SECRET", DEFAULT_SECRET_NAME)
        api_key = _load_secret_from_file(secret_name) or ""

    if not api_key:
        raise EnvironmentError(
            "Home Assistant API token missing. Set HA_API_KEY or configure HA_SECRETS_PATH and secret name."  # noqa: E501
        )

    mariadb_dsn = os.environ.get("MARIADB_DSN")
    if not mariadb_dsn:
        raise EnvironmentError("MARIADB_DSN must be set, e.g. mysql+pymysql://user:pass@host/dbname")
    # Optional DB controls for writing plan slots
    db_enabled = os.environ.get("DB_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
    db_url = os.environ.get("DB_URL") or mariadb_dsn

    timezone = os.environ.get("ENERGY_TIMEZONE", "Europe/Copenhagen")
    lookahead = int(os.environ.get("LOOKAHEAD_HOURS", DEFAULT_LOOKAHEAD_HOURS))
    resolution = int(os.environ.get("RESOLUTION_MINUTES", DEFAULT_RESOLUTION_MINUTES))
    consumption_history_hours = int(os.environ.get("CONSUMPTION_HISTORY_HOURS", 96))
    learning_history_days = int(os.environ.get("LEARNING_HISTORY_DAYS", 30))
    grid_sell_price_multiplier = float(os.environ.get("GRID_SELL_PRICE_MULTIPLIER", 0.6))
    grid_sell_penalty_dkk_per_kwh = float(os.environ.get("GRID_SELL_PENALTY_DKK_PER_KWH", 0.0))
    ev_default_daily_kwh = float(os.environ.get("EV_DEFAULT_DAILY_KWH", 16.0))
    # Use env override when provided; otherwise use dataclass default for cumulative EV energy sensor
    ev_energy_sensor = os.environ.get("EV_ENERGY_SENSOR", Settings.__dataclass_fields__["ev_energy_sensor"].default)
    if ev_energy_sensor:
        ev_energy_sensor = ev_energy_sensor.strip() or None
    ev_energy_history_days = int(os.environ.get("EV_ENERGY_HISTORY_DAYS", 14))
    future_extra_load_sensor = os.environ.get("FUTURE_EXTRA_LOAD_SENSOR")
    if future_extra_load_sensor:
        future_extra_load_sensor = future_extra_load_sensor.strip() or None
    ev_future_daily_buffer_kwh = float(os.environ.get("EV_FUTURE_DAILY_BUFFER_KWH", 0.0))

    consumption_sensor = os.environ.get("HOUSE_CONSUMPTION_SENSOR")
    ev_planning_switch = os.environ.get("EV_PLANNING_SWITCH")

    field_defaults = Settings.__dataclass_fields__

    def _default(name: str):
        field = field_defaults.get(name)
        return field.default if field is not None else None

    reserve_bias = float(os.environ.get("BATTERY_RESERVE_BIAS", _default("battery_reserve_bias")))

    def _clean_sensor(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    ev_window_local_buffer_pct = float(
        os.environ.get("EV_WINDOW_LOCAL_BUFFER_PCT", _default("ev_window_local_buffer_pct"))
    )

    price_forecast_url = os.environ.get("PRICE_FORECAST_URL", _default("price_forecast_url"))
    if price_forecast_url:
        price_forecast_url = price_forecast_url.strip() or None
    price_forecast_zone = os.environ.get("PRICE_FORECAST_ZONE", _default("price_forecast_zone")) or "DK1"
    price_forecast_zone = price_forecast_zone.strip() or "DK1"
    price_forecast_eur_to_dkk = float(
        os.environ.get("PRICE_FORECAST_EUR_TO_DKK", _default("price_forecast_eur_to_dkk"))
    )
    low_sell_price_threshold_dkk = float(
        os.environ.get("LOW_SELL_PRICE_THRESHOLD_DKK", _default("low_sell_price_threshold_dkk"))
    )
    price_future_margin_dkk = float(
        os.environ.get("PRICE_FUTURE_MARGIN_DKK", _default("price_future_margin_dkk"))
    )
    low_price_reserve_pct = float(os.environ.get("LOW_PRICE_RESERVE_PCT", _default("low_price_reserve_pct")))
    flat_price_reserve_pct = float(os.environ.get("FLAT_PRICE_RESERVE_PCT", _default("flat_price_reserve_pct")))

    price_buy_sensor = os.environ.get("PRICE_BUY_SENSOR", _default("price_buy_sensor"))
    price_sell_sensor = os.environ.get("PRICE_SELL_SENSOR", _default("price_sell_sensor"))
    battery_soc_sensor = os.environ.get("BATTERY_SOC_SENSOR", _default("battery_soc_sensor"))
    ev_start_sensor = os.environ.get("EV_START_SENSOR", _default("ev_start_sensor"))
    ev_end_sensor = os.environ.get("EV_END_SENSOR", _default("ev_end_sensor"))
    ev_soc_sensor = os.environ.get("EV_SOC_SENSOR", _default("ev_soc_sensor"))
    ev_target_soc_sensor = os.environ.get("EV_TARGET_SOC_SENSOR", _default("ev_target_soc_sensor"))
    ev_default_departure_sensor = os.environ.get(
        "EV_DEFAULT_DEPARTURE_SENSOR", _default("ev_default_departure_sensor")
    )
    ev_status_sensor = os.environ.get("EV_STATUS_SENSOR", _default("ev_status_sensor"))
    ev_planned_kwh_sensor = os.environ.get("EV_PLANNED_KWH_SENSOR", _default("ev_planned_kwh_sensor"))
    ev_arrival_soc_sensor = os.environ.get("EV_ARRIVAL_SOC_SENSOR", _default("ev_arrival_soc_sensor"))
    ev_departure_soc_sensor = os.environ.get("EV_DEPARTURE_SOC_SENSOR", _default("ev_departure_soc_sensor"))
    house_expected_daily_kwh_sensor = os.environ.get("HOUSE_EXPECTED_DAILY_KWH_SENSOR", _default("house_expected_daily_kwh_sensor"))
    # Optional cumulative sensors
    house_total_sensor = os.environ.get("HOUSE_TOTAL_SENSOR", _default("house_total_sensor"))
    pv_total_sensor = os.environ.get("PV_TOTAL_SENSOR", _default("pv_total_sensor"))
    battery_charge_total_sensor = os.environ.get("BATTERY_CHARGE_TOTAL_SENSOR", _default("battery_charge_total_sensor"))
    battery_discharge_total_sensor = os.environ.get("BATTERY_DISCHARGE_TOTAL_SENSOR", _default("battery_discharge_total_sensor"))
    grid_buy_total_sensor = os.environ.get("GRID_BUY_TOTAL_SENSOR", _default("grid_buy_total_sensor"))
    grid_sell_total_sensor = os.environ.get("GRID_SELL_TOTAL_SENSOR", _default("grid_sell_total_sensor"))

    # Optional recorder-based quarter-hour sensors (kWh per slot)
    ev_qh_sensor = os.environ.get("EV_QH_SENSOR")
    pv_qh_sensor = os.environ.get("PV_QH_SENSOR")
    grid_buy_qh_sensor = os.environ.get("GRID_BUY_QH_SENSOR")
    grid_sell_qh_sensor = os.environ.get("GRID_SELL_QH_SENSOR")
    battery_charge_qh_sensor = os.environ.get("BATTERY_CHARGE_QH_SENSOR")
    battery_discharge_qh_sensor = os.environ.get("BATTERY_DISCHARGE_QH_SENSOR")

    # Optional user control sensors
    optimistic_charging_pct_sensor = os.environ.get(
        "OPTIMISTIC_CHARGING_PCT_SENSOR",
        Settings.__dataclass_fields__["optimistic_charging_pct_sensor"].default,
    )
    cheap_price_threshold_sensor = os.environ.get(
        "CHEAP_PRICE_THRESHOLD_SENSOR",
        Settings.__dataclass_fields__["cheap_price_threshold_sensor"].default,
    )
    planning_profile_sensor = os.environ.get(
        "PLANNING_PROFILE_SENSOR",
        Settings.__dataclass_fields__["planning_profile_sensor"].default,
    )
    allow_multi_windows_switch = os.environ.get(
        "ALLOW_MULTI_WINDOWS_SWITCH",
        Settings.__dataclass_fields__["allow_multi_windows_switch"].default,
    )
    battery_capacity_sensor = os.environ.get(
        "BATTERY_CAPACITY_SENSOR",
        Settings.__dataclass_fields__["battery_capacity_sensor"].default,
    )
    battery_charge_power_sensor = os.environ.get(
        "BATTERY_CHARGE_POWER_SENSOR",
        Settings.__dataclass_fields__["battery_charge_power_sensor"].default,
    )
    battery_discharge_power_sensor = os.environ.get(
        "BATTERY_DISCHARGE_POWER_SENSOR",
        Settings.__dataclass_fields__["battery_discharge_power_sensor"].default,
    )
    battery_min_soc_sensor = os.environ.get(
        "BATTERY_MIN_SOC_SENSOR",
        Settings.__dataclass_fields__["battery_min_soc_sensor"].default,
    )
    battery_max_soc_sensor = os.environ.get(
        "BATTERY_MAX_SOC_SENSOR",
        Settings.__dataclass_fields__["battery_max_soc_sensor"].default,
    )

    use_linear_solver = os.environ.get("USE_LINEAR_SOLVER", "false").strip().lower() in {"1", "true", "yes", "on"}
    use_ml_forecast = os.environ.get("USE_ML_FORECAST", "false").strip().lower() in {"1", "true", "yes", "on"}

    return Settings(
        ha_base_url=base_url,
        ha_api_key=api_key,
        mariadb_dsn=mariadb_dsn,
        db_enabled=db_enabled,
        db_url=db_url,
        timezone=timezone,
        lookahead_hours=lookahead,
        resolution_minutes=resolution,
        consumption_history_hours=consumption_history_hours,
        learning_history_days=max(1, learning_history_days),
        battery_reserve_bias=max(0.0, min(1.0, reserve_bias)),
        grid_sell_price_multiplier=max(0.0, min(1.0, grid_sell_price_multiplier)),
        grid_sell_penalty_dkk_per_kwh=max(0.0, grid_sell_penalty_dkk_per_kwh),
        ev_default_daily_kwh=max(0.0, ev_default_daily_kwh),
        ev_planned_kwh_sensor=_clean_sensor(ev_planned_kwh_sensor),
        ev_arrival_soc_sensor=_clean_sensor(ev_arrival_soc_sensor),
        ev_departure_soc_sensor=_clean_sensor(ev_departure_soc_sensor),
        house_expected_daily_kwh_sensor=_clean_sensor(house_expected_daily_kwh_sensor),
        ev_energy_sensor=ev_energy_sensor,
        ev_energy_history_days=max(1, ev_energy_history_days),
        future_extra_load_sensor=future_extra_load_sensor,
        ev_future_daily_buffer_kwh=max(0.0, ev_future_daily_buffer_kwh),
        ev_window_local_buffer_pct=max(0.0, min(1.0, ev_window_local_buffer_pct)),
        house_consumption_sensor=consumption_sensor,
        price_buy_sensor=price_buy_sensor,
        price_sell_sensor=price_sell_sensor,
        battery_soc_sensor=battery_soc_sensor,
        ev_start_sensor=ev_start_sensor,
        ev_end_sensor=ev_end_sensor,
        ev_soc_sensor=ev_soc_sensor,
        ev_target_soc_sensor=ev_target_soc_sensor,
        ev_default_departure_sensor=_clean_sensor(ev_default_departure_sensor),
        ev_status_sensor=ev_status_sensor,
        ev_planning_switch=_clean_sensor(ev_planning_switch),
        price_forecast_url=price_forecast_url,
        price_forecast_zone=price_forecast_zone,
        price_forecast_eur_to_dkk=max(0.0, price_forecast_eur_to_dkk),
        low_sell_price_threshold_dkk=max(0.0, low_sell_price_threshold_dkk),
        price_future_margin_dkk=max(0.0, price_future_margin_dkk),
        low_price_reserve_pct=max(0.0, min(1.0, low_price_reserve_pct)),
        flat_price_reserve_pct=max(0.0, min(1.0, flat_price_reserve_pct)),
        house_total_sensor=_clean_sensor(house_total_sensor),
        pv_total_sensor=_clean_sensor(pv_total_sensor),
        battery_charge_total_sensor=_clean_sensor(battery_charge_total_sensor),
        battery_discharge_total_sensor=_clean_sensor(battery_discharge_total_sensor),
        grid_buy_total_sensor=_clean_sensor(grid_buy_total_sensor),
        grid_sell_total_sensor=_clean_sensor(grid_sell_total_sensor),
        ev_qh_sensor=_clean_sensor(ev_qh_sensor),
        pv_qh_sensor=_clean_sensor(pv_qh_sensor),
        grid_buy_qh_sensor=_clean_sensor(grid_buy_qh_sensor),
        grid_sell_qh_sensor=_clean_sensor(grid_sell_qh_sensor),
        battery_charge_qh_sensor=_clean_sensor(battery_charge_qh_sensor),
        battery_discharge_qh_sensor=_clean_sensor(battery_discharge_qh_sensor),
        optimistic_charging_pct_sensor=_clean_sensor(optimistic_charging_pct_sensor),
        cheap_price_threshold_sensor=_clean_sensor(cheap_price_threshold_sensor),
        planning_profile_sensor=_clean_sensor(planning_profile_sensor),
        allow_multi_windows_switch=_clean_sensor(allow_multi_windows_switch),
        battery_capacity_sensor=_clean_sensor(battery_capacity_sensor),
        battery_charge_power_sensor=_clean_sensor(battery_charge_power_sensor),
        battery_discharge_power_sensor=_clean_sensor(battery_discharge_power_sensor),
        battery_min_soc_sensor=_clean_sensor(battery_min_soc_sensor),
        battery_max_soc_sensor=_clean_sensor(battery_max_soc_sensor),
        use_linear_solver=use_linear_solver,
        use_ml_forecast=use_ml_forecast,
    )


__all__ = ["Settings", "load_settings", "get_loaded_env_files"]
