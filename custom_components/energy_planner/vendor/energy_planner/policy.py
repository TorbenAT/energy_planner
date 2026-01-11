"""Adaptive policy helpers for dynamic battery and EV planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sqlalchemy import select  # type: ignore

from .config import Settings
from .constants import (
    BATTERY_CAPACITY_KWH,
    BATTERY_MIN_SOC_KWH,
    BATTERY_CYCLE_COST_DKK_PER_KWH,
    EV_BATTERY_CAPACITY_KWH,
    MAX_EV_CHARGE_KWH,
    MAX_EV_CHARGE_QH,
)
from .db import session_scope
from .learning import LearningSnapshot, load_learning_snapshot
from .models import ActualQuarterHour
from .utils.time import ensure_timezone


ALLOWED_EV_STATUSES = {
    "awaiting_start",
    "charging",
    "ready_to_charge",
    "ready",
    "completed",
    "connected",
    "plugged_in",
}


@dataclass(slots=True)
class ChargeRecommendation:
    start: datetime
    end: datetime
    energy_kwh: float
    average_price_dkk: float
    slot_count: int
    percentile: float


@dataclass(slots=True)
class AdaptivePolicy:
    reserve_schedule: List[float]
    sell_price_override: List[float]
    reserve_penalty_per_kwh: float
    expected_ev_daily_kwh: float
    expected_house_daily_kwh: float
    future_extra_load_kwh: float
    ev_buffer_target_kwh: float
    charge_recommendation: Optional[ChargeRecommendation]
    history_sample_days: int
    notes: List[str]
    battery_hold_value_dkk: float = 0.0
    price_buy_high_threshold: float = 0.0
    learned_house_daily_sample_count: int = 0
    learned_ev_daily_sample_count: int = 0
    learned_battery_min_soc_pct: Optional[float] = None
    planned_ev_kwh: float = 0.0
    planned_ev_source: Optional[str] = None
    planned_ev_arrival_soc_pct: Optional[float] = None
    planned_ev_departure_soc_pct: Optional[float] = None
    house_expected_override_kwh: Optional[float] = None
    planned_ev_schedule_day: Optional[str] = None
    dynamic_margin_dkk: List[float] = field(default_factory=list)
    dynamic_low_reserve_pct_series: List[float] = field(default_factory=list)
    battery_hold_value_series: List[float] = field(default_factory=list)
    price_future_min_series: List[float] = field(default_factory=list)
    price_future_max_series: List[float] = field(default_factory=list)
    price_future_p75_series: List[float] = field(default_factory=list)
    price_future_std_series: List[float] = field(default_factory=list)
    wait_flags: List[bool] = field(default_factory=list)
    wait_reasons: List[Optional[str]] = field(default_factory=list)
    slot_diagnostics: List[dict] = field(default_factory=list)
    # Recommendation for vehicle charge limit (Tesla charge limit), based on price outlook and weekly EV needs
    recommended_ev_limit_pct: Optional[float] = None
    recommended_ev_limit_kwh: Optional[float] = None
    recommended_ev_limit_reason: Optional[str] = None


def compute_adaptive_policy(
    forecast: pd.DataFrame,
    settings: Settings,
    ha_client,
    SessionFactory,
    now: datetime,
    battery_soc_kwh: float,
    ev_required_kwh: float,
    ev_window: Tuple[int, int],
    ev_status: str,
    ev_planning_disabled: bool,
) -> AdaptivePolicy:
    period_minutes = max(settings.resolution_minutes, 1)
    period_hours = period_minutes / 60.0
    tz = settings.timezone

    prices_buy = forecast["price_buy"].astype(float).fillna(0.0)
    prices_sell = forecast["price_sell"].astype(float).fillna(0.0)
    price_signal_candidate = forecast.get("price_buy_signal")
    if isinstance(price_signal_candidate, pd.Series):
        price_signal_series = price_signal_candidate.astype(float).ffill().fillna(prices_buy)
    else:
        price_signal_series = prices_buy

    price_allin_series = forecast.get("price_allin_buy", price_signal_series).astype(float).fillna(0.0)
    future_min_series = forecast.get("price_future_min", price_allin_series).astype(float).fillna(price_allin_series.min())
    future_max_series = forecast.get("price_future_max", price_allin_series).astype(float).fillna(price_allin_series.max())
    future_p75_series = forecast.get("price_future_p75", price_allin_series).astype(float).fillna(price_allin_series.quantile(0.75))
    future_std_series = forecast.get("price_future_std", pd.Series([0.0] * len(forecast))).astype(float).fillna(0.0)
    consumption_kw = forecast["consumption_estimate_kw"].astype(float).fillna(0.0)
    production_kw = forecast["pv_forecast_kw"].astype(float).fillna(0.0)

    notes: List[str] = []
    house_samples = 0
    ev_samples = 0
    planned_ev_kwh = 0.0
    planned_ev_source: Optional[str] = None
    house_override: Optional[float] = None
    planned_arrival_soc_pct: Optional[float] = None
    planned_departure_soc_pct: Optional[float] = None
    planned_schedule_day: Optional[str] = None

    def _read_numeric(sensor: Optional[str], clamp: Optional[Tuple[float, float]] = None) -> Optional[float]:
        if not sensor:
            return None
        try:
            value = ha_client.fetch_numeric_state(sensor)
        except Exception as exc:  # pragma: no cover - defensive path
            notes.append(f"Kunne ikke læse {sensor}: {exc}")
            return None
        if value is None:
            return None
        number = float(value)
        if clamp is not None:
            low, high = clamp
            number = max(low, min(high, number))
        return number

    planned_arrival_soc_pct = _read_numeric(settings.ev_arrival_soc_sensor, (0.0, 100.0))
    planned_departure_soc_pct = _read_numeric(settings.ev_departure_soc_sensor, (0.0, 100.0))

    house_avg, sample_days = _estimate_house_daily_average(
        SessionFactory,
        now,
        settings,
        period_hours,
        tz,
        fallback=float(consumption_kw.sum() * period_hours / max(len(forecast) * period_hours / 24.0, 1e-6))
        if len(forecast)
        else 0.0,
    )
    if sample_days == 0:
        notes.append("Household history unavailable; using forecast-derived average.")

    manual_house = _read_numeric(settings.house_expected_daily_kwh_sensor, (0.0, 500.0))
    if manual_house is not None and manual_house > 0:
        house_override = manual_house
        house_avg = house_override
        notes.append(f"Husforbrug sat til ca. {house_override:.1f} kWh/døgn via sensor.")

    if planned_arrival_soc_pct is not None or planned_departure_soc_pct is not None:
        arrival_display = f"{planned_arrival_soc_pct:.0f}%" if planned_arrival_soc_pct is not None else "?"
        departure_display = (
            f"{planned_departure_soc_pct:.0f}%" if planned_departure_soc_pct is not None else "?"
        )
        notes.append(f"EV SOC plan: parkering ≈ {arrival_display}, afgang ≈ {departure_display}.")

    def _to_float(value: object, clamp: Optional[Tuple[float, float]] = None) -> Optional[float]:
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        if clamp is not None:
            low, high = clamp
            number = max(low, min(high, number))
        return number

    raw_planned_ev = None
    if settings.ev_planned_kwh_sensor:
        try:
            planned_payload = ha_client.fetch_state(settings.ev_planned_kwh_sensor)
        except Exception as exc:  # pragma: no cover - defensive path
            planned_payload = None
            notes.append(f"Kunne ikke læse {settings.ev_planned_kwh_sensor}: {exc}")
        if isinstance(planned_payload, dict):
            raw_planned_ev = _to_float(planned_payload.get("state"), (0.0, EV_BATTERY_CAPACITY_KWH))
            attributes = planned_payload.get("attributes")
            if isinstance(attributes, dict):
                inferred_source = attributes.get("source")
                schedule_day = attributes.get("schedule_day")
                attr_arrival = _to_float(attributes.get("arrival_soc_pct"), (0.0, 100.0))
                attr_departure = _to_float(attributes.get("departure_soc_pct"), (0.0, 100.0))

                if attr_arrival is not None and (planned_arrival_soc_pct is None or planned_arrival_soc_pct <= 0):
                    planned_arrival_soc_pct = attr_arrival
                if attr_departure is not None and (planned_departure_soc_pct is None or planned_departure_soc_pct <= 0):
                    planned_departure_soc_pct = attr_departure
                if schedule_day:
                    planned_schedule_day = str(schedule_day)
                if raw_planned_ev is not None and raw_planned_ev > 0:
                    planned_ev_kwh = raw_planned_ev
                    planned_ev_source = str(inferred_source or settings.ev_planned_kwh_sensor)
            elif raw_planned_ev is not None and raw_planned_ev > 0:
                planned_ev_kwh = raw_planned_ev
                planned_ev_source = settings.ev_planned_kwh_sensor

    if raw_planned_ev is None:
        raw_planned_ev = _read_numeric(settings.ev_planned_kwh_sensor, (0.0, EV_BATTERY_CAPACITY_KWH))
        if raw_planned_ev is not None and raw_planned_ev > 0:
            planned_ev_kwh = raw_planned_ev
            planned_ev_source = settings.ev_planned_kwh_sensor

    if planned_ev_kwh > 0:
        source_label_raw = planned_ev_source or settings.ev_planned_kwh_sensor or "sensor"
        friendly_sources = {
            "vehicle_soc": "bilens SOC",
            "vehicle": "bilens SOC",
            "soc_delta": "bilens SOC",
            "weekly_schedule": "ugeskema",
            "manual": "manuel kWh",
            "auto": "auto",
        }
        source_label = friendly_sources.get(str(planned_ev_source).lower(), source_label_raw.replace("_", " "))
        if planned_schedule_day and planned_schedule_day.lower() != "ingen":
            notes.append(
                f"Planlagt EV opladning {planned_ev_kwh:.1f} kWh via {source_label} (dag: {planned_schedule_day})."
            )
        else:
            notes.append(f"Planlagt EV opladning {planned_ev_kwh:.1f} kWh via {source_label}.")

    ev_daily_estimate, ev_history_note = _estimate_ev_daily_average(
        ha_client,
        settings,
        now,
        tz,
    )
    if ev_history_note:
        notes.append(ev_history_note)

    # Optional: read weekly EV kWh estimates from Home Assistant inputs (one per weekday)
    # and apply today's value to expected_ev_daily_kwh with explicit logging.
    try:
        weekday_keys = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        idx = ensure_timezone(now, tz).weekday()  # 0=Mon
        key = weekday_keys[idx]
        weekly_map = {}
        if hasattr(ha_client, "fetch_weekly_ev_kwh_inputs"):
            weekly_map = ha_client.fetch_weekly_ev_kwh_inputs() or {}
        else:
            # Fallback: query only today's key if helper is unavailable
            entity_id = f"input_number.energy_planner_ev_week_{key}_kwh"
            val = ha_client.fetch_numeric_state(entity_id)
            weekly_map = {key: float(val) if (val is not None) else 0.0}

        today_value = float(weekly_map.get(key, 0.0) or 0.0)
        if today_value > 0:
            ev_daily_estimate = today_value
            entity_id_today = getattr(ha_client, "weekly_ev_entity_id", lambda k: f"input_number.energy_planner_ev_week_{k}_kwh")(key)
            # EXACT logging line as required for verifiable linkage
            notes.append(f"Weekly EV input used: {today_value:.1f} kWh (source: {entity_id_today})")
    except Exception:
        # Non-fatal: inputs may not exist; keep learned/default estimates
        pass

    extra_load_kwh, extra_note = _fetch_future_extra_load(ha_client, settings)
    if extra_note:
        notes.append(extra_note)

    try:
        learning_snapshot = load_learning_snapshot(SessionFactory)
    except Exception:  # pragma: no cover - defensive fallback
        learning_snapshot = LearningSnapshot(metrics={})
        notes.append("Learning metrics unavailable; using history-based defaults.")

    def _blend(existing: float, learned: float, samples: int, max_samples: int = 30) -> float:
        if learned <= 0:
            return existing
        if existing <= 0:
            return learned
        weight = min(1.0, samples / max_samples) if samples > 0 else 0.5
        return (1.0 - weight) * existing + weight * learned

    weekday = ensure_timezone(now, tz).weekday()
    learned_house = learning_snapshot.house_daily_kwh(weekday)
    if learned_house is not None and learned_house > 0 and house_override is None:
        house_samples = learning_snapshot.sample_count("house_daily_kwh", f"weekday_{weekday}")
        house_avg = _blend(house_avg, learned_house, house_samples)
        notes.append(
            f"Learned weekday consumption approx. {learned_house:.1f} kWh (samples={house_samples})."
        )

    learned_ev = learning_snapshot.ev_daily_kwh()
    if learned_ev is not None and learned_ev > 0:
        ev_samples = learning_snapshot.sample_count("ev_daily_kwh", "global")
        ev_daily_estimate = _blend(ev_daily_estimate, learned_ev, ev_samples)
    learned_ev_note = learned_ev if learned_ev is not None else 0.0
    notes.append(f"Learned EV daily buffer approx. {learned_ev_note:.1f} kWh (samples={ev_samples}).")

    price_high_threshold = learning_snapshot.price_buy_high_threshold(0.0)
    learned_min_soc_pct = learning_snapshot.battery_min_soc_pct(None)

    price_allin_np = price_allin_series.to_numpy(dtype=float)
    # Use actual buy price from Home Assistant as the wait-logic basis so comparisons match grid cost perception
    price_wait_np = prices_buy.to_numpy(dtype=float)
    future_min_np = future_min_series.to_numpy(dtype=float)
    future_max_np = future_max_series.to_numpy(dtype=float)
    future_p75_np = future_p75_series.to_numpy(dtype=float)
    future_std_np = future_std_series.to_numpy(dtype=float)

    margin_from_quartile = np.nan_to_num(future_p75_np - future_min_np, nan=0.0)
    std_component = np.nan_to_num(future_std_np, nan=0.0)
    dynamic_margin_np = np.maximum.reduce([np.zeros_like(price_allin_np), margin_from_quartile, std_component])

    # Optional user controls from Home Assistant helpers
    optimism_factor = 0.0
    user_cheap_threshold = None
    profile_factor = 1.0
    try:
        if settings.optimistic_charging_pct_sensor:
            val = ha_client.fetch_numeric_state(settings.optimistic_charging_pct_sensor)
            if val is not None:
                optimism_factor = float(max(0.0, min(100.0, val))) / 100.0
        if settings.cheap_price_threshold_sensor:
            thr = ha_client.fetch_numeric_state(settings.cheap_price_threshold_sensor)
            if thr is not None and thr > 0:
                user_cheap_threshold = float(thr)
        if settings.planning_profile_sensor:
            prof_state = ha_client.fetch_string_state(settings.planning_profile_sensor) or "balanced"
            prof = str(prof_state).strip().lower()
            if prof in ("conservative", "konservativ"):
                profile_factor = 1.4
            elif prof in ("aggressive", "aggressiv"):
                profile_factor = 0.75
            else:
                profile_factor = 1.0
            notes.append(f"Planning profile: {prof} (margin x{profile_factor:.2f}).")
    except Exception:
        # Non-fatal if helpers are missing
        pass

    if profile_factor != 1.0:
        dynamic_margin_np = dynamic_margin_np * profile_factor

    reserve_shortfall_avg_d1 = learning_snapshot.get("reserve_shortfall_avg_d1", "global", 0.0) or 0.0
    if reserve_shortfall_avg_d1 > 1e-6:
        dynamic_margin_np *= 1.25
    else:
        dynamic_margin_np *= 0.75
    dynamic_margin_np = np.clip(dynamic_margin_np, 0.0, None)

    battery_hold_series_np = np.nan_to_num(future_min_np, nan=0.0) + dynamic_margin_np
    hold_value = float(battery_hold_series_np[0]) if battery_hold_series_np.size else 0.0

    denominator = np.maximum(future_max_np - future_min_np, 1e-6)
    dynamic_low_reserve_np = np.nan_to_num((price_allin_np - future_min_np) / denominator, nan=0.0)
    consumption_total_kwh = float(consumption_kw.sum() * period_hours)
    pv_total_kwh = float(production_kw.sum() * period_hours)
    pv_deficit_kwh = consumption_total_kwh - pv_total_kwh
    pv_ratio = 0.0
    if consumption_total_kwh > 1e-6:
        pv_ratio = max(0.0, min(1.0, pv_total_kwh / consumption_total_kwh))
    if pv_deficit_kwh > 0:
        dynamic_low_reserve_np = np.clip(dynamic_low_reserve_np + 0.1, 0.0, 1.0)
    elif pv_deficit_kwh < 0:
        dynamic_low_reserve_np = np.clip(dynamic_low_reserve_np - 0.1, 0.0, 1.0)
    else:
        dynamic_low_reserve_np = np.clip(dynamic_low_reserve_np, 0.0, 1.0)
    # Winter scaling: fewer soltimer -> lower reserve bias
    winter_scale = 0.35 + 0.65 * pv_ratio
    dynamic_low_reserve_np = np.clip(dynamic_low_reserve_np * winter_scale, 0.0, 1.0)

    # Size of the forward-looking window (12 hours). Use exact step count based on current resolution.
    window_slots = max(1, int(round(12.0 / period_hours)))
    wait_flags: List[bool] = []
    wait_reasons: List[Optional[str]] = []

    def _safe_float(value: float) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return float(value)

    slot_diagnostics: List[dict] = []
    count = len(price_allin_np)
    for idx in range(count):
        price_now_wait = price_wait_np[idx] if idx < len(price_wait_np) else float("nan")
        wait_flag = False
        wait_reason: Optional[str] = None
        start = idx + 1
        end = min(count, idx + 1 + window_slots)
        min_future_12: Optional[float] = None

        if start < end and not math.isnan(price_now_wait):
            future_window = price_wait_np[start:end]
            future_window = future_window[~np.isnan(future_window)]
            if future_window.size > 0:
                min_future_12 = float(np.min(future_window))
                # Margin uses same basis on both sides (actual buy price)
                if (min_future_12 + 0.10) < (price_now_wait - 1e-6):
                    wait_flag = True
                    wait_reason = "wait_cheaper"

        # Apply user cheap price threshold if set: wait whenever current price is above threshold
        if user_cheap_threshold is not None and not math.isnan(price_now_wait):
            if price_now_wait > user_cheap_threshold + 1e-6:
                wait_flag = True
                wait_reason = (wait_reason or "") + (";" if wait_reason else "") + "user_threshold"
            else:
                # If under threshold, don't wait
                wait_flag = False
                wait_reason = "user_threshold_ok"

        wait_flags.append(wait_flag)
        wait_reasons.append(wait_reason)
        # Diagnostics: explicitly state the price basis and the values actually used
        slot_diag = {
            "policy_price_basis": "buy",
            "policy_price_now_dkk": _safe_float(price_now_wait),
            # Use 12h window minimum for clarity; also retain horizon stats below
            "policy_future_min_dkk": _safe_float(min_future_12 if min_future_12 is not None else future_min_np[idx]),
            "policy_future_min_12h_dkk": _safe_float(min_future_12) if min_future_12 is not None else None,
            "policy_future_max_dkk": _safe_float(future_max_np[idx]),
            "policy_future_p75_dkk": _safe_float(future_p75_np[idx]),
            "policy_price_std_dkk": _safe_float(future_std_np[idx]),
            "policy_dynamic_margin_dkk": _safe_float(dynamic_margin_np[idx]),
            "policy_dynamic_low_reserve_pct": _safe_float(dynamic_low_reserve_np[idx]),
            "policy_wait_flag": wait_flag,
            "policy_wait_reason": wait_reason,
            "policy_hold_value_dkk": _safe_float(battery_hold_series_np[idx]),
            "policy_grid_charge_allowed": (not wait_flag),
        }
        slot_diagnostics.append(slot_diag)

    if hold_value > 0 and battery_hold_series_np.size:
        notes.append(
            f"Dynamic hold value ≈ {hold_value:.2f} DKK/kWh (future min {future_min_np[0]:.2f} + margin {dynamic_margin_np[0]:.2f})."
        )
    if price_high_threshold > hold_value and price_high_threshold > 0:
        notes.append(f"High price threshold learned approx. {price_high_threshold:.2f} DKK/kWh.")

    buffer_candidates: List[float] = [max(0.0, ev_required_kwh)]
    if planned_ev_kwh > 0:
        buffer_candidates.append(planned_ev_kwh)
    if settings.ev_future_daily_buffer_kwh > 0:
        buffer_candidates.append(settings.ev_future_daily_buffer_kwh)
    if ev_daily_estimate > 0:
        buffer_candidates.append(ev_daily_estimate)
    elif settings.ev_default_daily_kwh > 0 and planned_ev_kwh <= 0 and ev_required_kwh <= 0:
        buffer_candidates.append(settings.ev_default_daily_kwh)

    ev_buffer = max(buffer_candidates) if any(value > 0 for value in buffer_candidates) else 0.0
    # Optimistic charging: add a fraction of the expected daily EV need to the buffer
    if optimism_factor > 0 and ev_daily_estimate > 0:
        extra = float(ev_daily_estimate) * float(optimism_factor)
        ev_buffer = min(EV_BATTERY_CAPACITY_KWH, ev_buffer + extra)
        notes.append(f"Optimistic charging +{extra:.1f} kWh (factor {optimism_factor:.2f}).")
    ev_buffer = min(ev_buffer, EV_BATTERY_CAPACITY_KWH)
    reserve_schedule = _build_reserve_schedule(
        consumption_kw,
        production_kw,
        period_hours,
        settings,
        extra_load_kwh,
    )

    capacity_span = max(0.0, BATTERY_CAPACITY_KWH - BATTERY_MIN_SOC_KWH)
    seasonal_fraction = max(0.35, min(1.0, pv_ratio))
    seasonal_cap = (
        BATTERY_MIN_SOC_KWH + capacity_span * seasonal_fraction if capacity_span > 0 else BATTERY_MIN_SOC_KWH
    )
    max_buffer_allowed = max(0.0, seasonal_cap - BATTERY_MIN_SOC_KWH)

    planned_buffer_kwh = min(max(ev_buffer, 0.0), max_buffer_allowed)
    if max_buffer_allowed + 1e-6 < ev_buffer:
        notes.append(
            f"EV-buffer trimmed til ca. {planned_buffer_kwh:.1f} kWh (sæsonmaks {seasonal_cap:.1f} kWh)."
        )

    min_buffer_target = BATTERY_MIN_SOC_KWH + planned_buffer_kwh
    min_buffer_target = min(seasonal_cap, max(BATTERY_MIN_SOC_KWH, min_buffer_target))
    pre_window_target = min(seasonal_cap, max(BATTERY_MIN_SOC_KWH, battery_soc_kwh))
    window_start_index, _ = ev_window

    adjusted_schedule: List[float] = []
    for idx, target in enumerate(reserve_schedule):
        capped_target = min(target, seasonal_cap)
        if idx < window_start_index:
            adjusted_schedule.append(max(capped_target, pre_window_target))
        else:
            adjusted_schedule.append(max(capped_target, min_buffer_target))
    reserve_schedule = adjusted_schedule

    if capacity_span > 0 and seasonal_cap < BATTERY_CAPACITY_KWH - 0.1:
        notes.append(f"Winter reserve scaled til ca. {seasonal_cap:.1f} kWh (PV-forhold {pv_ratio:.2f}).")
    ev_buffer_pct = (planned_buffer_kwh / capacity_span) if capacity_span > 0 else 0.0
    current_soc_pct = (battery_soc_kwh / BATTERY_CAPACITY_KWH) if BATTERY_CAPACITY_KWH > 0 else 0.0

    reserve_schedule, price_notes, slot_diagnostics = _apply_price_guidance(
        reserve_schedule,
        price_allin_series,
        prices_sell,
        settings,
        battery_soc_kwh,
        dynamic_low_reserve_np,
        wait_flags,
        wait_reasons,
        ev_buffer_pct,
        current_soc_pct,
        slot_diagnostics,
    )
    notes.extend(price_notes)

    if learned_min_soc_pct is not None and BATTERY_CAPACITY_KWH > 0:
        min_target_kwh = max(
            BATTERY_MIN_SOC_KWH,
            min(BATTERY_CAPACITY_KWH, BATTERY_CAPACITY_KWH * learned_min_soc_pct / 100.0),
        )
        if min_target_kwh > BATTERY_MIN_SOC_KWH:
            reserve_schedule = [max(target, min_target_kwh) for target in reserve_schedule]
            notes.append(f"Learning raised minimum battery target to approx. {min_target_kwh:.1f} kWh.")

    if reserve_schedule and battery_soc_kwh < reserve_schedule[0] - 0.5:
        notes.append("Battery SoC below adaptive reserve; planner will top up when prices allow.")

    if ev_status:
        normalized_status = ev_status.strip().lower().replace(" ", "_").replace("-", "_")
        if normalized_status not in ALLOWED_EV_STATUSES and ev_buffer > 0.5:
            readable_status = normalized_status.replace("_", " ") or "ukendt"
            notes.append(f"EV ikke klar til opladning (status: {readable_status}). Holder buffer til senere.")

    reserve_penalty = _compute_reserve_penalty(price_signal_series, reserve_schedule)
    penalty_floor = max(hold_value, price_high_threshold, 0.0)
    penalty_cap = penalty_floor + BATTERY_CYCLE_COST_DKK_PER_KWH + 0.25 if penalty_floor > 0 else None
    if penalty_cap is not None and penalty_cap > 0:
        reserve_penalty = min(reserve_penalty, penalty_cap)
    reserve_penalty = max(reserve_penalty, penalty_floor)

    sell_override = _build_sell_price_override(
        price_signal_series,
        prices_sell,
        consumption_kw,
        production_kw,
        period_hours,
        ev_buffer + extra_load_kwh,
        hold_value,
    )

    recommended_charge = _suggest_charge_window(
        forecast,
        period_minutes,
        ev_buffer,
        ev_required_kwh,
        ev_window,
        ev_planning_disabled,
    )

    if recommended_charge is None and ev_buffer > 0:
        notes.append("No cheap charge window identified; keep buffer available.")

    # Heuristic: recommend an EV charge limit (%) for the vehicle app (e.g., Tesla)
    # Inputs: current EV SOC, weekly EV kWh expectations for the next ~3 days, and
    # a quick look at the next 72h price outlook. Purpose: give a simple, actionable
    # target even when the EV is disconnected (advice mode).
    rec_ev_limit_pct: Optional[float] = None
    rec_ev_limit_kwh: Optional[float] = None
    rec_ev_limit_reason: Optional[str] = None

    try:
        # Current EV state of charge
        ev_soc_pct = _read_numeric(settings.ev_soc_sensor, (0.0, 100.0)) or 0.0
        ev_soc_kwh_now = EV_BATTERY_CAPACITY_KWH * (float(ev_soc_pct) / 100.0)

        # Sum weekly EV kWh expectations for today + next 2 days (3-day horizon)
        # If weekly inputs are missing, fall back to the learned/default daily buffer.
        weekday_keys = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        now_local = ensure_timezone(now, tz)
        needs_next3 = 0.0
        need_breakdown: List[str] = []
        for offset in range(0, 3):
            idx = (now_local.weekday() + offset) % 7
            key = weekday_keys[idx]
            entity_id = f"input_number.energy_planner_ev_week_{key}_kwh"
            val = ha_client.fetch_numeric_state(entity_id)
            val_f = float(val) if (val is not None) else 0.0
            if val_f <= 0:
                # fallback to the policy estimate if weekly for that day is not set
                val_f = float(ev_daily_estimate) if ev_daily_estimate > 0 else 0.0
            needs_next3 += max(0.0, val_f)
            need_breakdown.append(f"{key}:{val_f:.1f}")

        # Determine cheapest time in the next 72 hours and whether it lies inside today's EV window
        # This is purely informational for the reason string.
        period_hours = max(1, period_minutes) / 60.0
        slots_72h = max(1, int(round(72.0 / period_hours)))
        price_slice = forecast[["timestamp", "price_buy"]].copy()
        price_slice["price_buy"] = price_slice["price_buy"].astype(float).fillna(price_slice["price_buy"].mean())
        price_slice = price_slice.iloc[: min(slots_72h, len(price_slice))]
        cheapest_idx: Optional[int] = None
        cheap_in_window = False
        cheapest_ts_str = ""
        if not price_slice.empty:
            cheapest_idx = int(price_slice["price_buy"].idxmin())
            try:
                cheapest_ts = pd.to_datetime(price_slice.loc[cheapest_idx, "timestamp"])  # type: ignore[index]
                cheapest_ts_str = ensure_timezone(cheapest_ts.to_pydatetime(), tz).isoformat()
            except Exception:
                cheapest_ts_str = ""
            try:
                win_start, win_end = ev_window
                cheap_in_window = (cheapest_idx is not None) and (win_start <= cheapest_idx < win_end)
            except Exception:
                cheap_in_window = False

        # Compute a simple, safe recommendation:
        # - Ensure enough energy for the next ~3 days
        # - Cap to pack capacity
        # - Express as a rounded-up percentage (nearest 5%)
        target_kwh = min(EV_BATTERY_CAPACITY_KWH, max(0.0, ev_soc_kwh_now) + max(0.0, needs_next3))
        if EV_BATTERY_CAPACITY_KWH > 0:
            pct = (target_kwh / EV_BATTERY_CAPACITY_KWH) * 100.0
            # Round up to the nearest 5% to give a clean value for the car app UI
            pct_rounded = float(min(100.0, max(0.0, math.ceil(pct / 5.0) * 5.0)))
        else:
            pct_rounded = 0.0

        # Only publish if the recommendation is above the current SOC by at least ~1%
        if pct_rounded > (ev_soc_pct + 1.0):
            rec_ev_limit_pct = pct_rounded
            rec_ev_limit_kwh = float(target_kwh)
            reason_bits = [
                f"næste~3d behov ≈ {needs_next3:.1f} kWh ({', '.join(need_breakdown)})",
            ]
            if cheapest_ts_str:
                reason_bits.append(
                    f"billigste pris indenfor 72h: {cheapest_ts_str}{' (i vinduet)' if cheap_in_window else ''}"
                )
            rec_ev_limit_reason = ", ".join(reason_bits)
    except Exception:
        # Non-fatal: missing sensors or data; leave recommendation as None
        pass

    dynamic_margin_list = [float(x) for x in dynamic_margin_np.tolist()]
    dynamic_low_reserve_list = [float(np.clip(x, 0.0, 1.0)) for x in dynamic_low_reserve_np.tolist()]
    battery_hold_list = [float(x) for x in battery_hold_series_np.tolist()]
    future_min_list = [float(x) if not math.isnan(x) else None for x in future_min_np.tolist()]
    future_max_list = [float(x) if not math.isnan(x) else None for x in future_max_np.tolist()]
    future_p75_list = [float(x) if not math.isnan(x) else None for x in future_p75_np.tolist()]
    future_std_list = [float(x) if not math.isnan(x) else 0.0 for x in future_std_np.tolist()]

    return AdaptivePolicy(
        reserve_schedule=reserve_schedule,
        sell_price_override=sell_override,
        reserve_penalty_per_kwh=reserve_penalty,
        expected_ev_daily_kwh=ev_daily_estimate,
        expected_house_daily_kwh=house_avg,
        future_extra_load_kwh=extra_load_kwh,
        ev_buffer_target_kwh=planned_ev_kwh if planned_ev_kwh > 0 else ev_buffer,
        charge_recommendation=recommended_charge,
        history_sample_days=sample_days,
        notes=notes,
        battery_hold_value_dkk=hold_value,
        price_buy_high_threshold=price_high_threshold,
        learned_house_daily_sample_count=house_samples,
        learned_ev_daily_sample_count=ev_samples,
        learned_battery_min_soc_pct=learned_min_soc_pct,
        planned_ev_kwh=planned_ev_kwh,
        planned_ev_source=planned_ev_source,
        planned_ev_arrival_soc_pct=planned_arrival_soc_pct,
        planned_ev_departure_soc_pct=planned_departure_soc_pct,
        house_expected_override_kwh=house_override,
        planned_ev_schedule_day=planned_schedule_day,
        dynamic_margin_dkk=dynamic_margin_list,
        dynamic_low_reserve_pct_series=dynamic_low_reserve_list,
        battery_hold_value_series=battery_hold_list,
        price_future_min_series=future_min_list,
        price_future_max_series=future_max_list,
        price_future_p75_series=future_p75_list,
        price_future_std_series=future_std_list,
        wait_flags=wait_flags,
        wait_reasons=wait_reasons,
        slot_diagnostics=slot_diagnostics,
        recommended_ev_limit_pct=rec_ev_limit_pct,
        recommended_ev_limit_kwh=rec_ev_limit_kwh,
        recommended_ev_limit_reason=rec_ev_limit_reason,
    )


def _estimate_house_daily_average(
    SessionFactory,
    now: datetime,
    settings: Settings,
    period_hours: float,
    timezone_name: str,
    fallback: float,
) -> Tuple[float, int]:
    history_days = max(1, settings.learning_history_days)
    start = now - timedelta(days=history_days)
    utc_start = ensure_timezone(start, "UTC").replace(tzinfo=None)

    with session_scope(SessionFactory) as session:
        stmt = (
            select(ActualQuarterHour.timestamp, ActualQuarterHour.consumption_kw)
            .where(ActualQuarterHour.timestamp >= utc_start)
        )
        records = [(row[0], row[1]) for row in session.execute(stmt)]

    if not records:
        return fallback, 0

    df = pd.DataFrame(records, columns=["timestamp", "consumption_kw"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["local_date"] = df["timestamp"].dt.tz_convert(timezone_name).dt.date
    df["kwh"] = df["consumption_kw"].astype(float) * period_hours
    daily = df.groupby("local_date")[["kwh"]].sum()
    if daily.empty:
        return fallback, 0

    usable_days = min(len(daily), history_days)
    average = float(daily.tail(usable_days)["kwh"].mean())
    return average, usable_days


def _estimate_ev_daily_average(
    ha_client,
    settings: Settings,
    now: datetime,
    timezone_name: str,
) -> Tuple[float, Optional[str]]:
    sensor = settings.ev_energy_sensor
    if not sensor:
        return settings.ev_default_daily_kwh, "EV energy sensor not configured; using default buffer."

    history_days = max(1, settings.ev_energy_history_days)
    start = now - timedelta(days=history_days)
    try:
        samples = ha_client.fetch_history_series(sensor, start, now)
    except Exception as exc:  # pragma: no cover - defensive path
        return settings.ev_default_daily_kwh, f"EV history fetch failed ({exc}); using default buffer."

    if not samples:
        return settings.ev_default_daily_kwh, "EV history missing; using default buffer."

    df = pd.DataFrame(samples, columns=["timestamp", "value"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values("timestamp", inplace=True)
    df["local_date"] = df["timestamp"].dt.tz_convert(timezone_name).dt.date

    daily_max = df.groupby("local_date")["value"].max().sort_index()
    if daily_max.empty:
        return settings.ev_default_daily_kwh, "EV history lacked usable samples; using default buffer."

    diffs = daily_max.diff().dropna()
    positive = diffs[diffs >= 0.1]
    if positive.empty:
        positive = daily_max

    window = positive.tail(history_days)
    average = float(window.mean()) if not window.empty else settings.ev_default_daily_kwh
    return max(average, 0.0), None


def _fetch_future_extra_load(ha_client, settings: Settings) -> Tuple[float, Optional[str]]:
    entity = settings.future_extra_load_sensor
    if not entity:
        return 0.0, None
    try:
        value = ha_client.fetch_numeric_state(entity)
    except Exception as exc:  # pragma: no cover - defensive path
        return 0.0, f"Could not read extra-load sensor ({exc}); ignoring."
    if value is None or value <= 0:
        return 0.0, None
    return float(value), f"Holding {value:.1f} kWh for declared future load."


def _build_reserve_schedule(
    consumption_kw: pd.Series,
    production_kw: pd.Series,
    period_hours: float,
    settings: Settings,
    initial_deficit_kwh: float,
) -> List[float]:
    capacity_span = max(0.0, BATTERY_CAPACITY_KWH - BATTERY_MIN_SOC_KWH)
    reserve_schedule: List[float] = []
    remaining_deficit = max(0.0, initial_deficit_kwh)

    net_kwh = (consumption_kw - production_kw) * period_hours

    for idx in range(len(net_kwh) - 1, -1, -1):
        delta = float(net_kwh.iloc[idx])
        if delta > 0:
            remaining_deficit += delta
        else:
            remaining_deficit = max(0.0, remaining_deficit + delta)

        if capacity_span > 0:
            normalized = min(1.0, remaining_deficit / capacity_span)
            reserve_level = settings.battery_reserve_bias + (1.0 - settings.battery_reserve_bias) * normalized
        else:
            reserve_level = 0.0

        target = BATTERY_MIN_SOC_KWH + reserve_level * capacity_span
        target = max(BATTERY_MIN_SOC_KWH, min(BATTERY_CAPACITY_KWH, target))
        reserve_schedule.append(target)

    reserve_schedule.reverse()
    return reserve_schedule


def _apply_price_guidance(
    reserve_schedule: Sequence[float],
    price_allin: pd.Series,
    prices_sell: pd.Series,
    settings: Settings,
    battery_soc_kwh: float,
    dynamic_low_reserve_pct: Sequence[float],
    wait_flags: Sequence[bool],
    wait_reasons: Sequence[Optional[str]],
    ev_buffer_pct: float,
    current_soc_pct: float,
    slot_diagnostics: List[dict],
) -> Tuple[List[float], List[str], List[dict]]:
    schedule = list(reserve_schedule)
    notes: List[str] = []

    if not isinstance(price_allin, pd.Series) or price_allin.empty:
        return schedule, notes, slot_diagnostics

    signal = price_allin.reset_index(drop=True).astype(float)
    sell_series = prices_sell.reset_index(drop=True).astype(float) if isinstance(prices_sell, pd.Series) else pd.Series(dtype=float)
    capacity_span = max(0.0, BATTERY_CAPACITY_KWH - BATTERY_MIN_SOC_KWH)
    ev_buffer_pct = float(np.clip(ev_buffer_pct, 0.0, 1.0)) if capacity_span > 0 else 0.0
    current_soc_pct = float(np.clip(current_soc_pct, 0.0, 1.0))

    wait_noted = False

    for idx, target in enumerate(schedule):
        if idx >= len(signal):
            break
        price_now = float(signal.iat[idx])
        sell_now = float(sell_series.iat[idx]) if idx < len(sell_series) else price_now
        low_pct = float(dynamic_low_reserve_pct[idx]) if idx < len(dynamic_low_reserve_pct) else 0.0
        low_pct = float(np.clip(low_pct, 0.0, 1.0))
        target_pct = max(low_pct, ev_buffer_pct, current_soc_pct)
        reserve_target = BATTERY_MIN_SOC_KWH + target_pct * capacity_span if capacity_span > 0 else BATTERY_MIN_SOC_KWH
        if reserve_target > schedule[idx]:
            schedule[idx] = reserve_target

        if idx < len(slot_diagnostics):
            slot_diagnostics[idx]["policy_reserve_target_pct"] = target_pct
            slot_diagnostics[idx]["policy_reserve_target_kwh"] = reserve_target
            slot_diagnostics[idx]["policy_ev_buffer_pct"] = ev_buffer_pct
            slot_diagnostics[idx]["policy_current_soc_pct"] = current_soc_pct
            slot_diagnostics[idx]["policy_effective_sell_price_dkk"] = sell_now

        if idx < len(wait_flags) and wait_flags[idx] and not wait_noted:
            notes.append("Cheaper slot forecasted; deferring grid charge where possible.")
            wait_noted = True
        if idx < len(slot_diagnostics) and idx < len(wait_reasons):
            slot_diagnostics[idx]["policy_wait_reason"] = wait_reasons[idx]

    return schedule, notes, slot_diagnostics


def _compute_reserve_penalty(prices_buy: pd.Series, reserve_schedule: Sequence[float]) -> float:
    if prices_buy.empty:
        return 0.0
    baseline = float(prices_buy.quantile(0.65))
    if baseline <= 0:
        return 0.0
    utilization = 0.0
    if reserve_schedule:
        max_reserve = max(reserve_schedule) - BATTERY_MIN_SOC_KWH
        capacity_span = max(0.0, BATTERY_CAPACITY_KWH - BATTERY_MIN_SOC_KWH)
        if capacity_span > 0:
            utilization = max_reserve / capacity_span
    return baseline * (1.0 + 0.5 * utilization)


def _build_sell_price_override(
    prices_buy: pd.Series,
    prices_sell: pd.Series,
    consumption_kw: pd.Series,
    production_kw: pd.Series,
    period_hours: float,
    buffer_kwh: float,
    hold_value: float,
) -> List[float]:
    sell_override: List[float] = []
    net_kwh = (production_kw - consumption_kw) * period_hours
    future_surplus = net_kwh.clip(lower=0.0).iloc[::-1].cumsum().iloc[::-1]

    margin = BATTERY_CYCLE_COST_DKK_PER_KWH + 0.05

    for idx in range(len(prices_sell)):
        sell_price = float(prices_sell.iloc[idx])
        buy_future = float(prices_buy.iloc[idx:].quantile(0.7)) if idx < len(prices_buy) else float(prices_buy.iloc[-1])
        opportunity = max(0.0, buy_future - sell_price - margin)

        if future_surplus.iloc[idx] <= buffer_kwh:
            effective_price = max(0.0, sell_price - opportunity)
        else:
            effective_price = max(0.0, sell_price - opportunity * 0.3)

        sell_override.append(effective_price)

    return sell_override


def _suggest_charge_window(
    forecast: pd.DataFrame,
    period_minutes: int,
    ev_buffer_kwh: float,
    ev_required_kwh: float,
    ev_window: Tuple[int, int],
    ev_planning_disabled: bool,
) -> Optional[ChargeRecommendation]:
    if ev_planning_disabled:
        return None

    required = max(ev_buffer_kwh, ev_required_kwh)
    if required <= 0.1:
        return None

    slots_per_hour = 60.0 / max(period_minutes, 1)
    slot_capacity = MAX_EV_CHARGE_KWH / slots_per_hour
    slots_needed = max(1, int(math.ceil(required / slot_capacity)))

    price_series = forecast[["timestamp", "price_buy"]].copy()
    price_series["price_buy"] = price_series["price_buy"].astype(float).fillna(price_series["price_buy"].mean())

    start_idx, end_idx = ev_window
    if 0 <= start_idx < end_idx <= len(price_series):
        candidates = price_series.iloc[start_idx:end_idx].copy()
    else:
        candidates = price_series

    if candidates.empty:
        return None

    cheapest = candidates.nsmallest(slots_needed, "price_buy")
    if cheapest.empty:
        return None

    start_ts = pd.to_datetime(cheapest["timestamp"].min())
    end_ts = pd.to_datetime(cheapest["timestamp"].max()) + timedelta(minutes=period_minutes)
    average_price = float(cheapest["price_buy"].mean())
    percentile_threshold = candidates["price_buy"].quantile(0.3)
    percentile = float((cheapest["price_buy"] <= percentile_threshold).mean())

    energy_slots = len(cheapest) * slot_capacity
    return ChargeRecommendation(
        start=start_ts.to_pydatetime(),
        end=end_ts.to_pydatetime(),
        energy_kwh=min(required, energy_slots),
        average_price_dkk=average_price,
        slot_count=len(cheapest),
        percentile=percentile,
    )


__all__ = [
    "AdaptivePolicy",
    "ChargeRecommendation",
    "compute_adaptive_policy",
]
