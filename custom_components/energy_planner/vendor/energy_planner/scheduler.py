"""Entrypoint for running the quarter-hour optimization cycle."""

from __future__ import annotations

import logging
import json
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)
from typing import Any, List, Optional, Tuple
from dataclasses import replace as dc_replace

import pandas as pd  # type: ignore

from .config import Settings, load_settings
from .constants import (
	BATTERY_CAPACITY_KWH,
	BATTERY_MIN_SOC_KWH,
	EV_BATTERY_CAPACITY_KWH,
	MAX_EV_CHARGE_KWH,
	MAX_EV_CHARGE_QH,
)
from .data_pipeline import DataPipeline
from .db import create_session_factory, session_scope, write_plan_to_mariadb
from .ha_client import HomeAssistantClient
from .models import OptimizerRun
from .optimizer.solver import OptimizationContext, solve_quarter_hour
# from .optimizer.simple_solver import solve_optimization_simple (Moved to local import to prevent load failure if pulp is missing)
from .policy import AdaptivePolicy, compute_adaptive_policy
from .utils.time import ensure_timezone, to_utc_naive


ALLOWED_EV_STATUSES = {
	"awaiting_start",
	"charging",
	"ready_to_charge",
	"ready",
	"completed",
	"connected",
	"plugged_in",
}


def _normalize_ev_status(value: Optional[str]) -> str:
	if not value:
		return "unknown"
	normalized = value.strip().lower().replace(" ", "_")
	return normalized or "unknown"


def _friendly_ev_status(value: Optional[str]) -> str:
	if not value:
		return "unknown"
	return value.replace("_", " ")


def _parse_time_setting(raw: Optional[str], base: datetime, *, roll_forward: bool) -> datetime:
	"""Parse a Home Assistant time selector value safely.

	HA often reports ``unknown``/``unavailable`` during startup; fallback to ``base`` in that case.
	"""

	if raw is None:
		return base

	value = str(raw).strip()
	if not value or value.lower() in {"unknown", "unavailable", "none", "null"}:
		return base

	try:
		parsed = datetime.fromisoformat(value)
		if parsed.tzinfo is None:
			parsed = parsed.replace(tzinfo=base.tzinfo)
		else:
			parsed = parsed.astimezone(base.tzinfo)
		return parsed
	except (TypeError, ValueError):
		parts = value.split(":")
		if len(parts) < 2:
			return base
		try:
			hour, minute = int(parts[0]), int(parts[1])
		except ValueError:
			return base

		candidate = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
		if roll_forward and candidate <= base:
			candidate += timedelta(days=1)
		return candidate


def build_context(
	forecast: pd.DataFrame,
	settings: Settings,
	ha: HomeAssistantClient,
	SessionFactory: Optional[Any] = None,
	now: Optional[datetime] = None,
	ev_done_today_kwh: Optional[float] = None,
) -> Tuple[OptimizationContext, AdaptivePolicy]:
	now = now or datetime.now(timezone.utc)
	tz = settings.timezone
	start_ts = ensure_timezone(pd.to_datetime(forecast.loc[0, "timestamp"]).to_pydatetime(), tz)
	period_minutes = settings.resolution_minutes
	period_hours = period_minutes / 60.0

	# Dynamiske hardware-parametre fra HA sensorer
	dyn_capacity_wh = ha.fetch_numeric_state(settings.battery_capacity_sensor)
	curr_batt_cap_kwh = (dyn_capacity_wh / 1000.0) if dyn_capacity_wh else BATTERY_CAPACITY_KWH
	
	dyn_min_soc_pct = ha.fetch_numeric_state(settings.battery_min_soc_sensor)
	curr_min_soc_pct = dyn_min_soc_pct if dyn_min_soc_pct is not None else 15.0
	curr_min_soc_kwh = (curr_min_soc_pct / 100.0) * curr_batt_cap_kwh

	dyn_max_soc_pct = ha.fetch_numeric_state(settings.battery_max_soc_sensor)
	curr_max_soc_pct = dyn_max_soc_pct if dyn_max_soc_pct is not None else 100.0

	from .constants import MAX_BATTERY_CHARGE_KWH, MAX_BATTERY_DISCHARGE_KWH
	dyn_charge_watts = ha.fetch_numeric_state(settings.battery_charge_power_sensor)
	if dyn_charge_watts is not None:
		# Slider kan begrænse yderligere, men aldrig overskride hardware loftet i constants.py
		curr_max_charge_kwh = min(dyn_charge_watts / 1000.0, MAX_BATTERY_CHARGE_KWH)
	else:
		curr_max_charge_kwh = MAX_BATTERY_CHARGE_KWH

	dyn_discharge_watts = ha.fetch_numeric_state(settings.battery_discharge_power_sensor)
	if dyn_discharge_watts is not None:
		curr_max_discharge_kwh = min(dyn_discharge_watts / 1000.0, MAX_BATTERY_DISCHARGE_KWH)
	else:
		curr_max_discharge_kwh = MAX_BATTERY_DISCHARGE_KWH

	battery_soc_pct = ha.fetch_numeric_state(settings.battery_soc_sensor)
	ev_soc_pct = ha.fetch_numeric_state(settings.ev_soc_sensor)
	
	# SOC VALIDATION with fallback values
	# If sensors are unavailable (e.g., during HA startup or Tesla sleep), use safe defaults
	battery_soc_pct_valid = battery_soc_pct
	if battery_soc_pct is None or battery_soc_pct < 0 or battery_soc_pct > 100:
		logger.warning(
			f"⚠️ Battery SOC sensor invalid (sensor={settings.battery_soc_sensor}, value={battery_soc_pct}). "
			f"Using fallback: 50%"
		)
		battery_soc_pct_valid = 50.0
	
	ev_soc_pct_valid = ev_soc_pct
	if ev_soc_pct is None or ev_soc_pct < 0 or ev_soc_pct > 100:
		logger.warning(
			f"⚠️ EV SOC sensor invalid (sensor={settings.ev_soc_sensor}, value={ev_soc_pct}). "
			f"Using fallback: 50%. Note: EV charging may not be optimal if car is actually connected."
		)
		ev_soc_pct_valid = 50.0
	
	# Convert validated percentages to kWh
	battery_soc_kwh = max(0.0, min(100.0, battery_soc_pct_valid)) / 100 * curr_batt_cap_kwh
	battery_soc_kwh = max(curr_min_soc_kwh, min(curr_batt_cap_kwh, battery_soc_kwh))
	
	ev_soc_kwh = max(0.0, min(100.0, ev_soc_pct_valid)) / 100 * EV_BATTERY_CAPACITY_KWH

	target_soc_pct = ha.fetch_numeric_state(settings.ev_target_soc_sensor) or 90.0
	target_soc_ratio = max(0.0, min(100.0, target_soc_pct)) / 100

	ev_status_raw = ha.fetch_string_state(settings.ev_status_sensor)
	ev_status = _normalize_ev_status(ev_status_raw)
	ev_planning_switch_state = None
	ev_planning_override = False
	ev_planning_disabled = False
	if settings.ev_planning_switch:
		ev_planning_switch_state = ha.fetch_string_state(settings.ev_planning_switch)
		normalized_switch = (ev_planning_switch_state or "").strip().lower()
		if normalized_switch in {"on", "true", "1", "available", "planning"}:
			ev_planning_override = True
		elif normalized_switch in {"off", "false", "0", "disabled", "unavailable"}:
			ev_planning_disabled = True
	# Always plan EV charging even if switch indicates disconnected; act as advisory
	if ev_planning_disabled:
		ev_planning_disabled = False

	ev_start_raw = ha.fetch_string_state(settings.ev_start_sensor)
	ev_end_raw = ha.fetch_string_state(settings.ev_end_sensor)

	# Parse user-provided EV window. We keep track of whether user explicitly set both values
	start_value_present = bool(ev_start_raw and str(ev_start_raw).strip().lower() not in {"unknown", "unavailable", "none", "null"})
	end_value_present = bool(ev_end_raw and str(ev_end_raw).strip().lower() not in {"unknown", "unavailable", "none", "null"})
	user_defined_window = start_value_present and end_value_present
	if not ev_start_raw or not ev_end_raw:
		ev_start_dt = start_ts
		ev_end_dt = start_ts + timedelta(hours=settings.lookahead_hours)
	else:
		ev_start_dt = _parse_time_setting(ev_start_raw, start_ts, roll_forward=False)
		ev_end_dt = _parse_time_setting(ev_end_raw, start_ts, roll_forward=True)
		if ev_start_dt < start_ts:
			ev_start_dt = start_ts
		while ev_end_dt <= start_ts:
			ev_end_dt += timedelta(days=1)
		while ev_end_dt <= ev_start_dt:
			ev_end_dt += timedelta(days=1)

	start_index = max(0, int((ev_start_dt - start_ts).total_seconds() // (period_minutes * 60)))
	end_index = min(len(forecast), int((ev_end_dt - start_ts).total_seconds() // (period_minutes * 60)))

	if end_index <= start_index:
		start_index = 0
		end_index = len(forecast)

	ev_windows: list[Tuple[int, int]] = []
	ev_window_defs: list[dict] = []
	day_keys = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
	day_index_map = {name: idx for idx, name in enumerate(day_keys)}

	def _next_weekday_key(key: str) -> str:
		idx = day_index_map.get(key)
		if idx is None:
			return day_keys[0]
		return day_keys[(idx + 1) % len(day_keys)]
	base_local = ensure_timezone(start_ts, tz)
	try:
		allow_multi = True
		if hasattr(settings, "allow_multi_windows_switch") and settings.allow_multi_windows_switch:
			state = ha.fetch_string_state(settings.allow_multi_windows_switch)
			allow_multi = str(state or "on").lower() not in {"off", "false", "0", "unavailable"}
		if allow_multi:
			horizon_end_local = ensure_timezone(pd.to_datetime(forecast.iloc[-1]["timestamp"]).to_pydatetime(), tz)
			for offset in range(0, 8):
				idx = (base_local.weekday() + offset) % 7
				key = day_keys[idx]
				ent_start = f"input_datetime.energy_planner_ev_start_{key}"
				ent_end = f"input_datetime.energy_planner_ev_end_{key}"
				s_raw = ha.fetch_string_state(ent_start)
				e_raw = ha.fetch_string_state(ent_end)
				if not s_raw or not e_raw:
					continue
				s_val = str(s_raw).strip().lower()
				e_val = str(e_raw).strip().lower()
				if s_val in {"unknown", "unavailable", "none", "null", ""} or e_val in {"unknown", "unavailable", "none", "null", ""}:
					continue
				try:
					sh, sm = [int(x) for x in s_raw.split(":")[:2]]
					eh, em = [int(x) for x in e_raw.split(":")[:2]]
				except Exception:
					continue
				base_day = ensure_timezone(
					(base_local + timedelta(days=offset)).replace(hour=0, minute=0, second=0, microsecond=0),
					tz,
				)
				s_dt_local = base_day.replace(hour=sh, minute=sm)
				e_dt_local = base_day.replace(hour=eh, minute=em)
				if e_dt_local <= s_dt_local:
					e_dt_local = e_dt_local + timedelta(days=1)
				s_dt_utc = s_dt_local.astimezone(timezone.utc).replace(tzinfo=timezone.utc)
				e_dt_utc = e_dt_local.astimezone(timezone.utc).replace(tzinfo=timezone.utc)
				if s_dt_local > horizon_end_local:
					continue
				a = max(0, int((s_dt_utc - start_ts.replace(tzinfo=timezone.utc)).total_seconds() // (period_minutes * 60)))
				b = max(a + 1, int((e_dt_utc - start_ts.replace(tzinfo=timezone.utc)).total_seconds() // (period_minutes * 60)))
				a = min(a, len(forecast))
				b = min(b, len(forecast))
				if b > a:
					ev_windows.append((a, b))
					ev_window_defs.append(
						{
							"start": a,
							"end": b,
							"weekday": key,
							"departure_weekday": _next_weekday_key(key),
							"local_start": s_dt_local,
							"local_end": e_dt_local,
						}
					)
	except Exception:
		ev_windows = []

	fallback_triggered = False
	if not ev_windows:
		fallback_triggered = True
		ev_windows = [(start_index, end_index)]
		default_weekday = day_keys[base_local.weekday()]
		ev_window_defs.append(
			{
				"start": start_index,
				"end": end_index,
				"weekday": default_weekday,
				"departure_weekday": _next_weekday_key(default_weekday),
				"local_start": ensure_timezone(start_ts, tz),
				"local_end": ensure_timezone(
					(start_ts + timedelta(minutes=(end_index - start_index) * period_minutes)), tz
				),
			}
		)
	ev_window_defs.sort(key=lambda item: item["start"])
	ev_window_start_index = max(0, min(a for a, _ in ev_windows))
	ev_window_end_index = min(len(forecast), max(b for _, b in ev_windows))
	start_index = ev_window_start_index
	end_index = ev_window_end_index

	try:
		weekly_ev_map = ha.fetch_weekly_ev_kwh_inputs()
	except Exception:
		weekly_ev_map = {key: 0.0 for key in day_keys}
	try:
		weekly_departure_pct = ha.fetch_weekly_ev_departure_pct_inputs()
	except Exception:
		weekly_departure_pct = {}
	default_departure_pct = None
	if settings.ev_default_departure_sensor:
		try:
			value = ha.fetch_numeric_state(settings.ev_default_departure_sensor)
			if value is not None and value == value:
				default_departure_pct = max(0.0, min(100.0, float(value)))
		except Exception:
			default_departure_pct = None
	if default_departure_pct is None:
		default_departure_pct = 90.0
	window_buffer_fraction = max(0.0, min(1.0, settings.ev_window_local_buffer_pct))

	window_departure_weekday_map: dict[str, str] = {}
	window_expected_consumption_map: dict[str, float] = {}
	window_departure_target_pct: dict[str, Optional[float]] = {}
	window_consumption_sensor_map: dict[str, str] = {}
	window_target_sensor_map: dict[str, str] = {}

	for key in day_keys:
		departure_key = _next_weekday_key(key)
		window_departure_weekday_map[key] = departure_key
		window_expected_consumption_map[key] = float(weekly_ev_map.get(departure_key, 0.0) or 0.0)
		target_val = weekly_departure_pct.get(departure_key)
		if target_val is None:
			target_val = default_departure_pct
		window_departure_target_pct[key] = target_val
		if hasattr(ha, "weekly_ev_entity_id"):
			try:
				window_consumption_sensor_map[key] = ha.weekly_ev_entity_id(departure_key)
			except Exception:
				window_consumption_sensor_map[key] = f"input_number.energy_planner_ev_week_{departure_key}_kwh"
		else:
			window_consumption_sensor_map[key] = f"input_number.energy_planner_ev_week_{departure_key}_kwh"
		window_target_sensor_map[key] = f"input_number.energy_planner_ev_departure_{departure_key}_pct"

	ev_target_pct_series: list[Optional[float]] = [None] * len(forecast)
	ev_window_requirements: list[dict] = []
	cumulative_required = 0.0
	soc_state = ev_soc_kwh
	total_window_capacity_slots = sum(max(0, b - a) for a, b in ev_windows)
	
	# Calculate per-slot limits based on resolution
	slots_per_hour = 60.0 / max(settings.resolution_minutes, 1)
	max_ev_charge_slot = MAX_EV_CHARGE_KWH / slots_per_hour
	
	max_deliverable_total = total_window_capacity_slots * max_ev_charge_slot
	default_target_pct = max(0.0, min(100.0, target_soc_ratio * 100.0))
	price_series = forecast.get("price_buy", pd.Series(dtype=float)).astype(float).fillna(0.0)

	# CRITICAL FIX: Account for EV consumption BEFORE first window starts
	# If current time is after previous window end but before next window start,
	# we must reduce SOC by the expected consumption during that period
	if ev_window_defs:
		first_window = ev_window_defs[0]
		first_window_start_index = first_window["start"]
		
		# If we're BEFORE the first window (i.e., there's a gap), we need to account for
		# consumption that happened AFTER the previous window ended
		if first_window_start_index > 0:
			# Get the departure weekday of the PREVIOUS day's window
			# This is the window that ended BEFORE the current first window
			current_weekday = first_window.get("weekday", "").lower()
			
# FIXED: Removed pre-window consumption adjustment that caused double-counting
		# Consumption is already accounted for in the effective_targets loop below (line 455)
		# Do NOT subtract consumption here - it will be subtracted AGAIN when computing required_kwh
		pass

	# Compute effective target for each window by looking ahead so earlier windows cover future consumption
	effective_targets: list[float] = [0.0] * len(ev_window_defs)
	required_soc_next = 0.0
	for idx in range(len(ev_window_defs) - 1, -1, -1):
		meta = ev_window_defs[idx]
		key = (meta.get("weekday") or day_keys[0]).lower()
		departure_key = (meta.get("departure_weekday") or window_departure_weekday_map.get(key) or _next_weekday_key(key))
		expected_consumption = float(window_expected_consumption_map.get(key, 0.0) or 0.0)
		target_pct = window_departure_target_pct.get(key)
		if target_pct is None or target_pct <= 0:
			target_pct = default_departure_pct or default_target_pct
		target_pct = max(0.0, min(100.0, float(target_pct)))
		target_energy = (target_pct / 100.0) * EV_BATTERY_CAPACITY_KWH
		effective_target = max(target_energy, required_soc_next)
		effective_target = min(EV_BATTERY_CAPACITY_KWH, effective_target)
		effective_targets[idx] = effective_target
		required_soc_next = max(0.0, effective_target - expected_consumption)
		meta["departure_weekday"] = departure_key

	for meta, effective_target in zip(ev_window_defs, effective_targets):
		key = (meta.get("weekday") or day_keys[0]).lower()
		departure_key = (meta.get("departure_weekday") or window_departure_weekday_map.get(key) or _next_weekday_key(key))
		expected_consumption = float(window_expected_consumption_map.get(key, 0.0) or 0.0)
		target_pct = window_departure_target_pct.get(key)
		if target_pct is None or target_pct <= 0:
			target_pct = default_departure_pct or default_target_pct
		target_pct = max(0.0, min(100.0, float(target_pct)))
		target_energy = (target_pct / 100.0) * EV_BATTERY_CAPACITY_KWH
		need_now = max(0.0, effective_target - soc_state)
		remaining_capacity = max(0.0, max_deliverable_total - cumulative_required)
		if need_now > remaining_capacity:
			need_now = remaining_capacity
		slot_count = max(0, int(meta["end"] - meta["start"]))
		window_capacity_kwh = slot_count * max_ev_charge_slot
		if window_capacity_kwh > 0 and need_now > window_capacity_kwh:
			need_now = window_capacity_kwh
		cumulative_required += need_now
		deadline_index = min(len(forecast), meta["end"])
		for i in range(meta["start"], meta["end"]):
			if 0 <= i < len(ev_target_pct_series):
				ev_target_pct_series[i] = target_pct
		buffer_kwh = 0.0
		if need_now > 0 and window_buffer_fraction > 0:
			buffer_kwh = min(need_now, window_capacity_kwh)
			buffer_kwh = min(buffer_kwh, window_buffer_fraction * need_now)
		ev_window_requirements.append(
			{
				"weekday": key,
				"window_weekday": key,
				"departure_weekday": departure_key,
				"start_index": meta["start"],
				"end_index": meta["end"],
				"slot_count": slot_count,
				"slot_capacity_kwh": window_capacity_kwh,
				"target_pct": target_pct,
				"target_kwh": target_energy,
				"goal_kwh": effective_target,
				"expected_consumption_kwh": expected_consumption,
				"needed_kwh": need_now,
				"buffer_kwh": buffer_kwh,
				"planned_kwh": 0.0,
				"deadline_index": deadline_index,
				"consumption_sensor": window_consumption_sensor_map.get(key),
				"target_sensor": window_target_sensor_map.get(key),
				"local_start": meta.get("local_start"),
				"local_end": meta.get("local_end"),
			}
		)
		soc_state = min(EV_BATTERY_CAPACITY_KWH, soc_state + need_now)
		soc_state = max(0.0, soc_state - expected_consumption)

	ev_required_kwh = cumulative_required

	ev_deadlines: list[tuple[int, float]] = []
	if ev_window_requirements:
		window_allocations: list[float] = []
		total_allocated = 0.0
		for req in ev_window_requirements:
			planned = min(float(req.get("needed_kwh", 0.0)), float(req.get("buffer_kwh", 0.0) or 0.0))
			req["planned_kwh"] = planned
			window_allocations.append(planned)
			total_allocated += planned
		remaining_need = max(0.0, ev_required_kwh - total_allocated)

		slot_candidates: list[tuple[float, int]] = []
		for idx, req in enumerate(ev_window_requirements):
			start_idx = int(req.get("start_index", 0))
			end_idx = int(req.get("end_index", 0))
			for slot_idx in range(start_idx, end_idx):
				if 0 <= slot_idx < len(price_series):
					slot_candidates.append((float(price_series.iloc[slot_idx]), idx))
		slot_candidates.sort(key=lambda item: item[0])

		for price, window_idx in slot_candidates:
			if remaining_need <= 1e-6:
				break
			needed_in_window = float(ev_window_requirements[window_idx].get("needed_kwh", 0.0))
			current_plan = window_allocations[window_idx]
			window_capacity = float(ev_window_requirements[window_idx].get("slot_capacity_kwh", 0.0))
			remaining_window = min(needed_in_window, window_capacity) - current_plan
			if remaining_window <= 1e-6:
				continue
			allocate = min(max_ev_charge_slot, remaining_need, remaining_window)
			window_allocations[window_idx] += allocate
			remaining_need -= allocate

		if remaining_need > 1e-6:
			last_idx = len(window_allocations) - 1
			if last_idx >= 0:
				window_allocations[last_idx] += remaining_need
				remaining_need = 0.0

		cumulative_plan = 0.0
		for idx, req in enumerate(ev_window_requirements):
			planned_kwh = min(float(req.get("needed_kwh", 0.0)), window_allocations[idx])
			req["planned_kwh"] = planned_kwh
			cumulative_plan += planned_kwh
			if planned_kwh > 0 or cumulative_plan > 0:
				ev_deadlines.append((int(req.get("deadline_index", len(forecast))), cumulative_plan))

	consumption_series = forecast.get("consumption_estimate_kw", pd.Series(dtype=float)).astype(float).fillna(0.0)
	pv_series = forecast.get("pv_forecast_kw", pd.Series(dtype=float)).astype(float).fillna(0.0)
	baseline_price = float(price_series.quantile(0.65)) if not price_series.empty else 0.0

	# Price ranking per local day: mark cheap24 and expensive24
	cheap_flags: list[bool] = [False] * len(forecast)
	expensive_flags: list[bool] = [False] * len(forecast)
	try:
		local_ts = ensure_timezone(pd.to_datetime(forecast["timestamp"], utc=True).dt.tz_convert(settings.timezone), settings.timezone)
		df_rank = pd.DataFrame({
			"idx": range(len(forecast)),
			"timestamp_local": local_ts,
			"price_buy": price_series,
		})
		df_rank["day"] = df_rank["timestamp_local"].dt.normalize()
		for _, grp in df_rank.groupby("day"):
			# sort ascending for cheapest
			order = grp.sort_values("price_buy", ascending=True)
			cheap_idx = order.head(24)["idx"].astype(int).tolist()
			for i in cheap_idx:
				if 0 <= i < len(cheap_flags):
					cheap_flags[i] = True
			# sort descending for most expensive
			order2 = grp.sort_values("price_buy", ascending=False)
			exp_idx = order2.head(24)["idx"].astype(int).tolist()
			for i in exp_idx:
				if 0 <= i < len(expensive_flags):
					expensive_flags[i] = True
	except Exception:
		pass

	reserve_signal: List[float] = [0.0] * len(forecast)
	remaining_deficit_kwh = 0.0
	for idx in range(len(forecast) - 1, -1, -1):
		load_kw = float(consumption_series.iloc[idx])
		prod_kw = float(pv_series.iloc[idx])
		deficit_kwh = max(0.0, load_kw - prod_kw) * period_hours
		surplus_kwh = max(0.0, prod_kw - load_kw) * period_hours
		remaining_deficit_kwh = max(0.0, remaining_deficit_kwh + deficit_kwh - surplus_kwh)
		if BATTERY_CAPACITY_KWH > 0:
			reserve_signal[idx] = min(1.0, remaining_deficit_kwh / BATTERY_CAPACITY_KWH)
		else:
			reserve_signal[idx] = 0.0

	# Reserve bias sættes til 0 for at muliggøre ren økonomisk optimering uden tvungen buffer.
	# Brugeren har anmodet om at fjerne "junk" logik der tvinger opladning i dyre timer.
	reserve_bias = 0.0 
	capacity_span = max(0.0, curr_batt_cap_kwh - curr_min_soc_kwh)
	reserve_schedule = []
	for signal in reserve_signal:
		reserve_level = reserve_bias + (1.0 - reserve_bias) * signal
		reserve_target = curr_min_soc_kwh + reserve_level * capacity_span
		reserve_schedule.append(min(curr_batt_cap_kwh, max(curr_min_soc_kwh, reserve_target)))

	# --- FJERN JUNK START ---
	# Vi fjerner tvungen buffer-boosting her i scheduler.py, 
	# da brugeren ønsker ren økonomisk optimering styret fra policy.py
	# ev_future_buffer_kwh = min(capacity_span, max(0.0, settings.ev_future_daily_buffer_kwh))
	# if ev_future_buffer_kwh > 0:
	# 	...
	# --- FJERN JUNK SLUT ---

	# Optional risk-based reserve boost using price volatility
	try:
		price_series = forecast.get("price_buy", pd.Series(dtype=float)).astype(float).fillna(0.0)
		if not price_series.empty:
			baseline_price = float(price_series.quantile(0.65)) if not price_series.empty else 0.0
			vol = float(price_series.std())
			if baseline_price > 0:
				boost_ratio = min(0.2, max(0.0, vol / (baseline_price * 3.0)))
				if boost_ratio > 0 and capacity_span > 0:
					boost = boost_ratio * capacity_span
					reserve_schedule = [min(curr_batt_cap_kwh, target + boost) for target in reserve_schedule]
	except Exception:
		# Non-fatal if price series missing or any math fails
		pass

	deficit_ratio = 0.0
	if capacity_span > 0:
		total_deficit = float(((consumption_series - pv_series).clip(lower=0)).sum() * period_hours)
		total_surplus = float(((pv_series - consumption_series).clip(lower=0)).sum() * period_hours)
		if total_deficit > total_surplus:
			deficit_ratio = min(1.5, (total_deficit - total_surplus) / (capacity_span or 1e-6))

	reserve_penalty = 0.0
	if reserve_bias > 0 and baseline_price > 0:
		reserve_penalty = baseline_price * (1.5 + 0.5 * deficit_ratio)

	policy: AdaptivePolicy
	if SessionFactory is not None and now is not None:
		try:
			policy = compute_adaptive_policy(
				forecast=forecast,
				settings=settings,
				ha_client=ha,
				SessionFactory=SessionFactory,
				now=now,
				battery_soc_kwh=battery_soc_kwh,
				ev_required_kwh=ev_required_kwh,
				ev_window=(start_index, end_index),
				ev_status=ev_status,
				ev_planning_disabled=ev_planning_disabled,
			)
			if fallback_triggered and allow_multi:
				policy.notes.append("⚠️ EV schedule fallback: Sensors unavailable, assuming always home.")
		except Exception as exc:  # pragma: no cover - defensive path
			policy = AdaptivePolicy(
				reserve_schedule=list(reserve_schedule),
				sell_price_override=[],
				reserve_penalty_per_kwh=reserve_penalty,
				expected_ev_daily_kwh=0.0,
				expected_house_daily_kwh=0.0,
				future_extra_load_kwh=0.0,
				ev_buffer_target_kwh=0.0,
				charge_recommendation=None,
				history_sample_days=0,
				notes=[f"Adaptive policy fallback: {exc}"],
				battery_hold_value_dkk=0.0,
				price_buy_high_threshold=0.0,
				learned_house_daily_sample_count=0,
				learned_ev_daily_sample_count=0,
				learned_battery_min_soc_pct=None,
			)
	else:
		policy = AdaptivePolicy(
			reserve_schedule=list(reserve_schedule),
			sell_price_override=[],
			reserve_penalty_per_kwh=reserve_penalty,
			expected_ev_daily_kwh=0.0,
			expected_house_daily_kwh=0.0,
			future_extra_load_kwh=0.0,
			ev_buffer_target_kwh=0.0,
			charge_recommendation=None,
			history_sample_days=0,
			notes=["Adaptive policy unavailable; history context missing."],
			battery_hold_value_dkk=0.0,
			price_buy_high_threshold=0.0,
			learned_house_daily_sample_count=0,
			learned_ev_daily_sample_count=0,
			learned_battery_min_soc_pct=None,
		)

	ev_capacity_slots = sum(max(0, b - a) for a, b in ev_windows)
	ev_capacity_total = ev_capacity_slots * MAX_EV_CHARGE_QH
	buffer_target = float(getattr(policy, "ev_buffer_target_kwh", 0.0) or 0.0)
	if ev_done_today_kwh is not None:
		buffer_target = max(0.0, buffer_target - float(ev_done_today_kwh))
	ev_required_candidate = max(ev_required_kwh, buffer_target)
	if ev_capacity_total > 0:
		ev_required_kwh = min(ev_required_candidate, ev_capacity_total)
	else:
		ev_required_kwh = 0.0

	# Compute precharge need targeting future expensive slots
	period_hours = period_minutes / 60.0
	try:
		# Expected house net need in expensive slots
		cons = forecast.get("consumption_estimate_kw", pd.Series(dtype=float)).astype(float).fillna(0.0)
		pv = forecast.get("pv_forecast_kw", pd.Series(dtype=float)).astype(float).fillna(0.0)
		net_need_exp = 0.0
		for i in range(len(forecast)):
			if i < len(expensive_flags) and expensive_flags[i]:
				net_need_exp += max(0.0, float(cons.iloc[i] - pv.iloc[i])) * period_hours
		# EV buffer no longer drives precharge decisions; focus solely on house peak shaving
		ev_min_buf = 0.0
		# PV expected before expensive slots (approximate using cheap24 total PV)
		pv_before = 0.0
		for i in range(len(forecast)):
			if i < len(cheap_flags) and cheap_flags[i]:
				pv_before += max(0.0, float(pv.iloc[i])) * period_hours
		precharge_need_kwh = max(0.0, net_need_exp + ev_min_buf - pv_before)
	except Exception:
		precharge_need_kwh = 0.0

	# EV allowed mask: allow all slots inside declared windows
	ev_allowed_mask: list[bool] = [False] * len(forecast)
	if ev_windows:
		for a, b in ev_windows:
			for idx in range(int(a), int(b)):
				if 0 <= idx < len(ev_allowed_mask):
					ev_allowed_mask[idx] = True
	else:
		ev_allowed_mask = [True] * len(forecast)

	resolved_reserve_schedule = policy.reserve_schedule or reserve_schedule
	resolved_reserve_penalty = policy.reserve_penalty_per_kwh if policy.reserve_penalty_per_kwh is not None else reserve_penalty
	resolved_reserve_schedule = list(resolved_reserve_schedule)
	sell_override = list(policy.sell_price_override)

	ev_charge_deadline_index: Optional[int] = None
	ev_min_charge_by_deadline_kwh: Optional[float] = None
	if (
		policy.charge_recommendation
		and not ev_planning_disabled
	):
		rec = policy.charge_recommendation
		try:
			rec_end_local = ensure_timezone(rec.end, tz)
		except Exception:
			rec_end_local = None
		if rec_end_local is not None and not forecast.empty:
			rec_end_utc = rec_end_local.astimezone(timezone.utc)
			start_reference = pd.to_datetime(forecast.loc[0, "timestamp"], utc=True)
			delta_seconds = (rec_end_utc - start_reference).total_seconds()
			slot = int(delta_seconds // (period_minutes * 60))
			horizon_last = max(0, len(forecast) - 1)
			if slot >= start_index:
				deadline_index = min(horizon_last, slot)
				deadline_index = min(deadline_index, max(end_index - 1, start_index))
				# Ensure the minimum-by-deadline reflects the strongest of:
				# - user/target driven requirement (ev_required_kwh),
				# - recommended window energy,
				# - recommended EV limit kWh (if available, e.g., 30 kWh)
				recommended_kwh_candidates = [max(0.0, rec.energy_kwh)]
				if ev_required_kwh is not None:
					recommended_kwh_candidates.append(max(0.0, ev_required_kwh))
				try:
					limit_abs_kwh = float(getattr(policy, "recommended_ev_limit_kwh", 0.0) or 0.0)
				except Exception:
					limit_abs_kwh = 0.0
				# Convert absolute limit (target SOC energy) to incremental kWh from current SOC
				if limit_abs_kwh > 0:
					limit_delta_kwh = max(0.0, limit_abs_kwh - ev_soc_kwh)
					if limit_delta_kwh > 0:
						recommended_kwh_candidates.append(limit_delta_kwh)

				recommended_kwh = max(recommended_kwh_candidates) if recommended_kwh_candidates else 0.0
				capacity_room = max(0.0, EV_BATTERY_CAPACITY_KWH - ev_soc_kwh)
				recommended_kwh = min(recommended_kwh, capacity_room)
				if recommended_kwh > 0 and deadline_index >= start_index:
					ev_charge_deadline_index = deadline_index
					ev_min_charge_by_deadline_kwh = recommended_kwh

	# If we have a recommended EV limit percentage higher than the current target cap,
	# relax the cap so the plan can reach the advised level (bounded at 100%).
	adj_target_soc_ratio = target_soc_ratio
	try:
		rec_limit_pct = getattr(policy, "recommended_ev_limit_pct", None)
		if rec_limit_pct is not None:
			adj_target_soc_ratio = max(adj_target_soc_ratio, float(rec_limit_pct) / 100.0)
	except Exception:
		pass

	# Calculate remaining minutes in current slot for partial slot scaling
	start_ts_local = ensure_timezone(start_ts, tz)
	minutes_into_slot = start_ts_local.minute % period_minutes
	remaining_minutes = float(period_minutes - minutes_into_slot)

	# Check if EV is physically connected
	ev_connected = False  # Default til False - kun tilsluttet hvis sensorer bekræfter det
	try:
		# Primary: binary_sensor.tessa_charger
		charger_state = ha.fetch_string_state("binary_sensor.tessa_charger")
		if charger_state and charger_state.lower() in {"on", "true", "connected", "available"}:
			ev_connected = True
		
		# Secondary fallback: sensor.easee_status (kun hvis charger ikke allerede siger connected)
		if not ev_connected:
			easee_state = ha.fetch_string_state("sensor.easee_status")
			if easee_state and easee_state.lower() not in {"disconnected", "unavailable", "unknown", "error", "offline"}:
				ev_connected = True
	except Exception:
		# Ved fejl: default til False (konservativ tilgang)
		ev_connected = False

	context = OptimizationContext(
		start_timestamp=start_ts,
		battery_soc_kwh=battery_soc_kwh,
		ev_soc_kwh=ev_soc_kwh,
		ev_target_soc_pct=adj_target_soc_ratio,
		ev_status=ev_status,
		ev_window_start_index=start_index,
		ev_window_end_index=max(start_index + 1, end_index),
		ev_windows=tuple(ev_windows),
		ev_required_kwh=ev_required_kwh,
		ev_planning_override=ev_planning_override,
		ev_planning_disabled=ev_planning_disabled,
		ev_planning_switch_state=ev_planning_switch_state,
		resolution_minutes=period_minutes,
		battery_reserve_schedule=resolved_reserve_schedule,
		reserve_penalty_per_kwh=resolved_reserve_penalty,
		sell_price_override=sell_override,
		grid_sell_price_multiplier=settings.grid_sell_price_multiplier,
		grid_sell_penalty_dkk_per_kwh=settings.grid_sell_penalty_dkk_per_kwh,
		battery_hold_value_dkk=policy.battery_hold_value_dkk,
		ev_charge_deadline_index=ev_charge_deadline_index,
		ev_min_charge_by_deadline_kwh=ev_min_charge_by_deadline_kwh,
		dynamic_margin_dkk=tuple(policy.dynamic_margin_dkk),
		dynamic_low_reserve_pct=tuple(policy.dynamic_low_reserve_pct_series),
		battery_hold_value_series=tuple(policy.battery_hold_value_series),
		price_future_min_series=tuple(policy.price_future_min_series),
		price_future_max_series=tuple(policy.price_future_max_series),
		price_future_p75_series=tuple(policy.price_future_p75_series),
		price_future_std_series=tuple(policy.price_future_std_series),
		wait_flags=tuple(policy.wait_flags),
		wait_reasons=tuple(policy.wait_reasons),
		slot_diagnostics=tuple(policy.slot_diagnostics),
		cheap_flags=tuple(cheap_flags),
		expensive_flags=tuple(expensive_flags),
		precharge_need_kwh=float(precharge_need_kwh),
		lambda_peak_dkk=3.0,
		ev_allowed_mask=tuple(ev_allowed_mask),
		ev_cumulative_deadlines=tuple(ev_deadlines),
		ev_window_requirements=tuple(ev_window_requirements),
		ev_target_pct_series=tuple(ev_target_pct_series),
		remaining_minutes_in_current_slot=remaining_minutes,
		ev_connected=ev_connected,
		# Dynamic hardware limits
		battery_capacity_kwh=curr_batt_cap_kwh,
		battery_min_soc_kwh=curr_min_soc_kwh,
		battery_maximum_pct=curr_max_soc_pct,
		max_charge_kwh=curr_max_charge_kwh,
		max_discharge_kwh=curr_max_discharge_kwh,
		use_linear_solver=settings.use_linear_solver,
	)

	return context, policy


def summarize_plan(
	plan: pd.DataFrame,
	forecast: pd.DataFrame,
	context: OptimizationContext,
	settings: Settings,
	policy: AdaptivePolicy,
) -> dict:
	period_hours = settings.resolution_minutes / 60.0
	weekday_lookup = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

	def _energy_total(series: pd.Series) -> float:
		return float(series.fillna(0).sum() * period_hours)

	energy_totals = {
		"grid_import_kwh": _energy_total(plan["g_buy"]),
		"grid_export_kwh": _energy_total(plan["g_sell"]),
		"battery_charge_kwh": _energy_total(plan["battery_in"]),
		"battery_discharge_kwh": _energy_total(plan["battery_out"]),
	}

	if plan.empty:
		energy_totals["ev_charge_kwh"] = 0.0
	else:
		start_ev_soc = float(context.ev_soc_kwh)
		end_ev_soc = float(plan["ev_soc"].iloc[-1])
		energy_totals["ev_charge_kwh"] = max(0.0, end_ev_soc - start_ev_soc)

	energy_totals["pv_generation_kwh"] = float(forecast["pv_forecast_kw"].fillna(0).sum() * period_hours)
	energy_totals["house_consumption_kwh"] = float(forecast["consumption_estimate_kw"].fillna(0).sum() * period_hours)
	horizon_days = len(plan) * period_hours / 24 if period_hours > 0 else 0.0
	if horizon_days > 0:
		energy_totals["house_consumption_per_day_kwh"] = energy_totals["house_consumption_kwh"] / horizon_days
	energy_totals["battery_charge_from_pv_kwh"] = _energy_total(plan.get("prod_to_batt", pd.Series(dtype=float)))
	energy_totals["battery_charge_from_grid_kwh"] = _energy_total(plan.get("grid_to_batt", pd.Series(dtype=float)))
	energy_totals["battery_discharge_to_house_kwh"] = _energy_total(plan.get("batt_to_house", pd.Series(dtype=float)))
	energy_totals["battery_discharge_to_ev_kwh"] = _energy_total(plan.get("batt_to_ev", pd.Series(dtype=float)))
	energy_totals["battery_discharge_to_sell_kwh"] = _energy_total(plan.get("batt_to_sell", pd.Series(dtype=float)))

	target_kwh = context.ev_target_soc_pct * EV_BATTERY_CAPACITY_KWH
	ev_ready_timestamp: Optional[datetime] = None
	for idx, ev_soc in enumerate(plan["ev_soc"].fillna(0)):
		if ev_soc >= target_kwh - 1e-3:
			ev_ready_timestamp = plan.loc[idx, "timestamp"]
			break

	ev_window_start = None
	ev_window_end = None
	if 0 <= context.ev_window_start_index < len(plan):
		ev_window_start = plan.loc[context.ev_window_start_index, "timestamp"]
	if 0 < context.ev_window_end_index <= len(plan):
		ev_window_end = plan.loc[context.ev_window_end_index - 1, "timestamp"]

	ev_window_targets_summary: List[dict] = []
	if context.ev_window_requirements:
		for req in context.ev_window_requirements:
			start_idx = int(req.get("start_index", 0))
			end_idx = int(req.get("end_index", 0))
			start_ts = plan.loc[start_idx, "timestamp"] if (0 <= start_idx < len(plan)) else None
			end_lookup = min(max(end_idx - 1, 0), len(plan) - 1)
			end_ts = plan.loc[end_lookup, "timestamp"] if not plan.empty else None
			local_start = req.get("local_start")
			local_end = req.get("local_end")
			window_weekday = req.get("window_weekday") or req.get("weekday")
			departure_weekday = req.get("departure_weekday")
			if departure_weekday is None and isinstance(window_weekday, str):
				weekday_lower = window_weekday.lower()
				if weekday_lower in weekday_lookup:
					idx = weekday_lookup.index(weekday_lower)
					departure_weekday = weekday_lookup[(idx + 1) % len(weekday_lookup)]
			start_soc_kwh = None
			end_soc_kwh = None
			if not plan.empty and "ev_soc" in plan:
				if 0 <= start_idx < len(plan):
					start_soc_kwh = float(plan["ev_soc"].iloc[start_idx])
				if 0 <= end_lookup < len(plan):
					end_soc_kwh = float(plan["ev_soc"].iloc[end_lookup])
			def _clamp_ev_soc(value: Optional[float]) -> Optional[float]:
				if value is None or EV_BATTERY_CAPACITY_KWH <= 0:
					return value
				return max(0.0, min(EV_BATTERY_CAPACITY_KWH, float(value)))

			start_soc_kwh = _clamp_ev_soc(start_soc_kwh)
			end_soc_kwh = _clamp_ev_soc(end_soc_kwh)
			start_soc_pct = (
				(start_soc_kwh / EV_BATTERY_CAPACITY_KWH * 100.0) if (start_soc_kwh is not None and EV_BATTERY_CAPACITY_KWH > 0) else None
			)
			end_soc_pct = (
				(end_soc_kwh / EV_BATTERY_CAPACITY_KWH * 100.0) if (end_soc_kwh is not None and EV_BATTERY_CAPACITY_KWH > 0) else None
			)
			goal_kwh = req.get("goal_kwh")
			goal_soc_pct = (
				(goal_kwh / EV_BATTERY_CAPACITY_KWH * 100.0) if (goal_kwh is not None and EV_BATTERY_CAPACITY_KWH > 0) else None
			)
			if goal_soc_pct is not None:
				goal_soc_pct = max(0.0, min(100.0, goal_soc_pct))
			actual_delta_kwh = None
			if start_soc_kwh is not None and end_soc_kwh is not None:
				actual_delta_kwh = end_soc_kwh - start_soc_kwh
			actual_delta_pct = (
				(actual_delta_kwh / EV_BATTERY_CAPACITY_KWH * 100.0) if (actual_delta_kwh is not None and EV_BATTERY_CAPACITY_KWH > 0) else None
			)
			window_ev_charge_kwh = None
			if "ev_charge" in plan.columns and not plan.empty and end_idx > start_idx >= 0:
				window_ev_charge_kwh = float(plan["ev_charge"].iloc[start_idx:end_idx].sum())
			slot_count = int(req.get("slot_count") or max(0, end_idx - start_idx))
			max_window_capacity = slot_count * MAX_EV_CHARGE_QH
			buffer_kwh = float(req.get("buffer_kwh") or 0.0)
			planned_kwh = float(req.get("planned_kwh") or 0.0)
			slot_capacity_kwh = float(req.get("slot_capacity_kwh", max_window_capacity))
			ev_window_targets_summary.append(
				{
					"weekday": req.get("weekday"),
					"window_weekday": window_weekday,
					"departure_weekday": departure_weekday,
					"start_index": start_idx,
					"end_index": end_idx,
					"start_timestamp": start_ts,
					"end_timestamp": end_ts,
					"local_start": local_start,
					"local_end": local_end,
					"target_pct": req.get("target_pct"),
					"target_kwh": req.get("target_kwh"),
					"goal_kwh": goal_kwh,
					"goal_soc_pct": goal_soc_pct,
					"expected_consumption_kwh": req.get("expected_consumption_kwh"),
					"needed_kwh": req.get("needed_kwh"),
					"buffer_kwh": buffer_kwh,
					"planned_kwh": planned_kwh,
					"start_soc_kwh": start_soc_kwh,
					"end_soc_kwh": end_soc_kwh,
					"start_soc_pct": start_soc_pct,
					"end_soc_pct": end_soc_pct,
					"actual_delta_kwh": actual_delta_kwh,
					"actual_delta_pct": actual_delta_pct,
					"ev_charge_within_window_kwh": window_ev_charge_kwh,
					"consumption_sensor": req.get("consumption_sensor"),
					"target_sensor": req.get("target_sensor"),
					"slot_count": slot_count,
					"slot_capacity_kwh": slot_capacity_kwh,
					"max_deliverable_kwh": max_window_capacity,
					"window_label": f"{window_weekday}->{departure_weekday}" if window_weekday and departure_weekday else None,
				}
			)

	ev_deadline_summary: List[dict] = []
	if context.ev_cumulative_deadlines:
		for deadline_index, required in context.ev_cumulative_deadlines:
			idx = int(max(0, min(len(plan), deadline_index)))
			ts = None
			if not plan.empty and idx > 0:
				ts = plan.loc[min(idx - 1, len(plan) - 1), "timestamp"]
			ev_deadline_summary.append(
				{
					"deadline_index": idx,
					"deadline_timestamp": ts,
					"cumulative_kwh": float(required),
				}
			)

	start_soc_kwh = context.battery_soc_kwh
	end_soc_kwh = float(plan["battery_soc"].iloc[-1]) if not plan.empty else context.battery_soc_kwh
	min_soc_kwh = float(plan["battery_soc"].min()) if not plan.empty else context.battery_soc_kwh
	max_soc_kwh = float(plan["battery_soc"].max()) if not plan.empty else context.battery_soc_kwh
	battery_summary = {
		"start_soc_kwh": start_soc_kwh,
		"end_soc_kwh": end_soc_kwh,
		"net_delta_kwh": end_soc_kwh - start_soc_kwh,
		"min_soc_kwh": min_soc_kwh,
		"max_soc_kwh": max_soc_kwh,
		"start_soc_pct": (start_soc_kwh / BATTERY_CAPACITY_KWH * 100) if BATTERY_CAPACITY_KWH else None,
		"end_soc_pct": (end_soc_kwh / BATTERY_CAPACITY_KWH * 100) if BATTERY_CAPACITY_KWH else None,
		"min_soc_pct": (min_soc_kwh / BATTERY_CAPACITY_KWH * 100) if BATTERY_CAPACITY_KWH else None,
		"max_soc_pct": (max_soc_kwh / BATTERY_CAPACITY_KWH * 100) if BATTERY_CAPACITY_KWH else None,
		"total_charge_kwh": energy_totals["battery_charge_kwh"],
		"total_discharge_kwh": energy_totals["battery_discharge_kwh"],
		"charge_from_pv_kwh": energy_totals["battery_charge_from_pv_kwh"],
		"charge_from_grid_kwh": energy_totals["battery_charge_from_grid_kwh"],
		"discharge_to_house_kwh": energy_totals["battery_discharge_to_house_kwh"],
		"discharge_to_ev_kwh": energy_totals["battery_discharge_to_ev_kwh"],
		"discharge_to_sell_kwh": energy_totals["battery_discharge_to_sell_kwh"],
	}

	policy_summary = {
		"reserve_schedule_applied": list(context.battery_reserve_schedule),
		"sell_price_override_applied": list(getattr(context, "sell_price_override", [])),
		"reserve_penalty_applied": context.reserve_penalty_per_kwh,
		"expected_ev_daily_kwh": policy.expected_ev_daily_kwh,
		"expected_house_daily_kwh": policy.expected_house_daily_kwh,
		"future_extra_load_kwh": policy.future_extra_load_kwh,
		"ev_buffer_target_kwh": policy.ev_buffer_target_kwh,
		"history_sample_days": policy.history_sample_days,
		"notes": policy.notes,
		"planned_ev_kwh": getattr(policy, "planned_ev_kwh", 0.0),
		"planned_ev_source": getattr(policy, "planned_ev_source", None),
		"planned_ev_arrival_soc_pct": getattr(policy, "planned_ev_arrival_soc_pct", None),
		"planned_ev_departure_soc_pct": getattr(policy, "planned_ev_departure_soc_pct", None),
		"house_expected_override_kwh": getattr(policy, "house_expected_override_kwh", None),
		"battery_hold_value_dkk": getattr(policy, "battery_hold_value_dkk", 0.0),
		"planned_ev_schedule_day": getattr(policy, "planned_ev_schedule_day", None),
		"recommended_ev_limit_pct": getattr(policy, "recommended_ev_limit_pct", None),
		"recommended_ev_limit_kwh": getattr(policy, "recommended_ev_limit_kwh", None),
		"recommended_ev_limit_reason": getattr(policy, "recommended_ev_limit_reason", None),
	}
	if policy.reserve_schedule:
		policy_summary["reserve_schedule_inferred"] = list(policy.reserve_schedule)
	if policy.sell_price_override:
		policy_summary["sell_price_override_inferred"] = list(policy.sell_price_override)
	policy_summary["reserve_penalty_inferred"] = policy.reserve_penalty_per_kwh
	if policy.charge_recommendation:
		recommendation = policy.charge_recommendation
		policy_summary["charge_recommendation"] = {
			"start": recommendation.start,
			"end": recommendation.end,
			"energy_kwh": recommendation.energy_kwh,
			"average_price_dkk": recommendation.average_price_dkk,
			"slot_count": recommendation.slot_count,
			"percentile": recommendation.percentile,
		}

	# Add unmet load diagnostics
	unmet_diagnostics = {}
	if not plan.empty and "house_load_unmet" in plan.columns:
		unmet_total = float(plan["house_load_unmet"].sum())
		unmet_diagnostics["total_unmet_kwh"] = unmet_total
		unmet_diagnostics["unmet_slots"] = int((plan["house_load_unmet"] > 0.01).sum())
		if "house_unmet_reason" in plan.columns:
			reason_counts = plan[plan["house_load_unmet"] > 0.01]["house_unmet_reason"].value_counts().to_dict()
			unmet_diagnostics["reason_breakdown"] = reason_counts

	required_ev_total = context.ev_required_kwh
	if required_ev_total is None:
		if context.ev_cumulative_deadlines:
			required_ev_total = float(context.ev_cumulative_deadlines[-1][1])
		else:
			required_ev_total = 0.0
	else:
		required_ev_total = float(required_ev_total)

	return {
		"energy_totals": energy_totals,
		"battery": battery_summary,
		"ev": {
			"status": _friendly_ev_status(context.ev_status),
			"required_kwh": required_ev_total,
			"planned_kwh": energy_totals["ev_charge_kwh"],
			"shortfall_kwh": max(0.0, required_ev_total - energy_totals["ev_charge_kwh"]),
			"target_soc_pct": context.ev_target_soc_pct * 100,
			"target_soc_kwh": target_kwh,
			"window_start": ev_window_start,
			"window_end": ev_window_end,
			"ready_timestamp": ev_ready_timestamp,
			"window_targets": ev_window_targets_summary,
			"window_summary": ev_window_targets_summary,
			"deadlines": ev_deadline_summary,
			"planning_mode": (
				"disabled"
				if context.ev_planning_disabled
				else (
					"forced" if context.ev_planning_override else (
						"advice" if (_normalize_ev_status(context.ev_status) not in ALLOWED_EV_STATUSES and (context.ev_required_kwh or 0.0) > 0)
						else "auto"
					)
				)
			),
			"switch_state": getattr(context, "ev_planning_switch_state", None),
		},
		"policy": policy_summary,
		"unmet_load": unmet_diagnostics,
	}


def _apply_ev_plan_to_forecast(forecast_df: pd.DataFrame, context: OptimizationContext, settings: 'PlannerSettings') -> None:
	"""
	Helper to apply EV constraints to the forecast dataframe before solving.
	Sets 'ev_available' (bool) and 'ev_driving_consumption_kwh' (float).
	
	CRITICAL FIX 2026-01-25:
	Consumption is now distributed based on WEEKDAY using ev_window_requirements.
	Each day's consumption (from input_number.energy_planner_ev_week_X_kwh) is distributed
	across the "away" slots for that specific weekday.
	"""
	import pandas as pd
	from datetime import timezone
	
	# Initialize columns if missing (safety check)
	if 'ev_available' not in forecast_df.columns:
		forecast_df['ev_available'] = False
	if 'ev_driving_consumption_kwh' not in forecast_df.columns:
		forecast_df['ev_driving_consumption_kwh'] = 0.0

	# 1. Mark availability based on ev_windows
	# ev_windows is list of (start_idx, end_idx) where car IS available (plugged in)
	forecast_df['ev_available'] = False
	
	if context.ev_windows:
		for start_i, end_i in context.ev_windows:
			s = max(0, start_i)
			e = min(len(forecast_df), end_i)
			if s < e:
				forecast_df.loc[s:e-1, 'ev_available'] = True

	# 2. Distribute consumption using ev_window_requirements
	# Each window has 'expected_consumption_kwh' and 'departure_weekday'
	# The consumption happens AFTER departure, so we distribute it across that day's "away" slots
	
	ev_window_reqs = getattr(context, 'ev_window_requirements', [])
	if not ev_window_reqs:
		logger.info("No ev_window_requirements found - skipping weekday-based consumption distribution")
		# Fallback to old uniform distribution if no window info
		val_kwh = getattr(context, 'ev_required_kwh', 0.0)
		if val_kwh and val_kwh > 0:
			mask_away = ~forecast_df['ev_available']
			away_count = mask_away.sum()
			if away_count > 0:
				drain_per_slot = val_kwh / away_count
				forecast_df.loc[mask_away, 'ev_driving_consumption_kwh'] = drain_per_slot
				logger.info(f"EV Consumption: Uniform distribution of {val_kwh:.1f} kWh across {away_count} away slots")
		return
	
	logger.info(f"Applying weekday-based EV consumption distribution using {len(ev_window_reqs)} window requirements")
	
	# Build weekday -> consumption map from window requirements
	# Key: departure_weekday (e.g., "mon" means car departs Monday morning and drives Monday)
	# Value: expected_consumption_kwh for that day
	day_consumption_map = {}
	for req in ev_window_reqs:
		departure_weekday = req.get("departure_weekday", "").lower()
		consumption_kwh = float(req.get("expected_consumption_kwh", 0.0))
		if departure_weekday and consumption_kwh > 0:
			day_consumption_map[departure_weekday] = consumption_kwh
			logger.info(f"  {departure_weekday}: {consumption_kwh:.1f} kWh expected consumption")
	
	if not day_consumption_map:
		logger.info("No consumption data in ev_window_requirements - skipping distribution")
		return  # No consumption to distribute
	
	day_keys = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
	tz = settings.timezone
	
	# Process each unique day in forecast
	forecast_df['ev_driving_consumption_kwh'] = 0.0
	
	for day_key in day_keys:
		if day_key not in day_consumption_map:
			continue
		
		consumption_kwh = day_consumption_map[day_key]
		
		# Find all timestamps for this weekday
		day_idx = day_keys.index(day_key)
		
		for idx in forecast_df.index:
			try:
				ts = pd.to_datetime(forecast_df.loc[idx, 'timestamp'])
				if ts.tzinfo is None:
					ts = ts.tz_localize(timezone.utc)
				ts_local = ts.astimezone(tz)
				slot_weekday_idx = ts_local.weekday()  # 0=Monday
				
				if slot_weekday_idx != day_idx:
					continue  # Not this weekday
				
				if forecast_df.loc[idx, 'ev_available']:
					continue  # Skip charging window slots
				
				# This slot is an "away" slot on the correct weekday
				# Mark it for consumption distribution
				forecast_df.loc[idx, f'_temp_day_{day_key}'] = 1
			except Exception as e:
				logger.warning(f"Failed to check weekday for slot {idx}: {e}")
				continue
	
	# Now distribute consumption across marked slots for each day
	for day_key, consumption_kwh in day_consumption_map.items():
		temp_col = f'_temp_day_{day_key}'
		if temp_col not in forecast_df.columns:
			continue
		
		day_mask = forecast_df[temp_col] == 1
		away_count_this_day = day_mask.sum()
		
		if away_count_this_day > 0:
			drain_per_slot = consumption_kwh / away_count_this_day
			forecast_df.loc[day_mask, 'ev_driving_consumption_kwh'] = drain_per_slot
			logger.info(
				f"EV Consumption: {day_key} = {consumption_kwh:.1f} kWh distributed across {away_count_this_day} away slots "
				f"({drain_per_slot:.3f} kWh/slot)"
			)
		else:
			logger.warning(f"EV Consumption: {day_key} has consumption ({consumption_kwh:.1f} kWh) but NO away slots found!")
		
		# Clean up temp column
		forecast_df.drop(columns=[temp_col], inplace=True)


def run_once(now: Optional[datetime] = None) -> dict:
	settings = load_settings()
	ha = HomeAssistantClient(settings.ha_base_url, settings.ha_api_key)
	SessionFactory = create_session_factory(settings.mariadb_dsn)

	pipeline = DataPipeline(settings, ha, SessionFactory)
	now = now or datetime.now(timezone.utc)
	forecast_frame = pipeline.build_forecast_dataframe(now)
	pipeline.persist_forecast(forecast_frame)
	
	# Record actuals from recent history to build dataset for analysis
	try:
		pipeline.fetch_and_record_actuals(hours_back=2)
	except Exception as e:
		print(f"Warning: Failed to record actuals: {e}")

	# Derive EV energy already charged today (local day) from reconciliation, if available
	ev_done_today_kwh: Optional[float] = None
	try:
		last_daily = getattr(pipeline, "last_daily_reconciliation", None) or []
		if last_daily:
			today_local = ensure_timezone(now, settings.timezone).date()
			for r in last_daily:
				try:
					d = pd.to_datetime(r.get("date")).date()
					if d == today_local:
						ev_done_today_kwh = float(r.get("ev_kwh", 0.0))
						break
				except Exception:
					continue
	except Exception:
		pass

	context, policy = build_context(
		forecast_frame,
		settings,
		ha,
		SessionFactory=SessionFactory,
		now=now,
		ev_done_today_kwh=ev_done_today_kwh,
	)
	
	# Task 2: Consumption estimate sanity check in EV window
	try:
		if context.ev_required_kwh and context.ev_required_kwh > 0:
			# Get EV window indices
			ev_start_idx = context.ev_window_start_index
			ev_end_idx = context.ev_window_end_index
			
			if ev_start_idx < ev_end_idx and ev_start_idx >= 0 and ev_end_idx <= len(forecast_frame):
				# Get forecast consumption in EV window
				forecast_consumption = forecast_frame.loc[ev_start_idx:ev_end_idx-1, "consumption_estimate_kw"]
				
				# Get historical consumption from pipeline if available
				hist_consumption = None
				if hasattr(pipeline, 'last_hist_quarter') and pipeline.last_hist_quarter is not None:
					hist_df = pipeline.last_hist_quarter
					if 'net_house_kw' in hist_df.columns and len(hist_df) > 0:
						# Align historical data with forecast timestamps
						forecast_timestamps = forecast_frame.loc[ev_start_idx:ev_end_idx-1, "timestamp"]
						hist_aligned = hist_df.set_index('timestamp').reindex(forecast_timestamps, method='nearest', tolerance=pd.Timedelta('15min'))
						hist_consumption = hist_aligned['net_house_kw'].fillna(0.0)
				
				if hist_consumption is not None and len(hist_consumption) > 0:
					# Compare forecast vs historical in EV window
					forecast_mean = forecast_consumption.mean()
					hist_mean = hist_consumption.mean()
					diff_pct = abs(forecast_mean - hist_mean) / max(hist_mean, 0.1) * 100
					
					if diff_pct > 20.0:  # More than 20% difference
						print(f"⚠️ Consumption estimate sanity check: EV window forecast ({forecast_mean:.2f} kW) differs significantly from historical ({hist_mean:.2f} kW, {diff_pct:.1f}% difference)")
						print("   This may indicate consumption calibration issues or unusual conditions.")
	except Exception as e:
		# Non-fatal: log but continue
		print(f"Consumption sanity check failed: {e}")
	
	# CRITICAL FIX: Apply EV consumption BEFORE solver selection
	# This ensures consumption columns exist regardless of solver type
	_apply_ev_plan_to_forecast(forecast_frame, context, settings)
	consumption_slots = (forecast_frame.get('ev_driving_consumption_kwh', pd.Series(dtype=float)) > 0).sum()
	logger.info("EV consumption setup complete: %d slots with consumption", consumption_slots)
	
	if settings.use_linear_solver:
		try:
			logger.info("Starting simple linear solver...")
			from .optimizer.simple_solver import solve_optimization_simple
			result = solve_optimization_simple(forecast_frame, context)
			logger.info("Simple solver completed with status: %s", result.status)
		except Exception as e:
			logger.error("Simple solver failed: %s", e, exc_info=True)
			result = solve_quarter_hour(forecast_frame, context)
			if hasattr(result, "notes"):
				result.notes.append(f"Simple solver FEJL: {str(e)[:100]}. Fallback til standard solver.")
	else:
		logger.info("Using standard heuristic solver.")
		result = solve_quarter_hour(forecast_frame, context)
		if hasattr(result, "notes"):
			result.notes.append("Standard Heuristisk Solver aktiv (Linear solver deaktiveret).")
	# Force garbage collection to free LP model memory
	import gc
	gc.collect()
	
	# CRITICAL FIX: Preserve EV consumption columns from forecast_frame to result.plan
	# Solvers don't automatically copy these columns, so we add them manually
	print(f"[SCHEDULER DEBUG] Before column copy: forecast_frame has {list(forecast_frame.columns)}")
	print(f"[SCHEDULER DEBUG] Before column copy: result.plan has {list(result.plan.columns)}")
	logger.info("Before column copy: result.plan has %d columns", len(result.plan.columns))
	for col in ['ev_driving_consumption_kwh', 'ev_available']:
		if col in forecast_frame.columns and col not in result.plan.columns:
			result.plan[col] = forecast_frame[col].values
			print(f"[SCHEDULER DEBUG] OK Copied '{col}' from forecast to plan ({(result.plan[col] > 0).sum()} non-zero)")
			logger.info("OK Copied column '%s' from forecast to plan (%d non-zero values)",
						col, (result.plan[col] > 0).sum())
		elif col in result.plan.columns:
			print(f"[SCHEDULER DEBUG] OK Column '{col}' already exists in result.plan")
			logger.info("OK Column '%s' already exists in result.plan", col)
		else:
			print(f"[SCHEDULER DEBUG] ERROR Column '{col}' NOT found in forecast_frame!")
			logger.warning("ERROR Column '%s' not found in forecast_frame!", col)
	print(f"[SCHEDULER DEBUG] After column copy: result.plan has {list(result.plan.columns)}")
	logger.info("After column copy: result.plan has %d columns: %s", 
				len(result.plan.columns), list(result.plan.columns))
	
	# Fallback: if model infeasible or not optimal, relax peak/EV gating and re-solve
	if str(result.status).lower() not in {"optimal", "feasible"}:
		period_count = len(forecast_frame)
		relaxed_mask = tuple([True] * period_count)
		false_flags = tuple([False] * period_count)
		try:
			relaxed_ctx = dc_replace(
				context,
				cheap_flags=false_flags,
				expensive_flags=false_flags,
				precharge_need_kwh=0.0,
				lambda_peak_dkk=0.0,
				ev_allowed_mask=relaxed_mask,
			)
		except Exception:
			# Construct a minimal relaxed context if replace fails
			relaxed_ctx = OptimizationContext(
				start_timestamp=context.start_timestamp,
				battery_soc_kwh=context.battery_soc_kwh,
				ev_soc_kwh=context.ev_soc_kwh,
				ev_target_soc_pct=context.ev_target_soc_pct,
				ev_status=context.ev_status,
				ev_window_start_index=context.ev_window_start_index,
				ev_window_end_index=context.ev_window_end_index,
				ev_windows=tuple(context.ev_windows or ()),
				ev_required_kwh=context.ev_required_kwh,
				ev_planning_override=context.ev_planning_override,
				ev_planning_disabled=context.ev_planning_disabled,
				ev_planning_switch_state=context.ev_planning_switch_state,
				resolution_minutes=context.resolution_minutes,
				battery_reserve_schedule=tuple(context.battery_reserve_schedule or ()),
				reserve_penalty_per_kwh=context.reserve_penalty_per_kwh,
				sell_price_override=tuple(getattr(context, "sell_price_override", ()) or ()),
				grid_sell_price_multiplier=context.grid_sell_price_multiplier,
				grid_sell_penalty_dkk_per_kwh=context.grid_sell_penalty_dkk_per_kwh,
				battery_hold_value_dkk=context.battery_hold_value_dkk,
				ev_charge_deadline_index=context.ev_charge_deadline_index,
				ev_min_charge_by_deadline_kwh=context.ev_min_charge_by_deadline_kwh,
				dynamic_margin_dkk=tuple(getattr(context, "dynamic_margin_dkk", ()) or ()),
				dynamic_low_reserve_pct=tuple(getattr(context, "dynamic_low_reserve_pct", ()) or ()),
				battery_hold_value_series=tuple(getattr(context, "battery_hold_value_series", ()) or ()),
				price_future_min_series=tuple(getattr(context, "price_future_min_series", ()) or ()),
				price_future_max_series=tuple(getattr(context, "price_future_max_series", ()) or ()),
				price_future_p75_series=tuple(getattr(context, "price_future_p75_series", ()) or ()),
				price_future_std_series=tuple(getattr(context, "price_future_std_series", ()) or ()),
				wait_flags=tuple(getattr(context, "wait_flags", ()) or ()),
				wait_reasons=tuple(getattr(context, "wait_reasons", ()) or ()),
				slot_diagnostics=tuple(getattr(context, "slot_diagnostics", ()) or ()),
				cheap_flags=false_flags,
				expensive_flags=false_flags,
				precharge_need_kwh=0.0,
				lambda_peak_dkk=0.0,
				ev_allowed_mask=relaxed_mask,
			)
		result_relaxed = solve_quarter_hour(forecast_frame, relaxed_ctx)
		gc.collect()  # Clean up second solve as well
		# Overwrite context/result if improved
		if str(result_relaxed.status).lower() in {"optimal", "feasible"}:
			context = relaxed_ctx
			result = result_relaxed
			# Also append a policy note indicating fallback was used
			try:
				policy.notes.append("Fallback: relaxed peak/EV gating used due to infeasible model.")
			except Exception:
				pass
	summary = summarize_plan(result.plan, forecast_frame, context, settings, policy)

	from .reporting import build_plan_report  # Local import to avoid circular dependency
	from .learning import LearningManager

	plan_report = build_plan_report(
		forecast=forecast_frame,
		result=result,
		context=context,
		settings=settings,
		pipeline=pipeline,
		summary=summary,
	)

	with session_scope(SessionFactory) as session:
		run = OptimizerRun(
			run_started_at=to_utc_naive(context.start_timestamp),
			run_completed_at=datetime.utcnow(),
			horizon_hours=settings.lookahead_hours,
			resolution_minutes=settings.resolution_minutes,
			objective_value=result.objective_value,
			status=result.status,
			details=json.dumps(summary, default=str),
		)
		session.add(run)
		session.flush()

		manager = LearningManager()
		manager.persist_plan_artifacts(session, run, plan_report)

	# Ensure the plan slots are written to the DB for the sensor to read
	# Use plan_report.plan (enriched) instead of result.plan (raw) to get activity, timestamp_local etc.
	write_plan_to_mariadb(plan_report.plan, settings.mariadb_dsn, settings.timezone)

	return {
		"objective_value": result.objective_value,
		"status": result.status,
		"plan_rows": len(result.plan),
		"summary": summary,
		"report": plan_report,
	}


__all__ = ["run_once", "build_context", "summarize_plan"]
