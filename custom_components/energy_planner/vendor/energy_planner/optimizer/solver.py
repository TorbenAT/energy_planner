"""Linear optimization solver for quarter-hour planning."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence

import pandas as pd  # type: ignore
import pulp as lp  # type: ignore

from ..constants import (
    BATTERY_CAPACITY_KWH,
    BATTERY_CYCLE_COST_DKK_PER_KWH,
    BATTERY_EFFICIENCY_IN,
    BATTERY_EFFICIENCY_OUT,
    BATTERY_MIN_SOC_KWH,
    EV_BATTERY_CAPACITY_KWH,
    EV_CHARGE_BONUS_DKK_PER_KWH,
    EV_DEFAULT_MIN_CHARGE_KWH,
    MAX_BATTERY_CHARGE_KWH,
    MAX_BATTERY_DISCHARGE_KWH,
    MAX_EV_CHARGE_KWH,
    MAX_GRID_BUY_KWH,
    MAX_GRID_SELL_KWH,
    MAX_INVERTER_OUTPUT_KWH,
    MAX_BATTERY_CHARGE_QH,
    MAX_BATTERY_DISCHARGE_QH,
    MAX_EV_CHARGE_QH,
    MAX_GRID_BUY_QH,
    MAX_GRID_SELL_QH,
    MAX_INVERTER_OUTPUT_QH,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OptimizationContext:
    start_timestamp: datetime
    battery_soc_kwh: float
    ev_soc_kwh: float
    ev_target_soc_pct: float
    ev_status: str
    ev_window_start_index: int
    ev_window_end_index: int
    ev_windows: Sequence[tuple[int, int]] = ()
    ev_required_kwh: Optional[float] = None
    ev_planning_override: bool = False
    ev_planning_disabled: bool = False
    ev_planning_switch_state: Optional[str] = None
    resolution_minutes: int = 15
    battery_reserve_schedule: Sequence[float] = ()
    reserve_penalty_per_kwh: float = 0.0
    sell_price_override: Sequence[float] = ()
    grid_sell_price_multiplier: float = 1.0
    grid_sell_penalty_dkk_per_kwh: float = 0.0
    battery_hold_value_dkk: float = 0.0
    ev_charge_deadline_index: Optional[int] = None
    ev_min_charge_by_deadline_kwh: Optional[float] = None
    dynamic_margin_dkk: Sequence[float] = ()
    dynamic_low_reserve_pct: Sequence[float] = ()
    battery_hold_value_series: Sequence[float] = ()
    price_future_min_series: Sequence[Optional[float]] = ()
    price_future_max_series: Sequence[Optional[float]] = ()
    price_future_p75_series: Sequence[Optional[float]] = ()
    price_future_std_series: Sequence[float] = ()
    wait_flags: Sequence[bool] = ()
    wait_reasons: Sequence[Optional[str]] = ()
    slot_diagnostics: Sequence[dict] = ()
    # Load-shaving / peak gating additions
    cheap_flags: Sequence[bool] = ()
    expensive_flags: Sequence[bool] = ()
    precharge_need_kwh: float = 0.0
    lambda_peak_dkk: float = 3.0
    ev_allowed_mask: Sequence[bool] = ()
    ev_cumulative_deadlines: Sequence[tuple[int, float]] = ()
    ev_window_requirements: Sequence[dict] = ()
    ev_target_pct_series: Sequence[Optional[float]] = ()
    # Current slot remaining time (for partial slot scaling)
    remaining_minutes_in_current_slot: float = 15.0
    ev_connected: bool = True  # Whether EV is physically connected
    # Dynamic hardware limits
    battery_capacity_kwh: float = BATTERY_CAPACITY_KWH
    battery_min_soc_kwh: float = BATTERY_MIN_SOC_KWH
    max_charge_kwh: float = MAX_BATTERY_CHARGE_KWH
    max_discharge_kwh: float = MAX_BATTERY_DISCHARGE_KWH
    max_ev_charge_kwh: float = MAX_EV_CHARGE_KWH
    ev_battery_capacity_kwh: float = EV_BATTERY_CAPACITY_KWH


@dataclass(slots=True)
class OptimizationResult:
    plan: pd.DataFrame
    objective_value: float
    status: str


def _diagnose_unmet_slot(t: int, plan_row: pd.Series, forecast_row: pd.Series, ctx: OptimizationContext) -> str:
    """Diagnose why a specific time slot has unmet house load.
    
    Returns a string describing the most likely reason for unmet demand.
    """
    reasons = []
    
    # Calculate per-slot limits based on resolution
    slots_per_hour = 60.0 / max(ctx.resolution_minutes, 1)
    max_grid_buy_slot = MAX_GRID_BUY_KWH / slots_per_hour
    max_inverter_output_slot = MAX_INVERTER_OUTPUT_KWH / slots_per_hour
    
    # Get values (with defaults)
    unmet = float(plan_row.get("house_load_unmet", 0.0))
    if unmet < 0.01:
        return ""
    
    load = float(forecast_row.get("consumption_estimate_kw", 0.0))
    pv_available = float(forecast_row.get("pv_forecast_kw", 0.0))
    
    # Available supply sources
    pv_to_house = float(plan_row.get("prod_to_house", 0.0))
    batt_to_house = float(plan_row.get("batt_to_house", 0.0))
    grid_to_house = float(plan_row.get("grid_to_house", 0.0))
    
    total_supply = pv_to_house + batt_to_house + grid_to_house
    
    # Check battery constraints
    battery_soc = float(plan_row.get("battery_soc", 0.0))
    if battery_soc <= BATTERY_MIN_SOC_KWH + 0.1:
        reasons.append("battery_min_soc")
    
    # Check grid buy limits
    g_buy = float(plan_row.get("g_buy", 0.0))
    if g_buy >= max_grid_buy_slot - 0.01:
        reasons.append("grid_buy_limit")
    
    # Check inverter output limits
    inverter_output = (
        float(plan_row.get("prod_to_house", 0.0)) +
        float(plan_row.get("prod_to_ev", 0.0)) +
        float(plan_row.get("prod_to_batt", 0.0)) +
        float(plan_row.get("batt_to_house", 0.0)) +
        float(plan_row.get("batt_to_ev", 0.0)) +
        float(plan_row.get("batt_to_sell", 0.0)) +
        float(plan_row.get("grid_to_house", 0.0)) +
        float(plan_row.get("grid_to_ev", 0.0))
    )
    if inverter_output >= max_inverter_output_slot - 0.01:
        reasons.append("inverter_limit")
    
    # Check PV availability
    pv_used = (
        float(plan_row.get("prod_to_house", 0.0)) +
        float(plan_row.get("prod_to_batt", 0.0)) +
        float(plan_row.get("prod_to_ev", 0.0))
    )
    if pv_used >= pv_available - 0.01:
        reasons.append("pv_insufficient")
    
    # Check reserve requirements
    reserve_target = float(plan_row.get("battery_reserve_target", BATTERY_MIN_SOC_KWH))
    if battery_soc <= reserve_target + 0.1:
        reasons.append("reserve_constraint")
    
    # If no specific reasons found, it might be a constraint combination
    if not reasons:
        reasons.append("constraint_combination")
    
    return "+".join(reasons)


EV_CHARGE_ALLOWED_STATUSES = {
    "awaiting_start",
    "charging",
    "ready_to_charge",
    "ready",
    "completed",
    "connected",
    "plugged_in",
}


def solve_quarter_hour(forecast: pd.DataFrame, ctx: OptimizationContext) -> OptimizationResult:
    periods = len(forecast)
    if periods == 0:
        empty = pd.DataFrame(columns=["g_buy", "g_sell", "battery_in", "battery_out", "ev_charge"])
        return OptimizationResult(empty, 0.0, "empty")

    # Calculate per-slot limits based on resolution
    slots_per_hour = 60.0 / max(ctx.resolution_minutes, 1)
    max_grid_buy_slot = MAX_GRID_BUY_KWH / slots_per_hour
    max_grid_sell_slot = MAX_GRID_SELL_KWH / slots_per_hour
    max_battery_charge_slot = ctx.max_charge_kwh / slots_per_hour
    max_battery_discharge_slot = ctx.max_discharge_kwh / slots_per_hour
    max_ev_charge_slot = ctx.max_ev_charge_kwh / slots_per_hour
    max_inverter_output_slot = MAX_INVERTER_OUTPUT_KWH / slots_per_hour

    model = lp.LpProblem("QuarterHourOptimization", lp.LpMaximize)

    # Parameters for arbitrage gating and hysteresis
    ETA_RT = BATTERY_EFFICIENCY_IN * BATTERY_EFFICIENCY_OUT  # Use physical efficiency instead of hardcoded 0.88
    CYCLE_COST = BATTERY_CYCLE_COST_DKK_PER_KWH              # Use central cycle cost instead of hardcoded 0.15
    PRICE_DEADBAND = 0.05  # DKK deadband required to flip charge/discharge mode
    MIN_RUNTIME_STEPS = 3  # minimum on/off runtime for charge and discharge modes (in slots)

    # CRITICAL FIX: Scale max charge rates for partially elapsed first slot
    remaining_fraction = ctx.remaining_minutes_in_current_slot / ctx.resolution_minutes
    max_battery_charge_slot_first = max_battery_charge_slot * remaining_fraction
    max_battery_discharge_slot_first = max_battery_discharge_slot * remaining_fraction
    max_ev_charge_slot_first = max_ev_charge_slot * remaining_fraction

    # CRITICAL: No upBound on g_buy/g_sell variables - bounds enforced via big-M constraints
    # with y[t] binary to avoid solver conflicts with redundant bounds
    g_buy = lp.LpVariable.dicts("g_buy", range(periods), lowBound=0)
    g_sell = lp.LpVariable.dicts("g_sell", range(periods), lowBound=0)
    
    # Battery charge/discharge - slot 0 has reduced capacity based on remaining time
    b_in = {}
    b_out = {}
    for t in range(periods):
        if t == 0:
            b_in[t] = lp.LpVariable(f"b_in_{t}", lowBound=0, upBound=max_battery_charge_slot_first)
            b_out[t] = lp.LpVariable(f"b_out_{t}", lowBound=0, upBound=max_battery_discharge_slot_first)
        else:
            b_in[t] = lp.LpVariable(f"b_in_{t}", lowBound=0, upBound=max_battery_charge_slot)
            b_out[t] = lp.LpVariable(f"b_out_{t}", lowBound=0, upBound=max_battery_discharge_slot)
    
    # EV charge - slot 0 has reduced capacity based on remaining time
    e = {}
    for t in range(periods):
        if t == 0:
            e[t] = lp.LpVariable(f"ev_{t}", lowBound=0, upBound=max_ev_charge_slot_first)
        else:
            e[t] = lp.LpVariable(f"ev_{t}", lowBound=0, upBound=max_ev_charge_slot)
    
    prod_to_house = lp.LpVariable.dicts("prod_house", range(periods), lowBound=0)
    prod_to_batt = lp.LpVariable.dicts("prod_batt", range(periods), lowBound=0)
    prod_to_ev = lp.LpVariable.dicts("prod_ev", range(periods), lowBound=0)
    prod_curtailed = lp.LpVariable.dicts("prod_curtailed", range(periods), lowBound=0)
    batt_to_house = lp.LpVariable.dicts("batt_house", range(periods), lowBound=0)
    batt_to_ev = lp.LpVariable.dicts("batt_ev", range(periods), lowBound=0)
    batt_to_sell = lp.LpVariable.dicts("batt_sell", range(periods), lowBound=0)
    grid_to_house = lp.LpVariable.dicts("grid_house", range(periods), lowBound=0)
    grid_to_ev = lp.LpVariable.dicts("grid_ev", range(periods), lowBound=0)
    grid_to_batt = lp.LpVariable.dicts("grid_batt", range(periods), lowBound=0)

    # Slack variable for unmet house load (emergency measure to prevent infeasibility)
    house_load_unmet = lp.LpVariable.dicts("house_unmet", range(periods), lowBound=0)

    battery_soc = lp.LpVariable.dicts("SoC", range(periods), lowBound=ctx.battery_min_soc_kwh, upBound=ctx.battery_capacity_kwh)
    ev_soc = lp.LpVariable.dicts("EV_SoC", range(periods), lowBound=0, upBound=ctx.ev_battery_capacity_kwh)

    y = lp.LpVariable.dicts("y", range(periods), cat="Binary")
    # Battery mode binaries: charge (c) and discharge (d)
    c_mode = lp.LpVariable.dicts("c_mode", range(periods), cat="Binary")
    d_mode = lp.LpVariable.dicts("d_mode", range(periods), cat="Binary")
    # Startup/shutdown indicators for min up/down time
    c_start = lp.LpVariable.dicts("c_start", range(periods), lowBound=0, upBound=1, cat="Binary")
    c_stop = lp.LpVariable.dicts("c_stop", range(periods), lowBound=0, upBound=1, cat="Binary")
    d_start = lp.LpVariable.dicts("d_start", range(periods), lowBound=0, upBound=1, cat="Binary")
    d_stop = lp.LpVariable.dicts("d_stop", range(periods), lowBound=0, upBound=1, cat="Binary")
    reserve_shortfall = lp.LpVariable.dicts("reserve_shortfall", range(periods), lowBound=0)

    if ctx.ev_required_kwh is None:
        ctx.ev_required_kwh = EV_DEFAULT_MIN_CHARGE_KWH

    # CRITICAL FIX: Clamp initial SoC to variable bounds to prevent infeasibility
    battery_soc_prev = max(ctx.battery_min_soc_kwh, min(ctx.battery_capacity_kwh, ctx.battery_soc_kwh))
    if abs(battery_soc_prev - ctx.battery_soc_kwh) > 0.01:
        logger.warning(
            "⚠️ Battery SoC clamped: %.2f → %.2f kWh (bounds: [%.1f, %.1f])",
            ctx.battery_soc_kwh, battery_soc_prev, ctx.battery_min_soc_kwh, ctx.battery_capacity_kwh
        )
    
    ev_soc_prev = max(0.0, min(ctx.ev_battery_capacity_kwh, ctx.ev_soc_kwh))
    if abs(ev_soc_prev - ctx.ev_soc_kwh) > 0.01:
        logger.warning(
            "⚠️ EV SoC clamped: %.2f → %.2f kWh (bounds: [0, %.1f])",
            ctx.ev_soc_kwh, ev_soc_prev, ctx.ev_battery_capacity_kwh
        )
    
    # Store initial SoC values for constraint validation
    ev_soc_initial = ev_soc_prev
    battery_soc_initial = battery_soc_prev

    objective_terms = []
    ev_bonus_allowed = ctx.ev_status in EV_CHARGE_ALLOWED_STATUSES or ctx.ev_planning_override
    reserve_schedule = list(ctx.battery_reserve_schedule) if ctx.battery_reserve_schedule else [ctx.battery_min_soc_kwh] * periods
    reserve_penalty = max(0.0, ctx.reserve_penalty_per_kwh)
    hold_value_threshold = max(0.0, getattr(ctx, "battery_hold_value_dkk", 0.0) or 0.0)
    
    # REMOVED RAMP: The ramp caused artificial grid-charging in expensive slots.
    # The soft reserve constraint (slack variable) now handles undershoot gracefully
    # without forcing a linear uphill path that ignores prices.

    effective_sell_prices: list[float] = []
    hold_value_penalty_per_kwh: list[float] = []

    # Precompute effective sell prices (with overrides/penalties) and price arrays for diagnostics
    price_buy_series = forecast["price_buy"].astype(float).fillna(0.0).tolist()
    price_sell_series = forecast["price_sell"].astype(float).fillna(0.0).tolist()
    sell_multiplier = max(0.0, ctx.grid_sell_price_multiplier)
    sell_penalty = max(0.0, ctx.grid_sell_penalty_dkk_per_kwh)
    sell_override_seq = list(ctx.sell_price_override) if ctx.sell_price_override else [None] * periods
    eff_sell_series: list[float] = []
    cheap_flags = list(getattr(ctx, "cheap_flags", [False] * periods))
    expensive_flags = list(getattr(ctx, "expensive_flags", [False] * periods))
    lambda_peak = float(getattr(ctx, "lambda_peak_dkk", 3.0) or 3.0)
    precharge_cap = max(0.0, float(getattr(ctx, "precharge_need_kwh", 0.0) or 0.0))
    ev_allowed_mask = list(getattr(ctx, "ev_allowed_mask", []))
    if not ev_allowed_mask:
        ev_allowed_mask = [True] * periods
    
    # Peak-buy gating binary (only meaningful on expensive slots)
    must_buy_house = lp.LpVariable.dicts("must_buy_house", range(periods), lowBound=0, upBound=1, cat="Binary")

    for t in range(periods):
        base = price_sell_series[t] * sell_multiplier
        override = sell_override_seq[t] if t < len(sell_override_seq) else None
        if override is not None:
            base = float(override)
        eff_sell_series.append(max(0.0, base - sell_penalty))

    # For each slot, compute the max future effective sell (t+1..end) for arbitrage gating
    future_max_eff_sell: list[float] = []
    current_max = 0.0
    # Walk from end to start accumulating maxima
    for t in range(periods - 1, -1, -1):
        if t + 1 < periods:
            current_max = max(current_max, eff_sell_series[t + 1])
        else:
            current_max = 0.0
        future_max_eff_sell.append(current_max)
    future_max_eff_sell.reverse()

    # Diagnostics holders
    arbitrage_gate_list: list[bool] = [False] * periods
    arbitrage_reason_list: list[str] = [""] * periods
    load_shave_gate_list: list[bool] = [False] * periods
    load_shave_reason_list: list[str] = [""] * periods

    # Precompute future max of buy prices (for load-shave gate)
    future_max_buy: list[float] = []
    mx = 0.0
    for t in range(periods - 1, -1, -1):
        if t + 1 < periods:
            mx = max(mx, float(price_buy_series[t + 1]))
        else:
            mx = 0.0
        future_max_buy.append(mx)
    future_max_buy.reverse()

    for t in range(periods):
        price_buy = float(price_buy_series[t])
        effective_sell_price = float(eff_sell_series[t])
        effective_sell_prices.append(effective_sell_price)
        hold_penalty = 0.0
        if hold_value_threshold > 0:
            hold_penalty = max(0.0, hold_value_threshold - effective_sell_price)
        hold_value_penalty_per_kwh.append(hold_penalty)
        # Scale kW to kWh per slot
        slot_hours = ctx.resolution_minutes / 60.0
        prod = float(forecast.loc[t, "pv_forecast_kw"] or 0.0) * slot_hours
        load = float(forecast.loc[t, "consumption_estimate_kw"] or 0.0) * slot_hours

        model += g_buy[t] == grid_to_house[t] + grid_to_ev[t] + grid_to_batt[t]
        model += g_sell[t] == batt_to_sell[t]

        # Battery flow split
        model += b_in[t] == prod_to_batt[t] + grid_to_batt[t]
        model += b_out[t] == batt_to_house[t] + batt_to_ev[t] + batt_to_sell[t]

        model += prod_to_house[t] + prod_to_batt[t] + prod_to_ev[t] + prod_curtailed[t] == prod
        # House load constraint with slack for unmet demand (penalized heavily in objective)
        model += prod_to_house[t] + batt_to_house[t] + grid_to_house[t] + house_load_unmet[t] == load
        model += prod_to_ev[t] + grid_to_ev[t] + batt_to_ev[t] == e[t]

        # Inverter / main-fuse output limit:
        # Fortolk MAX_INVERTER_OUTPUT_KWH som max samlet belastning på AC-siden
        # (hus + EV uanset om det kommer fra PV, batteri eller grid)
        inverter_output = (
            prod_to_house[t]
            + prod_to_ev[t]
            + prod_to_batt[t]
            + batt_to_house[t]
            + batt_to_ev[t]
            + batt_to_sell[t]
            + grid_to_house[t]
            + grid_to_ev[t]
        )
        model += (
            inverter_output <= max_inverter_output_slot,
            f"inverter_output_limit_{t}",
        )

        # Buy/sell exclusivity: y=1 means sell mode, y=0 means buy mode
        # CRITICAL FIX: Only grid_to_batt is blocked by sell mode (y=1).
        # grid_to_house must ALWAYS be allowed to meet house consumption!
        model += grid_to_batt[t] <= (1 - y[t]) * max_grid_buy_slot  # Can't buy to battery when selling
        model += g_sell[t] <= y[t] * max_grid_sell_slot              # Can only sell when y=1

        # Cap total grid buy per slot to enforce physical limit
        model += g_buy[t] <= max_grid_buy_slot

        # Add upper bounds on grid purchases to enforce max_grid_buy_slot per component
        model += grid_to_house[t] <= max_grid_buy_slot
        model += grid_to_ev[t] <= max_grid_buy_slot
        model += grid_to_batt[t] <= max_grid_buy_slot

        # Expensive-slot gating: soft preference managed via objective penalties (reserve, peak),
        # not hard constraint (would cause infeasibility when SoC < reserve and house needs grid)

        # No-overlap and mode binding for battery charge/discharge
        model += c_mode[t] + d_mode[t] <= 1
        # Bind the aggregate battery flows to modes (big-M)
        model += b_in[t] <= max_battery_charge_slot * c_mode[t]
        model += b_out[t] <= max_battery_discharge_slot * d_mode[t]
        # Bind sub-flows as well (strengthens LP relaxation and avoids leakage)
        model += grid_to_batt[t] <= max_battery_charge_slot * c_mode[t]
        model += prod_to_batt[t] <= max_battery_charge_slot * c_mode[t]
        model += batt_to_house[t] <= max_battery_discharge_slot * d_mode[t]
        model += batt_to_ev[t] <= max_battery_discharge_slot * d_mode[t]
        model += batt_to_sell[t] <= max_battery_discharge_slot * d_mode[t]

        # Enforce policy wait flag (unless reserve/EV override applies)
        if getattr(ctx, "wait_flags", None) is not None and len(ctx.wait_flags) > t:
            if bool(ctx.wait_flags[t]) and price_buy >= 0:
                reserve_override = False
                if reserve_schedule:
                    try:
                        current_reserve_target = reserve_schedule[t]
                        reserve_override = current_reserve_target > (float(getattr(ctx, "battery_soc_kwh", 0.0) or 0.0) + 0.5)
                    except Exception:
                        reserve_override = False
                ev_override = False
                try:
                    ev_required_now = float(getattr(ctx, "ev_required_kwh", 0.0) or 0.0)
                    if ev_required_now > 0:
                        ev_override = ctx.ev_window_start_index <= t < ctx.ev_window_end_index
                except Exception:
                    ev_override = False
                cheap_override = False
                if t < len(cheap_flags):
                    cheap_override = bool(cheap_flags[t])
                price_override = False
                if hold_value_threshold > 0:
                    price_override = price_buy <= hold_value_threshold
                if not (reserve_override or cheap_override or price_override):
                    model += grid_to_batt[t] == 0

        # Arbitrage gate (unless negative price): require profitable future opportunity to allow grid->battery
        arb_ok = False
        arb_reason = ""
        if price_buy < 0:
            arb_ok = True
            arb_reason = "negative_price_override"
        else:
            future_eff_sell = float(future_max_eff_sell[t]) if t < len(future_max_eff_sell) else 0.0
            # Correct arbitrage check: energy charged now returns at future sell price
            # scaled DOWN by round-trip efficiency. Profit condition:
            #   future_eff_sell * ETA_RT - price_buy - CYCLE_COST >= 0
            lhs = (future_eff_sell * ETA_RT) - price_buy - CYCLE_COST
            if lhs >= -1e-9:
                arb_ok = True
                arb_reason = "arbitrage_ok"
            else:
                arb_ok = False
                arb_reason = "arbitrage_fail"
        arbitrage_gate_list[t] = bool(arb_ok)
        arbitrage_reason_list[t] = arb_reason
        # Load-shaving gate: allow grid->battery if future buy price is markedly higher than now
        # Conservative penalty: cycle cost + round-trip energy loss valued at current buy price.
        allow_load_shave = False
        ls_reason = ""
        if price_buy >= 0:
            fut_max_buy = float(future_max_buy[t]) if t < len(future_max_buy) else 0.0
            roundtrip_penalty = CYCLE_COST + (1.0 - ETA_RT) * price_buy
            if (fut_max_buy - price_buy) >= roundtrip_penalty - 1e-9:
                allow_load_shave = True
                ls_reason = "load_shave_ok"
            else:
                ls_reason = "load_shave_fail"
        load_shave_gate_list[t] = bool(allow_load_shave)
        load_shave_reason_list[t] = ls_reason

        # DEBUG BASELINE: disable arbitrage/load-shave gating to avoid hard infeasibility
        # Final gating (arb/peak) is temporarily disabled; economic preferences remain in objective.
        # if not (arb_ok or allow_load_shave) and price_buy >= 0:
        #     model += grid_to_batt[t] == 0

        # EV per-slot constraints: allow charging only within planning window and mask
        if ctx.ev_planning_disabled or ctx.ev_required_kwh is None or ctx.ev_required_kwh <= 1e-6:
            model += e[t] == 0
        else:
            # Respect global EV window indices
            if not (ctx.ev_window_start_index <= t < ctx.ev_window_end_index):
                model += e[t] == 0
            else:
                model += e[t] <= max_ev_charge_slot
            # Optional fine-grained EV allowed mask (e.g., cheapest slots)
            if t < len(ev_allowed_mask) and not bool(ev_allowed_mask[t]):
                model += e[t] == 0

        soc_this = battery_soc[t]
        model += soc_this == battery_soc_prev + BATTERY_EFFICIENCY_IN * b_in[t] - (1 / BATTERY_EFFICIENCY_OUT) * b_out[t]
        battery_soc_prev = soc_this
        model += soc_this >= BATTERY_MIN_SOC_KWH, f"battery_min_soc_{t}"

        ev_soc_this = ev_soc[t]
        model += ev_soc_this == ev_soc_prev + e[t]
        ev_soc_prev = ev_soc_this

        term = -price_buy * g_buy[t] + effective_sell_price * g_sell[t]
        term -= BATTERY_CYCLE_COST_DKK_PER_KWH * (b_in[t] + b_out[t])
        if ev_bonus_allowed and not ctx.ev_planning_disabled:
            term += EV_CHARGE_BONUS_DKK_PER_KWH * e[t]
        disable_hold_penalty = False
        if hold_value_threshold > 0:
            if reserve_schedule[t] > (battery_soc_initial + 0.5):
                disable_hold_penalty = True
        if hold_value_threshold > 0 and not disable_hold_penalty:
            term -= max(0.0, price_buy - hold_value_threshold) * grid_to_batt[t]
            penalty = hold_value_penalty_per_kwh[t]
            if penalty > 0:
                term -= penalty * g_sell[t]
        elif hold_value_threshold > 0 and disable_hold_penalty:
            penalty = hold_value_penalty_per_kwh[t]
            if penalty > 0:
                term -= penalty * g_sell[t]
        # Cheap-slot precharge incentive and peak-buy penalty
        if t < len(cheap_flags) and bool(cheap_flags[t]):
            cheap_bonus = 0.03
            if hold_value_threshold > 0:
                cheap_bonus += max(0.0, hold_value_threshold - price_buy) * 0.05
            term += cheap_bonus * grid_to_batt[t]
        if t < len(expensive_flags) and bool(expensive_flags[t]) and lambda_peak > 0:
            term += -lambda_peak * must_buy_house[t]

        # Discourage PV curtailment slightly
        term -= 0.02 * prod_curtailed[t]
        
        # CRITICAL: Massively penalize unmet house load (forces solver to meet demand at all costs)
        term -= 1000.0 * house_load_unmet[t]
        
        objective_terms.append(term)

    # --- Soft reserve constraint (min. SoC med straf i målfunktionen) ---
    reserve_shortfall = lp.LpVariable.dicts(
        "reserve_shortfall",
        range(periods),
        lowBound=0,
        upBound=None,
        cat="Continuous",
    )

    for t in range(periods):
        model += (
            battery_soc[t] + reserve_shortfall[t] >= reserve_schedule[t],
            f"reserve_min_soc_soft_{t}",
        )

    total_reserve_shortfall = lp.lpSum(reserve_shortfall[t] for t in range(periods))
    # Dimensions fix: multiply reserve_penalty (DKK/kWh) by slot_hours to get DKK per slot
    slot_hours = ctx.resolution_minutes / 60.0
    objective_terms.append(-ctx.reserve_penalty_per_kwh * slot_hours * total_reserve_shortfall)

    # EV soft target: require total energy over allowed window, but allow slack if impossible
    ev_required = max(0.0, float(ctx.ev_required_kwh or 0.0))
    ev_shortfall_var = None
    if ctx.ev_planning_disabled or ev_required <= 1e-6:
        # Ensure no EV schedule when planning is disabled or no requirement
        model += lp.lpSum(e[t] for t in range(periods)) == 0
    else:
        # Build allowed EV indices based on global window and mask
        allowed_indices: list[int] = []
        for t in range(periods):
            if not (ctx.ev_window_start_index <= t < ctx.ev_window_end_index):
                continue
            if t < len(ev_allowed_mask) and not bool(ev_allowed_mask[t]):
                continue
            allowed_indices.append(t)

        # If nothing is allowed, fall back to no EV (avoid infeasibility)
        if not allowed_indices:
            model += lp.lpSum(e[t] for t in range(periods)) == 0
        else:
            ev_shortfall = lp.LpVariable("ev_shortfall", lowBound=0)
            ev_shortfall_var = ev_shortfall
            model += lp.lpSum(e[t] for t in allowed_indices) + ev_shortfall == ev_required
            # Penalise EV shortfall strongly in the objective (discourage under-delivery)
            LARGE_EV_PENALTY = 10.0  # DKK per kWh of unmet EV energy (tunable)
            objective_terms.append(-LARGE_EV_PENALTY * ev_shortfall)

    # Enforce cumulative EV deadlines (per window targets)
    if getattr(ctx, "ev_cumulative_deadlines", None):
        slack_term = ev_shortfall_var if ev_shortfall_var is not None else 0.0
        for deadline_index, required in ctx.ev_cumulative_deadlines:
            idx = int(max(0, min(periods, deadline_index)))
            if idx <= 0:
                continue
            model += lp.lpSum(e[t] for t in range(idx)) + slack_term >= float(required)

    # Hysteresis / min-runtime logic remains disabled; only per-slot c_mode/d_mode constraints apply.

    # Cap total precharge from grid across cheap slots
    try:
        if precharge_cap > 0:
            model += lp.lpSum(grid_to_batt[t] for t in range(periods) if t < len(cheap_flags) and bool(cheap_flags[t])) <= precharge_cap
    except Exception:
        pass

    model += lp.lpSum(objective_terms)

    model.solve(lp.PULP_CBC_CMD(msg=False))

    status = lp.LpStatus.get(model.status, "unknown")
    objective_value = float(lp.value(model.objective)) if model.objective is not None else 0.0

    ev_target_series = list(getattr(ctx, "ev_target_pct_series", []))
    if len(ev_target_series) < periods:
        ev_target_series = (ev_target_series + [None] * periods)[:periods]

    plan = pd.DataFrame(
        {
            "timestamp": forecast["timestamp"],
            "g_buy": [g_buy[t].varValue for t in range(periods)],
            "g_sell": [g_sell[t].varValue for t in range(periods)],
            "battery_in": [b_in[t].varValue for t in range(periods)],
            "battery_out": [b_out[t].varValue for t in range(periods)],
            "ev_charge": [e[t].varValue for t in range(periods)],
            "battery_soc": [battery_soc[t].varValue for t in range(periods)],
            "ev_soc": [ev_soc[t].varValue for t in range(periods)],
            "prod_to_house": [prod_to_house[t].varValue for t in range(periods)],
            "prod_to_batt": [prod_to_batt[t].varValue for t in range(periods)],
            "prod_to_ev": [prod_to_ev[t].varValue for t in range(periods)],
            "prod_curtailed": [prod_curtailed[t].varValue for t in range(periods)],
            "batt_to_house": [batt_to_house[t].varValue for t in range(periods)],
            "batt_to_ev": [batt_to_ev[t].varValue for t in range(periods)],
            "batt_to_sell": [batt_to_sell[t].varValue for t in range(periods)],
            "grid_to_house": [grid_to_house[t].varValue for t in range(periods)],
            "grid_to_ev": [grid_to_ev[t].varValue for t in range(periods)],
            "grid_to_batt": [grid_to_batt[t].varValue for t in range(periods)],
            "house_load_unmet": [house_load_unmet[t].varValue for t in range(periods)],  # DEBUG: track slack usage
            "battery_reserve_target": [reserve_schedule[t] if t < len(reserve_schedule) else BATTERY_MIN_SOC_KWH for t in range(periods)],
            "battery_reserve_shortfall": [reserve_shortfall[t].varValue for t in range(periods)],
            "ev_target_pct": ev_target_series,
            "effective_sell_price": effective_sell_prices,
            "hold_value_penalty_per_kwh": hold_value_penalty_per_kwh,
            "hold_value_penalty_dkk": [
                hold_value_penalty_per_kwh[t] * (g_sell[t].varValue or 0.0) for t in range(periods)
            ],
            # Diagnostics for arbitrage/wait logic
            "arb_gate": arbitrage_gate_list,
            "arb_reason": arbitrage_reason_list,
            "arb_basis": ["eff_sell"] * periods,
            "arb_eta_rt": [ETA_RT] * periods,
            "arb_c_cycle": [CYCLE_COST] * periods,
            "price_buy_now": price_buy_series,
            "future_max_sell_eff": future_max_eff_sell,
            "load_shave_gate": load_shave_gate_list,
            "load_shave_reason": load_shave_reason_list,
            "future_max_buy": future_max_buy,
            "charge_mode": [c_mode[t].varValue for t in range(periods)],
            "discharge_mode": [d_mode[t].varValue for t in range(periods)],
            "cheap24": [bool(cheap_flags[t]) for t in range(periods)],
            "expensive24": [bool(expensive_flags[t]) for t in range(periods)],
            "must_buy_house": [must_buy_house[t].varValue for t in range(periods)],
        }
    )

    # Numerical hygiene: enforce gate semantics in post-processing to avoid tiny leakage
    # from solver tolerances. If arbitrage gate is false and price >= 0, grid_to_batt must be 0.
    try:
        # DISABLED: This post-processing was too aggressive and hid valid flows (e.g. load shaving)
        # causing g_buy != sum(destinations). We trust the solver's output.
        # if "arb_gate" in plan.columns and "grid_to_batt" in plan.columns:
        #     price_now = plan.get("price_buy_now", plan.get("price_buy"))
        #     mask = (~plan["arb_gate"]) & (price_now >= 0)
        #     if "battery_reserve_shortfall" in plan.columns:
        #         mask &= plan["battery_reserve_shortfall"] < 1e-3
        #     try:
        #         if float(getattr(ctx, "ev_required_kwh", 0.0) or 0.0) > 0:
        #             ev_start = max(0, int(getattr(ctx, "ev_window_start_index", 0)))
        #             ev_end = min(len(plan), int(getattr(ctx, "ev_window_end_index", len(plan))))
        #             ev_mask = pd.Series(True, index=plan.index)
        #             ev_mask.iloc[ev_start:ev_end] = False
        #             mask &= ev_mask
        #     except Exception:
        #         pass
        #     plan.loc[mask, "grid_to_batt"] = 0.0
        # Clamp tiny numerical noise to zero for key flow columns
        for col in [
            "g_buy",
            "g_sell",
            "battery_in",
            "battery_out",
            "grid_to_batt",
            "prod_to_batt",
            "batt_to_sell",
            "batt_to_house",
            "batt_to_ev",
            "grid_to_ev",
            "grid_to_house",
        ]:
            if col in plan.columns:
                plan[col] = plan[col].apply(lambda v: 0.0 if (abs(float(v or 0.0)) < 1e-9) else float(v or 0.0))
    except Exception:  # pragma: no cover - defensive error path
        pass

    # CRITICAL ASSERTION: Check for unmet house load (slack variable usage)
    if "house_load_unmet" in plan.columns:
        unmet_total = float(plan["house_load_unmet"].sum())
        if unmet_total > 0.01:
            logger.warning(
                "⚠️ WARNING: UNMET HOUSE LOAD DETECTED! Total unmet: %.3f kWh across %d slots",
                unmet_total,
                (plan["house_load_unmet"] > 0.01).sum(),
            )
            # Log which time slots have unmet load
            unmet_slots = plan[plan["house_load_unmet"] > 0.01][["timestamp", "house_load_unmet"]]
            if not unmet_slots.empty:
                logger.warning("Slots with unmet load:\n%s", unmet_slots.head(10))

    # Add house_unmet_reason column with diagnostic reasons
    house_unmet_reasons = []
    for t in range(periods):
        plan_row = plan.iloc[t] if t < len(plan) else pd.Series()
        forecast_row = forecast.iloc[t] if t < len(forecast) else pd.Series()
        reason = _diagnose_unmet_slot(t, plan_row, forecast_row, ctx)
        house_unmet_reasons.append(reason)
    
    plan["house_unmet_reason"] = house_unmet_reasons

    return OptimizationResult(plan=plan, objective_value=objective_value, status=status)
