"""Simple Economic Linear Solver for quarter-hour planning."""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np

from ..constants import (
    BATTERY_CAPACITY_KWH,
    BATTERY_CYCLE_COST_DKK_PER_KWH,
    BATTERY_EFFICIENCY_IN,
    BATTERY_EFFICIENCY_OUT,
    EV_BATTERY_CAPACITY_KWH,
    EV_CHARGE_BONUS_DKK_PER_KWH,
    MAX_BATTERY_CHARGE_QH,
    MAX_BATTERY_DISCHARGE_QH,
    MAX_EV_CHARGE_QH,
    MAX_GRID_BUY_QH,
    MAX_GRID_SELL_QH,
)
from .solver import OptimizationContext, OptimizationResult

logger = logging.getLogger(__name__)


def solve_optimization_simple(
    forecast_df: pd.DataFrame,
    ctx: OptimizationContext
) -> OptimizationResult:
    """
    Solve the energy planning problem using pure Economic Linear Programming.
    
    Objective:
      Minimize (GridCost - GridRevenue + BatteryWear)
      
    Constraints:
      - Physical limits (Grid, Battery, Inverter)
      - Hard Battery Minimum (User defined floor)
      - EV Driving Consumption (Modeled as drain when away)
    """
    
    import pulp as lp
    # Time horizon
    T = len(forecast_df)
    if T == 0:
        return OptimizationResult(pd.DataFrame(), 0.0, "Empty Forecast")

    # Resolution handling
    res_min = ctx.resolution_minutes if ctx.resolution_minutes > 0 else 15
    slots_per_hour = 60.0 / res_min
    
    # --- Dynamic Limits ---
    max_charge_qh = (ctx.max_charge_kwh / slots_per_hour) if ctx.max_charge_kwh else MAX_BATTERY_CHARGE_QH
    max_discharge_qh = (ctx.max_discharge_kwh / slots_per_hour) if ctx.max_discharge_kwh else MAX_BATTERY_DISCHARGE_QH
    max_ev_charge_qh = (ctx.max_ev_charge_kwh / slots_per_hour) if ctx.max_ev_charge_kwh else MAX_EV_CHARGE_QH
    max_grid_buy_qh = MAX_GRID_BUY_QH
    max_grid_sell_qh = MAX_GRID_SELL_QH

    # --- Problem Setup ---
    prob = lp.LpProblem("EnergyPlan_Simple_Economic", lp.LpMinimize)

    # --- Variables ---
    # Gross flows (total per slot)
    g_buy = lp.LpVariable.dicts("g_buy", range(T), lowBound=0, upBound=max_grid_buy_qh)
    g_sell = lp.LpVariable.dicts("g_sell", range(T), lowBound=0, upBound=max_grid_sell_qh)
    batt_charge = lp.LpVariable.dicts("batt_charge", range(T), lowBound=0, upBound=max_charge_qh)
    batt_discharge = lp.LpVariable.dicts("batt_discharge", range(T), lowBound=0, upBound=max_discharge_qh)
    ev_charge = lp.LpVariable.dicts("ev_charge", range(T), lowBound=0, upBound=max_ev_charge_qh)

    # State variables (kWh)
    # Physical Ceiling (from input_number.battery_maximum)
    max_soc_kwh = ctx.battery_capacity_kwh * (ctx.battery_maximum_pct / 100.0)
    # Physical Floor (from constants - usually 10% or 5kWh)
    batt_soc = lp.LpVariable.dicts("batt_soc", range(T + 1), lowBound=ctx.battery_min_soc_kwh, upBound=max_soc_kwh)
    ev_soc = lp.LpVariable.dicts("ev_soc", range(T + 1), lowBound=0, upBound=EV_BATTERY_CAPACITY_KWH)
    
    # Track shortfalls for reporting
    shortfall_vars = []

    # Flows (Routing)
    grid_to_house = lp.LpVariable.dicts("grid_to_house", range(T), lowBound=0)
    grid_to_batt = lp.LpVariable.dicts("grid_to_batt", range(T), lowBound=0)
    grid_to_ev = lp.LpVariable.dicts("grid_to_ev", range(T), lowBound=0)
    
    pv_to_house = lp.LpVariable.dicts("pv_to_house", range(T), lowBound=0)
    pv_to_batt = lp.LpVariable.dicts("pv_to_batt", range(T), lowBound=0)
    pv_to_ev = lp.LpVariable.dicts("pv_to_ev", range(T), lowBound=0)
    pv_to_sell = lp.LpVariable.dicts("pv_to_sell", range(T), lowBound=0)
    
    batt_to_house = lp.LpVariable.dicts("batt_to_house", range(T), lowBound=0)
    batt_to_sell = lp.LpVariable.dicts("batt_to_sell", range(T), lowBound=0)
    batt_to_ev = lp.LpVariable.dicts("batt_to_ev", range(T), lowBound=0) # V2H preparation, likely 0 bound

    # --- Initial State ---
    prob += batt_soc[0] == ctx.battery_soc_kwh
    prob += ev_soc[0] == ctx.ev_soc_kwh

    # --- Objective ---
    total_cost = 0

    # Efficiency Roundtrip
    rt_efficiency = BATTERY_EFFICIENCY_IN * BATTERY_EFFICIENCY_OUT

    # --- Loop over time ---
    for t in range(T):
        row = forecast_df.iloc[t]
        
        # 1. Physics Conversion (kW -> kWh per slot)
        pv_avail = float(row["pv_forecast_kw"]) * (ctx.resolution_minutes / 60.0)
        house_load = float(row["consumption_estimate_kw"]) * (ctx.resolution_minutes / 60.0)
        
        # 2. Prices
        price_buy = float(row["price_allin_buy"])
        price_sell = float(row["price_eff_sell"])

        # 3. Price-Gating (Professional Logic)
        # Lookahead for future cheapest buy price to determine if charging NOW makes sense.
        # We look ahead up to 24 hours (96 slots) or remainder of forecast.
        lookahead_window = 96 
        future_prices = forecast_df["price_allin_buy"].iloc[t+1 : t+1+lookahead_window].astype(float)
        
        if not future_prices.empty:
            min_future_buy = future_prices.min()
            # Profit/Savings Potential: V_store = (FutureMin * Eff) - CurrentBuy - Wear
            # Note: Cycle cost is applied per kWh charged+discharged effectively.
            v_store = (min_future_buy * rt_efficiency) - price_buy - BATTERY_CYCLE_COST_DKK_PER_KWH
        else:
            # End of forecast, assume no future benefit
            v_store = -1.0
        
        # If V_store <= 0, we penalize GRID->BATT to only happen if forced by floor limits
        gate_penalty_coeff = 0.0
        if v_store <= 0:
            # Heavy penalty to prevent grid charging at high prices
            # Penalty should be higher than any possible price difference
            gate_penalty_coeff = 10.0 
        
        # 4. EV Logic
        is_ev_home = bool(row["ev_available"]) if "ev_available" in row else True
        ev_consumption_kwh = float(row.get("ev_driving_consumption_kwh", 0.0))

        if is_ev_home:
            prob += ev_soc[t+1] == ev_soc[t] + ev_charge[t] * 0.95 
            prob += batt_to_ev[t] == 0 
        else:
            prob += ev_charge[t] == 0
            prob += ev_soc[t+1] == ev_soc[t] - ev_consumption_kwh
        
        # 5. Battery Logic
        # HARD physical floor is enforced by batt_soc variable lowBound=ctx.battery_min_soc_kwh
        
        # SOFT floor (Reserve target)
        # We still keep it but with a very low penalty (0.001) so it doesn't trigger expensive buys.
        slot_reserve = ctx.battery_reserve_schedule[t] if t < len(ctx.battery_reserve_schedule) else ctx.battery_min_soc_kwh
        shortfall_v = lp.LpVariable(f"shortfall_{t}", lowBound=0)
        prob += batt_soc[t+1] >= slot_reserve - shortfall_v
        shortfall_vars.append(shortfall_v)

        # Battery Balance
        prob += batt_soc[t+1] == batt_soc[t] + \
                (batt_charge[t] * BATTERY_EFFICIENCY_IN) - \
                (batt_discharge[t] / BATTERY_EFFICIENCY_OUT)
                
        # 6. Energy Balance (The Node Constraints)
        prob += house_load == grid_to_house[t] + pv_to_house[t] + batt_to_house[t]
        prob += pv_avail >= pv_to_house[t] + pv_to_batt[t] + pv_to_ev[t] + pv_to_sell[t]
        prob += g_buy[t] == grid_to_house[t] + grid_to_batt[t] + grid_to_ev[t]
        prob += g_sell[t] == pv_to_sell[t] + batt_to_sell[t]
        prob += batt_charge[t] == grid_to_batt[t] + pv_to_batt[t]
        prob += batt_discharge[t] == batt_to_house[t] + batt_to_ev[t] + batt_to_sell[t]
        prob += ev_charge[t] == grid_to_ev[t] + pv_to_ev[t] + batt_to_ev[t]

        # 7. Costs (Objective)
        # Cost = Grid Buy - Grid Sell - EV Bonus + Wear + Gate Penalty + Reserve Penalty
        cost_slot = (g_buy[t] * price_buy) - (g_sell[t] * price_sell) - (ev_charge[t] * EV_CHARGE_BONUS_DKK_PER_KWH)
        
        # Cycle cost per kWh DISCHARGED (standard convention)
        wear_cost = batt_discharge[t] * BATTERY_CYCLE_COST_DKK_PER_KWH
        
        # Gate Penalty for Grid->Batt
        gate_penalty = grid_to_batt[t] * gate_penalty_coeff
        
        # Trivial penalty for reserve (only to break ties if prices are identical)
        penalty = shortfall_v * 0.001
        
        total_cost += cost_slot + wear_cost + gate_penalty + penalty

    # 8. Terminal Value (Salvage value of energy left in battery)
    # Without this, the solver will try to empty the battery at any sub-optimal time 
    # just to avoid leaving "worthless" energy behind at the end of the horizon.
    # We value the remaining SOC at the cheapest possible replacement cost (future min).
    # If battery_hold_value_dkk is too high (due to heuristic margins), it causes 'hoarding'.
    # We use the absolute minimum price in the forecast as a conservative replacement value.
    future_prices = forecast_df["price_allin_buy"].astype(float)
    hold_value = future_prices.min() if not future_prices.empty else 1.0
    
    terminal_value = batt_soc[T] * hold_value
    
    prob += total_cost - terminal_value

    # 8. EV Deadlines (Hard Constraints)
    # Note: ctx.ev_cumulative_deadlines is a list of (index, cumulative_kwh)
    deadlines = getattr(ctx, "ev_cumulative_deadlines", [])
    if deadlines:
        for deadline_idx, cumulative_kwh in deadlines:
            t_deadline = min(T, int(deadline_idx))
            if t_deadline > 0:
                # Enforce that we have charged at least 'cumulative_kwh' into the EV by this point
                # Since we don't discharge the EV in this model, sum(ev_charge) is sufficient.
                prob += lp.lpSum([ev_charge[t] for t in range(t_deadline)]) >= float(cumulative_kwh)

    # --- Solve ---
    status = prob.solve(lp.PULP_CBC_CMD(msg=0))
    
    # --- Results ---
    res_df = forecast_df.copy()
    
    # Ensure price aliases exist for reporting
    res_df["price_buy"] = res_df["price_allin_buy"]
    res_df["price_sell"] = res_df["price_eff_sell"]
    
    # Flow conversion multiplier: kWh/slot -> kW
    # (Energy / Time) = Power. Time = 1/slots_per_hour.
    # So Power = Energy * slots_per_hour
    kw_factor = slots_per_hour

    # Extract variable values and convert to kW where appropriate
    res_df["g_buy"] = [lp.value(g_buy[i]) * kw_factor for i in range(T)]
    res_df["g_sell"] = [lp.value(g_sell[i]) * kw_factor for i in range(T)]
    
    res_df["battery_in"] = [lp.value(batt_charge[i]) * kw_factor for i in range(T)]
    res_df["battery_out"] = [lp.value(batt_discharge[i]) * kw_factor for i in range(T)]
    
    # Detailed flows (kW)
    res_df["grid_to_batt"] = [lp.value(grid_to_batt[i]) * kw_factor for i in range(T)]
    res_df["grid_to_ev"] = [lp.value(grid_to_ev[i]) * kw_factor for i in range(T)]
    
    res_df["prod_to_batt"] = [lp.value(pv_to_batt[i]) * kw_factor for i in range(T)]
    res_df["prod_to_house"] = [lp.value(pv_to_house[i]) * kw_factor for i in range(T)]
    res_df["prod_to_ev"] = [lp.value(pv_to_ev[i]) * kw_factor for i in range(T)]
    
    # Implicit PV export (needed for curtailment calc)
    pv_to_sell_kw = [lp.value(pv_to_sell[i]) * kw_factor for i in range(T)]
    
    res_df["batt_to_house"] = [lp.value(batt_to_house[i]) * kw_factor for i in range(T)]
    res_df["batt_to_ev"] = [lp.value(batt_to_ev[i]) * kw_factor for i in range(T)]
    res_df["batt_to_sell"] = [lp.value(batt_to_sell[i]) * kw_factor for i in range(T)]
    
    res_df["grid_to_house"] = [lp.value(grid_to_house[i]) * kw_factor for i in range(T)]
    
    # Curtailment (Available - Used)
    # Using 'pv_forecast_kw' from original DF
    res_df["prod_curtailed"] = (
        res_df["pv_forecast_kw"] - 
        (res_df["prod_to_house"] + res_df["prod_to_batt"] + res_df["prod_to_ev"] + pv_to_sell_kw)
    ).clip(lower=0)
    
    # EV flows (kW) and state (kWh)
    res_df["ev_charge"] = [lp.value(ev_charge[i]) * kw_factor for i in range(T)]
    res_df["ev_soc"] = [lp.value(ev_soc[i+1]) for i in range(T)] # State is at end of slot
    
    # Battery state (kWh)
    res_df["battery_soc_kwh"] = [lp.value(batt_soc[i+1]) for i in range(T)] # End of slot
    res_df["battery_soc"] = res_df["battery_soc_kwh"] # Alias required by scheduler.py
    
    # Reserve and Shortfall
    res_df["battery_reserve_target"] = [
        ctx.battery_reserve_schedule[i] if i < len(ctx.battery_reserve_schedule) else ctx.battery_min_soc_kwh
        for i in range(T)
    ]
    res_df["battery_reserve_shortfall"] = [lp.value(shortfall_vars[i]) for i in range(T)]

    
    # Derived columns
    if ctx.battery_capacity_kwh > 0:
        res_df["battery_soc_pct"] = (res_df["battery_soc_kwh"] / ctx.battery_capacity_kwh) * 100
    else:
        res_df["battery_soc_pct"] = 0.0

    # Calculate costs for reporting (using kW * price/h / slots = kWh * price)
    # Actually just use original lp vars (kWh) * price
    # prices are DKK/kWh.
    # cost = Buy(kWh) * Price_Buy - Sell(kWh) * Price_Sell
    # We can reconstruct from kW columns: kW / slots_per_hour * price
    res_df["cost"] = (
        (res_df["g_buy"] / slots_per_hour * res_df["price_allin_buy"]) - 
        (res_df["g_sell"] / slots_per_hour * res_df["price_eff_sell"])
    )

    # Drop columns that will be merged in by reporting.py to avoid suffixes (e.g. price_buy_x)
    # reporting.py merges: "timestamp", "pv_forecast_kw", "consumption_estimate_kw", "price_buy", "price_sell"
    # We keep "timestamp" for the merge key.
    # Also drop "created_at" to avoid conflicts with DB writing (let scheduler/db handle it)
    cols_to_drop = ["price_buy", "price_sell", "pv_forecast_kw", "consumption_estimate_kw", "created_at"]
    res_df.drop(columns=[c for c in cols_to_drop if c in res_df.columns], inplace=True)

    # Final result construction
    notes = ["Optimeret med Simple Economic Linear Solver (PuLP)."]
    if lp.value(prob.objective) is None:
        notes.append("Advarsel: Objektivfunktion returnerede ingen vメrdi.")

    return OptimizationResult(
        plan=res_df,
        objective_value=lp.value(prob.objective) or 0.0,
        status=lp.LpStatus[status],
        notes=notes
    )
