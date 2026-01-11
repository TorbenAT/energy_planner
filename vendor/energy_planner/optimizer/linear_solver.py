"""Simplified Linear Programming solver for quarter-hour planning."""

from __future__ import annotations

import logging
import pulp as lp
import pandas as pd
import numpy as np

from ..constants import (
    BATTERY_CAPACITY_KWH,
    BATTERY_CYCLE_COST_DKK_PER_KWH,
    BATTERY_EFFICIENCY_IN,
    BATTERY_EFFICIENCY_OUT,
    BATTERY_MIN_SOC_KWH,
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

def solve_optimization_linear(
    forecast_df: pd.DataFrame,
    ctx: OptimizationContext
) -> OptimizationResult:
    """Solve the energy planning problem using pure Linear Programming (no binary vars)."""
    
    # Time horizon
    T = len(forecast_df)
    if T == 0:
        return OptimizationResult(pd.DataFrame(), 0.0, "Empty Forecast")

    # Create LP problem
    prob = lp.LpProblem("EnergyPlan_Linear", lp.LpMinimize)

    # --- Variables ---
    # Grid
    g_buy = lp.LpVariable.dicts("g_buy", range(T), lowBound=0, upBound=MAX_GRID_BUY_QH)
    g_sell = lp.LpVariable.dicts("g_sell", range(T), lowBound=0, upBound=MAX_GRID_SELL_QH)
    
    # Battery
    batt_charge = lp.LpVariable.dicts("batt_charge", range(T), lowBound=0, upBound=MAX_BATTERY_CHARGE_QH)
    batt_discharge = lp.LpVariable.dicts("batt_discharge", range(T), lowBound=0, upBound=MAX_BATTERY_DISCHARGE_QH)
    batt_soc = lp.LpVariable.dicts("batt_soc", range(T+1), lowBound=BATTERY_MIN_SOC_KWH, upBound=BATTERY_CAPACITY_KWH)
    
    # EV
    ev_charge = lp.LpVariable.dicts("ev_charge", range(T), lowBound=0, upBound=MAX_EV_CHARGE_QH)
    ev_soc = lp.LpVariable.dicts("ev_soc", range(T+1), lowBound=0, upBound=EV_BATTERY_CAPACITY_KWH)

    # Flows (Internal routing)
    grid_to_house = lp.LpVariable.dicts("grid_to_house", range(T), lowBound=0)
    grid_to_batt = lp.LpVariable.dicts("grid_to_batt", range(T), lowBound=0)
    grid_to_ev = lp.LpVariable.dicts("grid_to_ev", range(T), lowBound=0)
    
    pv_to_house = lp.LpVariable.dicts("pv_to_house", range(T), lowBound=0)
    pv_to_batt = lp.LpVariable.dicts("pv_to_batt", range(T), lowBound=0)
    pv_to_ev = lp.LpVariable.dicts("pv_to_ev", range(T), lowBound=0)
    pv_to_sell = lp.LpVariable.dicts("pv_to_sell", range(T), lowBound=0)
    
    batt_to_house = lp.LpVariable.dicts("batt_to_house", range(T), lowBound=0)
    batt_to_sell = lp.LpVariable.dicts("batt_to_sell", range(T), lowBound=0)
    batt_to_ev = lp.LpVariable.dicts("batt_to_ev", range(T), lowBound=0)

    # Initial State
    prob += batt_soc[0] == ctx.battery_soc_kwh
    prob += ev_soc[0] == ctx.ev_soc_kwh

    # Objective Function components
    total_cost = 0

    for t in range(T):
        # --- Parameters ---
        row = forecast_df.iloc[t]
        pv_avail = float(row["pv_forecast_kw"]) * (ctx.resolution_minutes / 60.0) # Convert kW to kWh
        house_load = float(row["consumption_estimate_kw"]) * (ctx.resolution_minutes / 60.0)
        
        price_buy = float(row["price_allin_buy"])
        price_sell = float(row["price_eff_sell"])
        
        # --- Constraints ---
        
        # 1. Balance: House Load
        prob += (grid_to_house[t] + pv_to_house[t] + batt_to_house[t] >= house_load), f"HouseBalance_{t}"
        
        # 2. Balance: PV Generation
        prob += (pv_to_house[t] + pv_to_batt[t] + pv_to_ev[t] + pv_to_sell[t] <= pv_avail), f"PVBalance_{t}"
        
        # 3. Balance: Grid Buy
        prob += (g_buy[t] == grid_to_house[t] + grid_to_batt[t] + grid_to_ev[t]), f"GridBuyDef_{t}"
        
        # 4. Balance: Grid Sell
        prob += (g_sell[t] == pv_to_sell[t] + batt_to_sell[t]), f"GridSellDef_{t}"
        
        # 5. Balance: Battery Charge (In)
        prob += (batt_charge[t] == grid_to_batt[t] + pv_to_batt[t]), f"BattChargeDef_{t}"
        
        # 6. Balance: Battery Discharge (Out)
        prob += (batt_discharge[t] == batt_to_house[t] + batt_to_sell[t] + batt_to_ev[t]), f"BattDischargeDef_{t}"
        
        # 7. Balance: EV Charge
        prob += (ev_charge[t] == grid_to_ev[t] + pv_to_ev[t] + batt_to_ev[t]), f"EVChargeDef_{t}"
        
        # 8. Battery State of Charge
        prob += (batt_soc[t+1] == batt_soc[t] + batt_charge[t] * BATTERY_EFFICIENCY_IN - batt_discharge[t] / BATTERY_EFFICIENCY_OUT), f"BattSoC_{t}"
        
        # 9. EV State of Charge
        prob += (ev_soc[t+1] == ev_soc[t] + ev_charge[t]), f"EVSoC_{t}"
        
        # 10. EV Constraints
        if not ctx.ev_allowed_mask[t]:
             prob += ev_charge[t] == 0
             
        # --- Objective ---
        cost_grid = g_buy[t] * price_buy
        rev_grid = g_sell[t] * price_sell
        cost_batt = (batt_discharge[t] + batt_charge[t]) * (BATTERY_CYCLE_COST_DKK_PER_KWH / 2.0)
        bonus_ev = ev_charge[t] * EV_CHARGE_BONUS_DKK_PER_KWH
        
        total_cost += cost_grid - rev_grid + cost_batt - bonus_ev
        
        # Reserve Penalty
        if t < len(ctx.battery_reserve_schedule):
            reserve = ctx.battery_reserve_schedule[t]
            shortfall = lp.LpVariable(f"shortfall_{t}", lowBound=0)
            prob += batt_soc[t+1] >= reserve - shortfall
            total_cost += shortfall * ctx.reserve_penalty_per_kwh

    # EV Target Constraint
    if ctx.ev_required_kwh and ctx.ev_required_kwh > 0:
         deadline = T
         if ctx.ev_charge_deadline_index and ctx.ev_charge_deadline_index < T:
             deadline = ctx.ev_charge_deadline_index
         prob += ev_soc[deadline] >= ctx.ev_soc_kwh + ctx.ev_required_kwh

    prob += total_cost

    # Solve
    status = prob.solve(lp.PULP_CBC_CMD(msg=False))
    
    # Extract results
    rows = []
    for t in range(T):
        row = {}
        row["timestamp"] = forecast_df.index[t]
        
        # Flows
        row["g_buy"] = lp.value(g_buy[t])
        row["g_sell"] = lp.value(g_sell[t])
        row["grid_to_batt"] = lp.value(grid_to_batt[t])
        row["grid_to_house"] = lp.value(grid_to_house[t])
        row["grid_to_ev"] = lp.value(grid_to_ev[t])
        
        row["prod_to_batt"] = lp.value(pv_to_batt[t])
        row["prod_to_house"] = lp.value(pv_to_house[t])
        row["prod_to_ev"] = lp.value(pv_to_ev[t])
        
        row["batt_to_house"] = lp.value(batt_to_house[t])
        row["batt_to_sell"] = lp.value(batt_to_sell[t])
        row["batt_to_ev"] = lp.value(batt_to_ev[t])
        
        row["battery_in"] = lp.value(batt_charge[t])
        row["battery_out"] = lp.value(batt_discharge[t])
        row["battery_soc"] = lp.value(batt_soc[t+1])
        row["battery_soc_pct"] = (row["battery_soc"] / BATTERY_CAPACITY_KWH) * 100.0
        
        row["ev_charge"] = lp.value(ev_charge[t])
        row["ev_soc_kwh"] = lp.value(ev_soc[t+1])
        row["ev_soc_pct"] = (row["ev_soc_kwh"] / EV_BATTERY_CAPACITY_KWH) * 100.0 if EV_BATTERY_CAPACITY_KWH > 0 else 0
        
        # Economics
        p_buy = float(forecast_df.iloc[t]["price_allin_buy"])
        p_sell = float(forecast_df.iloc[t]["price_eff_sell"])
        
        row["grid_cost"] = row["g_buy"] * p_buy
        row["grid_revenue_effective"] = row["g_sell"] * p_sell
        row["battery_cycle_cost"] = (row["battery_in"] + row["battery_out"]) * (BATTERY_CYCLE_COST_DKK_PER_KWH / 2.0)
        row["ev_bonus"] = row["ev_charge"] * EV_CHARGE_BONUS_DKK_PER_KWH
        
        rows.append(row)
        
    result_df = pd.DataFrame(rows)
    return OptimizationResult(result_df, lp.value(prob.objective), lp.LpStatus[status])
