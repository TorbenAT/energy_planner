"""Centralized technical limits and coefficients reused across modules."""

from __future__ import annotations

# --- Technical limits (hourly) ---
MAX_GRID_BUY_KWH = 17.0          # Max grid import per hour [kWh]
MAX_GRID_SELL_KWH = 12.0         # Max grid export per hour [kWh]
MAX_BATTERY_CHARGE_KWH = 11.0    # Max battery charge rate per hour [kWh]
MAX_BATTERY_DISCHARGE_KWH = 11.0 # Max battery discharge rate per hour [kWh]
MAX_EV_CHARGE_KWH = 10.0         # Max EV charge per hour [kWh]
MAX_INVERTER_OUTPUT_KWH = 12.0   # Max combined inverter output per hour [kWh]

SLOTS_PER_HOUR = 4  # 15-min opløsning

# --- Per-slot limits (kWh per 15-minut slot) ---
MAX_GRID_BUY_QH = MAX_GRID_BUY_KWH / SLOTS_PER_HOUR          # 4.25 kWh/slot
MAX_GRID_SELL_QH = MAX_GRID_SELL_KWH / SLOTS_PER_HOUR        # 3.00 kWh/slot
MAX_BATTERY_CHARGE_QH = MAX_BATTERY_CHARGE_KWH / SLOTS_PER_HOUR
MAX_BATTERY_DISCHARGE_QH = MAX_BATTERY_DISCHARGE_KWH / SLOTS_PER_HOUR
MAX_EV_CHARGE_QH = MAX_EV_CHARGE_KWH / SLOTS_PER_HOUR
MAX_INVERTER_OUTPUT_QH = MAX_INVERTER_OUTPUT_KWH / SLOTS_PER_HOUR

# --- Battery parameters ---
BATTERY_CAPACITY_KWH = 50.0
BATTERY_EFFICIENCY_IN = 0.98
BATTERY_EFFICIENCY_OUT = 0.98
BATTERY_MIN_SOC_KWH = 5.0

# --- EV parameters ---
EV_BATTERY_CAPACITY_KWH = 75.0
EV_DEFAULT_MIN_CHARGE_KWH = 25.0

# --- Economics ---
BATTERY_CYCLE_COST_DKK_PER_KWH = 0.15
EV_CHARGE_BONUS_DKK_PER_KWH = 0.15
MIN_SELL_PRICE_DKK = 0.04

# --- Planning horizon ---
DEFAULT_LOOKAHEAD_HOURS = 168
DEFAULT_RESOLUTION_MINUTES = 15

# --- Database defaults ---
DEFAULT_SCHEMA = "energy_planner"
FORECAST_TABLE = "forecast_quarter_hour"
ACTUAL_TABLE = "actual_quarter_hour"
RUN_HISTORY_TABLE = "optimizer_runs"
LEARNING_METRICS_TABLE = "learning_metrics"
PLAN_EVALUATIONS_TABLE = "plan_evaluations"
PLAN_QUARTER_ECON_TABLE = "plan_quarter_economics"
PLAN_SLOTS_TABLE = "energy_plan_slots"

