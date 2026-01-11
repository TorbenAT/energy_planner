"""Central schema definition for energy plan columns.

This is the SINGLE SOURCE OF TRUTH for all plan columns.
Use this everywhere to avoid duplication and inconsistency.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ColumnDef:
    """Definition of a single plan column."""
    name: str
    category: str  # flow, state, price, economics, forecast, time, diagnostic
    dtype: str  # float, bool, str, timestamp
    db_required: bool = True  # Should be in database?
    csv_export: bool = True  # Should be in CSV export?
    ha_sensor: bool = True  # Should be in HA sensor?
    description: str = ""


# ============================================================================
# MASTER COLUMN SCHEMA
# This defines ALL columns that can exist in a plan
# ============================================================================

PLAN_COLUMNS: List[ColumnDef] = [
    # Timestamps
    # Note: HA uses timestamp/timestamp_local, DB uses timestamp_utc/local_time
    ColumnDef("timestamp", "time", "timestamp", db_required=False, ha_sensor=True, description="UTC timestamp (HA only)"),
    ColumnDef("timestamp_local", "time", "timestamp", db_required=False, ha_sensor=True, description="Local time (HA only)"),
    # Note: HA uses activity, DB uses note
    ColumnDef("activity", "flow", "str", db_required=False, ha_sensor=True, description="Human-readable activity (HA only)"),
    
    # Grid flows
    ColumnDef("g_buy", "flow", "float", description="Total grid purchase (kWh)"),
    ColumnDef("g_sell", "flow", "float", description="Total grid export (kWh)"),
    ColumnDef("grid_to_batt", "flow", "float", description="Grid → Battery (kWh)"),
    ColumnDef("grid_to_house", "flow", "float", description="Grid → House (kWh)"),
    ColumnDef("grid_to_ev", "flow", "float", description="Grid → EV (kWh)"),
    
    # PV flows (DB uses pv_*, solver uses prod_*)
    ColumnDef("pv_to_batt", "flow", "float", description="PV → Battery (kWh)"),
    ColumnDef("pv_to_house", "flow", "float", description="PV → House (kWh)"),
    ColumnDef("pv_to_ev", "flow", "float", description="PV → EV (kWh)"),
    
    # Battery flows
    ColumnDef("batt_to_house", "flow", "float", description="Battery → House (kWh)"),
    ColumnDef("batt_to_sell", "flow", "float", description="Battery → Grid (kWh)"),
    ColumnDef("batt_to_ev", "flow", "float", description="Battery → EV (kWh)"),
    
    # State
    ColumnDef("battery_in", "state", "float", description="Total battery charge (kWh)"),
    ColumnDef("battery_out", "state", "float", description="Total battery discharge (kWh)"),
    ColumnDef("battery_soc", "state", "float", description="Battery State of Charge (kWh)"),
    ColumnDef("battery_soc_pct", "state", "float", description="Battery SoC (%)"),
    ColumnDef("battery_reserve_target", "state", "float", description="Battery reserve target (kWh)"),
    ColumnDef("battery_reserve_shortfall", "state", "float", description="Battery reserve shortfall (kWh)"),
    ColumnDef("ev_charge", "state", "float", description="EV charging (kWh)"),
    ColumnDef("ev_soc_kwh", "state", "float", description="EV State of Charge (kWh)"),
    ColumnDef("ev_soc_pct", "state", "float", description="EV SoC (%)"),
    
    # Prices
    ColumnDef("price_buy", "price", "float", description="Grid buy price (DKK/kWh)"),
    ColumnDef("price_sell", "price", "float", description="Grid sell price (DKK/kWh)"),
    ColumnDef("effective_sell_price", "price", "float", description="Effective sell price after penalties (DKK/kWh)"),
    
    # Economics
    ColumnDef("grid_cost", "economics", "float", description="Grid purchase cost (DKK)"),
    ColumnDef("grid_revenue_effective", "economics", "float", description="Grid export revenue effective (DKK)"),
    ColumnDef("grid_revenue", "economics", "float", description="Grid export revenue nominal (DKK)"),
    ColumnDef("cash_cost_dkk", "economics", "float", description="Cash cost (DKK)"),
    ColumnDef("ev_bonus", "economics", "float", description="EV charging bonus (DKK)"),
    ColumnDef("battery_cycle_cost", "economics", "float", description="Battery degradation cost (DKK)"),
    ColumnDef("battery_value_dkk", "economics", "float", description="Battery state value (DKK)"),
    ColumnDef("objective_component", "economics", "float", description="Optimizer objective component"),
    
    # Forecast inputs
    ColumnDef("consumption_estimate_kw", "forecast", "float", description="Estimated consumption (kW)"),
    ColumnDef("pv_forecast_kw", "forecast", "float", description="PV generation forecast (kW)"),
    
    # House supply breakdown (calculated fields - MUST be in DB for full visibility)
    ColumnDef("house_from_grid", "flow", "float", db_required=True, ha_sensor=True, description="House consumption from grid (kWh) - calculated"),
    ColumnDef("house_from_battery", "flow", "float", db_required=True, ha_sensor=True, description="House consumption from battery (kWh) - calculated"),
    ColumnDef("house_from_pv", "flow", "float", db_required=True, ha_sensor=True, description="House consumption from PV (kWh) - calculated"),
    
    # Time classifications
    ColumnDef("cheap24", "time", "bool", description="In cheap 24h period"),
    ColumnDef("expensive24", "time", "bool", description="In expensive 24h period"),
    
    # Arbitrage diagnostics
    ColumnDef("arb_gate", "diagnostic", "bool", description="Arbitrage allowed (True/False)"),
    ColumnDef("arb_reason", "diagnostic", "str", description="Arbitrage decision reason"),
    ColumnDef("arb_basis", "diagnostic", "str", description="Arbitrage pricing basis"),
    ColumnDef("arb_eta_rt", "diagnostic", "float", description="Round-trip efficiency"),
    ColumnDef("arb_c_cycle", "diagnostic", "float", description="Cycle cost (DKK/kWh)"),
    ColumnDef("price_buy_now", "diagnostic", "float", description="Current buy price (DKK/kWh)"),
    ColumnDef("future_max_sell_eff", "diagnostic", "float", description="Future max effective sell price (DKK/kWh)"),
    ColumnDef("arb_margin", "diagnostic", "float", description="Arbitrage margin (DKK/kWh)"),
    
    # Mode classification (energy system mode for dashboard)
    ColumnDef("recommended_mode", "classification", "str", ha_sensor=True, db_required=True, csv_export=True, description="Recommended energy system mode"),
    
    # Policy diagnostics
    ColumnDef("policy_wait_flag", "diagnostic", "bool", description="Waiting for cheaper prices"),
    ColumnDef("policy_wait_reason", "diagnostic", "str", description="Wait reason"),
    ColumnDef("policy_price_basis", "diagnostic", "str", description="Policy price basis (buy/sell)"),
    ColumnDef("policy_price_now_dkk", "diagnostic", "float", description="Current policy price (DKK/kWh)"),
    ColumnDef("policy_future_min_12h_dkk", "diagnostic", "float", description="Future min price 12h (DKK/kWh)"),
    ColumnDef("policy_grid_charge_allowed", "diagnostic", "bool", description="Grid charging allowed"),
    ColumnDef("policy_hold_value_dkk", "diagnostic", "float", description="Battery hold value (DKK/kWh)"),
    
    # Unmet load diagnostics
    ColumnDef("house_load_unmet", "diagnostic", "float", description="Unmet house load (kWh)"),
    ColumnDef("house_unmet_reason", "diagnostic", "str", description="Reason for unmet house load"),
    
    # DB-only columns (not in HA sensor or CSV)
    ColumnDef("timestamp_utc", "time", "timestamp", db_required=True, ha_sensor=False, csv_export=False, description="UTC timestamp (DB PRIMARY KEY)"),
    ColumnDef("local_time", "time", "timestamp", db_required=True, ha_sensor=False, csv_export=False, description="Local time (DB column)"),
    ColumnDef("note", "flow", "str", db_required=True, ha_sensor=False, csv_export=False, description="Activity note (DB column)"),
    ColumnDef("day_id", "time", "date", db_required=True, ha_sensor=False, csv_export=False, description="Day ID (DB column)"),
    ColumnDef("house_load", "flow", "float", db_required=True, ha_sensor=False, csv_export=False, description="House load (DB column)"),
]


# ============================================================================
# DERIVED LISTS (auto-generated from PLAN_COLUMNS)
# ============================================================================

def get_column_names(filter_by: Dict[str, any] = None) -> List[str]:
    """Get column names, optionally filtered."""
    cols = PLAN_COLUMNS
    if filter_by:
        for key, value in filter_by.items():
            cols = [c for c in cols if getattr(c, key) == value]
    return [c.name for c in cols]


def get_columns_by_category() -> Dict[str, List[str]]:
    """Group columns by category."""
    result = {}
    for col in PLAN_COLUMNS:
        result.setdefault(col.category, []).append(col.name)
    return result


# Standard lists for different consumers
PLAN_FIELDS_HA = get_column_names({"ha_sensor": True})  # For Home Assistant sensor.py
PLAN_FIELDS_DB = get_column_names({"db_required": True})  # For database writes
PLAN_FIELDS_CSV = get_column_names({"csv_export": True})  # For CSV export

# Column info lookup
COLUMN_INFO: Dict[str, ColumnDef] = {c.name: c for c in PLAN_COLUMNS}


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_dataframe(df, source: str = "unknown") -> Tuple[List[str], List[str]]:
    """Validate a DataFrame against the schema.
    
    Returns:
        (missing_required, extra_columns)
    """
    expected = set(PLAN_FIELDS_HA)
    actual = set(df.columns)
    
    missing = expected - actual
    extra = actual - expected
    
    return sorted(missing), sorted(extra)


def get_column_mapping() -> Dict[str, str]:
    """Get mapping for renamed columns (e.g., prod_to_* → pv_to_*)."""
    return {
        "prod_to_batt": "pv_to_batt",
        "prod_to_house": "pv_to_house",
        "prod_to_ev": "pv_to_ev",
        "ev_soc": "ev_soc_kwh",  # Alias
    }


def print_schema_summary():
    """Print a human-readable schema summary."""
    print("=" * 80)
    print("ENERGY PLAN SCHEMA - MASTER DEFINITION")
    print("=" * 80)
    print(f"\nTotal columns defined: {len(PLAN_COLUMNS)}")
    
    by_category = get_columns_by_category()
    for category, cols in sorted(by_category.items()):
        print(f"\n{category.upper()} ({len(cols)} columns):")
        for col in cols:
            info = COLUMN_INFO[col]
            flags = []
            if info.db_required:
                flags.append("DB")
            if info.ha_sensor:
                flags.append("HA")
            if info.csv_export:
                flags.append("CSV")
            flags_str = f"[{','.join(flags)}]"
            print(f"  {col:35s} {flags_str:15s} {info.description}")
    
    print(f"\n{'=' * 80}")
    print(f"HA Sensor columns: {len(PLAN_FIELDS_HA)}")
    print(f"Database columns: {len(PLAN_FIELDS_DB)}")
    print(f"CSV export columns: {len(PLAN_FIELDS_CSV)}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    print_schema_summary()
