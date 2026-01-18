"""Helpers for preparing optimisation outputs for Home Assistant/visualisation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union

import json
import math
import pandas as pd  # type: ignore
import pytz  # type: ignore

from .config import Settings, load_settings
from .constants import (
    BATTERY_CAPACITY_KWH,
    BATTERY_CYCLE_COST_DKK_PER_KWH,
    BATTERY_MIN_SOC_KWH,
    EV_BATTERY_CAPACITY_KWH,
    EV_CHARGE_BONUS_DKK_PER_KWH,
    MAX_EV_CHARGE_KWH,
    MAX_EV_CHARGE_QH,
    SLOTS_PER_HOUR,
)
from .data_pipeline import DataPipeline
from .db import create_session_factory
from .ha_client import HomeAssistantClient
from .optimizer.solver import EV_CHARGE_ALLOWED_STATUSES, OptimizationContext, OptimizationResult, solve_quarter_hour
from .scheduler import build_context, summarize_plan


@dataclass(slots=True)
class DebugSnapshot:
    """Container for debug-oriented diagnostics."""

    inputs: Dict[str, Union[float, int, str]]
    house_balance: pd.DataFrame

    def inputs_summary(self) -> Dict[str, float]:
        return dict(self.inputs)

    def house_balance_records(self, limit: Optional[int] = None, digits: int = 3) -> List[dict]:
        df = self.house_balance
        if limit is not None:
            df = df.head(limit)

        records: List[dict] = []
        for row in df.itertuples(index=False):
            row_dict = {}
            for column, value in zip(df.columns, row):
                if isinstance(value, float):
                    row_dict[column] = round(value, digits)
                else:
                    row_dict[column] = value.isoformat() if isinstance(value, pd.Timestamp) else value
            records.append(row_dict)
        return records


@dataclass(slots=True)
class PlanReport:
    """Structured representation of a single optimisation run."""

    status: str
    objective_value: float
    plan: pd.DataFrame
    hourly: pd.DataFrame
    summary: dict
    day_summary: List[dict]
    notes: List[str]
    generated_at: datetime
    updated_at: Optional[datetime]  # When plan was last updated in DB
    timezone: str
    ev_window_start: Optional[pd.Timestamp]
    ev_window_end: Optional[pd.Timestamp]
    ev_planning_mode: str
    ev_switch_state: Optional[str]
    debug: DebugSnapshot
    context: Optional[OptimizationContext] = None  # NEW: Store context for attribute access

    def plan_records(self, limit: Optional[int] = None, digits: int = 3) -> List[dict]:
        """Return quarter-hour plan rows as JSON-serialisable dictionaries."""

        df = self.plan
        # Default limit to 192 slots (48 hours) for HA sensor size management
        if limit is None:
            limit = 192
        if limit is not None:
            df = df.head(limit)

        records: List[dict] = []
        for row in df.itertuples(index=False):
            timestamp: pd.Timestamp = getattr(row, "timestamp")
            timestamp_local: Optional[pd.Timestamp] = getattr(row, "timestamp_local", None)

            def _clean(value: Optional[float]) -> Optional[float]:
                if value is None:
                    return None
                if isinstance(value, float) and math.isnan(value):
                    return None
                return round(float(value), digits)

            record = {
                "timestamp": timestamp.tz_convert(timezone.utc).isoformat() if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc).isoformat(),
                "timestamp_local": timestamp_local.isoformat() if isinstance(timestamp_local, pd.Timestamp) else None,
                "activity": getattr(row, "activity", None),
                # Grid flows
                "g_buy": _clean(getattr(row, "g_buy", None)),
                "g_sell": _clean(getattr(row, "g_sell", None)),
                "grid_to_batt": _clean(getattr(row, "grid_to_batt", None)),
                "grid_to_house": _clean(getattr(row, "grid_to_house", None)),
                "grid_to_ev": _clean(getattr(row, "grid_to_ev", None)),
                # PV flows (using pv_to_* names for HA)
                "pv_to_batt": _clean(getattr(row, "pv_to_batt", None)),
                "pv_to_house": _clean(getattr(row, "pv_to_house", None)),
                "pv_to_ev": _clean(getattr(row, "pv_to_ev", None)),
                # Battery flows
                "batt_to_house": _clean(getattr(row, "batt_to_house", None)),
                "batt_to_sell": _clean(getattr(row, "batt_to_sell", None)),
                "batt_to_ev": _clean(getattr(row, "batt_to_ev", None)),
                # State
                "battery_in": _clean(getattr(row, "battery_in", None)),
                "battery_out": _clean(getattr(row, "battery_out", None)),
                "ev_charge": _clean(getattr(row, "ev_charge", None)),
                "battery_soc": _clean(getattr(row, "battery_soc", None)),
                "battery_soc_pct": _clean(getattr(row, "battery_soc_pct", None)),
                "battery_soc_delta": _clean(getattr(row, "battery_soc_pct_delta", None)),
                "battery_reserve_target": _clean(getattr(row, "battery_reserve_target", None)),
                "battery_reserve_shortfall": _clean(getattr(row, "battery_reserve_shortfall", None)),
                "ev_soc_kwh": _clean(getattr(row, "ev_soc_kwh", getattr(row, "ev_soc", None))),
                "ev_soc_pct": _clean(getattr(row, "ev_soc_pct", None)),
                "ev_target_pct": _clean(getattr(row, "ev_target_pct", None)),
                # Prices
                "pv_forecast_kw": _clean(getattr(row, "pv_forecast_kw", None)),
                "consumption_estimate_kw": _clean(getattr(row, "consumption_estimate_kw", None)),
                "price_buy": _clean(getattr(row, "price_buy", None)),
                "price_sell": _clean(getattr(row, "price_sell", None)),
                "effective_sell_price": _clean(getattr(row, "effective_sell_price", None)),
                # Economics
                "grid_cost": _clean(getattr(row, "grid_cost", None)),
                "grid_revenue_effective": _clean(getattr(row, "grid_revenue_effective", None)),
                "grid_revenue": _clean(getattr(row, "grid_revenue", None)),
                "cash_cost_dkk": _clean(getattr(row, "cash_cost_dkk", None)),
                "ev_bonus": _clean(getattr(row, "ev_bonus", None)),
                "battery_cycle_cost": _clean(getattr(row, "battery_cycle_cost", None)),
                "battery_value_dkk": _clean(getattr(row, "battery_value_dkk", None)),
                "objective_component": _clean(getattr(row, "objective_component", None)),
                "battery_in_kw": _clean(getattr(row, "battery_in_kw", None)),
                "battery_out_kw": _clean(getattr(row, "battery_out_kw", None)),
                "battery_net_flow": _clean(getattr(row, "battery_net_flow", None)),
                "battery_net_flow_kw": _clean(getattr(row, "battery_net_flow_kw", None)),
                "battery_charge_from_pv": _clean(getattr(row, "battery_charge_from_pv", None)),
                "battery_charge_from_grid": _clean(getattr(row, "battery_charge_from_grid", None)),
                "battery_discharge_to_house": _clean(getattr(row, "battery_discharge_to_house", None)),
                "battery_discharge_to_ev": _clean(getattr(row, "battery_discharge_to_ev", None)),
                "battery_discharge_to_sell": _clean(getattr(row, "battery_discharge_to_sell", None)),
                "house_from_pv": _clean(getattr(row, "house_from_pv", None)),
                "house_from_battery": _clean(getattr(row, "house_from_battery", None)),
                "house_from_grid": _clean(getattr(row, "house_from_grid", None)),
                "house_supply_total_kw": _clean(getattr(row, "house_supply_total_kw", None)),
                "house_balance_error": _clean(getattr(row, "house_balance_error", None)),
                "house_from_pv_pct": _clean(getattr(row, "house_from_pv_pct", None)),
                "house_from_battery_pct": _clean(getattr(row, "house_from_battery_pct", None)),
                "house_from_grid_pct": _clean(getattr(row, "house_from_grid_pct", None)),
                "pv_used_total": _clean(getattr(row, "pv_used_total", None)),
                "pv_curtailed": _clean(getattr(row, "pv_curtailed", None)),
                "price_buy_percentile": _clean(getattr(row, "price_buy_percentile", None)),
                "battery_avoided_cost_dkk": _clean(getattr(row, "battery_avoided_cost_dkk", None)),
                "battery_sell_revenue_dkk": _clean(getattr(row, "battery_sell_revenue_dkk", None)),
                "battery_value_dkk": _clean(getattr(row, "battery_value_dkk", None)),
                "price_spread_dkk": _clean(getattr(row, "price_spread_dkk", None)),
                # Time classifications
                "cheap24": getattr(row, "cheap24", None),
                "expensive24": getattr(row, "expensive24", None),
                # Diagnostics (arbitrage)
                "arb_gate": getattr(row, "arb_gate", None),
                "arb_reason": getattr(row, "arb_reason", None),
                "arb_basis": getattr(row, "arb_basis", None),
                "arb_eta_rt": _clean(getattr(row, "arb_eta_rt", None)),
                "arb_c_cycle": _clean(getattr(row, "arb_c_cycle", None)),
                "price_buy_now": _clean(getattr(row, "price_buy_now", None)),
                "future_max_sell_eff": _clean(getattr(row, "future_max_sell_eff", None)),
                "arb_margin": _clean(getattr(row, "arb_margin", None)),
                # Diagnostics (policy/wait)
                "policy_wait_flag": getattr(row, "policy_wait_flag", None),
                "policy_wait_reason": getattr(row, "policy_wait_reason", None),
                "policy_price_basis": getattr(row, "policy_price_basis", None),
                "policy_price_now_dkk": _clean(getattr(row, "policy_price_now_dkk", None)),
                "policy_future_min_12h_dkk": _clean(getattr(row, "policy_future_min_12h_dkk", None)),
                "policy_grid_charge_allowed": getattr(row, "policy_grid_charge_allowed", None),
                "policy_hold_value_dkk": _clean(getattr(row, "policy_hold_value_dkk", None)),
                # Mode classification
                "recommended_mode": getattr(row, "recommended_mode", None),
            }
            records.append(record)
        return records

    def hourly_records(self, limit: Optional[int] = None, digits: int = 3) -> List[dict]:
        df = self.hourly
        if limit is not None:
            df = df.head(limit)
        records: List[dict] = []
        for row in df.itertuples(index=False):
            hour_ts: pd.Timestamp = getattr(row, "hour")

            def _clean(value: Optional[float]) -> Optional[float]:
                if value is None:
                    return None
                if isinstance(value, float) and math.isnan(value):
                    return None
                return round(float(value), digits)

            records.append(
                {
                    "hour": hour_ts.tz_convert(timezone.utc).isoformat() if hour_ts.tzinfo else hour_ts.replace(tzinfo=timezone.utc).isoformat(),
                    "g_buy": _clean(getattr(row, "g_buy", None)),
                    "g_sell": _clean(getattr(row, "g_sell", None)),
                    "battery_in": _clean(getattr(row, "battery_in", None)),
                    "battery_out": _clean(getattr(row, "battery_out", None)),
                    "ev_charge": _clean(getattr(row, "ev_charge", None)),
                    "pv_forecast_kw": _clean(getattr(row, "pv_forecast_kw", None)),
                    "consumption_estimate_kw": _clean(getattr(row, "consumption_estimate_kw", None)),
                    "objective_component": _clean(getattr(row, "objective_component", None)),
                    "grid_cost": _clean(getattr(row, "grid_cost", None)),
                    "grid_revenue": _clean(getattr(row, "grid_revenue", None)),
                    "battery_cycle_cost": _clean(getattr(row, "battery_cycle_cost", None)),
                    "ev_bonus": _clean(getattr(row, "ev_bonus", None)),
                    "battery_charge_from_pv": _clean(getattr(row, "battery_charge_from_pv", None)),
                    "battery_charge_from_grid": _clean(getattr(row, "battery_charge_from_grid", None)),
                    "battery_discharge_to_house": _clean(getattr(row, "battery_discharge_to_house", None)),
                    "battery_discharge_to_ev": _clean(getattr(row, "battery_discharge_to_ev", None)),
                    "battery_discharge_to_sell": _clean(getattr(row, "battery_discharge_to_sell", None)),
                    "battery_net_flow": _clean(getattr(row, "battery_net_flow", None)),
                    "battery_soc_start": _clean(getattr(row, "battery_soc_start", None)),
                    "battery_soc_end": _clean(getattr(row, "battery_soc_end", None)),
                    "battery_soc_min": _clean(getattr(row, "battery_soc_min", None)),
                    "battery_soc_max": _clean(getattr(row, "battery_soc_max", None)),
                    "battery_soc_min_pct": _clean(getattr(row, "battery_soc_min_pct", None)),
                    "battery_soc_max_pct": _clean(getattr(row, "battery_soc_max_pct", None)),
                }
            )
        return records

    def day_summary_records(self) -> List[dict]:
        return list(self.day_summary)

    def to_markdown(self, limit: Optional[int] = 24, digits: int = 2) -> str:
        if self.plan.empty:
            return "*Ingen plan er tilgængelig endnu.*"
        
        # Adjust limit based on resolution if it was the default
        if limit == 96:
            limit = 24 * SLOTS_PER_HOUR

        column_defs = [
            ("timestamp_local", "Tid"),
            ("activity", "Handling"),
            ("g_buy", "Net køb (kWh)"),
            ("g_sell", "Net salg (kWh)"),
            ("grid_to_batt", "Net→Batt (kWh)"),
            ("prod_to_batt", "PV→Batt (kWh)"),
            ("batt_to_house", "Batt→Hus (kWh)"),
            ("batt_to_sell", "Batt→Salg (kWh)"),
            ("price_buy", "Købspris"),
            ("price_sell", "Salgspris"),
            ("grid_cost", "Grid-køb"),
            ("grid_revenue_effective", "Grid-salg"),
            ("battery_cycle_cost", "Batterislid"),
            ("cash_cost_dkk", "Netto"),
        ]
        available_defs = [item for item in column_defs if item[0] in self.plan.columns]
        if not available_defs:
            return "*Ingen plan er tilgængelig endnu.*"

        data = self.plan[[col for col, _ in available_defs]]
        if limit is not None and limit > 0:
            data = data.head(limit)
        if data.empty:
            return "*Ingen plan er tilgængelig endnu.*"

        df = data.copy()
        if "timestamp_local" in df.columns:
            df["timestamp_local"] = df["timestamp_local"].dt.strftime("%Y-%m-%d %H:%M")

        numeric_cols = [col for col, _ in available_defs if col not in {"activity", "timestamp_local"}]
        for col in numeric_cols:
            df[col] = df[col].apply(lambda v: "-" if pd.isna(v) else f"{float(v):.{digits}f}")

        header = "| " + " | ".join(label for _, label in available_defs) + " |"
        sep = "| " + " | ".join(["---"] * len(available_defs)) + " |"
        rows = [
            "| " + " | ".join(str(df.iloc[i, j]) for j in range(len(available_defs))) + " |"
            for i in range(len(df))
        ]
        return "\n".join([header, sep] + rows)

    def to_json(self, limit: Optional[int] = None, digits: int = 3) -> str:
        return json.dumps(self.plan_records(limit=limit, digits=digits), ensure_ascii=False)

    def debug_inputs_summary(self) -> Dict[str, Union[float, int, str]]:
        return self.debug.inputs_summary()

    def debug_house_balance_records(self, limit: Optional[int] = None, digits: int = 3) -> List[dict]:
        return self.debug.house_balance_records(limit=limit, digits=digits)


def _prepare_plan_dataframe(
    forecast: pd.DataFrame,
    result: OptimizationResult,
    context: OptimizationContext,
    settings: Settings,
    pipeline: DataPipeline,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    List[str],
    Optional[pd.Timestamp],
    Optional[pd.Timestamp],
    str,
    Optional[str],
    DebugSnapshot,
]:
    plan = result.plan.copy()
    plan = plan.merge(
        forecast[["timestamp", "pv_forecast_kw", "consumption_estimate_kw", "price_buy", "price_sell"]],
        on="timestamp",
        how="left",
    )

    period_minutes = max(int(settings.resolution_minutes), 1)
    kwh_to_kw_factor = 60.0 / period_minutes
    period_hours = period_minutes / 60.0

    planning_override = getattr(context, "ev_planning_override", False)
    planning_disabled = getattr(context, "ev_planning_disabled", False)
    allowed_ev_bonus = (context.ev_status in EV_CHARGE_ALLOWED_STATUSES or planning_override) and not planning_disabled

    plan["battery_cycle_cost"] = BATTERY_CYCLE_COST_DKK_PER_KWH * (plan["battery_in"] + plan["battery_out"])
    plan["grid_cost"] = plan["price_buy"] * plan["g_buy"]
    effective_sell_price = plan.get("effective_sell_price", plan["price_sell"]).fillna(0.0)
    plan["grid_revenue_effective"] = effective_sell_price * plan["g_sell"]
    plan["grid_revenue"] = plan["price_sell"].fillna(0) * plan["g_sell"]
    plan["net_grid_cost_dkk"] = plan["grid_cost"] - plan["grid_revenue_effective"]

    hold_value_threshold = max(0.0, float(getattr(context, "battery_hold_value_dkk", 0.0) or 0.0))
    if "hold_value_penalty_dkk" in plan:
        plan["hold_value_penalty_dkk"] = plan["hold_value_penalty_dkk"].fillna(0.0)
    else:
        plan["hold_value_penalty_dkk"] = 0.0
    if "hold_value_penalty_per_kwh" in plan:
        plan["hold_value_penalty_per_kwh"] = plan["hold_value_penalty_per_kwh"].fillna(0.0)
    else:
        plan["hold_value_penalty_per_kwh"] = 0.0
    if hold_value_threshold > 0:
        plan["grid_charge_hold_penalty_dkk"] = ((plan["price_buy"] - hold_value_threshold).clip(lower=0.0)) * plan["grid_to_batt"]
    else:
        plan["grid_charge_hold_penalty_dkk"] = 0.0

    plan["ev_bonus"] = 0.0
    if allowed_ev_bonus:
        plan["ev_bonus"] = EV_CHARGE_BONUS_DKK_PER_KWH * plan["ev_charge"]
    plan["objective_component"] = (
        -plan["grid_cost"]
        + plan["grid_revenue_effective"]
        - plan["battery_cycle_cost"]
        + plan["ev_bonus"]
        - plan["grid_charge_hold_penalty_dkk"]
        - plan["hold_value_penalty_dkk"]
    )
    plan["cash_cost_dkk"] = plan["grid_cost"] - plan["grid_revenue"] + plan["battery_cycle_cost"]

    # Arbitrage margin for transparency if diagnostics are present
    if {
        "future_max_sell_eff",
        "arb_eta_rt",
        "price_buy_now",
        "arb_c_cycle",
    }.issubset(set(plan.columns)):
        denom = plan["arb_eta_rt"].replace(0, float("nan")).astype(float)
        plan["arb_margin"] = (
            plan["future_max_sell_eff"].astype(float) / denom
            - plan["price_buy_now"].astype(float)
            - plan["arb_c_cycle"].astype(float)
        )
    else:
        plan["arb_margin"] = float("nan")

    plan["house_from_pv"] = plan["prod_to_house"]
    plan["house_from_battery"] = plan["batt_to_house"]
    plan["house_from_grid"] = plan["grid_to_house"]
    plan["pv_used_total"] = plan["prod_to_house"] + plan["prod_to_batt"] + plan["prod_to_ev"]
    plan["pv_curtailed"] = plan["prod_curtailed"]

    house_supply_total = plan["house_from_pv"] + plan["house_from_battery"] + plan["house_from_grid"]
    plan["house_supply_total_kw"] = house_supply_total
    plan["house_balance_error"] = house_supply_total - plan["consumption_estimate_kw"]

    consumption_nonzero = plan["consumption_estimate_kw"].where(plan["consumption_estimate_kw"].abs() > 1e-6)
    plan["house_from_grid_pct"] = (plan["house_from_grid"] / consumption_nonzero).fillna(0.0)
    plan["house_from_battery_pct"] = (plan["house_from_battery"] / consumption_nonzero).fillna(0.0)
    plan["house_from_pv_pct"] = (plan["house_from_pv"] / consumption_nonzero).fillna(0.0)

    plan["battery_charge_from_pv"] = plan["prod_to_batt"]
    plan["battery_charge_from_grid"] = plan["grid_to_batt"]
    plan["battery_discharge_to_house"] = plan["batt_to_house"]
    plan["battery_discharge_to_ev"] = plan["batt_to_ev"]
    plan["battery_discharge_to_sell"] = plan["batt_to_sell"]

    plan["battery_net_flow"] = plan["battery_out"] - plan["battery_in"]
    plan["battery_in_kw"] = plan["battery_in"] * kwh_to_kw_factor
    plan["battery_out_kw"] = plan["battery_out"] * kwh_to_kw_factor
    plan["battery_net_flow_kw"] = plan["battery_net_flow"] * kwh_to_kw_factor

    plan["price_buy_percentile"] = plan["price_buy"].rank(method="average", pct=True)
    plan["battery_avoided_cost_dkk"] = plan["battery_discharge_to_house"] * plan["price_buy"]
    plan["battery_sell_revenue_dkk"] = plan["battery_discharge_to_sell"] * plan["price_sell"].fillna(0.0)
    plan["battery_value_dkk"] = (
        plan["battery_avoided_cost_dkk"]
        + plan["battery_sell_revenue_dkk"]
        - plan["battery_cycle_cost"]
    )
    plan["price_spread_dkk"] = plan["price_sell"].fillna(0.0) - plan["price_buy"]

    # Safety: Clip tiny negative numerical artefacts to zero for non-negative flows
    for col in [
        "ev_charge",
        "g_buy",
        "g_sell",
        "battery_in",
        "battery_out",
        "prod_to_house",
        "prod_to_batt",
        "prod_to_ev",
        "batt_to_house",
        "batt_to_ev",
        "batt_to_sell",
        "grid_to_house",
        "grid_to_ev",
        "grid_to_batt",
    ]:
        if col in plan.columns:
            plan[col] = plan[col].clip(lower=0.0)

    if BATTERY_CAPACITY_KWH > 0:
        plan["battery_soc_pct"] = plan["battery_soc"].astype(float) / BATTERY_CAPACITY_KWH * 100.0
    else:
        plan["battery_soc_pct"] = float("nan")

    plan["battery_soc_delta"] = plan["battery_soc"].diff()
    if not plan.empty:
        plan.loc[0, "battery_soc_delta"] = plan.loc[0, "battery_soc"] - context.battery_soc_kwh

    if BATTERY_CAPACITY_KWH > 0:
        initial_soc_pct = context.battery_soc_kwh / BATTERY_CAPACITY_KWH * 100.0
        plan["battery_soc_pct_delta"] = plan["battery_soc_pct"].diff()
        if not plan.empty:
            plan.loc[0, "battery_soc_pct_delta"] = plan.loc[0, "battery_soc_pct"] - initial_soc_pct
    else:
        plan["battery_soc_pct_delta"] = float("nan")

    if EV_BATTERY_CAPACITY_KWH > 0:
        plan["ev_soc_pct"] = plan["ev_soc"].astype(float) / EV_BATTERY_CAPACITY_KWH * 100.0
    else:
        plan["ev_soc_pct"] = float("nan")

    def classify_activity(row: pd.Series) -> str:
        """Classify the primary activity for this time slot.
        
        Shows combinations when multiple significant flows occur simultaneously.
        Threshold: 0.01 kWh (10 Wh) to filter out numerical noise.
        """
        threshold = 0.01
        activities = []
        
        # Priority 1: EV charging (most important to show)
        if row["ev_charge"] > threshold:
            if row["battery_discharge_to_ev"] > threshold:
                activities.append("EV(Batt)")
            elif row["grid_to_ev"] > threshold:
                activities.append("EV(Grid)")
            elif row["prod_to_ev"] > threshold:
                activities.append("EV(Sol)")
            else:
                activities.append("EV")
            return "+".join(activities)  # EV charging takes full priority
        
        # Priority 2: Battery arbitrage (selling from battery)
        if row["battery_out"] > threshold and row["g_sell"] > threshold:
            activities.append("Batt→Salg")
        
        # Priority 3: Grid charging battery (important to track)
        if row["battery_charge_from_grid"] > threshold:
            activities.append("Grid→Batt")
        
        # Priority 4: Solar charging battery
        if row["battery_charge_from_pv"] > threshold:
            activities.append("Sol→Batt")
        
        # Priority 5: Battery discharging to house
        if row["battery_out"] > threshold and row["battery_discharge_to_house"] > threshold:
            activities.append("Batt→Hus")
        
        # Priority 6: Grid supplying house
        if row["g_buy"] > threshold and row["house_from_grid"] > threshold:
            activities.append("Grid→Hus")
        
        # Priority 7: Solar to house (direct consumption)
        if row["house_from_pv"] > threshold:
            activities.append("Sol→Hus")
        
        # Return combined activities or "Normal" if nothing significant
        if activities:
            return "+".join(activities)
        return "Normal"

    def classify_mode(row: pd.Series) -> str:
        """Classify energy system mode for this time slot.
        
        Translates flows into user-friendly mode names matching input_select.energy_system_mode options.
        Combines modes when multiple activities occur simultaneously.
        """
        threshold = 0.01
        modes = []
        
        # Debug: Log first row to see what columns are available
        if not hasattr(classify_mode, '_logged'):
            print(f"[DEBUG] classify_mode columns: {list(row.index)}")
            print(f"[DEBUG] Sample values: ev_charge={row.get('ev_charge', 'MISSING')}, grid_to_batt={row.get('grid_to_batt', 'MISSING')}, battery_in={row.get('battery_in', 'MISSING')}")
            classify_mode._logged = True
        
        # Check EV charging
        if row.get("ev_charge", 0) > threshold:
            grid_to_ev = row.get("grid_to_ev", 0)
            pv_to_ev = row.get("pv_to_ev", 0) + row.get("prod_to_ev", 0)
            batt_to_ev = row.get("batt_to_ev", 0)
            if grid_to_ev > (pv_to_ev + batt_to_ev):
                modes.append("EV(Grid)")
            else:
                modes.append("EV(Solar)")
        
        # Check battery grid charging
        grid_to_batt = row.get("grid_to_batt", 0)
        if row.get("battery_in", 0) > threshold and grid_to_batt > threshold:
            modes.append("Batt(Grid)")
        
        # Check battery selling
        batt_to_sell = row.get("batt_to_sell", 0)
        if row.get("battery_out", 0) > threshold and batt_to_sell > threshold:
            modes.append("Batt→Salg")
        
        # Check sell priority (minimal battery charge)
        pv_to_house = row.get("pv_to_house", 0) + row.get("house_from_pv", 0)
        pv_to_batt = row.get("pv_to_batt", 0)
        if pv_to_house > threshold and pv_to_batt < threshold:
            modes.append("Sælg Overskud")
        
        # Combine or default
        if len(modes) >= 2:
            return " + ".join(modes)
        elif len(modes) == 1:
            return modes[0]
        else:
            return "Selvforbrug"

    plan["activity"] = plan.apply(classify_activity, axis=1)
    plan["recommended_mode"] = plan.apply(classify_mode, axis=1)

    tz = pytz.timezone(settings.timezone)
    timestamp_utc = pd.to_datetime(plan["timestamp"], utc=True)
    plan["timestamp"] = timestamp_utc
    plan["timestamp_local"] = timestamp_utc.dt.tz_convert(tz)
    plan["timestamp_local_naive"] = plan["timestamp_local"].dt.tz_localize(None)

    diagnostics = getattr(context, "slot_diagnostics", ())
    if diagnostics:
        diag_df = pd.DataFrame(list(diagnostics))
        diag_df = diag_df.reindex(range(len(plan)))
        diag_df = diag_df.reset_index(drop=True)
        plan = plan.reset_index(drop=True)
        plan = pd.concat([plan, diag_df], axis=1)
    else:
        plan = plan.reset_index(drop=True)

    balance_columns = [
        "timestamp_local",
        "consumption_estimate_kw",
        "house_from_pv",
        "house_from_battery",
        "house_from_grid",
        "house_from_pv_pct",
        "house_from_battery_pct",
        "house_from_grid_pct",
        "house_balance_error",
        "price_buy",
        "price_buy_percentile",
    ]
    balance_df = plan[balance_columns].copy()

    day_summary: List[dict] = []
    if not plan.empty:
        day_frame = plan.copy()
        day_frame["day"] = day_frame["timestamp_local"].dt.normalize()

        for day, group in day_frame.groupby("day"):
            period_start = group["timestamp_local"].min()
            period_end = group["timestamp_local"].max() + pd.Timedelta(minutes=settings.resolution_minutes)

            def _sum_flow(column: str) -> float:
                if column not in group.columns:
                    return 0.0
                return float(group[column].astype(float).sum())

            grid_to_house_kwh = _sum_flow("grid_to_house")
            grid_to_ev_kwh = _sum_flow("grid_to_ev")
            grid_to_batt_kwh = _sum_flow("grid_to_batt")
            grid_import_kwh = grid_to_house_kwh + grid_to_ev_kwh + grid_to_batt_kwh
            grid_export_kwh = _sum_flow("g_sell")

            total_consumption = float(group["consumption_estimate_kw"].sum() * period_hours)
            total_pv = float(group["pv_forecast_kw"].sum() * period_hours)
            total_batt_charge = float(group["battery_in"].sum())
            total_batt_discharge = float(group["battery_out"].sum())

            pv_to_house_kwh = _sum_flow("prod_to_house")
            pv_to_ev_kwh = _sum_flow("prod_to_ev")
            pv_to_batt_kwh = _sum_flow("prod_to_batt")
            pv_curtailed_kwh = _sum_flow("prod_curtailed")

            batt_to_house_kwh = _sum_flow("batt_to_house")
            batt_to_ev_kwh = _sum_flow("batt_to_ev")
            batt_to_sell_kwh = _sum_flow("batt_to_sell")

            if {"grid_to_house", "batt_to_house", "prod_to_house"}.issubset(set(group.columns)):
                total_house_actual = float(
                    (
                        group["grid_to_house"].astype(float)
                        + group["batt_to_house"].astype(float)
                        + group["prod_to_house"].astype(float)
                    ).sum()
                )
            else:
                total_house_actual = float("nan")

            total_ev_kwh = float(group["ev_charge"].sum()) if "ev_charge" in group.columns else 0.0
            total_grid_cost = float(group["grid_cost"].sum())
            total_grid_revenue = float(group["grid_revenue"].sum())
            total_cash_cost = float(group["cash_cost_dkk"].sum())
            total_objective = float(group["objective_component"].sum())
            total_battery_cycle_cost = float(group["battery_cycle_cost"].sum())

            try:
                batt_soc_start = float(group.iloc[0]["battery_soc"]) if "battery_soc" in group.columns else float("nan")
                batt_soc_end = float(group.iloc[-1]["battery_soc"]) if "battery_soc" in group.columns else float("nan")
            except Exception:
                batt_soc_start = float("nan")
                batt_soc_end = float("nan")

            total_buy_variable = _sum_flow("g_buy")
            weighted_buy = (group["price_buy"].astype(float) * group["g_buy"].astype(float)).sum()
            weighted_sell = (group["price_sell"].fillna(0.0).astype(float) * group["g_sell"].astype(float)).sum()

            avg_buy_price = float(weighted_buy / total_buy_variable) if total_buy_variable > 1e-6 else None
            avg_sell_price = float(weighted_sell / grid_export_kwh) if grid_export_kwh > 1e-6 else None

            total_house_actual_val = total_house_actual
            if isinstance(total_house_actual_val, float) and math.isnan(total_house_actual_val):
                total_house_actual_val = 0.0

            day_summary.append(
                {
                    "date": period_start.date().isoformat(),
                    "period_start": period_start.isoformat(),
                    "period_end": period_end.isoformat(),
                    "slots": int(group.shape[0]),
                    "grid_buy_kwh": float(grid_import_kwh),
                    "grid_sell_kwh": float(grid_export_kwh),
                    "grid_to_house_kwh": float(grid_to_house_kwh),
                    "grid_to_ev_kwh": float(grid_to_ev_kwh),
                    "grid_to_batt_kwh": float(grid_to_batt_kwh),
                    "consumption_kwh": float(total_consumption),
                    "house_actual_kwh": float(total_house_actual),
                    "ev_charge_kwh": float(total_ev_kwh),
                    "total_house_ev_kwh": float(total_house_actual_val + total_ev_kwh),
                    "pv_generation_kwh": float(total_pv),
                    "pv_to_house_kwh": float(pv_to_house_kwh),
                    "pv_to_ev_kwh": float(pv_to_ev_kwh),
                    "pv_to_batt_kwh": float(pv_to_batt_kwh),
                    "pv_curtailed_kwh": float(pv_curtailed_kwh),
                    "battery_charge_kwh": float(total_batt_charge),
                    "battery_discharge_kwh": float(total_batt_discharge),
                    "batt_to_house_kwh": float(batt_to_house_kwh),
                    "batt_to_ev_kwh": float(batt_to_ev_kwh),
                    "batt_to_sell_kwh": float(batt_to_sell_kwh),
                    "battery_soc_start_kwh": batt_soc_start,
                    "battery_soc_end_kwh": batt_soc_end,
                    "grid_cost_dkk": float(total_grid_cost),
                    "grid_revenue_dkk": float(total_grid_revenue),
                    "net_grid_cost_dkk": float(total_grid_cost - total_grid_revenue),
                    "cash_cost_dkk": float(total_cash_cost),
                    "battery_cycle_cost_dkk": float(total_battery_cycle_cost),
                    "objective_component_dkk": float(total_objective),
                    "avg_price_buy": avg_buy_price,
                    "avg_price_sell": avg_sell_price,
                }
            )

    hourly_source = plan.copy()
    hourly_source["hour"] = timestamp_utc.dt.floor("h")
    sum_columns = [
        "g_buy",
        "g_sell",
        "battery_in",
        "battery_out",
        "ev_charge",
        "pv_forecast_kw",
        "consumption_estimate_kw",
        "objective_component",
        "grid_cost",
        "grid_revenue",
        "battery_cycle_cost",
        "ev_bonus",
        "prod_to_house",
        "prod_to_batt",
        "prod_to_ev",
        "prod_curtailed",
        "batt_to_house",
        "batt_to_ev",
        "batt_to_sell",
        "grid_to_house",
        "grid_to_ev",
        "grid_to_batt",
        "battery_charge_from_pv",
        "battery_charge_from_grid",
        "battery_discharge_to_house",
        "battery_discharge_to_ev",
        "battery_discharge_to_sell",
        "battery_net_flow",
    ]
    hourly = hourly_source.groupby("hour", as_index=False)[sum_columns].sum()

    soc_group = hourly_source.groupby("hour")["battery_soc"].agg(["min", "max", "first", "last"]).reset_index()
    soc_group = soc_group.rename(
        columns={
            "min": "battery_soc_min",
            "max": "battery_soc_max",
            "first": "battery_soc_start",
            "last": "battery_soc_end",
        }
    )
    if BATTERY_CAPACITY_KWH > 0:
        soc_group["battery_soc_min_pct"] = soc_group["battery_soc_min"] / BATTERY_CAPACITY_KWH * 100.0
        soc_group["battery_soc_max_pct"] = soc_group["battery_soc_max"] / BATTERY_CAPACITY_KWH * 100.0
    else:
        soc_group["battery_soc_min_pct"] = float("nan")
        soc_group["battery_soc_max_pct"] = float("nan")
    hourly = hourly.merge(soc_group, on="hour", how="left")

    notes: List[str] = []
    if pipeline.last_consumption_note:
        notes.append(pipeline.last_consumption_note)

    if not plan.empty:
        min_soc = float(plan["battery_soc"].min())
        max_soc = float(plan["battery_soc"].max())
        if min_soc <= BATTERY_MIN_SOC_KWH + 0.25:
            notes.append(
                f"Batteriet rammer nær minimum (≈{min_soc:.1f} kWh). Overvej at hæve sikkerhedsbufferen eller reducere salg."
            )
        if max_soc >= BATTERY_CAPACITY_KWH - 0.25:
            notes.append(
                f"Batteriet fyldes næsten helt (≈{max_soc:.1f} kWh). Tjek om ekstra produktion skal sælges tidligere."
            )

    planning_mode = "disabled" if planning_disabled else "forced" if planning_override else "auto"
    if planning_disabled:
        notes.append("EV charging disabled by planning switch; no energy scheduled for the vehicle.")
    elif planning_override and context.ev_status not in EV_CHARGE_ALLOWED_STATUSES:
        notes.append(f"EV planning forced despite charger status '{context.ev_status}'. Ensure cable is connected.")

    ev_window_start = None
    if len(plan) and 0 <= context.ev_window_start_index < len(plan):
        ev_window_start = plan.loc[context.ev_window_start_index, "timestamp"]

    ev_window_end = None
    if len(plan):
        end_candidate = max(0, context.ev_window_end_index - 1)
        end_idx = min(end_candidate, len(plan) - 1)
        ev_window_end = plan.loc[end_idx, "timestamp"]

    price_buy_series = forecast["price_buy"].astype(float)
    price_sell_series = forecast["price_sell"].fillna(0.0).astype(float)
    consumption_series = forecast["consumption_estimate_kw"].astype(float)
    pv_series = forecast["pv_forecast_kw"].astype(float)

    debug_inputs: Dict[str, Union[float, int, str]] = {
        "periods": len(plan),
        "resolution_minutes": settings.resolution_minutes,
        "lookahead_hours": settings.lookahead_hours,
        "battery_capacity_kwh": BATTERY_CAPACITY_KWH,
        "battery_soc_kwh_initial": context.battery_soc_kwh,
        "battery_soc_pct_initial": (context.battery_soc_kwh / BATTERY_CAPACITY_KWH * 100) if BATTERY_CAPACITY_KWH else float("nan"),
        "ev_soc_kwh_initial": context.ev_soc_kwh,
        "ev_soc_pct_initial": (context.ev_soc_kwh / EV_BATTERY_CAPACITY_KWH * 100) if EV_BATTERY_CAPACITY_KWH else float("nan"),
        "ev_status": context.ev_status,
        "ev_required_kwh": context.ev_required_kwh or 0.0,
        "ev_window_start_index": context.ev_window_start_index,
        "ev_window_end_index": context.ev_window_end_index,
        "forecast_pv_total_kwh": float(pv_series.sum() * period_hours),
        "forecast_consumption_total_kwh": float(consumption_series.sum() * period_hours),
        "price_buy_min": float(price_buy_series.min()) if not price_buy_series.empty else float("nan"),
        "price_buy_max": float(price_buy_series.max()) if not price_buy_series.empty else float("nan"),
        "price_buy_mean": float(price_buy_series.mean()) if not price_buy_series.empty else float("nan"),
        "price_sell_mean": float(price_sell_series.mean()) if not price_sell_series.empty else float("nan"),
        "price_buy_now": float(price_buy_series.iloc[0]) if len(price_buy_series) else float("nan"),
        "price_sell_now": float(price_sell_series.iloc[0]) if len(price_sell_series) else float("nan"),
        "consumption_source": "home_assistant" if pipeline.last_consumption_note is None else "fallback_profile",
    }
    if pipeline.last_consumption_note:
        debug_inputs["consumption_note"] = pipeline.last_consumption_note

    debug_snapshot = DebugSnapshot(inputs=debug_inputs, house_balance=balance_df)

    return (
        plan,
        hourly,
        day_summary,
        notes,
        ev_window_start,
        ev_window_end,
        planning_mode,
        getattr(context, "ev_planning_switch_state", None),
        debug_snapshot,
    )


def build_plan_report(
    forecast: pd.DataFrame,
    result: OptimizationResult,
    context: OptimizationContext,
    settings: Settings,
    pipeline: DataPipeline,
    summary: dict,
) -> PlanReport:
    (
        plan,
        hourly,
        day_summary,
        notes,
        ev_window_start,
        ev_window_end,
        planning_mode,
        switch_state,
        debug_snapshot,
    ) = _prepare_plan_dataframe(
        forecast,
        result,
        context,
        settings,
        pipeline,
    )

    # Policy notes and acceptance checks for load-shaving and peak gating
    try:
        # Append explanatory policy notes
        precharge_need = float(getattr(context, "precharge_need_kwh", 0.0) or 0.0)
        if precharge_need > 0:
            summary.setdefault("policy", {}).setdefault("notes", []).append(
                f"Load-shave: aktiveret (cheap24→batt), forventet {precharge_need:.2f} kWh."
            )
        summary.setdefault("policy", {}).setdefault("notes", []).append(
            "Peak-køb: blokeret; undtagelser kun ved min SoC + EV-urgent."
        )
        # EV buffer recommendation block inside window (cheapest slots)
        if not plan.empty:
            buy_prices = plan["price_buy"].astype(float)
            # Pick EV window mask
            ev_mask = pd.Series(False, index=plan.index)
            if context.ev_window_end_index > context.ev_window_start_index:
                ev_mask.iloc[context.ev_window_start_index:context.ev_window_end_index] = True
            # Need estimate
            ev_need = float(context.ev_required_kwh or 0.0)
            n_slots = 0
            slots_per_hour = 60.0 / max(settings.resolution_minutes, 1)
            max_ev_charge_slot = MAX_EV_CHARGE_KWH / slots_per_hour
            if ev_need > 0 and max_ev_charge_slot > 0:
                import math as _math
                n_slots = max(6, int(_math.ceil(ev_need / max_ev_charge_slot)))
            ev_candidates = plan[ev_mask][["timestamp", "timestamp_local", "price_buy"]].copy()
            ev_candidates = ev_candidates.sort_values("price_buy", ascending=True)
            chosen = ev_candidates.head(n_slots) if n_slots > 0 else ev_candidates.head(0)
            if not chosen.empty:
                start_local = pd.to_datetime(chosen["timestamp_local"].min())
                end_local = pd.to_datetime(chosen["timestamp_local"].max()) + pd.Timedelta(minutes=settings.resolution_minutes)
                avg_price = float(chosen["price_buy"].mean()) if not chosen["price_buy"].empty else float("nan")
                summary.setdefault("policy", {}).setdefault("notes", []).append(
                    f"EV-buffer: {ev_need:.2f} kWh i billigste blokke {start_local.isoformat()}–{end_local.isoformat()}, ~{avg_price:.2f}."
                )
        # Weekly EV input note if present in summary
        weekly_kwh = summary.get("policy", {}).get("expected_ev_daily_kwh")
        if weekly_kwh is not None:
            summary.setdefault("policy", {}).setdefault("notes", []).append(
                f"Ugentlig EV-input: {float(weekly_kwh):.2f} kWh/dag."
            )
        else:
            summary.setdefault("policy", {}).setdefault("notes", []).append("Ugentlig EV-input: IKKE FUNDET.")

        # Acceptance checks A1–A3
        status_override: Optional[str] = None
        violations: list[str] = []
        # A1: Peak-buy hits in expensive24
        if {"expensive24", "g_buy"}.issubset(set(plan.columns)):
            peak_mask = (plan["expensive24"] == True) & (plan["g_buy"] > 0.05)  # noqa: E712
            peak_hits = int(peak_mask.sum())
            if peak_hits > 0:
                # Check SoC at min and EV urgent within 2h
                bad_rows = []
                for i in plan[peak_mask].index:
                    soc_ok = bool(plan.loc[i, "battery_soc"] <= BATTERY_MIN_SOC_KWH + 0.01) if "battery_soc" in plan.columns else False
                    # EV urgent if within last 8 slots of window
                    urgent = False
                    try:
                        urgent = i >= (context.ev_window_end_index - 8) and (context.ev_required_kwh or 0.0) > 0
                    except Exception:
                        urgent = False
                    if not (soc_ok and urgent):
                        bad_rows.append(i)
                if len(bad_rows) > 2:
                        violations.append(f"A1: For mange peak-køb ({len(bad_rows)} > 2) eller uden min-SoC/EV-urgent.")
        # A2: Cheap precharge at least 50% of need (TEMPORARILY DISABLED for debugging)
        # if precharge_need > 0 and {"cheap24", "grid_to_batt"}.issubset(set(plan.columns)):
        #     cheap_precharge = float(plan.loc[plan["cheap24"] == True, "grid_to_batt"].sum())  # noqa: E712
        #     if cheap_precharge < 0.5 * precharge_need - 1e-6:
        #         violations.append(f"A2: Precharge i cheap24 for lav ({cheap_precharge:.2f} < 50% af {precharge_need:.2f}).")
        # A3: No reserve charging in expensive24 (approximate: no grid->batt in expensive slots)
        if {"expensive24", "grid_to_batt"}.issubset(set(plan.columns)):
            if float(plan.loc[plan["expensive24"] == True, "grid_to_batt"].sum()) > 1e-3:  # noqa: E712
                violations.append("A3: Reserve-opladning til batteri i dyre slots er forbudt (grid→batt>0).")

        if violations:
            status_override = "Suboptimal (policy violation)"
            summary.setdefault("policy", {}).setdefault("notes", []).extend(violations)
    except Exception:
        # Do not break report on policy checks
        pass

    # Enrich summary with daily reconciliation from cumulative sensors when available
    try:
        last_daily = getattr(pipeline, "last_daily_reconciliation", None)
        if last_daily:
            summary["daily_reconciliation"] = list(last_daily)  # copy
            # Convenience shortcuts: latest day and previous day
            dates = []
            for r in last_daily:
                d = r.get("date")
                if d is not None:
                    try:
                        dates.append(pd.to_datetime(d))
                    except Exception:
                        continue
            if dates:
                last_date = max(dates)
                prev = [d for d in dates if d < last_date]
                prev_date = max(prev) if prev else None
                def _find(target):
                    if target is None:
                        return None
                    for r in last_daily:
                        try:
                            if pd.to_datetime(r.get("date")) == target:
                                return r
                        except Exception:
                            continue
                    return None
                today_rec = _find(last_date)
                yday_rec = _find(prev_date)
                if today_rec is not None:
                    summary["reconciliation_today"] = today_rec
                    try:
                        summary["ev_done_today_kwh"] = float(today_rec.get("ev_kwh", 0.0))
                    except Exception:
                        pass
                if yday_rec is not None:
                    summary["reconciliation_yesterday"] = yday_rec
    except Exception:
        # Non-fatal: leave summary as-is if anything goes wrong
        pass

    # Surface consumption calibration note if the forecast included it
    try:
        if "consumption_calibration_note" in forecast.columns:
            notes = forecast["consumption_calibration_note"].dropna().unique().tolist()
            if notes:
                summary["consumption_calibration_note"] = str(notes[0])
    except Exception:
        pass

    return PlanReport(
        status=status_override or result.status,
        objective_value=float(result.objective_value),
        plan=plan,
        hourly=hourly,
        summary=summary,
        day_summary=day_summary,
        notes=list(notes),
        generated_at=datetime.now(timezone.utc),
        updated_at=None,  # Only set when reading from DB
        timezone=settings.timezone,
        ev_window_start=ev_window_start,
        ev_window_end=ev_window_end,
        ev_planning_mode=planning_mode,
        ev_switch_state=switch_state,
        context=context,
        debug=debug_snapshot,
    )


def read_plan_from_db(now: Optional[datetime] = None) -> PlanReport:
    """Read the latest saved plan from MariaDB without running optimization.
    
    This is much faster than compute_plan_report() and ensures HA displays
    the same plan that was generated by run_planner.py.
    
    Returns:
        PlanReport with the saved plan data, or raises if no plan exists.
    """
    from sqlalchemy import create_engine, text
    
    settings = load_settings()
    now = now or datetime.now(timezone.utc)
    tz = pytz.timezone(settings.timezone)
    
    engine = create_engine(settings.mariadb_dsn, future=True, pool_pre_ping=True)
    
    # Read all slots from the latest optimizer run (based on created_at)
    # This ensures we get the complete plan from the most recent run,
    # including past slots that were just written
    query = text("""
        SELECT 
            timestamp_utc,
            local_time,
            price_buy,
            price_sell,
            COALESCE(g_buy, 0) as g_buy,
            COALESCE(g_sell, 0) as g_sell,
            COALESCE(battery_in, 0) as battery_in,
            COALESCE(battery_out, 0) as battery_out,
            COALESCE(grid_to_house, 0) as grid_to_house,
            COALESCE(grid_to_batt, 0) as grid_to_batt,
            COALESCE(grid_to_ev, 0) as grid_to_ev,
            COALESCE(batt_to_house, 0) as batt_to_house,
            COALESCE(batt_to_ev, 0) as batt_to_ev,
            COALESCE(batt_to_sell, 0) as batt_to_sell,
            COALESCE(ev_charge, 0) as ev_charge,
            COALESCE(pv_to_house, 0) as pv_to_house,
            COALESCE(pv_to_batt, 0) as pv_to_batt,
            COALESCE(pv_to_ev, 0) as pv_to_ev,
            COALESCE(house_from_grid, 0) as house_from_grid,
            COALESCE(house_from_battery, 0) as house_from_battery,
            COALESCE(house_from_pv, 0) as house_from_pv,
            COALESCE(house_load, 0) as house_load,
            recommended_mode,
            COALESCE(soc_kwh, 0) as soc_kwh,
            COALESCE(battery_soc, soc_kwh, 0) as battery_soc,
            COALESCE(battery_soc_pct, 0) as battery_soc_pct,
            COALESCE(ev_soc_kwh, 0) as ev_soc_kwh,
            COALESCE(ev_soc_pct, 0) as ev_soc_pct,
            COALESCE(battery_reserve_target, 0) as battery_reserve_target,
            COALESCE(battery_reserve_shortfall, 0) as battery_reserve_shortfall,
            COALESCE(effective_sell_price, 0) as effective_sell_price,
            COALESCE(grid_cost, 0) as grid_cost,
            COALESCE(grid_revenue_effective, 0) as grid_revenue_effective,
            COALESCE(grid_revenue, 0) as grid_revenue,
            COALESCE(ev_bonus, 0) as ev_bonus,
            COALESCE(battery_cycle_cost, 0) as battery_cycle_cost,
            COALESCE(battery_value_dkk, 0) as battery_value_dkk,
            COALESCE(cash_cost_dkk, 0) as cash_cost_dkk,
            COALESCE(objective_component, 0) as objective_component,
            COALESCE(pv_forecast_kw, 0) as pv_forecast_kw,
            COALESCE(consumption_estimate_kw, 0) as consumption_estimate_kw,
            COALESCE(cheap24, FALSE) as cheap24,
            COALESCE(expensive24, FALSE) as expensive24,
            arb_gate,
            arb_reason,
            arb_basis,
            COALESCE(arb_eta_rt, 0) as arb_eta_rt,
            COALESCE(arb_c_cycle, 0) as arb_c_cycle,
            COALESCE(price_buy_now, 0) as price_buy_now,
            COALESCE(future_max_sell_eff, 0) as future_max_sell_eff,
            COALESCE(arb_margin, 0) as arb_margin,
            policy_wait_flag,
            policy_wait_reason,
            policy_price_basis,
            COALESCE(policy_price_now_dkk, 0) as policy_price_now_dkk,
            COALESCE(policy_future_min_12h_dkk, 0) as policy_future_min_12h_dkk,
            policy_grid_charge_allowed,
            COALESCE(policy_hold_value_dkk, 0) as policy_hold_value_dkk,
            note,
            created_at
        FROM energy_planner.energy_plan_slots
        WHERE created_at >= (
            SELECT MAX(created_at) - INTERVAL 1 MINUTE
            FROM energy_planner.energy_plan_slots
        )
        ORDER BY timestamp_utc
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query)
        rows = result.fetchall()
    
    if not rows:
        # Fallback: try to get last 288 rows (72 hours at 15-min resolution)
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 
                    timestamp_utc,
                    local_time,
                    price_buy,
                    price_sell,
                    grid_to_batt,
                    batt_to_house,
                    batt_to_ev,
                    grid_to_ev,
                    ev_charge,
                    pv_to_batt,
                    pv_to_house,
                    house_load,
                    recommended_mode,
                    soc_kwh,
                    note,
                    created_at
                FROM energy_planner.energy_plan_slots
                ORDER BY timestamp_utc DESC
                LIMIT 288
            """))
            rows = result.fetchall()
            # Reverse to get chronological order
            rows = list(reversed(rows))
    
    if not rows:
        raise ValueError("No plan data found in database. Run 'python scripts/run_planner.py' first.")
    
    # Extract created_at timestamp before converting to DataFrame
    # Note: created_at in DB is stored as naive local time, not UTC
    timestamp_from_db = None
    if hasattr(rows[0], 'created_at') and rows[0].created_at:
        # Treat as local time and convert to UTC
        naive_ts = pd.to_datetime(rows[0].created_at)
        local_ts = tz.localize(naive_ts)
        timestamp_from_db = local_ts.astimezone(pytz.UTC)
    elif hasattr(rows[0], '_mapping') and 'created_at' in rows[0]._mapping:
        naive_ts = pd.to_datetime(rows[0]._mapping['created_at'])
        local_ts = tz.localize(naive_ts)
        timestamp_from_db = local_ts.astimezone(pytz.UTC)
    else:
        timestamp_from_db = now  # Fallback to current time
    
    # Convert to DataFrame matching the expected plan structure
    plan_df = pd.DataFrame([dict(row._mapping) for row in rows])
    
    # Ensure timestamp is tz-aware UTC
    plan_df["timestamp"] = pd.to_datetime(plan_df["timestamp_utc"], utc=True)
    plan_df["timestamp_local"] = pd.to_datetime(plan_df["local_time"]).dt.tz_localize(tz)
    
    # Map database columns to expected plan columns
    plan_df["activity"] = plan_df["note"].fillna("Normal")
    # plan_df["battery_soc"] = plan_df["soc_kwh"]  # REMOVED: Do not overwrite battery_soc with legacy soc_kwh
    plan_df["ev_soc"] = plan_df["ev_soc_kwh"]
    
    # Alias PV flows (prod_* is internal optimizer naming, pv_* is DB naming)
    plan_df["prod_to_house"] = plan_df["pv_to_house"]
    plan_df["prod_to_batt"] = plan_df["pv_to_batt"]
    plan_df["prod_to_ev"] = plan_df["pv_to_ev"]
    
    # If g_buy/g_sell are zero (old data), recalculate from flows
    if "g_buy" not in plan_df.columns or (plan_df["g_buy"] == 0).all():
        plan_df["g_buy"] = (
            plan_df["grid_to_house"].fillna(0)
            + plan_df["grid_to_batt"].fillna(0)
            + plan_df["grid_to_ev"].fillna(0)
        )
    
    # If battery_in/out are zero (old data), recalculate from flows
    if "battery_in" not in plan_df.columns or (plan_df["battery_in"] == 0).all():
        plan_df["battery_in"] = plan_df["grid_to_batt"].fillna(0) + plan_df["pv_to_batt"].fillna(0)
    if "battery_out" not in plan_df.columns or (plan_df["battery_out"] == 0).all():
        plan_df["battery_out"] = (
            plan_df["batt_to_house"].fillna(0)
            + plan_df["batt_to_ev"].fillna(0)
            + plan_df["batt_to_sell"].fillna(0)
        )
    
    # Calculate battery SoC percentage
    plan_df["battery_soc_pct"] = (plan_df["battery_soc"] / BATTERY_CAPACITY_KWH * 100).clip(0, 100)
    
    # Add ev columns if missing
    if "ev_soc_pct" not in plan_df.columns:
        plan_df["ev_soc_pct"] = 0.0
    if "ev_target_pct" not in plan_df.columns:
        plan_df["ev_target_pct"] = None
    
    # Add missing flow columns with defaults if not present
    for col in ["grid_to_house", "grid_to_batt", "grid_to_ev", "pv_to_house", "pv_to_batt", "pv_to_ev",
                "batt_to_house", "batt_to_sell", "batt_to_ev"]:
        if col not in plan_df.columns:
            plan_df[col] = 0.0
    
    # Add missing state/economics columns
    for col in ["battery_reserve_target", "battery_reserve_shortfall", "effective_sell_price",
                "grid_revenue", "cash_cost_dkk", "objective_component"]:
        if col not in plan_df.columns:
            plan_df[col] = 0.0
    
    # Add missing forecast input columns
    for col in ["consumption_estimate_kw", "pv_forecast_kw"]:
        if col not in plan_df.columns:
            plan_df[col] = 0.0
    
    # Add time classification columns
    for col in ["cheap24", "expensive24"]:
        if col not in plan_df.columns:
            plan_df[col] = False
    
    # Add diagnostic columns (arbitrage)
    for col in ["arb_gate", "arb_reason", "arb_basis"]:
        if col not in plan_df.columns:
            plan_df[col] = None
    for col in ["arb_eta_rt", "arb_c_cycle", "price_buy_now", "future_max_sell_eff", "arb_margin"]:
        if col not in plan_df.columns:
            plan_df[col] = 0.0
    
    # Add diagnostic columns (policy)
    for col in ["policy_wait_flag", "policy_wait_reason", "policy_price_basis", "policy_grid_charge_allowed"]:
        if col not in plan_df.columns:
            plan_df[col] = None
    for col in ["policy_price_now_dkk", "policy_future_min_12h_dkk", "policy_hold_value_dkk"]:
        if col not in plan_df.columns:
            plan_df[col] = 0.0
    
    # Add house supply breakdown columns (for dashboard compatibility)
    plan_df["house_from_pv"] = plan_df["pv_to_house"].fillna(0)
    plan_df["house_from_battery"] = plan_df["batt_to_house"].fillna(0)
    plan_df["house_from_grid"] = plan_df["grid_to_house"].fillna(0)
    
    # Calculate costs (simplified since we don't have all optimizer metadata)
    plan_df["grid_cost"] = plan_df["g_buy"] * plan_df["price_buy"]
    plan_df["grid_revenue_effective"] = plan_df["g_sell"] * plan_df["price_sell"]
    plan_df["battery_cycle_cost"] = (plan_df["battery_in"] + plan_df["battery_out"]) * BATTERY_CYCLE_COST_DKK_PER_KWH
    plan_df["ev_bonus"] = plan_df["ev_charge"] * EV_CHARGE_BONUS_DKK_PER_KWH
    
    # timestamp_from_db was extracted earlier before DataFrame conversion
    
    # Build a minimal summary (we don't have all the optimizer context)
    total_grid_import = plan_df["g_buy"].sum()
    total_grid_export = plan_df["g_sell"].sum()
    
    # House consumption: use house_load if available, otherwise estimate from flows
    has_house_load = (plan_df["house_load"] > 0.01).any()
    if has_house_load:
        total_house = plan_df["house_load"].sum()
    else:
        # Estimate house consumption from: grid_to_house + batt_to_house + pv_to_house
        total_house = (
            plan_df["grid_to_house"].sum() 
            + plan_df["batt_to_house"].sum() 
            + plan_df["pv_to_house"].sum()
        )
    
    total_ev = plan_df["ev_charge"].sum()
    total_pv = (plan_df["pv_to_house"].fillna(0) + plan_df["pv_to_batt"].fillna(0)).sum()
    total_batt_in = plan_df["battery_in"].sum()
    total_batt_out = plan_df["battery_out"].sum()
    
    total_grid_cost = plan_df["grid_cost"].sum()
    total_grid_revenue = plan_df["grid_revenue_effective"].sum()
    total_batt_cycle_cost = plan_df["battery_cycle_cost"].sum()
    total_ev_bonus = plan_df["ev_bonus"].sum()
    
    net_dkk = total_grid_cost - total_grid_revenue + total_batt_cycle_cost - total_ev_bonus
    
    # Build summary dict matching PlanReport structure
    summary = {
        "energy_totals": {
            "grid_import_kwh": float(total_grid_import),
            "grid_export_kwh": float(total_grid_export),
            "house_consumption_kwh": float(total_house),
            "ev_charge_kwh": float(total_ev),
            "pv_generation_kwh": float(total_pv),
            "battery_charge_kwh": float(total_batt_in),
            "battery_discharge_kwh": float(total_batt_out),
        },
        "battery": {
            "start_soc_kwh": float(plan_df.iloc[0]["battery_soc"]),
            "end_soc_kwh": float(plan_df.iloc[-1]["battery_soc"]),
            "start_soc_pct": float(plan_df.iloc[0]["battery_soc_pct"]),
            "end_soc_pct": float(plan_df.iloc[-1]["battery_soc_pct"]),
            "min_soc_kwh": float(plan_df["battery_soc"].min()),
            "max_soc_kwh": float(plan_df["battery_soc"].max()),
            "min_soc_pct": float(plan_df["battery_soc_pct"].min()),
            "max_soc_pct": float(plan_df["battery_soc_pct"].max()),
            "total_charge_kwh": float(total_batt_in),
            "total_discharge_kwh": float(total_batt_out),
        },
        "economics": {
            "grid_cost_dkk": float(total_grid_cost),
            "grid_revenue_eff_dkk": float(total_grid_revenue),
            "battery_cycle_cost_dkk": float(total_batt_cycle_cost),
            "ev_bonus_dkk": float(total_ev_bonus),
            "net_dkk": float(net_dkk),
        },
        "policy": {
            "notes": ["Plan loaded from database (not freshly optimized)"],
        },
    }
    
    # Build report with proper structure matching PlanReport dataclass
    report = PlanReport(
        status="Optimal",
        objective_value=float(-net_dkk),
        plan=plan_df,
        hourly=pd.DataFrame(),  # Empty for now (could aggregate by hour if needed)
        summary=summary,
        day_summary=[],  # Empty for now
        notes=["Plan loaded from database (not freshly optimized)"],
        generated_at=timestamp_from_db,
        updated_at=timestamp_from_db,  # Use same timestamp since OLD table doesn't track updates
        timezone=settings.timezone,
        ev_window_start=None,  # Not stored in OLD table
        ev_window_end=None,  # Not stored in OLD table
        ev_planning_mode="unknown",  # Not stored in OLD table
        ev_switch_state=None,  # Not stored in OLD table
        context=None,
        debug=DebugSnapshot(
            inputs={
                "grid_import_kwh": float(total_grid_import),
                "grid_export_kwh": float(total_grid_export),
                "house_consumption_kwh": float(total_house),
                "ev_energy_kwh": float(total_ev),
            },
            house_balance=pd.DataFrame(),  # Empty for now
        ),
    )
    
    return report


def compute_plan_report(now: Optional[datetime] = None) -> PlanReport:
    """Run the optimisation pipeline once and prepare a structured report."""

    settings = load_settings()
    ha = HomeAssistantClient(settings.ha_base_url, settings.ha_api_key)
    SessionFactory = create_session_factory(settings.mariadb_dsn)

    pipeline = DataPipeline(settings, ha, SessionFactory)
    now = now or datetime.now(timezone.utc)
    forecast = pipeline.build_forecast_dataframe(now)
    pipeline.persist_forecast(forecast)

    # Derive EV energy already charged today (local day) from reconciliation, if available
    ev_done_today_kwh: Optional[float] = None
    try:
        last_daily = getattr(pipeline, "last_daily_reconciliation", None) or []
        if last_daily:
            today_local = now.astimezone(pytz.timezone(settings.timezone)).date()
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
        forecast,
        settings,
        ha,
        SessionFactory=SessionFactory,
        now=now,
        ev_done_today_kwh=ev_done_today_kwh,
    )
    result = solve_quarter_hour(forecast, context)
    summary = summarize_plan(result.plan, forecast, context, settings, policy)

    return build_plan_report(
        forecast=forecast,
        result=result,
        context=context,
        settings=settings,
        pipeline=pipeline,
        summary=summary,
    )


__all__ = ["PlanReport", "DebugSnapshot", "compute_plan_report", "read_plan_from_db", "build_plan_report"]
