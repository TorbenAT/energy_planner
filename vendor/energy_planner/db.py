"""Database session management for MariaDB using SQLAlchemy.

Also provides a utility to persist the quarter-hour plan into a dedicated
slots table for HA dashboards and external analytics.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, text

from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd  # type: ignore
import pytz
from sqlalchemy.orm import DeclarativeBase, scoped_session, sessionmaker

# Import central schema for DB column mapping
try:
    from .plan_schema import PLAN_FIELDS_DB, get_column_mapping
    _SCHEMA_AVAILABLE = True
except ImportError:
    PLAN_FIELDS_DB = []
    _SCHEMA_AVAILABLE = False


class Base(DeclarativeBase):
    pass


def create_session_factory(dsn: str):
    # Import models to ensure metadata is registered before creating tables.
    from . import models  # noqa: F401

    engine = create_engine(dsn, future=True, pool_pre_ping=True)

    # Ensure the target schema exists before creating tables.
    from .constants import DEFAULT_SCHEMA, RUN_HISTORY_TABLE

    with engine.begin() as connection:
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {DEFAULT_SCHEMA}"))
        Base.metadata.create_all(connection)

        # Ensure optimizer run details column can store large JSON payloads.
        column_result = connection.execute(
            text(
                """
                SELECT DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
                FROM information_schema.columns
                WHERE table_schema = :schema
                  AND table_name = :table
                  AND column_name = 'details'
                """
            ),
            {"schema": DEFAULT_SCHEMA, "table": RUN_HISTORY_TABLE},
        ).first()

        if column_result is not None:
            data_type = (column_result[0] or "").lower()
            length = column_result[1] or 0
            if data_type != "longtext" and length < 65535:
                connection.execute(
                    text(
                        f"ALTER TABLE `{DEFAULT_SCHEMA}`.`{RUN_HISTORY_TABLE}` "
                        "MODIFY `details` LONGTEXT NULL"
                    )
                )

    return scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True))


@contextmanager
def session_scope(SessionFactory) -> Iterator:
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


__all__ = ["Base", "create_session_factory", "session_scope", "write_plan_to_mariadb"]


def write_plan_to_mariadb(plan_df: "pd.DataFrame", mariadb_dsn: str, tz_name: str) -> int:
    """Upsert planens 15-min slots til MariaDB (energy_planner.energy_plan_slots).

    - Bruger kun mariadb_dsn (ingen .env nødvendig).
    - Opretter schema/tabel hvis de mangler (DDL svarer til kravene).
    - Upsert pr. timestamp_utc (PRIMARY KEY).
    """
    if plan_df is None or plan_df.empty:
        return 0

    engine = create_engine(mariadb_dsn, future=True, pool_pre_ping=True)

    # DDL: schema + tabel
    create_schema_sql = text("""
        CREATE SCHEMA IF NOT EXISTS energy_planner
    """)
    create_table_sql = text(
        """
        CREATE TABLE IF NOT EXISTS energy_planner.energy_plan_slots (
          timestamp_utc      DATETIME NOT NULL PRIMARY KEY,
          local_time         DATETIME NOT NULL,
          day_id             DATE     NOT NULL,

          -- Prices
          price_buy          DOUBLE   DEFAULT 0,
          price_sell         DOUBLE   DEFAULT 0,
          effective_sell_price DOUBLE DEFAULT 0,

          -- Total grid flows (essential for dashboard)
          g_buy              DOUBLE   DEFAULT 0,
          g_sell             DOUBLE   DEFAULT 0,

          -- Battery totals (essential for dashboard)
          battery_in         DOUBLE   DEFAULT 0,
          battery_out        DOUBLE   DEFAULT 0,
          battery_soc        DOUBLE   DEFAULT 0,
          battery_soc_pct    DOUBLE   DEFAULT 0,

          -- Individual flow breakdowns (for detailed analysis)
          grid_to_house      DOUBLE   DEFAULT 0,
          grid_to_batt       DOUBLE   DEFAULT 0,
          grid_to_ev         DOUBLE   DEFAULT 0,
          batt_to_house      DOUBLE   DEFAULT 0,
          batt_to_ev         DOUBLE   DEFAULT 0,
          batt_to_sell       DOUBLE   DEFAULT 0,
          pv_to_house        DOUBLE   DEFAULT 0,
          pv_to_batt         DOUBLE   DEFAULT 0,
          pv_to_ev           DOUBLE   DEFAULT 0,
          
          -- EV
          ev_charge          DOUBLE   DEFAULT 0,
          ev_soc_kwh         DOUBLE   DEFAULT 0,
          ev_soc_pct         DOUBLE   DEFAULT 0,

          -- House consumption
          house_load         DOUBLE   DEFAULT 0,
          consumption_estimate_kw DOUBLE DEFAULT 0,
          pv_forecast_kw     DOUBLE   DEFAULT 0,

          -- Economics
          grid_cost          DOUBLE   DEFAULT 0,
          grid_revenue_effective DOUBLE DEFAULT 0,
          grid_revenue       DOUBLE   DEFAULT 0,
          battery_cycle_cost DOUBLE   DEFAULT 0,
          battery_value_dkk  DOUBLE   DEFAULT 0,
          ev_bonus           DOUBLE   DEFAULT 0,
          cash_cost_dkk      DOUBLE   DEFAULT 0,
          objective_component DOUBLE  DEFAULT 0,

          -- Reserves and constraints
          battery_reserve_target DOUBLE DEFAULT 0,
          battery_reserve_shortfall DOUBLE DEFAULT 0,

          -- Arbitrage diagnostics
          arb_gate           BOOLEAN  DEFAULT FALSE,
          arb_reason         VARCHAR(50) DEFAULT NULL,
          arb_basis          VARCHAR(50) DEFAULT NULL,
          arb_eta_rt         DOUBLE   DEFAULT 0,
          arb_c_cycle        DOUBLE   DEFAULT 0,
          price_buy_now      DOUBLE   DEFAULT 0,
          future_max_sell_eff DOUBLE  DEFAULT 0,
          arb_margin         DOUBLE   DEFAULT 0,

          -- Policy diagnostics
          policy_wait_flag   BOOLEAN  DEFAULT FALSE,
          policy_wait_reason TEXT     DEFAULT NULL,
          policy_price_basis VARCHAR(20) DEFAULT NULL,
          policy_price_now_dkk DOUBLE DEFAULT 0,
          policy_future_min_12h_dkk DOUBLE DEFAULT 0,
          policy_grid_charge_allowed BOOLEAN DEFAULT FALSE,
          policy_hold_value_dkk DOUBLE DEFAULT 0,

          -- Time classifications
          cheap24            BOOLEAN  DEFAULT FALSE,
          expensive24        BOOLEAN  DEFAULT FALSE,

          -- Activity label
          note               VARCHAR(100) DEFAULT NULL,

          -- Audit columns
          created_at         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
          updated_at         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

          KEY idx_day (day_id),
          KEY idx_local (local_time),
          KEY idx_cheap (cheap24),
          KEY idx_expensive (expensive24)
        )
        """
    )

    with engine.begin() as cx:
        cx.execute(create_schema_sql)
        cx.execute(create_table_sql)

    # Map DataFrame -> DB kolonner
    df = plan_df.copy()
    cols = set(df.columns)
    tz = pytz.timezone(tz_name or "Europe/Copenhagen")

    # timestamp: accept both naive and tz-aware; normalize to UTC for storage
    if "timestamp" not in cols:
        raise ValueError("Plan DataFrame mangler 'timestamp'")
    ts = pd.to_datetime(df["timestamp"], utc=True)  # ensures tz-aware in UTC
    df["timestamp_utc"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    local = ts.dt.tz_convert(tz)
    df["local_time"] = local.dt.tz_localize(None)
    df["day_id"] = local.dt.date

    # Core totals (essential for dashboard)
    if "g_buy" in cols:
        df["g_buy"] = df["g_buy"].astype(float)
    else:
        df["g_buy"] = 0.0
    if "g_sell" in cols:
        df["g_sell"] = df["g_sell"].astype(float)
    else:
        df["g_sell"] = 0.0
    if "battery_in" in cols:
        df["battery_in"] = df["battery_in"].astype(float)
    else:
        df["battery_in"] = 0.0
    if "battery_out" in cols:
        df["battery_out"] = df["battery_out"].astype(float)
    else:
        df["battery_out"] = 0.0

    # Flow breakdowns - grid flows
    if "grid_to_house" in cols:
        df["grid_to_house"] = df["grid_to_house"].astype(float)
    else:
        df["grid_to_house"] = 0.0
    if "grid_to_batt" in cols:
        df["grid_to_batt"] = df["grid_to_batt"].astype(float)
    else:
        df["grid_to_batt"] = 0.0
    if "grid_to_ev" in cols:
        df["grid_to_ev"] = df["grid_to_ev"].astype(float)
    else:
        df["grid_to_ev"] = 0.0
    
    # Flow breakdowns - battery flows
    if "batt_to_house" in cols:
        df["batt_to_house"] = df["batt_to_house"].astype(float)
    else:
        df["batt_to_house"] = 0.0
    if "batt_to_ev" in cols:
        df["batt_to_ev"] = df["batt_to_ev"].astype(float)
    else:
        df["batt_to_ev"] = 0.0
    if "batt_to_sell" in cols:
        df["batt_to_sell"] = df["batt_to_sell"].astype(float)
    else:
        df["batt_to_sell"] = 0.0
    
    # Flow breakdowns - PV flows (map prod_to_* to pv_to_*)
    if "prod_to_batt" in cols:
        df["pv_to_batt"] = df["prod_to_batt"].astype(float)
    else:
        df["pv_to_batt"] = 0.0
    if "prod_to_house" in cols:
        df["pv_to_house"] = df["prod_to_house"].astype(float)
    else:
        df["pv_to_house"] = 0.0
    if "prod_to_ev" in cols:
        df["pv_to_ev"] = df["prod_to_ev"].astype(float)
    else:
        df["pv_to_ev"] = 0.0
    
    # EV charge
    if "ev_charge" in cols:
        df["ev_charge"] = df["ev_charge"].astype(float)
    else:
        df["ev_charge"] = 0.0
    
    # Prefer consumption_estimate_kw from forecast if available (preserves low/night consumption)
    # Otherwise calculate from flows (backward compatibility)
    if "consumption_estimate_kw" in cols:
        df["house_load"] = pd.to_numeric(df["consumption_estimate_kw"], errors="coerce").fillna(0.0).astype(float)
    else:
        g2h = df.get("grid_to_house", 0.0)
        b2h = df.get("batt_to_house", 0.0)
        p2h = df.get("pv_to_house", 0.0)
        df["house_load"] = pd.Series(g2h, dtype=float).fillna(0) + pd.Series(b2h, dtype=float).fillna(0) + pd.Series(p2h, dtype=float).fillna(0)

    # SoC and note
    if "battery_soc" in cols:
        df["battery_soc"] = df["battery_soc"].astype(float)
    else:
        df["battery_soc"] = 0.0
    if "battery_soc_pct" in cols:
        df["battery_soc_pct"] = df["battery_soc_pct"].astype(float)
    else:
        df["battery_soc_pct"] = 0.0
    if "ev_soc" in cols:
        df["ev_soc_kwh"] = df["ev_soc"].astype(float)
    else:
        df["ev_soc_kwh"] = 0.0
    if "ev_soc_pct" in cols:
        df["ev_soc_pct"] = df["ev_soc_pct"].astype(float)
    else:
        df["ev_soc_pct"] = 0.0
    if "activity" in cols:
        df["note"] = df["activity"].astype(str)
    else:
        df["note"] = None

    # Auto-map all diagnostic columns that match DB schema
    # This allows adding new columns without code changes
    diagnostic_mappings = {
        # Economics
        "grid_cost": float,
        "grid_revenue": float,
        "grid_revenue_effective": float,
        "battery_cycle_cost": float,
        "battery_value_dkk": float,
        "ev_bonus": float,
        "cash_cost_dkk": float,
        "objective_component": float,
        # Reserves
        "battery_reserve_target": float,
        "battery_reserve_shortfall": float,
        # Arbitrage
        "arb_gate": bool,
        "arb_reason": str,
        "arb_basis": str,
        "arb_eta_rt": float,
        "arb_c_cycle": float,
        "price_buy_now": float,
        "future_max_sell_eff": float,
        "arb_margin": float,
        # Policy
        "policy_wait_flag": bool,
        "policy_wait_reason": str,
        "policy_price_basis": str,
        "policy_price_now_dkk": float,
        "policy_future_min_12h_dkk": float,
        "policy_grid_charge_allowed": bool,
        "policy_hold_value_dkk": float,
        # Time classifications
        "cheap24": bool,
        "expensive24": bool,
        # Mode and unmet load
        "recommended_mode": str,
        "house_load_unmet": float,
        "house_unmet_reason": str,
        # Prices
        "effective_sell_price": float,
        # Forecast inputs
        "consumption_estimate_kw": float,
        "pv_forecast_kw": float,
        # House consumption breakdown (NEW)
        "house_from_grid": float,
        "house_from_battery": float,
        "house_from_pv": float,
    }
    
    for col_name, col_type in diagnostic_mappings.items():
        if col_name in cols:
            if col_type == bool:
                df[col_name] = df[col_name].astype(bool)
            elif col_type == str:
                # For string columns, fillna with empty string before converting to avoid 'nan' strings
                df[col_name] = df[col_name].fillna("").astype(str)
                # Replace empty strings with None for proper NULL handling in database
                df[col_name] = df[col_name].replace("", None)
            else:
                df[col_name] = pd.to_numeric(df[col_name], errors="coerce").fillna(0.0).astype(float)
    
    # DEBUG: Check recommended_mode
    if "recommended_mode" in df.columns:
        non_null = df["recommended_mode"].notna().sum()
        print(f"[DB DEBUG] recommended_mode: {non_null}/{len(df)} non-null rows")
        if non_null > 0:
            print(f"[DB DEBUG] Sample values: {df['recommended_mode'].dropna().head(3).tolist()}")
    else:
        print("[DB DEBUG] recommended_mode column MISSING from DataFrame!")



    # Ensure defaults for missing numeric columns
    numeric_defaults = [
        "price_buy","price_sell",
        "g_buy","g_sell","battery_in","battery_out",
        "grid_to_house","grid_to_batt","grid_to_ev",
        "batt_to_house","batt_to_ev","batt_to_sell",
        "pv_to_house","pv_to_batt","pv_to_ev",
        "ev_charge","house_load","soc_kwh","ev_soc_kwh"
    ]
    for c in numeric_defaults:
        if c not in df.columns:
            df[c] = 0.0

    # Tidsstempler til audit-kolonner, hvis de findes i tabellen
    now_dt = datetime.now()
    if "created_at" not in df.columns:
        df["created_at"] = now_dt
    if "updated_at" not in df.columns:
        df["updated_at"] = now_dt

    # Find eksisterende kolonner i tabellen og lav dynamisk UPSERT over snittet
    with engine.begin() as cx:
        existing_cols = [r[0] for r in cx.execute(text("SHOW COLUMNS FROM energy_planner.energy_plan_slots")).all()]

    # Use centrally-defined schema if available, otherwise fallback
    if _SCHEMA_AVAILABLE and PLAN_FIELDS_DB:
        # Start with schema-defined DB columns
        desired_cols = PLAN_FIELDS_DB.copy()
        # Ensure audit columns are included
        for col in ["created_at", "updated_at", "note"]:
            if col not in desired_cols:
                desired_cols.append(col)
    else:
        # Fallback to hardcoded list (for backwards compatibility)
        desired_cols = [
            # Core identifiers
            "timestamp_utc","local_time","day_id",
            # Prices
            "price_buy","price_sell","effective_sell_price",
            # Grid flows
            "g_buy","g_sell",
            # Battery totals
            "battery_in","battery_out","battery_soc","battery_soc_pct",
            # Flow breakdowns
            "grid_to_house","grid_to_batt","grid_to_ev",
            "batt_to_house","batt_to_ev","batt_to_sell",
            "pv_to_house","pv_to_batt","pv_to_ev",
            # EV
            "ev_charge","ev_soc_kwh","ev_soc_pct",
            # House
            "house_load","consumption_estimate_kw","pv_forecast_kw",
            # Economics
            "grid_cost","grid_revenue","grid_revenue_effective",
            "battery_cycle_cost","battery_value_dkk","ev_bonus",
            "cash_cost_dkk","objective_component",
            # Reserves
            "battery_reserve_target","battery_reserve_shortfall",
            # Arbitrage diagnostics
            "arb_gate","arb_reason","arb_basis","arb_eta_rt","arb_c_cycle",
            "price_buy_now","future_max_sell_eff","arb_margin",
            # Policy diagnostics
            "policy_wait_flag","policy_wait_reason","policy_price_basis",
            "policy_price_now_dkk","policy_future_min_12h_dkk",
            "policy_grid_charge_allowed","policy_hold_value_dkk",
            # Time classifications
            "cheap24","expensive24",
            # Activity and audit
            "note","created_at","updated_at",
        ]
    
    present = [c for c in desired_cols if c in existing_cols]
    if "timestamp_utc" not in present:
        raise RuntimeError("DB-tabellen mangler 'timestamp_utc' kolonnen")
    
    # DEBUG: Check if recommended_mode is in present columns
    print(f"[DB DEBUG] Columns to write: {len(present)} total")
    print(f"[DB DEBUG] recommended_mode in present: {'recommended_mode' in present}")
    if "recommended_mode" not in present:
        print(f"[DB DEBUG] Missing! - In desired: {'recommended_mode' in desired_cols}, In existing: {'recommended_mode' in existing_cols}")

    # Trim rækker til kun de kolonner der findes i DB
    rows_all: List[Dict[str, Any]] = df.to_dict(orient="records")
    rows: List[Dict[str, Any]] = [{k: r.get(k) for k in present} for r in rows_all]
    
    # DEBUG: Check first row's recommended_mode value
    if rows and "recommended_mode" in rows[0]:
        print(f"[DB DEBUG] First row recommended_mode: '{rows[0]['recommended_mode']}'")
        # Check for None/NaN
        non_null_count = sum(1 for r in rows if r.get('recommended_mode') is not None and r.get('recommended_mode') is not pd.NA and str(r.get('recommended_mode')) != 'nan')
        print(f"[DB DEBUG] Rows with valid recommended_mode: {non_null_count}/{len(rows)}")
    
    if not rows:
        return 0

    fields_sql = ", ".join(present)
    values_sql = ", ".join(f":{c}" for c in present)
    update_cols = [c for c in present if c != "timestamp_utc"]
    update_sql = ",\n          ".join(f"{c}=VALUES({c})" for c in update_cols)

    upsert_sql = text(
        f"""
        INSERT INTO energy_planner.energy_plan_slots
        ({fields_sql})
        VALUES
        ({values_sql})
        ON DUPLICATE KEY UPDATE
          {update_sql}
        """
    )

    with engine.begin() as cx:
        cx.execute(upsert_sql, rows)
    return len(rows)
