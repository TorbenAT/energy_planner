"""SQLAlchemy ORM models for forecast, actual, and optimizer metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .constants import (
    ACTUAL_TABLE,
    FORECAST_TABLE,
    RUN_HISTORY_TABLE,
    DEFAULT_SCHEMA,
    LEARNING_METRICS_TABLE,
    PLAN_EVALUATIONS_TABLE,
    PLAN_QUARTER_ECON_TABLE,
    PLAN_SLOTS_TABLE,
)
from .db import Base


class ForecastQuarterHour(Base):
    __tablename__ = FORECAST_TABLE
    __table_args__ = (UniqueConstraint("timestamp", name="uq_forecast_timestamp"), {"schema": DEFAULT_SCHEMA})

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    production_kw: Mapped[float] = mapped_column(Float, nullable=False)
    consumption_kw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    price_buy: Mapped[float] = mapped_column(Float, nullable=False)
    price_sell: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ev_required_kw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=datetime.utcnow, nullable=False)


class ActualQuarterHour(Base):
    __tablename__ = ACTUAL_TABLE
    __table_args__ = (UniqueConstraint("timestamp", name="uq_actual_timestamp"), {"schema": DEFAULT_SCHEMA})

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    production_kw: Mapped[float] = mapped_column(Float, nullable=False)
    consumption_kw: Mapped[float] = mapped_column(Float, nullable=False)
    grid_import_kw: Mapped[float] = mapped_column(Float, nullable=False)
    grid_export_kw: Mapped[float] = mapped_column(Float, nullable=False)
    battery_soc_kw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ev_soc_kw: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=datetime.utcnow, nullable=False)


class OptimizerRun(Base):
    __tablename__ = RUN_HISTORY_TABLE
    __table_args__ = {"schema": DEFAULT_SCHEMA}

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_started_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    run_completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False), nullable=True)
    horizon_hours: Mapped[int] = mapped_column(Integer, nullable=False)
    resolution_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    objective_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class LearningMetric(Base):
    __tablename__ = LEARNING_METRICS_TABLE
    __table_args__ = (
        UniqueConstraint("metric", "scope", name="uq_learning_metric_scope"),
        {"schema": DEFAULT_SCHEMA},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    metric: Mapped[str] = mapped_column(String(64), nullable=False)
    scope: Mapped[str] = mapped_column(String(64), nullable=False, default="global")
    value: Mapped[float] = mapped_column(Float, nullable=False)
    variance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class PlanEvaluation(Base):
    __tablename__ = PLAN_EVALUATIONS_TABLE
    __table_args__ = (
        UniqueConstraint("run_id", name="uq_plan_evaluation_run"),
        {"schema": DEFAULT_SCHEMA},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(f"{DEFAULT_SCHEMA}.{RUN_HISTORY_TABLE}.id"),
        nullable=False,
    )
    plan_generated_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    horizon_hours: Mapped[int] = mapped_column(Integer, nullable=False)
    planned_objective: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    planned_cost_dkk: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    realized_objective: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    realized_cost_dkk: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    regret_dkk: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )


class PlanQuarterEconomics(Base):
    __tablename__ = PLAN_QUARTER_ECON_TABLE
    __table_args__ = (
        UniqueConstraint("run_id", "timestamp", name="uq_plan_quarter_run_ts"),
        {"schema": DEFAULT_SCHEMA},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(f"{DEFAULT_SCHEMA}.{RUN_HISTORY_TABLE}.id"),
        nullable=False,
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    price_buy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    price_sell: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    g_buy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    g_sell: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    battery_in: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    battery_out: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    battery_soc: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    battery_soc_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    grid_cost_dkk: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    grid_revenue_dkk: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    grid_revenue_effective_dkk: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    battery_cycle_cost_dkk: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    battery_value_dkk: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    objective_component_dkk: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    activity: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)


class EnergyPlanSlot(Base):
    __tablename__ = PLAN_SLOTS_TABLE
    __table_args__ = (
        UniqueConstraint("timestamp_utc", name="uq_plan_slots_timestamp"),
        {"schema": DEFAULT_SCHEMA},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # Core keys and times
    timestamp_utc: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    local_time: Mapped[str] = mapped_column(String(32), nullable=False)
    day_id: Mapped[str] = mapped_column(String(10), nullable=False)  # YYYY-MM-DD (local)

    # Prices (DKK/kWh)
    price_buy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    price_sell: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Flows (kWh per 15 min)
    grid_to_batt: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    batt_to_house: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    batt_to_ev: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    grid_to_ev: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ev_charge: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pv_to_batt: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pv_to_house: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    house_load: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # State & note
    soc_kwh: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=datetime.utcnow, nullable=False)

__all__ = [
    "ForecastQuarterHour",
    "ActualQuarterHour",
    "OptimizerRun",
    "LearningMetric",
    "PlanEvaluation",
    "PlanQuarterEconomics",
    "EnergyPlanSlot",
]
