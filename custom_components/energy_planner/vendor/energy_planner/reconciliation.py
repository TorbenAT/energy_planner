"""Utilities for comparing forecasts and actuals to refine models over time."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable

import pandas as pd  # type: ignore
from sqlalchemy import select  # type: ignore

from .models import ActualQuarterHour, ForecastQuarterHour


@dataclass(slots=True)
class ErrorMetrics:
    horizon_start: datetime
    horizon_end: datetime
    mean_absolute_error_kw: float
    mean_bias_kw: float
    total_cost_delta: float


class Reconciler:
    def __init__(self, SessionFactory) -> None:
        self.SessionFactory = SessionFactory

    def load_frames(self, start: datetime, end: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
        with self.SessionFactory() as session:
            forecast_rows = session.execute(
                select(ForecastQuarterHour).where(
                    ForecastQuarterHour.timestamp >= start,
                    ForecastQuarterHour.timestamp < end,
                )
            ).scalars().all()

            actual_rows = session.execute(
                select(ActualQuarterHour).where(
                    ActualQuarterHour.timestamp >= start,
                    ActualQuarterHour.timestamp < end,
                )
            ).scalars().all()

        forecast_df = pd.DataFrame([
            {
                "timestamp": row.timestamp,
                "production_kw": row.production_kw,
                "consumption_kw": row.consumption_kw,
                "price_buy": row.price_buy,
                "price_sell": row.price_sell,
            }
            for row in forecast_rows
        ])

        actual_df = pd.DataFrame([
            {
                "timestamp": row.timestamp,
                "production_kw": row.production_kw,
                "consumption_kw": row.consumption_kw,
                "grid_import_kw": row.grid_import_kw,
                "grid_export_kw": row.grid_export_kw,
            }
            for row in actual_rows
        ])

        return forecast_df, actual_df

    def calculate_metrics(self, forecast_df: pd.DataFrame, actual_df: pd.DataFrame) -> ErrorMetrics:
        if forecast_df.empty or actual_df.empty:
            raise ValueError("Forecast and actual data are required to compute metrics")

        merged = forecast_df.merge(actual_df, on="timestamp", suffixes=("_forecast", "_actual"))
        merged["production_error"] = merged["production_kw_actual"] - merged["production_kw_forecast"]
        merged["consumption_error"] = merged["consumption_kw_actual"] - merged["consumption_kw_forecast"]

        mae = merged["consumption_error"].abs().mean()
        bias = merged["consumption_error"].mean()
        cost_delta = (merged["grid_import_kw"] * merged.get("price_buy", 0)).sum()

        return ErrorMetrics(
            horizon_start=merged["timestamp"].min(),
            horizon_end=merged["timestamp"].max(),
            mean_absolute_error_kw=float(mae or 0.0),
            mean_bias_kw=float(bias or 0.0),
            total_cost_delta=float(cost_delta or 0.0),
        )
