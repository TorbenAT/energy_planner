"""Learning manager for adaptive optimisation metrics persisted in MariaDB."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, Optional

import pandas as pd  # type: ignore
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .constants import BATTERY_CAPACITY_KWH
from .db import session_scope
from .models import (
    LearningMetric,
    OptimizerRun,
    PlanEvaluation,
    PlanQuarterEconomics,
)
if TYPE_CHECKING:
    from .reporting import PlanReport


@dataclass(slots=True)
class MetricValue:
    """Lightweight representation of a learned metric."""

    value: float
    sample_count: int


@dataclass(slots=True)
class LearningSnapshot:
    """Container with convenient accessors for learned metrics."""

    metrics: Dict[str, Dict[str, MetricValue]]

    def get(self, metric: str, scope: str = "global", default: Optional[float] = None) -> Optional[float]:
        scoped = self.metrics.get(metric, {})
        record = scoped.get(scope)
        return record.value if record else default

    def sample_count(self, metric: str, scope: str = "global") -> int:
        scoped = self.metrics.get(metric, {})
        record = scoped.get(scope)
        return record.sample_count if record else 0

    def house_daily_kwh(self, weekday: int, fallback: Optional[float] = None) -> Optional[float]:
        scope = f"weekday_{weekday}"
        return self.get("house_daily_kwh", scope, fallback)

    def ev_daily_kwh(self, fallback: Optional[float] = None) -> Optional[float]:
        return self.get("ev_daily_kwh", "global", fallback)

    def battery_hold_value(self, fallback: float = 0.0) -> float:
        return self.get("battery_hold_value_dkk", "global", fallback) or fallback

    def price_buy_high_threshold(self, fallback: float = 0.0) -> float:
        return self.get("price_buy_high_threshold", "global", fallback) or fallback

    def battery_min_soc_pct(self, fallback: Optional[float] = None) -> Optional[float]:
        return self.get("battery_min_soc_pct", "global", fallback)


def load_learning_snapshot(SessionFactory) -> LearningSnapshot:
    """Fetch all learned metrics for reuse in the adaptive policy."""

    metrics: Dict[str, Dict[str, MetricValue]] = {}
    with session_scope(SessionFactory) as session:
        rows = session.execute(select(LearningMetric)).scalars().all()
        for row in rows:
            metric_map = metrics.setdefault(row.metric, {})
            metric_map[row.scope] = MetricValue(value=float(row.value), sample_count=int(row.sample_count or 0))
    return LearningSnapshot(metrics=metrics)


class LearningManager:
    """Persist and update learning-related artefacts for optimisation."""

    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = max(0.0, min(1.0, alpha)) or 0.2

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def persist_plan_artifacts(
        self,
        session: Session,
        run: OptimizerRun,
    report: "PlanReport",
    ) -> None:
        """Store quarter-hour economics and learning metrics for a plan run."""

        self._persist_quarter_economics(session, run, report)
        self._persist_plan_evaluation(session, run, report)
        self._update_learning_metrics(session, report)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _persist_quarter_economics(
        self,
        session: Session,
        run: OptimizerRun,
    report: "PlanReport",
    ) -> None:
        plan_frame = report.plan
        session.execute(
            delete(PlanQuarterEconomics).where(PlanQuarterEconomics.run_id == run.id)
        )
        if plan_frame.empty:
            return

        records: list[PlanQuarterEconomics] = []
        for row in plan_frame.itertuples(index=False):
            timestamp = self._to_utc_naive(getattr(row, "timestamp", None))
            if timestamp is None:
                continue
            records.append(
                PlanQuarterEconomics(
                    run_id=run.id,
                    timestamp=timestamp,
                    price_buy=self._clean_float(getattr(row, "price_buy", None)),
                    price_sell=self._clean_float(getattr(row, "price_sell", None)),
                    g_buy=self._clean_float(getattr(row, "g_buy", None)),
                    g_sell=self._clean_float(getattr(row, "g_sell", None)),
                    battery_in=self._clean_float(getattr(row, "battery_in", None)),
                    battery_out=self._clean_float(getattr(row, "battery_out", None)),
                    battery_soc=self._clean_float(getattr(row, "battery_soc", None)),
                    battery_soc_pct=self._clean_float(getattr(row, "battery_soc_pct", None)),
                    grid_cost_dkk=self._clean_float(getattr(row, "grid_cost", None)),
                    grid_revenue_dkk=self._clean_float(getattr(row, "grid_revenue", None)),
                    grid_revenue_effective_dkk=self._clean_float(
                        getattr(row, "grid_revenue_effective", None)
                    ),
                    battery_cycle_cost_dkk=self._clean_float(
                        getattr(row, "battery_cycle_cost", None)
                    ),
                    battery_value_dkk=self._clean_float(getattr(row, "battery_value_dkk", None)),
                    objective_component_dkk=self._clean_float(
                        getattr(row, "objective_component", None)
                    ),
                    activity=getattr(row, "activity", None),
                )
            )
        if records:
            session.bulk_save_objects(records)

    def _persist_plan_evaluation(
        self,
        session: Session,
        run: OptimizerRun,
    report: "PlanReport",
    ) -> None:
        plan_frame = report.plan
        grid_cost = float(plan_frame.get("grid_cost", pd.Series(dtype=float)).sum()) if not plan_frame.empty else 0.0
        grid_revenue_effective = (
            float(plan_frame.get("grid_revenue_effective", pd.Series(dtype=float)).sum())
            if not plan_frame.empty
            else 0.0
        )
        battery_cycle_cost = (
            float(plan_frame.get("battery_cycle_cost", pd.Series(dtype=float)).sum())
            if not plan_frame.empty
            else 0.0
        )
        ev_bonus = float(plan_frame.get("ev_bonus", pd.Series(dtype=float)).sum()) if not plan_frame.empty else 0.0
        planned_cost = grid_cost - grid_revenue_effective + battery_cycle_cost - ev_bonus

        notes_payload: Dict[str, Any] = {
            "grid_cost_dkk": round(grid_cost, 3),
            "grid_revenue_effective_dkk": round(grid_revenue_effective, 3),
            "battery_cycle_cost_dkk": round(battery_cycle_cost, 3),
            "ev_bonus_dkk": round(ev_bonus, 3),
            "plan_rows": len(plan_frame),
        }

        session.execute(delete(PlanEvaluation).where(PlanEvaluation.run_id == run.id))
        session.add(
            PlanEvaluation(
                run_id=run.id,
                plan_generated_at=run.run_started_at,
                horizon_hours=run.horizon_hours,
                planned_objective=report.objective_value,
                planned_cost_dkk=planned_cost,
                realized_objective=None,
                realized_cost_dkk=None,
                regret_dkk=None,
                notes=json.dumps(notes_payload),
            )
        )

    def _update_learning_metrics(self, session: Session, report: PlanReport) -> None:
        plan_frame = report.plan
        if plan_frame.empty:
            return

        # Daily house consumption per weekday
        for day in getattr(report, "day_summary", []):
            try:
                period_start = datetime.fromisoformat(day["period_start"])
            except Exception:
                continue
            weekday_scope = f"weekday_{period_start.weekday()}"
            consumption_kwh = float(day.get("consumption_kwh", 0.0))
            if consumption_kwh <= 0:
                continue
            self._ewma_metric(session, "house_daily_kwh", weekday_scope, consumption_kwh)

        # EV expected daily energy
        totals = getattr(report, "summary", {}).get("energy_totals", {})
        ev_charge_kwh = float(totals.get("ev_charge_kwh", 0.0) or 0.0)
        horizon_days = max(1, len(getattr(report, "day_summary", [])) or 1)
        ev_daily_kwh = ev_charge_kwh / horizon_days
        if ev_daily_kwh >= 0:
            self._ewma_metric(session, "ev_daily_kwh", "global", ev_daily_kwh)

        # Battery round-trip efficiency estimate
        battery_in = float(plan_frame.get("battery_in", pd.Series(dtype=float)).sum())
        battery_out = float(plan_frame.get("battery_out", pd.Series(dtype=float)).sum())
        if battery_in > 0 and battery_out >= 0:
            efficiency = max(0.0, min(1.2, battery_out / battery_in))
            self._ewma_metric(session, "battery_roundtrip_efficiency", "global", efficiency)

        # Battery hold value (price threshold for buying energy to store)
        price_buy = plan_frame.get("price_buy", pd.Series(dtype=float)).astype(float)
        if not price_buy.empty:
            hold_value = float(price_buy.quantile(0.35))
            expensive_value = float(price_buy.quantile(0.75))
            self._ewma_metric(session, "battery_hold_value_dkk", "global", hold_value)
            self._ewma_metric(session, "price_buy_high_threshold", "global", expensive_value)

        # Battery reserve target (average minimum SoC recorded)
        battery_soc = plan_frame.get("battery_soc", pd.Series(dtype=float)).astype(float)
        if not battery_soc.empty and BATTERY_CAPACITY_KWH > 0:
            min_soc_pct = float((battery_soc.min() / BATTERY_CAPACITY_KWH) * 100.0)
            self._ewma_metric(session, "battery_min_soc_pct", "global", min_soc_pct)

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------
    def _ewma_metric(self, session: Session, metric: str, scope: str, sample: float) -> None:
        sample = float(sample)
        result = session.execute(
            select(LearningMetric).where(
                LearningMetric.metric == metric,
                LearningMetric.scope == scope,
            )
        ).scalar_one_or_none()

        now = datetime.utcnow()
        if result is None:
            session.add(
                LearningMetric(
                    metric=metric,
                    scope=scope,
                    value=sample,
                    variance=None,
                    sample_count=1,
                    metadata_json=json.dumps({"last_sample": sample, "alpha": self.alpha}),
                    created_at=now,
                    updated_at=now,
                )
            )
            return

        previous = result.value
        updated_value = self.alpha * sample + (1.0 - self.alpha) * previous
        result.value = updated_value
        result.sample_count = int(result.sample_count or 0) + 1
        result.metadata_json = json.dumps({"last_sample": sample, "alpha": self.alpha})
        result.updated_at = now

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_float(value: Optional[Any]) -> Optional[float]:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if pd.isna(numeric):  # type: ignore[arg-type]
            return None
        return numeric

    @staticmethod
    def _to_utc_naive(value: Optional[Any]) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is not None:
                return value.astimezone(timezone.utc).replace(tzinfo=None)
            return value
        if isinstance(value, pd.Timestamp):
            ts = value
            if ts.tzinfo is None:
                return ts.to_pydatetime()
            return ts.tz_convert("UTC").to_pydatetime().replace(tzinfo=None)
        try:
            parsed = pd.Timestamp(value)
        except Exception:
            return None
        if parsed.tzinfo is None:
            return parsed.to_pydatetime()
        return parsed.tz_convert("UTC").to_pydatetime().replace(tzinfo=None)


__all__ = ["LearningManager", "LearningSnapshot", "load_learning_snapshot"]
