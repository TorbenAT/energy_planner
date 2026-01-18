"""Data ingestion and persistence pipeline."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd  # type: ignore
import requests  # type: ignore
import numpy as np  # type: ignore
from sqlalchemy.dialects.mysql import insert  # type: ignore

from .config import Settings
from .constants import (
    BATTERY_CAPACITY_KWH,
    DEFAULT_SCHEMA,
    EV_BATTERY_CAPACITY_KWH,
)
from .constants import SLOTS_PER_HOUR, DEFAULT_RESOLUTION_MINUTES
from .db import session_scope
from .ha_client import HomeAssistantClient
from .models import ActualQuarterHour, ForecastQuarterHour
from .optimizer.solver import OptimizationContext
from .utils.time import ensure_timezone, floor_to_resolution, generate_quarter_range, to_utc_naive


logger = logging.getLogger(__name__)


FALLBACK_EV_SENSOR = "sensor.carport_lifetime_energy"


class DataPipeline:
    def __init__(self, settings: Settings, ha: HomeAssistantClient, SessionFactory) -> None:
        self.settings = settings
        self.ha = ha
        self.SessionFactory = SessionFactory
        self.last_consumption_note: Optional[str] = None
        self.last_hist_quarter: Optional[pd.DataFrame] = None
        self.last_daily_reconciliation: Optional[list[dict]] = None

    # ------------------------------------------------------------------
    # Helpers: reset-robust deltas from cumulative sensors
    # ------------------------------------------------------------------
    def _load_cumulative_qh(
        self,
        sensor: Optional[str],
        start: datetime,
        end: datetime,
        period_minutes: int,
    ) -> pd.Series:
        """Return kWh per period (resampled) from an increasing cumulative sensor.

        Handles resets/rollovers by treating negative diffs as a reset and
        using the new absolute value as the first delta after reset.
        """
        if not sensor:
            return pd.Series(dtype=float)

        try:
            samples = self.ha.fetch_history_series(sensor, start, end)
        except Exception:  # pragma: no cover - defensive path
            return pd.Series(dtype=float)

        if not samples:
            return pd.Series(dtype=float)

        df = (
            pd.DataFrame(samples, columns=["timestamp", "value"])
            .assign(
                timestamp=lambda x: pd.to_datetime(x["timestamp"], utc=True),
                value=lambda x: pd.to_numeric(x["value"], errors="coerce"),
            )
            .dropna()
            .sort_values("timestamp")
        )

        if df.empty:
            return pd.Series(dtype=float)

        delta = df["value"].diff()
        resets = delta < 0
        delta.loc[resets] = df.loc[resets, "value"]
        delta = delta.fillna(0.0).clip(lower=0.0)

        freq = f"{max(period_minutes, 1)}min"
        qh_kwh = (
            df.assign(delta=delta)
            .set_index("timestamp")["delta"]
            .resample(freq, label="right", closed="right")
            .sum()
        )
        return qh_kwh.astype(float)

    def _load_ev_qh_and_daily(
        self,
        sensor: Optional[str],
        start: datetime,
        end: datetime,
        period_minutes: int,
        tz: str,
    ) -> tuple[pd.Series, pd.Series]:
        """Load EV energy as kWh per period and daily totals from a cumulative sensor.

        Returns (quarter_hour_kwh_series, daily_totals_kwh). Empty series if missing.
        """
        ev_qh = self._load_cumulative_qh(sensor, start, end, period_minutes)
        if ev_qh.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        tmp = ev_qh.to_frame("ev_kwh")
        tmp["local"] = tmp.index.tz_convert(tz)
        tmp["date"] = pd.to_datetime(tmp["local"]).dt.date
        daily = tmp.groupby("date")["ev_kwh"].sum()
        return ev_qh, daily

    def _daily_reconciliation(self, hist: pd.DataFrame, tz: str, period_minutes: int) -> pd.DataFrame:
        if hist.empty:
            return pd.DataFrame(columns=[
                "date",
                "house_total_kwh",
                "ev_kwh",
                "house_no_ev_kwh",
                "pv_total_kwh",
                "grid_buy_kwh",
                "grid_sell_kwh",
                "batt_in_kwh",
                "batt_out_kwh",
                "qh_points",
            ])

        loc = hist.index.tz_convert(tz)
        date = pd.to_datetime(loc.date)
        hist_with_date = hist.assign(date=date)
        agg = {
            "house_total_kwh": ("house_total_kwh", "sum"),
            "ev_kwh": ("ev_kwh", "sum"),
            "pv_kwh": ("pv_kwh", "sum"),
            "grid_buy_kwh": ("grid_buy_kwh", "sum"),
            "grid_sell_kwh": ("grid_sell_kwh", "sum"),
            "batt_in_kwh": ("batt_in_kwh", "sum"),
            "batt_out_kwh": ("batt_out_kwh", "sum"),
        }
        g = hist_with_date.groupby("date").agg(**agg)
        g["house_no_ev_kwh"] = (g["house_total_kwh"] - g["ev_kwh"]).clip(lower=0.0)
        counts = hist_with_date.groupby("date").size().rename("qh_points")
        daily = g.join(counts, how="left").reset_index(names="date")
        return daily

    def _load_ev_adjustment(
        self,
        start: datetime,
        end: datetime,
        period_minutes: int,
    ) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[str], bool]:
        sensor = self.settings.ev_energy_sensor
        fallback_used = False
        if not sensor:
            sensor = FALLBACK_EV_SENSOR
            fallback_used = True

        if not sensor:
            return None, None, None, fallback_used

        try:
            samples = self.ha.fetch_history_series(sensor, start, end)
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning("Failed to fetch EV history for %s: %s", sensor, exc)
            return None, None, sensor, fallback_used

        if not samples:
            return None, None, sensor, fallback_used

        df = pd.DataFrame(samples, columns=["timestamp", "value"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df.dropna(subset=["value"], inplace=True)
        df.sort_values("timestamp", inplace=True)

        if df.empty:
            return None, None, sensor, fallback_used

        delta = df["value"].diff()
        resets = delta < 0
        delta.loc[resets] = df.loc[resets, "value"]
        delta.fillna(0.0, inplace=True)
        delta = delta.clip(lower=0.0)

        freq = f"{max(period_minutes, 1)}min"
        resampled_kwh = (
            df.assign(delta=delta)
            .set_index("timestamp")["delta"]
            .resample(freq, label="right", closed="right")
            .sum()
        )

        if resampled_kwh.empty:
            return None, None, sensor, fallback_used

        period_hours = max(period_minutes / 60.0, 1e-9)
        kw_series = (resampled_kwh / period_hours).astype(float)
        kw_series.index = kw_series.index.tz_convert(self.settings.timezone)
        kw_series.index = kw_series.index.floor(freq)
        kw_series = kw_series.groupby(kw_series.index).sum()

        daily_kwh_frame = resampled_kwh.to_frame(name="ev_kwh")
        daily_kwh_frame["local_ts"] = daily_kwh_frame.index.to_series().apply(
            lambda ts: ensure_timezone(ts.to_pydatetime(), self.settings.timezone)
        )
        daily_kwh_frame["local_date"] = pd.to_datetime(daily_kwh_frame["local_ts"]).dt.date
        daily_totals = daily_kwh_frame.groupby("local_date")["ev_kwh"].sum()

        return kw_series, daily_totals, sensor, fallback_used

    def _load_price_forecast(self, freq: str) -> Optional[pd.Series]:
        url = self.settings.price_forecast_url
        if not url:
            return None

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:  # pragma: no cover - network failure path
            logger.warning("Failed to fetch external price forecast from %s: %s", url, exc)
            return None

        zone = (self.settings.price_forecast_zone or "DK1").upper()
        entries = payload.get(zone)
        if not isinstance(entries, list):
            logger.debug("Price forecast payload lacked zone %s", zone)
            return None

        values: List[float] = []
        index: List[pd.Timestamp] = []
        fx_rate = max(0.0, float(self.settings.price_forecast_eur_to_dkk))

        for item in entries:
            if not isinstance(item, dict):
                continue
            ts_raw = item.get("Time") or item.get("time") or item.get("timestamp")
            price_raw = item.get("Price") or item.get("price")
            if ts_raw is None or price_raw is None:
                continue
            try:
                ts = pd.Timestamp(ts_raw)
            except (TypeError, ValueError):
                continue
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            local_ts = ts.tz_convert(self.settings.timezone)
            try:
                price_val = float(price_raw)
            except (TypeError, ValueError):
                continue
            price_dkk = price_val * fx_rate / 1000.0 if fx_rate > 0 else price_val / 1000.0
            index.append(local_ts)
            values.append(price_dkk)

        if not values:
            return None

        series = pd.Series(values, index=index, dtype=float).sort_index()
        series = series[~series.index.duplicated(keep="last")]
        try:
            resampled = series.resample(freq).ffill()
        except ValueError:
            # If resample fails due to sparse data, fall back to the original series
            resampled = series
        return resampled

    def _prepare_history_frame(
        self,
        entries: List[tuple[pd.Timestamp, float]],
        freq: str,
        ev_lookup: Dict[pd.Timestamp, float],
    ) -> pd.DataFrame:
        if not entries:
            return pd.DataFrame(columns=["timestamp", "kw", "slot_ts", "slot", "ev_kw", "net_kw"])

        history_df = pd.DataFrame(entries, columns=["timestamp", "kw"])
        history_df.sort_values("timestamp", inplace=True)
        history_df["kw"] = history_df["kw"].astype(float)
        history_df["slot_ts"] = history_df["timestamp"].dt.floor(freq)
        history_df["slot"] = history_df["slot_ts"].dt.strftime("%H:%M")
        if ev_lookup:
            history_df["ev_kw"] = history_df["slot_ts"].map(ev_lookup).fillna(0.0)
        else:
            history_df["ev_kw"] = 0.0
        history_df["net_kw"] = (history_df["kw"] - history_df["ev_kw"]).clip(lower=0.0)
        return history_df

    def _format_ev_note(
        self,
        sensor: Optional[str],
        ev_kw_series: Optional[pd.Series],
        ev_daily_kwh: Optional[pd.Series],
        period_hours: float,
        fallback_used: bool,
    ) -> str:
        if ev_kw_series is None or ev_kw_series.empty:
            return ""

        if ev_daily_kwh is not None and not ev_daily_kwh.empty:
            total_kwh = float(ev_daily_kwh.sum())
        else:
            total_kwh = float(ev_kw_series.sum() * period_hours)

        sensor_label = sensor or "unknown sensor"
        suffix = ""
        if fallback_used and not self.settings.ev_energy_sensor:
            suffix = " (fallback sensor)"

        return f" EV subtraction applied from {sensor_label} (~{total_kwh:.1f} kWh)." + suffix

    def _default_consumption_series(self, index: pd.Index) -> pd.Series:
        hourly_profile = [
            0.4, 0.4, 0.4, 0.4, 0.4, 0.4,  # 00-05
            0.6, 0.6, 0.6,  # 06-08
            0.8, 0.8, 0.8, 0.8, 0.8, 0.8,  # 09-14
            1.0, 1.0, 1.0,  # 15-17
            1.2, 1.2, 1.2,  # 18-20
            0.8, 0.8,  # 21-22
            0.6,  # 23
        ]

        values: List[float] = []
        tz_name = self.settings.timezone
        for ts in index:
            localized = ensure_timezone(ts, tz_name)
            hour = localized.hour
            values.append(hourly_profile[hour if hour < len(hourly_profile) else hour % 24])
        return pd.Series(values, index=index, dtype=float)

    def build_forecast_dataframe(self, now: datetime, ctx: Optional[OptimizationContext] = None) -> pd.DataFrame:
        tz_now = ensure_timezone(now, self.settings.timezone)
        period_minutes = self.settings.resolution_minutes
        freq_str = f"{period_minutes}min"
        horizon_periods = int((self.settings.lookahead_hours * 60) / period_minutes)
        start = floor_to_resolution(tz_now, period_minutes)
        index = generate_quarter_range(start, horizon_periods, period_minutes)
        df = pd.DataFrame(index=index)
        self.last_consumption_note = None

        prod_points = self.ha.fetch_solcast_forecast(self.settings.forecast_sensors)
        if prod_points:
            prod_series = (
                pd.Series(
                    data=[p.pv_estimate_kw for p in prod_points],
                    index=[ensure_timezone(p.timestamp, self.settings.timezone) for p in prod_points],
                    dtype=float,
                )
                .sort_index()
            )
            prod_series = prod_series[~prod_series.index.duplicated(keep="last")]
            aligned_prod = prod_series.reindex(df.index, method="ffill")
            df["pv_forecast_kw"] = aligned_prod.fillna(0.0).to_numpy()
        else:
            df["pv_forecast_kw"] = 0.0

        price_buy_state = self.ha.fetch_state(self.settings.price_buy_sensor)
        price_buy = self.ha.fetch_price_series(self.settings.price_buy_sensor)
        
        # Convert to a Series for easier resampling if sensor is high resolution (e.g. 15min)
        if price_buy:
            price_series = pd.Series(
                data=[p.value for p in price_buy],
                index=[ensure_timezone(p.starts_at, self.settings.timezone) for p in price_buy]
            ).sort_index()
            # If the market is 15-min, we take the mean to match our 60-min slot
            price_series = price_series.resample(freq_str).mean().ffill()
            buy_map = price_series.to_dict()
        else:
            buy_map = {}

        df["price_buy"] = [buy_map.get(ts) for ts in df.index]

        price_sell = self.ha.fetch_price_series(self.settings.price_sell_sensor)
        if price_sell:
            sell_series = pd.Series(
                data=[p.value for p in price_sell],
                index=[ensure_timezone(p.starts_at, self.settings.timezone) for p in price_sell]
            ).sort_index()
            sell_series = sell_series.resample(freq_str).mean().ffill()
            sell_map = sell_series.to_dict()
        else:
            sell_map = {}

        df["price_sell"] = [sell_map.get(ts) for ts in df.index]

        period_hours = max(self.settings.resolution_minutes / 60.0, 1e-9)

        external_buy = self._load_price_forecast(freq_str)
        if external_buy is not None:
            aligned_external = external_buy.reindex(df.index, method="ffill")
            
            # Attempt to enrich forecast with tariffs from the price sensor attributes
            hourly_tariffs = {}
            fixed_sum = 0.0
            tariffs_found = False
            
            try:
                attrs = price_buy_state.get("attributes", {})
                # Sum fixed tariffs (DKK/kWh)
                for key in ["transmissions_nettarif", "systemtarif", "elafgift"]:
                    val = attrs.get(key)
                    if val is not None:
                        fixed_sum += float(val)
                
                # Parse hourly tariffs (hour index 0-23 -> DKK/kWh)
                raw_tariffs = attrs.get("tariffs", {})
                if raw_tariffs:
                    for h_str, val in raw_tariffs.items():
                        try:
                            hourly_tariffs[int(h_str)] = float(val)
                        except (ValueError, TypeError):
                            pass
                
                if fixed_sum > 0 or hourly_tariffs:
                    tariffs_found = True
            except Exception as exc:
                logger.warning("Failed to extract tariffs from price sensor: %s", exc)

            if tariffs_found:
                # Calculate tariff adder for each slot
                forecast_values = aligned_external.astype(float).fillna(0.0).to_numpy()
                tariff_adder = np.zeros(len(df))
                
                for i, ts in enumerate(df.index):
                    # ts is a Timestamp (local time)
                    h = ts.hour
                    var_tariff = hourly_tariffs.get(h, 0.0)
                    tariff_adder[i] = fixed_sum + var_tariff
                
                raw_with_tariffs = forecast_values + tariff_adder
                
                # Calibration: Compare current estimated price with actual sensor state
                # to account for VAT or missing components.
                current_actual = float(price_buy_state.get("state") or 0.0)
                final_forecast = raw_with_tariffs
                
                if current_actual > 0 and len(raw_with_tariffs) > 0:
                    # Use the first slot (closest to 'now') for calibration
                    estimated_now = raw_with_tariffs[0]
                    offset = current_actual - estimated_now
                    
                    # Only apply offset if it's reasonable (e.g., < 5 DKK) to avoid garbage data ruining the plan
                    if abs(offset) < 5.0:
                        final_forecast = raw_with_tariffs + offset
                        logger.info(
                            "Calibrated price forecast: spot+tariffs=%.2f, actual=%.2f, offset=%.2f",
                            estimated_now, current_actual, offset
                        )
                    else:
                        logger.warning(
                            "Price calibration offset too large (%.2f); using uncalibrated forecast.",
                            offset
                        )
                
                df["price_buy_forecast"] = final_forecast
            else:
                # Fallback: use raw forecast if no tariffs found
                df["price_buy_forecast"] = aligned_external.astype(float).to_numpy()
        else:
            df["price_buy_forecast"] = float("nan")

        df["price_buy"] = pd.to_numeric(df["price_buy"], errors="coerce")
        df["price_sell"] = pd.to_numeric(df["price_sell"], errors="coerce")
        df["price_buy_forecast"] = pd.to_numeric(df["price_buy_forecast"], errors="coerce")
        
        # Prefer sensor data (price_buy) where available, fill with forecast
        # Note: price_buy comes from HA sensor attributes (raw_today/tomorrow) which are usually
        # the most accurate source for near-term prices including tariffs if the sensor supports it.
        df["price_buy_signal"] = df["price_buy"].combine_first(df["price_buy_forecast"])

        df["price_allin_buy"] = df["price_buy_signal"].astype(float)
        sell_multiplier = max(0.0, float(self.settings.grid_sell_price_multiplier))
        sell_penalty = max(0.0, float(self.settings.grid_sell_penalty_dkk_per_kwh))
        df["price_eff_sell"] = (df["price_sell"].fillna(0.0).astype(float) * sell_multiplier) - sell_penalty
        df["price_eff_sell"] = df["price_eff_sell"].clip(lower=0.0)

        prices_array = df["price_allin_buy"].to_numpy(dtype=float)
        count = len(prices_array)
        future_min = np.full(count, np.nan)
        future_max = np.full(count, np.nan)
        future_p25 = np.full(count, np.nan)
        future_p75 = np.full(count, np.nan)
        future_std = np.full(count, np.nan)

        for idx in range(count):
            tail = prices_array[idx:]
            tail = tail[~np.isnan(tail)]
            if tail.size == 0:
                continue
            future_min[idx] = float(np.min(tail))
            future_max[idx] = float(np.max(tail))
            future_p25[idx] = float(np.percentile(tail, 25))
            future_p75[idx] = float(np.percentile(tail, 75))
            future_std[idx] = float(np.std(tail, ddof=0))

        df["price_future_min"] = future_min
        df["price_future_max"] = future_max
        df["price_future_p25"] = future_p25
        df["price_future_p75"] = future_p75
        df["price_future_std"] = future_std

        def _kwh_to_kw(raw_value: Optional[float]) -> Optional[float]:
            if raw_value is None:
                return None
            try:
                return float(raw_value) / period_hours
            except (TypeError, ValueError):
                return None

        history_used = False

        # ML Forecast Path
        if self.settings.use_ml_forecast:
            try:
                from .forecasting import ConsumptionForecaster
                forecaster = ConsumptionForecaster(self.settings.timezone)
                forecaster.train(
                    self.SessionFactory, 
                    days_history=self.settings.learning_history_days,
                    ha_client=self.ha,
                    settings=self.settings
                )
                
                # Predict for the forecast horizon
                timestamps = df.index.tolist()
                predictions = forecaster.predict(timestamps)
                df["consumption_estimate_kw"] = predictions
                
                self.last_consumption_note = f"ML Forecast (Linear Regression) used. Trained on {self.settings.learning_history_days} days."
                history_used = True
                
            except Exception as e:
                logger.warning(f"ML Forecast unavailable: {e}. Using legacy profile.")
                # Fall through to legacy logic

        # Preferred path: build profile from cumulative (increasing) sensors if configured
        if not history_used and self.settings.house_total_sensor:
            history_window = max(self.settings.consumption_history_hours, self.settings.lookahead_hours)
            history_start = tz_now - timedelta(hours=history_window)
            period_minutes = self.settings.resolution_minutes
            period_hours = max(period_minutes / 60.0, 1e-9)

            total_qh = self._load_cumulative_qh(
                self.settings.house_total_sensor, history_start, tz_now, period_minutes
            )
            # Guard: only use cumulative path if we have at least a day's worth of slots;
            # otherwise fall back to recorder-based slot averaging to avoid misprofiling
            # sparse/non-cumulative data sources.
            try:
                slots_per_day = int(round(24 * 60 / period_minutes))
            except Exception:
                slots_per_day = 24 * SLOTS_PER_HOUR
            if total_qh is not None and not total_qh.empty and len(total_qh) < slots_per_day:
                total_qh = pd.Series(dtype=float)
            ev_sensor = self.settings.ev_energy_sensor or FALLBACK_EV_SENSOR
            ev_qh, ev_daily = self._load_ev_qh_and_daily(
                ev_sensor, history_start - timedelta(hours=24), tz_now, period_minutes, self.settings.timezone
            )
            pv_qh = self._load_cumulative_qh(self.settings.pv_total_sensor, history_start, tz_now, period_minutes)
            grid_buy_qh = self._load_cumulative_qh(self.settings.grid_buy_total_sensor, history_start, tz_now, period_minutes)
            grid_sell_qh = self._load_cumulative_qh(self.settings.grid_sell_total_sensor, history_start, tz_now, period_minutes)
            batt_in_qh = self._load_cumulative_qh(self.settings.battery_charge_total_sensor, history_start, tz_now, period_minutes)
            batt_out_qh = self._load_cumulative_qh(self.settings.battery_discharge_total_sensor, history_start, tz_now, period_minutes)

            if not total_qh.empty:
                hist = pd.DataFrame(index=total_qh.index)
                hist["house_total_kwh"] = total_qh.reindex(hist.index).fillna(0.0)
                hist["ev_kwh"] = ev_qh.reindex(hist.index).fillna(0.0)
                hist["pv_kwh"] = pv_qh.reindex(hist.index).fillna(0.0)
                hist["grid_buy_kwh"] = grid_buy_qh.reindex(hist.index).fillna(0.0)
                hist["grid_sell_kwh"] = grid_sell_qh.reindex(hist.index).fillna(0.0)
                hist["batt_in_kwh"] = batt_in_qh.reindex(hist.index).fillna(0.0)
                hist["batt_out_kwh"] = batt_out_qh.reindex(hist.index).fillna(0.0)
                hist["net_house_kw"] = (hist["house_total_kwh"] - hist["ev_kwh"]).clip(lower=0.0) / period_hours

                loc = hist.index.tz_convert(self.settings.timezone)
                # Profile: prefer 15-min slot×weekday when we have enough history, else hour×weekday
                try:
                    slot_index = (loc.hour * int(round(60 / period_minutes)) + (loc.minute // period_minutes)).astype(int)
                except Exception:
                    slot_index = (loc.hour * 4 + (loc.minute // 15)).astype(int)
                has_depth = len(hist) >= max(7 * int(round(24 * 60 / period_minutes)), 200)
                if has_depth:
                    key = pd.MultiIndex.from_arrays([loc.weekday, slot_index], names=["wday", "slot"])  # 15-min slot
                else:
                    key = pd.MultiIndex.from_arrays([loc.weekday, loc.hour], names=["wday", "hour"])  # hourly
                profile = hist.groupby(key)["net_house_kw"].mean()

                def _predict(local_ts: pd.Timestamp) -> float:
                    if has_depth:
                        sidx = int(local_ts.hour * int(round(60 / period_minutes)) + (local_ts.minute // period_minutes))
                        return float(profile.get((local_ts.weekday(), sidx), float(hist["net_house_kw"].mean())))
                    else:
                        return float(profile.get((local_ts.weekday(), local_ts.hour), float(hist["net_house_kw"].mean())))

                df["consumption_estimate_kw"] = [
                    _predict(pd.Timestamp(ts).tz_convert(self.settings.timezone)) for ts in df.index
                ]
                self.last_consumption_note = (
                    f"Consumption profile built from {len(hist)} QH deltas on cumulative sensors; "
                    f"EV subtraction: {'fallback' if (self.settings.ev_energy_sensor is None) else 'configured'}."
                )
                # Persist reconciliation artefacts for reporting/summary
                self.last_hist_quarter = hist
                daily_df = self._daily_reconciliation(hist, self.settings.timezone, period_minutes)
                self.last_daily_reconciliation = daily_df.round(3).to_dict("records")
                # Calibration: scale forecast per slot using recent observed net house load
                try:
                    cal_window_slots = 7 * int(round(24 * 60 / period_minutes))
                    recent = hist.tail(cal_window_slots)
                    if not recent.empty:
                        loc_recent = recent.index.tz_convert(self.settings.timezone)
                        slot_recent = (
                            loc_recent.hour * int(round(60 / period_minutes)) + (loc_recent.minute // period_minutes)
                        ).astype(int)
                        period_hours = max(period_minutes / 60.0, 1e-9)
                        obs_per_slot = (recent["house_total_kwh"] - recent["ev_kwh"]).clip(lower=0.0) / period_hours
                        obs_mean = obs_per_slot.groupby(slot_recent).mean()

                        # Raw profile mean per slot from current forecast
                        raw_profile: dict[int, list[float]] = {}
                        tz_name = self.settings.timezone
                        for ts, val in zip(df.index, df["consumption_estimate_kw" ]):
                            local_ts = pd.Timestamp(ts).tz_convert(tz_name)
                            sidx = int(local_ts.hour * int(round(60 / period_minutes)) + (local_ts.minute // period_minutes))
                            raw_profile.setdefault(sidx, []).append(float(val))
                        raw_mean = {s: (sum(vals) / max(1, len(vals))) for s, vals in raw_profile.items()}

                        # Compute clamped scale factors per slot
                        slots_per_day = int(round(24 * 60 / period_minutes))
                        scale = {}
                        for s in range(slots_per_day):
                            obs = float(obs_mean.get(s, float("nan")))
                            raw = float(raw_mean.get(s, float("nan")))
                            if not pd.isna(obs) and not pd.isna(raw) and raw > 1e-6:
                                scale[s] = max(0.5, min(1.5, obs / raw))
                            else:
                                scale[s] = 1.0

                        df["consumption_estimate_kw"] = [
                            float(val)
                            * scale[
                                int(
                                    pd.Timestamp(ts).tz_convert(self.settings.timezone).hour * int(round(60 / period_minutes))
                                    + (pd.Timestamp(ts).tz_convert(self.settings.timezone).minute // period_minutes)
                                )
                            ]
                            for ts, val in zip(df.index, df["consumption_estimate_kw"])
                        ]
                        df["consumption_calibration_note"] = (
                            f"Slot scaling applied (0.5–1.5 clamp) from last {min(len(recent), cal_window_slots)} QH points."
                        )
                except Exception:
                    # Calibration is optional; continue silently on failures
                    pass
                history_used = True

        # Fallback path: legacy recorder-based quarter-hour series and helpers
        if not history_used and self.settings.house_consumption_sensor:
            history_entries: List[tuple[pd.Timestamp, float]] = []
            history_window = max(self.settings.consumption_history_hours, self.settings.lookahead_hours)
            history_start = tz_now - timedelta(hours=history_window)

            # For recorder-based fallback, do not subtract EV by default; these sensors
            # may already reflect net house usage in some setups and unit tests expect
            # raw slot averages. Build EV lookup from ev_qh_sensor if configured.
            ev_kw_series = None
            ev_daily_kwh = None
            ev_sensor_used = None
            ev_fallback_used = False
            ev_lookup: Dict[pd.Timestamp, float] = {}
            ev_note = ""

            # Fetch EV history for subtraction if ev_qh_sensor is configured
            if self.settings.ev_qh_sensor:
                try:
                    ev_raw_history = self.ha.fetch_history_series(
                        self.settings.ev_qh_sensor,
                        history_start,
                        tz_now,
                    )
                    ev_entries: List[tuple[pd.Timestamp, float]] = []
                    for sample_ts, sample_state in ev_raw_history:
                        kw_val = _kwh_to_kw(sample_state)
                        if kw_val is None:
                            continue
                        localized_ts = ensure_timezone(sample_ts, self.settings.timezone)
                        ev_entries.append((pd.Timestamp(localized_ts), float(kw_val)))
                    
                    if ev_entries:
                        ev_df = pd.DataFrame(ev_entries, columns=["timestamp", "kw"])
                        ev_df.sort_values("timestamp", inplace=True)
                        ev_df["slot_ts"] = ev_df["timestamp"].dt.floor(freq_str)
                        # Build lookup: slot_ts -> average kW in that slot
                        ev_lookup = ev_df.groupby("slot_ts")["kw"].mean().to_dict()
                        ev_sensor_used = self.settings.ev_qh_sensor
                        logger.info(
                            "Built EV lookup from %s: %d slots with EV data",
                            self.settings.ev_qh_sensor,
                            len(ev_lookup),
                        )
                except Exception as exc:
                    logger.warning(
                        "Failed to fetch EV history from %s: %s",
                        self.settings.ev_qh_sensor,
                        exc,
                    )

            try:
                raw_history = self.ha.fetch_history_series(
                    self.settings.house_consumption_sensor,
                    history_start,
                    tz_now,
                )
            except Exception as exc:  # pragma: no cover - network failure path
                logger.warning(
                    "Failed to fetch Home Assistant history for %s: %s",
                    self.settings.house_consumption_sensor,
                    exc,
                )
                raw_history = []

            for sample_ts, sample_state in raw_history:
                kw_val = _kwh_to_kw(sample_state)
                if kw_val is None:
                    continue
                localized_ts = ensure_timezone(sample_ts, self.settings.timezone)
                history_entries.append((pd.Timestamp(localized_ts), float(kw_val)))

            if history_entries:
                history_df = self._prepare_history_frame(history_entries, freq_str, ev_lookup)
                slot_series = history_df.groupby("slot")["net_kw"].mean()
                fallback_mean = float(history_df["net_kw"].mean()) if not history_df["net_kw"].empty else 0.0
                slot_values: List[float] = []
                for ts in df.index:
                    key = ensure_timezone(ts, self.settings.timezone).strftime("%H:%M")
                    value = slot_series.get(key, fallback_mean)
                    if value is None or pd.isna(value):
                        value = fallback_mean
                    slot_values.append(float(value))
                df["consumption_estimate_kw"] = slot_values
                total_points = len(history_df)
                unique_slots = int(slot_series.count())
                ev_note_part = ""
                if ev_sensor_used and ev_lookup:
                    total_ev_kwh = sum(ev_lookup.values()) * (self.settings.resolution_minutes / 60.0)
                    ev_note_part = f" EV subtracted from {ev_sensor_used} (~{total_ev_kwh:.1f} kWh)."
                self.last_consumption_note = (
                    f"Consumption forecast derived from {total_points} recorder samples "
                    f"across {unique_slots} slots using {self.settings.house_consumption_sensor}.{ev_note_part}"
                )
                history_used = True

            consumption_state = self.ha.fetch_state(self.settings.house_consumption_sensor)
            attributes = consumption_state.get("attributes", {}) if isinstance(consumption_state, dict) else {}

            if not history_used:
                history_serialized = attributes.get("history_serialized")
                if isinstance(history_serialized, str) and history_serialized.strip():
                    parsed_entries: List[tuple[pd.Timestamp, float]] = []
                    for raw_item in history_serialized.split("|"):
                        if not raw_item:
                            continue
                        ts_part, sep, value_part = raw_item.partition(",")
                        if not sep:
                            continue
                        try:
                            ts = pd.Timestamp(ts_part)
                        except (TypeError, ValueError):
                            continue
                        kw_val = _kwh_to_kw(value_part)
                        if kw_val is None:
                            continue
                        localized_ts = ensure_timezone(ts.to_pydatetime(), self.settings.timezone)
                        parsed_entries.append((pd.Timestamp(localized_ts), float(kw_val)))

                    if parsed_entries:
                        history_df = self._prepare_history_frame(parsed_entries, freq_str, ev_lookup)
                        slot_series = history_df.groupby("slot")["net_kw"].mean()
                        fallback_mean = float(history_df["net_kw"].mean()) if not history_df["net_kw"].empty else 0.0
                        slot_values = []
                        for ts in df.index:
                            key = ensure_timezone(ts, self.settings.timezone).strftime("%H:%M")
                            value = slot_series.get(key, fallback_mean)
                            if value is None or pd.isna(value):
                                value = fallback_mean
                            slot_values.append(float(value))
                        df["consumption_estimate_kw"] = slot_values
                        total_points = len(history_df)
                        unique_slots = int(slot_series.count())
                        self.last_consumption_note = (
                            f"Consumption forecast derived from {total_points} historical quarter-hours "
                            f"across {unique_slots} slots using {self.settings.house_consumption_sensor} history_serialized.{ev_note}"
                        )
                        history_used = True

            if not history_used:
                series_attr = attributes.get("quarter_hour_kwh")
                if isinstance(series_attr, list):
                    series_entries: List[tuple[pd.Timestamp, float]] = []
                    for entry in series_attr:
                        ts_str = entry.get("timestamp")
                        val = _kwh_to_kw(entry.get("value"))
                        if not ts_str or val is None:
                            continue
                        ts = ensure_timezone(pd.Timestamp(ts_str).to_pydatetime(), self.settings.timezone)
                        series_entries.append((pd.Timestamp(ts), float(val)))

                    if series_entries:
                        history_df = self._prepare_history_frame(series_entries, freq_str, ev_lookup)
                        for _, row in history_df.iterrows():
                            df.loc[df.index == row["slot_ts"], "consumption_estimate_kw"] = row["net_kw"]
                        self.last_consumption_note = (
                            f"Consumption forecast seeded from recent list provided by {self.settings.house_consumption_sensor}.{ev_note}"
                        )
                        history_used = True
                else:
                    fallback_val = _kwh_to_kw(series_attr)
                    if fallback_val is not None:
                        df["consumption_estimate_kw"] = max(float(fallback_val), 0.0)
                        self.last_consumption_note = (
                            f"Using last quarter-hour reading from {self.settings.house_consumption_sensor} due to limited history."
                        )
                        history_used = True

        if "consumption_estimate_kw" not in df.columns:
            df["consumption_estimate_kw"] = float("nan")

        df["consumption_estimate_kw"] = df["consumption_estimate_kw"].astype(float)

        # As a robust fallback, if the series is missing or effectively zero, try to synthesize a
        # simple slot-average directly from recorder history one more time before using defaults.
        if (
            "consumption_estimate_kw" not in df.columns
            or df["consumption_estimate_kw"].isna().all()
            or (df["consumption_estimate_kw"].abs() < 1e-6).all()
        ) and self.settings.house_consumption_sensor:
            try:
                history_window = max(self.settings.consumption_history_hours, self.settings.lookahead_hours)
                history_start = tz_now - timedelta(hours=history_window)
                raw_history = self.ha.fetch_history_series(
                    self.settings.house_consumption_sensor,
                    history_start,
                    tz_now,
                )
                history_entries: List[tuple[pd.Timestamp, float]] = []
                for sample_ts, sample_state in raw_history:
                    kw_val = _kwh_to_kw(sample_state)
                    if kw_val is None:
                        continue
                    localized_ts = ensure_timezone(sample_ts, self.settings.timezone)
                    history_entries.append((pd.Timestamp(localized_ts), float(kw_val)))
                if history_entries:
                    hist_df = self._prepare_history_frame(history_entries, freq_str, {})
                    slot_series = hist_df.groupby("slot")["net_kw"].mean()
                    fallback_mean = float(hist_df["net_kw"].mean()) if not hist_df["net_kw"].empty else 0.0
                    slot_values: List[float] = []
                    for ts in df.index:
                        key = ensure_timezone(ts, self.settings.timezone).strftime("%H:%M")
                        value = slot_series.get(key, fallback_mean)
                        if value is None or pd.isna(value):
                            value = fallback_mean
                        slot_values.append(float(value))
                    df["consumption_estimate_kw"] = slot_values
                    total_points = len(hist_df)
                    unique_slots = int(slot_series.count())
                    self.last_consumption_note = (
                        f"Consumption forecast derived from {total_points} recorder samples "
                        f"across {unique_slots} slots using {self.settings.house_consumption_sensor}."
                    )
            except Exception:
                pass

        if (
            "consumption_estimate_kw" not in df.columns
            or df["consumption_estimate_kw"].isna().all()
            or (df["consumption_estimate_kw"].abs() < 1e-6).all()
        ):
            df["consumption_estimate_kw"] = self._default_consumption_series(df.index)
            self.last_consumption_note = (
                "Default consumption profile (0.4–1.2 kWh/h) applied due to missing Home Assistant data."
            )
        df = df.ffill().fillna(0.0).infer_objects(copy=False)
        df["created_at"] = datetime.now(timezone.utc)
        df = df.reset_index(names="timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
        
        # CRITICAL: Verify consumption forecast has reasonable values
        zero_count = (df["consumption_estimate_kw"] == 0).sum()
        if zero_count > 4:
            logger.error(
                "❌ HOUSE LOAD FORECAST BROKEN – %d slots with ZERO consumption detected! "
                "Sample timestamps: %s",
                zero_count,
                df[df["consumption_estimate_kw"] == 0]["timestamp"].head(10).tolist(),
            )
            # Log sample of the data for debugging
            logger.error("Consumption forecast sample:\n%s", df[["timestamp", "consumption_estimate_kw"]].head(20))
            raise ValueError(
                f"❌ HOUSE LOAD FORECAST BROKEN – {zero_count} slots with ZERO consumption detected. "
                f"Check consumption sensor configuration and data availability. "
                f"Last consumption note: {self.last_consumption_note}"
            )
        
        # Log consumption forecast summary for visibility
        logger.info(
            "Consumption forecast: min=%.3f, max=%.3f, mean=%.3f kW | Zero slots: %d/%d | Note: %s",
            df["consumption_estimate_kw"].min(),
            df["consumption_estimate_kw"].max(),
            df["consumption_estimate_kw"].mean(),
            zero_count,
            len(df),
            self.last_consumption_note or "N/A",
        )
        
        return df

    def fetch_and_record_actuals(self, hours_back: int = 6) -> None:
        """Fetch recent history from HA and persist to actual_quarter_hour table."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=hours_back)
        period_minutes = self.settings.resolution_minutes
        period_hours = max(period_minutes / 60.0, 1e-9)
        
        # 1. Fetch cumulative flows (kWh -> kW)
        # We use _load_cumulative_qh which returns kWh per slot
        pv_qh = self._load_cumulative_qh(self.settings.pv_total_sensor, start, now, period_minutes)
        house_qh = self._load_cumulative_qh(self.settings.house_total_sensor, start, now, period_minutes)
        grid_buy_qh = self._load_cumulative_qh(self.settings.grid_buy_total_sensor, start, now, period_minutes)
        grid_sell_qh = self._load_cumulative_qh(self.settings.grid_sell_total_sensor, start, now, period_minutes)
        
        # EV subtraction for house load
        ev_qh = pd.Series(dtype=float)
        if self.settings.ev_energy_sensor:
             ev_qh = self._load_cumulative_qh(self.settings.ev_energy_sensor, start, now, period_minutes)
        
        # 2. Fetch state sensors (SoC) -> Resample to mean
        def _fetch_state_mean(sensor: Optional[str]) -> pd.Series:
            if not sensor:
                return pd.Series(dtype=float)
            try:
                samples = self.ha.fetch_history_series(sensor, start, now)
                if not samples:
                    return pd.Series(dtype=float)
                df = pd.DataFrame(samples, columns=["timestamp", "value"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                df.dropna(inplace=True)
                df.set_index("timestamp", inplace=True)
                freq = f"{period_minutes}min"
                return df["value"].resample(freq).mean()
            except Exception as e:
                logger.warning(f"Failed to fetch state history for {sensor}: {e}")
                return pd.Series(dtype=float)

        batt_soc_series = _fetch_state_mean(self.settings.battery_soc_sensor)
        ev_soc_series = _fetch_state_mean(self.settings.ev_soc_sensor)

        # 3. Align everything
        # Create a common index
        all_indices = pv_qh.index.union(house_qh.index).union(grid_buy_qh.index)
        if all_indices.empty:
            return

        df = pd.DataFrame(index=all_indices)
        
        # Fill flows (convert kWh -> kW)
        df["production_kw"] = pv_qh.reindex(df.index).fillna(0.0) / period_hours
        
        # House load = Total - EV
        raw_house = house_qh.reindex(df.index).fillna(0.0)
        raw_ev = ev_qh.reindex(df.index).fillna(0.0) if not ev_qh.empty else 0.0
        df["consumption_kw"] = (raw_house - raw_ev).clip(lower=0.0) / period_hours
        
        df["grid_import_kw"] = grid_buy_qh.reindex(df.index).fillna(0.0) / period_hours
        df["grid_export_kw"] = grid_sell_qh.reindex(df.index).fillna(0.0) / period_hours
        
        # Fill states
        batt_cap = BATTERY_CAPACITY_KWH
        
        # Helper to convert % to kWh if needed
        def _to_kwh(val, cap):
            if cap > 0 and val <= 100.0:
                 return (val / 100.0) * cap
            return val

        batt_vals = batt_soc_series.reindex(df.index).ffill()
        if batt_cap > 0:
             batt_vals = batt_vals.apply(lambda x: _to_kwh(x, batt_cap))
        df["battery_soc_kw"] = batt_vals
        
        ev_cap = EV_BATTERY_CAPACITY_KWH
        ev_vals = ev_soc_series.reindex(df.index).ffill()
        if ev_cap > 0:
             ev_vals = ev_vals.apply(lambda x: _to_kwh(x, ev_cap))
        df["ev_soc_kw"] = ev_vals

        # 4. Persist
        df = df.dropna(how="all")
        df = df.reset_index().rename(columns={"index": "timestamp"})
        
        records = df.to_dict(orient="records")
        if records:
            self.record_actuals(records)
            print(f"DEBUG: Recorded {len(records)} actuals from history (last {hours_back}h)")
            logger.info(f"Recorded {len(records)} actuals from history (last {hours_back}h)")
        else:
            print("DEBUG: No actuals records found to persist.")

    def persist_forecast(self, frame: pd.DataFrame) -> None:
        normalized = frame.copy()

        def _normalize_ts(value):
            if isinstance(value, pd.Timestamp):
                return to_utc_naive(value.to_pydatetime())
            if isinstance(value, datetime):
                return to_utc_naive(value)
            return to_utc_naive(pd.Timestamp(value).to_pydatetime())

        normalized["timestamp"] = normalized["timestamp"].apply(_normalize_ts)
        if "created_at" in normalized.columns:

            def _normalize_created(value):
                if value is None:
                    return to_utc_naive(datetime.now(timezone.utc))
                if isinstance(value, pd.Timestamp):
                    return to_utc_naive(value.to_pydatetime())
                if isinstance(value, datetime):
                    return to_utc_naive(value)
                return to_utc_naive(pd.Timestamp(value).to_pydatetime())

            normalized["created_at"] = normalized["created_at"].apply(_normalize_created)
        records = normalized.to_dict(orient="records")
        with session_scope(self.SessionFactory) as session:
            for record in records:
                stmt = insert(ForecastQuarterHour).values(
                    timestamp=record["timestamp"],
                    production_kw=record.get("pv_forecast_kw", 0.0),
                    consumption_kw=record.get("consumption_estimate_kw"),
                    price_buy=record.get("price_buy", 0.0),
                    price_sell=record.get("price_sell"),
                    created_at=record.get("created_at"),
                )
                update_cols = {
                    "production_kw": stmt.inserted.production_kw,
                    "consumption_kw": stmt.inserted.consumption_kw,
                    "price_buy": stmt.inserted.price_buy,
                    "price_sell": stmt.inserted.price_sell,
                }
                session.execute(stmt.on_duplicate_key_update(**update_cols))

    def record_actuals(self, actual_rows: Iterable[dict]) -> None:
        normalized_rows = []

        for row in actual_rows:
            copy_row = dict(row)
            ts_val = copy_row.get("timestamp")
            if ts_val is not None:
                if isinstance(ts_val, pd.Timestamp):
                    copy_row["timestamp"] = to_utc_naive(ts_val.to_pydatetime())
                elif isinstance(ts_val, datetime):
                    copy_row["timestamp"] = to_utc_naive(ts_val)
                else:
                    copy_row["timestamp"] = to_utc_naive(pd.Timestamp(ts_val).to_pydatetime())
            
            # Replace NaN with None for SQL compatibility
            for k, v in copy_row.items():
                if isinstance(v, float) and np.isnan(v):
                    copy_row[k] = None
                    
            normalized_rows.append(copy_row)

        with session_scope(self.SessionFactory) as session:
            for row in normalized_rows:
                stmt = insert(ActualQuarterHour).values(**row)
                update_cols = {
                    "production_kw": stmt.inserted.production_kw,
                    "consumption_kw": stmt.inserted.consumption_kw,
                    "grid_import_kw": stmt.inserted.grid_import_kw,
                    "grid_export_kw": stmt.inserted.grid_export_kw,
                    "battery_soc_kw": stmt.inserted.battery_soc_kw,
                    "ev_soc_kw": stmt.inserted.ev_soc_kw,
                }
                session.execute(stmt.on_duplicate_key_update(**update_cols))


__all__ = ["DataPipeline"]
