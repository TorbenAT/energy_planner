"""Forecasting module using Linear Regression and Holidays."""

import logging
from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np
import pandas as pd
from sqlalchemy import select, text
import holidays

from .models import ActualQuarterHour
from .db import session_scope
from .constants import SLOTS_PER_HOUR, DEFAULT_RESOLUTION_MINUTES
from .utils.time import ensure_timezone

logger = logging.getLogger(__name__)

# Try to import scikit-learn, otherwise fall back to custom implementation
try:
    from sklearn.linear_model import LinearRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    class LinearRegression:
        """A simple implementation of Linear Regression using numpy."""
        def __init__(self):
            self.coef_ = None
            self.intercept_ = None

        def fit(self, X, y):
            # Add column of ones for intercept
            X_b = np.c_[np.ones((len(X), 1)), X]
            # Use lstsq for robust solution
            theta_best, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=None)
            self.intercept_ = theta_best[0]
            self.coef_ = theta_best[1:]

        def predict(self, X):
            X_b = np.c_[np.ones((len(X), 1)), X]
            theta = np.r_[self.intercept_, self.coef_]
            return X_b.dot(theta)

        def score(self, X, y):
            """Calculate R^2 score."""
            y_pred = self.predict(X)
            u = ((y - y_pred) ** 2).sum()
            v = ((y - np.mean(y)) ** 2).sum()
            return 1 - u/v

class ConsumptionForecaster:
    def __init__(self, timezone_str: str = "Europe/Copenhagen"):
        self.timezone = timezone_str
        self.model = LinearRegression()
        self.is_trained = False
        self.dk_holidays = holidays.DK()
        
        if HAS_SKLEARN:
            logger.info("Using scikit-learn for forecasting.")
        else:
            logger.info("Using internal numpy-based LinearRegression (scikit-learn not found).")

    def _get_features(self, dt: datetime) -> List[float]:
        """Extract features from a datetime object."""
        # Cyclic hour
        hour_sin = np.sin(2 * np.pi * dt.hour / 24)
        hour_cos = np.cos(2 * np.pi * dt.hour / 24)
        
        # Cyclic day of week
        dow_sin = np.sin(2 * np.pi * dt.weekday() / 7)
        dow_cos = np.cos(2 * np.pi * dt.weekday() / 7)
        
        is_holiday = 1.0 if dt in self.dk_holidays else 0.0
        is_weekend = 1.0 if dt.weekday() >= 5 else 0.0
        
        return [hour_sin, hour_cos, dow_sin, dow_cos, is_holiday, is_weekend]

    def train(self, SessionFactory, days_history: int = 60, ha_client=None, settings=None) -> None:
        """Train the model on historical data.
        
        If DB is empty, attempts to backfill from Home Assistant if ha_client is provided.
        """
        start_time = datetime.utcnow() - timedelta(days=days_history)
        
        try:
            with session_scope(SessionFactory) as session:
                # 1. Fetch existing timestamps using raw SQL to be sure
                # This avoids potential ORM mapping issues or stale session state
                stmt = text("SELECT timestamp FROM energy_planner.actual_quarter_hour WHERE timestamp >= :start_time")
                existing_rows = session.execute(stmt, {"start_time": start_time}).fetchall()
                existing_timestamps = {row[0] for row in existing_rows}
                
                # 2. Check if we need to backfill
                # If we have very little data (e.g. < 1 day) and HA client is available
                if len(existing_timestamps) < (24 * SLOTS_PER_HOUR) and ha_client and settings:
                    logger.info(f"Insufficient DB history ({len(existing_timestamps)} records). Attempting backfill from Home Assistant...")
                    
                    # Prefer cumulative total sensor if available, else rate sensor
                    sensor_id = settings.house_total_sensor or settings.house_consumption_sensor
                    is_cumulative = bool(settings.house_total_sensor)
                    
                    if sensor_id:
                        try:
                            history = ha_client.fetch_history_series(
                                sensor_id,
                                start_time,
                                datetime.utcnow()
                            )
                            
                            if history:
                                # Convert to DataFrame for resampling
                                df = pd.DataFrame(history, columns=['timestamp', 'value'])
                                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                                df.set_index('timestamp', inplace=True)
                                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                                df.dropna(inplace=True)
                                
                                # Resample to 15min
                                if is_cumulative:
                                    # For cumulative, calculate diffs on raw data first
                                    delta = df["value"].diff()
                                    resets = delta < 0
                                    delta.loc[resets] = df.loc[resets, "value"]
                                    delta = delta.fillna(0.0).clip(lower=0.0)
                                    
                                    # Sum of deltas = kWh per slot
                                    freq = f"{DEFAULT_RESOLUTION_MINUTES}min"
                                    qh_kwh = df.assign(delta=delta).resample(freq, label='right', closed='right')['delta'].sum()
                                    
                                    # Convert kWh to kW (average power)
                                    qh_val = qh_kwh * float(SLOTS_PER_HOUR)
                                else:
                                    # For power/rate, just take the mean
                                    freq = f"{DEFAULT_RESOLUTION_MINUTES}min"
                                    qh_val = df['value'].resample(freq, label='right', closed='right').mean()
                                
                                count = 0
                                new_records = []
                                
                                for ts, val in qh_val.items():
                                    if pd.isna(val):
                                        continue
                                        
                                    # Convert timestamp to naive UTC
                                    ts_naive = ts.to_pydatetime().replace(tzinfo=None)
                                    
                                    if ts_naive not in existing_timestamps:
                                        record = ActualQuarterHour(
                                            timestamp=ts_naive,
                                            consumption_kw=float(val),
                                            production_kw=0.0,
                                            grid_import_kw=0.0,
                                            grid_export_kw=0.0
                                        )
                                        new_records.append(record)
                                        existing_timestamps.add(ts_naive)
                                        count += 1
                                
                                if new_records:
                                    session.add_all(new_records)
                                    session.commit()
                                    logger.info(f"Backfilled {count} new records from HA (resampled).")
                                else:
                                    logger.info("No new records to backfill.")
                            else:
                                logger.warning(f"No history found for sensor {sensor_id}")
                            
                        except Exception as e:
                            logger.warning(f"Backfill failed: {e}")
                            session.rollback()
                    else:
                        logger.warning("No consumption sensor configured for backfill.")

            # 3. Fetch full objects for training (now including backfilled data)
            training_data = []
            with session_scope(SessionFactory) as session:
                stmt = select(ActualQuarterHour).where(ActualQuarterHour.timestamp >= start_time)
                results = session.execute(stmt).scalars().all()
                
                # Extract data inside session scope to avoid DetachedInstanceError
                for record in results:
                    training_data.append({
                        'timestamp': record.timestamp,
                        'consumption_kw': record.consumption_kw
                    })

            if not training_data:
                logger.warning("No historical data found for training forecast model.")
                return

            X = []
            y = []
            
            for item in training_data:
                # Assuming timestamp is naive UTC in DB
                ts = item['timestamp']
                # Convert to local time for feature extraction
                ts_local = ensure_timezone(ts, self.timezone)
                
                features = self._get_features(ts_local)
                X.append(features)
                y.append(item['consumption_kw'])
                
            if len(X) < 100:
                 logger.warning(f"Insufficient data points ({len(X)}) for robust training.")
                 return

            self.model.fit(X, y)
            self.is_trained = True
            score = self.model.score(X, y)
            logger.info(f"Consumption forecast model trained on {len(X)} points. R2 score: {score:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to train forecast model: {e}")

    def predict(self, timestamps: List[datetime]) -> List[float]:
        """Predict consumption for a list of timestamps."""
        if not self.is_trained:
            logger.warning("Model not trained. Returning zeros.")
            return [0.0] * len(timestamps)
            
        X_pred = []
        for ts in timestamps:
            ts_local = ensure_timezone(ts, self.timezone)
            X_pred.append(self._get_features(ts_local))
            
        predictions = self.model.predict(X_pred)
        # Clip negative values
        return [max(0.0, float(p)) for p in predictions]
