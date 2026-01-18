"""Client for interacting with Home Assistant REST API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests  # type: ignore
from dateutil import parser  # type: ignore


@dataclass(slots=True)
class PricePoint:
    starts_at: datetime
    value: float


@dataclass(slots=True)
class ProductionPoint:
    timestamp: datetime
    pv_estimate_kw: float


@dataclass(slots=True)
class ConsumptionPoint:
    timestamp: datetime
    consumption_kw: float


class HomeAssistantClient:
    def __init__(self, base_url: str, api_key: str, timeout: int = 15) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    def _get(self, path: str) -> Dict:
        url = f"{self.base_url}{path}"
        response = self.session.get(url, timeout=self.timeout)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return {}
            raise
        try:
            return response.json()
        except ValueError:
            return {}

    def _post(self, path: str, json: Optional[Dict[str, Any]] = None) -> Dict:
        url = f"{self.base_url}{path}"
        response = self.session.post(url, json=json or {}, timeout=self.timeout)
        response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            return {}

    def fetch_state(self, entity_id: str) -> Dict:
        return self._get(f"/states/{entity_id}")

    def fetch_history_series(
        self,
        entity_id: str,
        start: datetime,
        end: Optional[datetime] = None,
        minimal_response: bool = True,
    ) -> List[Tuple[datetime, float]]:
        """Return history samples for ``entity_id`` between ``start`` and ``end``.

        Values are returned as a list of ``(timestamp, state)`` tuples. The ``state``
        is parsed as ``float``; any entries that cannot be parsed are skipped.
        """

        params: Dict[str, Any] = {
            "filter_entity_id": entity_id,
            "significant_changes_only": "false",
            "no_attributes": "true",
        }
        if minimal_response:
            params["minimal_response"] = "true"
        if end is not None:
            params["end_time"] = end.astimezone(timezone.utc).isoformat()

        start_iso = start.astimezone(timezone.utc).isoformat()
        url = f"{self.base_url}/history/period/{start_iso}"
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()

        payload = response.json()
        samples: List[Tuple[datetime, float]] = []
        if isinstance(payload, list):
            for entity_entries in payload:
                if not isinstance(entity_entries, list):
                    continue
                for item in entity_entries:
                    if not isinstance(item, dict):
                        continue
                    ts_raw = item.get("last_changed") or item.get("last_updated") or item.get("time")
                    state_raw = item.get("state")
                    if ts_raw is None or state_raw is None:
                        continue
                    try:
                        ts = parser.isoparse(ts_raw)
                        value = float(state_raw)
                    except (TypeError, ValueError):
                        continue
                    samples.append((ts, value))
        return samples

    def fetch_price_series(self, sensor_id: str) -> List[PricePoint]:
        payload = self.fetch_state(sensor_id)
        attrs = payload.get("attributes", {})
        all_points: List[PricePoint] = []
        # Process forecast first, then raw_today/tomorrow so that confirmed spot prices
        # (which often have higher resolution or certainty) overwrite forecast values
        # when creating the map in data_pipeline.
        for field in ("forecast", "raw_today", "raw_tomorrow"):
            entries: Iterable[Dict] = attrs.get(field, []) or []
            for entry in entries:
                ts_raw = entry.get("start") or entry.get("datetime") or entry.get("hour")
                price = entry.get("price") if entry.get("price") is not None else entry.get("value")
                if ts_raw is None or price is None:
                    continue
                ts = parser.isoparse(ts_raw)
                all_points.append(PricePoint(starts_at=ts, value=float(price)))
        return sorted(all_points, key=lambda p: p.starts_at)

    def fetch_solcast_forecast(self, sensor_ids: Iterable[str]) -> List[ProductionPoint]:
        forecasts: List[ProductionPoint] = []
        for sensor in sensor_ids:
            payload = self.fetch_state(sensor)
            details = payload.get("attributes", {}).get("detailedHourly", []) or []
            for item in details:
                ts_raw = item.get("period_end") or item.get("period_start")
                pv_estimate = item.get("pv_estimate")
                if ts_raw is None or pv_estimate is None:
                    continue
                ts = parser.isoparse(ts_raw)
                forecasts.append(ProductionPoint(timestamp=ts, pv_estimate_kw=float(pv_estimate)))
        forecasts.sort(key=lambda p: p.timestamp)
        return forecasts

    def fetch_numeric_state(self, entity_id: str) -> Optional[float]:
        payload = self.fetch_state(entity_id)
        state = payload.get("state")
        try:
            return float(state)
        except (TypeError, ValueError):
            return None

    def fetch_string_state(self, entity_id: str) -> Optional[str]:
        payload = self.fetch_state(entity_id)
        state = payload.get("state")
        return str(state) if state is not None else None

    def list_services(self, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return available Home Assistant services as normalized entries.

        Each returned item contains at least ``domain`` and ``service`` keys.
        When ``domain`` is provided, restrict the response to that domain.
        """

        def _normalize(domain_name: Optional[str], services: Any) -> List[Dict[str, Any]]:
            entries: List[Dict[str, Any]] = []
            if isinstance(services, dict):
                for service_name, payload in services.items():
                    entry: Dict[str, Any] = {
                        "domain": domain_name,
                        "service": service_name,
                    }
                    if isinstance(payload, dict):
                        entry.update(payload)
                    else:
                        entry["value"] = payload
                    entries.append(entry)
            elif isinstance(services, list):
                for item in services:
                    if isinstance(item, dict):
                        entry = dict(item)
                        entry.setdefault("domain", domain_name)
                        entries.append(entry)
            return entries

        data = self._get("/services")

        normalized: List[Dict[str, Any]] = []
        if isinstance(data, list):
            for item in data:
                domain_name = item.get("domain") if isinstance(item, dict) else None
                services = item.get("services") if isinstance(item, dict) else None
                normalized.extend(_normalize(domain_name, services))

        if domain is not None:
            normalized = [entry for entry in normalized if entry.get("domain") == domain]

        return normalized

    def call_service(self, domain: str, service: str, data: Optional[Dict[str, Any]] = None) -> Dict:
        """Call a Home Assistant service via REST API.

        Example: call_service("homeassistant", "update_entity", {"entity_id": "sensor.energy_plan"})
        """
        domain = (domain or "").strip()
        service = (service or "").strip()
        if not domain or not service:
            raise ValueError("Both domain and service are required to call a HA service")
        return self._post(f"/services/{domain}/{service}", json=data or {})

    # --- Weekly EV inputs helpers ---
    def fetch_weekly_ev_kwh_inputs(self) -> Dict[str, float]:
        """Return weekly EV expected energy inputs from HA inputs as a dict.

        Keys: mon,tue,wed,thu,fri,sat,sun. Values are floats (kWh), unknown/unavailable -> 0.0.
        """
        result: Dict[str, float] = {k: 0.0 for k in ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]}
        for key in list(result.keys()):
            entity_id = f"input_number.energy_planner_ev_week_{key}_kwh"
            try:
                value = self.fetch_numeric_state(entity_id)
                result[key] = float(value) if (value is not None and value == value) else 0.0  # NaN-safe
            except Exception:
                result[key] = 0.0
        return result

    def weekly_ev_entity_id(self, weekday_key: str) -> str:
        """Utility to build the entity_id for a weekday key (mon..sun)."""
        return f"input_number.energy_planner_ev_week_{weekday_key}_kwh"

    def fetch_weekly_ev_departure_pct_inputs(self) -> Dict[str, Optional[float]]:
        """Return per-weekday EV departure charge limits (percentages)."""
        result: Dict[str, Optional[float]] = {k: None for k in ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]}
        for key in list(result.keys()):
            entity_id = f"input_number.energy_planner_ev_departure_{key}_pct"
            try:
                value = self.fetch_numeric_state(entity_id)
                if value is None or value != value:  # NaN-safe
                    result[key] = None
                else:
                    result[key] = max(0.0, min(100.0, float(value)))
            except Exception:
                result[key] = None
        return result


__all__ = [
    "HomeAssistantClient",
    "PricePoint",
    "ProductionPoint",
    "ConsumptionPoint",
]
