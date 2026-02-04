from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional
import math


@dataclass
class ProcessingResult:
    ok: bool
    event: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None


class DataProcessor:
    """
    Cleans, validates, and normalizes incoming events.

    Expected input event (from BLE layer):
      {
        "timestamp": "...",   # ISO string or datetime
        "mac": "XX:XX:..",
        "device_label": "Device 1",
        "sensor_type": "temperature",
        "unit": "°C",
        "uuid": "...",
        "value": 23.5,
        "raw_hex": "...",
        "error": None | "..."
      }
    """

    def __init__(self) -> None:
        self._ranges = {
            "temperature": (-40.0, 85.0),
            "humidity": (0.0, 100.0),
        }

    def process(self, event: Dict[str, Any]) -> ProcessingResult:
        # Upstream parse error
        if event.get("error"):
            return ProcessingResult(ok=False, reason=f"upstream_error: {event['error']}")

        sensor_type = str(event.get("sensor_type") or "").strip().lower()
        if not sensor_type:
            return ProcessingResult(ok=False, reason="missing_sensor_type")

        # Normalize timestamp
        event["timestamp"] = self._normalize_timestamp(event.get("timestamp"))

        # Normalize MAC
        if event.get("mac"):
            event["mac"] = str(event["mac"]).upper()

        value = event.get("value", None)

        # Allow dict values (e.g., gas readings)
        if isinstance(value, dict):
            return ProcessingResult(ok=True, event=event)

        if value is None:
            return ProcessingResult(ok=False, reason="value_none")

        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return ProcessingResult(ok=False, reason="value_not_numeric")

        if math.isnan(value_f) or math.isinf(value_f):
            return ProcessingResult(ok=False, reason="value_nan_or_inf")

        # Range check if known
        if sensor_type in self._ranges:
            lo, hi = self._ranges[sensor_type]
            if not (lo <= value_f <= hi):
                return ProcessingResult(ok=False, reason=f"out_of_range:{lo}..{hi}")

        # Rounding policy
        if sensor_type in ("temperature", "humidity"):
            value_f = round(value_f, 2)

        event["value"] = value_f
        return ProcessingResult(ok=True, event=event)

    @staticmethod
    def _normalize_timestamp(ts: Any) -> str:
        if ts is None:
            return datetime.utcnow().isoformat(timespec="seconds")
        if isinstance(ts, datetime):
            return ts.isoformat()
        return str(ts)
