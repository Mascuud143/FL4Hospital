from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from persistence.database import session_scope
from persistence.models.utility_usage import UtilityUsage
from simulation_batch.csv_filestorage import write_model_rows

_UTILITY_USAGE_BUFFER: list[tuple[int, str, datetime, datetime, Optional[float], Optional[float], Optional[int]]] = []
_WRITE_BATCH_SIZE = 50000


def _flush_buffer() -> None:
    global _UTILITY_USAGE_BUFFER
    if not _UTILITY_USAGE_BUFFER:
        return
    batch = _UTILITY_USAGE_BUFFER
    _UTILITY_USAGE_BUFFER = []
    serialized_batch = []
    for room_id, category, start_time, end_time, power_kwh, water_liters, device_id in batch:
        row = {
            "room_id": room_id,
            "category": category,
            "start_time": start_time,
            "end_time": end_time,
            "power_consumption": power_kwh,
            "water_consumption": water_liters,
        }
        if device_id is not None and hasattr(UtilityUsage, "device_id"):
            row["device_id"] = int(device_id)
        serialized_batch.append(row)
    write_model_rows(UtilityUsage, serialized_batch)
    with session_scope() as session:
        session.bulk_insert_mappings(UtilityUsage, serialized_batch)


def flush_utility_usage_writes() -> None:
    _flush_buffer()


def insert_utility_usage(
    *,
    room_id: int,
    category: str,
    start_time: datetime,
    end_time: datetime,
    power_kwh: Optional[float] = None,
    water_liters: Optional[float] = None,
    device_id: Optional[int] = None,
) -> None:
    """
    Insert a UtilityUsage row.

    Works in BOTH cases:
      - UtilityUsage has a device_id column
      - UtilityUsage does NOT have a device_id column (we skip it safely)

    category examples:
      - "hvac"
      - "toilet_heater"
      - "toilet_light"
      - "water"
    """
    start_time_utc = start_time.astimezone(timezone.utc)
    end_time_utc = end_time.astimezone(timezone.utc)

    _UTILITY_USAGE_BUFFER.append(
        (
            int(room_id),
            str(category),
            start_time_utc,
            end_time_utc,
            power_kwh,
            water_liters,
            int(device_id) if device_id is not None else None,
        )
    )
    if len(_UTILITY_USAGE_BUFFER) >= _WRITE_BATCH_SIZE:
        _flush_buffer()
