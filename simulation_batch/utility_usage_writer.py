from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from persistence.database import session_scope
from persistence.models.utility_usage import UtilityUsage
from simulation_batch.csv_filestorage import write_model_row
from simulation_batch.config import ENABLE_UTILITY_USAGE


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
    if not ENABLE_UTILITY_USAGE:
        return
    start_time_utc = start_time.astimezone(timezone.utc)
    end_time_utc = end_time.astimezone(timezone.utc)

    kwargs = dict(
        room_id=room_id,
        category=category,
        start_time=start_time_utc,
        end_time=end_time_utc,
        power_consumption=power_kwh,
        water_consumption=water_liters,
    )

    # Only set device_id if the model actually has that column
    if device_id is not None and hasattr(UtilityUsage, "device_id"):
        kwargs["device_id"] = int(device_id)

    with session_scope() as session:
        row = UtilityUsage(**kwargs)
        write_model_row(row)
        session.add(row)
