from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from persistence.database import session_scope
from persistence.models.device import Device as DeviceModel
from persistence.models.sensor import Sensor as SensorModel

from .room_engine import RoomEngine


@dataclass
class SensorEmitRow:
    device_id: int
    room_id: int
    location: str           # "main" | "toilet"
    device_type: str        # should be "sensor"
    sensor_type: str        # temperature/humidity/co2/light/sound
    uuid: Optional[str]
    unit: str


class SensorSampler:
    """
    Discovers sensor devices & sensors from DB, then emits readings from RoomEngine state.
    """

    def __init__(self, *, seed: int = 999):
        self.rng = random.Random(seed)
        self.emit_map: List[SensorEmitRow] = []

    def build_emit_map_from_db(self) -> None:
        self.emit_map.clear()
        with session_scope() as session:
            sensor_devices = (
                session.query(DeviceModel)
                .filter(DeviceModel.device_type == "sensor")
                .all()
            )

            for dev in sensor_devices:
                if dev.room_id is None:
                    continue

                loc = (dev.location or "main").lower()

                sensors = (
                    session.query(SensorModel)
                    .filter(SensorModel.device_id == dev.device_id)
                    .all()
                )

                for s in sensors:
                    st = (getattr(s, "sensor_type", "") or "").lower()
                    if not st:
                        continue

                    # enforce your rule set
                    if loc == "main":
                        if st not in {"temperature", "humidity", "co2", "light", "sound"}:
                            continue
                    elif loc == "toilet":
                        if st != "temperature":
                            continue
                    else:
                        continue

                    self.emit_map.append(
                        SensorEmitRow(
                            device_id=dev.device_id,
                            room_id=dev.room_id,
                            location=loc,
                            device_type=dev.device_type,
                            sensor_type=st,
                            uuid=getattr(s, "uuid", None),
                            unit=getattr(s, "unit", "") or "",
                        )
                    )

    async def emit(self, now: datetime, *, room_engine: RoomEngine, on_event) -> None:
        ts = now.isoformat()

        for row in self.emit_map:
            rs = room_engine.rooms.get(row.room_id)
            if rs is None:
                continue

            if row.location == "toilet":
                # toilet: temperature only
                value = round(rs.toilet.temperature + self.rng.gauss(0.0, 0.05), 2)
            else:
                if row.sensor_type == "temperature":
                    value = round(rs.main.temperature + self.rng.gauss(0.0, 0.05), 2)
                elif row.sensor_type == "humidity":
                    value = round(rs.main.humidity + self.rng.gauss(0.0, 0.25), 1)
                elif row.sensor_type == "co2":
                    value = round(rs.main.co2 + self.rng.gauss(0.0, 8.0), 0)
                elif row.sensor_type == "light":
                    value = round(rs.main.light + self.rng.gauss(0.0, 2.0), 1)
                elif row.sensor_type == "sound":
                    value = round(rs.main.sound + self.rng.gauss(0.0, 0.6), 1)
                else:
                    continue

            event: Dict[str, Any] = {
                "timestamp": ts,
                "device_id": row.device_id,
                "room_id": row.room_id,
                "location": row.location,
                "device_type": row.device_type,
                "sensor_type": row.sensor_type,
                "unit": row.unit,
                "uuid": row.uuid,
                "value": value,
                "raw_hex": None,
                "error": None,
            }

            await on_event(event)
