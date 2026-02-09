from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

from .sensor import Sensor


@dataclass
class Device:
    """
    Runtime representation of a BLE device.
    """
    mac_address: str
    name: Optional[str] = None
    room_id: Optional[int] = None
    label: Optional[str] = None  
    sensors: List[Sensor] = field(default_factory=list)

    def add_sensor(self, sensor: Sensor) -> None:
        self.sensors.append(sensor)

    def get_sensor_by_uuid(self, uuid: str) -> Optional[Sensor]:
        for s in self.sensors:
            if s.uuid.lower() == uuid.lower():
                return s
        return None
