from __future__ import annotations

# Load sensor data.
# - Stores one sensor record - SensorRuntime
# - Reads sensor ids, room ids, and sensor types
# - Builds the sensor list - load_sensor_registry()

from dataclasses import dataclass
from typing import List

from persistence.database import session_scope
from persistence.models.device import Device as DeviceModel
from persistence.models.sensor import Sensor as SensorModel


@dataclass(frozen=True)
class SensorRuntime:
    sensor_id: int
    device_id: int
    room_id: int
    mac: str
    location: str
    sensor_type: str
    uuid: str


def load_sensor_registry() -> list[SensorRuntime]:
    sensors: List[SensorRuntime] = []
    with session_scope() as session:
        rows = (
            session.query(
                SensorModel.sensor_id,
                SensorModel.sensor_type,
                SensorModel.uuid,
                DeviceModel.device_id,
                DeviceModel.room_id,
                DeviceModel.mac_address,
                DeviceModel.location,
            )
            .join(DeviceModel, SensorModel.device_id == DeviceModel.device_id)
            .all()
        )
        for row in rows:
            sensors.append(
                SensorRuntime(
                    sensor_id=row.sensor_id,
                    device_id=row.device_id,
                    room_id=row.room_id,
                    mac=row.mac_address,
                    location=row.location,
                    sensor_type=row.sensor_type,
                    uuid=row.uuid,
                )
            )
    return sensors
