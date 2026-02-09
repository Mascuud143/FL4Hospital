from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from persistence.database import session_scope
from persistence.models.data import Data
from persistence.models.device import Device as DeviceModel
from persistence.models.sensor import Sensor as SensorModel


def _parse_ts(ts: Any) -> Optional[datetime]:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts)
        except ValueError:
            return None
    return None


async def db_sink(event: dict):
    """
    Assumes devices/sensors are already seeded in DB.
    Resolves sensor_id via (mac_address + uuid) or (mac_address + sensor_type).
    Inserts into Data table.
    """
    mac = event.get("mac_address") or event.get("device_mac")
    if not mac:
        raise ValueError(f"db_sink: missing mac_address in event: {event}")

    value = event.get("value")
    if value is None:
        return

    uuid = event.get("uuid")
    sensor_type = event.get("sensor_type")
    ts = _parse_ts(event.get("timestamp"))

    with session_scope() as session:
        db_device = session.query(DeviceModel).filter(DeviceModel.mac_address == mac).one_or_none()
        if db_device is None:
            raise ValueError(f"db_sink: device {mac} not found in DB (did seeding run?)")

        q = session.query(SensorModel).filter(SensorModel.device_id == db_device.device_id)

        if uuid is not None and hasattr(SensorModel, "uuid"):
            db_sensor = q.filter(SensorModel.uuid == uuid).one_or_none()
        else:
            if not sensor_type:
                raise ValueError(f"db_sink: missing sensor_type (and no uuid) in event: {event}")
            db_sensor = q.filter(SensorModel.sensor_type == sensor_type).one_or_none()

        if db_sensor is None:
            raise ValueError(f"db_sink: sensor not found for device {mac} (uuid={uuid}, type={sensor_type})")

        row = Data(sensor_id=db_sensor.sensor_id, value=float(value))
        if ts is not None:
            row.timestamp = ts

        session.add(row)
        