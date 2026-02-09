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


def _normalize_uuid(u: Any) -> Optional[str]:
    """
    Accepts:
      - None
      - a UUID string
      - a BleakGATTCharacteristic (has .uuid)
    Returns a string uuid or None.
    """
    if u is None:
        return None

    # BleakGATTCharacteristic -> use its uuid attribute
    if hasattr(u, "uuid"):
        try:
            u = u.uuid
        except Exception:
            pass

    # Now u should be a string-like UUID
    if isinstance(u, str):
        return u

    # Some UUID objects stringify cleanly
    try:
        return str(u)
    except Exception:
        return None


async def db_sink(event: dict):
    """
    Resolves device by MAC and sensor by (device_id + uuid) OR (device_id + sensor_type),
    then inserts a row into Data.
    """
    # ✅ accept your actual key name "mac"
    mac = event.get("mac_address") or event.get("device_mac") or event.get("mac")
    if not mac:
        raise ValueError(f"db_sink: missing mac in event: {event}")

    value = event.get("value")
    if value is None:
        return

    sensor_type = event.get("sensor_type")
    ts = _parse_ts(event.get("timestamp"))

    # ✅ normalize uuid (your event currently carries a BleakGATTCharacteristic object)
    uuid = _normalize_uuid(event.get("uuid"))

    with session_scope() as session:
        db_device = (
            session.query(DeviceModel)
            .filter(DeviceModel.mac_address == mac)
            .one_or_none()
        )
        if db_device is None:
            raise ValueError(f"db_sink: device {mac} not found in DB (did seeding run?)")

        q = session.query(SensorModel).filter(SensorModel.device_id == db_device.device_id)

        db_sensor = None
        if uuid and hasattr(SensorModel, "uuid"):
            db_sensor = q.filter(SensorModel.uuid == uuid).one_or_none()

        if db_sensor is None:
            if not sensor_type:
                raise ValueError(f"db_sink: missing sensor_type in event: {event}")
            db_sensor = q.filter(SensorModel.sensor_type == sensor_type).one_or_none()

        if db_sensor is None:
            raise ValueError(f"db_sink: sensor not found for device {mac} (uuid={uuid}, type={sensor_type})")

        row = Data(sensor_id=db_sensor.sensor_id, value=float(value))
        if ts is not None:
            row.timestamp = ts

        session.add(row)
        # ❌ DO NOT call session.commit() here; session_scope() already commits
