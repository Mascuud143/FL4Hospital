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
    if u is None:
        return None
    if hasattr(u, "uuid"):
        try:
            u = u.uuid
        except Exception:
            pass
    if isinstance(u, str):
        return u
    try:
        return str(u)
    except Exception:
        return None


def _get_device_from_event(session, event: dict) -> Optional[DeviceModel]:
    """
    Supported device identifiers (in priority order):
      1) device_id
      2) mac / mac_address / device_mac
      3) composite: room_id + device_type + location
    """
    # 1) device_id
    device_id = event.get("device_id")
    if device_id is not None:
        try:
            device_id = int(device_id)
        except Exception:
            device_id = None
        if device_id is not None:
            return (
                session.query(DeviceModel)
                .filter(DeviceModel.device_id == device_id)
                .one_or_none()
            )

    # 2) MAC
    mac = event.get("mac_address") or event.get("device_mac") or event.get("mac")
    if isinstance(mac, str):
        mac = mac.strip().upper()
    if mac:
        return (
            session.query(DeviceModel)
            .filter(DeviceModel.mac_address == mac)
            .one_or_none()
        )

    # 3) composite identity: room_id + device_type + location
    room_id = event.get("room_id")
    device_type = event.get("device_type")
    location = event.get("location")

    if room_id is not None and device_type is not None and location is not None:
        try:
            room_id = int(room_id)
        except Exception:
            return None

        q = session.query(DeviceModel).filter(DeviceModel.room_id == room_id)

        if hasattr(DeviceModel, "device_type"):
            q = q.filter(DeviceModel.device_type == str(device_type))
        if hasattr(DeviceModel, "location"):
            q = q.filter(DeviceModel.location == str(location))

        return q.one_or_none()

    return None


async def db_sink(event: dict):
    print(f"db_sink received event: {event}")
    """
    Resolves device and sensor, then inserts a row into Data.

    Device resolution (priority):
      - event["device_id"]
      - event["mac"] / ["mac_address"] / ["device_mac"]
      - event["room_id"] + event["device_type"] + event["location"]

    Sensor resolution (priority):
      - device_id + uuid
      - device_id + sensor_type
    """
    value = event.get("value")
    if value is None:
        return

    sensor_type = event.get("sensor_type")
    ts = _parse_ts(event.get("timestamp"))
    uuid = _normalize_uuid(event.get("uuid"))

    with session_scope() as session:
        db_device = _get_device_from_event(session, event)

        if db_device is None:
            raise ValueError(
                "db_sink: device not found. Provide one of: "
                "device_id OR mac OR (room_id+device_type+location). "
                f"event={event}"
            )

        q = session.query(SensorModel).filter(SensorModel.device_id == db_device.device_id)

        db_sensor = None

        # Prefer UUID match if available
        if uuid and hasattr(SensorModel, "uuid"):
            db_sensor = q.filter(SensorModel.uuid == uuid).one_or_none()

        # Fallback to sensor_type
        if db_sensor is None:
            if not sensor_type:
                raise ValueError(f"db_sink: missing sensor_type in event: {event}")
            db_sensor = q.filter(SensorModel.sensor_type == sensor_type).one_or_none()

        if db_sensor is None:
            raise ValueError(
                f"db_sink: sensor not found for device_id={db_device.device_id} "
                f"(uuid={uuid}, type={sensor_type}) event={event}"
            )

        row = Data(sensor_id=db_sensor.sensor_id, value=float(value))
        if ts is not None:
            row.timestamp = ts

    
        # before saving, we change if there is big change in temperarture, humifity etc
        # check from data base if there big change and then we only save if we have +- number
        # for this type of sensor type and sensor id


        current_value = session.query

        session.add(row)
        # session_scope() handles commit/rollback
