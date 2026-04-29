from __future__ import annotations

from typing import Iterable, Optional

from persistence.database import session_scope
from persistence.models.device import Device as DeviceModel
from persistence.models.sensor import Sensor as SensorModel
from simulation_batch.csv_storage import write_model_row


def _device_lookup_query(session, *, mac: Optional[str], room_id: Optional[int], device_type: str, location: Optional[str]):
    """
    Prefer MAC when available (unique). If MAC is None, fall back to a composite identity:
      (room_id, device_type, location)
    """
    q = session.query(DeviceModel)

    if mac:
        return q.filter(DeviceModel.mac_address == mac)

    # No MAC: must have enough info to uniquely identify the device
    q = q.filter(DeviceModel.device_type == device_type)

    if room_id is not None and hasattr(DeviceModel, "room_id"):
        q = q.filter(DeviceModel.room_id == room_id)

    if location is not None and hasattr(DeviceModel, "location"):
        q = q.filter(DeviceModel.location == location)

    return q


def seed_devices_and_sensors(devices: Iterable, *, room_id_default=None) -> None:
    """
    Ensures each runtime device exists in DB, and ensures its sensors exist.

    Expects runtime device objects to have (as applicable):
      - mac_address (nullable)
      - device_type (e.g. sensor | ventilation | speaker | toilet_heater | toilet_light)
      - room_id (or uses room_id_default)
      - location (e.g. main | toilet)
      - sensors: list of sensor objects with uuid, sensor_type, unit
    """
    with session_scope() as session:
        for d in devices:
            mac = getattr(d, "mac_address", None)
            if isinstance(mac, str) and mac.strip() == "":
                mac = None

            room_id = getattr(d, "room_id", room_id_default)
            device_type = getattr(d, "device_type", None) or "sensor"
            location = getattr(d, "location", None)

            # Find existing device (MAC preferred, otherwise composite)
            db_device = (
                _device_lookup_query(
                    session,
                    mac=mac,
                    room_id=room_id,
                    device_type=device_type,
                    location=location,
                )
                .one_or_none()
            )

            if db_device is None:
                db_device = DeviceModel(
                    mac_address=mac,
                    device_type=device_type,
                    room_id=room_id,
                    location=location,
                )
                write_model_row(db_device)
                session.add(db_device)
                session.flush()  # ensures device_id

            else:
                # keep DB aligned with runtime in case seeding is re-run with updates
                if hasattr(db_device, "device_type"):
                    db_device.device_type = device_type
                if hasattr(db_device, "room_id") and room_id is not None:
                    db_device.room_id = room_id
                if hasattr(db_device, "location"):
                    db_device.location = location
                if hasattr(db_device, "mac_address"):
                    db_device.mac_address = mac

            # Ensure sensors exist (only for devices that have them)
            for s in getattr(d, "sensors", []) or []:
                sensor_type = getattr(s, "sensor_type", None)
                unit = getattr(s, "unit", None)
                uuid = getattr(s, "uuid", None)

                q = session.query(SensorModel).filter(SensorModel.device_id == db_device.device_id)

                db_sensor = None
                if uuid is not None and hasattr(SensorModel, "uuid"):
                    db_sensor = q.filter(SensorModel.uuid == str(uuid)).one_or_none()

                if db_sensor is None and sensor_type is not None and hasattr(SensorModel, "sensor_type"):
                    db_sensor = q.filter(SensorModel.sensor_type == sensor_type).one_or_none()

                if db_sensor is None:
                    skwargs = {"device_id": db_device.device_id}

                    if sensor_type is not None and hasattr(SensorModel, "sensor_type"):
                        skwargs["sensor_type"] = sensor_type
                    if unit is not None and hasattr(SensorModel, "unit"):
                        skwargs["unit"] = unit
                    if uuid is not None and hasattr(SensorModel, "uuid"):
                        skwargs["uuid"] = str(uuid)

                    db_sensor = SensorModel(**skwargs)
                    write_model_row(db_sensor)
                    session.add(db_sensor)
