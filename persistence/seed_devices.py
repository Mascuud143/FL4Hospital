from __future__ import annotations

from typing import Iterable

from persistence.database import session_scope
from persistence.models.device import Device as DeviceModel
from persistence.models.sensor import Sensor as SensorModel


def seed_devices_and_sensors(devices: Iterable, *, room_id_default=None) -> None:
    """
    Ensures each BLE device (by mac_address) exists in DB, and ensures its sensors exist.

    Expects your runtime 'devices' objects to have:
      - mac_address
      - label
      - sensors: list of Sensor objects with uuid, sensor_type, unit
    """
    with session_scope() as session:
        for d in devices:
            mac = d.mac_address
            label = getattr(d, "label", None)

            db_device = (
                session.query(DeviceModel)
                .filter(DeviceModel.mac_address == mac)
                .one_or_none()
            )

            if db_device is None:
                kwargs = {"mac_address": mac}
                # only set fields if the ORM model actually has them
                if hasattr(DeviceModel, "label"):
                    kwargs["label"] = label
                if room_id_default is not None and hasattr(DeviceModel, "room_id"):
                    kwargs["room_id"] = room_id_default

                db_device = DeviceModel(**kwargs)
                session.add(db_device)
                session.flush()  # assigns device_id

            # Ensure sensors for this device exist
            for s in getattr(d, "sensors", []):
                sensor_type = getattr(s, "sensor_type", None)
                unit = getattr(s, "unit", None)
                uuid = getattr(s, "uuid", None)

                q = session.query(SensorModel).filter(SensorModel.device_id == db_device.device_id)

                # Prefer UUID uniqueness if your Sensor table has uuid column
                if uuid is not None and hasattr(SensorModel, "uuid"):
                    db_sensor = q.filter(SensorModel.uuid == uuid).one_or_none()
                else:
                    db_sensor = q.filter(SensorModel.sensor_type == sensor_type).one_or_none()

                if db_sensor is None:
                    skwargs = {
                        "device_id": db_device.device_id,
                    }
                    if hasattr(SensorModel, "sensor_type"):
                        skwargs["sensor_type"] = sensor_type
                    if unit is not None and hasattr(SensorModel, "unit"):
                        skwargs["unit"] = unit
                    if uuid is not None and hasattr(SensorModel, "uuid"):
                        skwargs["uuid"] = uuid

                    session.add(SensorModel(**skwargs))
