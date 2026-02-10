from __future__ import annotations

import random
from datetime import datetime
from typing import Callable, List


class SensorRuntime:
    """
    Plain runtime sensor object (NO ORM).
    """
    def __init__(
        self,
        *,
        sensor_id: int,
        device_id: int,
        room_id: int,
        mac: str,
        location: str,
        sensor_type: str,
        uuid: str,
    ):
        self.sensor_id = sensor_id
        self.device_id = device_id
        self.room_id = room_id
        self.mac = mac
        self.location = location
        self.sensor_type = sensor_type
        self.uuid = uuid


class SensorSampler:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.sensors: List[SensorRuntime] = []

        # ✅ Load everything ONCE, flattening relationships
        from persistence.database import session_scope
        from persistence.models.sensor import Sensor as SensorModel
        from persistence.models.device import Device as DeviceModel

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

            for r in rows:
                self.sensors.append(
                    SensorRuntime(
                        sensor_id=r.sensor_id,
                        device_id=r.device_id,
                        room_id=r.room_id,
                        mac=r.mac_address,
                        location=r.location,
                        sensor_type=r.sensor_type,
                        uuid=r.uuid,
                    )
                )

        print(f"[SensorSampler] Loaded {len(self.sensors)} sensors")

    async def emit(
        self,
        now: datetime,
        *,
        room_engine,
        on_event: Callable[[dict], None],
    ) -> None:
        for sensor in self.sensors:
            room = room_engine.rooms.get(sensor.room_id)
            if not room:
                continue

            value = self._sample(sensor.sensor_type, room)

            event = {
                "timestamp": now,
                "room_id": sensor.room_id,
                "device_id": sensor.device_id,
                "mac": sensor.mac,
                "device_type": "sensor",
                "location": sensor.location,
                "sensor_type": sensor.sensor_type,
                "uuid": sensor.uuid,
                "unit": self._unit(sensor.sensor_type),
                "value": value,
            }

            await on_event(event)

    def _sample(self, sensor_type: str, room) -> float:
        if sensor_type == "temperature":
            return room.temperature + self.rng.gauss(0, 0.05)
        if sensor_type == "humidity":
            return room.humidity + self.rng.gauss(0, 0.2)
        if sensor_type == "co2":
            return max(400.0, room.co2 + self.rng.gauss(0, 20))
        if sensor_type == "light":
            return max(0.0, room.light + self.rng.gauss(0, 2))
        if sensor_type == "sound":
            return max(0.0, room.sound + self.rng.gauss(0, 1))
        return 0.0

    def _unit(self, sensor_type: str) -> str:
        return {
            "temperature": "°C",
            "humidity": "%",
            "co2": "ppm",
            "light": "lux",
            "sound": "dB",
        }.get(sensor_type, "")
