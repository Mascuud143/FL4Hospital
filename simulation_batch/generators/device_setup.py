from __future__ import annotations

# Build room devices.
# - Builds room MAC addresses - _random_mac()
# - Picks sensor units - _unit_for()
# - Adds main room sensors
# - Adds toilet sensors
# - Adds room devices - create_room_devices()

import random
from typing import List

from ble import Device as BLEDevice, Sensor as BLESensor
from persistence.models.room import Room


def _random_mac(rng: random.Random) -> str:
    return "02:00:00:%02x:%02x:%02x" % (
        rng.randint(0, 255),
        rng.randint(0, 255),
        rng.randint(0, 255),
    )


def _unit_for(sensor_type: str) -> str:
    return {
        "temperature": "C",
        "humidity": "%",
        "co2": "ppm",
        "light": "lux",
        "sound": "dB",
    }.get(sensor_type, "")


def create_room_devices(*, rooms: list[Room], rng: random.Random, include_speaker: bool = True) -> List[BLEDevice]:
    main_sensor_types = ("temperature",)
    toilet_sensor_types = ("temperature",)
    devices: List[BLEDevice] = []
    for room in rooms:
        mac_main = _random_mac(rng)
        main_device = BLEDevice(mac_address=mac_main, label=f"Room {room.room_number} Main Sensor")
        main_device.room_id = room.room_id
        main_device.device_type = "sensor"
        main_device.location = "main"
        for sensor_type in main_sensor_types:
            uuid = f"{mac_main}-{sensor_type}"
            main_device.add_sensor(BLESensor(uuid=uuid, sensor_type=sensor_type, unit=_unit_for(sensor_type), parser=lambda b: None))
        devices.append(main_device)

        mac_toilet = _random_mac(rng)
        toilet_device = BLEDevice(mac_address=mac_toilet, label=f"Room {room.room_number} Toilet Sensor")
        toilet_device.room_id = room.room_id
        toilet_device.device_type = "sensor"
        toilet_device.location = "toilet"
        for sensor_type in toilet_sensor_types:
            uuid = f"{mac_toilet}-{sensor_type}"
            toilet_device.add_sensor(BLESensor(uuid=uuid, sensor_type=sensor_type, unit=_unit_for(sensor_type), parser=lambda b: None))
        devices.append(toilet_device)

        for label, device_type, location in [
            ("Ventilation", "ventilation", "main"),
            ("Toilet Heater", "toilet_heater", "toilet"),
            ("Toilet Light", "toilet_light", "toilet"),
        ]:
            device = BLEDevice(mac_address=None, label=f"Room {room.room_number} {label}")
            device.room_id = room.room_id
            device.device_type = device_type
            device.location = location
            devices.append(device)

        if include_speaker:
            speaker = BLEDevice(mac_address=None, label=f"Room {room.room_number} Speaker")
            speaker.room_id = room.room_id
            speaker.device_type = "speaker"
            speaker.location = "main"
            devices.append(speaker)
    return devices
