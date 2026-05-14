from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Optional

from ble import BLEManager, Device as RuntimeDevice, Sensor as RuntimeSensor
from ble.sensor import (
    parse_air_quality_thingy,
    parse_humidity_thingy,
    parse_light_thingy,
    parse_sound_thingy,
    parse_temp_thingy,
)
from hybrid_prototype.controller import process_ble_event
from hybrid_prototype.db_ingest import db_sink
from persistence.database import session_scope
from persistence.models import Device as DeviceModel


PARSER_BY_SENSOR_TYPE = {
    "temperature": parse_temp_thingy,
    "humidity": parse_humidity_thingy,
    "co2": parse_air_quality_thingy,
    "light": parse_light_thingy,
    "sound": parse_sound_thingy,
}


async def _on_event(event: dict) -> None:
    await db_sink(event)
    if event.get("sensor_type") == "temperature":
        await process_ble_event(event)


@dataclass
class RuntimeHandle:
    device_id: int
    mac_address: str
    manager: BLEManager
    stop_event: threading.Event
    thread: Optional[threading.Thread] = None


_HANDLES: dict[int, RuntimeHandle] = {}


def _build_runtime_device(device_id: int) -> RuntimeDevice:
    with session_scope() as session:
        device = session.get(DeviceModel, device_id)
        if device is None:
            raise ValueError("Device not found.")
        if not device.mac_address:
            raise ValueError("Device has no MAC address.")

        runtime = RuntimeDevice(
            mac_address=device.mac_address,
            name=f"Nordic-{device.device_id}",
            room_id=device.room_id,
            label=f"Room {device.room_id} {device.location or 'main'} Nordic",
        )
        for sensor in device.sensors:
            parser = PARSER_BY_SENSOR_TYPE.get(sensor.sensor_type)
            if parser is None:
                continue
            runtime.add_sensor(
                RuntimeSensor(
                    uuid=sensor.uuid,
                    sensor_type=sensor.sensor_type,
                    unit=sensor.unit or "",
                    parser=parser,
                )
            )
        if not runtime.sensors:
            raise ValueError("No supported sensor UUIDs are registered for this Nordic device.")
        return runtime


async def _run_handle(handle: RuntimeHandle) -> None:
    await handle.manager.start()
    try:
        while not handle.stop_event.is_set():
            await asyncio.sleep(0.5)
    finally:
        await handle.manager.stop()


def _thread_target(handle: RuntimeHandle) -> None:
    asyncio.run(_run_handle(handle))


def start_runtime(device_id: int) -> None:
    stop_runtime(device_id)
    runtime_device = _build_runtime_device(device_id)
    stop_event = threading.Event()
    manager = BLEManager(devices=[runtime_device], on_event=_on_event)
    handle = RuntimeHandle(
        device_id=device_id,
        mac_address=runtime_device.mac_address.upper(),
        manager=manager,
        stop_event=stop_event,
    )
    thread = threading.Thread(
        target=_thread_target,
        args=(handle,),
        name=f"hybrid_ble_runtime_{device_id}",
        daemon=True,
    )
    handle.thread = thread
    _HANDLES[device_id] = handle
    thread.start()


def stop_runtime(device_id: int) -> None:
    handle = _HANDLES.pop(device_id, None)
    if handle is None:
        return
    handle.stop_event.set()
    if handle.thread and handle.thread.is_alive():
        handle.thread.join(timeout=1.0)


def get_runtime_status(device_id: int) -> dict[str, str | bool | None]:
    handle = _HANDLES.get(device_id)
    if handle is None:
        return {
            "state": "saved",
            "connected": False,
            "last_error": None,
            "last_seen": None,
        }
    status = handle.manager.get_status().get(handle.mac_address, {})
    connected = bool(status.get("connected"))
    last_error = status.get("last_error")
    last_seen = status.get("last_seen")
    if connected:
        state = "connected"
    elif last_error:
        state = "failed"
    else:
        state = "disconnected"
    return {
        "state": state,
        "connected": connected,
        "last_error": last_error,
        "last_seen": last_seen,
    }
