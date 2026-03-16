from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

from ble.ble_manager import probe_device_connection


_DEVICE_STATUS: dict[int, dict[str, str | None]] = {}
_DEVICE_MONITORS: dict[int, tuple[threading.Thread, threading.Event]] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_device_status(device_id: int) -> dict[str, str | None]:
    return _DEVICE_STATUS.get(
        device_id,
        {
            "state": "saved",
            "last_error": None,
            "last_connected_at": None,
            "updated_at": None,
        },
    )


def mark_connecting(device_id: int) -> None:
    current = get_device_status(device_id).copy()
    current["state"] = "connecting"
    current["updated_at"] = _now_iso()
    _DEVICE_STATUS[device_id] = current


def mark_connected(device_id: int) -> None:
    current = get_device_status(device_id).copy()
    current["state"] = "connected"
    current["last_error"] = None
    current["last_connected_at"] = _now_iso()
    current["updated_at"] = current["last_connected_at"]
    _DEVICE_STATUS[device_id] = current


def mark_failed(device_id: int, error: str) -> None:
    current = get_device_status(device_id).copy()
    current["state"] = "failed"
    current["last_error"] = error
    current["updated_at"] = _now_iso()
    _DEVICE_STATUS[device_id] = current


def mark_disconnected(device_id: int) -> None:
    current = get_device_status(device_id).copy()
    current["state"] = "disconnected"
    current["updated_at"] = _now_iso()
    _DEVICE_STATUS[device_id] = current


def _monitor_loop(device_id: int, mac_address: str, stop_event: threading.Event, poll_interval_s: float) -> None:
    while not stop_event.is_set():
        try:
            connected = probe_device_connection(mac_address, timeout=5.0)
        except Exception as exc:
            mark_failed(device_id, str(exc))
        else:
            if connected:
                mark_connected(device_id)
            else:
                mark_disconnected(device_id)
        stop_event.wait(poll_interval_s)


def start_device_monitor(device_id: int, mac_address: str, poll_interval_s: float = 5.0) -> None:
    stop_device_monitor(device_id)
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_monitor_loop,
        args=(device_id, mac_address.strip().upper(), stop_event, poll_interval_s),
        name=f"hybrid_ble_monitor_{device_id}",
        daemon=True,
    )
    _DEVICE_MONITORS[device_id] = (thread, stop_event)
    thread.start()


def stop_device_monitor(device_id: int) -> None:
    existing = _DEVICE_MONITORS.pop(device_id, None)
    if existing is None:
        return
    thread, stop_event = existing
    stop_event.set()
    if thread.is_alive():
        thread.join(timeout=0.2)
