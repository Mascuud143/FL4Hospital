"""
BLE & Device Layer

This package handles:
- BLE discovery and connections
- Device and sensor abstractions
- BLE characteristic UUID definitions

No business logic, database access, or UI code should live here.
"""

from .ble_manager import BLEManager
from .device import Device
from .sensor import Sensor

__all__ = [
    "BLEManager",
    "Device",
    "Sensor",
]
