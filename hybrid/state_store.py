from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class RoomState:
    # Runtime-only state for the hybrid control loop.
    virtual_temp: Optional[float] = None
    last_timestamp: Optional[datetime] = None
    last_ble_temp: Optional[float] = None
    hvac_mode: str = "off"
    active_hvac_usage_id: Optional[int] = None

    def reset(self) -> None:
        self.virtual_temp = None
        self.last_timestamp = None
        self.last_ble_temp = None
        self.hvac_mode = "off"
        self.active_hvac_usage_id = None


ROOM_STATE = RoomState()
