from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class ZoneState:
    virtual_temp: Optional[float] = None
    last_timestamp: Optional[datetime] = None
    last_ble_temp: Optional[float] = None
    hvac_mode: str = "off"

    def reset(self) -> None:
        self.virtual_temp = None
        self.last_timestamp = None
        self.last_ble_temp = None
        self.hvac_mode = "off"


@dataclass
class RoomState:
    # Runtime-only state for the hybrid control loop.
    main: ZoneState = field(default_factory=ZoneState)
    toilet: ZoneState = field(default_factory=ZoneState)
    active_hvac_usage_id: Optional[int] = None

    def reset(self) -> None:
        self.main.reset()
        self.toilet.reset()
        self.active_hvac_usage_id = None


ROOM_STATES: dict[int, RoomState] = {}


def get_room_state(room_id: int) -> RoomState:
    state = ROOM_STATES.get(room_id)
    if state is None:
        state = RoomState()
        ROOM_STATES[room_id] = state
    return state


def get_zone_state(room_id: int, location: str | None) -> ZoneState:
    room_state = get_room_state(room_id)
    if location == "toilet":
        return room_state.toilet
    return room_state.main
