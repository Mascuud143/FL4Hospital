from dataclasses import dataclass


@dataclass
class Room:
    id: int
    capacity: int
    is_active: bool = True


def is_room_available(room: Room, current_occupancy: int) -> bool:
    """
    Check if a room can accept more patients.
    """
    if not room.is_active:
        return False

    return current_occupancy < room.capacity
