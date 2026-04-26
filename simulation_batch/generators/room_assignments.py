from __future__ import annotations

# Build room assignments.
# - Checks room time overlap - overlaps()
# - Creates room rows - create_room()
# - Finds free rooms for each stay window - find_or_create_room_for_window()
# - Creates assignment rows - create_assignment()
# - Splits one stay for transfers - apply_transfer()

from datetime import datetime
from typing import Tuple

from persistence.models.room import Room
from persistence.models.room_assignment import RoomAssignment
from simulation_batch.csv_storage import write_model_row


def overlaps(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
    return not (a_end <= b_start or a_start >= b_end)


def create_room(*, session, room_number: int) -> Room:
    room = Room(room_number=room_number)
    write_model_row(room)
    session.add(room)
    session.flush()
    return room


def find_or_create_room_for_window(
    *,
    session,
    rooms: list[Room],
    room_occupancy: dict[int, list[Tuple[datetime, datetime]]],
    window_start: datetime,
    window_end: datetime,
    room_number_seed: int,
) -> Room:
    for room in rooms:
        occupancy = room_occupancy.get(room.room_id, [])
        if all(not overlaps(window_start, window_end, start, end) for (start, end) in occupancy):
            return room
    next_number = room_number_seed if not rooms else max(int(room.room_number) for room in rooms) + 1
    room = create_room(session=session, room_number=next_number)
    rooms.append(room)
    room_occupancy[room.room_id] = []
    return room


def create_assignment(
    *,
    session,
    admission_id: int,
    patient_id: int,
    room_id: int,
    start_time: datetime,
    end_time: datetime,
) -> RoomAssignment:
    assignment = RoomAssignment(
        admission_id=admission_id,
        patient_id=patient_id,
        room_id=room_id,
        start_time=start_time,
        end_time=end_time,
    )
    write_model_row(assignment)
    session.add(assignment)
    return assignment


def apply_transfer(
    *,
    session,
    admission_id: int,
    patient_id: int,
    admit: datetime,
    discharge: datetime,
    transfer_time: datetime,
    rooms: list[Room],
    transfer_rooms: list[Room],
    room_occupancy: dict[int, list[Tuple[datetime, datetime]]],
    max_room_number: int,
) -> int:
    original = (
        session.query(RoomAssignment)
        .filter(RoomAssignment.admission_id == admission_id)
        .order_by(RoomAssignment.start_time.asc())
        .first()
    )
    if not original:
        return max_room_number
    original_room_id = original.room_id
    old_end = original.end_time
    original.end_time = transfer_time
    updated: list[Tuple[datetime, datetime]] = []
    replaced = False
    for start, end in room_occupancy.get(original_room_id, []):
        if not replaced and start == admit and end == old_end:
            updated.append((admit, transfer_time))
            replaced = True
        else:
            updated.append((start, end))
    room_occupancy[original_room_id] = updated
    destination = None
    for room in transfer_rooms:
        if room.room_id == original_room_id:
            continue
        occupancy = room_occupancy.get(room.room_id, [])
        if all(not overlaps(transfer_time, discharge, start, end) for (start, end) in occupancy):
            destination = room
            break
    if destination is None:
        max_room_number += 1
        destination = create_room(session=session, room_number=max_room_number)
        rooms.append(destination)
        transfer_rooms.append(destination)
        room_occupancy.setdefault(destination.room_id, [])
    create_assignment(
        session=session,
        admission_id=admission_id,
        patient_id=patient_id,
        room_id=destination.room_id,
        start_time=transfer_time,
        end_time=discharge,
    )
    room_occupancy.setdefault(destination.room_id, [])
    room_occupancy[destination.room_id].append((transfer_time, discharge))
    return max_room_number
