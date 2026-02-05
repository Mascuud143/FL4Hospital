from domain.room_logic import is_room_available, Room
from domain.patient_logic import can_patient_be_assigned, Patient


def can_assign_patient_to_room(
    patient: Patient,
    room: Room,
    current_occupancy: int
) -> bool:
    """
    Validate whether a patient can be assigned to a room.
    """
    if not can_patient_be_assigned(patient):
        return False

    if not is_room_available(room, current_occupancy):
        return False

    return True
