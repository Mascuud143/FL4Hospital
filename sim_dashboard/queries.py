from persistence.database import session_scope
from persistence.models.room import Room
from persistence.models.patient import Patient
from persistence.models.room_assignment import RoomAssignment
from persistence.models.comfort_preference import ComfortPreference


def get_rooms():
    print("Getting rooms...")
    with session_scope() as session:
        return session.query(Room).all()


def get_room_detail(room_id):
    print(f"Getting details for room_id={room_id}")
    with session_scope() as session:
        room = session.get(Room, room_id)
        assignments = (
            session.query(RoomAssignment)
            .filter_by(room_id=room_id)
            .all()
        )
        return room, assignments


def get_patient_detail(patient_id):
    with session_scope() as session:
        patient = session.get(Patient, patient_id)
        comforts = (
            session.query(ComfortPreference)
            .filter_by(patient_id=patient_id)
            .order_by(ComfortPreference.timestamp)
            .all()
        )
        return patient, comforts


# get devives , sensors, ventialtion speaker for each room
def get_room_devices(room_id):
    with session_scope() as session:
        room = session.get(Room, room_id)
        devices = room.devices  # Assuming a relationship is defined
        return devices
