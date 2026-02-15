from datetime import date, datetime
from persistence.database import session_scope
from persistence.models import Room, Patient, Device, RoomAssignment


ROOM_ID = 1
PATIENT_ID = 1


def ensure_room_and_patient(runtime_devices):
    """
    Create a real hospital stay:
    - Room
    - Patient (full data)
    - RoomAssignment
    - Attach BLE devices
    """

    with session_scope() as session:

        # ---------------- ROOM ----------------
        room = session.get(Room, ROOM_ID)
        if room is None:
            room = Room(
                room_id=ROOM_ID,
                room_number="H-01"
            )
            session.add(room)

        # ---------------- PATIENT ----------------
        patient = session.get(Patient, PATIENT_ID)
        if patient is None:
            patient = Patient(
                patient_id=PATIENT_ID,
                name="John Doe",
                age=67,
                weight=82.5,
                height=178.0,
                gender="male",
                ethnicity="unknown",
                current_diagnosis="post_surgery_recovery",
                stay_days=5,
                admission_date=date.today(),
                release_date=None,
            )
            session.add(patient)

        session.flush()

        # ---------------- ASSIGNMENT ----------------
        assignment = (
            session.query(RoomAssignment)
            .filter(RoomAssignment.room_id == ROOM_ID)
            .filter(RoomAssignment.patient_id == PATIENT_ID)
            .first()
        )

        if assignment is None:
            session.add(
                RoomAssignment(
                    room_id=ROOM_ID,
                    patient_id=PATIENT_ID,
                    start_time=datetime.utcnow(),
                    end_time=None,
                )
            )

        # ---------------- ATTACH DEVICES ----------------
        for runtime_dev in runtime_devices:
            db_dev = (
                session.query(Device)
                .filter(Device.mac_address == runtime_dev.mac_address.upper())
                .first()
            )

            if db_dev:
                db_dev.room_id = ROOM_ID
