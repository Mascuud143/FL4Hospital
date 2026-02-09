from flask import Blueprint, render_template

from persistence.database import session_scope
from persistence.models.room import Room
from persistence.models.patient import Patient
# import timedelta for date calculation
from datetime import timedelta

# import config
from simulation_batch.config import START_DATE, DAYS

sim_bp = Blueprint("sim_bp", __name__)


@sim_bp.route("/")
def rooms():
    """
    Display all rooms.
    Uses plain dicts to avoid DetachedInstanceError.
    """
    with session_scope() as session:
        rooms = session.query(Room).all()

        # Convert ORM objects → plain dicts (safe for Jinja)
        room_data = [
            {
                "room_id": room.room_id,
                "room_number": room.room_number,
            }
            for room in rooms
        ]
    

    # calcuate simualtion period with start date + days USE timedelta
    simulation_period = f"{START_DATE} to {(START_DATE + timedelta(days=DAYS)).strftime('%Y-%m-%d')}"

    # get simulaton info, how many patients, how many rooms, how many devices etc, the date start etc
    simulation_info = {
        "total_rooms": len(room_data),
        "total_patients": session.query(Patient).count(),
        "simulation_period": simulation_period,
    }

    return render_template("rooms.html", rooms=room_data, simulation_info=simulation_info)

@sim_bp.route("/rooms/<int:room_id>")
def room_detail(room_id):
    """
    Display details for a specific room, including patient assignments.
    """

    print(f"Fetching details for room_id={room_id}")  # Debug log
    with session_scope() as session:
        room = session.query(Room).get(room_id)

        if not room:
            return "Room not found", 404


        # get room devices, sensors etc
        devices = room.devices  # Assuming a relationship is defined
        print(f"Devices in room {room_id}: {devices}")  # Debug log

        # Convert devices to match template expectations
        device_data = [
            {
                "device_id": device.device_id,
                "device_type": device.device_type,
            }
            for device in devices
        ]

        # Convert ORM object → plain dict (safe for Jinja)
        room_data = {
            "room_id": room.room_id,
            "room_number": room.room_number,
            "devices": device_data,
        }
        
        # Convert assignments separately to match template expectations
        assignments = [
            {
                "assignment_id": assignment.assignment_id,
                "patient": {
                    "patient_id": assignment.patient.patient_id,
                    "name": assignment.patient.name,
                },
                "start_time": assignment.start_time,
                "end_time": assignment.end_time,
            }
            for assignment in room.assignments
        ]

    return render_template("room_detail.html", room=room_data, assignments=assignments)


@sim_bp.route("/patients")
def patients():
    """
    Display all patients.
    """
    with session_scope() as session:
        patients = session.query(Patient).all()

        # Convert ORM objects → plain dicts (safe for Jinja)
        patient_data = [
            {
                "patient_id": patient.patient_id,
                "name": patient.name,
                "age": patient.age,
                "admission_date": patient.admission_date,
                "release_date": patient.release_date,
            }
            for patient in patients
        ]

    return render_template("patients.html", patients=patient_data)


@sim_bp.route("/patients/<int:patient_id>")
def patient_detail(patient_id):
    """
    Display details for a specific patient, including comfort preferences.
    """

    with session_scope() as session:
        patient = session.query(Patient).get(patient_id)
        print(f"Fetched patient: {patient}")  # Debug log
        if not patient:
            return "Patient not found", 404

        # Convert ORM object → plain dict (safe for Jinja)
        patient_data = {
            "patient_id": patient.patient_id,
            "name": patient.name,
            "age": patient.age,
            "gender": patient.gender,
            "height": patient.height,
            "weight": patient.weight,
            "ethnicity": patient.ethnicity,
            "current_diagnosis": patient.current_diagnosis,
            "admission_date": patient.admission_date,
            "release_date": patient.release_date,
        }

        print(f"Patient data prepared for template: {patient_data}")  # Debug log
        
        # Convert comfort preferences to match template expectations
        comforts = [
            {
                "comfort_pref_id": pref.comfort_pref_id,
                "timestamp": pref.timestamp,
                "temperature": pref.temperature,
                "light_intensity": pref.light_intensity,
                "sound_level": pref.sound_level,
                "ventilation": pref.ventilation,
            }
            for pref in patient.comfort_preferences
        ]

    return render_template("patient_detail.html", patient=patient_data, comforts=comforts)
