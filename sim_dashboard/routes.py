from flask import Blueprint, render_template

from persistence.database import session_scope
from persistence.models.room import Room
from persistence.models.patient import Patient
from persistence.models.data import Data
from persistence.models.utility_usage import UtilityUsage
from persistence.models.comfort_preference import ComfortPreference
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

        sensors = []
        for device in devices:
            for sensor in device.sensors:
                sensors.append(
                    {
                        "sensor_id": sensor.sensor_id,
                        "sensor_type": sensor.sensor_type,
                        "unit": sensor.unit,
                        "device_id": device.device_id,
                    }
                )

        comfort_rows = (
            session.query(ComfortPreference)
            .filter(ComfortPreference.room_id == room_id)
            .order_by(ComfortPreference.timestamp.desc())
            .all()
        )
        comfort_preferences = []
        for pref in comfort_rows:
            window_start = pref.timestamp - timedelta(hours=1)
            window_end = pref.timestamp + timedelta(hours=1)
            sensor_windows = []
            for s in sensors:
                before_rows = (
                    session.query(Data)
                    .filter(
                        Data.sensor_id == s["sensor_id"],
                        Data.timestamp >= window_start,
                        Data.timestamp <= pref.timestamp,
                    )
                    .order_by(Data.timestamp.desc())
                    .limit(10)
                    .all()
                )
                after_rows = (
                    session.query(Data)
                    .filter(
                        Data.sensor_id == s["sensor_id"],
                        Data.timestamp >= pref.timestamp,
                        Data.timestamp <= window_end,
                    )
                    .order_by(Data.timestamp.asc())
                    .limit(10)
                    .all()
                )
                sensor_windows.append(
                    {
                        **s,
                        "before_rows": [
                            {"value": r.value, "timestamp": r.timestamp}
                            for r in before_rows
                        ],
                        "after_rows": [
                            {"value": r.value, "timestamp": r.timestamp}
                            for r in after_rows
                        ],
                    }
                )
            comfort_preferences.append(
                {
                    "comfort_pref_id": pref.comfort_pref_id,
                    "timestamp": pref.timestamp,
                    "temperature_main": pref.temperature_main,
                    "temperature_toilet": pref.temperature_toilet,
                    "light_intensity": pref.light_intensity,
                    "sound_level": pref.sound_level,
                    "airflow": pref.airflow,
                    "source": pref.source,
                    "patient_name": pref.patient.name if pref.patient else None,
                    "sensor_windows": sensor_windows,
                }
            )

        utility_rows = (
            session.query(UtilityUsage)
            .filter(UtilityUsage.room_id == room_id)
            .order_by(UtilityUsage.start_time.desc())
            .limit(10)
            .all()
        )
        utility_usages = [
            {
                "category": row.category,
                "power_consumption": row.power_consumption,
                "water_consumption": row.water_consumption,
                "start_time": row.start_time,
                "end_time": row.end_time,
                "device_id": row.device_id,
            }
            for row in utility_rows
        ]

        ventilation_data = []
        for device in devices:
            vent = device.ventilation
            if vent:
                ventilation_data.append(
                    {
                        "device_id": device.device_id,
                        "mode": vent.mode,
                        "level": vent.level,
                        "timestamp": vent.timestamp,
                    }
                )

        ventilation_data.sort(
            key=lambda v: v["timestamp"] or 0,
            reverse=True,
        )

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

    return render_template(
        "room_detail.html",
        room=room_data,
        assignments=assignments,
        comfort_preferences=comfort_preferences,
        utility_usages=utility_usages,
        ventilation_data=ventilation_data,
    )


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
