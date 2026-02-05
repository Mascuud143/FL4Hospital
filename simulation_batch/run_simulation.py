import random
from datetime import timedelta

from persistence.database import init_db, session_scope
from persistence.models.data import Data
from persistence.models.room import Room
from persistence.models.patient import Patient
from persistence.models.sensor import Sensor
from persistence.models.device import Device
from persistence.models.comfort_preference import ComfortPreference

from simulation_batch.config import *
from simulation_batch.generators.rooms import generate_rooms
from simulation_batch.generators.patients import generate_patients
from simulation_batch.generators.comfort import generate_comfort
from simulation_batch.generators.environment import generate_environment_for_day


def run():
    random.seed(42)

    init_db()

    with session_scope() as session:
        # 1. Rooms
        rooms = []
        for r in generate_rooms(ROOM_COUNT):
            room = Room(room_number=r["room_number"])
            session.add(room)
            rooms.append(room)

        session.flush()

        # 2. Patients + comfort
        patients = []
        for p in generate_patients(PATIENT_COUNT):
            patient = Patient(
                name=p["name"],
                current_diagnosis=p["diagnosis"],
                height=random.randint(150, 190),
                weight=random.randint(50, 100)

            )
            session.add(patient)
            patients.append(patient)

            comfort = generate_comfort()
            session.add(ComfortPreference(
                patient=patient,
                temperature=comfort["temperature"],
            ))

        session.flush()

        # 3. Assign patients to rooms
        # first add release and admission dates for patients
        # make and save RoomAssignment
        # make sure no room has more than 1 patient at a time
        # and give them random admission and release dates within the simulation period
        assigned_rooms = random.sample(rooms, k=len(patients))
        for patient, room in zip(patients, assigned_rooms):
            admission_offset = random.randint(0, DAYS - 5)
            admission_date = START_DATE + timedelta(days=admission_offset)
            release_date = admission_date + timedelta(days=random.randint(1, 5))

            patient.admission_date = admission_date
            patient.release_date = release_date

            assignment = RoomAssignment(
                patient=patient,
                room=room,
                start_time=admission_date,
                end_time=release_date
            )
            session.add(assignment)

        # devices nordic, generate 3 per room
        devices = []
        for room in rooms:
            # noridc devices
            for _ in range(3):
                # random mac address
                mac_address = "02:00:00:%02x:%02x:%02x" % (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )
                sensor_device = Device(room_id=room.room_id, device_type="nordic", mac_address=mac_address)
                session.add(sensor_device)
                devices.append(sensor_device)


                # 4. Sensors for each device
                for sensor_type in ["temperature"]:
                    sensor = Sensor(
                        device=sensor_device,
                        sensor_type=sensor_type,
                        unit="C" if sensor_type == "temperature" else "%",
                        uuid=f"{sensor_device.mac_address}-{sensor_type}"
                    )
                    session.add(sensor)
            
            # ventilation and speaker devices 
            for dtype in ["ventilation", "speaker"]:
                mac_address = "02:00:00:%02x:%02x:%02x" % (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                )
                other_device = Device(room_id=room.room_id, device_type=dtype, mac_address=mac_address)
                session.add(other_device)
                devices.append(other_device)
        session.flush()
        sensors = session.query(Sensor).all()




        


        # 5. Generate environment data
        current_date = START_DATE

        for _ in range(DAYS):
            for room in rooms:
                day_readings = generate_environment_for_day(
                    room.room_id,
                    current_date,
                    temp_range= TEMP_RANGE,
                    readings_per_day= READINGS_PER_DAY,
                    humidity_range = HUMIDITY_RANGE
                )

                for r in day_readings:
                    data_entry = Data(
                        sensor_id=random.choice([s.sensor_id for s in sensors if s.device.room_id == room.room_id and s.sensor_type == "temperature"]),
                        timestamp=r["timestamp"],
                        value=r["temperature"],

                    )
                    session.add(data_entry)

            current_date += timedelta(days=1)

    print("✅ Simulation complete. Data saved.")


if __name__ == "__main__":
    run()
