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

from simulation_batch.generators.name_gen import generate_name 
from simulation_batch.generators.age_height_weight_gen import age_height_weight_generator

def run():
    random.seed(42)

    init_db()
    # config user input values  
    # how many rooms, patients, simulation date range, readings per day
    ROOM_COUNT = int(input("Enter number of rooms to simulate: "))
    PATIENT_COUNT = int(input("Enter number of patients to simulate: "))
    DAYS = int(input("Enter number of days to simulate: "))
    READINGS_PER_DAY = int(input("Enter number of readings per day: "))




    with session_scope() as session:

          # 2. Patients + comfort
        patients = []
        patient_stay_days = []
        for p in generate_patients(PATIENT_COUNT):
            age_height_weight = age_height_weight_generator()
            patient = Patient(
                age=age_height_weight["age"],
                name=generate_name(),
                current_diagnosis=p["diagnosis"],
                ethnicity=p["ethnicity"],
                gender = random.choice(["Male", "Female"]),

                height=age_height_weight["height"],
                weight=age_height_weight["weight"]
            
            )
            stay_days = random.randint(5, 20)
            patient.stay_days = stay_days
            patient_stay_days.append(stay_days)
            session.add(patient)
            patients.append(patient)

           

        session.flush()


        # 1. Rooms
        day_num = 0

        for s in patient_stay_days:
            day_num += patient_stay_days[]
            if day_num > DAYS:
                day_num= 0

            
        

        rooms = []
        for r in generate_rooms(ROOM_COUNT):
            room = Room(room_number=r["room_number"])
            session.add(room)
            rooms.append(room)

        session.flush()

      


        #room assignments



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



    print("✅ Simulation complete. Data saved.")


if __name__ == "__main__":
    run()
