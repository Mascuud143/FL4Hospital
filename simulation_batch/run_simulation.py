
import random
from datetime import timedelta, datetime, date

from persistence.database import init_db, session_scope
from persistence.models.data import Data
from persistence.models.room import Room
from persistence.models.patient import Patient
from persistence.models.sensor import Sensor
from persistence.models.device import Device
from persistence.models.comfort_preference import ComfortPreference
from persistence.models.room_assignment import RoomAssignment

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
    # ROOM_COUNT = int(input("Enter number of rooms to simulate: "))
    # PATIENT_COUNT = int(input("Enter number of patients to simulate: "))
    # DAYS = int(input("Enter number of days to simulate: "))
    # READINGS_PER_DAY = int(input("Enter number of readings per day: "))




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
        rooms = []
        current_room_stay_sum=0
        current_room = 1

        
        COMFORT_HOURS = [0, 6, 12, 18]

        room= Room(room_number=100)
        session.add(room)
        rooms.append(room)
        session.flush()
        room_occupancy = {}  # room_id -> list of (start_time, end_time)


        print("Assigning patients to rooms...")
        print("Patient stay days:", patient_stay_days)

        for i, s in enumerate(patient_stay_days):
            patient = patients[i]

            print(
                f"Processing patient {patient.patient_id} "
                f"with stay of {s} days. "
                f"Current room stay sum: {current_room_stay_sum}, "
                f"current room: {current_room}"
            )

            # create new room if needed
            if current_room_stay_sum + s > DAYS:
                current_room += 1
                current_room_stay_sum = 0

                room = Room(room_number=100 + current_room)
                session.add(room)
                session.flush()
                rooms.append(room)

            # admission & release dates
            patient.admission_date = START_DATE + timedelta(days=current_room_stay_sum)
            patient.release_date = patient.admission_date + timedelta(days=s)

            # room assignment
            assignment = RoomAssignment(
                patient_id=patient.patient_id,
                room_id=room.room_id,
                start_time=patient.admission_date,
                end_time=patient.release_date,
            )
            session.add(assignment)

            room_occupancy.setdefault(room.room_id, []).append(
            (patient.admission_date, patient.release_date)
            )

            # ---- EXACTLY 4 comfort preferences (one-day profile) ----
            base_date = patient.admission_date
            
            # Convert to datetime and zero out time components
            if isinstance(base_date, date) and not isinstance(base_date, datetime):
                base_date = datetime.combine(base_date, datetime.min.time())
            else:
                base_date = base_date.replace(hour=0, minute=0, second=0, microsecond=0)

            for hour in COMFORT_HOURS:
                comfort_pref = generate_comfort()

                # check the hour and adjust the temperature range accordingly
                if hour == 0:  # midnight
                    comfort_pref["temperature"] = round(random.uniform(20.0, 22.0), 2)
                    # light intensity should be off at night
                    comfort_pref["light_intensity"] = 0.0
                    # sound level should be low at night
                    comfort_pref["sound_level"] = round(random.uniform(0, 20), 2)
                elif hour == 6:  # morning
                    comfort_pref["temperature"] = round(random.uniform(21.0, 23.0), 2)
                elif hour == 12:  # afternoon
                    comfort_pref["temperature"] = round(random.uniform(22.0, 24.0), 2)
                    comfort_pref["light_intensity"] = round(random.uniform(20, 50), 2)
                elif hour == 18:  # evening
                    comfort_pref["temperature"] = round(random.uniform(20.0, 21.0), 2)
                    # light intensity should be lower in the evening
                    comfort_pref["sound_level"] = round(random.uniform(0, 20), 2)
                    comfort_pref["light_intensity"] = round(random.uniform(20, 50), 2)

                comfort = ComfortPreference(
                    patient_id=patient.patient_id,
                    room_id=room.room_id,
                    timestamp=base_date + timedelta(hours=hour),
                    temperature=comfort_pref["temperature"],
                    light_intensity=comfort_pref["light_intensity"],
                    sound_level=comfort_pref["sound_level"],
                    ventilation=comfort_pref["ventilation"],
                    source=comfort_pref["source"],
                )
                session.add(comfort)

            # advance room timeline
            current_room_stay_sum += s

        session.flush()

        # ============================================================
        # EXTRA PHASE: ROOM CHANGE WHILE ADMITTED
        # ============================================================

        def overlaps(a_start, a_end, b_start, b_end):
            return not (a_end <= b_start or a_start >= b_end)

        original_last_room = max(rooms, key=lambda r: r.room_number)
        transfer_rooms = [original_last_room]
        max_room_number = original_last_room.room_number

        print(f">>> TRANSFER PHASE START (prob={CHANGE_ROOM_PROB}) <<<")

        for patient in patients:
            if random.random() > CHANGE_ROOM_PROB:
                continue

            admit = patient.admission_date
            discharge = patient.release_date
            total_days = (discharge - admit).days

            if total_days < MIN_DAYS_BEFORE_TRANSFER + MIN_DAYS_AFTER_TRANSFER + 1:
                continue

            transfer_day = random.randint(
                MIN_DAYS_BEFORE_TRANSFER,
                total_days - MIN_DAYS_AFTER_TRANSFER,
            )
            transfer_time = admit + timedelta(days=transfer_day)

            orig = (
            session.query(RoomAssignment)
            .filter(RoomAssignment.patient_id == patient.patient_id)
            .order_by(RoomAssignment.start_time.asc())
            .first()
            )
            if not orig:
                print(f"SKIP patient {patient.patient_id}: no orig assignment found")
                continue
            

            orig_room_id = orig.room_id
            orig.end_time = transfer_time

            room_occupancy[orig_room_id] = [
                (s, e)
                for (s, e) in room_occupancy[orig_room_id]
                if not (s == admit and e == discharge)
            ]
            room_occupancy[orig_room_id].append((admit, transfer_time))

            destination = None
            for r in transfer_rooms:
                if r.room_id == orig_room_id:
                    continue

                if all(
                    not overlaps(transfer_time, discharge, s, e)
                    for (s, e) in room_occupancy.get(r.room_id, [])
                ):
                    destination = r
                    break

            if destination is None:
                max_room_number += 1
                destination = Room(room_number=max_room_number)
                session.add(destination)
                session.flush()

                rooms.append(destination)
                transfer_rooms.append(destination)
                room_occupancy[destination.room_id] = []

            session.add(
                RoomAssignment(
                    patient_id=patient.patient_id,
                    room_id=destination.room_id,
                    start_time=transfer_time,
                    end_time=discharge,
                )
            )

            room_occupancy[destination.room_id].append((transfer_time, discharge))
            session.flush()



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



#mian questions
# is the simulation created for traning the data and only that 
# or 
# for traning and also simulating real world model 



# tables that need to be fixed:
# device, speaker, ventilation, data
# 
# others things to consider:
# also include light in the sensors table with unit or as a device?
# since the powerconsumtion and the waterconsumtion will not be used in traning do we need to generate this in the simulation?
# change comfort hours logic to include reading_per_day input 
# room assigment should be done at a appropiate time?
#


