# simulation_batch/generators/visit_generator.py

from __future__ import annotations

import random
from datetime import datetime, timedelta

from persistence.database import session_scope
from persistence.models.admission import Admission
from persistence.models.medication import Medication
from persistence.models.visit import Visit

#reuse UTC helper
from simulation_batch.room_engine import _as_utc
from simulation_batch.csv_filestorage import write_model_row


class VisitGenerator:
    """
    Generates nurse visits:
      - 3 fixed visits per day
      - If >4 medications that day → reduce fixed visits to 1
      - +2h follow-up visit after each medication
    """

    def __init__(self, *, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_for_horizon(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> int:

        start_time = _as_utc(start_time)
        end_time = _as_utc(end_time)

        inserted = 0

        with session_scope() as session:

            admissions = (
                session.query(Admission)
                .filter(Admission.discharged_at > start_time)
                .filter(Admission.admitted_at < end_time)
                .all()
            )

            for adm in admissions:

                adm_start = _as_utc(adm.admitted_at)
                adm_end = _as_utc(adm.discharged_at)

                stay_start = max(adm_start, start_time)
                stay_end = min(adm_end, end_time)

                day_cursor = stay_start.replace(
                    hour=0, minute=0, second=0, microsecond=0
                )

                while day_cursor < stay_end:
                    day_end = day_cursor + timedelta(days=1)

                    meds = (
                        session.query(Medication)
                        .filter(Medication.patient_id == adm.patient_id)
                        .filter(Medication.medication_time >= day_cursor)
                        .filter(Medication.medication_time < day_end)
                        .all()
                    )

                    daily_med_count = len(meds)

                    # Rule
                    fixed_visits = 1 if daily_med_count > 4 else 3

                    visit_times = []

                    # Fixed daytime visits
                    for _ in range(fixed_visits):
                        hour = self.rng.randint(7, 21)
                        ts = day_cursor + timedelta(hours=hour)
                        if stay_start <= ts < stay_end:
                            visit_times.append(ts)

                    # +2h medication follow-ups
                    for med in meds:
                        follow_up = _as_utc(med.medication_time) + timedelta(hours=2)
                        if follow_up < stay_end:
                            visit_times.append(follow_up)

                    # Remove duplicates
                    visit_times = sorted(set(visit_times))

                    def generate_a_symptom_or_none(diagnosis: str) -> str:
                        symptoms = DIAGNOSES.get(diagnosis, {}).get("symptoms", [])
                        if symptoms and self.rng.random() < 0.5:
                            return self.rng.choice(symptoms)
                        return ""
                

                    def generate_blood_pressure_based_on_symptom(symptom: str) -> str:
                        symptom = symptom.lower()

                        if symptom == "fever":
                            systolic = random.randint(100, 120)
                            diastolic = random.randint(60, 80)

                        elif symptom == "cough":
                            systolic = random.randint(110, 130)
                            diastolic = random.randint(70, 85)

                        else:
                            systolic = random.randint(110, 140)
                            diastolic = random.randint(70, 90)

                        return f"{systolic}/{diastolic}"


                    def generate_body_temperature_based_on_symptom(symptom: str) -> float:
                        if symptom == "fever":
                            return round(self.rng.uniform(38.0, 40.0), 1)
                        elif symptom == "cough":
                            return round(self.rng.uniform(36.5, 37.5), 1)
                        else:
                            return round(self.rng.uniform(36.0, 38.5), 1)
                        
                    for ts in visit_times:
                        row = Visit(
                            patient_id=adm.patient_id,
                            visit_time=ts,
                            body_temperature=round(
                                self.rng.uniform(36.0, 38.5), 1
                            ),
                            blood_pressure=f"{self.rng.randint(110,140)}/{self.rng.randint(70,90)}",
                            # get random symptoms from diagnosis, some times empty, some times one sytom only no multiple symptoms for simplicity
                            symptoms=generate_a_symptom_or_none(adm.current_diagnosis),
                        )
                        write_model_row(row)
                        session.add(row)
                        inserted += 1

                    day_cursor += timedelta(days=1)

        return inserted
