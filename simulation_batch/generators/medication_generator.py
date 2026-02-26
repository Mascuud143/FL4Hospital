# simulation_batch/generators/medication_generator.py

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Dict

from persistence.database import session_scope
from persistence.models.admission import Admission
from persistence.models.medication import Medication

# ✅ reuse same UTC normalizer as engine
from simulation_batch.room_engine import _as_utc


class MedicationGenerator:
    """
    Generates Medication rows per patient stay.
    Handles naive vs aware datetime correctly.
    """

    def __init__(self, *, seed: int = 42, diagnoses: Dict):
        self.rng = random.Random(seed)
        self.diagnoses = diagnoses

    def generate_for_horizon(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> int:

        # Normalize horizon times
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

                diagnosis = adm.current_diagnosis
                if diagnosis not in self.diagnoses:
                    continue

                med_schedule = self.diagnoses[diagnosis]["medications"]

                # ✅ Normalize DB timestamps
                adm_start = _as_utc(adm.admitted_at)
                adm_end = _as_utc(adm.discharged_at)

                stay_start = max(adm_start, start_time)
                stay_end = min(adm_end, end_time)

                day_cursor = stay_start.replace(
                    hour=0, minute=0, second=0, microsecond=0
                )

                while day_cursor < stay_end:
                    day_end = day_cursor + timedelta(days=1)

                    for drug_name, hours in med_schedule.items():
                        for h in hours:

                            if h >= 0:
                                ts = day_cursor + timedelta(hours=h)
                            else:
                                # PRN medication → random time that day
                                random_hour = self.rng.randint(0, 23)
                                ts = day_cursor + timedelta(hours=random_hour)

                            if ts < stay_start or ts >= stay_end:
                                continue

                            session.add(
                                Medication(
                                    patient_id=adm.patient_id,
                                    medication_time=ts,
                                    drug_name=drug_name,
                                    route="oral",
                                    dose="1 dose",
                                    status="taken",
                                )
                            )
                            inserted += 1

                    day_cursor += timedelta(days=1)

        return inserted