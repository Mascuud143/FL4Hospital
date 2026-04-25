from __future__ import annotations

# Build visit rows.
# - Splits work into chunks - _chunked()
# - Builds rows for each chunk - _build_rows_for_chunk()
# - Uses stay windows and medication times
# - Adds fixed daily visits and follow-up visits
# - Writes visit rows - VisitGenerator

import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from typing import Iterable

from persistence.database import session_scope
from persistence.models.admission import Admission
from persistence.models.medication import Medication
from persistence.models.visit import Visit
from simulation_batch.csv_storage import write_model_rows
from simulation_batch.generators.diagnosis_profiles import DIAGNOSES
from simulation_batch.simulation_steps.room_simulation import as_utc


def _chunked(items: list[dict], size: int) -> Iterable[list[dict]]:
    for idx in range(0, len(items), size):
        yield items[idx:idx + size]


def _symptom_for_diagnosis(rng: random.Random, diagnosis: str) -> str:
    symptoms = DIAGNOSES.get(diagnosis, {}).get("symptoms", [])
    if symptoms and rng.random() < 0.5:
        return str(rng.choice(symptoms))
    return ""


def _blood_pressure_for_symptom(rng: random.Random, symptom: str) -> str:
    symptom = symptom.lower()
    if symptom == "fever":
        systolic = rng.randint(100, 120)
        diastolic = rng.randint(60, 80)
    elif symptom == "cough":
        systolic = rng.randint(110, 130)
        diastolic = rng.randint(70, 85)
    else:
        systolic = rng.randint(110, 140)
        diastolic = rng.randint(70, 90)
    return f"{systolic}/{diastolic}"


def _body_temperature_for_symptom(rng: random.Random, symptom: str) -> float:
    if symptom == "fever":
        return round(rng.uniform(38.0, 40.0), 1)
    if symptom == "cough":
        return round(rng.uniform(36.5, 37.5), 1)
    return round(rng.uniform(36.0, 38.5), 1)


def _build_rows_for_chunk(chunk: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for item in chunk:
        patient_id = int(item["patient_id"])
        diagnosis = str(item["diagnosis"])
        stay_start = as_utc(item["stay_start"])
        stay_end = as_utc(item["stay_end"])
        medication_times = [as_utc(ts) for ts in item.get("medication_times", [])]
        day_cursor = stay_start.replace(hour=0, minute=0, second=0, microsecond=0)
        while day_cursor < stay_end:
            day_end = day_cursor + timedelta(days=1)
            meds = [ts for ts in medication_times if day_cursor <= ts < day_end]
            fixed_visits = 1 if len(meds) > 4 else 3
            visit_times: list[datetime] = []
            for _ in range(fixed_visits):
                hour = rng.randint(7, 21)
                ts = day_cursor + timedelta(hours=hour)
                if stay_start <= ts < stay_end:
                    visit_times.append(ts)
            for med_time in meds:
                follow_up = med_time + timedelta(hours=2)
                if follow_up < stay_end:
                    visit_times.append(follow_up)
            for ts in sorted(set(visit_times)):
                symptom = _symptom_for_diagnosis(rng, diagnosis)
                rows.append({"patient_id": patient_id, "visit_time": ts, "body_temperature": _body_temperature_for_symptom(rng, symptom), "blood_pressure": _blood_pressure_for_symptom(rng, symptom), "symptoms": symptom})
            day_cursor += timedelta(days=1)
    return rows


class VisitGenerator:
    def __init__(self, *, seed: int = 42, workers: int | None = None, chunk_size: int = 50000, write_batch_size: int = 50000):
        self.seed = seed
        self.workers = workers if workers is not None else max(1, min((os.cpu_count() or 2) - 1, 8))
        self.chunk_size = max(1, int(chunk_size))
        self.write_batch_size = max(1, int(write_batch_size))

    def generate_for_horizon(self, start_time: datetime, end_time: datetime) -> int:
        start_time = as_utc(start_time)
        end_time = as_utc(end_time)
        with session_scope() as session:
            admissions = (
                session.query(Admission)
                .filter(Admission.discharged_at > start_time)
                .filter(Admission.admitted_at < end_time)
                .all()
            )
            medications = (
                session.query(Medication)
                .filter(Medication.medication_time >= start_time)
                .filter(Medication.medication_time < end_time)
                .all()
            )
            meds_by_patient: dict[int, list[datetime]] = defaultdict(list)
            for med in medications:
                meds_by_patient[int(med.patient_id)].append(as_utc(med.medication_time))
            for values in meds_by_patient.values():
                values.sort()
            work_items: list[dict] = []
            for adm in admissions:
                adm_start = as_utc(adm.admitted_at)
                adm_end = as_utc(adm.discharged_at)
                stay_start = max(adm_start, start_time)
                stay_end = min(adm_end, end_time)
                if stay_start >= stay_end:
                    continue
                patient_id = int(adm.patient_id)
                medication_times = [ts for ts in meds_by_patient.get(patient_id, []) if stay_start <= ts < stay_end]
                work_items.append({"patient_id": patient_id, "diagnosis": adm.current_diagnosis or "", "stay_start": stay_start, "stay_end": stay_end, "medication_times": medication_times})
            chunks = list(_chunked(work_items, self.chunk_size))
            if not chunks:
                return 0
            if self.workers <= 1 or len(chunks) == 1:
                generated_iter = (_build_rows_for_chunk(chunk, self.seed + idx) for idx, chunk in enumerate(chunks))
            else:
                executor = ProcessPoolExecutor(max_workers=self.workers)
                generated_iter = executor.map(_build_rows_for_chunk, chunks, [self.seed + idx for idx in range(len(chunks))])
            inserted = 0
            try:
                for rows in generated_iter:
                    for batch in _chunked(rows, self.write_batch_size):
                        write_model_rows(Visit, batch)
                        session.bulk_insert_mappings(Visit, batch)
                        inserted += len(batch)
            finally:
                if self.workers > 1 and len(chunks) > 1:
                    executor.shutdown(wait=True)
        return inserted


__all__ = ["VisitGenerator"]
