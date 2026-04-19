from __future__ import annotations

import os
import random
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from typing import Dict, Iterable

from persistence.database import session_scope
from persistence.models.admission import Admission
from persistence.models.medication import Medication
from simulation_batch.csv_filestorage import write_model_rows
from simulation_batch.room_engine import _as_utc


def _chunked(items: list[dict], size: int) -> Iterable[list[dict]]:
    for idx in range(0, len(items), size):
        yield items[idx:idx + size]


def _build_rows_for_chunk(chunk: list[dict], diagnoses: Dict, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []

    for admission in chunk:
        diagnosis = admission["diagnosis"]
        if diagnosis not in diagnoses:
            continue

        med_schedule = diagnoses[diagnosis]["medications"]
        stay_start = _as_utc(admission["stay_start"])
        stay_end = _as_utc(admission["stay_end"])
        day_cursor = stay_start.replace(hour=0, minute=0, second=0, microsecond=0)

        while day_cursor < stay_end:
            for drug_name, hours in med_schedule.items():
                for hour in hours:
                    ts = day_cursor + timedelta(hours=hour if hour >= 0 else rng.randint(0, 23))
                    if ts < stay_start or ts >= stay_end:
                        continue
                    rows.append(
                        {
                            "patient_id": admission["patient_id"],
                            "medication_time": ts,
                            "drug_name": drug_name,
                            "route": "oral",
                            "dose": "1 dose",
                            "status": "taken",
                        }
                    )
            day_cursor += timedelta(days=1)

    return rows


class MedicationGenerator:
    """
    Generates medication rows per patient stay.
    Row scheduling can be parallelized, but writes stay in the parent process.
    """

    def __init__(
        self,
        *,
        seed: int = 42,
        diagnoses: Dict,
        workers: int | None = None,
        chunk_size: int = 50000,
        write_batch_size: int = 50000,
    ):
        self.seed = seed
        self.diagnoses = diagnoses
        self.workers = workers if workers is not None else max(1, min((os.cpu_count() or 2) - 1, 8))
        self.chunk_size = max(1, int(chunk_size))
        self.write_batch_size = max(1, int(write_batch_size))

    def generate_for_horizon(self, start_time: datetime, end_time: datetime) -> int:
        start_time = _as_utc(start_time)
        end_time = _as_utc(end_time)

        with session_scope() as session:
            admissions = (
                session.query(Admission)
                .filter(Admission.discharged_at > start_time)
                .filter(Admission.admitted_at < end_time)
                .all()
            )

            work_items = []
            for adm in admissions:
                diagnosis = adm.current_diagnosis
                if diagnosis not in self.diagnoses:
                    continue

                adm_start = _as_utc(adm.admitted_at)
                adm_end = _as_utc(adm.discharged_at)
                stay_start = max(adm_start, start_time)
                stay_end = min(adm_end, end_time)
                if stay_start >= stay_end:
                    continue

                work_items.append(
                    {
                        "patient_id": adm.patient_id,
                        "diagnosis": diagnosis,
                        "stay_start": stay_start,
                        "stay_end": stay_end,
                    }
                )

            chunks = list(_chunked(work_items, self.chunk_size))
            if not chunks:
                return 0

            if self.workers <= 1 or len(chunks) == 1:
                generated = [
                    _build_rows_for_chunk(chunk, self.diagnoses, self.seed + idx)
                    for idx, chunk in enumerate(chunks)
                ]
            else:
                with ProcessPoolExecutor(max_workers=self.workers) as executor:
                    generated = list(
                        executor.map(
                            _build_rows_for_chunk,
                            chunks,
                            [self.diagnoses] * len(chunks),
                            [self.seed + idx for idx in range(len(chunks))],
                        )
                    )

            inserted = 0
            for rows in generated:
                for batch in _chunked(rows, self.write_batch_size):
                    write_model_rows(Medication, batch)
                    session.bulk_insert_mappings(Medication, batch)
                    inserted += len(batch)

        return inserted
