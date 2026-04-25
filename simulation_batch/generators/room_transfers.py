from __future__ import annotations

# Build room transfers.
# - Picks stays to transfer
# - Splits one stay into two room parts
# - Creates the new room assignment - apply_room_transfers()

from datetime import timedelta

from simulation_batch.generators.room_assignments import apply_transfer


def apply_room_transfers(
    *,
    session,
    admissions_created: list[dict],
    change_room_prob: float,
    min_days_before_transfer: int,
    min_days_after_transfer: int,
    rooms,
    room_occupancy,
    rng,
) -> None:
    transfer_rooms = list(rooms)
    max_room_number = max(int(room.room_number) for room in rooms) if rooms else 100
    for admission_info in admissions_created:
        if rng.random() > change_room_prob:
            continue
        admit = admission_info["admitted_at"]
        discharge = admission_info["discharged_at"]
        admission_id = admission_info["admission_id"]
        patient_id = admission_info["patient_id"]
        total_days = (discharge - admit).days
        if total_days < min_days_before_transfer + min_days_after_transfer + 1:
            continue
        transfer_day = rng.randint(
            min_days_before_transfer,
            max(min_days_before_transfer, total_days - min_days_after_transfer),
        )
        transfer_time = admit + timedelta(days=transfer_day)
        if transfer_time <= admit or transfer_time >= discharge:
            continue
        max_room_number = apply_transfer(
            session=session,
            admission_id=admission_id,
            patient_id=patient_id,
            admit=admit,
            discharge=discharge,
            transfer_time=transfer_time,
            rooms=rooms,
            transfer_rooms=transfer_rooms,
            room_occupancy=room_occupancy,
            max_room_number=max_room_number,
        )
