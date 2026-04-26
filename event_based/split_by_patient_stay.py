import argparse
import csv
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone

from per_room_data import room_file_path


DEFAULT_INPUT_DIR = "rows"
DEFAULT_OUTPUT_DIR = "splits"
DEFAULT_CHUNK_SIZE = 50000
CSV_WRITE_BATCH_SIZE = 50000


def parse_ts(value: str | None) -> datetime:
    if value is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    raw = value.strip()
    if raw == "":
        return datetime.min.replace(tzinfo=timezone.utc)
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def resolve_dir(value: str, script_dir: str) -> str:
    return value if os.path.isabs(value) else os.path.join(script_dir, value)


def initialize_csv(path: str, fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_rows(path: str, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for start_idx in range(0, len(rows), CSV_WRITE_BATCH_SIZE):
            writer.writerows(rows[start_idx:start_idx + CSV_WRITE_BATCH_SIZE])


def _flush_room_chunk_groups(
    chunk_groups: dict[str, list[dict[str, str]]],
    room_paths: dict[str, str],
    initialized: set[str],
    fieldnames: list[str],
) -> None:
    for room_id, rows in chunk_groups.items():
        room_path = room_paths[room_id]
        if room_path not in initialized:
            initialize_csv(room_path, fieldnames)
            initialized.add(room_path)
        append_rows(room_path, fieldnames, rows)


def spool_room_files(
    in_path: str,
    chunk_size: int,
    temp_rooms_dir: str,
) -> tuple[list[str], list[str], int, list[str]]:
    room_paths: dict[str, str] = {}
    room_order: list[str] = []
    initialized: set[str] = set()
    total_rows = 0

    with open(in_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            return [], [], 0, []

        chunk_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in reader:
            room_id = str(row.get("room_id") or "").strip()
            if room_id == "":
                continue
            if room_id not in room_paths:
                room_index = len(room_order)
                room_order.append(room_id)
                room_paths[room_id] = os.path.join(temp_rooms_dir, f"room_{room_index:06d}.csv")
            chunk_groups[room_id].append(row)
            total_rows += 1
            if total_rows % chunk_size == 0:
                _flush_room_chunk_groups(chunk_groups, room_paths, initialized, fieldnames)
                print(f"[split] rows_read={total_rows} rooms_seen={len(room_order)}", flush=True)
                chunk_groups = defaultdict(list)

        if chunk_groups:
            _flush_room_chunk_groups(chunk_groups, room_paths, initialized, fieldnames)
            print(f"[split] rows_read={total_rows} rooms_seen={len(room_order)}", flush=True)

    room_files = [room_paths[room_id] for room_id in room_order]
    return room_order, room_files, total_rows, fieldnames


def split_room_file(room_path: str, train_ratio: float, min_train_rows: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with open(room_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    rows.sort(key=lambda row: parse_ts(row.get("event_time")))
    split_idx = max(min_train_rows, int(len(rows) * train_ratio))
    split_idx = min(split_idx, len(rows))
    return rows[:split_idx], rows[split_idx:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split event-based rows into time-based per-room train/test sets.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directory containing event_based_rows.csv")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to write train/test CSV files")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio per room")
    parser.add_argument("--min-train-rows", type=int, default=1, help="Minimum rows kept in train per room")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Rows read per CSV chunk while spooling rooms")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = resolve_dir(args.input_dir, script_dir)
    output_dir = resolve_dir(args.output_dir, script_dir)
    input_path = os.path.join(input_dir, "event_based_rows.csv")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing input rows file: {input_path}")

    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    temp_root = os.path.join(output_dir, "_split_tmp")
    temp_rooms_dir = os.path.join(temp_root, "rooms")

    shutil.rmtree(train_dir, ignore_errors=True)
    shutil.rmtree(test_dir, ignore_errors=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root, ignore_errors=True)
    os.makedirs(temp_rooms_dir, exist_ok=True)

    total_rows = 0
    train_rows = 0
    test_rows = 0

    try:
        room_ids, room_files, total_rows, fieldnames = spool_room_files(
            input_path,
            args.chunk_size,
            temp_rooms_dir,
        )

        if not fieldnames:
            print("split_by_patient_stay.py complete")
            print("rows_total=0")
            print("rows_train=0")
            print("rows_test=0")
            print(f"train_dir={train_dir}")
            print(f"test_dir={test_dir}")
            return

        total_rooms = len(room_files)
        for idx, (room_id, room_path) in enumerate(zip(room_ids, room_files, strict=False), start=1):
            train_part, test_part = split_room_file(room_path, args.train_ratio, args.min_train_rows)
            train_path = room_file_path(output_dir, "train", room_id)
            test_path = room_file_path(output_dir, "test", room_id)
            initialize_csv(train_path, fieldnames)
            initialize_csv(test_path, fieldnames)
            append_rows(train_path, fieldnames, train_part)
            append_rows(test_path, fieldnames, test_part)
            train_rows += len(train_part)
            test_rows += len(test_part)
            if idx % 100 == 0 or idx == total_rooms:
                print(
                    f"[split] rooms_processed={idx}/{total_rooms} train_rows={train_rows} test_rows={test_rows}",
                    flush=True,
                )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    print("split_by_patient_stay.py complete")
    print(f"rows_total={total_rows}")
    print(f"rows_train={train_rows}")
    print(f"rows_test={test_rows}")
    print(f"train_dir={train_dir}")
    print(f"test_dir={test_dir}")


if __name__ == "__main__":
    main()
