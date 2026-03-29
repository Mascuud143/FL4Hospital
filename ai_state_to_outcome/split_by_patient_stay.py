import argparse
import csv
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone

DEFAULT_INPUT_DIR = "rows"
DEFAULT_OUTPUT_DIR = "splits"
DEFAULT_CHUNK_SIZE = 50000


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
        writer.writerows(rows)


def _flush_admission_chunk_groups(
    chunk_groups: dict[str, list[dict[str, str]]],
    admission_paths: dict[str, str],
    initialized: set[str],
    fieldnames: list[str],
) -> None:
    for admission_id, rows in chunk_groups.items():
        admission_path = admission_paths[admission_id]
        if admission_path not in initialized:
            initialize_csv(admission_path, fieldnames)
            initialized.add(admission_path)
        append_rows(admission_path, fieldnames, rows)


def spool_admission_files(
    in_path: str,
    chunk_size: int,
    temp_admissions_dir: str,
) -> tuple[list[str], list[str], int, list[str]]:
    admission_paths: dict[str, str] = {}
    admission_order: list[str] = []
    initialized: set[str] = set()
    total_rows = 0

    with open(in_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            return [], [], 0, []

        chunk_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in reader:
            admission_id = (row.get("admission_id") or "").strip()
            if admission_id == "":
                continue
            if admission_id not in admission_paths:
                admission_index = len(admission_order)
                admission_order.append(admission_id)
                admission_paths[admission_id] = os.path.join(temp_admissions_dir, f"admission_{admission_index:06d}.csv")
            chunk_groups[admission_id].append(row)
            total_rows += 1
            if total_rows % chunk_size == 0:
                _flush_admission_chunk_groups(chunk_groups, admission_paths, initialized, fieldnames)
                print(f"[split] rows_read={total_rows} admissions_seen={len(admission_order)}", flush=True)
                chunk_groups = defaultdict(list)

        if chunk_groups:
            _flush_admission_chunk_groups(chunk_groups, admission_paths, initialized, fieldnames)
            print(f"[split] rows_read={total_rows} admissions_seen={len(admission_order)}", flush=True)

    admission_files = [admission_paths[admission_id] for admission_id in admission_order]
    return admission_order, admission_files, total_rows, fieldnames


def split_admission_file(
    admission_path: str,
    test_fraction: float,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with open(admission_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    rows.sort(key=lambda row: parse_ts(row.get("event_time")))
    if len(rows) <= 1:
        return rows, []

    split_idx = int(len(rows) * (1.0 - test_fraction))
    split_idx = max(1, min(split_idx, len(rows) - 1))
    return rows[:split_idx], rows[split_idx:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split Task #2 event rows by patient stay in time order.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directory containing state_to_outcome_rows.csv")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to write train/test CSV files")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Fraction of each admission to reserve for testing")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Rows read per CSV chunk while spooling admissions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = resolve_dir(args.input_dir, script_dir)
    output_dir = resolve_dir(args.output_dir, script_dir)
    input_path = os.path.join(input_dir, "state_to_outcome_rows.csv")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing input rows file: {input_path}")

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "state_to_outcome_train.csv")
    test_path = os.path.join(output_dir, "state_to_outcome_test.csv")
    temp_root = os.path.join(output_dir, "_split_tmp")
    temp_admissions_dir = os.path.join(temp_root, "admissions")

    if os.path.exists(temp_root):
        shutil.rmtree(temp_root, ignore_errors=True)
    os.makedirs(temp_admissions_dir, exist_ok=True)

    total_rows = 0
    train_rows = 0
    test_rows = 0

    try:
        admission_ids, admission_files, total_rows, fieldnames = spool_admission_files(
            input_path,
            args.chunk_size,
            temp_admissions_dir,
        )

        if not fieldnames:
            initialize_csv(train_path, ["admission_id"])
            initialize_csv(test_path, ["admission_id"])
            print("split_by_patient_stay.py complete")
            print("rows_total=0")
            print("rows_train=0")
            print("rows_test=0")
            print(f"train_path={train_path}")
            print(f"test_path={test_path}")
            return

        initialize_csv(train_path, fieldnames)
        initialize_csv(test_path, fieldnames)

        total_admissions = len(admission_files)
        for idx, admission_path in enumerate(admission_files, start=1):
            train_part, test_part = split_admission_file(admission_path, args.test_fraction)
            append_rows(train_path, fieldnames, train_part)
            append_rows(test_path, fieldnames, test_part)
            train_rows += len(train_part)
            test_rows += len(test_part)
            if idx % 500 == 0 or idx == total_admissions:
                print(
                    f"[split] admissions_processed={idx}/{total_admissions} train_rows={train_rows} test_rows={test_rows}",
                    flush=True,
                )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    print("split_by_patient_stay.py complete")
    print(f"rows_total={total_rows}")
    print(f"rows_train={train_rows}")
    print(f"rows_test={test_rows}")
    print(f"train_path={train_path}")
    print(f"test_path={test_path}")


if __name__ == "__main__":
    main()
