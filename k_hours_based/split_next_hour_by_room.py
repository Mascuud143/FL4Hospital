import argparse
import csv
import os
import shutil
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from typing import Any

from per_room_data import room_file_path
from runtime_defaults import (
    SPLIT_CSV_WRITE_BATCH_SIZE,
    SPLIT_DEFAULT_CHUNK_SIZE,
    SPLIT_DEFAULT_ROOM_SPLIT_WORKERS,
)


DEFAULT_INPUT_DIR = "outputs_next_hour"
DEFAULT_OUTPUT_DIR = "splits_next_hour"
DEFAULT_CHUNK_SIZE = SPLIT_DEFAULT_CHUNK_SIZE
DEFAULT_ROOM_SPLIT_WORKERS = SPLIT_DEFAULT_ROOM_SPLIT_WORKERS
CSV_WRITE_BATCH_SIZE = SPLIT_CSV_WRITE_BATCH_SIZE


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


def resolve_input_path(path_value: str, project_root: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    candidate_cwd = os.path.abspath(path_value)
    if os.path.exists(candidate_cwd):
        return candidate_cwd
    return os.path.abspath(os.path.join(project_root, path_value))


def initialize_csv(path: str, fieldnames: list[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_rows(path: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for start_idx in range(0, len(rows), CSV_WRITE_BATCH_SIZE):
            writer.writerows(rows[start_idx:start_idx + CSV_WRITE_BATCH_SIZE])


def split_room_rows(rows: list[dict[str, Any]], train_ratio: float, min_train_rows: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows.sort(key=lambda row: parse_ts(row.get("t")))
    split_idx = max(min_train_rows, int(len(rows) * train_ratio))
    split_idx = min(split_idx, len(rows))
    return rows[:split_idx], rows[split_idx:]


def write_room_split(
    client_id: str,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    split_dir: str,
    train_ratio: float,
    min_train_rows: int,
) -> tuple[int | None, int, int]:
    train_rows, test_rows = split_room_rows(rows, train_ratio, min_train_rows)
    train_path = room_file_path(split_dir, "train", client_id)
    test_path = room_file_path(split_dir, "test", client_id)
    initialize_csv(train_path, fieldnames)
    initialize_csv(test_path, fieldnames)
    append_rows(train_path, fieldnames, train_rows)
    append_rows(test_path, fieldnames, test_rows)
    try:
        client_int = int(client_id)
    except ValueError:
        client_int = None
    return client_int, len(train_rows), len(test_rows)


def split_room_rows_worker(
    client_id: str,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    split_dir: str,
    train_ratio: float,
    min_train_rows: int,
) -> tuple[str, int | None, int, int]:
    client_int, train_count, test_count = write_room_split(
        client_id,
        rows,
        fieldnames,
        split_dir,
        train_ratio,
        min_train_rows,
    )
    return client_id, client_int, train_count, test_count


def write_stats(path: str, train_counts: Counter[int], test_counts: Counter[int]) -> None:
    room_ids = sorted(set(train_counts) | set(test_counts))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["client_id", "train_rows", "test_rows"])
        writer.writeheader()
        pending_rows: list[dict[str, int]] = []
        for rid in room_ids:
            pending_rows.append({"client_id": rid, "train_rows": train_counts.get(rid, 0), "test_rows": test_counts.get(rid, 0)})
            if len(pending_rows) >= CSV_WRITE_BATCH_SIZE:
                writer.writerows(pending_rows)
                pending_rows = []
        if pending_rows:
            writer.writerows(pending_rows)


def dirty_room_path(temp_root: str, client_id: str) -> str:
    safe_client_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in client_id).strip("._-") or "unknown"
    return os.path.join(temp_root, f"dirty_room_{safe_client_id}.csv")


def read_csv_rows(path: str) -> tuple[list[str], list[dict[str, Any]]]:
    if not os.path.exists(path):
        return [], []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def append_dirty_rows(temp_root: str, client_id: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = dirty_room_path(temp_root, client_id)
    if not os.path.exists(path):
        initialize_csv(path, fieldnames)
    append_rows(path, fieldnames, rows)


def reprocess_dirty_room(
    client_id: str,
    split_dir: str,
    dirty_root: str,
    train_ratio: float,
    min_train_rows: int,
) -> tuple[int | None, int, int]:
    train_path = room_file_path(split_dir, "train", client_id)
    test_path = room_file_path(split_dir, "test", client_id)
    train_fields, train_rows = read_csv_rows(train_path)
    test_fields, test_rows = read_csv_rows(test_path)
    dirty_fields, dirty_rows = read_csv_rows(dirty_room_path(dirty_root, client_id))
    fieldnames = train_fields or test_fields or dirty_fields
    all_rows = [*train_rows, *test_rows, *dirty_rows]
    return write_room_split(client_id, all_rows, fieldnames, split_dir, train_ratio, min_train_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split next-hour rows into time-based per-room train/test sets.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directory containing next_hour_rows.csv")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for split output CSV files")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio per room")
    parser.add_argument("--min-train-rows", type=int, default=1, help="Minimum rows kept in train per room")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Progress logging interval while scanning")
    parser.add_argument("--workers", type=int, default=DEFAULT_ROOM_SPLIT_WORKERS, help="Workers used for parallel room sorting and splitting")
    parser.add_argument("--csv-write-batch-size", type=int, default=CSV_WRITE_BATCH_SIZE, help="Rows buffered per CSV write")
    return parser.parse_args()
def run_split_mode(
    in_path: str,
    output_dir: str,
    stats_out: str,
    dirty_root: str,
    *,
    train_ratio: float,
    min_train_rows: int,
    chunk_size: int,
    workers: int,
) -> tuple[int, int, int, int]:
    train_counts: Counter[int] = Counter()
    test_counts: Counter[int] = Counter()
    dirty_rooms: set[str] = set()
    closed_rooms: set[str] = set()
    total_rows = 0
    rows_read = 0
    rooms_dispatched = 0
    fieldnames: list[str] = []
    current_client_id: str | None = None
    current_rows: list[dict[str, Any]] = []
    current_is_dirty = False

    def drain_done(future_map: dict[Any, str], *, wait_for_all: bool, block_until_done: bool = False) -> None:
        if not future_map:
            return
        if wait_for_all:
            done_futures = list(future_map)
        elif block_until_done:
            done, _ = wait(tuple(future_map), return_when=FIRST_COMPLETED)
            done_futures = list(done)
        else:
            done_futures = [future for future in list(future_map) if future.done()]
        for future in done_futures:
            client_id, client_int, train_count, test_count = future.result()
            future_map.pop(future, None)
            if client_int is not None:
                train_counts[client_int] = train_count
                test_counts[client_int] = test_count

    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        future_map: dict[Any, str] = {}

        def flush_current_room() -> None:
            nonlocal current_rows, current_client_id, rooms_dispatched
            if current_client_id is None or not current_rows:
                return
            if current_is_dirty:
                append_dirty_rows(dirty_root, current_client_id, fieldnames, current_rows)
            else:
                while len(future_map) >= max(1, workers):
                    drain_done(future_map, wait_for_all=False, block_until_done=True)
                future = executor.submit(
                    split_room_rows_worker,
                    current_client_id,
                    current_rows,
                    fieldnames,
                    output_dir,
                    train_ratio,
                    min_train_rows,
                )
                future_map[future] = current_client_id
                rooms_dispatched += 1
            closed_rooms.add(current_client_id)
            current_rows = []

        with open(in_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            for row in reader:
                client_id = str(row.get("client_id") or "").strip()
                if client_id == "":
                    continue
                rows_read += 1
                total_rows += 1
                if current_client_id is None:
                    current_client_id = client_id
                    current_is_dirty = False
                elif client_id != current_client_id:
                    flush_current_room()
                    current_client_id = client_id
                    current_is_dirty = client_id in closed_rooms
                    if current_is_dirty:
                        dirty_rooms.add(client_id)
                current_rows.append(row)
                drain_done(future_map, wait_for_all=False)
                if rows_read % max(1, chunk_size) == 0:
                    print(
                        f"[split] rows_read={rows_read} rooms_dispatched={rooms_dispatched} active_workers={len(future_map)} dirty_rooms={len(dirty_rooms)}",
                        flush=True,
                    )
            flush_current_room()
        drain_done(future_map, wait_for_all=True)

    for client_id in sorted(dirty_rooms, key=lambda value: int(value) if value.isdigit() else value):
        client_int, train_count, test_count = reprocess_dirty_room(
            client_id,
            output_dir,
            dirty_root,
            train_ratio,
            min_train_rows,
        )
        if client_int is not None:
            train_counts[client_int] = train_count
            test_counts[client_int] = test_count
    write_stats(stats_out, train_counts, test_counts)
    return total_rows, sum(train_counts.values()), sum(test_counts.values()), len(dirty_rooms)


def main() -> None:
    args = parse_args()
    global CSV_WRITE_BATCH_SIZE
    CSV_WRITE_BATCH_SIZE = max(1, int(args.csv_write_batch_size))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if os.path.isabs(args.input_dir):
        input_dir = args.input_dir
    elif args.input_dir == DEFAULT_INPUT_DIR:
        input_dir = os.path.join(script_dir, args.input_dir)
    else:
        input_dir = resolve_input_path(args.input_dir, project_root)

    if os.path.isabs(args.output_dir):
        output_dir = args.output_dir
    elif args.output_dir == DEFAULT_OUTPUT_DIR:
        output_dir = os.path.join(script_dir, args.output_dir)
    else:
        output_dir = os.path.abspath(args.output_dir)

    in_path = os.path.join(input_dir, "next_hour_rows.csv")
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Missing input file: {in_path}")

    stats_out = os.path.join(output_dir, "split_stats_by_room.csv")
    temp_root = os.path.join(output_dir, "_split_tmp")
    dirty_root = os.path.join(temp_root, "dirty_rooms")
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")

    print(f"[split] input_path={in_path}", flush=True)
    print(f"[split] csv_write_batch_size={CSV_WRITE_BATCH_SIZE}", flush=True)
    print("[split] mode=stream_workers", flush=True)
    print(f"[split] workers={args.workers}", flush=True)

    os.makedirs(output_dir, exist_ok=True)
    shutil.rmtree(train_dir, ignore_errors=True)
    shutil.rmtree(test_dir, ignore_errors=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root, ignore_errors=True)
    os.makedirs(dirty_root, exist_ok=True)

    try:
        total_rows, train_total, test_total, dirty_room_count = run_split_mode(
            in_path,
            output_dir,
            stats_out,
            dirty_root,
            train_ratio=args.train_ratio,
            min_train_rows=args.min_train_rows,
            chunk_size=max(1, int(args.chunk_size)),
            workers=max(1, int(args.workers)),
        )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
    print("split_next_hour_by_room.py complete")
    print(f"next_hour_total={total_rows} train={train_total} test={test_total}")
    print(f"dirty_rooms_detected={dirty_room_count}")
    print(f"next_hour_train_dir={train_dir}")
    print(f"next_hour_test_dir={test_dir}")
    print(f"split_stats_path={stats_out}")


if __name__ == "__main__":
    main()
