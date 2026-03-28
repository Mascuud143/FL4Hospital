import argparse
import csv
import os
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

DEFAULT_INPUT_DIR = "outputs_next_hour"
DEFAULT_OUTPUT_DIR = "splits_next_hour"
DEFAULT_CHUNK_SIZE = 50000
DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) - 1)


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
        writer.writerows(rows)


def merge_part_files(part_paths: list[str], out_path: str, fieldnames: list[str]) -> None:
    initialize_csv(out_path, fieldnames)
    with open(out_path, "a", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        for part_path in part_paths:
            if not os.path.exists(part_path):
                continue
            with open(part_path, "r", encoding="utf-8", newline="") as in_f:
                reader = csv.DictReader(in_f)
                for row in reader:
                    writer.writerow(row)


def spool_room_files(
    in_path: str, chunk_size: int, temp_rooms_dir: str
) -> tuple[list[str], list[str], int]:
    room_paths: dict[str, str] = {}
    room_order: list[str] = []
    initialized: set[str] = set()
    total_rows = 0
    fieldnames: list[str] = []

    with open(in_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            return [], [], 0

        chunk_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in reader:
            client_id = (row.get("client_id") or "").strip()
            if client_id == "":
                continue
            if client_id not in room_paths:
                room_index = len(room_order)
                room_order.append(client_id)
                room_paths[client_id] = os.path.join(temp_rooms_dir, f"room_{room_index:06d}.csv")
            chunk_groups[client_id].append(row)
            total_rows += 1
            if total_rows % chunk_size == 0:
                _flush_room_chunk_groups(chunk_groups, room_paths, initialized, fieldnames)
                print(f"[split] rows_read={total_rows} rooms_seen={len(room_order)}", flush=True)
                chunk_groups = defaultdict(list)

        if chunk_groups:
            _flush_room_chunk_groups(chunk_groups, room_paths, initialized, fieldnames)
            print(f"[split] rows_read={total_rows} rooms_seen={len(room_order)}", flush=True)

    room_files = [room_paths[client_id] for client_id in room_order]
    return room_order, room_files, total_rows


def _flush_room_chunk_groups(
    chunk_groups: dict[str, list[dict[str, str]]],
    room_paths: dict[str, str],
    initialized: set[str],
    fieldnames: list[str],
) -> None:
    for client_id, rows in chunk_groups.items():
        room_path = room_paths[client_id]
        if room_path not in initialized:
            initialize_csv(room_path, fieldnames)
            initialized.add(room_path)
        append_rows(room_path, fieldnames, rows)


def _split_room_file(
    client_id: str,
    room_path: str,
    train_ratio: float,
    min_train_rows: int,
    fieldnames: list[str],
    train_part_dir: str,
    test_part_dir: str,
    part_index: int,
) -> dict[str, Any]:
    with open(room_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    rows.sort(key=lambda row: parse_ts(row.get("t")))
    split_idx = max(min_train_rows, int(len(rows) * train_ratio))
    split_idx = min(split_idx, len(rows))
    train_rows = rows[:split_idx]
    test_rows = rows[split_idx:]

    train_part_path = os.path.join(train_part_dir, f"train_{part_index:06d}.csv")
    test_part_path = os.path.join(test_part_dir, f"test_{part_index:06d}.csv")
    initialize_csv(train_part_path, fieldnames)
    initialize_csv(test_part_path, fieldnames)
    append_rows(train_part_path, fieldnames, train_rows)
    append_rows(test_part_path, fieldnames, test_rows)

    try:
        client_int = int(client_id)
    except ValueError:
        client_int = None

    return {
        "client_int": client_int,
        "train_count": len(train_rows),
        "test_count": len(test_rows),
        "train_part_path": train_part_path,
        "test_part_path": test_part_path,
    }


def split_room_files(
    room_ids: list[str],
    room_files: list[str],
    train_ratio: float,
    min_train_rows: int,
    workers: int,
    fieldnames: list[str],
    train_part_dir: str,
    test_part_dir: str,
) -> tuple[list[str], list[str], Counter[int], Counter[int], int, int]:
    train_part_paths: list[str] = []
    test_part_paths: list[str] = []
    train_counts: Counter[int] = Counter()
    test_counts: Counter[int] = Counter()
    train_total = 0
    test_total = 0

    jobs = list(enumerate(zip(room_ids, room_files)))
    total_clients = max(1, len(jobs))
    worker_count = max(1, min(workers, total_clients))

    if worker_count == 1:
        results = (
            _split_room_file(client_id, room_path, train_ratio, min_train_rows, fieldnames, train_part_dir, test_part_dir, index)
            for index, (client_id, room_path) in jobs
        )
        for idx, result in enumerate(results, start=1):
            _accumulate_split_result(
                result, train_part_paths, test_part_paths, train_counts, test_counts
            )
            train_total += result["train_count"]
            test_total += result["test_count"]
            if idx % 25 == 0 or idx == total_clients:
                print(
                    f"[split] rooms_processed={idx}/{total_clients} train_rows={train_total} test_rows={test_total}",
                    flush=True,
                )
        return train_part_paths, test_part_paths, train_counts, test_counts, train_total, test_total

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(
                _split_room_file,
                client_id,
                room_path,
                train_ratio,
                min_train_rows,
                fieldnames,
                train_part_dir,
                test_part_dir,
                index,
            ): index
            for index, (client_id, room_path) in jobs
        }
        completed = 0
        for future in as_completed(future_map):
            result = future.result()
            _accumulate_split_result(
                result, train_part_paths, test_part_paths, train_counts, test_counts
            )
            train_total += result["train_count"]
            test_total += result["test_count"]
            completed += 1
            if completed % 25 == 0 or completed == total_clients:
                print(
                    f"[split] rooms_processed={completed}/{total_clients} train_rows={train_total} test_rows={test_total}",
                    flush=True,
                )

    train_part_paths.sort()
    test_part_paths.sort()
    return train_part_paths, test_part_paths, train_counts, test_counts, train_total, test_total


def _accumulate_split_result(
    result: dict[str, Any],
    train_part_paths: list[str],
    test_part_paths: list[str],
    train_counts: Counter[int],
    test_counts: Counter[int],
) -> None:
    train_part_paths.append(result["train_part_path"])
    test_part_paths.append(result["test_part_path"])
    if result["client_int"] is not None:
        train_counts[result["client_int"]] = result["train_count"]
        test_counts[result["client_int"]] = result["test_count"]


def write_stats(path: str, train_counts: Counter[int], test_counts: Counter[int]) -> None:
    room_ids = sorted(set(train_counts) | set(test_counts))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["client_id", "train_rows", "test_rows"])
        writer.writeheader()
        for rid in room_ids:
            writer.writerow({"client_id": rid, "train_rows": train_counts.get(rid, 0), "test_rows": test_counts.get(rid, 0)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split next-hour rows into time-based per-room train/test sets.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directory containing next_hour_rows.csv")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for split output CSV files")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio per room")
    parser.add_argument("--min-train-rows", type=int, default=1, help="Minimum rows kept in train per room")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Rows read per CSV chunk")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Worker processes used for per-room splitting")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    train_out = os.path.join(output_dir, "next_hour_train.csv")
    test_out = os.path.join(output_dir, "next_hour_test.csv")
    stats_out = os.path.join(output_dir, "split_stats_by_room.csv")
    temp_root = os.path.join(output_dir, "_split_tmp")
    temp_rooms_dir = os.path.join(temp_root, "rooms")
    train_part_dir = os.path.join(temp_root, "train_parts")
    test_part_dir = os.path.join(temp_root, "test_parts")

    print(f"[split] input_path={in_path}", flush=True)
    print(f"[split] chunk_size={args.chunk_size}", flush=True)
    print(f"workers_used={max(1, args.workers)}", flush=True)

    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root, ignore_errors=True)
    os.makedirs(temp_rooms_dir, exist_ok=True)
    os.makedirs(train_part_dir, exist_ok=True)
    os.makedirs(test_part_dir, exist_ok=True)

    try:
        room_ids, room_files, total_rows = spool_room_files(in_path, args.chunk_size, temp_rooms_dir)
        if not room_files:
            initialize_csv(train_out, ["client_id"])
            initialize_csv(test_out, ["client_id"])
            write_stats(stats_out, Counter(), Counter())
            print("split_next_hour_by_room.py complete")
            print("next_hour_total=0 train=0 test=0")
            print(f"next_hour_train_path={train_out}")
            print(f"next_hour_test_path={test_out}")
            print(f"split_stats_path={stats_out}")
            return

        with open(room_files[0], "r", encoding="utf-8", newline="") as f:
            fieldnames = csv.DictReader(f).fieldnames or ["client_id"]

        initialize_csv(train_out, fieldnames)
        initialize_csv(test_out, fieldnames)

        train_parts, test_parts, train_counts, test_counts, train_total, test_total = split_room_files(
            room_ids,
            room_files,
            args.train_ratio,
            args.min_train_rows,
            args.workers,
            fieldnames,
            train_part_dir,
            test_part_dir,
        )
        merge_part_files(train_parts, train_out, fieldnames)
        merge_part_files(test_parts, test_out, fieldnames)
        write_stats(stats_out, train_counts, test_counts)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    print("split_next_hour_by_room.py complete")
    print(f"next_hour_total={total_rows} train={train_total} test={test_total}")
    print(f"next_hour_train_path={train_out}")
    print(f"next_hour_test_path={test_out}")
    print(f"split_stats_path={stats_out}")


if __name__ == "__main__":
    main()
