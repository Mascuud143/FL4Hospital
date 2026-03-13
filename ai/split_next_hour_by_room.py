import argparse
import csv
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

DEFAULT_INPUT_DIR = "outputs_next_hour"
DEFAULT_OUTPUT_DIR = "splits_next_hour"


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


def read_csv(path: str) -> tuple[list[dict[str, str]], list[str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return rows, fieldnames


def split_rows_by_client(
    rows: list[dict[str, str]], train_ratio: float, min_train_rows: int
) -> tuple[list[dict[str, str]], list[dict[str, str]], Counter[int], Counter[int]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        client_id = (row.get("client_id") or "").strip()
        if client_id == "":
            continue
        grouped[client_id].append(row)

    train_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []
    train_counts: Counter[int] = Counter()
    test_counts: Counter[int] = Counter()

    for client_id, client_rows in grouped.items():
        client_rows.sort(key=lambda r: parse_ts(r.get("t")))
        split_idx = max(min_train_rows, int(len(client_rows) * train_ratio))
        split_idx = min(split_idx, len(client_rows))
        train_chunk = client_rows[:split_idx]
        test_chunk = client_rows[split_idx:]
        train_rows.extend(train_chunk)
        test_rows.extend(test_chunk)
        try:
            cid = int(client_id)
        except ValueError:
            continue
        train_counts[cid] = len(train_chunk)
        test_counts[cid] = len(test_chunk)

    return train_rows, test_rows, train_counts, test_counts


def write_csv(path: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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

    rows, fieldnames = read_csv(in_path)
    train_rows, test_rows, train_counts, test_counts = split_rows_by_client(rows, args.train_ratio, args.min_train_rows)
    train_out = os.path.join(output_dir, "next_hour_train.csv")
    test_out = os.path.join(output_dir, "next_hour_test.csv")
    stats_out = os.path.join(output_dir, "split_stats_by_room.csv")
    write_csv(train_out, fieldnames, train_rows)
    write_csv(test_out, fieldnames, test_rows)
    write_stats(stats_out, train_counts, test_counts)

    print("split_next_hour_by_room.py complete")
    print(f"next_hour_total={len(rows)} train={len(train_rows)} test={len(test_rows)}")
    print(f"next_hour_train_path={train_out}")
    print(f"next_hour_test_path={test_out}")
    print(f"split_stats_path={stats_out}")


if __name__ == "__main__":
    main()
