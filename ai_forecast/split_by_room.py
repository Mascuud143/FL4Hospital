import argparse
import csv
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

DEFAULT_INPUT_DIR = "outputs"
DEFAULT_OUTPUT_DIR = "splits"


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
    rows: list[dict[str, str]],
    train_ratio: float,
    min_train_rows: int,
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
        n = len(client_rows)
        split_idx = int(n * train_ratio)
        split_idx = max(min_train_rows, split_idx)
        split_idx = min(split_idx, n)

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


def write_stats(
    path: str,
    train_counts_a: Counter[int],
    test_counts_a: Counter[int],
    train_counts_b: Counter[int],
    test_counts_b: Counter[int],
) -> None:
    room_ids = sorted(set(train_counts_a) | set(test_counts_a) | set(train_counts_b) | set(test_counts_b))
    with open(path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "client_id",
            "model_a_train_rows",
            "model_a_test_rows",
            "model_b_train_rows",
            "model_b_test_rows",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rid in room_ids:
            writer.writerow(
                {
                    "client_id": rid,
                    "model_a_train_rows": train_counts_a.get(rid, 0),
                    "model_a_test_rows": test_counts_a.get(rid, 0),
                    "model_b_train_rows": train_counts_b.get(rid, 0),
                    "model_b_test_rows": test_counts_b.get(rid, 0),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split model_a/model_b rows into time-based per-room train/test.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directory containing model_a_rows.csv and model_b_rows.csv")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for split output CSV files")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio per room (default 0.8)")
    parser.add_argument(
        "--min-train-rows",
        type=int,
        default=1,
        help="Minimum rows kept in train per room (default 1)",
    )
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

    model_a_in = os.path.join(input_dir, "model_a_rows.csv")
    model_b_in = os.path.join(input_dir, "model_b_rows.csv")

    if not os.path.exists(model_a_in):
        raise FileNotFoundError(f"Missing input file: {model_a_in}")
    if not os.path.exists(model_b_in):
        raise FileNotFoundError(f"Missing input file: {model_b_in}")

    model_a_rows, model_a_fields = read_csv(model_a_in)
    model_b_rows, model_b_fields = read_csv(model_b_in)

    a_train, a_test, a_train_counts, a_test_counts = split_rows_by_client(
        model_a_rows, args.train_ratio, args.min_train_rows
    )
    b_train, b_test, b_train_counts, b_test_counts = split_rows_by_client(
        model_b_rows, args.train_ratio, args.min_train_rows
    )

    model_a_train_out = os.path.join(output_dir, "model_a_train.csv")
    model_a_test_out = os.path.join(output_dir, "model_a_test.csv")
    model_b_train_out = os.path.join(output_dir, "model_b_train.csv")
    model_b_test_out = os.path.join(output_dir, "model_b_test.csv")
    stats_out = os.path.join(output_dir, "split_stats_by_room.csv")

    write_csv(model_a_train_out, model_a_fields, a_train)
    write_csv(model_a_test_out, model_a_fields, a_test)
    write_csv(model_b_train_out, model_b_fields, b_train)
    write_csv(model_b_test_out, model_b_fields, b_test)
    write_stats(stats_out, a_train_counts, a_test_counts, b_train_counts, b_test_counts)

    print("split_by_room.py complete")
    print(f"model_a_total={len(model_a_rows)} train={len(a_train)} test={len(a_test)}")
    print(f"model_b_total={len(model_b_rows)} train={len(b_train)} test={len(b_test)}")
    print(f"model_a_train_path={model_a_train_out}")
    print(f"model_a_test_path={model_a_test_out}")
    print(f"model_b_train_path={model_b_train_out}")
    print(f"model_b_test_path={model_b_test_out}")
    print(f"split_stats_path={stats_out}")


if __name__ == "__main__":
    main()
