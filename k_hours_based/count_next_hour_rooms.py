import argparse
import csv
import os
import time

DEFAULT_INPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "outputs_next_hour_dashboard",
    "next_hour_rows.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count distinct room_id values in a next-hour dashboard CSV."
    )
    parser.add_argument(
        "--input-path",
        default=DEFAULT_INPUT_PATH,
        help="Path to the CSV file to inspect.",
    )
    parser.add_argument(
        "--show-ids",
        action="store_true",
        help="Print the distinct room_id values after the count.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1_000_000,
        help="Print a progress update every N rows. Use 0 to disable.",
    )
    return parser.parse_args()


def collect_room_ids(input_path: str, progress_every: int) -> tuple[list[str], int]:
    room_ids: set[str] = set()
    row_count = 0
    started = time.perf_counter()

    with open(input_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            return [], 0
        try:
            room_id_index = header.index("room_id")
        except ValueError as exc:
            raise ValueError(f"'room_id' column not found in {input_path}") from exc

        for row in reader:
            row_count += 1
            if room_id_index >= len(row):
                continue
            room_id = row[room_id_index].strip()
            if room_id:
                room_ids.add(room_id)
            if progress_every > 0 and row_count % progress_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"rows_processed={row_count} unique_room_ids={len(room_ids)} elapsed_seconds={elapsed:.1f}"
                )

    return sorted(room_ids), row_count


def main() -> None:
    args = parse_args()
    room_ids, row_count = collect_room_ids(args.input_path, args.progress_every)
    print(f"rows_processed={row_count}")
    print(f"room_count={len(room_ids)}")
    if args.show_ids:
        for room_id in room_ids:
            print(room_id)


if __name__ == "__main__":
    main()
