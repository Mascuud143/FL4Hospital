import argparse
import os

import pandas as pd

DEFAULT_INPUT_DIR = "rows"
DEFAULT_OUTPUT_DIR = "splits"


def resolve_dir(value: str, script_dir: str) -> str:
    return value if os.path.isabs(value) else os.path.join(script_dir, value)


def split_rows(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, group in df.groupby("admission_id", sort=False):
        ordered = group.sort_values("event_time", kind="stable")
        if len(ordered) == 1:
            train_parts.append(ordered)
            continue
        split_idx = int(len(ordered) * (1.0 - test_fraction))
        split_idx = max(1, min(split_idx, len(ordered) - 1))
        train_parts.append(ordered.iloc[:split_idx])
        test_parts.append(ordered.iloc[split_idx:])

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=df.columns)
    return train_df, test_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split Task #2 event rows by patient stay in time order.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Directory containing state_to_outcome_rows.csv")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to write train/test CSV files")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Fraction of each admission to reserve for testing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = resolve_dir(args.input_dir, script_dir)
    output_dir = resolve_dir(args.output_dir, script_dir)
    input_path = os.path.join(input_dir, "state_to_outcome_rows.csv")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing input rows file: {input_path}")

    df = pd.read_csv(input_path)
    if "event_time" not in df.columns:
        raise ValueError("Input rows are missing event_time")
    train_df, test_df = split_rows(df, test_fraction=args.test_fraction)

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "state_to_outcome_train.csv")
    test_path = os.path.join(output_dir, "state_to_outcome_test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("split_by_patient_stay.py complete")
    print(f"rows_total={len(df)}")
    print(f"rows_train={len(train_df)}")
    print(f"rows_test={len(test_df)}")
    print(f"train_path={train_path}")
    print(f"test_path={test_path}")


if __name__ == "__main__":
    main()
