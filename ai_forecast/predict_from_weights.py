import argparse
import os

import numpy as np
import pandas as pd

from fl_client import make_model, sanitize_targets, set_params
from next_hour_schema import CHANGE_BASELINE_COLUMNS, TARGET_COLUMNS, row_to_input_vector

DEFAULT_SPLIT_DIR = "splits_next_hour"
DEFAULT_WEIGHTS_PATH = "fl_weights_sim/latest_global_weights.npz"
DEFAULT_OUT_DIR = "fl_predictions"


def resolve_path(value: str, script_dir: str) -> str:
    return value if os.path.isabs(value) else os.path.join(script_dir, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ai_forecast predictions from saved FL global weights.")
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR, help="Directory with next_hour_test.csv")
    parser.add_argument("--weights-path", default=DEFAULT_WEIGHTS_PATH, help="Path to saved global weights .npz")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory to write predictions CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    split_dir = resolve_path(args.split_dir, script_dir)
    weights_path = resolve_path(args.weights_path, script_dir)
    out_dir = resolve_path(args.out_dir, script_dir)

    test_path = os.path.join(split_dir, "next_hour_test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test split: {test_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing weights file: {weights_path}")

    test_df = sanitize_targets(pd.read_csv(test_path))
    x_test = row_to_input_vector(test_df)

    model = make_model(x_test.shape[1])
    raw = np.load(weights_path)
    params = [raw[key] for key in sorted(raw.files, key=lambda item: int(item.split("_")[1]))]
    set_params(model, params)
    y_pred = pd.DataFrame(model.predict(x_test), columns=TARGET_COLUMNS, index=test_df.index)

    pred_df = test_df[["client_id", "t"]].copy()
    for target in TARGET_COLUMNS:
        pred_df[f"{target}_true"] = test_df[target].values
        pred_df[f"{target}_pred"] = y_pred[target].values
    pred_df["y_airflow_pred_binary"] = (y_pred["y_airflow"] >= 0.5).astype(int).values
    pred_df["y_any_change_true"] = test_df["y_any_change"].round().astype(int).values
    current_values = test_df[CHANGE_BASELINE_COLUMNS].astype(float).to_numpy()
    predicted_values = y_pred[TARGET_COLUMNS].astype(float).to_numpy()
    from next_hour_schema import next_hour_change_flags

    pred_df["y_any_change_pred"] = next_hour_change_flags(current_values, predicted_values)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "next_hour_predictions.csv")
    pred_df.to_csv(out_path, index=False)

    print("predict_from_weights.py complete")
    print(f"weights_path={weights_path}")
    print(f"test_rows={len(test_df)}")
    print(f"predictions_path={out_path}")


if __name__ == "__main__":
    main()
