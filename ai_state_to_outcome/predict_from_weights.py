import argparse
import os
from collections import OrderedDict

import numpy as np
import pandas as pd

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "PyTorch is required for ai_state_to_outcome inference from FL weights. Install torch before running this mode."
    ) from exc

from fl_client import get_input_dim, make_model, sanitize_rows
from schema import REGRESSION_TARGET_COLUMNS, row_to_input_vector

DEFAULT_SPLIT_DIR = "splits"
DEFAULT_WEIGHTS_PATH = "fl_weights/latest_global_weights.npz"
DEFAULT_OUT_DIR = "fl_predictions"


def resolve_path(value: str, script_dir: str) -> str:
    return value if os.path.isabs(value) else os.path.join(script_dir, value)


def load_model(weights_path: str):
    model = make_model(get_input_dim())
    state_dict = model.state_dict()
    raw = np.load(weights_path)
    params = [raw[key] for key in sorted(raw.files, key=lambda item: int(item.split("_")[1]))]
    if len(params) != len(state_dict):
        raise ValueError(f"Parameter count mismatch: expected {len(state_dict)}, got {len(params)}")
    new_state = OrderedDict()
    for (name, tensor), value in zip(state_dict.items(), params):
        new_state[name] = torch.tensor(value, dtype=tensor.dtype)
    model.load_state_dict(new_state, strict=True)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Task #2 predictions from saved FL global weights.")
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR, help="Directory with state_to_outcome_test.csv")
    parser.add_argument("--weights-path", default=DEFAULT_WEIGHTS_PATH, help="Path to saved global weights .npz")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory to write predictions CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    split_dir = resolve_path(args.split_dir, script_dir)
    weights_path = resolve_path(args.weights_path, script_dir)
    out_dir = resolve_path(args.out_dir, script_dir)

    test_path = os.path.join(split_dir, "state_to_outcome_test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test split: {test_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing weights file: {weights_path}")

    test_df = sanitize_rows(pd.read_csv(test_path))
    x_test = row_to_input_vector(test_df).astype(np.float32)

    model = load_model(weights_path)
    with torch.no_grad():
        logits = model(torch.tensor(x_test, dtype=torch.float32)).cpu().numpy()

    pred_df = test_df[["admission_id", "patient_id", "room_id", "event_time", "target_time"]].copy()
    for idx, target in enumerate(REGRESSION_TARGET_COLUMNS):
        pred_df[f"{target}_true"] = test_df[target].values
        pred_df[f"{target}_pred"] = logits[:, idx]
    airflow_prob = 1.0 / (1.0 + np.exp(-logits[:, 4]))
    pred_df["y_target_airflow_true"] = test_df["y_target_airflow"].round().astype(int).values
    pred_df["y_target_airflow_pred"] = (airflow_prob >= 0.5).astype(int)
    pred_df["y_target_airflow_score"] = airflow_prob

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "state_to_outcome_predictions.csv")
    pred_df.to_csv(out_path, index=False)

    print("predict_from_weights.py complete")
    print(f"weights_path={weights_path}")
    print(f"test_rows={len(test_df)}")
    print(f"predictions_path={out_path}")


if __name__ == "__main__":
    main()
