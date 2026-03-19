import argparse
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "PyTorch is required for ai_forecast LSTM+MLP feature importance. Install torch before running this mode."
    ) from exc

from fl_client_lstm_mlp import build_hybrid_arrays, make_model, sanitize_targets
from next_hour_schema import CHANGE_BASELINE_COLUMNS, INPUT_COLUMNS, TARGET_COLUMNS, next_hour_change_flags

REGRESSION_TARGETS = ["y_temp_main", "y_temp_toilet", "y_light", "y_sound"]
FEATURE_GROUPS = {
    "time_features": [col for col in INPUT_COLUMNS if col.startswith("time_slot_")],
    "demographics": [col for col in INPUT_COLUMNS if col in {"age", "height", "weight", "gender_binary"}],
    "diagnosis": [col for col in INPUT_COLUMNS if col.startswith("diagnosis_")],
    "symptoms": [col for col in INPUT_COLUMNS if col.startswith("symptom_")],
    "medication_type": [col for col in INPUT_COLUMNS if "_type_" in col],
    "medication_schedule": [col for col in INPUT_COLUMNS if "_sched_" in col],
}
DEFAULT_SPLIT_DIR = "..\\ai\\splits_next_hour"
DEFAULT_WEIGHTS_PATH = "fl_weights_sim_lstm_mlp/latest_global_weights.npz"
DEFAULT_OUT_DIR = "feature_importance_lstm_mlp"


def resolve_path(value: str, script_dir: str) -> str:
    return value if os.path.isabs(value) else os.path.join(script_dir, value)


def load_rows(path: str) -> pd.DataFrame:
    return sanitize_targets(pd.read_csv(path))


def load_model(weights_path: str, input_dim: int):
    model = make_model(input_dim)
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


def hybrid_predict(model, df: pd.DataFrame, sequence_length: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    x_seq, x_flat, y_true, current_rows, change_true = build_hybrid_arrays(df, sequence_length)
    with torch.no_grad():
        change_logits = model.change_mlp(torch.tensor(x_flat, dtype=torch.float32)).cpu().numpy()
        target_logits = model.target_lstm(torch.tensor(x_seq, dtype=torch.float32)).cpu().numpy()
    pred_df = pd.DataFrame(target_logits, columns=TARGET_COLUMNS)
    pred_df["y_airflow"] = (1.0 / (1.0 + np.exp(-pred_df["y_airflow"])) >= 0.5).astype(int)
    change_prob = 1.0 / (1.0 + np.exp(-change_logits))
    change_pred = (change_prob >= 0.5).astype(int)
    return pred_df, current_rows, change_pred


def compute_metrics(df: pd.DataFrame, pred_df: pd.DataFrame, current_rows: np.ndarray, change_true: np.ndarray) -> dict[str, float]:
    metrics: dict[str, float] = {}
    mae_values: list[float] = []
    for target in REGRESSION_TARGETS:
        mae = float(mean_absolute_error(df[target].iloc[-len(pred_df):], pred_df[target]))
        metrics[f"mae_{target}"] = mae
        mae_values.append(mae)
    airflow_true = df["y_airflow"].iloc[-len(pred_df):].round().astype(int)
    airflow_pred = pred_df["y_airflow"].round().astype(int)
    metrics["airflow_f1"] = float(f1_score(airflow_true, airflow_pred, zero_division=0))
    predicted_values = pred_df[TARGET_COLUMNS].astype(float).to_numpy()
    change_pred = next_hour_change_flags(current_rows, predicted_values)
    metrics["change_f1"] = float(f1_score(change_true.astype(int), change_pred, zero_division=0))
    metrics["composite_loss"] = float(np.mean(mae_values) + (1.0 - metrics["airflow_f1"]) + (1.0 - metrics["change_f1"]))
    return metrics


def permutation_importance(df: pd.DataFrame, model, sequence_length: int, random_state: int) -> tuple[pd.DataFrame, dict[str, float]]:
    baseline_pred, baseline_current, baseline_change = hybrid_predict(model, df, sequence_length)
    baseline_metrics = compute_metrics(df, baseline_pred, baseline_current, baseline_change)
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | str]] = []
    for feature in INPUT_COLUMNS:
        shuffled = df.copy()
        shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
        pred_df, current_rows, change_pred = hybrid_predict(model, shuffled, sequence_length)
        metrics = compute_metrics(shuffled, pred_df, current_rows, change_pred)
        rows.append(
            {
                "feature": feature,
                "delta_composite_loss": metrics["composite_loss"] - baseline_metrics["composite_loss"],
                "delta_airflow_f1": baseline_metrics["airflow_f1"] - metrics["airflow_f1"],
                "delta_change_f1": baseline_metrics["change_f1"] - metrics["change_f1"],
            }
        )
    return pd.DataFrame(rows).sort_values("delta_composite_loss", ascending=False, kind="stable"), baseline_metrics


def ablation_importance(df: pd.DataFrame, model, sequence_length: int) -> tuple[pd.DataFrame, dict[str, float]]:
    baseline_pred, baseline_current, baseline_change = hybrid_predict(model, df, sequence_length)
    baseline_metrics = compute_metrics(df, baseline_pred, baseline_current, baseline_change)
    rows: list[dict[str, float | str]] = []
    for group_name, columns in FEATURE_GROUPS.items():
        ablated = df.copy()
        for column in columns:
            if column in ablated.columns:
                ablated[column] = 0.0
        pred_df, current_rows, change_pred = hybrid_predict(model, ablated, sequence_length)
        metrics = compute_metrics(ablated, pred_df, current_rows, change_pred)
        rows.append(
            {
                "group": group_name,
                "delta_composite_loss": metrics["composite_loss"] - baseline_metrics["composite_loss"],
                "delta_airflow_f1": baseline_metrics["airflow_f1"] - metrics["airflow_f1"],
                "delta_change_f1": baseline_metrics["change_f1"] - metrics["change_f1"],
            }
        )
    return pd.DataFrame(rows).sort_values("delta_composite_loss", ascending=False, kind="stable"), baseline_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ai_forecast LSTM+MLP permutation importance and ablation analysis.")
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR, help="Directory with next_hour train/test splits")
    parser.add_argument("--weights-path", default=DEFAULT_WEIGHTS_PATH, help="Path to LSTM+MLP global weights")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for feature importance outputs")
    parser.add_argument("--sequence-length", type=int, default=4, help="Sequence length used by the hybrid model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for permutation shuffling")
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

    test_df = load_rows(test_path)
    model = load_model(weights_path, len(INPUT_COLUMNS))
    os.makedirs(out_dir, exist_ok=True)

    permutation_df, baseline_metrics = permutation_importance(test_df, model, args.sequence_length, args.seed)
    ablation_df, ablation_baseline_metrics = ablation_importance(test_df, model, args.sequence_length)

    permutation_df.to_csv(os.path.join(out_dir, "permutation_importance_fl_lstm_mlp.csv"), index=False)
    ablation_df.to_csv(os.path.join(out_dir, "ablation_fl_lstm_mlp.csv"), index=False)
    pd.DataFrame([baseline_metrics]).to_csv(os.path.join(out_dir, "baseline_metrics_fl_lstm_mlp.csv"), index=False)
    pd.DataFrame([ablation_baseline_metrics]).to_csv(os.path.join(out_dir, "ablation_baseline_metrics_fl_lstm_mlp.csv"), index=False)

    print("feature_importance_lstm_mlp.py complete")
    print(f"weights_path={weights_path}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
