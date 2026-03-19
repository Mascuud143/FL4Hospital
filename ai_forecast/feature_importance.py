import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor

from fl_client import MODEL_HIDDEN_LAYER_SIZES, make_model, sanitize_targets, set_params
from next_hour_schema import CHANGE_BASELINE_COLUMNS, INPUT_COLUMNS, TARGET_COLUMNS, next_hour_change_flags, row_to_input_vector

REGRESSION_TARGETS = ["y_temp_main", "y_temp_toilet", "y_light", "y_sound"]
FEATURE_GROUPS = {
    "time_features": [col for col in INPUT_COLUMNS if col.startswith("time_slot_")],
    "demographics": [col for col in INPUT_COLUMNS if col in {"age", "height", "weight", "gender_binary"}],
    "diagnosis": [col for col in INPUT_COLUMNS if col.startswith("diagnosis_")],
    "symptoms": [col for col in INPUT_COLUMNS if col.startswith("symptom_")],
    "medication_type": [col for col in INPUT_COLUMNS if "_type_" in col],
    "medication_schedule": [col for col in INPUT_COLUMNS if "_sched_" in col],
}
DEFAULT_SPLIT_DIR = "splits_next_hour"
DEFAULT_OUT_DIR = "feature_importance"
DEFAULT_WEIGHTS_PATH = "fl_weights_sim/latest_global_weights.npz"


def resolve_path(value: str, script_dir: str) -> str:
    return value if os.path.isabs(value) else os.path.join(script_dir, value)


def load_rows(path: str) -> pd.DataFrame:
    return sanitize_targets(pd.read_csv(path))


def make_baseline_model(input_dim: int) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=MODEL_HIDDEN_LAYER_SIZES,
        activation="relu",
        random_state=42,
        early_stopping=True,
        max_iter=300,
    )


def load_fl_model(weights_path: str, input_dim: int) -> MLPRegressor:
    model = make_model(input_dim)
    raw = np.load(weights_path)
    params = [raw[key] for key in sorted(raw.files, key=lambda item: int(item.split("_")[1]))]
    set_params(model, params)
    return model


def forecast_metrics(df: pd.DataFrame, pred_df: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {}
    mae_values: list[float] = []
    for target in REGRESSION_TARGETS:
        mae = float(mean_absolute_error(df[target], pred_df[target]))
        metrics[f"mae_{target}"] = mae
        mae_values.append(mae)
    airflow_true = df["y_airflow"].round().astype(int)
    airflow_pred = (pred_df["y_airflow"] >= 0.5).astype(int)
    metrics["airflow_f1"] = float(f1_score(airflow_true, airflow_pred, zero_division=0))
    current_values = df[CHANGE_BASELINE_COLUMNS].astype(float).to_numpy()
    predicted_values = pred_df[TARGET_COLUMNS].astype(float).to_numpy()
    change_true = df["y_any_change"].round().astype(int)
    change_pred = next_hour_change_flags(current_values, predicted_values)
    metrics["change_f1"] = float(f1_score(change_true, change_pred, zero_division=0))
    metrics["composite_loss"] = float(np.mean(mae_values) + (1.0 - metrics["airflow_f1"]) + (1.0 - metrics["change_f1"]))
    return metrics


def permutation_importance(df: pd.DataFrame, predict_fn, features: list[str], random_state: int) -> tuple[pd.DataFrame, dict[str, float]]:
    baseline_pred = predict_fn(df)
    baseline_metrics = forecast_metrics(df, baseline_pred)
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | str]] = []
    for feature in features:
        shuffled = df.copy()
        shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
        metrics = forecast_metrics(shuffled, predict_fn(shuffled))
        rows.append(
            {
                "feature": feature,
                "delta_composite_loss": metrics["composite_loss"] - baseline_metrics["composite_loss"],
                "delta_airflow_f1": baseline_metrics["airflow_f1"] - metrics["airflow_f1"],
                "delta_change_f1": baseline_metrics["change_f1"] - metrics["change_f1"],
                "delta_mae_y_temp_main": metrics["mae_y_temp_main"] - baseline_metrics["mae_y_temp_main"],
                "delta_mae_y_temp_toilet": metrics["mae_y_temp_toilet"] - baseline_metrics["mae_y_temp_toilet"],
                "delta_mae_y_light": metrics["mae_y_light"] - baseline_metrics["mae_y_light"],
                "delta_mae_y_sound": metrics["mae_y_sound"] - baseline_metrics["mae_y_sound"],
            }
        )
    return pd.DataFrame(rows).sort_values("delta_composite_loss", ascending=False, kind="stable"), baseline_metrics


def run_ablation_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    x_train = row_to_input_vector(train_df)
    x_test = row_to_input_vector(test_df)
    full_model = make_baseline_model(x_train.shape[1])
    full_model.fit(x_train, train_df[TARGET_COLUMNS].to_numpy(dtype=float))
    baseline_pred = pd.DataFrame(full_model.predict(x_test), columns=TARGET_COLUMNS, index=test_df.index)
    baseline_metrics = forecast_metrics(test_df, baseline_pred)

    rows: list[dict[str, float | str]] = []
    for group_name, columns in FEATURE_GROUPS.items():
        remaining = [col for col in INPUT_COLUMNS if col not in columns]
        reduced_train = train_df[remaining]
        reduced_test = test_df[remaining]
        model = make_baseline_model(reduced_train.shape[1])
        model.fit(reduced_train.to_numpy(dtype=float), train_df[TARGET_COLUMNS].to_numpy(dtype=float))
        pred_df = pd.DataFrame(model.predict(reduced_test.to_numpy(dtype=float)), columns=TARGET_COLUMNS, index=test_df.index)
        metrics = forecast_metrics(test_df, pred_df)
        rows.append(
            {
                "group": group_name,
                "delta_composite_loss": metrics["composite_loss"] - baseline_metrics["composite_loss"],
                "delta_airflow_f1": baseline_metrics["airflow_f1"] - metrics["airflow_f1"],
                "delta_change_f1": baseline_metrics["change_f1"] - metrics["change_f1"],
            }
        )
    return pd.DataFrame(rows).sort_values("delta_composite_loss", ascending=False, kind="stable"), baseline_metrics


def ablation_fl(model: MLPRegressor, test_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    x_test = row_to_input_vector(test_df)
    baseline_pred = pd.DataFrame(model.predict(x_test), columns=TARGET_COLUMNS, index=test_df.index)
    baseline_metrics = forecast_metrics(test_df, baseline_pred)
    rows: list[dict[str, float | str]] = []
    for group_name, columns in FEATURE_GROUPS.items():
        ablated = test_df.copy()
        for column in columns:
            if column in ablated.columns:
                ablated[column] = 0.0
        pred_df = pd.DataFrame(model.predict(row_to_input_vector(ablated)), columns=TARGET_COLUMNS, index=ablated.index)
        metrics = forecast_metrics(ablated, pred_df)
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
    parser = argparse.ArgumentParser(description="ai_forecast permutation importance and ablation analysis.")
    parser.add_argument("--model-type", choices=["baseline", "fl"], default="baseline", help="Which model family to analyze")
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR, help="Directory with next_hour train/test splits")
    parser.add_argument("--weights-path", default=DEFAULT_WEIGHTS_PATH, help="Required for model-type=fl")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for feature importance outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for permutation shuffling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    split_dir = resolve_path(args.split_dir, script_dir)
    out_dir = resolve_path(args.out_dir, script_dir)
    train_path = os.path.join(split_dir, "next_hour_train.csv")
    test_path = os.path.join(split_dir, "next_hour_test.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing train split: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test split: {test_path}")

    train_df = load_rows(train_path)
    test_df = load_rows(test_path)
    os.makedirs(out_dir, exist_ok=True)

    if args.model_type == "baseline":
        model = make_baseline_model(len(INPUT_COLUMNS))
        model.fit(row_to_input_vector(train_df), train_df[TARGET_COLUMNS].to_numpy(dtype=float))
        importance_df, baseline_metrics = permutation_importance(
            test_df,
            predict_fn=lambda df: pd.DataFrame(model.predict(row_to_input_vector(df)), columns=TARGET_COLUMNS, index=df.index),
            features=INPUT_COLUMNS,
            random_state=args.seed,
        )
        ablation_df, ablation_baseline_metrics = run_ablation_baseline(train_df, test_df)
    else:
        weights_path = resolve_path(args.weights_path, script_dir)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Missing weights file: {weights_path}")
        model = load_fl_model(weights_path, len(INPUT_COLUMNS))
        importance_df, baseline_metrics = permutation_importance(
            test_df,
            predict_fn=lambda df: pd.DataFrame(model.predict(row_to_input_vector(df)), columns=TARGET_COLUMNS, index=df.index),
            features=INPUT_COLUMNS,
            random_state=args.seed,
        )
        ablation_df, ablation_baseline_metrics = ablation_fl(model, test_df)

    importance_df.to_csv(os.path.join(out_dir, f"permutation_importance_{args.model_type}.csv"), index=False)
    ablation_df.to_csv(os.path.join(out_dir, f"ablation_{args.model_type}.csv"), index=False)
    pd.DataFrame([baseline_metrics]).to_csv(os.path.join(out_dir, f"baseline_metrics_{args.model_type}.csv"), index=False)
    pd.DataFrame([ablation_baseline_metrics]).to_csv(os.path.join(out_dir, f"ablation_baseline_metrics_{args.model_type}.csv"), index=False)

    print("feature_importance.py complete")
    print(f"model_type={args.model_type}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
