import argparse
import os

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score
from sklearn.neural_network import MLPRegressor

from next_hour_schema import CHANGE_BASELINE_COLUMNS, INPUT_COLUMNS, TARGET_COLUMNS, next_hour_change_flags, row_to_input_vector

DEFAULT_SPLIT_DIR = "splits_next_hour"
DEFAULT_OUT_DIR = "baseline_results_next_hour"
RANDOM_STATE = 42
REGRESSION_TARGETS = ["y_temp_main", "y_temp_toilet", "y_light", "y_sound"]
AIRFLOW_TARGET = "y_airflow"
CHANGE_METADATA_TARGETS = ["curr_temp_main_eval", "curr_temp_toilet_eval", "curr_light_eval", "curr_sound_eval", "curr_airflow_eval"]


def resolve_split_dir(split_dir: str, script_dir: str) -> str:
    if os.path.isabs(split_dir):
        return split_dir
    if split_dir == DEFAULT_SPLIT_DIR:
        return os.path.join(script_dir, split_dir)
    return os.path.abspath(split_dir)


def load_rows(path: str, max_rows: int | None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing split file: {path}")
    keep_cols = ["client_id", "t", *CHANGE_BASELINE_COLUMNS, "y_any_change", *INPUT_COLUMNS, *TARGET_COLUMNS]
    df = pd.read_csv(path, usecols=lambda c: c in keep_cols)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=RANDOM_STATE)
    for col in INPUT_COLUMNS + TARGET_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=CHANGE_METADATA_TARGETS + INPUT_COLUMNS + TARGET_COLUMNS).copy()


def compute_metrics(test_df: pd.DataFrame, y_pred: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    y_test = test_df[TARGET_COLUMNS].astype(float)
    metrics: dict[str, float] = {}
    for target in REGRESSION_TARGETS:
        metrics[f"mae_{target}"] = mean_absolute_error(y_test[target], y_pred[target])
        metrics[f"rmse_{target}"] = mean_squared_error(y_test[target], y_pred[target]) ** 0.5

    airflow_pred = (y_pred[AIRFLOW_TARGET] >= 0.5).astype(int)
    airflow_true = y_test[AIRFLOW_TARGET].round().astype(int)
    metrics["airflow_accuracy"] = accuracy_score(airflow_true, airflow_pred)
    metrics["airflow_precision"] = precision_score(airflow_true, airflow_pred, zero_division=0)
    metrics["airflow_recall"] = recall_score(airflow_true, airflow_pred, zero_division=0)
    metrics["airflow_f1"] = f1_score(airflow_true, airflow_pred, zero_division=0)

    current_values = test_df[CHANGE_METADATA_TARGETS].astype(float).to_numpy()
    predicted_values = y_pred[TARGET_COLUMNS].astype(float).to_numpy()
    change_true = test_df["y_any_change"].round().astype(int)
    change_pred = next_hour_change_flags(current_values, predicted_values)
    metrics["change_accuracy"] = accuracy_score(change_true, change_pred)
    metrics["change_precision"] = precision_score(change_true, change_pred, zero_division=0)
    metrics["change_recall"] = recall_score(change_true, change_pred, zero_division=0)
    metrics["change_f1"] = f1_score(change_true, change_pred, zero_division=0)
    metrics["change_tp"] = int(((change_true == 1) & (change_pred == 1)).sum())
    metrics["change_tn"] = int(((change_true == 0) & (change_pred == 0)).sum())
    metrics["change_fp"] = int(((change_true == 0) & (change_pred == 1)).sum())
    metrics["change_fn"] = int(((change_true == 1) & (change_pred == 0)).sum())

    pred_df = test_df[["client_id", "t"]].copy()
    for target in TARGET_COLUMNS:
        pred_df[f"{target}_true"] = y_test[target].values
        pred_df[f"{target}_pred"] = y_pred[target].values
    pred_df["y_airflow_pred_binary"] = airflow_pred.values
    pred_df["y_any_change_true"] = change_true.values
    pred_df["y_any_change_pred"] = change_pred
    return metrics, pred_df


def train_and_eval_next_hour(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    X_train = row_to_input_vector(train_df)
    X_test = row_to_input_vector(test_df)
    y_train = train_df[TARGET_COLUMNS].astype(float)

    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        random_state=RANDOM_STATE,
        early_stopping=True,
        max_iter=300,
    )
    model.fit(X_train, y_train)
    y_pred = pd.DataFrame(
        model.predict(X_test),
        columns=TARGET_COLUMNS,
        index=test_df.index,
    )
    metrics, pred_df = compute_metrics(test_df, y_pred)

    per_room_rows: list[dict[str, float | str | int]] = []
    for client_id, room_test_df in test_df.groupby("client_id"):
        room_pred = y_pred.loc[room_test_df.index]
        room_metrics, _ = compute_metrics(room_test_df, room_pred)
        per_room_rows.append(
            {
                "client_id": client_id,
                "rows": int(len(room_test_df)),
                **room_metrics,
            }
        )
    per_room_df = pd.DataFrame(per_room_rows)
    return metrics, pred_df, per_room_df


def write_metrics(path: str, metrics: dict[str, float]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())}).to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate a local next-hour MLP model from split CSV files.")
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR, help="Directory with next_hour_train/test.csv")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for output metrics and predictions")
    parser.add_argument("--max-train", type=int, default=250000, help="Optional row cap for training")
    parser.add_argument("--max-test", type=int, default=150000, help="Optional row cap for testing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    split_dir = resolve_split_dir(args.split_dir, script_dir)
    out_dir = os.path.join(script_dir, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    train_df = load_rows(os.path.join(split_dir, "next_hour_train.csv"), args.max_train)
    test_df = load_rows(os.path.join(split_dir, "next_hour_test.csv"), args.max_test)
    metrics, pred_df, per_room_df = train_and_eval_next_hour(train_df, test_df)
    os.makedirs(out_dir, exist_ok=True)
    write_metrics(os.path.join(out_dir, "next_hour_metrics.csv"), metrics)
    pred_df.to_csv(os.path.join(out_dir, "next_hour_predictions.csv"), index=False)
    per_room_df.to_csv(os.path.join(out_dir, "next_hour_per_room_metrics.csv"), index=False)

    print("train_baseline_next_hour.py complete")
    print(f"train_rows_used={len(train_df)}")
    print(f"test_rows_used={len(test_df)}")
    print(f"airflow_f1={metrics['airflow_f1']:.4f}")
    print(f"change_f1={metrics['change_f1']:.4f}")
    print(f"mae_y_temp_main={metrics['mae_y_temp_main']:.4f}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
