import argparse
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

from schema import FEATURE_COLUMNS, REGRESSION_TARGET_COLUMNS, TARGET_COLUMNS, row_to_input_vector

DEFAULT_SPLIT_DIR = "splits"
DEFAULT_OUT_DIR = "baseline_results"


def resolve_dir(value: str, script_dir: str) -> str:
    return value if os.path.isabs(value) else os.path.join(script_dir, value)


def load_rows(path: str, max_rows: int | None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing split file: {path}")
    keep_cols = [*FEATURE_COLUMNS, *TARGET_COLUMNS, "admission_id", "patient_id", "room_id", "event_time", "target_time"]
    df = pd.read_csv(path, usecols=lambda col: col in keep_cols)
    if max_rows is not None and len(df) > max_rows:
        df = df.iloc[:max_rows].copy()
    numeric_cols = [col for col in FEATURE_COLUMNS + TARGET_COLUMNS if col in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    return df


def make_regression_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))),
        ]
    )


def make_airflow_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ]
    )


def compute_metrics(test_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for target in REGRESSION_TARGET_COLUMNS:
        metrics[f"mae_{target}"] = float(mean_absolute_error(test_df[target], pred_df[target]))
        metrics[f"rmse_{target}"] = float(mean_squared_error(test_df[target], pred_df[target]) ** 0.5)

    airflow_true = test_df["y_target_airflow"].round().astype(int)
    airflow_pred = pred_df["y_target_airflow"].round().astype(int)
    metrics["airflow_accuracy"] = float(accuracy_score(airflow_true, airflow_pred))
    metrics["airflow_precision"] = float(precision_score(airflow_true, airflow_pred, zero_division=0))
    metrics["airflow_recall"] = float(recall_score(airflow_true, airflow_pred, zero_division=0))
    metrics["airflow_f1"] = float(f1_score(airflow_true, airflow_pred, zero_division=0))
    metrics["evaluated_examples"] = int(len(test_df))
    return metrics


def train_and_eval(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    clean_train = train_df.dropna(subset=TARGET_COLUMNS).copy()
    clean_test = test_df.dropna(subset=TARGET_COLUMNS).copy()
    x_train = row_to_input_vector(clean_train)
    x_test = row_to_input_vector(clean_test)

    reg_model = make_regression_model()
    reg_model.fit(x_train, clean_train[REGRESSION_TARGET_COLUMNS].to_numpy(dtype=float))
    reg_pred = pd.DataFrame(reg_model.predict(x_test), columns=REGRESSION_TARGET_COLUMNS, index=clean_test.index)

    airflow_model = make_airflow_model()
    airflow_model.fit(x_train, clean_train["y_target_airflow"].round().astype(int).to_numpy())
    airflow_pred = airflow_model.predict(x_test)

    pred_df = clean_test[["admission_id", "patient_id", "room_id", "event_time", "target_time"]].copy()
    for target in REGRESSION_TARGET_COLUMNS:
        pred_df[f"{target}_true"] = clean_test[target].values
        pred_df[f"{target}_pred"] = reg_pred[target].values
    pred_df["y_target_airflow_true"] = clean_test["y_target_airflow"].round().astype(int).values
    pred_df["y_target_airflow_pred"] = airflow_pred

    merged_pred = pd.DataFrame(index=clean_test.index)
    for target in REGRESSION_TARGET_COLUMNS:
        merged_pred[target] = reg_pred[target]
    merged_pred["y_target_airflow"] = airflow_pred
    metrics = compute_metrics(clean_test, merged_pred)

    per_room_rows: list[dict[str, float | int]] = []
    for room_id, room_group in clean_test.groupby("room_id"):
        room_pred = merged_pred.loc[room_group.index]
        room_metrics = compute_metrics(room_group, room_pred)
        per_room_rows.append({"room_id": int(room_id), "rows": int(len(room_group)), **room_metrics})
    per_room_df = pd.DataFrame(per_room_rows)
    return metrics, pred_df, per_room_df


def write_metrics(path: str, metrics: dict[str, float]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())}).to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a local baseline model for Task #2 state-to-outcome prediction.")
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR, help="Directory containing Task #2 train/test CSV files.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for metrics and predictions.")
    parser.add_argument("--max-train", type=int, default=None, help="Optional cap for train rows.")
    parser.add_argument("--max-test", type=int, default=None, help="Optional cap for test rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    split_dir = resolve_dir(args.split_dir, script_dir)
    out_dir = resolve_dir(args.out_dir, script_dir)
    train_path = os.path.join(split_dir, "state_to_outcome_train.csv")
    test_path = os.path.join(split_dir, "state_to_outcome_test.csv")

    train_df = load_rows(train_path, args.max_train)
    test_df = load_rows(test_path, args.max_test)
    metrics, pred_df, per_room_df = train_and_eval(train_df, test_df)

    os.makedirs(out_dir, exist_ok=True)
    write_metrics(os.path.join(out_dir, "state_to_outcome_metrics.csv"), metrics)
    pred_df.to_csv(os.path.join(out_dir, "state_to_outcome_predictions.csv"), index=False)
    per_room_df.to_csv(os.path.join(out_dir, "state_to_outcome_per_room_metrics.csv"), index=False)

    print("train_baseline.py complete")
    print(f"train_rows_used={len(train_df)}")
    print(f"test_rows_used={len(test_df)}")
    print(f"airflow_f1={metrics['airflow_f1']:.4f}")
    print(f"mae_y_target_temp_main={metrics['mae_y_target_temp_main']:.4f}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
