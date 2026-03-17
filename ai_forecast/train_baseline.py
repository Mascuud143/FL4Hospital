import argparse
import os
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DEFAULT_SPLIT_DIR = "splits"
DEFAULT_OUT_DIR = "baseline_results"
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "hour",
    "weekday",
    "day_of_stay",
    "age",
    "height",
    "minutes_since_last_med",
    "curr_temp_main",
    "curr_temp_toilet",
    "curr_light",
    "curr_sound",
    "curr_airflow",
    "minutes_since_last_change",
    "gender",
    "ethnicity",
    "diagnosis",
    "latest_symptom",
    "last_medication",
    "last_med_status",
]

NUMERIC_FEATURES = [
    "hour",
    "weekday",
    "day_of_stay",
    "age",
    "height",
    "minutes_since_last_med",
    "curr_temp_main",
    "curr_temp_toilet",
    "curr_light",
    "curr_sound",
    "curr_airflow",
    "minutes_since_last_change",
]

CATEGORICAL_FEATURES = [
    "gender",
    "ethnicity",
    "diagnosis",
    "latest_symptom",
    "last_medication",
    "last_med_status",
]

MODEL_B_NUMERIC_TARGETS = [
    "y_temp_main",
    "y_temp_toilet",
    "y_light",
    "y_sound",
    "y_when_minutes",
]
MODEL_B_AIRFLOW_TARGET = "y_airflow"



def resolve_split_dir(split_dir: str, script_dir: str) -> str:
    if os.path.isabs(split_dir):
        return split_dir
    if split_dir == DEFAULT_SPLIT_DIR:
        return os.path.join(script_dir, split_dir)
    return os.path.abspath(split_dir)



def as_float(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")



def as_int(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).round().astype(int)



def load_rows(path: str, max_rows: int | None, for_model_b: bool = False) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing split file: {path}")

    target_cols = ["y_event"] if not for_model_b else MODEL_B_NUMERIC_TARGETS + [MODEL_B_AIRFLOW_TARGET]
    keep_cols = ["client_id", "t"] + FEATURE_COLUMNS + target_cols
    df = pd.read_csv(path, usecols=lambda c: c in keep_cols)

    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=RANDOM_STATE)

    as_float(df, NUMERIC_FEATURES + MODEL_B_NUMERIC_TARGETS)
    as_int(df, ["y_event", MODEL_B_AIRFLOW_TARGET])

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    return df



def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )



def train_and_eval_model_a(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["y_event"].astype(int)
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["y_event"].astype(int)

    model = Pipeline(
        steps=[
            ("prep", make_preprocessor()),
            (
                "clf",
                LogisticRegression(
                    max_iter=600,
                    class_weight="balanced",
                    solver="saga",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
    }

    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
    except Exception:
        metrics["roc_auc"] = float("nan")

    pred_df = test_df[["client_id"]].copy()
    pred_df["y_true"] = y_test.values
    pred_df["y_pred"] = y_pred
    return metrics, pred_df



def per_room_model_a_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for room_id, grp in pred_df.groupby("client_id"):
        yt = grp["y_true"]
        yp = grp["y_pred"]
        rows.append(
            {
                "client_id": room_id,
                "rows": len(grp),
                "precision": precision_score(yt, yp, zero_division=0),
                "recall": recall_score(yt, yp, zero_division=0),
                "f1": f1_score(yt, yp, zero_division=0),
                "accuracy": accuracy_score(yt, yp),
            }
        )
    return pd.DataFrame(rows).sort_values("client_id")



def train_and_eval_model_b(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    X_train = train_df[FEATURE_COLUMNS]
    X_test = test_df[FEATURE_COLUMNS]

    regression_metrics: dict[str, float] = {}
    baseline_values: dict[str, float] = {}

    for target in MODEL_B_NUMERIC_TARGETS:
        train_mask = train_df[target].notna()
        test_mask = test_df[target].notna()
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            regression_metrics[f"mae_{target}"] = float("nan")
            regression_metrics[f"rmse_{target}"] = float("nan")
            baseline_values[f"ridge_intercept_{target}"] = float("nan")
            continue

        X_train_target = X_train.loc[train_mask]
        y_train = train_df.loc[train_mask, target]
        X_test_target = X_test.loc[test_mask]
        y_test = test_df.loc[test_mask, target]

        reg_model = Pipeline(
            steps=[
                ("prep", make_preprocessor()),
                ("reg", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
            ]
        )
        reg_model.fit(X_train_target, y_train)
        y_pred = reg_model.predict(X_test_target)

        regression_metrics[f"mae_{target}"] = mean_absolute_error(y_test, y_pred)
        regression_metrics[f"rmse_{target}"] = mean_squared_error(y_test, y_pred) ** 0.5
        baseline_values[f"ridge_intercept_{target}"] = float(getattr(reg_model.named_steps["reg"], "intercept_", float("nan")))

    train_air_mask = train_df[MODEL_B_AIRFLOW_TARGET].notna()
    test_air_mask = test_df[MODEL_B_AIRFLOW_TARGET].notna()
    if train_air_mask.sum() == 0 or test_air_mask.sum() == 0:
        regression_metrics["airflow_precision"] = float("nan")
        regression_metrics["airflow_recall"] = float("nan")
        regression_metrics["airflow_f1"] = float("nan")
        regression_metrics["airflow_accuracy"] = float("nan")
        return regression_metrics, baseline_values

    X_train_air = X_train.loc[train_air_mask]
    X_test_air = X_test.loc[test_air_mask]
    y_train_air = train_df.loc[train_air_mask, MODEL_B_AIRFLOW_TARGET].astype(int)
    y_test_air = test_df.loc[test_air_mask, MODEL_B_AIRFLOW_TARGET].astype(int)

    air_model = Pipeline(
        steps=[
            ("prep", make_preprocessor()),
            (
                "clf",
                LogisticRegression(
                    max_iter=600,
                    class_weight="balanced",
                    solver="saga",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    air_model.fit(X_train_air, y_train_air)
    y_pred_air = air_model.predict(X_test_air)

    regression_metrics["airflow_precision"] = precision_score(y_test_air, y_pred_air, zero_division=0)
    regression_metrics["airflow_recall"] = recall_score(y_test_air, y_pred_air, zero_division=0)
    regression_metrics["airflow_f1"] = f1_score(y_test_air, y_pred_air, zero_division=0)
    regression_metrics["airflow_accuracy"] = accuracy_score(y_test_air, y_pred_air)
    return regression_metrics, baseline_values



def write_metrics(path: str, metrics: dict[str, float]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())}).to_csv(path, index=False)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate scikit-learn baseline models from split CSV files.")
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR, help="Directory with model_a_train/test.csv and model_b_train/test.csv")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for baseline metrics output")
    parser.add_argument("--max-train-a", type=int, default=250000, help="Optional row cap for Model A train")
    parser.add_argument("--max-test-a", type=int, default=150000, help="Optional row cap for Model A test")
    parser.add_argument("--max-train-b", type=int, default=200000, help="Optional row cap for Model B train")
    parser.add_argument("--max-test-b", type=int, default=120000, help="Optional row cap for Model B test")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    split_dir = resolve_split_dir(args.split_dir, script_dir)
    out_dir = os.path.join(script_dir, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir

    a_train = load_rows(os.path.join(split_dir, "model_a_train.csv"), args.max_train_a, for_model_b=False)
    a_test = load_rows(os.path.join(split_dir, "model_a_test.csv"), args.max_test_a, for_model_b=False)
    b_train = load_rows(os.path.join(split_dir, "model_b_train.csv"), args.max_train_b, for_model_b=True)
    b_test = load_rows(os.path.join(split_dir, "model_b_test.csv"), args.max_test_b, for_model_b=True)

    metrics_a, pred_a = train_and_eval_model_a(a_train, a_test)
    metrics_b, model_b_values = train_and_eval_model_b(b_train, b_test)
    per_room_a_df = per_room_model_a_metrics(pred_a)

    os.makedirs(out_dir, exist_ok=True)
    write_metrics(os.path.join(out_dir, "model_a_metrics.csv"), metrics_a)
    write_metrics(os.path.join(out_dir, "model_b_metrics.csv"), metrics_b)
    write_metrics(os.path.join(out_dir, "model_b_baseline_values.csv"), model_b_values)
    per_room_a_df.to_csv(os.path.join(out_dir, "model_a_per_room_metrics.csv"), index=False)

    print("train_baseline.py complete")
    print(f"model_a_train_rows_used={len(a_train)}")
    print(f"model_a_test_rows_used={len(a_test)}")
    print(f"model_b_train_rows_used={len(b_train)}")
    print(f"model_b_test_rows_used={len(b_test)}")
    print(f"model_a_f1={metrics_a['f1']:.4f}")
    print(f"model_a_roc_auc={metrics_a['roc_auc']:.4f}")
    print(f"model_b_mae_y_when_minutes={metrics_b['mae_y_when_minutes']:.4f}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
