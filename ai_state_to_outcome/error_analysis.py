import argparse
import os

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score

REGRESSION_TARGETS = [
    "y_target_temp_main",
    "y_target_temp_toilet",
    "y_target_light",
    "y_target_sound",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task #2 error analysis on a predictions CSV.")
    parser.add_argument("--predictions-path", default="ai_state_to_outcome/fl_predictions/state_to_outcome_predictions.csv", help="Predictions CSV produced by baseline or FL inference")
    parser.add_argument("--rows-path", default="ai_state_to_outcome/splits/state_to_outcome_test.csv", help="Original labeled rows used for test evaluation")
    parser.add_argument("--out-dir", default="ai_state_to_outcome/error_analysis", help="Directory for analysis outputs")
    return parser.parse_args()


def _regression_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for target in REGRESSION_TARGETS:
        true_col = f"{target}_true"
        pred_col = f"{target}_pred"
        residual = df[true_col] - df[pred_col]
        rows.append(
            {
                "target": target,
                "mae": float(mean_absolute_error(df[true_col], df[pred_col])),
                "rmse": float(mean_squared_error(df[true_col], df[pred_col]) ** 0.5),
                "residual_mean": float(residual.mean()),
                "residual_std": float(residual.std(ddof=0)),
                "residual_median": float(residual.median()),
                "residual_p10": float(residual.quantile(0.10)),
                "residual_p90": float(residual.quantile(0.90)),
            }
        )
    return pd.DataFrame(rows)


def _classification_summary(df: pd.DataFrame) -> pd.DataFrame:
    truth = df["y_target_airflow_true"].astype(int)
    pred = df["y_target_airflow_pred"].astype(int)
    tp = int(((truth == 1) & (pred == 1)).sum())
    tn = int(((truth == 0) & (pred == 0)).sum())
    fp = int(((truth == 0) & (pred == 1)).sum())
    fn = int(((truth == 1) & (pred == 0)).sum())
    return pd.DataFrame(
        [
            {
                "target": "y_target_airflow",
                "accuracy": float(accuracy_score(truth, pred)),
                "precision": float(precision_score(truth, pred, zero_division=0)),
                "recall": float(recall_score(truth, pred, zero_division=0)),
                "f1": float(f1_score(truth, pred, zero_division=0)),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }
        ]
    )


def _slice_metrics(df: pd.DataFrame, slice_col: str) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for value, group in df.groupby(slice_col):
        entry: dict[str, float | str | int] = {slice_col: value, "rows": int(len(group))}
        for target in REGRESSION_TARGETS:
            true_col = f"{target}_true"
            pred_col = f"{target}_pred"
            entry[f"mae_{target}"] = float(mean_absolute_error(group[true_col], group[pred_col]))
        truth = group["y_target_airflow_true"].astype(int)
        pred = group["y_target_airflow_pred"].astype(int)
        entry["airflow_accuracy"] = float(accuracy_score(truth, pred))
        entry["airflow_f1"] = float(f1_score(truth, pred, zero_division=0))
        rows.append(entry)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    predictions_path = os.path.abspath(args.predictions_path)
    rows_path = os.path.abspath(args.rows_path)
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Missing predictions file: {predictions_path}")
    if not os.path.exists(rows_path):
        raise FileNotFoundError(f"Missing rows file: {rows_path}")

    pred_df = pd.read_csv(predictions_path)
    rows_df = pd.read_csv(rows_path)
    merged = pred_df.merge(
        rows_df[["admission_id", "patient_id", "room_id", "event_time", "event_type", "event_detail", "target_time"]],
        on=["admission_id", "patient_id", "room_id", "event_time", "target_time"],
        how="left",
    )
    merged["target_delay_hours"] = (
        pd.to_datetime(merged["target_time"], utc=True) - pd.to_datetime(merged["event_time"], utc=True)
    ).dt.total_seconds() / 3600.0

    os.makedirs(out_dir, exist_ok=True)
    _regression_summary(merged).to_csv(os.path.join(out_dir, "regression_summary.csv"), index=False)
    _classification_summary(merged).to_csv(os.path.join(out_dir, "classification_summary.csv"), index=False)
    _slice_metrics(merged, "room_id").to_csv(os.path.join(out_dir, "slice_by_room.csv"), index=False)
    if "event_type" in merged.columns:
        _slice_metrics(merged, "event_type").to_csv(os.path.join(out_dir, "slice_by_event_type.csv"), index=False)

    residual_rows: list[pd.DataFrame] = []
    for target in REGRESSION_TARGETS:
        residual_rows.append(
            pd.DataFrame(
                {
                    "target": target,
                    "admission_id": merged["admission_id"],
                    "patient_id": merged["patient_id"],
                    "room_id": merged["room_id"],
                    "event_time": merged["event_time"],
                    "target_time": merged["target_time"],
                    "target_delay_hours": merged["target_delay_hours"],
                    "residual": merged[f"{target}_true"] - merged[f"{target}_pred"],
                }
            )
        )
    pd.concat(residual_rows, ignore_index=True).to_csv(os.path.join(out_dir, "residuals.csv"), index=False)
    merged.to_csv(os.path.join(out_dir, "merged_predictions.csv"), index=False)

    print("error_analysis.py complete")
    print(f"predictions_path={predictions_path}")
    print(f"rows_path={rows_path}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
