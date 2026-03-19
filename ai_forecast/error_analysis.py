import argparse
import os

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score

REGRESSION_TARGETS = ["y_temp_main", "y_temp_toilet", "y_light", "y_sound"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ai_forecast error analysis on a predictions CSV.")
    parser.add_argument("--predictions-path", default="ai_forecast/fl_predictions/next_hour_predictions.csv", help="Predictions CSV produced by baseline or FL inference")
    parser.add_argument("--rows-path", default="ai_forecast/splits_next_hour/next_hour_test.csv", help="Original labeled test rows")
    parser.add_argument("--out-dir", default="ai_forecast/error_analysis", help="Directory for analysis outputs")
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


def _binary_summary(df: pd.DataFrame, true_col: str, pred_col: str, target: str) -> dict[str, float | int | str]:
    truth = df[true_col].astype(int)
    pred = df[pred_col].astype(int)
    return {
        "target": target,
        "accuracy": float(accuracy_score(truth, pred)),
        "precision": float(precision_score(truth, pred, zero_division=0)),
        "recall": float(recall_score(truth, pred, zero_division=0)),
        "f1": float(f1_score(truth, pred, zero_division=0)),
        "tp": int(((truth == 1) & (pred == 1)).sum()),
        "tn": int(((truth == 0) & (pred == 0)).sum()),
        "fp": int(((truth == 0) & (pred == 1)).sum()),
        "fn": int(((truth == 1) & (pred == 0)).sum()),
    }


def _slice_metrics(df: pd.DataFrame, slice_col: str) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for value, group in df.groupby(slice_col):
        entry: dict[str, float | str | int] = {slice_col: value, "rows": int(len(group))}
        for target in REGRESSION_TARGETS:
            entry[f"mae_{target}"] = float(mean_absolute_error(group[f"{target}_true"], group[f"{target}_pred"]))
        airflow = _binary_summary(group, "y_airflow_true", "y_airflow_pred_binary", "y_airflow")
        change = _binary_summary(group, "y_any_change_true", "y_any_change_pred", "y_any_change")
        entry["airflow_f1"] = float(airflow["f1"])
        entry["change_f1"] = float(change["f1"])
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
    merged = pred_df.merge(rows_df[["client_id", "t"]], on=["client_id", "t"], how="left")

    os.makedirs(out_dir, exist_ok=True)
    _regression_summary(merged).to_csv(os.path.join(out_dir, "regression_summary.csv"), index=False)
    pd.DataFrame(
        [
            _binary_summary(merged, "y_airflow_true", "y_airflow_pred_binary", "y_airflow"),
            _binary_summary(merged, "y_any_change_true", "y_any_change_pred", "y_any_change"),
        ]
    ).to_csv(os.path.join(out_dir, "classification_summary.csv"), index=False)
    _slice_metrics(merged, "client_id").to_csv(os.path.join(out_dir, "slice_by_room.csv"), index=False)

    residual_rows: list[pd.DataFrame] = []
    for target in REGRESSION_TARGETS:
        residual_rows.append(
            pd.DataFrame(
                {
                    "target": target,
                    "client_id": merged["client_id"],
                    "t": merged["t"],
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
