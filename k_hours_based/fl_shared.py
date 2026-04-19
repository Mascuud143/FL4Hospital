import os

import flwr as fl
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score

from next_hour_schema import AIRFLOW_INDEX, TARGET_COLUMNS
from per_room_data import load_room_df as load_room_df_from_split
from per_room_data import list_room_ids
from runtime_defaults import DEFAULT_AIRFLOW_THRESHOLD, DEFAULT_OUTPUT_THRESHOLDS


METRICS_CSV_WRITE_BATCH_SIZE = 1000


def room_sort_key(value: str) -> int | str:
    return int(value) if str(value).isdigit() else str(value)


def room_ids_from_stats(stats_path: str) -> list[str]:
    if not os.path.exists(stats_path):
        return []
    df = pd.read_csv(stats_path, usecols=["client_id"])
    ids = df["client_id"].astype(str).dropna().unique().tolist()
    return sorted(ids, key=room_sort_key)


def load_split_room_df(
    split_dir: str,
    subset: str,
    room_id: str,
    keep_cols: list[str],
) -> pd.DataFrame:
    df = load_room_df_from_split(split_dir, subset, room_id, usecols=keep_cols)
    return df.loc[:, ~df.columns.duplicated()]


def available_client_workers(client_cpu: float) -> int:
    cpu_count = os.cpu_count() or 1
    client_cpu = max(float(client_cpu), 0.001)
    return max(1, int(cpu_count // client_cpu))


def count_split_rooms(split_dir: str) -> int:
    return len(list_room_ids(split_dir, "train"))


def save_parameters_npz(params: fl.common.Parameters, out_path: str) -> None:
    ndarrays = fl.common.parameters_to_ndarrays(params)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **{f"param_{idx}": array for idx, array in enumerate(ndarrays)})


def write_dataframe_in_batches(df: pd.DataFrame, out_path: str, *, mode: str, header: bool) -> None:
    for start_idx in range(0, len(df), METRICS_CSV_WRITE_BATCH_SIZE):
        batch = df.iloc[start_idx:start_idx + METRICS_CSV_WRITE_BATCH_SIZE]
        batch.to_csv(out_path, mode=mode, header=header, index=False)
        mode = "a"
        header = False


def append_metric_rows(out_path: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.DataFrame(rows)
    if os.path.exists(out_path):
        write_dataframe_in_batches(df, out_path, mode="a", header=False)
    else:
        write_dataframe_in_batches(df, out_path, mode="w", header=True)


def upsert_metric_rows(out_path: str, rows: list[dict[str, object]], key_cols: list[str]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    new_df = pd.DataFrame(rows)
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        for key in key_cols:
            if key in existing.columns and key in new_df.columns:
                existing[key] = existing[key].astype(str)
                new_df[key] = new_df[key].astype(str)
        existing_keys = set(existing.columns)
        all_cols = list(existing.columns)
        for col in new_df.columns:
            if col not in existing_keys:
                all_cols.append(col)
        for col in all_cols:
            if col not in existing.columns:
                existing[col] = np.nan
            if col not in new_df.columns:
                new_df[col] = np.nan
        merged = pd.concat([existing[all_cols], new_df[all_cols]], ignore_index=True)
        merged = merged.drop_duplicates(subset=key_cols, keep="last")
        write_dataframe_in_batches(merged, out_path, mode="w", header=True)
    else:
        write_dataframe_in_batches(new_df, out_path, mode="w", header=True)


def metric_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def empty_eval_metric_totals(*, include_temperature: bool = False) -> dict[str, float]:
    summary = {
        "evaluated_examples": 0.0,
        "regression_correct": 0.0,
        "regression_wrong": 0.0,
        "airflow_correct": 0.0,
        "airflow_incorrect": 0.0,
        "airflow_tp": 0.0,
        "airflow_tn": 0.0,
        "airflow_fp": 0.0,
        "airflow_fn": 0.0,
        "airflow_accuracy_sum": 0.0,
        "airflow_precision_sum": 0.0,
        "airflow_recall_sum": 0.0,
        "airflow_f1_sum": 0.0,
        "change_accuracy_sum": 0.0,
        "change_precision_sum": 0.0,
        "change_recall_sum": 0.0,
        "change_f1_sum": 0.0,
        "change_correct": 0.0,
        "change_incorrect": 0.0,
        "change_tp": 0.0,
        "change_tn": 0.0,
        "change_fp": 0.0,
        "change_fn": 0.0,
        "mae_sum_y_temp_main": 0.0,
        "mae_sum_y_temp_toilet": 0.0,
        "mae_sum_y_light": 0.0,
        "mae_sum_y_sound": 0.0,
        "mse_sum_y_temp_main": 0.0,
        "mse_sum_y_temp_toilet": 0.0,
        "mse_sum_y_light": 0.0,
        "mse_sum_y_sound": 0.0,
        "threshold_correct_y_temp_main": 0.0,
        "threshold_wrong_y_temp_main": 0.0,
        "threshold_correct_y_temp_toilet": 0.0,
        "threshold_wrong_y_temp_toilet": 0.0,
        "threshold_correct_y_light": 0.0,
        "threshold_wrong_y_light": 0.0,
        "threshold_correct_y_sound": 0.0,
        "threshold_wrong_y_sound": 0.0,
    }
    if include_temperature:
        summary["temperature_correct"] = 0.0
        summary["temperature_wrong"] = 0.0
    return summary


def prefix_eval_metrics(prefix: str, loss: float, num_examples: int, metrics: dict[str, float | int | str]) -> dict[str, float | int | str]:
    prefixed: dict[str, float | int | str] = {
        f"{prefix}__global_loss": float(loss),
        f"{prefix}__evaluated_examples": int(num_examples),
    }
    for key, value in metrics.items():
        prefixed[f"{prefix}__{key}"] = value
    return prefixed


def extract_prefixed_eval_metrics(
    prefix: str,
    metrics: dict[str, float | int | str],
) -> tuple[float, int, dict[str, float | int | str]]:
    marker = f"{prefix}__"
    extracted: dict[str, float | int | str] = {}
    for key, value in metrics.items():
        if key.startswith(marker):
            extracted[key[len(marker):]] = value
    loss = float(extracted.pop("global_loss", 0.0) or 0.0)
    num_examples = int(extracted.pop("evaluated_examples", 0) or 0)
    return loss, num_examples, extracted


def target_correct_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int]:
    thresholds = np.array(
        [
            DEFAULT_OUTPUT_THRESHOLDS["y_temp_main"],
            DEFAULT_OUTPUT_THRESHOLDS["y_temp_toilet"],
            DEFAULT_OUTPUT_THRESHOLDS["y_light"],
            DEFAULT_OUTPUT_THRESHOLDS["y_sound"],
        ],
        dtype=np.float64,
    )
    regression_true = y_true[:, :AIRFLOW_INDEX]
    regression_pred = y_pred[:, :AIRFLOW_INDEX]
    correct = int(np.sum(np.all(np.abs(regression_true - regression_pred) <= thresholds, axis=1)))
    return correct, int(y_true.shape[0] - correct)


def per_target_threshold_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, tuple[int, int]]:
    counts: dict[str, tuple[int, int]] = {}
    for idx, target in enumerate(TARGET_COLUMNS[:AIRFLOW_INDEX]):
        threshold = DEFAULT_OUTPUT_THRESHOLDS[target]
        within_threshold = np.abs(y_true[:, idx] - y_pred[:, idx]) <= threshold
        correct = int(np.sum(within_threshold))
        counts[target] = (correct, int(y_true.shape[0] - correct))
    return counts


def temperature_correct_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int]:
    threshold = np.array([DEFAULT_OUTPUT_THRESHOLDS["y_temp_main"]], dtype=np.float64)
    temp_true = y_true[:, :2]
    temp_pred = y_pred[:, :2]
    correct = int(np.sum(np.all(np.abs(temp_true - temp_pred) <= threshold, axis=1)))
    return correct, int(y_true.shape[0] - correct)


def empty_next_hour_eval_metrics(
    room_id: str,
    *,
    include_temperature: bool = False,
    extra_metrics: dict[str, float | int | str] | None = None,
) -> dict[str, float | int | str]:
    metrics: dict[str, float | int | str] = {
        "mae_sum_y_temp_main": 0.0,
        "mae_sum_y_temp_toilet": 0.0,
        "mae_sum_y_light": 0.0,
        "mae_sum_y_sound": 0.0,
        "mse_sum_y_temp_main": 0.0,
        "mse_sum_y_temp_toilet": 0.0,
        "mse_sum_y_light": 0.0,
        "mse_sum_y_sound": 0.0,
        "regression_correct": 0,
        "regression_wrong": 0,
        "threshold_correct_y_temp_main": 0,
        "threshold_wrong_y_temp_main": 0,
        "threshold_correct_y_temp_toilet": 0,
        "threshold_wrong_y_temp_toilet": 0,
        "threshold_correct_y_light": 0,
        "threshold_wrong_y_light": 0,
        "threshold_correct_y_sound": 0,
        "threshold_wrong_y_sound": 0,
        "airflow_accuracy_sum": 0.0,
        "airflow_precision_sum": 0.0,
        "airflow_recall_sum": 0.0,
        "airflow_f1_sum": 0.0,
        "airflow_correct": 0,
        "airflow_incorrect": 0,
        "airflow_tp": 0,
        "airflow_tn": 0,
        "airflow_fp": 0,
        "airflow_fn": 0,
        "change_accuracy_sum": 0.0,
        "change_precision_sum": 0.0,
        "change_recall_sum": 0.0,
        "change_f1_sum": 0.0,
        "change_correct": 0,
        "change_incorrect": 0,
        "change_tp": 0,
        "change_tn": 0,
        "change_fp": 0,
        "change_fn": 0,
        "room_id": room_id,
    }
    if include_temperature:
        metrics["temperature_correct"] = 0
        metrics["temperature_wrong"] = 0
    if extra_metrics:
        metrics.update(extra_metrics)
    return metrics


def summarize_next_hour_predictions(
    room_id: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    change_true: np.ndarray,
    change_pred: np.ndarray,
    *,
    airflow_pred: np.ndarray | None = None,
    include_temperature: bool = False,
    extra_metrics: dict[str, float | int | str] | None = None,
) -> tuple[float, int, dict[str, float | int | str]]:
    if y_true.size == 0:
        return 0.0, 0, empty_next_hour_eval_metrics(
            room_id,
            include_temperature=include_temperature,
            extra_metrics=extra_metrics,
        )

    reg_true = y_true[:, :AIRFLOW_INDEX]
    reg_pred = y_pred[:, :AIRFLOW_INDEX]
    airflow_true = y_true[:, AIRFLOW_INDEX].round().astype(int)
    airflow_pred_array = (
        (np.clip(y_pred[:, AIRFLOW_INDEX], 0.0, 1.0) >= DEFAULT_AIRFLOW_THRESHOLD).astype(int)
        if airflow_pred is None
        else np.asarray(airflow_pred, dtype=np.int64)
    )
    change_true_array = np.asarray(change_true, dtype=np.int64)
    change_pred_array = np.asarray(change_pred, dtype=np.int64)

    regression_mae = {
        target: float(mean_absolute_error(reg_true[:, idx], reg_pred[:, idx]))
        for idx, target in enumerate(TARGET_COLUMNS[:AIRFLOW_INDEX])
    }
    regression_mse = {
        target: float(mean_squared_error(reg_true[:, idx], reg_pred[:, idx]))
        for idx, target in enumerate(TARGET_COLUMNS[:AIRFLOW_INDEX])
    }
    regression_correct, regression_wrong = target_correct_counts(y_true, y_pred)
    threshold_counts = per_target_threshold_counts(reg_true, reg_pred)
    temperature_correct = 0
    temperature_wrong = 0
    if include_temperature:
        temperature_correct, temperature_wrong = temperature_correct_counts(y_true, y_pred)

    airflow_tp = int(np.sum((airflow_true == 1) & (airflow_pred_array == 1)))
    airflow_tn = int(np.sum((airflow_true == 0) & (airflow_pred_array == 0)))
    airflow_fp = int(np.sum((airflow_true == 0) & (airflow_pred_array == 1)))
    airflow_fn = int(np.sum((airflow_true == 1) & (airflow_pred_array == 0)))
    airflow_accuracy = float(accuracy_score(airflow_true, airflow_pred_array))
    airflow_precision = float(precision_score(airflow_true, airflow_pred_array, zero_division=0))
    airflow_recall = float(recall_score(airflow_true, airflow_pred_array, zero_division=0))
    airflow_f1 = float(f1_score(airflow_true, airflow_pred_array, zero_division=0))

    change_tp = int(np.sum((change_true_array == 1) & (change_pred_array == 1)))
    change_tn = int(np.sum((change_true_array == 0) & (change_pred_array == 0)))
    change_fp = int(np.sum((change_true_array == 0) & (change_pred_array == 1)))
    change_fn = int(np.sum((change_true_array == 1) & (change_pred_array == 0)))
    change_accuracy = float(accuracy_score(change_true_array, change_pred_array))
    change_precision = float(precision_score(change_true_array, change_pred_array, zero_division=0))
    change_recall = float(recall_score(change_true_array, change_pred_array, zero_division=0))
    change_f1 = float(f1_score(change_true_array, change_pred_array, zero_division=0))

    count = int(y_true.shape[0])
    metrics = empty_next_hour_eval_metrics(
        room_id,
        include_temperature=include_temperature,
        extra_metrics=extra_metrics,
    )
    metrics.update(
        {
            "mae_sum_y_temp_main": regression_mae["y_temp_main"] * count,
            "mae_sum_y_temp_toilet": regression_mae["y_temp_toilet"] * count,
            "mae_sum_y_light": regression_mae["y_light"] * count,
            "mae_sum_y_sound": regression_mae["y_sound"] * count,
            "mse_sum_y_temp_main": regression_mse["y_temp_main"] * count,
            "mse_sum_y_temp_toilet": regression_mse["y_temp_toilet"] * count,
            "mse_sum_y_light": regression_mse["y_light"] * count,
            "mse_sum_y_sound": regression_mse["y_sound"] * count,
            "regression_correct": regression_correct,
            "regression_wrong": regression_wrong,
            "threshold_correct_y_temp_main": threshold_counts["y_temp_main"][0],
            "threshold_wrong_y_temp_main": threshold_counts["y_temp_main"][1],
            "threshold_correct_y_temp_toilet": threshold_counts["y_temp_toilet"][0],
            "threshold_wrong_y_temp_toilet": threshold_counts["y_temp_toilet"][1],
            "threshold_correct_y_light": threshold_counts["y_light"][0],
            "threshold_wrong_y_light": threshold_counts["y_light"][1],
            "threshold_correct_y_sound": threshold_counts["y_sound"][0],
            "threshold_wrong_y_sound": threshold_counts["y_sound"][1],
            "airflow_accuracy_sum": airflow_accuracy * count,
            "airflow_precision_sum": airflow_precision * count,
            "airflow_recall_sum": airflow_recall * count,
            "airflow_f1_sum": airflow_f1 * count,
            "airflow_correct": int(np.sum(airflow_true == airflow_pred_array)),
            "airflow_incorrect": int(count - np.sum(airflow_true == airflow_pred_array)),
            "airflow_tp": airflow_tp,
            "airflow_tn": airflow_tn,
            "airflow_fp": airflow_fp,
            "airflow_fn": airflow_fn,
            "change_accuracy_sum": change_accuracy * count,
            "change_precision_sum": change_precision * count,
            "change_recall_sum": change_recall * count,
            "change_f1_sum": change_f1 * count,
            "change_correct": int(np.sum(change_true_array == change_pred_array)),
            "change_incorrect": int(count - np.sum(change_true_array == change_pred_array)),
            "change_tp": change_tp,
            "change_tn": change_tn,
            "change_fp": change_fp,
            "change_fn": change_fn,
        }
    )
    if include_temperature:
        metrics["temperature_correct"] = temperature_correct
        metrics["temperature_wrong"] = temperature_wrong
    overall_loss = float(np.mean(list(regression_mae.values())))
    return overall_loss, count, metrics


def write_next_hour_prediction_csv(
    predictions_out_dir: str,
    room_id: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    change_true: np.ndarray,
    change_pred: np.ndarray,
    *,
    airflow_pred: np.ndarray | None = None,
) -> str:
    os.makedirs(predictions_out_dir, exist_ok=True)
    reg_true = y_true[:, :AIRFLOW_INDEX]
    reg_pred = y_pred[:, :AIRFLOW_INDEX]
    airflow_true = y_true[:, AIRFLOW_INDEX].round().astype(int)
    airflow_pred_array = (
        (np.clip(y_pred[:, AIRFLOW_INDEX], 0.0, 1.0) >= DEFAULT_AIRFLOW_THRESHOLD).astype(int)
        if airflow_pred is None
        else np.asarray(airflow_pred, dtype=np.int64)
    )
    prediction_frame = pd.DataFrame(
        {
            "room_id": room_id,
            "y_temp_main_true": reg_true[:, 0],
            "y_temp_main_pred": reg_pred[:, 0],
            "y_temp_toilet_true": reg_true[:, 1],
            "y_temp_toilet_pred": reg_pred[:, 1],
            "y_light_true": reg_true[:, 2],
            "y_light_pred": reg_pred[:, 2],
            "y_sound_true": reg_true[:, 3],
            "y_sound_pred": reg_pred[:, 3],
            "y_airflow_true": airflow_true,
            "y_airflow_pred": airflow_pred_array,
            "y_any_change_true": np.asarray(change_true, dtype=np.int64),
            "y_any_change_pred": np.asarray(change_pred, dtype=np.int64),
        }
    )
    prediction_path = os.path.join(predictions_out_dir, f"room_{room_id}_predictions.csv")
    prediction_frame.to_csv(prediction_path, index=False)
    return prediction_path


def accumulate_eval_metrics(summary: dict[str, float], metrics: dict[str, float | int | str], count: float) -> None:
    summary["evaluated_examples"] += float(count)
    for key in summary:
        if key == "evaluated_examples":
            continue
        summary[key] += float(metrics.get(key, 0.0) or 0.0)


def build_latest_fit_summary(server_round: int, fit_examples: float, train_loss_sum: float) -> dict[str, float | int]:
    return {
        "round": int(server_round),
        "trained_examples": int(fit_examples),
        "train_loss": float(train_loss_sum) / max(float(fit_examples), 1.0),
    }


def build_room_metric_row(
    server_round: int,
    num_examples: int,
    local_loss: float,
    metrics: dict[str, float | int | str],
) -> dict[str, float | int | str]:
    count = float(num_examples)
    return {
        "round": int(server_round),
        "room_id": str(metrics.get("room_id", "")),
        "num_examples": int(num_examples),
        "local_loss": float(local_loss),
        "mae_y_temp_main": float(metrics.get("mae_sum_y_temp_main", 0.0)) / max(count, 1.0),
        "mse_y_temp_main": float(metrics.get("mse_sum_y_temp_main", 0.0)) / max(count, 1.0),
        "rmse_y_temp_main": (float(metrics.get("mse_sum_y_temp_main", 0.0)) / max(count, 1.0)) ** 0.5,
        "threshold_accuracy_y_temp_main": metric_ratio(
            metrics.get("threshold_correct_y_temp_main", 0.0),
            float(metrics.get("threshold_correct_y_temp_main", 0.0)) + float(metrics.get("threshold_wrong_y_temp_main", 0.0)),
        ),
        "mae_y_temp_toilet": float(metrics.get("mae_sum_y_temp_toilet", 0.0)) / max(count, 1.0),
        "mse_y_temp_toilet": float(metrics.get("mse_sum_y_temp_toilet", 0.0)) / max(count, 1.0),
        "rmse_y_temp_toilet": (float(metrics.get("mse_sum_y_temp_toilet", 0.0)) / max(count, 1.0)) ** 0.5,
        "threshold_accuracy_y_temp_toilet": metric_ratio(
            metrics.get("threshold_correct_y_temp_toilet", 0.0),
            float(metrics.get("threshold_correct_y_temp_toilet", 0.0)) + float(metrics.get("threshold_wrong_y_temp_toilet", 0.0)),
        ),
        "mae_y_light": float(metrics.get("mae_sum_y_light", 0.0)) / max(count, 1.0),
        "mse_y_light": float(metrics.get("mse_sum_y_light", 0.0)) / max(count, 1.0),
        "rmse_y_light": (float(metrics.get("mse_sum_y_light", 0.0)) / max(count, 1.0)) ** 0.5,
        "threshold_accuracy_y_light": metric_ratio(
            metrics.get("threshold_correct_y_light", 0.0),
            float(metrics.get("threshold_correct_y_light", 0.0)) + float(metrics.get("threshold_wrong_y_light", 0.0)),
        ),
        "mae_y_sound": float(metrics.get("mae_sum_y_sound", 0.0)) / max(count, 1.0),
        "mse_y_sound": float(metrics.get("mse_sum_y_sound", 0.0)) / max(count, 1.0),
        "rmse_y_sound": (float(metrics.get("mse_sum_y_sound", 0.0)) / max(count, 1.0)) ** 0.5,
        "threshold_accuracy_y_sound": metric_ratio(
            metrics.get("threshold_correct_y_sound", 0.0),
            float(metrics.get("threshold_correct_y_sound", 0.0)) + float(metrics.get("threshold_wrong_y_sound", 0.0)),
        ),
        "airflow_accuracy": float(metrics.get("airflow_accuracy_sum", 0.0)) / max(count, 1.0),
        "airflow_precision": float(metrics.get("airflow_precision_sum", 0.0)) / max(count, 1.0),
        "airflow_recall": float(metrics.get("airflow_recall_sum", 0.0)) / max(count, 1.0),
        "airflow_f1": float(metrics.get("airflow_f1_sum", 0.0)) / max(count, 1.0),
        "airflow_tp": int(metrics.get("airflow_tp", 0)),
        "airflow_fp": int(metrics.get("airflow_fp", 0)),
        "airflow_tn": int(metrics.get("airflow_tn", 0)),
        "airflow_fn": int(metrics.get("airflow_fn", 0)),
        "change_accuracy": float(metrics.get("change_accuracy_sum", 0.0)) / max(count, 1.0),
        "change_precision": float(metrics.get("change_precision_sum", 0.0)) / max(count, 1.0),
        "change_recall": float(metrics.get("change_recall_sum", 0.0)) / max(count, 1.0),
        "change_f1": float(metrics.get("change_f1_sum", 0.0)) / max(count, 1.0),
        "change_tp": int(metrics.get("change_tp", 0)),
        "change_fp": int(metrics.get("change_fp", 0)),
        "change_tn": int(metrics.get("change_tn", 0)),
        "change_fn": int(metrics.get("change_fn", 0)),
    }


def build_latest_eval_summary(
    server_round: int,
    global_loss: float | None,
    summary: dict[str, float],
    *,
    include_temperature: bool = False,
    include_threshold_counts: bool = False,
    recompute_binary_metrics: bool = False,
) -> dict[str, float | int]:
    count = max(float(summary["evaluated_examples"]), 1.0)
    latest_eval_summary: dict[str, float | int] = {
        "round": int(server_round),
        "global_loss": float(global_loss) if global_loss is not None else 0.0,
        "mae_y_temp_main": float(summary["mae_sum_y_temp_main"]) / count,
        "mse_y_temp_main": float(summary["mse_sum_y_temp_main"]) / count,
        "mae_y_temp_toilet": float(summary["mae_sum_y_temp_toilet"]) / count,
        "mse_y_temp_toilet": float(summary["mse_sum_y_temp_toilet"]) / count,
        "mae_y_light": float(summary["mae_sum_y_light"]) / count,
        "mse_y_light": float(summary["mse_sum_y_light"]) / count,
        "mae_y_sound": float(summary["mae_sum_y_sound"]) / count,
        "mse_y_sound": float(summary["mse_sum_y_sound"]) / count,
        "rmse_y_temp_main": (float(summary["mse_sum_y_temp_main"]) / count) ** 0.5,
        "rmse_y_temp_toilet": (float(summary["mse_sum_y_temp_toilet"]) / count) ** 0.5,
        "rmse_y_light": (float(summary["mse_sum_y_light"]) / count) ** 0.5,
        "rmse_y_sound": (float(summary["mse_sum_y_sound"]) / count) ** 0.5,
        "threshold_accuracy_y_temp_main": metric_ratio(
            summary["threshold_correct_y_temp_main"],
            summary["threshold_correct_y_temp_main"] + summary["threshold_wrong_y_temp_main"],
        ),
        "threshold_accuracy_y_temp_toilet": metric_ratio(
            summary["threshold_correct_y_temp_toilet"],
            summary["threshold_correct_y_temp_toilet"] + summary["threshold_wrong_y_temp_toilet"],
        ),
        "threshold_accuracy_y_light": metric_ratio(
            summary["threshold_correct_y_light"],
            summary["threshold_correct_y_light"] + summary["threshold_wrong_y_light"],
        ),
        "threshold_accuracy_y_sound": metric_ratio(
            summary["threshold_correct_y_sound"],
            summary["threshold_correct_y_sound"] + summary["threshold_wrong_y_sound"],
        ),
        "regression_correct": int(summary["regression_correct"]),
        "regression_wrong": int(summary["regression_wrong"]),
        "regression_correct_rate": metric_ratio(
            summary["regression_correct"],
            summary["regression_correct"] + summary["regression_wrong"],
        ),
        "airflow_correct": int(summary["airflow_correct"]),
        "airflow_incorrect": int(summary["airflow_incorrect"]),
        "airflow_tp": int(summary["airflow_tp"]),
        "airflow_tn": int(summary["airflow_tn"]),
        "airflow_fp": int(summary["airflow_fp"]),
        "airflow_fn": int(summary["airflow_fn"]),
        "change_correct": int(summary["change_correct"]),
        "change_incorrect": int(summary["change_incorrect"]),
        "change_tp": int(summary["change_tp"]),
        "change_tn": int(summary["change_tn"]),
        "change_fp": int(summary["change_fp"]),
        "change_fn": int(summary["change_fn"]),
        "evaluated_examples": int(summary["evaluated_examples"]),
    }
    if include_temperature:
        latest_eval_summary["temperature_correct"] = int(summary["temperature_correct"])
        latest_eval_summary["temperature_wrong"] = int(summary["temperature_wrong"])
        latest_eval_summary["temperature_correct_rate"] = metric_ratio(
            summary["temperature_correct"],
            summary["temperature_correct"] + summary["temperature_wrong"],
        )
    if include_threshold_counts:
        latest_eval_summary["threshold_correct_y_temp_main"] = int(summary["threshold_correct_y_temp_main"])
        latest_eval_summary["threshold_wrong_y_temp_main"] = int(summary["threshold_wrong_y_temp_main"])
        latest_eval_summary["threshold_correct_y_temp_toilet"] = int(summary["threshold_correct_y_temp_toilet"])
        latest_eval_summary["threshold_wrong_y_temp_toilet"] = int(summary["threshold_wrong_y_temp_toilet"])
        latest_eval_summary["threshold_correct_y_light"] = int(summary["threshold_correct_y_light"])
        latest_eval_summary["threshold_wrong_y_light"] = int(summary["threshold_wrong_y_light"])
        latest_eval_summary["threshold_correct_y_sound"] = int(summary["threshold_correct_y_sound"])
        latest_eval_summary["threshold_wrong_y_sound"] = int(summary["threshold_wrong_y_sound"])

    if recompute_binary_metrics:
        airflow_tp = float(summary["airflow_tp"])
        airflow_tn = float(summary["airflow_tn"])
        airflow_fp = float(summary["airflow_fp"])
        airflow_fn = float(summary["airflow_fn"])
        latest_eval_summary["airflow_accuracy"] = metric_ratio(airflow_tp + airflow_tn, airflow_tp + airflow_tn + airflow_fp + airflow_fn)
        latest_eval_summary["airflow_precision"] = metric_ratio(airflow_tp, airflow_tp + airflow_fp)
        latest_eval_summary["airflow_recall"] = metric_ratio(airflow_tp, airflow_tp + airflow_fn)
        airflow_precision = float(latest_eval_summary["airflow_precision"])
        airflow_recall = float(latest_eval_summary["airflow_recall"])
        latest_eval_summary["airflow_f1"] = (
            0.0
            if airflow_precision + airflow_recall <= 0.0
            else 2.0 * airflow_precision * airflow_recall / (airflow_precision + airflow_recall)
        )

        change_tp = float(summary["change_tp"])
        change_tn = float(summary["change_tn"])
        change_fp = float(summary["change_fp"])
        change_fn = float(summary["change_fn"])
        latest_eval_summary["change_accuracy"] = metric_ratio(change_tp + change_tn, change_tp + change_tn + change_fp + change_fn)
        latest_eval_summary["change_precision"] = metric_ratio(change_tp, change_tp + change_fp)
        latest_eval_summary["change_recall"] = metric_ratio(change_tp, change_tp + change_fn)
        change_precision = float(latest_eval_summary["change_precision"])
        change_recall = float(latest_eval_summary["change_recall"])
        latest_eval_summary["change_f1"] = (
            0.0
            if change_precision + change_recall <= 0.0
            else 2.0 * change_precision * change_recall / (change_precision + change_recall)
        )
    else:
        latest_eval_summary["airflow_accuracy"] = float(summary["airflow_accuracy_sum"]) / count
        latest_eval_summary["airflow_precision"] = float(summary["airflow_precision_sum"]) / count
        latest_eval_summary["airflow_recall"] = float(summary["airflow_recall_sum"]) / count
        latest_eval_summary["airflow_f1"] = float(summary["airflow_f1_sum"]) / count
        latest_eval_summary["change_accuracy"] = float(summary["change_accuracy_sum"]) / count
        latest_eval_summary["change_precision"] = float(summary["change_precision_sum"]) / count
        latest_eval_summary["change_recall"] = float(summary["change_recall_sum"]) / count
        latest_eval_summary["change_f1"] = float(summary["change_f1_sum"]) / count
    return latest_eval_summary
