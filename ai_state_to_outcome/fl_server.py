import argparse
import os
from typing import Any

import flwr as fl
import numpy as np
import pandas as pd

from fl_client import get_input_dim, get_params, make_model


CSV_WRITE_BATCH_SIZE = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower server for Task #2 state-to-outcome training.")
    parser.add_argument("--server-address", default="127.0.0.1:8080", help="Server bind address")
    parser.add_argument("--rounds", type=int, default=3, help="Federated rounds")
    parser.add_argument("--min-fit-clients", type=int, default=2, help="Minimum clients for fit")
    parser.add_argument("--min-evaluate-clients", type=int, default=2, help="Minimum clients for evaluate")
    parser.add_argument("--min-available-clients", type=int, default=2, help="Minimum connected clients")
    parser.add_argument("--fraction-fit", type=float, default=1.0, help="Fraction of clients sampled for fit")
    parser.add_argument("--fraction-evaluate", type=float, default=1.0, help="Fraction of clients sampled for evaluate")
    parser.add_argument("--aggregation-method", choices=["fedavg", "fedprox"], default="fedavg", help="Server aggregation strategy")
    parser.add_argument("--proximal-mu", type=float, default=0.0, help="FedProx proximal coefficient")
    parser.add_argument("--weights-out-dir", default="ai_state_to_outcome/fl_weights", help="Directory to write global weights and metrics")
    parser.add_argument("--hidden-layers", default="128,64,32", help="Comma-separated hidden layer sizes")
    parser.add_argument("--activation", choices=["relu", "tanh", "logistic"], default="relu", help="Activation function for hidden layers")
    return parser.parse_args()


def make_initial_parameters(hidden_layers: str = "128,64,32", activation: str = "relu") -> fl.common.Parameters:
    model = make_model(get_input_dim(), hidden_layers=hidden_layers, activation=activation)
    return fl.common.ndarrays_to_parameters(get_params(model))


def save_parameters(params: fl.common.Parameters, out_path: str) -> None:
    ndarrays = fl.common.parameters_to_ndarrays(params)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **{f"param_{idx}": array for idx, array in enumerate(ndarrays)})


def _write_dataframe_in_batches(df: pd.DataFrame, out_path: str, *, mode: str, header: bool) -> None:
    for start_idx in range(0, len(df), CSV_WRITE_BATCH_SIZE):
        batch = df.iloc[start_idx:start_idx + CSV_WRITE_BATCH_SIZE]
        batch.to_csv(out_path, mode=mode, header=header, index=False)
        mode = "a"
        header = False


def upsert_rows(out_path: str, rows: list[dict[str, Any]], key_cols: list[str]) -> None:
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
        all_cols = list(existing.columns)
        for col in new_df.columns:
            if col not in all_cols:
                all_cols.append(col)
        for col in all_cols:
            if col not in existing.columns:
                existing[col] = np.nan
            if col not in new_df.columns:
                new_df[col] = np.nan
        merged = pd.concat([existing[all_cols], new_df[all_cols]], ignore_index=True)
        merged = merged.drop_duplicates(subset=key_cols, keep="last")
        _write_dataframe_in_batches(merged, out_path, mode="w", header=True)
    else:
        _write_dataframe_in_batches(new_df, out_path, mode="w", header=True)


def _summary_template() -> dict[str, float]:
    return {
        "evaluated_examples": 0.0,
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
        "mae_sum_y_target_temp_main": 0.0,
        "mae_sum_y_target_temp_toilet": 0.0,
        "mae_sum_y_target_light": 0.0,
        "mae_sum_y_target_sound": 0.0,
        "mse_sum_y_target_temp_main": 0.0,
        "mse_sum_y_target_temp_toilet": 0.0,
        "mse_sum_y_target_light": 0.0,
        "mse_sum_y_target_sound": 0.0,
        "threshold_correct_y_target_temp_main": 0.0,
        "threshold_wrong_y_target_temp_main": 0.0,
        "threshold_correct_y_target_temp_toilet": 0.0,
        "threshold_wrong_y_target_temp_toilet": 0.0,
        "threshold_correct_y_target_light": 0.0,
        "threshold_wrong_y_target_light": 0.0,
        "threshold_correct_y_target_sound": 0.0,
        "threshold_wrong_y_target_sound": 0.0,
    }


class TrackingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, weights_out_dir: str, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.weights_out_dir = weights_out_dir
        self.latest_parameters: fl.common.Parameters | None = None
        self.latest_eval_summary: dict[str, float] | None = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            self.latest_parameters = aggregated_parameters
            save_parameters(aggregated_parameters, os.path.join(self.weights_out_dir, f"round_{server_round}_global_weights.npz"))
            save_parameters(aggregated_parameters, os.path.join(self.weights_out_dir, "latest_global_weights.npz"))
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        summary = _summary_template()
        room_rows: list[dict[str, Any]] = []
        for _, eval_res in results:
            metrics = eval_res.metrics or {}
            count = float(eval_res.num_examples)
            summary["evaluated_examples"] += count
            for key in summary:
                if key == "evaluated_examples":
                    continue
                summary[key] += float(metrics.get(key, 0.0))
            room_rows.append(
                {
                    "round": int(server_round),
                    "room_id": str(metrics.get("room_id", "")),
                    "num_examples": int(eval_res.num_examples),
                    "local_loss": float(eval_res.loss),
                    "mae_y_target_temp_main": float(metrics.get("mae_sum_y_target_temp_main", 0.0)) / max(count, 1.0),
                    "mae_y_target_temp_toilet": float(metrics.get("mae_sum_y_target_temp_toilet", 0.0)) / max(count, 1.0),
                    "mae_y_target_light": float(metrics.get("mae_sum_y_target_light", 0.0)) / max(count, 1.0),
                    "mae_y_target_sound": float(metrics.get("mae_sum_y_target_sound", 0.0)) / max(count, 1.0),
                    "rmse_y_target_temp_main": (float(metrics.get("mse_sum_y_target_temp_main", 0.0)) / max(count, 1.0)) ** 0.5,
                    "rmse_y_target_temp_toilet": (float(metrics.get("mse_sum_y_target_temp_toilet", 0.0)) / max(count, 1.0)) ** 0.5,
                    "rmse_y_target_light": (float(metrics.get("mse_sum_y_target_light", 0.0)) / max(count, 1.0)) ** 0.5,
                    "rmse_y_target_sound": (float(metrics.get("mse_sum_y_target_sound", 0.0)) / max(count, 1.0)) ** 0.5,
                    "threshold_accuracy_y_temp_main": float(metrics.get("threshold_correct_y_target_temp_main", 0.0)) / max(float(metrics.get("threshold_correct_y_target_temp_main", 0.0)) + float(metrics.get("threshold_wrong_y_target_temp_main", 0.0)), 1.0),
                    "threshold_accuracy_y_temp_toilet": float(metrics.get("threshold_correct_y_target_temp_toilet", 0.0)) / max(float(metrics.get("threshold_correct_y_target_temp_toilet", 0.0)) + float(metrics.get("threshold_wrong_y_target_temp_toilet", 0.0)), 1.0),
                    "threshold_accuracy_y_light": float(metrics.get("threshold_correct_y_target_light", 0.0)) / max(float(metrics.get("threshold_correct_y_target_light", 0.0)) + float(metrics.get("threshold_wrong_y_target_light", 0.0)), 1.0),
                    "threshold_accuracy_y_sound": float(metrics.get("threshold_correct_y_target_sound", 0.0)) / max(float(metrics.get("threshold_correct_y_target_sound", 0.0)) + float(metrics.get("threshold_wrong_y_target_sound", 0.0)), 1.0),
                    "airflow_accuracy": float(metrics.get("airflow_accuracy_sum", 0.0)) / max(count, 1.0),
                    "airflow_precision": float(metrics.get("airflow_precision_sum", 0.0)) / max(count, 1.0),
                    "airflow_recall": float(metrics.get("airflow_recall_sum", 0.0)) / max(count, 1.0),
                    "airflow_f1": float(metrics.get("airflow_f1_sum", 0.0)) / max(count, 1.0),
                    "airflow_tp": int(metrics.get("airflow_tp", 0)),
                    "airflow_tn": int(metrics.get("airflow_tn", 0)),
                    "airflow_fp": int(metrics.get("airflow_fp", 0)),
                    "airflow_fn": int(metrics.get("airflow_fn", 0)),
                }
            )

        count = max(summary["evaluated_examples"], 1.0)
        self.latest_eval_summary = {
            "round": int(server_round),
            "global_loss": float(aggregated_loss) if aggregated_loss is not None else 0.0,
            "mae_y_target_temp_main": summary["mae_sum_y_target_temp_main"] / count,
            "mae_y_target_temp_toilet": summary["mae_sum_y_target_temp_toilet"] / count,
            "mae_y_target_light": summary["mae_sum_y_target_light"] / count,
            "mae_y_target_sound": summary["mae_sum_y_target_sound"] / count,
            "rmse_y_target_temp_main": (summary["mse_sum_y_target_temp_main"] / count) ** 0.5,
            "rmse_y_target_temp_toilet": (summary["mse_sum_y_target_temp_toilet"] / count) ** 0.5,
            "rmse_y_target_light": (summary["mse_sum_y_target_light"] / count) ** 0.5,
            "rmse_y_target_sound": (summary["mse_sum_y_target_sound"] / count) ** 0.5,
            "threshold_correct_y_temp_main": int(summary["threshold_correct_y_target_temp_main"]),
            "threshold_wrong_y_temp_main": int(summary["threshold_wrong_y_target_temp_main"]),
            "threshold_correct_y_temp_toilet": int(summary["threshold_correct_y_target_temp_toilet"]),
            "threshold_wrong_y_temp_toilet": int(summary["threshold_wrong_y_target_temp_toilet"]),
            "threshold_correct_y_light": int(summary["threshold_correct_y_target_light"]),
            "threshold_wrong_y_light": int(summary["threshold_wrong_y_target_light"]),
            "threshold_correct_y_sound": int(summary["threshold_correct_y_target_sound"]),
            "threshold_wrong_y_sound": int(summary["threshold_wrong_y_target_sound"]),
            "threshold_accuracy_y_temp_main": summary["threshold_correct_y_target_temp_main"] / max(summary["threshold_correct_y_target_temp_main"] + summary["threshold_wrong_y_target_temp_main"], 1.0),
            "threshold_accuracy_y_temp_toilet": summary["threshold_correct_y_target_temp_toilet"] / max(summary["threshold_correct_y_target_temp_toilet"] + summary["threshold_wrong_y_target_temp_toilet"], 1.0),
            "threshold_accuracy_y_light": summary["threshold_correct_y_target_light"] / max(summary["threshold_correct_y_target_light"] + summary["threshold_wrong_y_target_light"], 1.0),
            "threshold_accuracy_y_sound": summary["threshold_correct_y_target_sound"] / max(summary["threshold_correct_y_target_sound"] + summary["threshold_wrong_y_target_sound"], 1.0),
            "airflow_accuracy": summary["airflow_accuracy_sum"] / count,
            "airflow_precision": summary["airflow_precision_sum"] / count,
            "airflow_recall": summary["airflow_recall_sum"] / count,
            "airflow_f1": summary["airflow_f1_sum"] / count,
            "airflow_correct": int(summary["airflow_correct"]),
            "airflow_incorrect": int(summary["airflow_incorrect"]),
            "airflow_tp": int(summary["airflow_tp"]),
            "airflow_tn": int(summary["airflow_tn"]),
            "airflow_fp": int(summary["airflow_fp"]),
            "airflow_fn": int(summary["airflow_fn"]),
            "evaluated_examples": int(summary["evaluated_examples"]),
        }
        total_path = os.path.join(self.weights_out_dir, f"round_{server_round}_total_metrics.csv")
        room_path = os.path.join(self.weights_out_dir, f"round_{server_round}_room_metrics.csv")
        os.makedirs(self.weights_out_dir, exist_ok=True)
        _write_dataframe_in_batches(pd.DataFrame([self.latest_eval_summary]), total_path, mode="w", header=True)
        _write_dataframe_in_batches(pd.DataFrame(room_rows), room_path, mode="w", header=True)
        upsert_rows(os.path.join(self.weights_out_dir, "room_metrics.csv"), room_rows, ["round", "room_id"])
        return aggregated_loss, aggregated_metrics


class TrackingFedProx(fl.server.strategy.FedProx):
    def __init__(self, weights_out_dir: str, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.weights_out_dir = weights_out_dir
        self.latest_parameters: fl.common.Parameters | None = None
        self.latest_eval_summary: dict[str, float] | None = None

    aggregate_fit = TrackingFedAvg.aggregate_fit
    aggregate_evaluate = TrackingFedAvg.aggregate_evaluate


def make_strategy(aggregation_method: str, weights_out_dir: str, proximal_mu: float, **kwargs: Any):
    if aggregation_method == "fedprox":
        return TrackingFedProx(weights_out_dir=weights_out_dir, proximal_mu=proximal_mu, **kwargs)
    return TrackingFedAvg(weights_out_dir=weights_out_dir, **kwargs)


def main() -> None:
    args = parse_args()
    strategy = make_strategy(
        aggregation_method=args.aggregation_method,
        weights_out_dir=os.path.abspath(args.weights_out_dir),
        proximal_mu=args.proximal_mu,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=args.min_fit_clients,
        min_evaluate_clients=args.min_evaluate_clients,
        min_available_clients=args.min_available_clients,
        initial_parameters=make_initial_parameters(hidden_layers=args.hidden_layers, activation=args.activation),
    )
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
