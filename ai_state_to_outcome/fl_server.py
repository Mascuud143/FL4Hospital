import argparse
import os
from typing import Any

import flwr as fl
import numpy as np
import pandas as pd

from fl_client import get_input_dim, get_params, make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower server for Task #2 state-to-outcome training.")
    parser.add_argument("--server-address", default="127.0.0.1:8080", help="Server bind address")
    parser.add_argument("--rounds", type=int, default=3, help="Federated rounds")
    parser.add_argument("--min-fit-clients", type=int, default=2, help="Minimum clients for fit")
    parser.add_argument("--min-evaluate-clients", type=int, default=2, help="Minimum clients for evaluate")
    parser.add_argument("--min-available-clients", type=int, default=2, help="Minimum connected clients")
    parser.add_argument("--fraction-fit", type=float, default=1.0, help="Fraction of clients sampled for fit")
    parser.add_argument("--fraction-evaluate", type=float, default=1.0, help="Fraction of clients sampled for evaluate")
    parser.add_argument("--weights-out-dir", default="ai_state_to_outcome/fl_weights", help="Directory to write global weights and metrics")
    return parser.parse_args()


def make_initial_parameters() -> fl.common.Parameters:
    model = make_model(get_input_dim())
    return fl.common.ndarrays_to_parameters(get_params(model))


def save_parameters(params: fl.common.Parameters, out_path: str) -> None:
    ndarrays = fl.common.parameters_to_ndarrays(params)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **{f"param_{idx}": array for idx, array in enumerate(ndarrays)})


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
        for _, eval_res in results:
            metrics = eval_res.metrics or {}
            count = float(eval_res.num_examples)
            summary["evaluated_examples"] += count
            for key in summary:
                if key == "evaluated_examples":
                    continue
                summary[key] += float(metrics.get(key, 0.0))

        count = max(summary["evaluated_examples"], 1.0)
        self.latest_eval_summary = {
            "mae_y_target_temp_main": summary["mae_sum_y_target_temp_main"] / count,
            "mae_y_target_temp_toilet": summary["mae_sum_y_target_temp_toilet"] / count,
            "mae_y_target_light": summary["mae_sum_y_target_light"] / count,
            "mae_y_target_sound": summary["mae_sum_y_target_sound"] / count,
            "rmse_y_target_temp_main": (summary["mse_sum_y_target_temp_main"] / count) ** 0.5,
            "rmse_y_target_temp_toilet": (summary["mse_sum_y_target_temp_toilet"] / count) ** 0.5,
            "rmse_y_target_light": (summary["mse_sum_y_target_light"] / count) ** 0.5,
            "rmse_y_target_sound": (summary["mse_sum_y_target_sound"] / count) ** 0.5,
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
        os.makedirs(self.weights_out_dir, exist_ok=True)
        pd.DataFrame([self.latest_eval_summary]).to_csv(total_path, index=False)
        return aggregated_loss, aggregated_metrics


def main() -> None:
    args = parse_args()
    strategy = TrackingFedAvg(
        weights_out_dir=os.path.abspath(args.weights_out_dir),
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=args.min_fit_clients,
        min_evaluate_clients=args.min_evaluate_clients,
        min_available_clients=args.min_available_clients,
        initial_parameters=make_initial_parameters(),
    )
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
