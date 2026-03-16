import argparse
import os
from typing import Any

import flwr as fl
import numpy as np
import pandas as pd

from fl_client_lstm import TARGET_COLUMNS, get_input_dim, get_params, make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower server for next-hour LSTM training.")
    parser.add_argument("--split-dir", default="ai/splits_next_hour", help="Directory with next_hour train split")
    parser.add_argument("--server-address", default="127.0.0.1:8080", help="Server bind address")
    parser.add_argument("--rounds", type=int, default=3, help="Federated rounds")
    parser.add_argument("--min-fit-clients", type=int, default=2, help="Minimum clients for fit")
    parser.add_argument("--min-evaluate-clients", type=int, default=2, help="Minimum clients for evaluate")
    parser.add_argument("--min-available-clients", type=int, default=2, help="Minimum connected clients")
    parser.add_argument("--fraction-fit", type=float, default=1.0, help="Fraction of clients sampled for fit")
    parser.add_argument("--fraction-evaluate", type=float, default=1.0, help="Fraction of clients sampled for evaluate")
    parser.add_argument("--weights-out-dir", default="ai/fl_weights_next_hour_lstm", help="Directory to write global weights per round")
    return parser.parse_args()


def make_initial_parameters() -> fl.common.Parameters:
    model = make_model(get_input_dim())
    return fl.common.ndarrays_to_parameters(get_params(model))


def count_rooms(split_dir: str) -> int:
    train_path = os.path.join(split_dir, "next_hour_train.csv")
    if not os.path.exists(train_path):
        return 0
    df = pd.read_csv(train_path, usecols=["client_id"])
    return int(df["client_id"].astype(str).nunique())


def save_parameters(params: fl.common.Parameters, out_path: str) -> None:
    ndarrays = fl.common.parameters_to_ndarrays(params)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **{f"param_{idx}": array for idx, array in enumerate(ndarrays)})


def save_parameters_readable(params: fl.common.Parameters, out_prefix: str) -> None:
    ndarrays = fl.common.parameters_to_ndarrays(params)
    summary_txt = f"{out_prefix}_summary.txt"
    os.makedirs(os.path.dirname(summary_txt), exist_ok=True)
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"param_count={len(ndarrays)}\n")
        for idx, array in enumerate(ndarrays):
            flat = array.reshape(-1)
            csv_path = f"{out_prefix}_param_{idx}.csv"
            pd.DataFrame({"value": flat}).to_csv(csv_path, index=False)
            f.write(f"param_{idx}_shape={array.shape}\n")
            f.write(f"param_{idx}_min={float(np.min(flat))}\n")
            f.write(f"param_{idx}_max={float(np.max(flat))}\n")
            f.write(f"param_{idx}_mean={float(np.mean(flat))}\n")
            f.write(f"param_{idx}_std={float(np.std(flat))}\n")


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
            round_weights_path = os.path.join(self.weights_out_dir, f"round_{server_round}_global_weights.npz")
            latest_weights_path = os.path.join(self.weights_out_dir, "latest_global_weights.npz")
            save_parameters(aggregated_parameters, round_weights_path)
            save_parameters(aggregated_parameters, latest_weights_path)
            print(f"saved_global_weights={round_weights_path}")
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
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
        }
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
            "mae_y_temp_main": summary["mae_sum_y_temp_main"] / count,
            "mae_y_temp_toilet": summary["mae_sum_y_temp_toilet"] / count,
            "mae_y_light": summary["mae_sum_y_light"] / count,
            "mae_y_sound": summary["mae_sum_y_sound"] / count,
            "rmse_y_temp_main": (summary["mse_sum_y_temp_main"] / count) ** 0.5,
            "rmse_y_temp_toilet": (summary["mse_sum_y_temp_toilet"] / count) ** 0.5,
            "rmse_y_light": (summary["mse_sum_y_light"] / count) ** 0.5,
            "rmse_y_sound": (summary["mse_sum_y_sound"] / count) ** 0.5,
            "regression_correct": int(summary["regression_correct"]),
            "regression_wrong": int(summary["regression_wrong"]),
            "regression_correct_rate": summary["regression_correct"] / max(summary["regression_correct"] + summary["regression_wrong"], 1.0),
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
            "change_accuracy": summary["change_accuracy_sum"] / count,
            "change_precision": summary["change_precision_sum"] / count,
            "change_recall": summary["change_recall_sum"] / count,
            "change_f1": summary["change_f1_sum"] / count,
            "change_correct": int(summary["change_correct"]),
            "change_incorrect": int(summary["change_incorrect"]),
            "change_tp": int(summary["change_tp"]),
            "change_tn": int(summary["change_tn"]),
            "change_fp": int(summary["change_fp"]),
            "change_fn": int(summary["change_fn"]),
            "evaluated_examples": int(summary["evaluated_examples"]),
        }
        os.makedirs(self.weights_out_dir, exist_ok=True)
        total_path = os.path.join(self.weights_out_dir, f"round_{server_round}_total_metrics.csv")
        pd.DataFrame([self.latest_eval_summary]).to_csv(total_path, index=False)
        print(f"saved_total_metrics={total_path}")
        print(
            f"[round {server_round}] evaluation_summary "
            f"regression_correct_rate={self.latest_eval_summary['regression_correct_rate']:.4f} "
            f"airflow_f1={self.latest_eval_summary['airflow_f1']:.4f} "
            f"change_f1={self.latest_eval_summary['change_f1']:.4f} "
            f"mae_y_temp_main={self.latest_eval_summary['mae_y_temp_main']:.4f} "
            f"evaluated_examples={self.latest_eval_summary['evaluated_examples']}"
        )
        return aggregated_loss, aggregated_metrics


def main() -> None:
    args = parse_args()
    split_dir = os.path.abspath(args.split_dir)
    weights_out_dir = os.path.abspath(args.weights_out_dir)
    rooms = count_rooms(split_dir)

    strategy = TrackingFedAvg(
        weights_out_dir=weights_out_dir,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=args.min_fit_clients,
        min_evaluate_clients=args.min_evaluate_clients,
        min_available_clients=args.min_available_clients,
        initial_parameters=make_initial_parameters(),
    )

    print("fl_server_lstm.py starting")
    print(f"split_dir={split_dir}")
    print(f"targets={','.join(TARGET_COLUMNS)}")
    print(f"rooms_detected={rooms}")
    print(f"server_address={args.server_address}")
    print(f"rounds={args.rounds}")
    print(f"weights_out_dir={weights_out_dir}")
    print("waiting_for_room_clients...")

    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
if __name__ == "__main__":
    main()
