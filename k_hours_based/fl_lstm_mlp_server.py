import argparse
import os
from typing import Any

import flwr as fl
import numpy as np
import pandas as pd

try:
    from k_hours_based.fl_lstm_mlp_client import TARGET_COLUMNS, get_input_dim, get_params, make_model
except ModuleNotFoundError:
    from fl_lstm_mlp_client import TARGET_COLUMNS, get_input_dim, get_params, make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower server for next-hour hybrid MLP+LSTM training.")
    parser.add_argument("--split-dir", default="ai/splits_next_hour", help="Directory with next_hour train split")
    parser.add_argument("--server-address", default="127.0.0.1:8080", help="Server bind address")
    parser.add_argument("--rounds", type=int, default=3, help="Federated rounds")
    parser.add_argument("--min-fit-clients", type=int, default=2, help="Minimum clients for fit")
    parser.add_argument("--min-evaluate-clients", type=int, default=2, help="Minimum clients for evaluate")
    parser.add_argument("--min-available-clients", type=int, default=2, help="Minimum connected clients")
    parser.add_argument("--fraction-fit", type=float, default=1.0, help="Fraction of clients sampled for fit")
    parser.add_argument("--fraction-evaluate", type=float, default=1.0, help="Fraction of clients sampled for evaluate")
    parser.add_argument("--weights-out-dir", default="ai/fl_weights_next_hour_lstm_mlp", help="Directory to write global weights per round")
    parser.add_argument("--change-hidden-layers", default="128,64", help="Comma-separated hidden layer sizes for the change MLP branch")
    parser.add_argument("--lstm-hidden-dim", type=int, default=64, help="Hidden dimension size for the target LSTM branch")
    parser.add_argument("--lstm-num-layers", type=int, default=1, help="Number of stacked LSTM layers in the target branch")
    parser.add_argument("--lstm-head-hidden-dim", type=int, default=64, help="Dense head hidden dimension after the target LSTM")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for local training")
    parser.add_argument("--change-activation", choices=["relu", "tanh", "gelu"], default="relu", help="Activation function for the hybrid change MLP branch")
    parser.add_argument("--lstm-activation", choices=["relu", "tanh", "gelu"], default="relu", help="Activation function for the hybrid LSTM dense head")
    parser.add_argument("--aggregation-method", choices=["fedavg", "fedprox"], default="fedavg", help="Server aggregation strategy")
    parser.add_argument("--proximal-mu", type=float, default=0.0, help="FedProx proximal coefficient")
    return parser.parse_args()


def make_initial_parameters(
    change_hidden_layers: str = "128,64",
    lstm_hidden_dim: int = 64,
    lstm_num_layers: int = 1,
    lstm_head_hidden_dim: int = 64,
    change_activation: str = "relu",
    lstm_activation: str = "relu",
) -> fl.common.Parameters:
    model = make_model(
        get_input_dim(),
        change_hidden_layers=tuple(int(part.strip()) for part in change_hidden_layers.split(",") if part.strip()),
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_num_layers=lstm_num_layers,
        lstm_head_hidden_dim=lstm_head_hidden_dim,
        change_activation=change_activation,
        lstm_activation=lstm_activation,
    )
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


def append_rows(out_path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.DataFrame(rows)
    if os.path.exists(out_path):
        df.to_csv(out_path, mode="a", header=False, index=False)
    else:
        df.to_csv(out_path, index=False)


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
        self.latest_fit_summary: dict[str, float] | None = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        fit_examples = 0.0
        train_loss_sum = 0.0
        for _, fit_res in results:
            metrics = fit_res.metrics or {}
            count = float(fit_res.num_examples)
            fit_examples += count
            train_loss_sum += float(metrics.get("train_loss_sum", 0.0))
        self.latest_fit_summary = {
            "round": int(server_round),
            "trained_examples": int(fit_examples),
            "train_loss": train_loss_sum / max(fit_examples, 1.0),
        }
        append_rows(os.path.join(self.weights_out_dir, "train_metrics.csv"), [self.latest_fit_summary])
        if aggregated_parameters is not None:
            self.latest_parameters = aggregated_parameters
            round_weights_path = os.path.join(self.weights_out_dir, f"round_{server_round}_global_weights.npz")
            latest_weights_path = os.path.join(self.weights_out_dir, "latest_global_weights.npz")
            save_parameters(aggregated_parameters, round_weights_path)
            save_parameters(aggregated_parameters, latest_weights_path)
            print(f"saved_global_weights={round_weights_path}")
        print(
            f"[round {server_round}] training_summary "
            f"train_loss={self.latest_fit_summary['train_loss']:.4f} "
            f"trained_examples={self.latest_fit_summary['trained_examples']}"
        )
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
            "threshold_correct_y_temp_main": 0.0,
            "threshold_wrong_y_temp_main": 0.0,
            "threshold_correct_y_temp_toilet": 0.0,
            "threshold_wrong_y_temp_toilet": 0.0,
            "threshold_correct_y_light": 0.0,
            "threshold_wrong_y_light": 0.0,
            "threshold_correct_y_sound": 0.0,
            "threshold_wrong_y_sound": 0.0,
        }
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
                    "mae_y_temp_main": float(metrics.get("mae_sum_y_temp_main", 0.0)) / max(count, 1.0),
                    "mse_y_temp_main": float(metrics.get("mse_sum_y_temp_main", 0.0)) / max(count, 1.0),
                    "rmse_y_temp_main": (float(metrics.get("mse_sum_y_temp_main", 0.0)) / max(count, 1.0)) ** 0.5,
                    "threshold_accuracy_y_temp_main": float(metrics.get("threshold_correct_y_temp_main", 0.0)) / max(float(metrics.get("threshold_correct_y_temp_main", 0.0)) + float(metrics.get("threshold_wrong_y_temp_main", 0.0)), 1.0),
                    "mae_y_temp_toilet": float(metrics.get("mae_sum_y_temp_toilet", 0.0)) / max(count, 1.0),
                    "mse_y_temp_toilet": float(metrics.get("mse_sum_y_temp_toilet", 0.0)) / max(count, 1.0),
                    "rmse_y_temp_toilet": (float(metrics.get("mse_sum_y_temp_toilet", 0.0)) / max(count, 1.0)) ** 0.5,
                    "threshold_accuracy_y_temp_toilet": float(metrics.get("threshold_correct_y_temp_toilet", 0.0)) / max(float(metrics.get("threshold_correct_y_temp_toilet", 0.0)) + float(metrics.get("threshold_wrong_y_temp_toilet", 0.0)), 1.0),
                    "mae_y_light": float(metrics.get("mae_sum_y_light", 0.0)) / max(count, 1.0),
                    "mse_y_light": float(metrics.get("mse_sum_y_light", 0.0)) / max(count, 1.0),
                    "rmse_y_light": (float(metrics.get("mse_sum_y_light", 0.0)) / max(count, 1.0)) ** 0.5,
                    "threshold_accuracy_y_light": float(metrics.get("threshold_correct_y_light", 0.0)) / max(float(metrics.get("threshold_correct_y_light", 0.0)) + float(metrics.get("threshold_wrong_y_light", 0.0)), 1.0),
                    "mae_y_sound": float(metrics.get("mae_sum_y_sound", 0.0)) / max(count, 1.0),
                    "mse_y_sound": float(metrics.get("mse_sum_y_sound", 0.0)) / max(count, 1.0),
                    "rmse_y_sound": (float(metrics.get("mse_sum_y_sound", 0.0)) / max(count, 1.0)) ** 0.5,
                    "threshold_accuracy_y_sound": float(metrics.get("threshold_correct_y_sound", 0.0)) / max(float(metrics.get("threshold_correct_y_sound", 0.0)) + float(metrics.get("threshold_wrong_y_sound", 0.0)), 1.0),
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
            )

        count = max(summary["evaluated_examples"], 1.0)
        airflow_tp = summary["airflow_tp"]
        airflow_tn = summary["airflow_tn"]
        airflow_fp = summary["airflow_fp"]
        airflow_fn = summary["airflow_fn"]
        change_tp = summary["change_tp"]
        change_tn = summary["change_tn"]
        change_fp = summary["change_fp"]
        change_fn = summary["change_fn"]

        airflow_total = max(airflow_tp + airflow_tn + airflow_fp + airflow_fn, 1.0)
        airflow_precision = airflow_tp / max(airflow_tp + airflow_fp, 1.0)
        airflow_recall = airflow_tp / max(airflow_tp + airflow_fn, 1.0)
        airflow_f1 = 0.0
        if airflow_precision + airflow_recall > 0:
            airflow_f1 = 2.0 * airflow_precision * airflow_recall / (airflow_precision + airflow_recall)
        airflow_accuracy = (airflow_tp + airflow_tn) / airflow_total

        change_total = max(change_tp + change_tn + change_fp + change_fn, 1.0)
        change_precision = change_tp / max(change_tp + change_fp, 1.0)
        change_recall = change_tp / max(change_tp + change_fn, 1.0)
        change_f1 = 0.0
        if change_precision + change_recall > 0:
            change_f1 = 2.0 * change_precision * change_recall / (change_precision + change_recall)
        change_accuracy = (change_tp + change_tn) / change_total

        self.latest_eval_summary = {
            "round": int(server_round),
            "global_loss": float(aggregated_loss) if aggregated_loss is not None else 0.0,
            "mae_y_temp_main": summary["mae_sum_y_temp_main"] / count,
            "mse_y_temp_main": summary["mse_sum_y_temp_main"] / count,
            "mae_y_temp_toilet": summary["mae_sum_y_temp_toilet"] / count,
            "mse_y_temp_toilet": summary["mse_sum_y_temp_toilet"] / count,
            "mae_y_light": summary["mae_sum_y_light"] / count,
            "mse_y_light": summary["mse_sum_y_light"] / count,
            "mae_y_sound": summary["mae_sum_y_sound"] / count,
            "mse_y_sound": summary["mse_sum_y_sound"] / count,
            "rmse_y_temp_main": (summary["mse_sum_y_temp_main"] / count) ** 0.5,
            "rmse_y_temp_toilet": (summary["mse_sum_y_temp_toilet"] / count) ** 0.5,
            "rmse_y_light": (summary["mse_sum_y_light"] / count) ** 0.5,
            "rmse_y_sound": (summary["mse_sum_y_sound"] / count) ** 0.5,
            "threshold_accuracy_y_temp_main": summary["threshold_correct_y_temp_main"] / max(summary["threshold_correct_y_temp_main"] + summary["threshold_wrong_y_temp_main"], 1.0),
            "threshold_accuracy_y_temp_toilet": summary["threshold_correct_y_temp_toilet"] / max(summary["threshold_correct_y_temp_toilet"] + summary["threshold_wrong_y_temp_toilet"], 1.0),
            "threshold_accuracy_y_light": summary["threshold_correct_y_light"] / max(summary["threshold_correct_y_light"] + summary["threshold_wrong_y_light"], 1.0),
            "threshold_accuracy_y_sound": summary["threshold_correct_y_sound"] / max(summary["threshold_correct_y_sound"] + summary["threshold_wrong_y_sound"], 1.0),
            "regression_correct": int(summary["regression_correct"]),
            "regression_wrong": int(summary["regression_wrong"]),
            "regression_correct_rate": summary["regression_correct"] / max(summary["regression_correct"] + summary["regression_wrong"], 1.0),
            "airflow_accuracy": airflow_accuracy,
            "airflow_precision": airflow_precision,
            "airflow_recall": airflow_recall,
            "airflow_f1": airflow_f1,
            "airflow_correct": int(summary["airflow_correct"]),
            "airflow_incorrect": int(summary["airflow_incorrect"]),
            "airflow_tp": int(summary["airflow_tp"]),
            "airflow_tn": int(summary["airflow_tn"]),
            "airflow_fp": int(summary["airflow_fp"]),
            "airflow_fn": int(summary["airflow_fn"]),
            "change_accuracy": change_accuracy,
            "change_precision": change_precision,
            "change_recall": change_recall,
            "change_f1": change_f1,
            "change_correct": int(summary["change_correct"]),
            "change_incorrect": int(summary["change_incorrect"]),
            "change_tp": int(summary["change_tp"]),
            "change_tn": int(summary["change_tn"]),
            "change_fp": int(summary["change_fp"]),
            "change_fn": int(summary["change_fn"]),
            "evaluated_examples": int(summary["evaluated_examples"]),
        }
        append_rows(os.path.join(self.weights_out_dir, "total_metrics.csv"), [self.latest_eval_summary])
        append_rows(os.path.join(self.weights_out_dir, "room_metrics.csv"), room_rows)
        print(f"saved_total_metrics={os.path.join(self.weights_out_dir, 'total_metrics.csv')}")
        print(
            f"[round {server_round}] evaluation_summary "
            f"regression_correct_rate={self.latest_eval_summary['regression_correct_rate']:.4f} "
            f"airflow_f1={self.latest_eval_summary['airflow_f1']:.4f} "
            f"change_f1={self.latest_eval_summary['change_f1']:.4f} "
            f"mae_y_temp_main={self.latest_eval_summary['mae_y_temp_main']:.4f} "
            f"evaluated_examples={self.latest_eval_summary['evaluated_examples']}"
        )
        return aggregated_loss, aggregated_metrics


class TrackingFedProx(fl.server.strategy.FedProx):
    def __init__(self, weights_out_dir: str, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.weights_out_dir = weights_out_dir
        self.latest_parameters: fl.common.Parameters | None = None
        self.latest_eval_summary: dict[str, float] | None = None
        self.latest_fit_summary: dict[str, float] | None = None

    aggregate_fit = TrackingFedAvg.aggregate_fit
    aggregate_evaluate = TrackingFedAvg.aggregate_evaluate


def make_strategy(aggregation_method: str, weights_out_dir: str, proximal_mu: float, **kwargs: Any):
    if aggregation_method == "fedprox":
        return TrackingFedProx(weights_out_dir=weights_out_dir, proximal_mu=proximal_mu, **kwargs)
    return TrackingFedAvg(weights_out_dir=weights_out_dir, **kwargs)


def main() -> None:
    args = parse_args()
    split_dir = os.path.abspath(args.split_dir)
    weights_out_dir = os.path.abspath(args.weights_out_dir)
    rooms = count_rooms(split_dir)

    strategy = make_strategy(
        aggregation_method=args.aggregation_method,
        weights_out_dir=weights_out_dir,
        proximal_mu=args.proximal_mu,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=args.min_fit_clients,
        min_evaluate_clients=args.min_evaluate_clients,
        min_available_clients=args.min_available_clients,
        initial_parameters=make_initial_parameters(
            change_hidden_layers=args.change_hidden_layers,
            lstm_hidden_dim=args.lstm_hidden_dim,
            lstm_num_layers=args.lstm_num_layers,
            lstm_head_hidden_dim=args.lstm_head_hidden_dim,
            change_activation=args.change_activation,
            lstm_activation=args.lstm_activation,
        ),
    )

    print("fl_server_lstm_mlp.py starting")
    print(f"split_dir={split_dir}")
    print(f"targets={','.join(TARGET_COLUMNS)}")
    print(f"rooms_detected={rooms}")
    print(f"server_address={args.server_address}")
    print(f"rounds={args.rounds}")
    print(f"aggregation_method={args.aggregation_method}")
    print(f"proximal_mu={args.proximal_mu}")
    print(f"weights_out_dir={weights_out_dir}")
    print("waiting_for_room_clients...")

    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
if __name__ == "__main__":
    main()
