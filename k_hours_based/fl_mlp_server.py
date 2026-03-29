import argparse
import os
from typing import Any

import flwr as fl
import numpy as np
import pandas as pd

try:
    from k_hours_based.fl_mlp_client import TARGET_COLUMNS, get_input_dim, get_params, make_model, parse_hidden_layers
except ModuleNotFoundError:
    from fl_mlp_client import TARGET_COLUMNS, get_input_dim, get_params, make_model, parse_hidden_layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower server for next-hour environment training.")
    parser.add_argument("--split-dir", default="ai/splits_next_hour", help="Directory with next_hour train split")
    parser.add_argument("--server-address", default="127.0.0.1:8080", help="Server bind address")
    parser.add_argument("--rounds", type=int, default=3, help="Federated rounds")
    parser.add_argument("--min-fit-clients", type=int, default=2, help="Minimum clients for fit")
    parser.add_argument("--min-evaluate-clients", type=int, default=2, help="Minimum clients for evaluate")
    parser.add_argument("--min-available-clients", type=int, default=2, help="Minimum connected clients")
    parser.add_argument("--fraction-fit", type=float, default=1.0, help="Fraction of clients sampled for fit")
    parser.add_argument("--fraction-evaluate", type=float, default=1.0, help="Fraction of clients sampled for evaluate")
    parser.add_argument("--n-features", type=int, default=256, help="Unused compatibility flag kept for compatibility")
    parser.add_argument("--weights-out-dir", default="ai/fl_weights_next_hour", help="Directory to write global weights per round")
    parser.add_argument("--hidden-layers", default="128,64,32", help="Comma-separated MLP hidden layer sizes")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for local training")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for local training")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam", help="Optimizer for local training")
    parser.add_argument("--activation", choices=["relu", "tanh", "logistic"], default="relu", help="Activation function for hidden layers")
    parser.add_argument("--aggregation-method", choices=["fedavg", "fedprox"], default="fedavg", help="Server aggregation strategy")
    parser.add_argument("--proximal-mu", type=float, default=0.0, help="FedProx proximal coefficient")
    return parser.parse_args()


def make_initial_parameters(
    _: int,
    hidden_layers: str = "128,64,32",
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    optimizer: str = "adam",
    activation: str = "relu",
) -> fl.common.Parameters:
    # Server round 1 starts from the same fresh MLP architecture used by every client.
    model = make_model(
        get_input_dim(),
        hidden_layer_sizes=parse_hidden_layers(hidden_layers),
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer=optimizer,
        activation=activation,
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
        existing_keys = set(existing.columns)
        new_keys = set(new_df.columns)
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
        merged.to_csv(out_path, index=False)
    else:
        new_df.to_csv(out_path, index=False)


class TrackingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, weights_out_dir: str, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.weights_out_dir = weights_out_dir
        self.latest_parameters: fl.common.Parameters | None = None
        self.latest_eval_summary: dict[str, float] | None = None
        self.latest_fit_summary: dict[str, float] | None = None

    def aggregate_fit(self, server_round, results, failures):
        # Flower averages the client-updated models here to form the next global model.
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
        upsert_rows(os.path.join(self.weights_out_dir, "train_metrics.csv"), [self.latest_fit_summary], ["round"])
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
        # Clients send sums and counts, and the server turns them into dataset-wide metrics.
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        summary = {
            "evaluated_examples": 0.0,
            "regression_correct": 0.0,
            "regression_wrong": 0.0,
            "temperature_correct": 0.0,
            "temperature_wrong": 0.0,
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
            "temperature_correct": int(summary["temperature_correct"]),
            "temperature_wrong": int(summary["temperature_wrong"]),
            "temperature_correct_rate": summary["temperature_correct"] / max(summary["temperature_correct"] + summary["temperature_wrong"], 1.0),
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
        upsert_rows(os.path.join(self.weights_out_dir, "total_metrics.csv"), [self.latest_eval_summary], ["round"])
        upsert_rows(os.path.join(self.weights_out_dir, "room_metrics.csv"), room_rows, ["round", "room_id"])
        print(f"saved_total_metrics={os.path.join(self.weights_out_dir, 'total_metrics.csv')}")
        print(
            f"[round {server_round}] evaluation_summary "
            f"regression_correct_rate={self.latest_eval_summary['regression_correct_rate']:.4f} "
            f"temperature_correct_rate={self.latest_eval_summary['temperature_correct_rate']:.4f} "
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
            args.n_features,
            hidden_layers=args.hidden_layers,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            optimizer=args.optimizer,
            activation=args.activation,
        ),
    )

    print("fl_server.py starting")
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
