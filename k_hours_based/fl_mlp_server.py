import argparse
import os
from typing import Any

import flwr as fl
from fl_shared import (
    accumulate_eval_metrics,
    build_latest_eval_summary,
    build_room_metric_row,
    count_split_rooms,
    empty_eval_metric_totals,
    extract_prefixed_eval_metrics,
    save_parameters_npz,
    upsert_metric_rows,
)

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
        train_summary = empty_eval_metric_totals(include_temperature=True)
        train_room_rows: list[dict[str, Any]] = []
        test_local_summary = empty_eval_metric_totals(include_temperature=True)
        test_local_room_rows: list[dict[str, Any]] = []
        for _, fit_res in results:
            metrics = fit_res.metrics or {}
            count = float(fit_res.num_examples)
            fit_examples += count
            train_loss_sum += float(metrics.get("train_loss_sum", 0.0))
            train_eval_loss, train_eval_examples, train_eval_metrics = extract_prefixed_eval_metrics("train_local", metrics)
            accumulate_eval_metrics(train_summary, train_eval_metrics, float(train_eval_examples))
            train_room_rows.append(
                build_room_metric_row(server_round, train_eval_examples, train_eval_loss, train_eval_metrics)
            )
            test_local_loss, test_local_examples, test_local_metrics = extract_prefixed_eval_metrics("test_local", metrics)
            accumulate_eval_metrics(test_local_summary, test_local_metrics, float(test_local_examples))
            test_local_room_rows.append(
                build_room_metric_row(server_round, test_local_examples, test_local_loss, test_local_metrics)
            )
        self.latest_fit_summary = build_latest_eval_summary(
            server_round,
            float(train_loss_sum) / max(float(fit_examples), 1.0),
            train_summary,
            include_temperature=True,
        )
        self.latest_fit_summary["trained_examples"] = int(fit_examples)
        self.latest_fit_summary["train_loss"] = float(train_loss_sum) / max(float(fit_examples), 1.0)
        local_test_summary = build_latest_eval_summary(
            server_round,
            float(test_local_summary["mae_sum_y_temp_main"] + test_local_summary["mae_sum_y_temp_toilet"] + test_local_summary["mae_sum_y_light"] + test_local_summary["mae_sum_y_sound"])
            / max(float(test_local_summary["evaluated_examples"]) * 4.0, 1.0),
            test_local_summary,
            include_temperature=True,
        )
        upsert_metric_rows(os.path.join(self.weights_out_dir, "train_metrics.csv"), [self.latest_fit_summary], ["round"])
        upsert_metric_rows(os.path.join(self.weights_out_dir, "train_room_metrics.csv"), train_room_rows, ["round", "room_id"])
        upsert_metric_rows(os.path.join(self.weights_out_dir, "local_test_metrics.csv"), [local_test_summary], ["round"])
        upsert_metric_rows(os.path.join(self.weights_out_dir, "local_test_room_metrics.csv"), test_local_room_rows, ["round", "room_id"])
        if aggregated_parameters is not None:
            self.latest_parameters = aggregated_parameters
            round_weights_path = os.path.join(self.weights_out_dir, f"round_{server_round}_global_weights.npz")
            latest_weights_path = os.path.join(self.weights_out_dir, "latest_global_weights.npz")
            save_parameters_npz(aggregated_parameters, round_weights_path)
            save_parameters_npz(aggregated_parameters, latest_weights_path)
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
        summary = empty_eval_metric_totals(include_temperature=True)
        room_rows: list[dict[str, Any]] = []
        for _, eval_res in results:
            metrics = eval_res.metrics or {}
            count = float(eval_res.num_examples)
            accumulate_eval_metrics(summary, metrics, count)
            room_rows.append(build_room_metric_row(server_round, int(eval_res.num_examples), float(eval_res.loss), metrics))

        self.latest_eval_summary = build_latest_eval_summary(
            server_round,
            aggregated_loss,
            summary,
            include_temperature=True,
        )
        upsert_metric_rows(os.path.join(self.weights_out_dir, "total_metrics.csv"), [self.latest_eval_summary], ["round"])
        upsert_metric_rows(os.path.join(self.weights_out_dir, "room_metrics.csv"), room_rows, ["round", "room_id"])
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
    rooms = count_split_rooms(split_dir)

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
