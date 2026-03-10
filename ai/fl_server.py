import argparse
import os
from typing import Any

import flwr as fl
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower server (rooms connect as clients).")
    parser.add_argument("--split-dir", default="ai/splits", help="Directory with model_a_train.csv")
    parser.add_argument("--server-address", default="127.0.0.1:8080", help="Server bind address")
    parser.add_argument("--rounds", type=int, default=3, help="Federated rounds")
    parser.add_argument("--min-fit-clients", type=int, default=2, help="Minimum clients for fit")
    parser.add_argument("--min-evaluate-clients", type=int, default=2, help="Minimum clients for evaluate")
    parser.add_argument("--min-available-clients", type=int, default=2, help="Minimum connected clients")
    parser.add_argument("--fraction-fit", type=float, default=1.0, help="Fraction of clients sampled for fit")
    parser.add_argument("--fraction-evaluate", type=float, default=1.0, help="Fraction of clients sampled for evaluate")
    parser.add_argument("--n-features", type=int, default=256, help="Model feature size")
    parser.add_argument("--weights-out-dir", default="ai/fl_weights", help="Directory to write global weights per round")
    return parser.parse_args()


def make_initial_parameters(n_features: int) -> fl.common.Parameters:
    coef = np.zeros((1, n_features), dtype=np.float64)
    intercept = np.zeros((1,), dtype=np.float64)
    return fl.common.ndarrays_to_parameters([coef, intercept])


def count_rooms(split_dir: str) -> int:
    train_path = os.path.join(split_dir, "model_a_train.csv")
    if not os.path.exists(train_path):
        return 0
    df = pd.read_csv(train_path, usecols=["client_id"])
    return int(df["client_id"].astype(str).nunique())


def save_parameters(params: fl.common.Parameters, out_path: str) -> None:
    ndarrays = fl.common.parameters_to_ndarrays(params)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, coef=ndarrays[0], intercept=ndarrays[1])


def save_parameters_readable(params: fl.common.Parameters, out_prefix: str) -> None:
    ndarrays = fl.common.parameters_to_ndarrays(params)
    coef = ndarrays[0]
    intercept = ndarrays[1]
    coef_flat = coef.reshape(-1)
    coef_csv = f"{out_prefix}_coef.csv"
    intercept_csv = f"{out_prefix}_intercept.csv"
    summary_txt = f"{out_prefix}_summary.txt"
    os.makedirs(os.path.dirname(coef_csv), exist_ok=True)
    coef_df = pd.DataFrame({"feature_index": np.arange(coef_flat.shape[0]), "coef": coef_flat})
    coef_df.to_csv(coef_csv, index=False)
    pd.DataFrame({"intercept": intercept.reshape(-1)}).to_csv(intercept_csv, index=False)
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"coef_shape={coef.shape}\n")
        f.write(f"intercept_shape={intercept.shape}\n")
        f.write(f"coef_min={float(np.min(coef_flat))}\n")
        f.write(f"coef_max={float(np.max(coef_flat))}\n")
        f.write(f"coef_mean={float(np.mean(coef_flat))}\n")
        f.write(f"coef_std={float(np.std(coef_flat))}\n")
        f.write(f"intercept={intercept.reshape(-1).tolist()}\n")


class TrackingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, weights_out_dir: str, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.weights_out_dir = weights_out_dir
        self.latest_parameters: fl.common.Parameters | None = None
        self.latest_eval_summary: dict[str, int] | None = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            self.latest_parameters = aggregated_parameters
            out_path = os.path.join(self.weights_out_dir, f"global_round_{server_round}.npz")
            save_parameters(aggregated_parameters, out_path)
            save_parameters_readable(
                aggregated_parameters,
                os.path.join(self.weights_out_dir, f"global_round_{server_round}"),
            )
            print(f"saved_global_weights={out_path}")
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        summary = {
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "correct": 0,
            "incorrect": 0,
            "evaluated_examples": 0,
        }
        for _, eval_res in results:
            metrics = eval_res.metrics or {}
            summary["tp"] += int(metrics.get("tp", 0))
            summary["tn"] += int(metrics.get("tn", 0))
            summary["fp"] += int(metrics.get("fp", 0))
            summary["fn"] += int(metrics.get("fn", 0))
            summary["correct"] += int(metrics.get("correct", 0))
            summary["incorrect"] += int(metrics.get("incorrect", 0))
            summary["evaluated_examples"] += int(eval_res.num_examples)

        self.latest_eval_summary = summary
        print(
            f"[round {server_round}] evaluation_summary "
            f"correct={summary['correct']} incorrect={summary['incorrect']} "
            f"tp={summary['tp']} tn={summary['tn']} fp={summary['fp']} fn={summary['fn']} "
            f"evaluated_examples={summary['evaluated_examples']}"
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
        initial_parameters=make_initial_parameters(args.n_features),
    )

    print("fl_server.py starting")
    print(f"split_dir={split_dir}")
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
    if strategy.latest_parameters is not None:
        final_path = os.path.join(weights_out_dir, "global_final.npz")
        save_parameters(strategy.latest_parameters, final_path)
        save_parameters_readable(strategy.latest_parameters, os.path.join(weights_out_dir, "global_final"))
        print(f"saved_global_weights_final={final_path}")


if __name__ == "__main__":
    main()
