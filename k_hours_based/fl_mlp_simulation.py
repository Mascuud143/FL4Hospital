import argparse
import json
import os

import flwr as fl
import numpy as np
from per_room_data import list_room_ids
from fl_shared import available_client_workers, room_ids_from_stats, room_sort_key

_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in os.sys.path:
    os.sys.path.insert(0, _PARENT_DIR)

try:
    from k_hours_based.fl_mlp_client import (
        CHANGE_BASELINE_COLUMNS,
        RoomClient,
        TARGET_COLUMNS,
        get_input_dim,
        load_room_df,
        row_to_input_vector,
        sanitize_targets,
    )
    from k_hours_based.fl_mlp_server import make_initial_parameters, make_strategy
except ModuleNotFoundError:
    from fl_mlp_client import (
        CHANGE_BASELINE_COLUMNS,
        RoomClient,
        TARGET_COLUMNS,
        get_input_dim,
        load_room_df,
        row_to_input_vector,
        sanitize_targets,
    )
    from fl_mlp_server import make_initial_parameters, make_strategy

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower simulation mode for next-hour environment training.")
    parser.add_argument("--split-dir", default="ai/splits_next_hour", help="Directory with next_hour train and test CSV files")
    parser.add_argument("--rounds", type=int, default=5, help="Federated rounds")
    parser.add_argument("--n-features", type=int, default=256, help="Unused compatibility flag kept for compatibility")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--stats-path", default=None, help="Optional path to split_stats_by_room.csv")
    parser.add_argument("--fraction-fit", type=float, default=1.0, help="Fraction of clients sampled for fit")
    parser.add_argument("--fraction-evaluate", type=float, default=1.0, help="Fraction of clients sampled for evaluate")
    parser.add_argument("--min-fit-clients", type=int, default=2, help="Minimum clients for fit")
    parser.add_argument("--min-evaluate-clients", type=int, default=2, help="Minimum clients for evaluate")
    parser.add_argument("--min-available-clients", type=int, default=2, help="Minimum connected clients")
    parser.add_argument("--max-rooms", type=int, default=None, help="Optional cap on number of rooms")
    parser.add_argument("--weights-out-dir", default="ai/fl_weights_next_hour", help="Directory to write global weights per round")
    parser.add_argument("--summary-out", default=None, help="Optional JSON path to write final evaluation summary")
    parser.add_argument("--client-cpu", type=float, default=1.0, help="CPU resources per simulated client")
    parser.add_argument("--chunksize", type=int, default=50000, help="CSV chunksize")
    parser.add_argument("--hidden-layers", default="128,64,32", help="Comma-separated MLP hidden layer sizes")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for local training")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for local training")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam", help="Optimizer for local training")
    parser.add_argument("--activation", choices=["relu", "tanh", "logistic"], default="relu", help="Activation function for hidden layers")
    parser.add_argument("--aggregation-method", choices=["fedavg", "fedprox"], default="fedavg", help="Server aggregation strategy")
    parser.add_argument("--proximal-mu", type=float, default=0.0, help="FedProx proximal coefficient")
    parser.add_argument("--prebuild-workers", type=int, default=0, help="Client CSV prebuild workers; 0 derives from available workers * 2")
    return parser.parse_args()

def load_room_dataset(split_dir: str, rid: str, chunksize: int) -> tuple:
    room_train = sanitize_targets(load_room_df(split_dir, "train", room_id=rid, chunksize=chunksize))
    if len(room_train) == 0:
        raise RuntimeError(f"No training rows for room_id={rid}")
    room_test = sanitize_targets(load_room_df(split_dir, "test", room_id=rid, chunksize=chunksize))
    return (
        row_to_input_vector(room_train),
        room_train[TARGET_COLUMNS].to_numpy(dtype=np.float64),
        room_train[CHANGE_BASELINE_COLUMNS].to_numpy(dtype=np.float64),
        room_train["y_any_change"].to_numpy(dtype=np.int64),
        row_to_input_vector(room_test),
        room_test[TARGET_COLUMNS].to_numpy(dtype=np.float64),
        room_test[CHANGE_BASELINE_COLUMNS].to_numpy(dtype=np.float64),
        room_test["y_any_change"].to_numpy(dtype=np.int64),
    )


def cleanup_previous_outputs(weights_out_dir: str) -> None:
    os.makedirs(weights_out_dir, exist_ok=True)
    for name in [
        "total_metrics.csv",
        "room_metrics.csv",
        "train_metrics.csv",
        "train_room_metrics.csv",
        "local_test_metrics.csv",
        "local_test_room_metrics.csv",
        "latest_global_weights.npz",
    ]:
        path = os.path.join(weights_out_dir, name)
        if os.path.exists(path):
            os.remove(path)
    for name in os.listdir(weights_out_dir):
        if name.startswith("round_") and name.endswith("_global_weights.npz"):
            os.remove(os.path.join(weights_out_dir, name))

def main() -> None:
    args = parse_args()
    split_dir = os.path.abspath(args.split_dir)
    stats_path = os.path.abspath(args.stats_path) if args.stats_path else os.path.join(split_dir, "split_stats_by_room.csv")
    train_dir = os.path.join(split_dir, "train")
    test_dir = os.path.join(split_dir, "test")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Missing directory: {train_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Missing directory: {test_dir}")

    train_room_id_set = set(list_room_ids(split_dir, "train"))
    test_room_id_set = set(list_room_ids(split_dir, "test"))
    client_worker_count = available_client_workers(args.client_cpu)
    required_ready_client_csvs = max(1, client_worker_count * 2)

    room_ids = room_ids_from_stats(stats_path)
    room_source = "split_stats_by_room.csv"
    if not room_ids:
        room_ids = sorted(train_room_id_set, key=room_sort_key)
        room_source = "train/*.csv"
    room_ids = [rid for rid in room_ids if rid in train_room_id_set and rid in test_room_id_set]
    if not room_ids:
        raise RuntimeError("No clients with both training and test rows were found.")
    if args.max_rooms is not None:
        room_ids = room_ids[: max(1, args.max_rooms)]
    effective_min_available_clients = min(len(room_ids), max(args.min_available_clients, required_ready_client_csvs))
    input_dim = get_input_dim()
    weights_out_dir = os.path.abspath(args.weights_out_dir)
    cleanup_previous_outputs(weights_out_dir)
    strategy = make_strategy(
        aggregation_method=args.aggregation_method,
        weights_out_dir=weights_out_dir,
        proximal_mu=args.proximal_mu,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=min(args.min_fit_clients, len(room_ids)),
        min_evaluate_clients=min(args.min_evaluate_clients, len(room_ids)),
        min_available_clients=effective_min_available_clients,
        initial_parameters=make_initial_parameters(
            args.n_features,
            hidden_layers=args.hidden_layers,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            optimizer=args.optimizer,
            activation=args.activation,
        ),
    )

    def client_fn(cid: str):
        # Flower identifies clients as "0", "1", ...; we map those ids back to room ids.
        rid = room_ids[int(cid)]
        x_train, y_train, current_train, change_train, x_test, y_test, current_test, change_true = load_room_dataset(split_dir, rid, args.chunksize)
        return RoomClient(
            room_id=rid,
            x_train=x_train,
            y_train=y_train,
            current_train=current_train,
            change_train=change_train,
            x_test=x_test,
            y_test=y_test,
            current_test=current_test,
            change_true=change_true,
            input_dim=input_dim,
            local_epochs=args.local_epochs,
            hidden_layer_sizes=tuple(int(part.strip()) for part in args.hidden_layers.split(",") if part.strip()),
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            optimizer=args.optimizer,
            activation=args.activation,
        ).to_client()

    print("fl_simulation.py starting")
    print(f"split_dir={split_dir}")
    print(f"targets={','.join(TARGET_COLUMNS)}")
    print(f"room_source={room_source}")
    print(f"stats_path={stats_path}")
    print(f"rooms_simulated={len(room_ids)}")
    print(f"rounds={args.rounds}")
    print(f"hidden_layers={args.hidden_layers}")
    print(f"batch_size={args.batch_size}")
    print(f"learning_rate={args.learning_rate}")
    print(f"optimizer={args.optimizer}")
    print(f"activation={args.activation}")
    print(f"aggregation_method={args.aggregation_method}")
    print(f"proximal_mu={args.proximal_mu}")
    print(f"available_client_workers={client_worker_count}")
    print(f"required_ready_client_csvs={required_ready_client_csvs}")
    print(f"min_available_clients={effective_min_available_clients}")
    print(f"weights_out_dir={weights_out_dir}")
    print("[start] simulation running...")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(room_ids),
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={"num_cpus": args.client_cpu},
    )

    print("[done] simulation finished")
    if history.losses_distributed:
        print(f"[done] distributed_losses={history.losses_distributed}")

    if strategy.latest_eval_summary is not None and args.summary_out:
        summary_out = os.path.abspath(args.summary_out)
        os.makedirs(os.path.dirname(summary_out), exist_ok=True)
        payload = {
            "targets": TARGET_COLUMNS,
            "rounds": args.rounds,
            "rooms_simulated": len(room_ids),
            "weights_out_dir": weights_out_dir,
            **strategy.latest_eval_summary,
        }
        with open(summary_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
