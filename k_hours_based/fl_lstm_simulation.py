import argparse
import json
import os

import flwr as fl
from flwr.common import Context
from per_room_data import list_room_ids
from fl_shared import available_client_workers, room_ids_from_stats, room_sort_key

_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in os.sys.path:
    os.sys.path.insert(0, _PARENT_DIR)

try:
    from k_hours_based.fl_lstm_client import (
        CHANGE_BASELINE_COLUMNS,
        INPUT_COLUMNS,
        RoomLSTMClient,
        TARGET_COLUMNS,
        build_sequence_arrays,
        get_input_dim,
        load_room_df,
        sanitize_targets,
    )
    from k_hours_based.fl_local_simulation import start_local_simulation
    from k_hours_based.fl_lstm_server import make_initial_parameters, make_strategy
except ModuleNotFoundError:
    from fl_lstm_client import (
        CHANGE_BASELINE_COLUMNS,
        INPUT_COLUMNS,
        RoomLSTMClient,
        TARGET_COLUMNS,
        build_sequence_arrays,
        get_input_dim,
        load_room_df,
        sanitize_targets,
    )
    from fl_local_simulation import start_local_simulation
    from fl_lstm_server import make_initial_parameters, make_strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower simulation mode for next-hour LSTM training.")
    parser.add_argument("--split-dir", default="ai/splits_next_hour", help="Directory with next_hour train and test CSV files")
    parser.add_argument("--rounds", type=int, default=5, help="Federated rounds")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--sequence-length", type=int, default=4, help="Number of historical rows per LSTM sample")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for local training")
    parser.add_argument("--stats-path", default=None, help="Optional path to split_stats_by_room.csv")
    parser.add_argument("--fraction-fit", type=float, default=1.0, help="Fraction of clients sampled for fit")
    parser.add_argument("--fraction-evaluate", type=float, default=1.0, help="Fraction of clients sampled for evaluate")
    parser.add_argument("--min-fit-clients", type=int, default=2, help="Minimum clients for fit")
    parser.add_argument("--min-evaluate-clients", type=int, default=2, help="Minimum clients for evaluate")
    parser.add_argument("--min-available-clients", type=int, default=2, help="Minimum connected clients")
    parser.add_argument("--max-rooms", type=int, default=None, help="Optional cap on number of rooms")
    parser.add_argument("--room-id", default=None, help="Optional specific room/client id to simulate")
    parser.add_argument("--weights-out-dir", default="ai/fl_weights_next_hour_lstm", help="Directory to write global weights per round")
    parser.add_argument("--summary-out", default=None, help="Optional JSON path to write final evaluation summary")
    parser.add_argument("--predictions-out-dir", default=None, help="Optional directory to write per-room evaluation predictions")
    parser.add_argument("--client-cpu", type=float, default=1.0, help="CPU resources per simulated client")
    parser.add_argument("--chunksize", type=int, default=50000, help="CSV chunksize")
    parser.add_argument("--prebuild-workers", type=int, default=0, help="Client CSV prebuild workers; 0 derives from available workers * 2")
    parser.add_argument("--hidden-dim", type=int, default=64, help="LSTM hidden dimension size")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of stacked LSTM layers")
    parser.add_argument("--head-hidden-dim", type=int, default=64, help="Dense head hidden dimension after the LSTM")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for local training")
    parser.add_argument("--optimizer", choices=["adam", "sgd", "rmsprop"], default="adam", help="Optimizer for local training")
    parser.add_argument("--activation", choices=["relu", "tanh", "gelu"], default="relu", help="Activation function for the dense head")
    parser.add_argument("--aggregation-method", choices=["fedavg", "fedprox"], default="fedavg", help="Server aggregation strategy")
    parser.add_argument("--proximal-mu", type=float, default=0.0, help="FedProx proximal coefficient")
    return parser.parse_args()

def load_room_sequences(split_dir: str, rid: str, chunksize: int, sequence_length: int) -> tuple:
    room_train_df = sanitize_targets(load_room_df(split_dir, "train", room_id=rid, chunksize=chunksize))
    room_test_df = sanitize_targets(load_room_df(split_dir, "test", room_id=rid, chunksize=chunksize))
    x_train, y_train, current_train, change_train = build_sequence_arrays(room_train_df, sequence_length)
    x_test, y_test, current_test, change_true = build_sequence_arrays(room_test_df, sequence_length)
    if len(y_train) == 0:
        raise RuntimeError(f"Not enough training rows for room_id={rid} and sequence_length={sequence_length}")
    return x_train, y_train, current_train, change_train, x_test, y_test, current_test, change_true


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
        if name.startswith("round_") and (
            name.endswith("_global_weights.npz")
            or name.endswith("_total_metrics.csv")
            or name.endswith("_room_metrics.csv")
        ):
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

    room_ids = room_ids_from_stats(stats_path)
    room_source = "split_stats_by_room.csv"
    if not room_ids:
        room_ids = sorted(train_room_id_set, key=room_sort_key)
        room_source = "train/*.csv"
    room_ids = [rid for rid in room_ids if rid in train_room_id_set and rid in test_room_id_set]
    if args.room_id is not None:
        requested_room_id = str(args.room_id)
        room_ids = [rid for rid in room_ids if str(rid) == requested_room_id]
        room_source = f"requested_room_id={requested_room_id}"
    if not room_ids:
        raise RuntimeError("No clients with both training and test rows were found.")
    if args.max_rooms is not None:
        room_ids = room_ids[: max(1, args.max_rooms)]

    input_dim = get_input_dim()
    client_worker_count = available_client_workers(args.client_cpu)
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
        min_available_clients=min(args.min_available_clients, len(room_ids)),
        initial_parameters=make_initial_parameters(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            head_hidden_dim=args.head_hidden_dim,
        ),
    )

    def build_client(cid: str) -> RoomLSTMClient:
        # Flower client ids are positional; here we map them back to actual room ids.
        rid = room_ids[int(cid)]
        x_train, y_train, current_train, change_train, x_test, y_test, current_test, change_true = load_room_sequences(split_dir, rid, args.chunksize, args.sequence_length)
        return RoomLSTMClient(
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
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            head_hidden_dim=args.head_hidden_dim,
            learning_rate=args.learning_rate,
            optimizer=args.optimizer,
            activation=args.activation,
            predictions_out_dir=os.path.abspath(args.predictions_out_dir) if args.predictions_out_dir else None,
        )

    def client_fn(context: Context):
        cid = str(context.node_config.get("partition-id", context.node_id))
        return build_client(cid).to_client()

    print("fl_simulation_lstm.py starting")
    print(f"split_dir={split_dir}")
    print(f"targets={','.join(TARGET_COLUMNS)}")
    print(f"room_source={room_source}")
    if args.room_id is not None:
        print(f"requested_room_id={args.room_id}")
    print(f"stats_path={stats_path}")
    print(f"rooms_simulated={len(room_ids)}")
    print(f"rounds={args.rounds}")
    print(f"sequence_length={args.sequence_length}")
    print(f"hidden_dim={args.hidden_dim}")
    print(f"num_layers={args.num_layers}")
    print(f"head_hidden_dim={args.head_hidden_dim}")
    print(f"learning_rate={args.learning_rate}")
    print(f"optimizer={args.optimizer}")
    print(f"activation={args.activation}")
    print(f"aggregation_method={args.aggregation_method}")
    print(f"proximal_mu={args.proximal_mu}")
    print(f"available_client_workers={client_worker_count}")
    print(f"weights_out_dir={weights_out_dir}")
    print("[start] simulation running...")
    try:
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=len(room_ids),
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
            client_resources={"num_cpus": args.client_cpu},
        )
    except ImportError as exc:
        if "Unable to import module `ray`" not in str(exc):
            raise
        print("[fallback] ray not installed; using local in-process simulation")
        history = start_local_simulation(
            client_factory=build_client,
            num_clients=len(room_ids),
            num_rounds=args.rounds,
            strategy=strategy,
            max_workers=client_worker_count,
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
            "sequence_length": args.sequence_length,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "head_hidden_dim": args.head_hidden_dim,
            "learning_rate": args.learning_rate,
            "optimizer": args.optimizer,
            "activation": args.activation,
            "aggregation_method": args.aggregation_method,
            "proximal_mu": args.proximal_mu,
            "weights_out_dir": weights_out_dir,
            **strategy.latest_eval_summary,
        }
        with open(summary_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
