import argparse
import json
import os

import flwr as fl
from per_room_data import list_room_ids
from fl_shared import room_ids_from_stats, room_sort_key

_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in os.sys.path:
    os.sys.path.insert(0, _PARENT_DIR)

try:
    from k_hours_based.fl_lstm_mlp_client import (
        CHANGE_BASELINE_COLUMNS,
        INPUT_COLUMNS,
        RoomHybridClient,
        TARGET_COLUMNS,
        build_hybrid_arrays,
        get_input_dim,
        load_room_df,
        sanitize_targets,
    )
    from k_hours_based.fl_lstm_mlp_server import TrackingFedAvg, TrackingFedProx, make_initial_parameters
except ModuleNotFoundError:
    from fl_lstm_mlp_client import (
        CHANGE_BASELINE_COLUMNS,
        INPUT_COLUMNS,
        RoomHybridClient,
        TARGET_COLUMNS,
        build_hybrid_arrays,
        get_input_dim,
        load_room_df,
        sanitize_targets,
    )
    from fl_lstm_mlp_server import TrackingFedAvg, TrackingFedProx, make_initial_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower simulation mode for next-hour hybrid MLP+LSTM training.")
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
    parser.add_argument("--weights-out-dir", default="ai/fl_weights_sim_lstm_mlp", help="Directory to write global weights per round")
    parser.add_argument("--summary-out", default=None, help="Optional JSON path to write final evaluation summary")
    parser.add_argument("--client-cpu", type=float, default=1.0, help="CPU resources per simulated client")
    parser.add_argument("--chunksize", type=int, default=50000, help="CSV chunksize")
    parser.add_argument("--prebuild-workers", type=int, default=0, help="Client CSV prebuild workers; 0 derives from available workers * 2")
    parser.add_argument("--change-hidden-layers", default="128,64", help="Comma-separated hidden layer sizes for the change MLP branch")
    parser.add_argument("--lstm-hidden-dim", type=int, default=64, help="Hidden dimension size for the target LSTM branch")
    parser.add_argument("--lstm-num-layers", type=int, default=1, help="Number of stacked LSTM layers in the target branch")
    parser.add_argument("--lstm-head-hidden-dim", type=int, default=64, help="Dense head hidden dimension after the target LSTM")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for local training")
    parser.add_argument("--optimizer", choices=["adam", "sgd", "rmsprop"], default="adam", help="Optimizer for local training")
    parser.add_argument("--change-activation", choices=["relu", "tanh", "gelu"], default="relu", help="Activation function for the hybrid change MLP branch")
    parser.add_argument("--lstm-activation", choices=["relu", "tanh", "gelu"], default="relu", help="Activation function for the hybrid LSTM dense head")
    parser.add_argument("--aggregation-method", choices=["fedavg", "fedprox"], default="fedavg", help="Server aggregation strategy")
    parser.add_argument("--proximal-mu", type=float, default=0.0, help="FedProx proximal coefficient")
    return parser.parse_args()

def load_room_hybrid_dataset(
    split_dir: str,
    rid: str,
    chunksize: int,
    sequence_length: int,
) -> tuple:
    room_train_df = sanitize_targets(load_room_df(split_dir, "train", room_id=rid, chunksize=chunksize))
    room_test_df = sanitize_targets(load_room_df(split_dir, "test", room_id=rid, chunksize=chunksize))
    x_train_seq, x_train_flat, y_train, current_train, change_train = build_hybrid_arrays(room_train_df, sequence_length)
    x_test_seq, x_test_flat, y_test, current_test, change_true = build_hybrid_arrays(room_test_df, sequence_length)
    if len(y_train) == 0:
        raise RuntimeError(f"Not enough training rows for room_id={rid} and sequence_length={sequence_length}")
    return (
        x_train_seq,
        x_train_flat,
        y_train,
        current_train,
        change_train,
        x_test_seq,
        x_test_flat,
        y_test,
        current_test,
        change_true,
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
        if name.startswith("round_") and (
            name.endswith("_global_weights.npz")
            or name.endswith("_total_metrics.csv")
            or name.endswith("_room_metrics.csv")
        ):
            os.remove(os.path.join(weights_out_dir, name))


class VerboseTrackingFedAvg(TrackingFedAvg):
    def configure_fit(self, server_round, parameters, client_manager):
        fit_cfg = super().configure_fit(server_round, parameters, client_manager)
        print(f"[round {server_round}] configure_fit sampled_clients={len(fit_cfg)}")
        return fit_cfg

    def aggregate_fit(self, server_round, results, failures):
        print(f"[round {server_round}] aggregate_fit received_results={len(results)} failures={len(failures)}")
        return super().aggregate_fit(server_round, results, failures)

    def configure_evaluate(self, server_round, parameters, client_manager):
        eval_cfg = super().configure_evaluate(server_round, parameters, client_manager)
        print(f"[round {server_round}] configure_evaluate sampled_clients={len(eval_cfg)}")
        return eval_cfg

    def aggregate_evaluate(self, server_round, results, failures):
        print(f"[round {server_round}] aggregate_evaluate received_results={len(results)} failures={len(failures)}")
        return super().aggregate_evaluate(server_round, results, failures)

class VerboseTrackingFedProx(TrackingFedProx):
    def configure_fit(self, server_round, parameters, client_manager):
        fit_cfg = super().configure_fit(server_round, parameters, client_manager)
        print(f"[round {server_round}] configure_fit sampled_clients={len(fit_cfg)}")
        return fit_cfg

    def aggregate_fit(self, server_round, results, failures):
        print(f"[round {server_round}] aggregate_fit received_results={len(results)} failures={len(failures)}")
        return super().aggregate_fit(server_round, results, failures)

    def configure_evaluate(self, server_round, parameters, client_manager):
        eval_cfg = super().configure_evaluate(server_round, parameters, client_manager)
        print(f"[round {server_round}] configure_evaluate sampled_clients={len(eval_cfg)}")
        return eval_cfg

    def aggregate_evaluate(self, server_round, results, failures):
        print(f"[round {server_round}] aggregate_evaluate received_results={len(results)} failures={len(failures)}")
        return super().aggregate_evaluate(server_round, results, failures)


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
    weights_out_dir = os.path.abspath(args.weights_out_dir)
    cleanup_previous_outputs(weights_out_dir)
    strategy_cls = VerboseTrackingFedProx if args.aggregation_method == "fedprox" else VerboseTrackingFedAvg
    strategy = strategy_cls(
        weights_out_dir=weights_out_dir,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=min(args.min_fit_clients, len(room_ids)),
        min_evaluate_clients=min(args.min_evaluate_clients, len(room_ids)),
        min_available_clients=min(args.min_available_clients, len(room_ids)),
        initial_parameters=make_initial_parameters(
            change_hidden_layers=args.change_hidden_layers,
            lstm_hidden_dim=args.lstm_hidden_dim,
            lstm_num_layers=args.lstm_num_layers,
            lstm_head_hidden_dim=args.lstm_head_hidden_dim,
        ),
        **({"proximal_mu": args.proximal_mu} if args.aggregation_method == "fedprox" else {}),
    )

    def client_fn(cid: str):
        rid = room_ids[int(cid)]
        (
            x_train_seq,
            x_train_flat,
            y_train,
            current_train,
            change_train,
            x_test_seq,
            x_test_flat,
            y_test,
            current_test,
            change_true,
        ) = load_room_hybrid_dataset(split_dir, rid, args.chunksize, args.sequence_length)
        return RoomHybridClient(
            room_id=rid,
            x_train_seq=x_train_seq,
            x_train_flat=x_train_flat,
            y_train=y_train,
            current_train=current_train,
            change_train=change_train,
            x_test_seq=x_test_seq,
            x_test_flat=x_test_flat,
            y_test=y_test,
            current_test=current_test,
            change_true=change_true,
            input_dim=input_dim,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            change_hidden_layers=tuple(int(part.strip()) for part in args.change_hidden_layers.split(",") if part.strip()),
            lstm_hidden_dim=args.lstm_hidden_dim,
            lstm_num_layers=args.lstm_num_layers,
            lstm_head_hidden_dim=args.lstm_head_hidden_dim,
            learning_rate=args.learning_rate,
            optimizer=args.optimizer,
            change_activation=args.change_activation,
            lstm_activation=args.lstm_activation,
        ).to_client()

    print("fl_simulation_lstm_mlp.py starting")
    print(f"split_dir={split_dir}")
    print(f"targets={','.join(TARGET_COLUMNS)}")
    print(f"room_source={room_source}")
    if args.room_id is not None:
        print(f"requested_room_id={args.room_id}")
    print(f"stats_path={stats_path}")
    print(f"rooms_simulated={len(room_ids)}")
    print(f"rounds={args.rounds}")
    print(f"sequence_length={args.sequence_length}")
    print(f"change_hidden_layers={args.change_hidden_layers}")
    print(f"lstm_hidden_dim={args.lstm_hidden_dim}")
    print(f"lstm_num_layers={args.lstm_num_layers}")
    print(f"lstm_head_hidden_dim={args.lstm_head_hidden_dim}")
    print(f"learning_rate={args.learning_rate}")
    print(f"optimizer={args.optimizer}")
    print(f"change_activation={args.change_activation}")
    print(f"lstm_activation={args.lstm_activation}")
    print(f"aggregation_method={args.aggregation_method}")
    print(f"proximal_mu={args.proximal_mu}")
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

    if args.summary_out and strategy.latest_eval_summary is not None:
        out_path = os.path.abspath(args.summary_out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([strategy.latest_eval_summary], f, indent=2)
        print(f"saved_summary={out_path}")


if __name__ == "__main__":
    main()
