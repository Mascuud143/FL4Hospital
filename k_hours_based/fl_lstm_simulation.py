import argparse
import json
import os

import flwr as fl
import pandas as pd

try:
    from k_hours_based.fl_lstm_client import (
        RoomLSTMClient,
        TARGET_COLUMNS,
        build_sequence_arrays,
        get_input_dim,
        sanitize_targets,
    )
    from k_hours_based.fl_lstm_server import TrackingFedAvg, make_initial_parameters
except ModuleNotFoundError:
    from fl_lstm_client import (
        RoomLSTMClient,
        TARGET_COLUMNS,
        build_sequence_arrays,
        get_input_dim,
        sanitize_targets,
    )
    from fl_lstm_server import TrackingFedAvg, make_initial_parameters


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
    parser.add_argument("--chunksize", type=int, default=200000, help="CSV chunksize")
    return parser.parse_args()


def unique_room_ids(path: str, chunksize: int) -> list[str]:
    ids: set[str] = set()
    for chunk in pd.read_csv(path, usecols=["client_id"], chunksize=chunksize):
        ids.update(chunk["client_id"].astype(str).tolist())
    return sorted(ids, key=lambda x: int(x) if x.isdigit() else x)


def room_ids_from_stats(stats_path: str) -> list[str]:
    if not os.path.exists(stats_path):
        return []
    df = pd.read_csv(stats_path, usecols=["client_id"])
    ids = df["client_id"].astype(str).dropna().unique().tolist()
    return sorted(ids, key=lambda x: int(x) if x.isdigit() else x)


def load_filtered(path: str, room_ids: set[str], chunksize: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=lambda c: True, chunksize=chunksize):
        filtered = chunk[chunk["client_id"].astype(str).isin(room_ids)]
        if not filtered.empty:
            parts.append(filtered)
    if not parts:
        return pd.DataFrame(columns=["client_id", "t", "y_any_change", *TARGET_COLUMNS])
    df = pd.concat(parts, ignore_index=True)
    df["client_id"] = df["client_id"].astype(str)
    return df.loc[:, ~df.columns.duplicated()]


def build_room_sequences(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    room_ids: list[str],
    sequence_length: int,
) -> dict[str, tuple]:
    room_data: dict[str, tuple] = {}
    for rid in room_ids:
        room_train_df = train_df[train_df["client_id"] == rid]
        room_test_df = test_df[test_df["client_id"] == rid]
        x_train, y_train, _, _ = build_sequence_arrays(room_train_df, sequence_length)
        x_test, y_test, current_test, change_true = build_sequence_arrays(room_test_df, sequence_length)
        if len(y_train) == 0:
            continue
        # Each room contributes overlapping sequence windows rather than flat rows.
        room_data[rid] = (x_train, y_train, x_test, y_test, current_test, change_true)
    return room_data


def main() -> None:
    args = parse_args()
    split_dir = os.path.abspath(args.split_dir)
    stats_path = os.path.abspath(args.stats_path) if args.stats_path else os.path.join(split_dir, "split_stats_by_room.csv")
    train_path = os.path.join(split_dir, "next_hour_train.csv")
    test_path = os.path.join(split_dir, "next_hour_test.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing file: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing file: {test_path}")

    room_ids = room_ids_from_stats(stats_path)
    room_source = "split_stats_by_room.csv"
    if not room_ids:
        room_ids = unique_room_ids(train_path, chunksize=args.chunksize)
        room_source = "next_hour_train.csv"
    if args.room_id is not None:
        requested_room_id = str(args.room_id)
        room_ids = [rid for rid in room_ids if str(rid) == requested_room_id]
        room_source = f"requested_room_id={requested_room_id}"
    if args.max_rooms is not None:
        room_ids = room_ids[: max(1, args.max_rooms)]

    room_set = set(room_ids)
    print("[load] reading train split...")
    train_df = sanitize_targets(load_filtered(train_path, room_set, chunksize=args.chunksize))
    print("[load] reading test split...")
    test_df = sanitize_targets(load_filtered(test_path, room_set, chunksize=args.chunksize))
    print(f"[load] train_rows={len(train_df)} test_rows={len(test_df)}")

    print("[load] preparing room sequences...")
    room_data = build_room_sequences(train_df, test_df, room_ids, args.sequence_length)
    room_ids = sorted(room_data.keys(), key=lambda x: int(x) if x.isdigit() else x)
    if not room_ids:
        raise RuntimeError("No room sequence training data loaded for LSTM simulation.")

    input_dim = get_input_dim()
    weights_out_dir = os.path.abspath(args.weights_out_dir)
    strategy = TrackingFedAvg(
        weights_out_dir=weights_out_dir,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=min(args.min_fit_clients, len(room_ids)),
        min_evaluate_clients=min(args.min_evaluate_clients, len(room_ids)),
        min_available_clients=min(args.min_available_clients, len(room_ids)),
        initial_parameters=make_initial_parameters(),
    )

    def client_fn(cid: str):
        # Flower client ids are positional; here we map them back to actual room ids.
        rid = room_ids[int(cid)]
        x_train, y_train, x_test, y_test, current_test, change_true = room_data[rid]
        return RoomLSTMClient(
            room_id=rid,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            current_test=current_test,
            change_true=change_true,
            input_dim=input_dim,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            predictions_out_dir=os.path.abspath(args.predictions_out_dir) if args.predictions_out_dir else None,
        ).to_client()

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
            "sequence_length": args.sequence_length,
            "weights_out_dir": weights_out_dir,
            **strategy.latest_eval_summary,
        }
        with open(summary_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
