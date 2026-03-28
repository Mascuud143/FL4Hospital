import argparse
import json
import os

import flwr as fl
import numpy as np
import pandas as pd

try:
    from k_hours_based.fl_mlp_client import (
        CHANGE_BASELINE_COLUMNS,
        RoomClient,
        TARGET_COLUMNS,
        get_input_dim,
        row_to_input_vector,
        sanitize_targets,
    )
    from k_hours_based.fl_mlp_server import TrackingFedAvg, make_initial_parameters
except ModuleNotFoundError:
    from fl_mlp_client import (
        CHANGE_BASELINE_COLUMNS,
        RoomClient,
        TARGET_COLUMNS,
        get_input_dim,
        row_to_input_vector,
        sanitize_targets,
    )
    from fl_mlp_server import TrackingFedAvg, make_initial_parameters


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
    keep_cols = ["client_id", *TARGET_COLUMNS]
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=lambda c: True, chunksize=chunksize):
        filtered = chunk[chunk["client_id"].astype(str).isin(room_ids)]
        if not filtered.empty:
            parts.append(filtered)
    if not parts:
        return pd.DataFrame(columns=keep_cols)
    df = pd.concat(parts, ignore_index=True)
    df["client_id"] = df["client_id"].astype(str)
    return df.loc[:, ~df.columns.duplicated()]


def build_room_datasets(train_df: pd.DataFrame, test_df: pd.DataFrame, room_ids: list[str]) -> dict[str, tuple]:
    room_data: dict[str, tuple] = {}
    for rid in room_ids:
        room_train = train_df[train_df["client_id"] == rid]
        if len(room_train) == 0:
            continue
        room_test = test_df[test_df["client_id"] == rid]
        # Each room becomes one federated client with its own local train/test arrays.
        room_data[rid] = (
            row_to_input_vector(room_train),
            room_train[TARGET_COLUMNS].to_numpy(dtype=np.float64),
            row_to_input_vector(room_test),
            room_test[TARGET_COLUMNS].to_numpy(dtype=np.float64),
            room_test[CHANGE_BASELINE_COLUMNS].to_numpy(dtype=np.float64),
            room_test["y_any_change"].to_numpy(dtype=np.int64),
        )
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
    if args.max_rooms is not None:
        room_ids = room_ids[: max(1, args.max_rooms)]
    room_set = set(room_ids)

    print("[load] reading train split...")
    train_df = sanitize_targets(load_filtered(train_path, room_set, chunksize=args.chunksize))
    print("[load] reading test split...")
    test_df = sanitize_targets(load_filtered(test_path, room_set, chunksize=args.chunksize))
    print(f"[load] train_rows={len(train_df)} test_rows={len(test_df)}")

    room_data = build_room_datasets(train_df, test_df, room_ids)
    room_ids = sorted(room_data.keys(), key=lambda x: int(x) if x.isdigit() else x)
    if not room_ids:
        raise RuntimeError("No room training data loaded for simulation.")

    input_dim = get_input_dim()
    weights_out_dir = os.path.abspath(args.weights_out_dir)
    strategy = TrackingFedAvg(
        weights_out_dir=weights_out_dir,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=min(args.min_fit_clients, len(room_ids)),
        min_evaluate_clients=min(args.min_evaluate_clients, len(room_ids)),
        min_available_clients=min(args.min_available_clients, len(room_ids)),
        initial_parameters=make_initial_parameters(args.n_features),
    )

    def client_fn(cid: str):
        # Flower identifies clients as "0", "1", ...; we map those ids back to room ids.
        rid = room_ids[int(cid)]
        x_train, y_train, x_test, y_test, current_test, change_true = room_data[rid]
        return RoomClient(
            room_id=rid,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            current_test=current_test,
            change_true=change_true,
            input_dim=input_dim,
            local_epochs=args.local_epochs,
        ).to_client()

    print("fl_simulation.py starting")
    print(f"split_dir={split_dir}")
    print(f"targets={','.join(TARGET_COLUMNS)}")
    print(f"room_source={room_source}")
    print(f"stats_path={stats_path}")
    print(f"rooms_simulated={len(room_ids)}")
    print(f"rounds={args.rounds}")
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
