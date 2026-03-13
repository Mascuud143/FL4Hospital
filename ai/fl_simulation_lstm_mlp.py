import argparse
import json
import os
import threading
import time

import flwr as fl
import pandas as pd

from fl_client_lstm_mlp import (
    RoomHybridClient,
    TARGET_COLUMNS,
    build_hybrid_arrays,
    get_input_dim,
    sanitize_targets,
)
from fl_server_lstm_mlp import TrackingFedAvg, make_initial_parameters


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
    parser.add_argument("--chunksize", type=int, default=200000, help="CSV chunksize")
    parser.add_argument("--server-address", default="127.0.0.1:8093", help="Used for threaded fallback mode")
    parser.add_argument("--server-start-wait", type=float, default=2.0, help="Seconds to wait before starting fallback clients")
    parser.add_argument("--client-retries", type=int, default=60, help="Retries for fallback client connections")
    parser.add_argument("--retry-wait", type=float, default=1.0, help="Seconds between fallback client retries")
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


def run_threaded_fallback(args: argparse.Namespace, strategy: TrackingFedAvg, room_ids: list[str], room_data: dict[str, tuple], input_dim: int) -> None:
    print("[fallback] starting local threaded Flower server+clients")

    server_errors: list[Exception] = []
    client_errors: list[tuple[str, str]] = []

    def server_target() -> None:
        try:
            fl.server.start_server(
                server_address=args.server_address,
                config=fl.server.ServerConfig(num_rounds=args.rounds),
                strategy=strategy,
            )
        except Exception as e:
            server_errors.append(e)

    def client_target(rid: str) -> None:
        (
            x_train_seq,
            x_train_flat,
            y_train,
            change_train,
            x_test_seq,
            x_test_flat,
            y_test,
            current_test,
            change_true,
        ) = room_data[rid]
        client = RoomHybridClient(
            room_id=rid,
            x_train_seq=x_train_seq,
            x_train_flat=x_train_flat,
            y_train=y_train,
            change_train=change_train,
            x_test_seq=x_test_seq,
            x_test_flat=x_test_flat,
            y_test=y_test,
            current_test=current_test,
            change_true=change_true,
            input_dim=input_dim,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
        )
        tries = 0
        while tries < args.client_retries:
            tries += 1
            try:
                fl.client.start_client(server_address=args.server_address, client=client.to_client())
                return
            except Exception as e:
                if tries >= args.client_retries:
                    client_errors.append((rid, str(e)))
                    return
                time.sleep(args.retry_wait)

    server_thread = threading.Thread(target=server_target, name="fl_lstm_mlp_server")
    server_thread.start()
    time.sleep(args.server_start_wait)

    client_threads: list[threading.Thread] = []
    for rid in room_ids:
        thread = threading.Thread(target=client_target, args=(rid,), name=f"fl_lstm_mlp_client_{rid}")
        thread.start()
        client_threads.append(thread)

    for thread in client_threads:
        thread.join()
    server_thread.join()

    if server_errors:
        raise RuntimeError(f"Threaded fallback server failed: {server_errors[0]}")
    if client_errors:
        sample = ", ".join([f"{rid}: {err}" for rid, err in client_errors[:3]])
        raise RuntimeError(f"Threaded fallback had {len(client_errors)} client failures. Sample: {sample}")
    print("[done] simulation finished (threaded fallback)")


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

    print("[load] preparing room hybrid datasets...")
    room_data: dict[str, tuple] = {}
    for idx, rid in enumerate(room_ids, start=1):
        room_train_df = train_df[train_df["client_id"] == rid]
        room_test_df = test_df[test_df["client_id"] == rid]
        x_train_seq, x_train_flat, y_train, _, change_train = build_hybrid_arrays(room_train_df, args.sequence_length)
        x_test_seq, x_test_flat, y_test, current_test, change_true = build_hybrid_arrays(room_test_df, args.sequence_length)
        if len(y_train) == 0:
            continue
        room_data[rid] = (
            x_train_seq,
            x_train_flat,
            y_train,
            change_train,
            x_test_seq,
            x_test_flat,
            y_test,
            current_test,
            change_true,
        )
        if idx % 25 == 0:
            print(f"[prep] rooms_prepared={idx}/{len(room_ids)}")

    room_ids = sorted(room_data.keys(), key=lambda x: int(x) if x.isdigit() else x)
    if not room_ids:
        raise RuntimeError("No room sequence training data loaded for hybrid LSTM+MLP simulation.")

    input_dim = get_input_dim()
    weights_out_dir = os.path.abspath(args.weights_out_dir)
    strategy = VerboseTrackingFedAvg(
        weights_out_dir=weights_out_dir,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=min(args.min_fit_clients, len(room_ids)),
        min_evaluate_clients=min(args.min_evaluate_clients, len(room_ids)),
        min_available_clients=min(args.min_available_clients, len(room_ids)),
        initial_parameters=make_initial_parameters(),
    )

    def client_fn(cid: str):
        rid = room_ids[int(cid)]
        (
            x_train_seq,
            x_train_flat,
            y_train,
            change_train,
            x_test_seq,
            x_test_flat,
            y_test,
            current_test,
            change_true,
        ) = room_data[rid]
        return RoomHybridClient(
            room_id=rid,
            x_train_seq=x_train_seq,
            x_train_flat=x_train_flat,
            y_train=y_train,
            change_train=change_train,
            x_test_seq=x_test_seq,
            x_test_flat=x_test_flat,
            y_test=y_test,
            current_test=current_test,
            change_true=change_true,
            input_dim=input_dim,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
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
        print("[done] simulation finished (ray backend)")
        if history.losses_distributed:
            print(f"[done] distributed_losses={history.losses_distributed}")
    except ImportError as exc:
        print(f"[fallback] ray simulation backend unavailable: {exc}")
        run_threaded_fallback(args, strategy, room_ids, room_data, input_dim)

    if args.summary_out and strategy.latest_eval_summary is not None:
        out_path = os.path.abspath(args.summary_out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([strategy.latest_eval_summary], f, indent=2)
        print(f"saved_summary={out_path}")


if __name__ == "__main__":
    main()
