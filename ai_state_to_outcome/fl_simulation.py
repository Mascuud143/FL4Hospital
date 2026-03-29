import argparse
import json
import os
import threading
import time

import flwr as fl
import pandas as pd

from fl_client import RoomClient, TARGET_COLUMNS, build_target_matrix, get_input_dim, row_to_input_vector, sanitize_rows
from fl_server import TrackingFedAvg, make_initial_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower simulation mode for Task #2 state-to-outcome training.")
    parser.add_argument("--split-dir", default="ai_state_to_outcome/splits", help="Directory with state_to_outcome train and test CSV files")
    parser.add_argument("--rounds", type=int, default=5, help="Federated rounds")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for local training")
    parser.add_argument("--fraction-fit", type=float, default=1.0, help="Fraction of clients sampled for fit")
    parser.add_argument("--fraction-evaluate", type=float, default=1.0, help="Fraction of clients sampled for evaluate")
    parser.add_argument("--min-fit-clients", type=int, default=2, help="Minimum clients for fit")
    parser.add_argument("--min-evaluate-clients", type=int, default=2, help="Minimum clients for evaluate")
    parser.add_argument("--min-available-clients", type=int, default=2, help="Minimum connected clients")
    parser.add_argument("--max-rooms", type=int, default=None, help="Optional cap on number of rooms")
    parser.add_argument("--weights-out-dir", default="ai_state_to_outcome/fl_weights", help="Directory to write global weights per round")
    parser.add_argument("--summary-out", default=None, help="Optional JSON path to write final evaluation summary")
    parser.add_argument("--client-cpu", type=float, default=1.0, help="CPU resources per simulated client")
    parser.add_argument("--chunksize", type=int, default=100000, help="CSV chunksize")
    parser.add_argument("--server-address", default="127.0.0.1:8096", help="Used for threaded fallback mode")
    parser.add_argument("--server-start-wait", type=float, default=2.0, help="Seconds to wait before starting fallback clients")
    parser.add_argument("--client-retries", type=int, default=60, help="Retries for fallback client connections")
    parser.add_argument("--retry-wait", type=float, default=1.0, help="Seconds between fallback client retries")
    return parser.parse_args()


def unique_room_ids(path: str, chunksize: int) -> list[str]:
    ids: set[str] = set()
    for chunk in pd.read_csv(path, usecols=["room_id"], chunksize=chunksize):
        ids.update(chunk["room_id"].astype(str).tolist())
    return sorted(ids, key=lambda x: int(x) if x.isdigit() else x)


def load_filtered(path: str, room_ids: set[str], chunksize: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=lambda c: True, chunksize=chunksize):
        filtered = chunk[chunk["room_id"].astype(str).isin(room_ids)]
        if not filtered.empty:
            parts.append(filtered)
    if not parts:
        return pd.DataFrame(columns=["room_id"])
    df = pd.concat(parts, ignore_index=True)
    df["room_id"] = df["room_id"].astype(str)
    return df.loc[:, ~df.columns.duplicated()]


def main() -> None:
    args = parse_args()
    split_dir = os.path.abspath(args.split_dir)
    train_path = os.path.join(split_dir, "state_to_outcome_train.csv")
    test_path = os.path.join(split_dir, "state_to_outcome_test.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing file: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing file: {test_path}")

    room_ids = unique_room_ids(train_path, chunksize=args.chunksize)
    if args.max_rooms is not None:
        room_ids = room_ids[: max(1, args.max_rooms)]
    room_set = set(room_ids)

    train_df = sanitize_rows(load_filtered(train_path, room_set, chunksize=args.chunksize))
    test_df = sanitize_rows(load_filtered(test_path, room_set, chunksize=args.chunksize))

    room_data: dict[str, tuple] = {}
    for rid in room_ids:
        room_train_df = train_df[train_df["room_id"] == rid]
        if len(room_train_df) == 0:
            continue
        room_test_df = test_df[test_df["room_id"] == rid]
        x_train = row_to_input_vector(room_train_df).astype("float32")
        y_train = build_target_matrix(room_train_df)
        x_test = row_to_input_vector(room_test_df).astype("float32")
        y_test = build_target_matrix(room_test_df)
        room_data[rid] = (x_train, y_train, x_test, y_test)

    room_ids = sorted(room_data.keys(), key=lambda x: int(x) if x.isdigit() else x)
    if not room_ids:
        raise RuntimeError("No room training data loaded for simulation.")

    strategy = TrackingFedAvg(
        weights_out_dir=os.path.abspath(args.weights_out_dir),
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=min(args.min_fit_clients, len(room_ids)),
        min_evaluate_clients=min(args.min_evaluate_clients, len(room_ids)),
        min_available_clients=min(args.min_available_clients, len(room_ids)),
        initial_parameters=make_initial_parameters(),
    )
    input_dim = get_input_dim()

    def client_fn(cid: str):
        rid = room_ids[int(cid)]
        x_train, y_train, x_test, y_test = room_data[rid]
        return RoomClient(
            room_id=rid,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            input_dim=input_dim,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
        ).to_client()

    try:
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=len(room_ids),
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
            client_resources={"num_cpus": args.client_cpu},
        )
    except ImportError:
        server_errors: list[Exception] = []
        client_errors: list[tuple[str, str]] = []

        def server_target() -> None:
            try:
                fl.server.start_server(
                    server_address=args.server_address,
                    config=fl.server.ServerConfig(num_rounds=args.rounds),
                    strategy=strategy,
                )
            except Exception as exc:  # pragma: no cover
                server_errors.append(exc)

        def client_target(rid: str) -> None:
            x_train, y_train, x_test, y_test = room_data[rid]
            client = RoomClient(
                room_id=rid,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
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
                except Exception as exc:  # pragma: no cover
                    if tries >= args.client_retries:
                        client_errors.append((rid, str(exc)))
                        return
                    time.sleep(args.retry_wait)

        server_thread = threading.Thread(target=server_target, name="state_to_outcome_server")
        server_thread.start()
        time.sleep(args.server_start_wait)
        client_threads: list[threading.Thread] = []
        for rid in room_ids:
            thread = threading.Thread(target=client_target, args=(rid,), name=f"state_to_outcome_client_{rid}")
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

    if strategy.latest_eval_summary is not None and args.summary_out:
        summary_out = os.path.abspath(args.summary_out)
        os.makedirs(os.path.dirname(summary_out), exist_ok=True)
        payload = {
            "targets": TARGET_COLUMNS,
            "rounds": args.rounds,
            "rooms_simulated": len(room_ids),
            "weights_out_dir": os.path.abspath(args.weights_out_dir),
            **strategy.latest_eval_summary,
        }
        with open(summary_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
