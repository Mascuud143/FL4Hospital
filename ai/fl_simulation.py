import argparse
import os
import threading
import time

import flwr as fl
import pandas as pd
from sklearn.feature_extraction import FeatureHasher

from fl_client import FEATURE_COLUMNS, RoomClient, rows_to_dicts
from fl_server import TrackingFedAvg, make_initial_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower simulation mode (rooms as virtual clients).")
    parser.add_argument("--split-dir", default="ai/splits", help="Directory with model_a_train.csv and model_a_test.csv")
    parser.add_argument(
        "--stats-path",
        default=None,
        help="Optional path to split_stats_by_room.csv (default: <split-dir>/split_stats_by_room.csv)",
    )
    parser.add_argument("--rounds", type=int, default=5, help="Federated rounds")
    parser.add_argument("--n-features", type=int, default=256, help="FeatureHasher output dimension")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--fraction-fit", type=float, default=1.0, help="Fraction of clients sampled for fit")
    parser.add_argument("--fraction-evaluate", type=float, default=1.0, help="Fraction of clients sampled for evaluate")
    parser.add_argument("--min-fit-clients", type=int, default=2, help="Minimum clients for fit")
    parser.add_argument("--min-evaluate-clients", type=int, default=2, help="Minimum clients for evaluate")
    parser.add_argument("--min-available-clients", type=int, default=2, help="Minimum connected clients")
    parser.add_argument("--max-rooms", type=int, default=None, help="Optional cap on number of rooms")
    parser.add_argument("--weights-out-dir", default="ai/fl_weights_sim", help="Directory to write global weights per round")
    parser.add_argument("--client-cpu", type=float, default=1.0, help="CPU resources per simulated client")
    parser.add_argument("--chunksize", type=int, default=200000, help="CSV chunksize")
    parser.add_argument("--server-address", default="127.0.0.1:8090", help="Used for threaded fallback mode")
    parser.add_argument("--server-start-wait", type=float, default=2.0, help="Seconds to wait before starting fallback clients")
    parser.add_argument("--client-retries", type=int, default=60, help="Retries for fallback client connections")
    parser.add_argument("--retry-wait", type=float, default=1.0, help="Seconds between fallback client retries")
    return parser.parse_args()


def unique_room_ids(path: str, chunksize: int) -> list[str]:
    ids: set[str] = set()
    chunk_idx = 0
    for chunk in pd.read_csv(path, usecols=["client_id"], chunksize=chunksize):
        chunk_idx += 1
        ids.update(chunk["client_id"].astype(str).tolist())
        if chunk_idx % 10 == 0:
            print(f"[rooms-scan] chunks={chunk_idx} unique_rooms_so_far={len(ids)}")
    return sorted(ids, key=lambda x: int(x) if x.isdigit() else x)


def room_ids_from_stats(stats_path: str) -> list[str]:
    if not os.path.exists(stats_path):
        return []
    df = pd.read_csv(stats_path, usecols=["client_id"])
    ids = df["client_id"].astype(str).dropna().unique().tolist()
    return sorted(ids, key=lambda x: int(x) if x.isdigit() else x)


def load_filtered(path: str, room_ids: set[str], chunksize: int) -> pd.DataFrame:
    keep_cols = ["client_id"] + FEATURE_COLUMNS + ["y_event"]
    parts: list[pd.DataFrame] = []
    chunk_idx = 0
    kept_rows = 0
    for chunk in pd.read_csv(path, usecols=lambda c: c in keep_cols, chunksize=chunksize):
        chunk_idx += 1
        filtered = chunk[chunk["client_id"].astype(str).isin(room_ids)]
        if not filtered.empty:
            parts.append(filtered)
            kept_rows += len(filtered)
        if chunk_idx % 10 == 0:
            print(f"[load] file={os.path.basename(path)} chunks={chunk_idx} kept_rows={kept_rows}")
    if not parts:
        return pd.DataFrame(columns=keep_cols)
    df = pd.concat(parts, ignore_index=True)
    df["client_id"] = df["client_id"].astype(str)
    df["y_event"] = pd.to_numeric(df["y_event"], errors="coerce").fillna(0).astype(int)
    return df


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


def main() -> None:
    args = parse_args()
    split_dir = os.path.abspath(args.split_dir)
    stats_path = (
        os.path.abspath(args.stats_path)
        if args.stats_path
        else os.path.join(split_dir, "split_stats_by_room.csv")
    )
    train_path = os.path.join(split_dir, "model_a_train.csv")
    test_path = os.path.join(split_dir, "model_a_test.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing file: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing file: {test_path}")

    room_ids = room_ids_from_stats(stats_path)
    room_source = "split_stats_by_room.csv"
    if not room_ids:
        room_ids = unique_room_ids(train_path, chunksize=args.chunksize)
        room_source = "model_a_train.csv"
    if args.max_rooms is not None:
        room_ids = room_ids[: max(1, args.max_rooms)]
    room_set = set(room_ids)

    print("[load] reading train split...")
    train_df = load_filtered(train_path, room_set, chunksize=args.chunksize)
    print("[load] reading test split...")
    test_df = load_filtered(test_path, room_set, chunksize=args.chunksize)
    print(f"[load] train_rows={len(train_df)} test_rows={len(test_df)}")

    hasher = FeatureHasher(n_features=args.n_features, input_type="dict", alternate_sign=False)
    room_data: dict[str, tuple] = {}
    for idx, rid in enumerate(room_ids, start=1):
        r_train = train_df[train_df["client_id"] == rid]
        if len(r_train) == 0:
            continue
        r_test = test_df[test_df["client_id"] == rid]
        x_train = hasher.transform(rows_to_dicts(r_train))
        y_train = r_train["y_event"].to_numpy(dtype="int64")
        x_test = hasher.transform(rows_to_dicts(r_test))
        y_test = r_test["y_event"].to_numpy(dtype="int64")
        room_data[rid] = (x_train, y_train, x_test, y_test)
        if idx % 25 == 0:
            print(f"[prep] rooms_prepared={idx}/{len(room_ids)}")

    room_ids = sorted(room_data.keys(), key=lambda x: int(x) if x.isdigit() else x)
    if not room_ids:
        raise RuntimeError("No room training data loaded for simulation.")

    weights_out_dir = os.path.abspath(args.weights_out_dir)
    strategy = VerboseTrackingFedAvg(
        weights_out_dir=weights_out_dir,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=min(args.min_fit_clients, len(room_ids)),
        min_evaluate_clients=min(args.min_evaluate_clients, len(room_ids)),
        min_available_clients=min(args.min_available_clients, len(room_ids)),
        initial_parameters=make_initial_parameters(args.n_features),
    )

    def client_fn(cid: str):
        rid = room_ids[int(cid)]
        x_train, y_train, x_test, y_test = room_data[rid]
        return RoomClient(
            room_id=rid,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            n_features=args.n_features,
            local_epochs=args.local_epochs,
        ).to_client()

    print("fl_simulation.py starting")
    print(f"split_dir={split_dir}")
    print(f"room_source={room_source}")
    print(f"stats_path={stats_path}")
    print(f"rooms_simulated={len(room_ids)}")
    print(f"rounds={args.rounds}")
    print(f"weights_out_dir={weights_out_dir}")
    print("[start] simulation running...")

    history = None
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
        print("[fallback] starting local threaded Flower server+clients")

        server_errors: list[Exception] = []
        client_errors: list[tuple[str, str]] = []
        lock = threading.Lock()

        def server_target() -> None:
            try:
                fl.server.start_server(
                    server_address=args.server_address,
                    config=fl.server.ServerConfig(num_rounds=args.rounds),
                    strategy=strategy,
                )
            except Exception as e:  # pragma: no cover
                with lock:
                    server_errors.append(e)

        def client_target(rid: str) -> None:
            x_train, y_train, x_test, y_test = room_data[rid]
            client = RoomClient(
                room_id=rid,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                n_features=args.n_features,
                local_epochs=args.local_epochs,
            )
            tries = 0
            while tries < args.client_retries:
                tries += 1
                try:
                    fl.client.start_client(
                        server_address=args.server_address,
                        client=client.to_client(),
                    )
                    return
                except Exception as e:
                    if tries >= args.client_retries:
                        with lock:
                            client_errors.append((rid, str(e)))
                        return
                    time.sleep(args.retry_wait)

        server_thread = threading.Thread(target=server_target, name="fl_sim_fallback_server")
        server_thread.start()
        time.sleep(args.server_start_wait)

        client_threads: list[threading.Thread] = []
        for rid in room_ids:
            t = threading.Thread(target=client_target, args=(rid,), name=f"fl_sim_client_{rid}")
            t.start()
            client_threads.append(t)

        for t in client_threads:
            t.join()
        server_thread.join()

        if server_errors:
            raise RuntimeError(f"Threaded fallback server failed: {server_errors[0]}")
        if client_errors:
            sample = ", ".join([f"{rid}: {err}" for rid, err in client_errors[:3]])
            raise RuntimeError(f"Threaded fallback had {len(client_errors)} client failures. Sample: {sample}")
        print("[done] simulation finished (threaded fallback)")

    if strategy.latest_parameters is not None:
        from fl_server import save_parameters, save_parameters_readable

        final_path = os.path.join(weights_out_dir, "global_final.npz")
        save_parameters(strategy.latest_parameters, final_path)
        save_parameters_readable(strategy.latest_parameters, os.path.join(weights_out_dir, "global_final"))
        print(f"saved_global_weights_final={final_path}")
    if strategy.latest_eval_summary is not None:
        summary = strategy.latest_eval_summary
        print("[final] evaluation prediction summary")
        print(f"[final] correct={summary['correct']} incorrect={summary['incorrect']}")
        print(f"[final] tp={summary['tp']} tn={summary['tn']} fp={summary['fp']} fn={summary['fn']}")
        print(f"[final] evaluated_examples={summary['evaluated_examples']}")


if __name__ == "__main__":
    main()
