import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import flwr as fl
import numpy as np
import pandas as pd
from per_room_data import list_room_ids

_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in os.sys.path:
    os.sys.path.insert(0, _PARENT_DIR)

from fl_client import FEATURE_COLUMNS, RoomClient, TARGET_COLUMNS, build_target_matrix, get_input_dim, load_room_df, row_to_input_vector, sanitize_rows
from fl_server import TrackingFedAvg, TrackingFedProx, make_initial_parameters, save_parameters, upsert_rows

MANUAL_MAX_CONCURRENT_ROOMS = 10
MANUAL_PROGRESS_EVERY = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower simulation mode for Task #2 state-to-outcome training.")
    parser.add_argument("--split-dir", default="ai_state_to_outcome/splits", help="Directory with state_to_outcome train and test CSV files")
    parser.add_argument("--rounds", type=int, default=5, help="Federated rounds")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per round")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for local training")
    parser.add_argument("--hidden-layers", default="128,64,32", help="Comma-separated hidden layer sizes")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for local training")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam", help="Optimizer for local training")
    parser.add_argument("--activation", choices=["relu", "tanh", "logistic"], default="relu", help="Activation function for hidden layers")
    parser.add_argument("--fraction-fit", type=float, default=1.0, help="Fraction of clients sampled for fit")
    parser.add_argument("--fraction-evaluate", type=float, default=1.0, help="Fraction of clients sampled for evaluate")
    parser.add_argument("--aggregation-method", choices=["fedavg", "fedprox"], default="fedavg", help="Server aggregation strategy")
    parser.add_argument("--proximal-mu", type=float, default=0.0, help="FedProx proximal coefficient")
    parser.add_argument("--min-fit-clients", type=int, default=2, help="Minimum clients for fit")
    parser.add_argument("--min-evaluate-clients", type=int, default=2, help="Minimum clients for evaluate")
    parser.add_argument("--min-available-clients", type=int, default=2, help="Minimum connected clients")
    parser.add_argument("--max-rooms", type=int, default=None, help="Optional cap on number of rooms")
    parser.add_argument("--weights-out-dir", default="ai_state_to_outcome/fl_weights", help="Directory to write global weights per round")
    parser.add_argument("--summary-out", default=None, help="Optional JSON path to write final evaluation summary")
    parser.add_argument("--client-cpu", type=float, default=1.0, help="CPU resources per simulated client")
    parser.add_argument("--chunksize", type=int, default=50000, help="CSV chunksize")
    parser.add_argument("--prebuild-workers", type=int, default=0, help="Room CSV prebuild workers; 0 derives from available workers * 2")
    parser.add_argument("--server-address", default="127.0.0.1:8096", help="Used for threaded fallback mode")
    parser.add_argument("--server-start-wait", type=float, default=2.0, help="Seconds to wait before starting fallback clients")
    parser.add_argument("--client-retries", type=int, default=60, help="Retries for fallback client connections")
    parser.add_argument("--retry-wait", type=float, default=1.0, help="Seconds between fallback client retries")
    return parser.parse_args()


def _available_client_workers(client_cpu: float) -> int:
    cpu_count = os.cpu_count() or 1
    client_cpu = max(float(client_cpu), 0.001)
    return max(1, int(cpu_count // client_cpu))


def load_room_dataset(split_dir: str, rid: str, chunksize: int) -> tuple:
    room_train_df = sanitize_rows(load_room_df(split_dir, "train", room_id=rid, chunksize=chunksize))
    if len(room_train_df) == 0:
        raise RuntimeError(f"No training rows for room_id={rid}")
    room_test_df = sanitize_rows(load_room_df(split_dir, "test", room_id=rid, chunksize=chunksize))
    x_train = row_to_input_vector(room_train_df).astype("float32")
    y_train = build_target_matrix(room_train_df)
    x_test = row_to_input_vector(room_test_df).astype("float32")
    y_test = build_target_matrix(room_test_df)
    return x_train, y_train, x_test, y_test


def _weighted_average_params(results: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    total_examples = sum(num_examples for _, num_examples in results)
    if total_examples <= 0:
        raise RuntimeError("No training examples were returned by rooms.")
    averaged: list[np.ndarray] = []
    for layer_idx in range(len(results[0][0])):
        weighted_sum = sum(params[layer_idx] * num_examples for params, num_examples in results)
        averaged.append(weighted_sum / float(total_examples))
    return averaged


def _run_manual_federated_loop(
    room_ids: list[str],
    *,
    args: argparse.Namespace,
    input_dim: int,
    strategy,
    load_dataset,
) -> None:
    global_parameters = fl.common.parameters_to_ndarrays(strategy.initial_parameters)
    total_rooms = len(room_ids)

    def _fit_one_room(rid: str):
        x_train, y_train, x_test, y_test = load_dataset(rid)
        client = RoomClient(
            room_id=rid,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            input_dim=input_dim,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            hidden_layers=args.hidden_layers,
            learning_rate=args.learning_rate,
            optimizer_name=args.optimizer,
            activation=args.activation,
        )
        params, num_examples, metrics = client.fit(
            [np.asarray(p, dtype=np.float32).copy() for p in global_parameters],
            {"proximal_mu": args.proximal_mu if args.aggregation_method == "fedprox" else 0.0},
        )
        return rid, params, num_examples, metrics

    def _evaluate_one_room(rid: str):
        x_train, y_train, x_test, y_test = load_dataset(rid)
        client = RoomClient(
            room_id=rid,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            input_dim=input_dim,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            hidden_layers=args.hidden_layers,
            learning_rate=args.learning_rate,
            optimizer_name=args.optimizer,
            activation=args.activation,
        )
        loss, num_examples, metrics = client.evaluate(
            [np.asarray(p, dtype=np.float32).copy() for p in global_parameters],
            {},
        )
        return rid, loss, num_examples, metrics

    for server_round in range(1, args.rounds + 1):
        print(f"[manual] round_progress={server_round}/{args.rounds} phase=starting rooms={total_rooms}")
        print(f"[manual] round={server_round} fit_start")
        fit_results = []
        with ThreadPoolExecutor(max_workers=min(MANUAL_MAX_CONCURRENT_ROOMS, max(1, total_rooms))) as executor:
            future_map = {executor.submit(_fit_one_room, rid): rid for rid in room_ids}
            completed_fit = 0
            for future in as_completed(future_map):
                rid, params, num_examples, metrics = future.result()
                fit_results.append((params, num_examples, metrics))
                completed_fit += 1
                if completed_fit % MANUAL_PROGRESS_EVERY == 0 or completed_fit == total_rooms:
                    print(
                        f"[manual] round={server_round} phase=fit rooms_completed={completed_fit}/{total_rooms}",
                        flush=True,
                    )

        global_parameters = _weighted_average_params([(params, num_examples) for params, num_examples, _ in fit_results])
        round_weights_path = os.path.join(strategy.weights_out_dir, f"round_{server_round}_global_weights.npz")
        latest_weights_path = os.path.join(strategy.weights_out_dir, "latest_global_weights.npz")
        save_parameters(fl.common.ndarrays_to_parameters(global_parameters), round_weights_path)
        save_parameters(fl.common.ndarrays_to_parameters(global_parameters), latest_weights_path)
        print(f"saved_global_weights={round_weights_path}")

        print(f"[manual] round={server_round} evaluate_start")
        eval_results = []
        with ThreadPoolExecutor(max_workers=min(MANUAL_MAX_CONCURRENT_ROOMS, max(1, total_rooms))) as executor:
            future_map = {executor.submit(_evaluate_one_room, rid): rid for rid in room_ids}
            completed_eval = 0
            for future in as_completed(future_map):
                rid, loss, num_examples, metrics = future.result()
                eval_results.append((loss, num_examples, metrics))
                completed_eval += 1
                if completed_eval % MANUAL_PROGRESS_EVERY == 0 or completed_eval == total_rooms:
                    print(
                        f"[manual] round={server_round} phase=evaluate rooms_completed={completed_eval}/{total_rooms}",
                        flush=True,
                    )

        summary = {
            "evaluated_examples": 0.0,
            "airflow_correct": 0.0,
            "airflow_incorrect": 0.0,
            "airflow_tp": 0.0,
            "airflow_tn": 0.0,
            "airflow_fp": 0.0,
            "airflow_fn": 0.0,
            "airflow_accuracy_sum": 0.0,
            "airflow_precision_sum": 0.0,
            "airflow_recall_sum": 0.0,
            "airflow_f1_sum": 0.0,
            "mae_sum_y_target_temp_main": 0.0,
            "mae_sum_y_target_temp_toilet": 0.0,
            "mae_sum_y_target_light": 0.0,
            "mae_sum_y_target_sound": 0.0,
            "mse_sum_y_target_temp_main": 0.0,
            "mse_sum_y_target_temp_toilet": 0.0,
            "mse_sum_y_target_light": 0.0,
            "mse_sum_y_target_sound": 0.0,
            "threshold_correct_y_target_temp_main": 0.0,
            "threshold_wrong_y_target_temp_main": 0.0,
            "threshold_correct_y_target_temp_toilet": 0.0,
            "threshold_wrong_y_target_temp_toilet": 0.0,
            "threshold_correct_y_target_light": 0.0,
            "threshold_wrong_y_target_light": 0.0,
            "threshold_correct_y_target_sound": 0.0,
            "threshold_wrong_y_target_sound": 0.0,
        }
        room_rows: list[dict[str, float | int | str]] = []
        loss_weighted_sum = 0.0
        for loss, num_examples, metrics in eval_results:
            count = float(num_examples)
            summary["evaluated_examples"] += count
            loss_weighted_sum += float(loss) * count
            for key in summary:
                if key == "evaluated_examples":
                    continue
                summary[key] += float(metrics.get(key, 0.0))
            room_rows.append(
                {
                    "round": int(server_round),
                    "room_id": str(metrics.get("room_id", "")),
                    "num_examples": int(num_examples),
                    "local_loss": float(loss),
                    "mae_y_target_temp_main": float(metrics.get("mae_sum_y_target_temp_main", 0.0)) / max(count, 1.0),
                    "mae_y_target_temp_toilet": float(metrics.get("mae_sum_y_target_temp_toilet", 0.0)) / max(count, 1.0),
                    "mae_y_target_light": float(metrics.get("mae_sum_y_target_light", 0.0)) / max(count, 1.0),
                    "mae_y_target_sound": float(metrics.get("mae_sum_y_target_sound", 0.0)) / max(count, 1.0),
                    "rmse_y_target_temp_main": (float(metrics.get("mse_sum_y_target_temp_main", 0.0)) / max(count, 1.0)) ** 0.5,
                    "rmse_y_target_temp_toilet": (float(metrics.get("mse_sum_y_target_temp_toilet", 0.0)) / max(count, 1.0)) ** 0.5,
                    "rmse_y_target_light": (float(metrics.get("mse_sum_y_target_light", 0.0)) / max(count, 1.0)) ** 0.5,
                    "rmse_y_target_sound": (float(metrics.get("mse_sum_y_target_sound", 0.0)) / max(count, 1.0)) ** 0.5,
                    "threshold_accuracy_y_temp_main": float(metrics.get("threshold_correct_y_target_temp_main", 0.0)) / max(float(metrics.get("threshold_correct_y_target_temp_main", 0.0)) + float(metrics.get("threshold_wrong_y_target_temp_main", 0.0)), 1.0),
                    "threshold_accuracy_y_temp_toilet": float(metrics.get("threshold_correct_y_target_temp_toilet", 0.0)) / max(float(metrics.get("threshold_correct_y_target_temp_toilet", 0.0)) + float(metrics.get("threshold_wrong_y_target_temp_toilet", 0.0)), 1.0),
                    "threshold_accuracy_y_light": float(metrics.get("threshold_correct_y_target_light", 0.0)) / max(float(metrics.get("threshold_correct_y_target_light", 0.0)) + float(metrics.get("threshold_wrong_y_target_light", 0.0)), 1.0),
                    "threshold_accuracy_y_sound": float(metrics.get("threshold_correct_y_target_sound", 0.0)) / max(float(metrics.get("threshold_correct_y_target_sound", 0.0)) + float(metrics.get("threshold_wrong_y_target_sound", 0.0)), 1.0),
                    "airflow_accuracy": float(metrics.get("airflow_accuracy_sum", 0.0)) / max(count, 1.0),
                    "airflow_precision": float(metrics.get("airflow_precision_sum", 0.0)) / max(count, 1.0),
                    "airflow_recall": float(metrics.get("airflow_recall_sum", 0.0)) / max(count, 1.0),
                    "airflow_f1": float(metrics.get("airflow_f1_sum", 0.0)) / max(count, 1.0),
                    "airflow_tp": int(metrics.get("airflow_tp", 0)),
                    "airflow_tn": int(metrics.get("airflow_tn", 0)),
                    "airflow_fp": int(metrics.get("airflow_fp", 0)),
                    "airflow_fn": int(metrics.get("airflow_fn", 0)),
                }
            )

        count = max(summary["evaluated_examples"], 1.0)
        strategy.latest_eval_summary = {
            "round": int(server_round),
            "global_loss": loss_weighted_sum / count,
            "mae_y_target_temp_main": summary["mae_sum_y_target_temp_main"] / count,
            "mae_y_target_temp_toilet": summary["mae_sum_y_target_temp_toilet"] / count,
            "mae_y_target_light": summary["mae_sum_y_target_light"] / count,
            "mae_y_target_sound": summary["mae_sum_y_target_sound"] / count,
            "rmse_y_target_temp_main": (summary["mse_sum_y_target_temp_main"] / count) ** 0.5,
            "rmse_y_target_temp_toilet": (summary["mse_sum_y_target_temp_toilet"] / count) ** 0.5,
            "rmse_y_target_light": (summary["mse_sum_y_target_light"] / count) ** 0.5,
            "rmse_y_target_sound": (summary["mse_sum_y_target_sound"] / count) ** 0.5,
            "threshold_correct_y_temp_main": int(summary["threshold_correct_y_target_temp_main"]),
            "threshold_wrong_y_temp_main": int(summary["threshold_wrong_y_target_temp_main"]),
            "threshold_correct_y_temp_toilet": int(summary["threshold_correct_y_target_temp_toilet"]),
            "threshold_wrong_y_temp_toilet": int(summary["threshold_wrong_y_target_temp_toilet"]),
            "threshold_correct_y_light": int(summary["threshold_correct_y_target_light"]),
            "threshold_wrong_y_light": int(summary["threshold_wrong_y_target_light"]),
            "threshold_correct_y_sound": int(summary["threshold_correct_y_target_sound"]),
            "threshold_wrong_y_sound": int(summary["threshold_wrong_y_target_sound"]),
            "threshold_accuracy_y_temp_main": summary["threshold_correct_y_target_temp_main"] / max(summary["threshold_correct_y_target_temp_main"] + summary["threshold_wrong_y_target_temp_main"], 1.0),
            "threshold_accuracy_y_temp_toilet": summary["threshold_correct_y_target_temp_toilet"] / max(summary["threshold_correct_y_target_temp_toilet"] + summary["threshold_wrong_y_target_temp_toilet"], 1.0),
            "threshold_accuracy_y_light": summary["threshold_correct_y_target_light"] / max(summary["threshold_correct_y_target_light"] + summary["threshold_wrong_y_target_light"], 1.0),
            "threshold_accuracy_y_sound": summary["threshold_correct_y_target_sound"] / max(summary["threshold_correct_y_target_sound"] + summary["threshold_wrong_y_target_sound"], 1.0),
            "airflow_accuracy": summary["airflow_accuracy_sum"] / count,
            "airflow_precision": summary["airflow_precision_sum"] / count,
            "airflow_recall": summary["airflow_recall_sum"] / count,
            "airflow_f1": summary["airflow_f1_sum"] / count,
            "airflow_correct": int(summary["airflow_correct"]),
            "airflow_incorrect": int(summary["airflow_incorrect"]),
            "airflow_tp": int(summary["airflow_tp"]),
            "airflow_tn": int(summary["airflow_tn"]),
            "airflow_fp": int(summary["airflow_fp"]),
            "airflow_fn": int(summary["airflow_fn"]),
            "evaluated_examples": int(summary["evaluated_examples"]),
        }
        total_path = os.path.join(strategy.weights_out_dir, f"round_{server_round}_total_metrics.csv")
        room_path = os.path.join(strategy.weights_out_dir, f"round_{server_round}_room_metrics.csv")
        os.makedirs(strategy.weights_out_dir, exist_ok=True)
        pd.DataFrame([strategy.latest_eval_summary]).to_csv(total_path, index=False)
        pd.DataFrame(room_rows).to_csv(room_path, index=False)
        upsert_rows(os.path.join(strategy.weights_out_dir, "room_metrics.csv"), room_rows, ["round", "room_id"])
        print(f"saved_total_metrics={total_path}")
        print(
            f"[round {server_round}] evaluation_summary "
            f"airflow_f1={strategy.latest_eval_summary['airflow_f1']:.4f} "
            f"mae_y_target_temp_main={strategy.latest_eval_summary['mae_y_target_temp_main']:.4f} "
            f"evaluated_examples={strategy.latest_eval_summary['evaluated_examples']}"
        )


def main() -> None:
    args = parse_args()
    split_dir = os.path.abspath(args.split_dir)
    train_dir = os.path.join(split_dir, "train")
    test_dir = os.path.join(split_dir, "test")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Missing directory: {train_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Missing directory: {test_dir}")

    train_room_id_set = set(list_room_ids(split_dir, "train"))
    test_room_id_set = set(list_room_ids(split_dir, "test"))

    room_ids = sorted(train_room_id_set, key=lambda x: int(x) if x.isdigit() else x)
    room_ids = [rid for rid in room_ids if rid in train_room_id_set and rid in test_room_id_set]
    if not room_ids:
        raise RuntimeError("No rooms with both training and test rows were found.")
    if args.max_rooms is not None:
        room_ids = room_ids[: max(1, args.max_rooms)]
    strategy_cls = TrackingFedProx if args.aggregation_method == "fedprox" else TrackingFedAvg
    strategy_kwargs = {
        "weights_out_dir": os.path.abspath(args.weights_out_dir),
        "fraction_fit": args.fraction_fit,
        "fraction_evaluate": args.fraction_evaluate,
        "min_fit_clients": min(args.min_fit_clients, len(room_ids)),
        "min_evaluate_clients": min(args.min_evaluate_clients, len(room_ids)),
        "min_available_clients": min(args.min_available_clients, len(room_ids)),
        "initial_parameters": make_initial_parameters(hidden_layers=args.hidden_layers, activation=args.activation),
    }
    if args.aggregation_method == "fedprox":
        strategy_kwargs["proximal_mu"] = args.proximal_mu
    strategy = strategy_cls(**strategy_kwargs)
    input_dim = get_input_dim()

    print("fl_simulation.py starting")
    print(f"split_dir={split_dir}")
    print(f"rooms_simulated={len(room_ids)}")
    print(f"weights_out_dir={os.path.abspath(args.weights_out_dir)}")
    print("[start] simulation running...")

    def client_fn(cid: str):
        rid = room_ids[int(cid)]
        x_train, y_train, x_test, y_test = load_room_dataset(split_dir, rid, args.chunksize)
        client = RoomClient(
            room_id=rid,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            input_dim=input_dim,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            hidden_layers=args.hidden_layers,
            learning_rate=args.learning_rate,
            optimizer_name=args.optimizer,
            activation=args.activation,
        )
        if args.aggregation_method == "fedprox":
            return client.to_client()
        return client.to_client()

    try:
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=len(room_ids),
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
            client_resources={"num_cpus": args.client_cpu},
        )
    except ImportError as exc:
        print(f"[fallback] flower simulation backend unavailable: {exc}")
        _run_manual_federated_loop(
            room_ids,
            args=args,
            input_dim=input_dim,
            strategy=strategy,
            load_dataset=lambda rid: load_room_dataset(split_dir, rid, args.chunksize),
        )

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

    print("[done] simulation finished")


if __name__ == "__main__":
    main()
