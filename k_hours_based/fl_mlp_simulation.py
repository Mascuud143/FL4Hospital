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
    from k_hours_based.fl_mlp_server import make_initial_parameters, make_strategy
except ModuleNotFoundError:
    from fl_mlp_client import (
        CHANGE_BASELINE_COLUMNS,
        RoomClient,
        TARGET_COLUMNS,
        get_input_dim,
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
    parser.add_argument("--chunksize", type=int, default=200000, help="CSV chunksize")
    parser.add_argument("--hidden-layers", default="128,64,32", help="Comma-separated MLP hidden layer sizes")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for local training")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for local training")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam", help="Optimizer for local training")
    parser.add_argument("--activation", choices=["relu", "tanh", "logistic"], default="relu", help="Activation function for hidden layers")
    parser.add_argument("--aggregation-method", choices=["fedavg", "fedprox"], default="fedavg", help="Server aggregation strategy")
    parser.add_argument("--proximal-mu", type=float, default=0.0, help="FedProx proximal coefficient")
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


def cleanup_previous_outputs(weights_out_dir: str) -> None:
    os.makedirs(weights_out_dir, exist_ok=True)
    for name in ["total_metrics.csv", "room_metrics.csv", "train_metrics.csv", "latest_global_weights.npz"]:
        path = os.path.join(weights_out_dir, name)
        if os.path.exists(path):
            os.remove(path)
    for name in os.listdir(weights_out_dir):
        if name.startswith("round_") and name.endswith("_global_weights.npz"):
            os.remove(os.path.join(weights_out_dir, name))


def _weighted_average_params(results: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    total_examples = sum(num_examples for _, num_examples in results)
    if total_examples <= 0:
        raise RuntimeError("No training examples were returned by clients.")
    averaged: list[np.ndarray] = []
    for layer_idx in range(len(results[0][0])):
        weighted_sum = sum(params[layer_idx] * num_examples for params, num_examples in results)
        averaged.append(weighted_sum / float(total_examples))
    return averaged


def _run_manual_federated_loop(
    room_ids: list[str],
    room_data: dict[str, tuple],
    *,
    args: argparse.Namespace,
    input_dim: int,
    strategy,
) -> None:
    global_parameters = fl.common.parameters_to_ndarrays(strategy.initial_parameters)
    hidden_layer_sizes = tuple(int(part.strip()) for part in args.hidden_layers.split(",") if part.strip())

    for server_round in range(1, args.rounds + 1):
        print(f"[manual] round={server_round} fit_start")
        fit_results = []
        for rid in room_ids:
            x_train, y_train, x_test, y_test, current_test, change_true = room_data[rid]
            client = RoomClient(
                room_id=rid,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                current_test=current_test,
                change_true=change_true,
                input_dim=input_dim,
                local_epochs=args.local_epochs,
                hidden_layer_sizes=hidden_layer_sizes,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                optimizer=args.optimizer,
                activation=args.activation,
            )
            params, num_examples, metrics = client.fit(
                [np.asarray(p, dtype=np.float64).copy() for p in global_parameters],
                {"proximal_mu": args.proximal_mu if args.aggregation_method == "fedprox" else 0.0},
            )
            fit_results.append((params, num_examples, metrics))

        global_parameters = _weighted_average_params([(params, num_examples) for params, num_examples, _ in fit_results])
        train_examples = sum(num_examples for _, num_examples, _ in fit_results)
        train_loss_sum = sum(float(metrics.get("train_loss_sum", 0.0)) for _, _, metrics in fit_results)
        strategy.latest_fit_summary = {
            "round": int(server_round),
            "trained_examples": int(train_examples),
            "train_loss": train_loss_sum / max(float(train_examples), 1.0),
        }
        from k_hours_based.fl_mlp_server import upsert_rows, save_parameters
        upsert_rows(os.path.join(strategy.weights_out_dir, "train_metrics.csv"), [strategy.latest_fit_summary], ["round"])
        round_weights_path = os.path.join(strategy.weights_out_dir, f"round_{server_round}_global_weights.npz")
        latest_weights_path = os.path.join(strategy.weights_out_dir, "latest_global_weights.npz")
        save_parameters(fl.common.ndarrays_to_parameters(global_parameters), round_weights_path)
        save_parameters(fl.common.ndarrays_to_parameters(global_parameters), latest_weights_path)
        print(f"saved_global_weights={round_weights_path}")
        print(
            f"[round {server_round}] training_summary "
            f"train_loss={strategy.latest_fit_summary['train_loss']:.4f} "
            f"trained_examples={strategy.latest_fit_summary['trained_examples']}"
        )

        print(f"[manual] round={server_round} evaluate_start")
        eval_results = []
        for rid in room_ids:
            x_train, y_train, x_test, y_test, current_test, change_true = room_data[rid]
            client = RoomClient(
                room_id=rid,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                current_test=current_test,
                change_true=change_true,
                input_dim=input_dim,
                local_epochs=args.local_epochs,
                hidden_layer_sizes=hidden_layer_sizes,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                optimizer=args.optimizer,
                activation=args.activation,
            )
            loss, num_examples, metrics = client.evaluate(
                [np.asarray(p, dtype=np.float64).copy() for p in global_parameters],
                {},
            )
            eval_results.append((loss, num_examples, metrics))

        summary = {
            "evaluated_examples": 0.0,
            "regression_correct": 0.0,
            "regression_wrong": 0.0,
            "temperature_correct": 0.0,
            "temperature_wrong": 0.0,
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
            "change_accuracy_sum": 0.0,
            "change_precision_sum": 0.0,
            "change_recall_sum": 0.0,
            "change_f1_sum": 0.0,
            "change_correct": 0.0,
            "change_incorrect": 0.0,
            "change_tp": 0.0,
            "change_tn": 0.0,
            "change_fp": 0.0,
            "change_fn": 0.0,
            "mae_sum_y_temp_main": 0.0,
            "mae_sum_y_temp_toilet": 0.0,
            "mae_sum_y_light": 0.0,
            "mae_sum_y_sound": 0.0,
            "mse_sum_y_temp_main": 0.0,
            "mse_sum_y_temp_toilet": 0.0,
            "mse_sum_y_light": 0.0,
            "mse_sum_y_sound": 0.0,
            "threshold_correct_y_temp_main": 0.0,
            "threshold_wrong_y_temp_main": 0.0,
            "threshold_correct_y_temp_toilet": 0.0,
            "threshold_wrong_y_temp_toilet": 0.0,
            "threshold_correct_y_light": 0.0,
            "threshold_wrong_y_light": 0.0,
            "threshold_correct_y_sound": 0.0,
            "threshold_wrong_y_sound": 0.0,
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
                    "mae_y_temp_main": float(metrics.get("mae_sum_y_temp_main", 0.0)) / max(count, 1.0),
                    "mse_y_temp_main": float(metrics.get("mse_sum_y_temp_main", 0.0)) / max(count, 1.0),
                    "rmse_y_temp_main": (float(metrics.get("mse_sum_y_temp_main", 0.0)) / max(count, 1.0)) ** 0.5,
                    "threshold_accuracy_y_temp_main": float(metrics.get("threshold_correct_y_temp_main", 0.0)) / max(float(metrics.get("threshold_correct_y_temp_main", 0.0)) + float(metrics.get("threshold_wrong_y_temp_main", 0.0)), 1.0),
                    "mae_y_temp_toilet": float(metrics.get("mae_sum_y_temp_toilet", 0.0)) / max(count, 1.0),
                    "mse_y_temp_toilet": float(metrics.get("mse_sum_y_temp_toilet", 0.0)) / max(count, 1.0),
                    "rmse_y_temp_toilet": (float(metrics.get("mse_sum_y_temp_toilet", 0.0)) / max(count, 1.0)) ** 0.5,
                    "threshold_accuracy_y_temp_toilet": float(metrics.get("threshold_correct_y_temp_toilet", 0.0)) / max(float(metrics.get("threshold_correct_y_temp_toilet", 0.0)) + float(metrics.get("threshold_wrong_y_temp_toilet", 0.0)), 1.0),
                    "mae_y_light": float(metrics.get("mae_sum_y_light", 0.0)) / max(count, 1.0),
                    "mse_y_light": float(metrics.get("mse_sum_y_light", 0.0)) / max(count, 1.0),
                    "rmse_y_light": (float(metrics.get("mse_sum_y_light", 0.0)) / max(count, 1.0)) ** 0.5,
                    "threshold_accuracy_y_light": float(metrics.get("threshold_correct_y_light", 0.0)) / max(float(metrics.get("threshold_correct_y_light", 0.0)) + float(metrics.get("threshold_wrong_y_light", 0.0)), 1.0),
                    "mae_y_sound": float(metrics.get("mae_sum_y_sound", 0.0)) / max(count, 1.0),
                    "mse_y_sound": float(metrics.get("mse_sum_y_sound", 0.0)) / max(count, 1.0),
                    "rmse_y_sound": (float(metrics.get("mse_sum_y_sound", 0.0)) / max(count, 1.0)) ** 0.5,
                    "threshold_accuracy_y_sound": float(metrics.get("threshold_correct_y_sound", 0.0)) / max(float(metrics.get("threshold_correct_y_sound", 0.0)) + float(metrics.get("threshold_wrong_y_sound", 0.0)), 1.0),
                    "airflow_accuracy": float(metrics.get("airflow_accuracy_sum", 0.0)) / max(count, 1.0),
                    "airflow_precision": float(metrics.get("airflow_precision_sum", 0.0)) / max(count, 1.0),
                    "airflow_recall": float(metrics.get("airflow_recall_sum", 0.0)) / max(count, 1.0),
                    "airflow_f1": float(metrics.get("airflow_f1_sum", 0.0)) / max(count, 1.0),
                    "airflow_tp": int(metrics.get("airflow_tp", 0)),
                    "airflow_fp": int(metrics.get("airflow_fp", 0)),
                    "airflow_tn": int(metrics.get("airflow_tn", 0)),
                    "airflow_fn": int(metrics.get("airflow_fn", 0)),
                    "change_accuracy": float(metrics.get("change_accuracy_sum", 0.0)) / max(count, 1.0),
                    "change_precision": float(metrics.get("change_precision_sum", 0.0)) / max(count, 1.0),
                    "change_recall": float(metrics.get("change_recall_sum", 0.0)) / max(count, 1.0),
                    "change_f1": float(metrics.get("change_f1_sum", 0.0)) / max(count, 1.0),
                    "change_tp": int(metrics.get("change_tp", 0)),
                    "change_fp": int(metrics.get("change_fp", 0)),
                    "change_tn": int(metrics.get("change_tn", 0)),
                    "change_fn": int(metrics.get("change_fn", 0)),
                }
            )

        count = max(summary["evaluated_examples"], 1.0)
        strategy.latest_eval_summary = {
            "round": int(server_round),
            "global_loss": loss_weighted_sum / count,
            "mae_y_temp_main": summary["mae_sum_y_temp_main"] / count,
            "mse_y_temp_main": summary["mse_sum_y_temp_main"] / count,
            "mae_y_temp_toilet": summary["mae_sum_y_temp_toilet"] / count,
            "mse_y_temp_toilet": summary["mse_sum_y_temp_toilet"] / count,
            "mae_y_light": summary["mae_sum_y_light"] / count,
            "mse_y_light": summary["mse_sum_y_light"] / count,
            "mae_y_sound": summary["mae_sum_y_sound"] / count,
            "mse_y_sound": summary["mse_sum_y_sound"] / count,
            "rmse_y_temp_main": (summary["mse_sum_y_temp_main"] / count) ** 0.5,
            "rmse_y_temp_toilet": (summary["mse_sum_y_temp_toilet"] / count) ** 0.5,
            "rmse_y_light": (summary["mse_sum_y_light"] / count) ** 0.5,
            "rmse_y_sound": (summary["mse_sum_y_sound"] / count) ** 0.5,
            "threshold_accuracy_y_temp_main": summary["threshold_correct_y_temp_main"] / max(summary["threshold_correct_y_temp_main"] + summary["threshold_wrong_y_temp_main"], 1.0),
            "threshold_accuracy_y_temp_toilet": summary["threshold_correct_y_temp_toilet"] / max(summary["threshold_correct_y_temp_toilet"] + summary["threshold_wrong_y_temp_toilet"], 1.0),
            "threshold_accuracy_y_light": summary["threshold_correct_y_light"] / max(summary["threshold_correct_y_light"] + summary["threshold_wrong_y_light"], 1.0),
            "threshold_accuracy_y_sound": summary["threshold_correct_y_sound"] / max(summary["threshold_correct_y_sound"] + summary["threshold_wrong_y_sound"], 1.0),
            "regression_correct": int(summary["regression_correct"]),
            "regression_wrong": int(summary["regression_wrong"]),
            "regression_correct_rate": summary["regression_correct"] / max(summary["regression_correct"] + summary["regression_wrong"], 1.0),
            "temperature_correct": int(summary["temperature_correct"]),
            "temperature_wrong": int(summary["temperature_wrong"]),
            "temperature_correct_rate": summary["temperature_correct"] / max(summary["temperature_correct"] + summary["temperature_wrong"], 1.0),
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
            "change_accuracy": summary["change_accuracy_sum"] / count,
            "change_precision": summary["change_precision_sum"] / count,
            "change_recall": summary["change_recall_sum"] / count,
            "change_f1": summary["change_f1_sum"] / count,
            "change_correct": int(summary["change_correct"]),
            "change_incorrect": int(summary["change_incorrect"]),
            "change_tp": int(summary["change_tp"]),
            "change_tn": int(summary["change_tn"]),
            "change_fp": int(summary["change_fp"]),
            "change_fn": int(summary["change_fn"]),
            "evaluated_examples": int(summary["evaluated_examples"]),
        }
        upsert_rows(os.path.join(strategy.weights_out_dir, "total_metrics.csv"), [strategy.latest_eval_summary], ["round"])
        upsert_rows(os.path.join(strategy.weights_out_dir, "room_metrics.csv"), room_rows, ["round", "room_id"])
        print(f"saved_total_metrics={os.path.join(strategy.weights_out_dir, 'total_metrics.csv')}")
        print(
            f"[round {server_round}] evaluation_summary "
            f"regression_correct_rate={strategy.latest_eval_summary['regression_correct_rate']:.4f} "
            f"temperature_correct_rate={strategy.latest_eval_summary['temperature_correct_rate']:.4f} "
            f"airflow_f1={strategy.latest_eval_summary['airflow_f1']:.4f} "
            f"change_f1={strategy.latest_eval_summary['change_f1']:.4f} "
            f"mae_y_temp_main={strategy.latest_eval_summary['mae_y_temp_main']:.4f} "
            f"evaluated_examples={strategy.latest_eval_summary['evaluated_examples']}"
        )


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
    print(f"weights_out_dir={weights_out_dir}")
    print("[start] simulation running...")
    history = None
    use_manual_fallback = os.name == "nt"
    if not use_manual_fallback:
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=len(room_ids),
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
            client_resources={"num_cpus": args.client_cpu},
        )
    else:
        _run_manual_federated_loop(
            room_ids,
            room_data,
            args=args,
            input_dim=input_dim,
            strategy=strategy,
        )

    print("[done] simulation finished")
    if history is not None and history.losses_distributed:
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
