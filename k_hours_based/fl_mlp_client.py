import argparse
import os
import time
from typing import Any

import flwr as fl
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score
from sklearn.neural_network import MLPRegressor

from next_hour_schema import AIRFLOW_INDEX, CHANGE_BASELINE_COLUMNS, INPUT_COLUMNS, TARGET_COLUMNS, next_hour_change_flags, row_to_input_vector

DEFAULT_OUTPUT_THRESHOLDS = {
    "y_temp_main": 1.0,
    "y_temp_toilet": 1.0,
    "y_light": 5.0,
    "y_sound": 5.0,
}
MODEL_HIDDEN_LAYER_SIZES = (128, 64, 32)
DEFAULT_BATCH_SIZE = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower room client for federated next-hour environment rows.")
    parser.add_argument("--split-dir", default="ai/splits_next_hour", help="Directory with next_hour train and test CSV files")
    parser.add_argument("--room-id", required=True, help="Room/client id to run")
    parser.add_argument("--server-address", default="127.0.0.1:8080", help="Flower server address")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per FL round")
    parser.add_argument("--connect-retries", type=int, default=60, help="Connection retries")
    parser.add_argument("--retry-wait", type=float, default=1.0, help="Seconds between retries")
    parser.add_argument("--chunksize", type=int, default=200000, help="CSV chunksize for room filtering")
    parser.add_argument("--hidden-layers", default="128,64,32", help="Comma-separated MLP hidden layer sizes")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for local training")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for local training")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam", help="Optimizer for local training")
    parser.add_argument("--activation", choices=["relu", "tanh", "logistic"], default="relu", help="Activation function for hidden layers")
    return parser.parse_args()


def get_input_dim() -> int:
    return len(INPUT_COLUMNS)


def parse_hidden_layers(spec: str) -> tuple[int, ...]:
    values = [part.strip() for part in str(spec).split(",")]
    layers = tuple(int(part) for part in values if part)
    if not layers or any(size <= 0 for size in layers):
        raise ValueError(f"Invalid hidden layer specification: {spec}")
    return layers


def make_model(
    input_dim: int,
    hidden_layer_sizes: tuple[int, ...] = MODEL_HIDDEN_LAYER_SIZES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = 1e-3,
    optimizer: str = "adam",
    activation: str = "relu",
) -> MLPRegressor:
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=optimizer,
        random_state=42,
        learning_rate_init=learning_rate,
        batch_size=batch_size,
        max_iter=1,
        shuffle=False,
    )
    # sklearn's MLPRegressor creates weight arrays only after an initial fit.
    # We bootstrap it with one dummy sample so Flower can exchange parameters.
    x0 = np.zeros((1, input_dim), dtype=np.float64)
    y0 = np.zeros((1, len(TARGET_COLUMNS)), dtype=np.float64)
    model.partial_fit(x0, y0)
    return model


def get_params(model: MLPRegressor) -> list[np.ndarray]:
    return [np.asarray(layer, dtype=np.float64) for layer in [*model.coefs_, *model.intercepts_]]


def set_params(model: MLPRegressor, params: list[np.ndarray]) -> None:
    split_idx = len(model.coefs_)
    model.coefs_ = [np.asarray(layer, dtype=np.float64).copy() for layer in params[:split_idx]]
    model.intercepts_ = [np.asarray(layer, dtype=np.float64).reshape(-1).copy() for layer in params[split_idx:]]


def load_room_df(path: str, room_id: str, chunksize: int) -> pd.DataFrame:
    keep_cols = ["client_id", *CHANGE_BASELINE_COLUMNS, "y_any_change", *INPUT_COLUMNS, *TARGET_COLUMNS]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=lambda c: c in keep_cols, chunksize=chunksize):
        room_chunk = chunk[chunk["client_id"].astype(str) == room_id]
        if not room_chunk.empty:
            chunks.append(room_chunk)
    if not chunks:
        return pd.DataFrame(columns=keep_cols)
    df = pd.concat(chunks, ignore_index=True)
    return df.loc[:, ~df.columns.duplicated()]


def sanitize_targets(df: pd.DataFrame) -> pd.DataFrame:
    # Older generated splits may not include newer optional input columns.
    # For backward compatibility we materialize missing feature columns and
    # default them to zero before building the numeric matrix.
    required_columns = CHANGE_BASELINE_COLUMNS + ["y_any_change"] + INPUT_COLUMNS + TARGET_COLUMNS
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0.0 if col in INPUT_COLUMNS else np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[INPUT_COLUMNS] = df[INPUT_COLUMNS].fillna(0.0)
    df = df.dropna(subset=CHANGE_BASELINE_COLUMNS + ["y_any_change"] + TARGET_COLUMNS).copy()
    df["y_airflow"] = df["y_airflow"].clip(0.0, 1.0)
    return df


def target_correct_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int]:
    thresholds = np.array(
        [
            DEFAULT_OUTPUT_THRESHOLDS["y_temp_main"],
            DEFAULT_OUTPUT_THRESHOLDS["y_temp_toilet"],
            DEFAULT_OUTPUT_THRESHOLDS["y_light"],
            DEFAULT_OUTPUT_THRESHOLDS["y_sound"],
        ],
        dtype=np.float64,
    )
    regression_true = y_true[:, :AIRFLOW_INDEX]
    regression_pred = y_pred[:, :AIRFLOW_INDEX]
    correct = int(np.sum(np.all(np.abs(regression_true - regression_pred) <= thresholds, axis=1)))
    return correct, int(y_true.shape[0] - correct)


def per_target_threshold_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, tuple[int, int]]:
    counts: dict[str, tuple[int, int]] = {}
    for idx, target in enumerate(TARGET_COLUMNS[:AIRFLOW_INDEX]):
        threshold = DEFAULT_OUTPUT_THRESHOLDS[target]
        within_threshold = np.abs(y_true[:, idx] - y_pred[:, idx]) <= threshold
        correct = int(np.sum(within_threshold))
        counts[target] = (correct, int(y_true.shape[0] - correct))
    return counts


def temperature_correct_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int]:
    thresholds = np.array(
        [
            DEFAULT_OUTPUT_THRESHOLDS["y_temp_main"],
        ],
        dtype=np.float64,
    )
    temp_true = y_true[:, :2]
    temp_pred = y_pred[:, :2]
    correct = int(np.sum(np.all(np.abs(temp_true - temp_pred) <= thresholds, axis=1)))
    return correct, int(y_true.shape[0] - correct)


class RoomClient(fl.client.NumPyClient):
    def __init__(
        self,
        room_id: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        current_test: np.ndarray,
        change_true: np.ndarray,
        input_dim: int,
        local_epochs: int,
        hidden_layer_sizes: tuple[int, ...] = MODEL_HIDDEN_LAYER_SIZES,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = 1e-3,
        optimizer: str = "adam",
        activation: str = "relu",
    ):
        self.room_id = room_id
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.current_test = current_test
        self.change_true = change_true
        self.local_epochs = local_epochs
        self.model = make_model(
            input_dim,
            hidden_layer_sizes=hidden_layer_sizes,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer=optimizer,
            activation=activation,
        )

    def get_parameters(self, config: dict[str, Any]):
        return get_params(self.model)

    def fit(self, parameters, config):
        # Each round starts from the current global model sent by the server.
        set_params(self.model, parameters)
        initial_parameters = [np.asarray(param, dtype=np.float64).copy() for param in parameters]
        proximal_mu = float(config.get("proximal_mu", 0.0) or 0.0)
        train_loss_sum = 0.0
        for _ in range(self.local_epochs):
            # partial_fit runs one local optimization pass over this room's rows.
            self.model.partial_fit(self.x_train, self.y_train)
            if proximal_mu > 0.0:
                current_parameters = get_params(self.model)
                blended_parameters = [
                    (current + proximal_mu * initial) / (1.0 + proximal_mu)
                    for current, initial in zip(current_parameters, initial_parameters)
                ]
                set_params(self.model, blended_parameters)
            train_pred = self.model.predict(self.x_train)
            train_loss_sum += float(mean_squared_error(self.y_train, train_pred)) * float(self.y_train.shape[0])
        return get_params(self.model), int(self.y_train.shape[0]), {
            "room_id": self.room_id,
            "train_loss_sum": train_loss_sum,
        }

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        if self.y_test.size == 0:
            return 0.0, 0, {
                "mae_sum_y_temp_main": 0.0,
                "mae_sum_y_temp_toilet": 0.0,
                "mae_sum_y_light": 0.0,
                "mae_sum_y_sound": 0.0,
                "mse_sum_y_temp_main": 0.0,
                "mse_sum_y_temp_toilet": 0.0,
                "mse_sum_y_light": 0.0,
                "mse_sum_y_sound": 0.0,
                "regression_correct": 0,
                "regression_wrong": 0,
                "threshold_correct_y_temp_main": 0,
                "threshold_wrong_y_temp_main": 0,
                "threshold_correct_y_temp_toilet": 0,
                "threshold_wrong_y_temp_toilet": 0,
                "threshold_correct_y_light": 0,
                "threshold_wrong_y_light": 0,
                "threshold_correct_y_sound": 0,
                "threshold_wrong_y_sound": 0,
                "temperature_correct": 0,
                "temperature_wrong": 0,
                "airflow_accuracy_sum": 0.0,
                "airflow_precision_sum": 0.0,
                "airflow_recall_sum": 0.0,
                "airflow_f1_sum": 0.0,
                "airflow_correct": 0,
                "airflow_incorrect": 0,
                "airflow_tp": 0,
                "airflow_tn": 0,
                "airflow_fp": 0,
                "airflow_fn": 0,
                "change_accuracy_sum": 0.0,
                "change_precision_sum": 0.0,
                "change_recall_sum": 0.0,
                "change_f1_sum": 0.0,
                "change_correct": 0,
                "change_incorrect": 0,
                "change_tp": 0,
                "change_tn": 0,
                "change_fp": 0,
                "change_fn": 0,
                "room_id": self.room_id,
            }

        y_pred = self.model.predict(self.x_test)
        # The network predicts all next-hour targets at once, then we score
        # regression accuracy, airflow classification, and change detection.
        reg_true = self.y_test[:, :AIRFLOW_INDEX]
        reg_pred = y_pred[:, :AIRFLOW_INDEX]
        airflow_true = self.y_test[:, AIRFLOW_INDEX].round().astype(int)
        airflow_pred = (np.clip(y_pred[:, AIRFLOW_INDEX], 0.0, 1.0) >= 0.5).astype(int)

        regression_mae = {
            target: float(mean_absolute_error(reg_true[:, idx], reg_pred[:, idx]))
            for idx, target in enumerate(TARGET_COLUMNS[:AIRFLOW_INDEX])
        }
        regression_mse = {
            target: float(mean_squared_error(reg_true[:, idx], reg_pred[:, idx]))
            for idx, target in enumerate(TARGET_COLUMNS[:AIRFLOW_INDEX])
        }
        regression_correct, regression_wrong = target_correct_counts(self.y_test, y_pred)
        threshold_counts = per_target_threshold_counts(reg_true, reg_pred)
        temperature_correct, temperature_wrong = temperature_correct_counts(self.y_test, y_pred)

        tp = int(np.sum((airflow_true == 1) & (airflow_pred == 1)))
        tn = int(np.sum((airflow_true == 0) & (airflow_pred == 0)))
        fp = int(np.sum((airflow_true == 0) & (airflow_pred == 1)))
        fn = int(np.sum((airflow_true == 1) & (airflow_pred == 0)))
        airflow_accuracy = float(accuracy_score(airflow_true, airflow_pred))
        airflow_precision = float(precision_score(airflow_true, airflow_pred, zero_division=0))
        airflow_recall = float(recall_score(airflow_true, airflow_pred, zero_division=0))
        airflow_f1 = float(f1_score(airflow_true, airflow_pred, zero_division=0))
        change_pred = next_hour_change_flags(self.current_test, y_pred)
        change_accuracy = float(accuracy_score(self.change_true, change_pred))
        change_precision = float(precision_score(self.change_true, change_pred, zero_division=0))
        change_recall = float(recall_score(self.change_true, change_pred, zero_division=0))
        change_f1 = float(f1_score(self.change_true, change_pred, zero_division=0))
        change_tp = int(np.sum((self.change_true == 1) & (change_pred == 1)))
        change_tn = int(np.sum((self.change_true == 0) & (change_pred == 0)))
        change_fp = int(np.sum((self.change_true == 0) & (change_pred == 1)))
        change_fn = int(np.sum((self.change_true == 1) & (change_pred == 0)))

        overall_loss = float(np.mean(list(regression_mae.values())))
        return overall_loss, int(self.y_test.shape[0]), {
            "mae_sum_y_temp_main": regression_mae["y_temp_main"] * self.y_test.shape[0],
            "mae_sum_y_temp_toilet": regression_mae["y_temp_toilet"] * self.y_test.shape[0],
            "mae_sum_y_light": regression_mae["y_light"] * self.y_test.shape[0],
            "mae_sum_y_sound": regression_mae["y_sound"] * self.y_test.shape[0],
            "mse_sum_y_temp_main": regression_mse["y_temp_main"] * self.y_test.shape[0],
            "mse_sum_y_temp_toilet": regression_mse["y_temp_toilet"] * self.y_test.shape[0],
            "mse_sum_y_light": regression_mse["y_light"] * self.y_test.shape[0],
            "mse_sum_y_sound": regression_mse["y_sound"] * self.y_test.shape[0],
            "regression_correct": regression_correct,
            "regression_wrong": regression_wrong,
            "threshold_correct_y_temp_main": threshold_counts["y_temp_main"][0],
            "threshold_wrong_y_temp_main": threshold_counts["y_temp_main"][1],
            "threshold_correct_y_temp_toilet": threshold_counts["y_temp_toilet"][0],
            "threshold_wrong_y_temp_toilet": threshold_counts["y_temp_toilet"][1],
            "threshold_correct_y_light": threshold_counts["y_light"][0],
            "threshold_wrong_y_light": threshold_counts["y_light"][1],
            "threshold_correct_y_sound": threshold_counts["y_sound"][0],
            "threshold_wrong_y_sound": threshold_counts["y_sound"][1],
            "temperature_correct": temperature_correct,
            "temperature_wrong": temperature_wrong,
            "airflow_accuracy_sum": airflow_accuracy * self.y_test.shape[0],
            "airflow_precision_sum": airflow_precision * self.y_test.shape[0],
            "airflow_recall_sum": airflow_recall * self.y_test.shape[0],
            "airflow_f1_sum": airflow_f1 * self.y_test.shape[0],
            "airflow_correct": int(np.sum(airflow_true == airflow_pred)),
            "airflow_incorrect": int(self.y_test.shape[0] - np.sum(airflow_true == airflow_pred)),
            "airflow_tp": tp,
            "airflow_tn": tn,
            "airflow_fp": fp,
            "airflow_fn": fn,
            "change_accuracy_sum": change_accuracy * self.y_test.shape[0],
            "change_precision_sum": change_precision * self.y_test.shape[0],
            "change_recall_sum": change_recall * self.y_test.shape[0],
            "change_f1_sum": change_f1 * self.y_test.shape[0],
            "change_correct": int(np.sum(self.change_true == change_pred)),
            "change_incorrect": int(self.y_test.shape[0] - np.sum(self.change_true == change_pred)),
            "change_tp": change_tp,
            "change_tn": change_tn,
            "change_fp": change_fp,
            "change_fn": change_fn,
            "room_id": self.room_id,
        }


def main() -> None:
    args = parse_args()
    room_id = str(args.room_id)
    split_dir = os.path.abspath(args.split_dir)
    train_path = os.path.join(split_dir, "next_hour_train.csv")
    test_path = os.path.join(split_dir, "next_hour_test.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing file: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing file: {test_path}")

    train_df = sanitize_targets(load_room_df(train_path, room_id=room_id, chunksize=args.chunksize))
    test_df = sanitize_targets(load_room_df(test_path, room_id=room_id, chunksize=args.chunksize))
    if len(train_df) == 0:
        raise RuntimeError(f"No training rows for room_id={room_id}")

    input_dim = get_input_dim()
    hidden_layer_sizes = parse_hidden_layers(args.hidden_layers)
    x_train = row_to_input_vector(train_df)
    y_train = train_df[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    x_test = row_to_input_vector(test_df)
    y_test = test_df[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    current_test = test_df[CHANGE_BASELINE_COLUMNS].to_numpy(dtype=np.float64)
    change_true = test_df["y_any_change"].to_numpy(dtype=np.int64)

    client = RoomClient(
        room_id=room_id,
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

    print("fl_client.py starting")
    print(f"room_id={room_id}")
    print(f"split_dir={split_dir}")
    print(f"server_address={args.server_address}")
    print(f"hidden_layers={','.join(str(size) for size in hidden_layer_sizes)}")
    print(f"batch_size={args.batch_size}")
    print(f"learning_rate={args.learning_rate}")
    print(f"optimizer={args.optimizer}")
    print(f"activation={args.activation}")
    print(f"train_rows={len(y_train)} test_rows={len(y_test)}")

    attempts = 0
    while attempts < args.connect_retries:
        attempts += 1
        try:
            fl.client.start_numpy_client(server_address=args.server_address, client=client)
            print(f"client_completed room_id={room_id}")
            return
        except Exception:
            if attempts >= args.connect_retries:
                raise
            time.sleep(args.retry_wait)


if __name__ == "__main__":
    main()
