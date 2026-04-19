import argparse
import os
import time
from typing import Any

import flwr as fl
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

from next_hour_schema import CHANGE_BASELINE_COLUMNS, INPUT_COLUMNS, TARGET_COLUMNS, next_hour_change_flags, row_to_input_vector
from fl_shared import load_split_room_df, prefix_eval_metrics, summarize_next_hour_predictions
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
    parser.add_argument("--chunksize", type=int, default=50000, help="CSV chunksize for room filtering")
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


def load_room_df(split_dir: str, subset: str, room_id: str, chunksize: int) -> pd.DataFrame:
    keep_cols = ["client_id", *CHANGE_BASELINE_COLUMNS, "y_any_change", *INPUT_COLUMNS, *TARGET_COLUMNS]
    return load_split_room_df(split_dir, subset, room_id, keep_cols)


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


class RoomClient(fl.client.NumPyClient):
    def __init__(
        self,
        room_id: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        current_train: np.ndarray,
        change_train: np.ndarray,
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
        self.current_train = current_train
        self.change_train = change_train
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

    def _evaluate_arrays(
        self,
        x_eval: np.ndarray,
        y_eval: np.ndarray,
        current_eval: np.ndarray,
        change_eval: np.ndarray,
    ) -> tuple[float, int, dict[str, float | int | str]]:
        y_pred = self.model.predict(x_eval)
        change_pred = next_hour_change_flags(current_eval, y_pred)
        return summarize_next_hour_predictions(
            self.room_id,
            y_eval,
            y_pred,
            change_eval,
            change_pred,
            include_temperature=True,
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
        train_eval_loss, train_eval_examples, train_eval_metrics = self._evaluate_arrays(
            self.x_train,
            self.y_train,
            self.current_train,
            self.change_train,
        )
        test_local_loss, test_local_examples, test_local_metrics = self._evaluate_arrays(
            self.x_test,
            self.y_test,
            self.current_test,
            self.change_true,
        )
        fit_metrics: dict[str, float | int | str] = {
            "room_id": self.room_id,
            "train_loss_sum": train_loss_sum,
        }
        fit_metrics.update(prefix_eval_metrics("train_local", train_eval_loss, train_eval_examples, train_eval_metrics))
        fit_metrics.update(prefix_eval_metrics("test_local", test_local_loss, test_local_examples, test_local_metrics))
        return get_params(self.model), int(self.y_train.shape[0]), fit_metrics

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        return self._evaluate_arrays(self.x_test, self.y_test, self.current_test, self.change_true)


def main() -> None:
    args = parse_args()
    room_id = str(args.room_id)
    split_dir = os.path.abspath(args.split_dir)
    train_dir = os.path.join(split_dir, "train")
    test_dir = os.path.join(split_dir, "test")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Missing directory: {train_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Missing directory: {test_dir}")

    train_df = sanitize_targets(load_room_df(split_dir, "train", room_id=room_id, chunksize=args.chunksize))
    test_df = sanitize_targets(load_room_df(split_dir, "test", room_id=room_id, chunksize=args.chunksize))
    if len(train_df) == 0:
        raise RuntimeError(f"No training rows for room_id={room_id}")

    input_dim = get_input_dim()
    hidden_layer_sizes = parse_hidden_layers(args.hidden_layers)
    x_train = row_to_input_vector(train_df)
    y_train = train_df[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    x_test = row_to_input_vector(test_df)
    y_test = test_df[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    current_train = train_df[CHANGE_BASELINE_COLUMNS].to_numpy(dtype=np.float64)
    change_train = train_df["y_any_change"].to_numpy(dtype=np.int64)
    current_test = test_df[CHANGE_BASELINE_COLUMNS].to_numpy(dtype=np.float64)
    change_true = test_df["y_any_change"].to_numpy(dtype=np.int64)

    client = RoomClient(
        room_id=room_id,
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
