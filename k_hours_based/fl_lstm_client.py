import argparse
import os
import time
from collections import OrderedDict
from typing import Any

import flwr as fl
import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "PyTorch is required for ai_fl_lstm. Install torch in the project environment before running this mode."
    ) from exc

from next_hour_schema import AIRFLOW_INDEX, CHANGE_BASELINE_COLUMNS, INPUT_COLUMNS, TARGET_COLUMNS, next_hour_change_flags, row_to_input_vector
from fl_shared import load_split_room_df, prefix_eval_metrics, summarize_next_hour_predictions, write_next_hour_prediction_csv


def make_activation(name: str) -> nn.Module:
    return {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "gelu": nn.GELU(),
    }[name]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower room client for federated next-hour LSTM training.")
    parser.add_argument("--split-dir", default="ai/splits_next_hour", help="Directory with next_hour train and test CSV files")
    parser.add_argument("--room-id", required=True, help="Room/client id to run")
    parser.add_argument("--server-address", default="127.0.0.1:8080", help="Flower server address")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per FL round")
    parser.add_argument("--connect-retries", type=int, default=60, help="Connection retries")
    parser.add_argument("--retry-wait", type=float, default=1.0, help="Seconds between retries")
    parser.add_argument("--chunksize", type=int, default=50000, help="CSV chunksize for room filtering")
    parser.add_argument("--sequence-length", type=int, default=4, help="Number of historical rows per LSTM sample")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for local training")
    parser.add_argument("--hidden-dim", type=int, default=64, help="LSTM hidden dimension size")
    parser.add_argument("--num-layers", type=int, default=1, help="Number of stacked LSTM layers")
    parser.add_argument("--head-hidden-dim", type=int, default=64, help="Dense head hidden dimension after the LSTM")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for local training")
    parser.add_argument("--optimizer", choices=["adam", "sgd", "rmsprop"], default="adam", help="Optimizer for local training")
    parser.add_argument("--activation", choices=["relu", "tanh", "gelu"], default="relu", help="Activation function for the dense head")
    parser.add_argument("--predictions-out-dir", default=None, help="Optional directory to write per-room evaluation predictions")
    return parser.parse_args()


def get_input_dim() -> int:
    return len(INPUT_COLUMNS)


class NextHourLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        head_hidden_dim: int = 64,
        activation: str = "relu",
        output_dim: int = len(TARGET_COLUMNS),
    ):
        super().__init__()
        # The LSTM reads a short sequence of past rows and compresses them into
        # one hidden representation, which the head maps to the next-hour targets.
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden_dim),
            make_activation(activation),
            nn.Linear(head_hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.head(last_hidden)


def make_model(
    input_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 1,
    head_hidden_dim: int = 64,
    activation: str = "relu",
) -> NextHourLSTM:
    return NextHourLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        head_hidden_dim=head_hidden_dim,
        activation=activation,
    )


def get_params(model: NextHourLSTM) -> list[np.ndarray]:
    return [tensor.detach().cpu().numpy() for _, tensor in model.state_dict().items()]


def set_params(model: NextHourLSTM, params: list[np.ndarray]) -> None:
    state_dict = model.state_dict()
    if len(params) != len(state_dict):
        raise ValueError(f"Parameter count mismatch: expected {len(state_dict)}, got {len(params)}")
    new_state = OrderedDict()
    for (name, tensor), value in zip(state_dict.items(), params):
        new_state[name] = torch.tensor(value, dtype=tensor.dtype)
    model.load_state_dict(new_state, strict=True)


def load_room_df(split_dir: str, subset: str, room_id: str, chunksize: int) -> pd.DataFrame:
    keep_cols = ["client_id", "t", *CHANGE_BASELINE_COLUMNS, "y_any_change", *INPUT_COLUMNS, *TARGET_COLUMNS]
    return load_split_room_df(split_dir, subset, room_id, keep_cols)


def sanitize_targets(df: pd.DataFrame) -> pd.DataFrame:
    for col in CHANGE_BASELINE_COLUMNS + ["y_any_change"] + INPUT_COLUMNS + TARGET_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["t", *CHANGE_BASELINE_COLUMNS, "y_any_change", *INPUT_COLUMNS, *TARGET_COLUMNS]).copy()
    df["y_airflow"] = df["y_airflow"].clip(0.0, 1.0)
    df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
    return df.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)


def build_sequence_arrays(df: pd.DataFrame, sequence_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_rows = row_to_input_vector(df)
    y_rows = df[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    current_rows = df[CHANGE_BASELINE_COLUMNS].to_numpy(dtype=np.float64)
    change_rows = df["y_any_change"].to_numpy(dtype=np.int64)
    if len(df) < sequence_length:
        empty_seq = np.zeros((0, sequence_length, len(INPUT_COLUMNS)), dtype=np.float32)
        empty_target = np.zeros((0, len(TARGET_COLUMNS)), dtype=np.float32)
        empty_current = np.zeros((0, len(CHANGE_BASELINE_COLUMNS)), dtype=np.float32)
        empty_change = np.zeros((0,), dtype=np.int64)
        return empty_seq, empty_target, empty_current, empty_change

    x_seq = []
    y_seq = []
    current_seq = []
    change_seq = []
    # Turn ordered room rows into overlapping windows:
    # [row1,row2,row3,row4] -> target from row4, then slide one step forward.
    for idx in range(sequence_length - 1, len(df)):
        start_idx = idx - sequence_length + 1
        x_seq.append(x_rows[start_idx : idx + 1])
        y_seq.append(y_rows[idx])
        current_seq.append(current_rows[idx])
        change_seq.append(change_rows[idx])
    return (
        np.asarray(x_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.float32),
        np.asarray(current_seq, dtype=np.float32),
        np.asarray(change_seq, dtype=np.int64),
    )


class RoomLSTMClient(fl.client.NumPyClient):
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
        batch_size: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        head_hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        optimizer: str = "adam",
        activation: str = "relu",
        predictions_out_dir: str | None = None,
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
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.predictions_out_dir = predictions_out_dir
        self.device = torch.device("cpu")
        self.model = make_model(
            input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            head_hidden_dim=head_hidden_dim,
            activation=activation,
        ).to(self.device)

    def get_parameters(self, config: dict[str, Any]):
        return get_params(self.model)

    def _evaluate_arrays(
        self,
        x_eval: np.ndarray,
        y_eval: np.ndarray,
        current_eval: np.ndarray,
        change_eval: np.ndarray,
    ) -> tuple[float, int, dict[str, float | int | str]]:
        self.model.eval()
        if y_eval.size == 0:
            return summarize_next_hour_predictions(
                self.room_id,
                y_eval,
                np.zeros((0, len(TARGET_COLUMNS)), dtype=np.float32),
                change_eval,
                change_eval,
            )
        with torch.no_grad():
            y_pred = self.model(torch.tensor(x_eval, dtype=torch.float32, device=self.device)).cpu().numpy()
        change_pred = next_hour_change_flags(current_eval, y_pred)
        return summarize_next_hour_predictions(
            self.room_id,
            y_eval,
            y_pred,
            change_eval,
            change_pred,
        )

    def fit(self, parameters, config):
        set_params(self.model, parameters)
        if self.y_train.size == 0:
            return get_params(self.model), 0, {"room_id": self.room_id}

        proximal_mu = float(config.get("proximal_mu", 0.0) or 0.0)
        self.model.train()
        dataset = TensorDataset(
            torch.tensor(self.x_train, dtype=torch.float32, device=self.device),
            torch.tensor(self.y_train, dtype=torch.float32, device=self.device),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        optimizer_cls = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
        }[self.optimizer_name]
        optimizer = optimizer_cls(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        train_loss_sum = 0.0
        global_params = [param.detach().clone() for param in self.model.parameters()]

        for _ in range(self.local_epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                # Forward pass: predict the next-hour targets from one sequence window.
                preds = self.model(batch_x)
                # MSE measures how far all predicted target values are from the true ones.
                loss = loss_fn(preds, batch_y)
                if proximal_mu > 0.0:
                    prox_reg = torch.zeros((), device=self.device)
                    for param, global_param in zip(self.model.parameters(), global_params):
                        prox_reg = prox_reg + torch.sum((param - global_param) ** 2)
                    loss = loss + (proximal_mu / 2.0) * prox_reg
                # Backprop computes gradients for every trainable weight in the LSTM and head.
                loss.backward()
                # Adam updates the local room model before the round finishes.
                optimizer.step()
                train_loss_sum += float(loss.item()) * float(batch_y.shape[0])
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
        loss, count, metrics = self._evaluate_arrays(
            self.x_test,
            self.y_test,
            self.current_test,
            self.change_true,
        )

        if self.predictions_out_dir:
            self.model.eval()
            with torch.no_grad():
                y_pred = self.model(torch.tensor(self.x_test, dtype=torch.float32, device=self.device)).cpu().numpy()
            change_pred = next_hour_change_flags(self.current_test, y_pred)
            write_next_hour_prediction_csv(
                self.predictions_out_dir,
                self.room_id,
                self.y_test,
                y_pred,
                self.change_true,
                change_pred,
            )

        return loss, count, metrics


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
    x_train, y_train, current_train, change_train = build_sequence_arrays(train_df, args.sequence_length)
    x_test, y_test, current_test, change_true = build_sequence_arrays(test_df, args.sequence_length)
    if len(y_train) == 0:
        raise RuntimeError(f"Not enough training rows for room_id={room_id} and sequence_length={args.sequence_length}")

    client = RoomLSTMClient(
        room_id=room_id,
        x_train=x_train,
        y_train=y_train,
        current_train=current_train,
        change_train=change_train,
        x_test=x_test,
        y_test=y_test,
        current_test=current_test,
        change_true=change_true,
        input_dim=get_input_dim(),
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        head_hidden_dim=args.head_hidden_dim,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        activation=args.activation,
        predictions_out_dir=os.path.abspath(args.predictions_out_dir) if args.predictions_out_dir else None,
    )

    print("fl_client_lstm.py starting")
    print(f"room_id={room_id}")
    print(f"split_dir={split_dir}")
    print(f"server_address={args.server_address}")
    print(f"sequence_length={args.sequence_length}")
    print(f"hidden_dim={args.hidden_dim}")
    print(f"num_layers={args.num_layers}")
    print(f"head_hidden_dim={args.head_hidden_dim}")
    print(f"learning_rate={args.learning_rate}")
    print(f"optimizer={args.optimizer}")
    print(f"activation={args.activation}")
    print(f"train_sequences={len(y_train)} test_sequences={len(y_test)}")

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
