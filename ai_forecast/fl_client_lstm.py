import argparse
import os
import time
from collections import OrderedDict
from typing import Any

import flwr as fl
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "PyTorch is required for ai_fl_lstm. Install torch in the project environment before running this mode."
    ) from exc

from next_hour_schema import AIRFLOW_INDEX, CHANGE_BASELINE_COLUMNS, INPUT_COLUMNS, TARGET_COLUMNS, next_hour_change_flags, row_to_input_vector

DEFAULT_OUTPUT_THRESHOLDS = {
    "y_temp_main": 1.0,
    "y_temp_toilet": 1.0,
    "y_light": 5.0,
    "y_sound": 5.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower room client for federated next-hour LSTM training.")
    parser.add_argument("--split-dir", default="ai/splits_next_hour", help="Directory with next_hour train and test CSV files")
    parser.add_argument("--room-id", required=True, help="Room/client id to run")
    parser.add_argument("--server-address", default="127.0.0.1:8080", help="Flower server address")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per FL round")
    parser.add_argument("--connect-retries", type=int, default=60, help="Connection retries")
    parser.add_argument("--retry-wait", type=float, default=1.0, help="Seconds between retries")
    parser.add_argument("--chunksize", type=int, default=200000, help="CSV chunksize for room filtering")
    parser.add_argument("--sequence-length", type=int, default=4, help="Number of historical rows per LSTM sample")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for local training")
    return parser.parse_args()


def get_input_dim() -> int:
    return len(INPUT_COLUMNS)


class NextHourLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, output_dim: int = len(TARGET_COLUMNS)):
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
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.head(last_hidden)


def make_model(input_dim: int) -> NextHourLSTM:
    return NextHourLSTM(input_dim=input_dim)


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


def load_room_df(path: str, room_id: str, chunksize: int) -> pd.DataFrame:
    keep_cols = ["client_id", "t", *CHANGE_BASELINE_COLUMNS, "y_any_change", *INPUT_COLUMNS, *TARGET_COLUMNS]
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


class RoomLSTMClient(fl.client.NumPyClient):
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
        batch_size: int,
    ):
        self.room_id = room_id
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.current_test = current_test
        self.change_true = change_true
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.device = torch.device("cpu")
        self.model = make_model(input_dim).to(self.device)

    def get_parameters(self, config: dict[str, Any]):
        return get_params(self.model)

    def fit(self, parameters, config):
        set_params(self.model, parameters)
        if self.y_train.size == 0:
            return get_params(self.model), 0, {"room_id": self.room_id}

        self.model.train()
        dataset = TensorDataset(
            torch.tensor(self.x_train, dtype=torch.float32, device=self.device),
            torch.tensor(self.y_train, dtype=torch.float32, device=self.device),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        for _ in range(self.local_epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                # Forward pass: predict the next-hour targets from one sequence window.
                preds = self.model(batch_x)
                # MSE measures how far all predicted target values are from the true ones.
                loss = loss_fn(preds, batch_y)
                # Backprop computes gradients for every trainable weight in the LSTM and head.
                loss.backward()
                # Adam updates the local room model before the round finishes.
                optimizer.step()
        return get_params(self.model), int(self.y_train.shape[0]), {"room_id": self.room_id}

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

        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(torch.tensor(self.x_test, dtype=torch.float32, device=self.device)).cpu().numpy()

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
    x_train, y_train, _, _ = build_sequence_arrays(train_df, args.sequence_length)
    x_test, y_test, current_test, change_true = build_sequence_arrays(test_df, args.sequence_length)
    if len(y_train) == 0:
        raise RuntimeError(f"Not enough training rows for room_id={room_id} and sequence_length={args.sequence_length}")

    client = RoomLSTMClient(
        room_id=room_id,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        current_test=current_test,
        change_true=change_true,
        input_dim=get_input_dim(),
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
    )

    print("fl_client_lstm.py starting")
    print(f"room_id={room_id}")
    print(f"split_dir={split_dir}")
    print(f"server_address={args.server_address}")
    print(f"sequence_length={args.sequence_length}")
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
