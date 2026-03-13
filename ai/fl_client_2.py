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
        "PyTorch is required for ai_fl_2. Install torch in the project environment before running this mode."
    ) from exc

from next_hour_schema import INPUT_COLUMNS, TARGET_COLUMNS, row_to_input_vector

REGRESSION_TARGETS = ["y_temp_main", "y_temp_toilet", "y_light", "y_sound"]
CLASSIFICATION_TARGETS = ["y_airflow", "y_any_change"]
MODEL_OUTPUTS = [*REGRESSION_TARGETS, *CLASSIFICATION_TARGETS]
AIRFLOW_OUTPUT_INDEX = 4
CHANGE_OUTPUT_INDEX = 5
DEFAULT_OUTPUT_THRESHOLDS = {
    "y_temp_main": 1.0,
    "y_temp_toilet": 1.0,
    "y_light": 5.0,
    "y_sound": 5.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower room client for federated next-hour mixed-loss MLP training.")
    parser.add_argument("--split-dir", default="ai/splits_next_hour", help="Directory with next_hour train and test CSV files")
    parser.add_argument("--room-id", required=True, help="Room/client id to run")
    parser.add_argument("--server-address", default="127.0.0.1:8080", help="Flower server address")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per FL round")
    parser.add_argument("--connect-retries", type=int, default=60, help="Connection retries")
    parser.add_argument("--retry-wait", type=float, default=1.0, help="Seconds between retries")
    parser.add_argument("--chunksize", type=int, default=200000, help="CSV chunksize for room filtering")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for local training")
    return parser.parse_args()


def get_input_dim() -> int:
    return len(INPUT_COLUMNS)


class NextHourMLP2(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = len(MODEL_OUTPUTS)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_model(input_dim: int) -> NextHourMLP2:
    return NextHourMLP2(input_dim=input_dim)


def get_params(model: NextHourMLP2) -> list[np.ndarray]:
    return [tensor.detach().cpu().numpy() for _, tensor in model.state_dict().items()]


def set_params(model: NextHourMLP2, params: list[np.ndarray]) -> None:
    state_dict = model.state_dict()
    if len(params) != len(state_dict):
        raise ValueError(f"Parameter count mismatch: expected {len(state_dict)}, got {len(params)}")
    new_state = OrderedDict()
    for (name, tensor), value in zip(state_dict.items(), params):
        new_state[name] = torch.tensor(value, dtype=tensor.dtype)
    model.load_state_dict(new_state, strict=True)


def load_room_df(path: str, room_id: str, chunksize: int) -> pd.DataFrame:
    keep_cols = ["client_id", "t", *INPUT_COLUMNS, *TARGET_COLUMNS, "y_any_change"]
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
    for col in [*INPUT_COLUMNS, *TARGET_COLUMNS, "y_any_change"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[*INPUT_COLUMNS, *TARGET_COLUMNS, "y_any_change"]).copy()
    df["y_airflow"] = df["y_airflow"].clip(0.0, 1.0)
    df["y_any_change"] = df["y_any_change"].clip(0.0, 1.0)
    return df


def build_target_matrix(df: pd.DataFrame) -> np.ndarray:
    target_df = pd.DataFrame(index=df.index)
    for target in REGRESSION_TARGETS:
        target_df[target] = df[target].astype(float)
    target_df["y_airflow"] = df["y_airflow"].astype(float)
    target_df["y_any_change"] = df["y_any_change"].astype(float)
    return target_df[MODEL_OUTPUTS].to_numpy(dtype=np.float32)


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
    regression_true = y_true[:, :4]
    regression_pred = y_pred[:, :4]
    correct = int(np.sum(np.all(np.abs(regression_true - regression_pred) <= thresholds, axis=1)))
    return correct, int(y_true.shape[0] - correct)


def _pos_weight_from_targets(values: np.ndarray) -> torch.Tensor:
    positives = float(np.sum(values >= 0.5))
    negatives = float(len(values) - positives)
    if positives <= 0.0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(max(negatives / positives, 1.0), dtype=torch.float32)


class RoomClient2(fl.client.NumPyClient):
    def __init__(
        self,
        room_id: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        input_dim: int,
        local_epochs: int,
        batch_size: int,
    ):
        self.room_id = room_id
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.device = torch.device("cpu")
        self.model = make_model(input_dim).to(self.device)
        self.airflow_pos_weight = _pos_weight_from_targets(self.y_train[:, AIRFLOW_OUTPUT_INDEX]).to(self.device)
        self.change_pos_weight = _pos_weight_from_targets(self.y_train[:, CHANGE_OUTPUT_INDEX]).to(self.device)

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
        loader = DataLoader(dataset, batch_size=max(1, min(self.batch_size, len(dataset))), shuffle=False)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        reg_loss_fn = nn.MSELoss()
        airflow_loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.airflow_pos_weight)
        change_loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.change_pos_weight)

        for _ in range(self.local_epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                preds = self.model(batch_x)
                reg_loss = reg_loss_fn(preds[:, :4], batch_y[:, :4])
                airflow_loss = airflow_loss_fn(preds[:, AIRFLOW_OUTPUT_INDEX], batch_y[:, AIRFLOW_OUTPUT_INDEX])
                change_loss = change_loss_fn(preds[:, CHANGE_OUTPUT_INDEX], batch_y[:, CHANGE_OUTPUT_INDEX])
                loss = reg_loss + airflow_loss + change_loss
                loss.backward()
                optimizer.step()
        return get_params(self.model), int(self.y_train.shape[0]), {"room_id": self.room_id}

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        if self.y_test.size == 0:
            return 0.0, 0, {
                "room_id": self.room_id,
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
            }

        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(self.x_test, dtype=torch.float32, device=self.device)).cpu().numpy()

        reg_pred = logits[:, :4]
        reg_true = self.y_test[:, :4]
        airflow_true = self.y_test[:, AIRFLOW_OUTPUT_INDEX].round().astype(int)
        airflow_prob = 1.0 / (1.0 + np.exp(-logits[:, AIRFLOW_OUTPUT_INDEX]))
        airflow_pred = (airflow_prob >= 0.5).astype(int)
        change_true = self.y_test[:, CHANGE_OUTPUT_INDEX].round().astype(int)
        change_prob = 1.0 / (1.0 + np.exp(-logits[:, CHANGE_OUTPUT_INDEX]))
        change_pred = (change_prob >= 0.5).astype(int)

        regression_mae = {
            target: float(mean_absolute_error(reg_true[:, idx], reg_pred[:, idx]))
            for idx, target in enumerate(REGRESSION_TARGETS)
        }
        regression_mse = {
            target: float(mean_squared_error(reg_true[:, idx], reg_pred[:, idx]))
            for idx, target in enumerate(REGRESSION_TARGETS)
        }
        regression_correct, regression_wrong = target_correct_counts(self.y_test, logits)

        airflow_tp = int(np.sum((airflow_true == 1) & (airflow_pred == 1)))
        airflow_tn = int(np.sum((airflow_true == 0) & (airflow_pred == 0)))
        airflow_fp = int(np.sum((airflow_true == 0) & (airflow_pred == 1)))
        airflow_fn = int(np.sum((airflow_true == 1) & (airflow_pred == 0)))
        change_tp = int(np.sum((change_true == 1) & (change_pred == 1)))
        change_tn = int(np.sum((change_true == 0) & (change_pred == 0)))
        change_fp = int(np.sum((change_true == 0) & (change_pred == 1)))
        change_fn = int(np.sum((change_true == 1) & (change_pred == 0)))

        overall_loss = float(np.mean(list(regression_mae.values())))
        return overall_loss, int(self.y_test.shape[0]), {
            "room_id": self.room_id,
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
            "airflow_accuracy_sum": float(accuracy_score(airflow_true, airflow_pred)) * self.y_test.shape[0],
            "airflow_precision_sum": float(precision_score(airflow_true, airflow_pred, zero_division=0)) * self.y_test.shape[0],
            "airflow_recall_sum": float(recall_score(airflow_true, airflow_pred, zero_division=0)) * self.y_test.shape[0],
            "airflow_f1_sum": float(f1_score(airflow_true, airflow_pred, zero_division=0)) * self.y_test.shape[0],
            "airflow_correct": int(np.sum(airflow_true == airflow_pred)),
            "airflow_incorrect": int(self.y_test.shape[0] - np.sum(airflow_true == airflow_pred)),
            "airflow_tp": airflow_tp,
            "airflow_tn": airflow_tn,
            "airflow_fp": airflow_fp,
            "airflow_fn": airflow_fn,
            "change_accuracy_sum": float(accuracy_score(change_true, change_pred)) * self.y_test.shape[0],
            "change_precision_sum": float(precision_score(change_true, change_pred, zero_division=0)) * self.y_test.shape[0],
            "change_recall_sum": float(recall_score(change_true, change_pred, zero_division=0)) * self.y_test.shape[0],
            "change_f1_sum": float(f1_score(change_true, change_pred, zero_division=0)) * self.y_test.shape[0],
            "change_correct": int(np.sum(change_true == change_pred)),
            "change_incorrect": int(self.y_test.shape[0] - np.sum(change_true == change_pred)),
            "change_tp": change_tp,
            "change_tn": change_tn,
            "change_fp": change_fp,
            "change_fn": change_fn,
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

    x_train = row_to_input_vector(train_df).astype(np.float32)
    y_train = build_target_matrix(train_df)
    x_test = row_to_input_vector(test_df).astype(np.float32)
    y_test = build_target_matrix(test_df)

    client = RoomClient2(
        room_id=room_id,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        input_dim=get_input_dim(),
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
    )

    print("fl_client_2.py starting")
    print(f"room_id={room_id}")
    print(f"split_dir={split_dir}")
    print(f"server_address={args.server_address}")
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
