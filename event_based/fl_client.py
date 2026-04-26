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
        "PyTorch is required for event_based federated learning. Install torch before running this mode."
    ) from exc

from schema import FEATURE_COLUMNS, REGRESSION_TARGET_COLUMNS, TARGET_COLUMNS, row_to_input_vector
from per_room_data import load_room_df as load_room_df_from_split

AIRFLOW_OUTPUT_INDEX = 4
DEFAULT_OUTPUT_THRESHOLDS = {
    "y_target_temp_main": 1.0,
    "y_target_temp_toilet": 1.0,
    "y_target_light": 5.0,
    "y_target_sound": 5.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower room client for Task #2 state-to-outcome training.")
    parser.add_argument("--split-dir", default="event_based/splits", help="Directory with event_based train/test CSV files")
    parser.add_argument("--room-id", required=True, help="Room/client id to run")
    parser.add_argument("--server-address", default="127.0.0.1:8080", help="Flower server address")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per FL round")
    parser.add_argument("--connect-retries", type=int, default=60, help="Connection retries")
    parser.add_argument("--retry-wait", type=float, default=1.0, help="Seconds between retries")
    parser.add_argument("--chunksize", type=int, default=50000, help="CSV chunksize for room filtering")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for local training")
    parser.add_argument("--hidden-layers", default="128,64,32", help="Comma-separated hidden layer sizes")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for local training")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam", help="Optimizer for local training")
    parser.add_argument("--activation", choices=["relu", "tanh", "logistic"], default="relu", help="Activation function for hidden layers")
    return parser.parse_args()


def get_input_dim() -> int:
    return len(FEATURE_COLUMNS)


def parse_hidden_layers(raw: str) -> list[int]:
    layers: list[int] = []
    for part in str(raw).split(","):
        value = part.strip()
        if not value:
            continue
        size = int(value)
        if size <= 0:
            raise ValueError("Hidden layer sizes must be positive integers")
        layers.append(size)
    if not layers:
        raise ValueError("At least one hidden layer size is required")
    return layers


def _activation_module(name: str) -> nn.Module:
    normalized = str(name).strip().lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "logistic":
        return nn.Sigmoid()
    raise ValueError(f"Unsupported activation: {name}")


class StateOutcomeMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list[int], activation: str, output_dim: int = len(TARGET_COLUMNS)):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(_activation_module(activation))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_model(input_dim: int, hidden_layers: str = "128,64,32", activation: str = "relu") -> StateOutcomeMLP:
    return StateOutcomeMLP(input_dim=input_dim, hidden_layers=parse_hidden_layers(hidden_layers), activation=activation)


def make_optimizer(model: nn.Module, optimizer_name: str, learning_rate: float) -> torch.optim.Optimizer:
    normalized = str(optimizer_name).strip().lower()
    if normalized == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    if normalized == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_params(model: StateOutcomeMLP) -> list[np.ndarray]:
    return [tensor.detach().cpu().numpy() for _, tensor in model.state_dict().items()]


def set_params(model: StateOutcomeMLP, params: list[np.ndarray]) -> None:
    state_dict = model.state_dict()
    if len(params) != len(state_dict):
        raise ValueError(f"Parameter count mismatch: expected {len(state_dict)}, got {len(params)}")
    new_state = OrderedDict()
    for (name, tensor), value in zip(state_dict.items(), params):
        new_state[name] = torch.tensor(value, dtype=tensor.dtype)
    model.load_state_dict(new_state, strict=True)


def load_room_df(split_dir: str, subset: str, room_id: str, chunksize: int) -> pd.DataFrame:
    keep_cols = ["room_id", *FEATURE_COLUMNS, *TARGET_COLUMNS, "event_type"]
    df = load_room_df_from_split(split_dir, subset, room_id, usecols=keep_cols)
    return df.loc[:, ~df.columns.duplicated()]


def sanitize_rows(df: pd.DataFrame) -> pd.DataFrame:
    # The federated client expects numeric features and complete target labels.
    for col in [*FEATURE_COLUMNS, *TARGET_COLUMNS]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[*TARGET_COLUMNS]).copy()
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    df["y_target_airflow"] = df["y_target_airflow"].clip(0.0, 1.0)
    return df


def build_target_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[TARGET_COLUMNS].to_numpy(dtype=np.float32)


def _pos_weight_from_targets(values: np.ndarray) -> torch.Tensor:
    positives = float(np.sum(values >= 0.5))
    negatives = float(len(values) - positives)
    if positives <= 0.0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(max(negatives / positives, 1.0), dtype=torch.float32)


def target_threshold_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, tuple[int, int]]:
    counts: dict[str, tuple[int, int]] = {}
    for idx, target in enumerate(REGRESSION_TARGET_COLUMNS):
        threshold = DEFAULT_OUTPUT_THRESHOLDS[target]
        correct = int(np.sum(np.abs(y_true[:, idx] - y_pred[:, idx]) <= threshold))
        wrong = int(y_true.shape[0] - correct)
        counts[target] = (correct, wrong)
    return counts


class RoomClient(fl.client.NumPyClient):
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
        hidden_layers: str,
        learning_rate: float,
        optimizer_name: str,
        activation: str,
    ):
        self.room_id = room_id
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.device = torch.device("cpu")
        self.model = make_model(input_dim, hidden_layers=hidden_layers, activation=activation).to(self.device)
        self.airflow_pos_weight = _pos_weight_from_targets(self.y_train[:, AIRFLOW_OUTPUT_INDEX]).to(self.device)

    def get_parameters(self, config: dict[str, Any]):
        return get_params(self.model)

    def fit(self, parameters, config):
        set_params(self.model, parameters)
        if self.y_train.size == 0:
            return get_params(self.model), 0, {"room_id": self.room_id}

        self.model.train()
        proximal_mu = float(config.get("proximal_mu", 0.0) or 0.0)
        global_params = [param.detach().clone() for param in self.model.parameters()]
        dataset = TensorDataset(
            torch.tensor(self.x_train, dtype=torch.float32, device=self.device),
            torch.tensor(self.y_train, dtype=torch.float32, device=self.device),
        )
        loader = DataLoader(dataset, batch_size=max(1, min(self.batch_size, len(dataset))), shuffle=False)
        optimizer = make_optimizer(self.model, self.optimizer_name, self.learning_rate)
        reg_loss_fn = nn.MSELoss()
        airflow_loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.airflow_pos_weight)

        for _ in range(self.local_epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                # One forward pass predicts all outcome targets for this event batch.
                preds = self.model(batch_x)
                # Continuous comfort targets use MSE, while airflow is trained as
                # a weighted binary classification problem.
                reg_loss = reg_loss_fn(preds[:, :4], batch_y[:, :4])
                airflow_loss = airflow_loss_fn(preds[:, AIRFLOW_OUTPUT_INDEX], batch_y[:, AIRFLOW_OUTPUT_INDEX])
                loss = reg_loss + airflow_loss
                if proximal_mu > 0.0:
                    prox_reg = torch.zeros((), device=self.device)
                    for param, global_param in zip(self.model.parameters(), global_params):
                        prox_reg = prox_reg + torch.sum((param - global_param) ** 2)
                    loss = loss + (proximal_mu / 2.0) * prox_reg
                loss.backward()
                optimizer.step()
        return get_params(self.model), int(self.y_train.shape[0]), {"room_id": self.room_id}

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        if self.y_test.size == 0:
            return 0.0, 0, {
                "room_id": self.room_id,
                "mae_sum_y_target_temp_main": 0.0,
                "mae_sum_y_target_temp_toilet": 0.0,
                "mae_sum_y_target_light": 0.0,
                "mae_sum_y_target_sound": 0.0,
                "mse_sum_y_target_temp_main": 0.0,
                "mse_sum_y_target_temp_toilet": 0.0,
                "mse_sum_y_target_light": 0.0,
                "mse_sum_y_target_sound": 0.0,
                "threshold_correct_y_target_temp_main": 0,
                "threshold_wrong_y_target_temp_main": 0,
                "threshold_correct_y_target_temp_toilet": 0,
                "threshold_wrong_y_target_temp_toilet": 0,
                "threshold_correct_y_target_light": 0,
                "threshold_wrong_y_target_light": 0,
                "threshold_correct_y_target_sound": 0,
                "threshold_wrong_y_target_sound": 0,
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
            }

        self.model.eval()
        with torch.no_grad():
            # The network returns raw airflow logits, so we convert them to
            # probabilities with the sigmoid formula before thresholding at 0.5.
            logits = self.model(torch.tensor(self.x_test, dtype=torch.float32, device=self.device)).cpu().numpy()

        reg_pred = logits[:, :4]
        reg_true = self.y_test[:, :4]
        airflow_true = self.y_test[:, AIRFLOW_OUTPUT_INDEX].round().astype(int)
        airflow_prob = 1.0 / (1.0 + np.exp(-logits[:, AIRFLOW_OUTPUT_INDEX]))
        airflow_pred = (airflow_prob >= 0.5).astype(int)

        regression_mae = {
            target: float(mean_absolute_error(reg_true[:, idx], reg_pred[:, idx]))
            for idx, target in enumerate(REGRESSION_TARGET_COLUMNS)
        }
        regression_mse = {
            target: float(mean_squared_error(reg_true[:, idx], reg_pred[:, idx]))
            for idx, target in enumerate(REGRESSION_TARGET_COLUMNS)
        }
        threshold_counts = target_threshold_counts(reg_true, reg_pred)
        airflow_tp = int(np.sum((airflow_true == 1) & (airflow_pred == 1)))
        airflow_tn = int(np.sum((airflow_true == 0) & (airflow_pred == 0)))
        airflow_fp = int(np.sum((airflow_true == 0) & (airflow_pred == 1)))
        airflow_fn = int(np.sum((airflow_true == 1) & (airflow_pred == 0)))

        overall_loss = float(np.mean(list(regression_mae.values())))
        count = self.y_test.shape[0]
        return overall_loss, int(count), {
            "room_id": self.room_id,
            "mae_sum_y_target_temp_main": regression_mae["y_target_temp_main"] * count,
            "mae_sum_y_target_temp_toilet": regression_mae["y_target_temp_toilet"] * count,
            "mae_sum_y_target_light": regression_mae["y_target_light"] * count,
            "mae_sum_y_target_sound": regression_mae["y_target_sound"] * count,
            "mse_sum_y_target_temp_main": regression_mse["y_target_temp_main"] * count,
            "mse_sum_y_target_temp_toilet": regression_mse["y_target_temp_toilet"] * count,
            "mse_sum_y_target_light": regression_mse["y_target_light"] * count,
            "mse_sum_y_target_sound": regression_mse["y_target_sound"] * count,
            "threshold_correct_y_target_temp_main": threshold_counts["y_target_temp_main"][0],
            "threshold_wrong_y_target_temp_main": threshold_counts["y_target_temp_main"][1],
            "threshold_correct_y_target_temp_toilet": threshold_counts["y_target_temp_toilet"][0],
            "threshold_wrong_y_target_temp_toilet": threshold_counts["y_target_temp_toilet"][1],
            "threshold_correct_y_target_light": threshold_counts["y_target_light"][0],
            "threshold_wrong_y_target_light": threshold_counts["y_target_light"][1],
            "threshold_correct_y_target_sound": threshold_counts["y_target_sound"][0],
            "threshold_wrong_y_target_sound": threshold_counts["y_target_sound"][1],
            "airflow_accuracy_sum": float(accuracy_score(airflow_true, airflow_pred)) * count,
            "airflow_precision_sum": float(precision_score(airflow_true, airflow_pred, zero_division=0)) * count,
            "airflow_recall_sum": float(recall_score(airflow_true, airflow_pred, zero_division=0)) * count,
            "airflow_f1_sum": float(f1_score(airflow_true, airflow_pred, zero_division=0)) * count,
            "airflow_correct": int(np.sum(airflow_true == airflow_pred)),
            "airflow_incorrect": int(count - np.sum(airflow_true == airflow_pred)),
            "airflow_tp": airflow_tp,
            "airflow_tn": airflow_tn,
            "airflow_fp": airflow_fp,
            "airflow_fn": airflow_fn,
        }


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

    train_df = sanitize_rows(load_room_df(split_dir, "train", room_id=room_id, chunksize=args.chunksize))
    test_df = sanitize_rows(load_room_df(split_dir, "test", room_id=room_id, chunksize=args.chunksize))
    if len(train_df) == 0:
        raise RuntimeError(f"No training rows for room_id={room_id}")

    x_train = row_to_input_vector(train_df).astype(np.float32)
    y_train = build_target_matrix(train_df)
    x_test = row_to_input_vector(test_df).astype(np.float32)
    y_test = build_target_matrix(test_df)

    client = RoomClient(
        room_id=room_id,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        input_dim=get_input_dim(),
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        hidden_layers=args.hidden_layers,
        learning_rate=args.learning_rate,
        optimizer_name=args.optimizer,
        activation=args.activation,
    )

    attempts = 0
    while attempts < args.connect_retries:
        attempts += 1
        try:
            fl.client.start_numpy_client(server_address=args.server_address, client=client)
            return
        except Exception:
            if attempts >= args.connect_retries:
                raise
            time.sleep(args.retry_wait)


if __name__ == "__main__":
    main()
