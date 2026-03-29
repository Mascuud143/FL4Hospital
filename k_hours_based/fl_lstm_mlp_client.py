import argparse
import os
import time
from collections import OrderedDict
from typing import Any

import flwr as fl
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "PyTorch is required for ai_fl_lstm_MLP. Install torch in the project environment before running this mode."
    ) from exc

from next_hour_schema import AIRFLOW_INDEX, CHANGE_BASELINE_COLUMNS, INPUT_COLUMNS, TARGET_COLUMNS, row_to_input_vector

DEFAULT_OUTPUT_THRESHOLDS = {
    "y_temp_main": 1.0,
    "y_temp_toilet": 1.0,
    "y_light": 5.0,
    "y_sound": 5.0,
}


def make_activation(name: str) -> nn.Module:
    return {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "gelu": nn.GELU(),
    }[name]


def _safe_logit(p: float) -> float:
    clipped = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return float(np.log(clipped / (1.0 - clipped)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower room client for federated next-hour hybrid MLP+LSTM training.")
    parser.add_argument("--split-dir", default="ai/splits_next_hour", help="Directory with next_hour train and test CSV files")
    parser.add_argument("--room-id", required=True, help="Room/client id to run")
    parser.add_argument("--server-address", default="127.0.0.1:8080", help="Flower server address")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per FL round")
    parser.add_argument("--connect-retries", type=int, default=60, help="Connection retries")
    parser.add_argument("--retry-wait", type=float, default=1.0, help="Seconds between retries")
    parser.add_argument("--chunksize", type=int, default=200000, help="CSV chunksize for room filtering")
    parser.add_argument("--sequence-length", type=int, default=4, help="Number of historical rows per LSTM sample")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for local training")
    parser.add_argument("--change-hidden-layers", default="128,64", help="Comma-separated hidden layer sizes for the change MLP branch")
    parser.add_argument("--lstm-hidden-dim", type=int, default=64, help="Hidden dimension size for the target LSTM branch")
    parser.add_argument("--lstm-num-layers", type=int, default=1, help="Number of stacked LSTM layers in the target branch")
    parser.add_argument("--lstm-head-hidden-dim", type=int, default=64, help="Dense head hidden dimension after the target LSTM")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for local training")
    parser.add_argument("--optimizer", choices=["adam", "sgd", "rmsprop"], default="adam", help="Optimizer for local training")
    parser.add_argument("--change-activation", choices=["relu", "tanh", "gelu"], default="relu", help="Activation function for the hybrid change MLP branch")
    parser.add_argument("--lstm-activation", choices=["relu", "tanh", "gelu"], default="relu", help="Activation function for the hybrid LSTM dense head")
    return parser.parse_args()


def get_input_dim() -> int:
    return len(INPUT_COLUMNS)


def parse_hidden_layers(spec: str) -> tuple[int, ...]:
    values = [part.strip() for part in str(spec).split(",")]
    layers = tuple(int(part) for part in values if part)
    if not layers or any(size <= 0 for size in layers):
        raise ValueError(f"Invalid hidden layer specification: {spec}")
    return layers


class ChangeMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: tuple[int, ...] = (128, 64), activation: str = "relu"):
        super().__init__()
        # This branch answers: "do we need any comfort change at all?"
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(make_activation(activation))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TargetLSTM(nn.Module):
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
        # This branch answers: "if we predict the next state, what should the
        # actual comfort values be after looking at recent history?"
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


class HybridNextHourModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        change_hidden_layers: tuple[int, ...] = (128, 64),
        lstm_hidden_dim: int = 64,
        lstm_num_layers: int = 1,
        lstm_head_hidden_dim: int = 64,
        change_activation: str = "relu",
        lstm_activation: str = "relu",
    ):
        super().__init__()
        # The model keeps the change-decision task separate from the target-value task.
        self.change_mlp = ChangeMLP(
            input_dim=input_dim,
            hidden_layers=change_hidden_layers,
            activation=change_activation,
        )
        self.target_lstm = TargetLSTM(
            input_dim=input_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            head_hidden_dim=lstm_head_hidden_dim,
            activation=lstm_activation,
        )


def make_model(
    input_dim: int,
    change_hidden_layers: tuple[int, ...] = (128, 64),
    lstm_hidden_dim: int = 64,
    lstm_num_layers: int = 1,
    lstm_head_hidden_dim: int = 64,
    change_activation: str = "relu",
    lstm_activation: str = "relu",
) -> HybridNextHourModel:
    return HybridNextHourModel(
        input_dim=input_dim,
        change_hidden_layers=change_hidden_layers,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_num_layers=lstm_num_layers,
        lstm_head_hidden_dim=lstm_head_hidden_dim,
        change_activation=change_activation,
        lstm_activation=lstm_activation,
    )


def get_params(model: HybridNextHourModel) -> list[np.ndarray]:
    return [tensor.detach().cpu().numpy() for _, tensor in model.state_dict().items()]


def set_params(model: HybridNextHourModel, params: list[np.ndarray]) -> None:
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
    for col in ["t", *CHANGE_BASELINE_COLUMNS, "y_any_change", *INPUT_COLUMNS, *TARGET_COLUMNS]:
        if col != "t":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["t", *CHANGE_BASELINE_COLUMNS, "y_any_change", *INPUT_COLUMNS, *TARGET_COLUMNS]).copy()
    df["y_airflow"] = df["y_airflow"].clip(0.0, 1.0)
    df["y_any_change"] = df["y_any_change"].clip(0.0, 1.0)
    df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
    return df.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)


def build_hybrid_arrays(
    df: pd.DataFrame,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_rows = row_to_input_vector(df)
    y_rows = df[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    current_rows = df[CHANGE_BASELINE_COLUMNS].to_numpy(dtype=np.float32)
    change_rows = df["y_any_change"].to_numpy(dtype=np.float32)
    if len(df) < sequence_length:
        empty_seq = np.zeros((0, sequence_length, len(INPUT_COLUMNS)), dtype=np.float32)
        empty_flat = np.zeros((0, len(INPUT_COLUMNS)), dtype=np.float32)
        empty_target = np.zeros((0, len(TARGET_COLUMNS)), dtype=np.float32)
        empty_current = np.zeros((0, len(CHANGE_BASELINE_COLUMNS)), dtype=np.float32)
        empty_change = np.zeros((0,), dtype=np.float32)
        return empty_seq, empty_flat, empty_target, empty_current, empty_change

    x_seq = []
    x_flat = []
    y_seq = []
    current_seq = []
    change_seq = []
    # Every sample contains:
    # - one flat row for the change classifier
    # - one sequence window for the target predictor
    # - the target values from the last row in that window
    for idx in range(sequence_length - 1, len(df)):
        start_idx = idx - sequence_length + 1
        x_seq.append(x_rows[start_idx : idx + 1])
        x_flat.append(x_rows[idx])
        y_seq.append(y_rows[idx])
        current_seq.append(current_rows[idx])
        change_seq.append(change_rows[idx])
    return (
        np.asarray(x_seq, dtype=np.float32),
        np.asarray(x_flat, dtype=np.float32),
        np.asarray(y_seq, dtype=np.float32),
        np.asarray(current_seq, dtype=np.float32),
        np.asarray(change_seq, dtype=np.float32),
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


def per_target_threshold_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, tuple[int, int]]:
    counts: dict[str, tuple[int, int]] = {}
    for idx, target in enumerate(TARGET_COLUMNS[:AIRFLOW_INDEX]):
        threshold = DEFAULT_OUTPUT_THRESHOLDS[target]
        within_threshold = np.abs(y_true[:, idx] - y_pred[:, idx]) <= threshold
        correct = int(np.sum(within_threshold))
        counts[target] = (correct, int(y_true.shape[0] - correct))
    return counts


def _pos_weight_from_targets(values: np.ndarray) -> torch.Tensor:
    positives = float(np.sum(values >= 0.5))
    negatives = float(len(values) - positives)
    if positives <= 0.0:
        return torch.tensor(1.0, dtype=torch.float32)
    return torch.tensor(max(negatives / positives, 1.0), dtype=torch.float32)


def _balanced_binary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: float,
    neg_weight: float,
) -> torch.Tensor:
    base = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = torch.where(
        targets >= 0.5,
        torch.full_like(targets, fill_value=float(pos_weight)),
        torch.full_like(targets, fill_value=float(neg_weight)),
    )
    return (base * weights).mean()


def _class_weights(values: np.ndarray) -> tuple[float, float]:
    if len(values) == 0:
        return 1.0, 1.0
    positives = float(np.sum(values >= 0.5))
    negatives = float(len(values) - positives)
    if positives <= 0.0:
        return 1.0, 2.0
    if negatives <= 0.0:
        return 2.0, 1.0
    pos_frac = positives / (positives + negatives)
    neg_frac = 1.0 - pos_frac
    # Inverse-frequency weighting with bounded scale to avoid unstable gradients.
    pos_weight = min(max(1.0 / max(pos_frac, 1e-6), 0.5), 5.0)
    neg_weight = min(max(1.0 / max(neg_frac, 1e-6), 0.5), 5.0)
    return float(pos_weight), float(neg_weight)


def _best_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    if len(probs) == 0:
        return 0.5
    if np.all(labels == labels[0]):
        return 0.5
    best_thr = 0.5
    best_f1 = -1.0
    best_prec = -1.0
    for thr in np.linspace(0.1, 0.9, 17):
        pred = (probs >= thr).astype(int)
        f1 = float(f1_score(labels, pred, zero_division=0))
        prec = float(precision_score(labels, pred, zero_division=0))
        if f1 > best_f1 or (f1 == best_f1 and prec > best_prec):
            best_f1 = f1
            best_prec = prec
            best_thr = float(thr)
    return best_thr


class RoomHybridClient(fl.client.NumPyClient):
    def __init__(
        self,
        room_id: str,
        x_train_seq: np.ndarray,
        x_train_flat: np.ndarray,
        y_train: np.ndarray,
        change_train: np.ndarray,
        x_test_seq: np.ndarray,
        x_test_flat: np.ndarray,
        y_test: np.ndarray,
        current_test: np.ndarray,
        change_true: np.ndarray,
        input_dim: int,
        local_epochs: int,
        batch_size: int,
        change_hidden_layers: tuple[int, ...] = (128, 64),
        lstm_hidden_dim: int = 64,
        lstm_num_layers: int = 1,
        lstm_head_hidden_dim: int = 64,
        learning_rate: float = 1e-3,
        optimizer: str = "adam",
        change_activation: str = "relu",
        lstm_activation: str = "relu",
    ):
        self.room_id = room_id
        self.x_train_seq = x_train_seq
        self.x_train_flat = x_train_flat
        self.y_train = y_train
        self.change_train = change_train
        self.x_test_seq = x_test_seq
        self.x_test_flat = x_test_flat
        self.y_test = y_test
        self.current_test = current_test
        self.change_true = change_true.astype(np.int64)
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.device = torch.device("cpu")
        self.model = make_model(
            input_dim,
            change_hidden_layers=change_hidden_layers,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            lstm_head_hidden_dim=lstm_head_hidden_dim,
            change_activation=change_activation,
            lstm_activation=lstm_activation,
        ).to(self.device)
        change_pos_weight, change_neg_weight = _class_weights(self.change_train)
        self.change_pos_weight = float(change_pos_weight)
        self.change_neg_weight = float(change_neg_weight)

        airflow_train = self.y_train[:, AIRFLOW_INDEX] if self.y_train.size else np.zeros((0,), dtype=np.float32)
        airflow_pos_weight, airflow_neg_weight = _class_weights(airflow_train)
        self.airflow_pos_weight = float(airflow_pos_weight)
        self.airflow_neg_weight = float(airflow_neg_weight)

        change_prior = float(np.mean(self.change_train >= 0.5)) if len(self.change_train) else 0.5
        airflow_prior = float(np.mean(airflow_train >= 0.5)) if len(airflow_train) else 0.5
        self.change_threshold = 0.5
        self.airflow_threshold = 0.5

        # Bias initialization helps prevent immediate collapse to one class.
        with torch.no_grad():
            final_layer = self.model.change_mlp.net[-1]
            if isinstance(final_layer, nn.Linear):
                final_layer.bias.fill_(_safe_logit(change_prior))
            airflow_head = self.model.target_lstm.head[-1]
            if isinstance(airflow_head, nn.Linear):
                airflow_head.bias[AIRFLOW_INDEX] = _safe_logit(airflow_prior)

    def get_parameters(self, config: dict[str, Any]):
        return get_params(self.model)

    def fit(self, parameters, config):
        set_params(self.model, parameters)
        if self.y_train.size == 0:
            return get_params(self.model), 0, {"room_id": self.room_id}

        proximal_mu = float(config.get("proximal_mu", 0.0) or 0.0)
        self.model.train()
        dataset = TensorDataset(
            torch.tensor(self.x_train_flat, dtype=torch.float32, device=self.device),
            torch.tensor(self.x_train_seq, dtype=torch.float32, device=self.device),
            torch.tensor(self.change_train, dtype=torch.float32, device=self.device),
            torch.tensor(self.y_train, dtype=torch.float32, device=self.device),
        )
        loader = DataLoader(dataset, batch_size=max(1, min(self.batch_size, len(dataset))), shuffle=True)
        optimizer_cls = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop,
        }[self.optimizer_name]
        optimizer_mlp = optimizer_cls(self.model.change_mlp.parameters(), lr=self.learning_rate)
        optimizer_lstm = optimizer_cls(self.model.target_lstm.parameters(), lr=self.learning_rate)
        train_loss_sum = 0.0
        global_params = [param.detach().clone() for param in self.model.parameters()]

        for _ in range(self.local_epochs):
            for batch_flat, batch_seq, batch_change, batch_targets in loader:
                optimizer_mlp.zero_grad()
                # Branch 1: classify whether any comfort change should happen.
                change_logits = self.model.change_mlp(batch_flat)
                change_loss = _balanced_binary_loss(
                    change_logits,
                    batch_change,
                    pos_weight=self.change_pos_weight,
                    neg_weight=self.change_neg_weight,
                )
                if proximal_mu > 0.0:
                    prox_reg = torch.zeros((), device=self.device)
                    for param, global_param in zip(self.model.parameters(), global_params):
                        prox_reg = prox_reg + torch.sum((param - global_param) ** 2)
                    change_loss = change_loss + (proximal_mu / 2.0) * prox_reg
                change_loss.backward()
                optimizer_mlp.step()

                optimizer_lstm.zero_grad()
                # Branch 2: predict the concrete next-hour settings from recent history.
                target_preds = self.model.target_lstm(batch_seq)
                reg_loss = torch.nn.functional.mse_loss(
                    target_preds[:, :AIRFLOW_INDEX],
                    batch_targets[:, :AIRFLOW_INDEX],
                )
                # Airflow is treated like a binary target, so we train it with
                # a weighted binary loss instead of plain regression.
                airflow_loss = _balanced_binary_loss(
                    target_preds[:, AIRFLOW_INDEX],
                    batch_targets[:, AIRFLOW_INDEX],
                    pos_weight=self.airflow_pos_weight,
                    neg_weight=self.airflow_neg_weight,
                )
                target_loss = reg_loss + airflow_loss
                if proximal_mu > 0.0:
                    prox_reg = torch.zeros((), device=self.device)
                    for param, global_param in zip(self.model.parameters(), global_params):
                        prox_reg = prox_reg + torch.sum((param - global_param) ** 2)
                    target_loss = target_loss + (proximal_mu / 2.0) * prox_reg
                target_loss.backward()
                optimizer_lstm.step()
                train_loss_sum += float(change_loss.item() + target_loss.item()) * float(batch_targets.shape[0])

        # After local training we re-estimate thresholds from local predictions,
        # because class balance differs from room to room.
        with torch.no_grad():
            train_change_logits = self.model.change_mlp(
                torch.tensor(self.x_train_flat, dtype=torch.float32, device=self.device)
            ).cpu().numpy()
            train_change_probs = 1.0 / (1.0 + np.exp(-train_change_logits))
            self.change_threshold = _best_threshold(
                train_change_probs.astype(np.float64),
                (self.change_train >= 0.5).astype(int),
            )

            train_target_logits = self.model.target_lstm(
                torch.tensor(self.x_train_seq, dtype=torch.float32, device=self.device)
            ).cpu().numpy()
            train_airflow_probs = 1.0 / (1.0 + np.exp(-train_target_logits[:, AIRFLOW_INDEX]))
            self.airflow_threshold = _best_threshold(
                train_airflow_probs.astype(np.float64),
                np.rint(self.y_train[:, AIRFLOW_INDEX]).astype(int),
            )

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
            # We evaluate both branches and then derive change metrics from the
            # calibrated change probabilities.
            y_pred = self.model.target_lstm(torch.tensor(self.x_test_seq, dtype=torch.float32, device=self.device)).cpu().numpy()
            change_logits = self.model.change_mlp(torch.tensor(self.x_test_flat, dtype=torch.float32, device=self.device)).cpu().numpy()

        reg_true = self.y_test[:, :AIRFLOW_INDEX]
        reg_pred = y_pred[:, :AIRFLOW_INDEX]
        airflow_true = self.y_test[:, AIRFLOW_INDEX].round().astype(int)
        airflow_prob = 1.0 / (1.0 + np.exp(-y_pred[:, AIRFLOW_INDEX]))
        airflow_pred = (airflow_prob >= self.airflow_threshold).astype(int)
        change_prob = 1.0 / (1.0 + np.exp(-change_logits))
        change_pred = (change_prob >= self.change_threshold).astype(int)

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

        tp = int(np.sum((airflow_true == 1) & (airflow_pred == 1)))
        tn = int(np.sum((airflow_true == 0) & (airflow_pred == 0)))
        fp = int(np.sum((airflow_true == 0) & (airflow_pred == 1)))
        fn = int(np.sum((airflow_true == 1) & (airflow_pred == 0)))
        airflow_accuracy = float(accuracy_score(airflow_true, airflow_pred))
        airflow_precision = float(precision_score(airflow_true, airflow_pred, zero_division=0))
        airflow_recall = float(recall_score(airflow_true, airflow_pred, zero_division=0))
        airflow_f1 = float(f1_score(airflow_true, airflow_pred, zero_division=0))

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
            "airflow_threshold": float(self.airflow_threshold),
            "change_threshold": float(self.change_threshold),
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
    x_train_seq, x_train_flat, y_train, _, change_train = build_hybrid_arrays(train_df, args.sequence_length)
    x_test_seq, x_test_flat, y_test, current_test, change_true = build_hybrid_arrays(test_df, args.sequence_length)
    if len(y_train) == 0:
        raise RuntimeError(f"Not enough training rows for room_id={room_id} and sequence_length={args.sequence_length}")

    client = RoomHybridClient(
        room_id=room_id,
        x_train_seq=x_train_seq,
        x_train_flat=x_train_flat,
        y_train=y_train,
        change_train=change_train,
        x_test_seq=x_test_seq,
        x_test_flat=x_test_flat,
        y_test=y_test,
        current_test=current_test,
        change_true=change_true,
        input_dim=get_input_dim(),
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        change_hidden_layers=parse_hidden_layers(args.change_hidden_layers),
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_num_layers=args.lstm_num_layers,
        lstm_head_hidden_dim=args.lstm_head_hidden_dim,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        change_activation=args.change_activation,
        lstm_activation=args.lstm_activation,
    )

    print("fl_client_lstm_mlp.py starting")
    print(f"room_id={room_id}")
    print(f"split_dir={split_dir}")
    print(f"server_address={args.server_address}")
    print(f"sequence_length={args.sequence_length}")
    print(f"change_hidden_layers={args.change_hidden_layers}")
    print(f"lstm_hidden_dim={args.lstm_hidden_dim}")
    print(f"lstm_num_layers={args.lstm_num_layers}")
    print(f"lstm_head_hidden_dim={args.lstm_head_hidden_dim}")
    print(f"learning_rate={args.learning_rate}")
    print(f"optimizer={args.optimizer}")
    print(f"change_activation={args.change_activation}")
    print(f"lstm_activation={args.lstm_activation}")
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
