import argparse
import os
import time
from typing import Any

import flwr as fl
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

FEATURE_COLUMNS = [
    "hour",
    "weekday",
    "day_of_stay",
    "age",
    "height",
    "minutes_since_last_med",
    "curr_temp_main",
    "curr_temp_toilet",
    "curr_light",
    "curr_sound",
    "curr_airflow",
    "minutes_since_last_change",
    "gender",
    "ethnicity",
    "diagnosis",
    "latest_symptom",
    "last_medication",
    "last_med_status",
]

NUMERIC_FEATURES = [
    "hour",
    "weekday",
    "day_of_stay",
    "age",
    "height",
    "minutes_since_last_med",
    "curr_temp_main",
    "curr_temp_toilet",
    "curr_light",
    "curr_sound",
    "curr_airflow",
    "minutes_since_last_change",
]

CATEGORICAL_FEATURES = [
    "gender",
    "ethnicity",
    "diagnosis",
    "latest_symptom",
    "last_medication",
    "last_med_status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower room client for model_a rows.")
    parser.add_argument("--split-dir", default="ai/splits", help="Directory with model_a_train.csv and model_a_test.csv")
    parser.add_argument("--room-id", required=True, help="Room/client id to run")
    parser.add_argument("--server-address", default="127.0.0.1:8080", help="Flower server address")
    parser.add_argument("--n-features", type=int, default=256, help="FeatureHasher output dimension")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local epochs per FL round")
    parser.add_argument("--connect-retries", type=int, default=60, help="Connection retries")
    parser.add_argument("--retry-wait", type=float, default=1.0, help="Seconds between retries")
    parser.add_argument("--chunksize", type=int, default=200000, help="CSV chunksize for room filtering")
    return parser.parse_args()


def make_model(n_features: int) -> SGDClassifier:
    model = SGDClassifier(loss="log_loss", random_state=42, alpha=1e-4, learning_rate="optimal")
    x0 = csr_matrix((1, n_features), dtype=np.float64)
    y0 = np.array([0], dtype=np.int64)
    model.partial_fit(x0, y0, classes=np.array([0, 1], dtype=np.int64))
    return model


def get_params(model: SGDClassifier) -> list[np.ndarray]:
    return [model.coef_.astype(np.float64), model.intercept_.astype(np.float64)]


def set_params(model: SGDClassifier, params: list[np.ndarray]) -> None:
    coef, intercept = params
    model.coef_ = coef.astype(np.float64, copy=True)
    model.intercept_ = intercept.astype(np.float64, copy=True)
    model.classes_ = np.array([0, 1], dtype=np.int64)


def load_room_df(path: str, room_id: str, chunksize: int) -> pd.DataFrame:
    keep_cols = ["client_id"] + FEATURE_COLUMNS + ["y_event"]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=lambda c: c in keep_cols, chunksize=chunksize):
        room_chunk = chunk[chunk["client_id"].astype(str) == room_id]
        if not room_chunk.empty:
            chunks.append(room_chunk)
    if not chunks:
        return pd.DataFrame(columns=keep_cols)
    df = pd.concat(chunks, ignore_index=True)
    df["y_event"] = pd.to_numeric(df["y_event"], errors="coerce").fillna(0).astype(int)
    return df


def rows_to_dicts(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        item: dict[str, Any] = {}
        for col in NUMERIC_FEATURES:
            value = pd.to_numeric(row.get(col), errors="coerce")
            if pd.notna(value):
                item[col] = float(value)
        for col in CATEGORICAL_FEATURES:
            value = row.get(col)
            if value is None:
                continue
            text = str(value).strip()
            if text == "":
                continue
            item[f"{col}={text}"] = 1.0
        rows.append(item)
    return rows


class RoomClient(fl.client.NumPyClient):
    def __init__(self, room_id: str, x_train: Any, y_train: np.ndarray, x_test: Any, y_test: np.ndarray, n_features: int, local_epochs: int):
        self.room_id = room_id
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.local_epochs = local_epochs
        self.model = make_model(n_features)

    def get_parameters(self, config: dict[str, Any]):
        return get_params(self.model)

    def fit(self, parameters, config):
        set_params(self.model, parameters)
        for _ in range(self.local_epochs):
            self.model.partial_fit(self.x_train, self.y_train, classes=np.array([0, 1], dtype=np.int64))
        return get_params(self.model), int(self.y_train.shape[0]), {"room_id": self.room_id}

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        if self.y_test.size == 0:
            return 0.0, 0, {
                "accuracy": 0.0,
                "f1": 0.0,
                "tp": 0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
                "correct": 0,
                "incorrect": 0,
                "room_id": self.room_id,
            }
        y_pred = self.model.predict(self.x_test)
        acc = float(accuracy_score(self.y_test, y_pred))
        f1 = float(f1_score(self.y_test, y_pred, zero_division=0))
        loss = float(1.0 - acc)
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred, labels=[0, 1]).ravel()
        correct = int(tp + tn)
        incorrect = int(fp + fn)
        return loss, int(self.y_test.shape[0]), {
            "accuracy": acc,
            "f1": f1,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "correct": correct,
            "incorrect": incorrect,
            "room_id": self.room_id,
        }


def main() -> None:
    args = parse_args()
    room_id = str(args.room_id)
    split_dir = os.path.abspath(args.split_dir)
    train_path = os.path.join(split_dir, "model_a_train.csv")
    test_path = os.path.join(split_dir, "model_a_test.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing file: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing file: {test_path}")

    train_df = load_room_df(train_path, room_id=room_id, chunksize=args.chunksize)
    test_df = load_room_df(test_path, room_id=room_id, chunksize=args.chunksize)
    if len(train_df) == 0:
        raise RuntimeError(f"No training rows for room_id={room_id}")

    hasher = FeatureHasher(n_features=args.n_features, input_type="dict", alternate_sign=False)
    x_train = hasher.transform(rows_to_dicts(train_df))
    y_train = train_df["y_event"].to_numpy(dtype=np.int64)
    x_test = hasher.transform(rows_to_dicts(test_df))
    y_test = test_df["y_event"].to_numpy(dtype=np.int64)

    client = RoomClient(
        room_id=room_id,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        n_features=args.n_features,
        local_epochs=args.local_epochs,
    )

    print("fl_client.py starting")
    print(f"room_id={room_id}")
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
