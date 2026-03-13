import argparse
import os

import flwr as fl
import pandas as pd

from fl_client_lstm_mlp_2 import TARGET_COLUMNS, get_input_dim, get_params, make_model
from fl_server_lstm_mlp import TrackingFedAvg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower server for next-hour hybrid MLP+LSTM training v2.")
    parser.add_argument("--split-dir", default="ai/splits_next_hour", help="Directory with next_hour train split")
    parser.add_argument("--server-address", default="127.0.0.1:8080", help="Server bind address")
    parser.add_argument("--rounds", type=int, default=3, help="Federated rounds")
    parser.add_argument("--min-fit-clients", type=int, default=2, help="Minimum clients for fit")
    parser.add_argument("--min-evaluate-clients", type=int, default=2, help="Minimum clients for evaluate")
    parser.add_argument("--min-available-clients", type=int, default=2, help="Minimum connected clients")
    parser.add_argument("--fraction-fit", type=float, default=1.0, help="Fraction of clients sampled for fit")
    parser.add_argument("--fraction-evaluate", type=float, default=1.0, help="Fraction of clients sampled for evaluate")
    parser.add_argument("--weights-out-dir", default="ai/fl_weights_next_hour_lstm_mlp_2", help="Directory to write global weights per round")
    return parser.parse_args()


def make_initial_parameters() -> fl.common.Parameters:
    model = make_model(get_input_dim())
    return fl.common.ndarrays_to_parameters(get_params(model))


def count_rooms(split_dir: str) -> int:
    train_path = os.path.join(split_dir, "next_hour_train.csv")
    if not os.path.exists(train_path):
        return 0
    df = pd.read_csv(train_path, usecols=["client_id"])
    return int(df["client_id"].astype(str).nunique())


def main() -> None:
    args = parse_args()
    split_dir = os.path.abspath(args.split_dir)
    weights_out_dir = os.path.abspath(args.weights_out_dir)
    rooms = count_rooms(split_dir)

    strategy = TrackingFedAvg(
        weights_out_dir=weights_out_dir,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=args.min_fit_clients,
        min_evaluate_clients=args.min_evaluate_clients,
        min_available_clients=args.min_available_clients,
        initial_parameters=make_initial_parameters(),
    )

    print("fl_server_lstm_mlp_2.py starting")
    print(f"split_dir={split_dir}")
    print(f"targets={','.join(TARGET_COLUMNS)}")
    print(f"rooms_detected={rooms}")
    print(f"server_address={args.server_address}")
    print(f"rounds={args.rounds}")
    print(f"weights_out_dir={weights_out_dir}")
    print("waiting_for_room_clients...")

    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
if __name__ == "__main__":
    main()
