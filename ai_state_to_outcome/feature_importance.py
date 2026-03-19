import argparse
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "PyTorch is required for FL-based feature importance. Install torch before running this mode."
    ) from exc

from fl_client import get_input_dim, make_model, sanitize_rows
from schema import FEATURE_COLUMNS, FEATURE_GROUPS, REGRESSION_TARGET_COLUMNS, row_to_input_vector

DEFAULT_SPLIT_DIR = "splits"
DEFAULT_OUT_DIR = "feature_importance"
DEFAULT_WEIGHTS_PATH = "fl_weights/latest_global_weights.npz"


def resolve_path(value: str, script_dir: str) -> str:
    return value if os.path.isabs(value) else os.path.join(script_dir, value)


def load_rows(path: str) -> pd.DataFrame:
    return sanitize_rows(pd.read_csv(path))


def make_regression_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))),
        ]
    )


def make_airflow_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ]
    )


def train_baseline_models(train_df: pd.DataFrame):
    x_train = row_to_input_vector(train_df)
    reg_model = make_regression_model()
    reg_model.fit(x_train, train_df[REGRESSION_TARGET_COLUMNS].to_numpy(dtype=float))

    airflow_model = make_airflow_model()
    airflow_model.fit(x_train, train_df["y_target_airflow"].round().astype(int).to_numpy())
    return reg_model, airflow_model


def load_fl_model(weights_path: str):
    model = make_model(get_input_dim())
    state_dict = model.state_dict()
    raw = np.load(weights_path)
    params = [raw[key] for key in sorted(raw.files, key=lambda item: int(item.split("_")[1]))]
    if len(params) != len(state_dict):
        raise ValueError(f"Parameter count mismatch: expected {len(state_dict)}, got {len(params)}")
    new_state = OrderedDict()
    for (name, tensor), value in zip(state_dict.items(), params):
        new_state[name] = torch.tensor(value, dtype=tensor.dtype)
    model.load_state_dict(new_state, strict=True)
    model.eval()
    return model


def predict_with_baseline(reg_model, airflow_model, df: pd.DataFrame) -> pd.DataFrame:
    x = row_to_input_vector(df)
    reg_pred = reg_model.predict(x)
    airflow_pred = airflow_model.predict(x)
    out = pd.DataFrame(reg_pred, columns=REGRESSION_TARGET_COLUMNS, index=df.index)
    out["y_target_airflow"] = airflow_pred
    return out


def predict_with_fl(model, df: pd.DataFrame) -> pd.DataFrame:
    x = row_to_input_vector(df).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32)).cpu().numpy()
    out = pd.DataFrame(logits[:, :4], columns=REGRESSION_TARGET_COLUMNS, index=df.index)
    airflow_prob = 1.0 / (1.0 + np.exp(-logits[:, 4]))
    out["y_target_airflow"] = (airflow_prob >= 0.5).astype(int)
    return out


def compute_metrics(df: pd.DataFrame, pred_df: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {}
    mae_values: list[float] = []
    for target in REGRESSION_TARGET_COLUMNS:
        mae = float(mean_absolute_error(df[target], pred_df[target]))
        metrics[f"mae_{target}"] = mae
        mae_values.append(mae)
    airflow_true = df["y_target_airflow"].round().astype(int)
    airflow_pred = pred_df["y_target_airflow"].round().astype(int)
    metrics["airflow_accuracy"] = float(accuracy_score(airflow_true, airflow_pred))
    metrics["airflow_f1"] = float(f1_score(airflow_true, airflow_pred, zero_division=0))
    metrics["composite_loss"] = float(np.mean(mae_values) + (1.0 - metrics["airflow_f1"]))
    return metrics


def permutation_importance(df: pd.DataFrame, predict_fn, random_state: int) -> tuple[pd.DataFrame, dict[str, float]]:
    baseline_pred = predict_fn(df)
    baseline_metrics = compute_metrics(df, baseline_pred)
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | str]] = []
    for feature in FEATURE_COLUMNS:
        shuffled = df.copy()
        shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
        perturbed_pred = predict_fn(shuffled)
        perturbed_metrics = compute_metrics(shuffled, perturbed_pred)
        rows.append(
            {
                "feature": feature,
                "delta_composite_loss": perturbed_metrics["composite_loss"] - baseline_metrics["composite_loss"],
                "delta_airflow_f1": baseline_metrics["airflow_f1"] - perturbed_metrics["airflow_f1"],
                "delta_mae_y_target_temp_main": perturbed_metrics["mae_y_target_temp_main"] - baseline_metrics["mae_y_target_temp_main"],
                "delta_mae_y_target_temp_toilet": perturbed_metrics["mae_y_target_temp_toilet"] - baseline_metrics["mae_y_target_temp_toilet"],
                "delta_mae_y_target_light": perturbed_metrics["mae_y_target_light"] - baseline_metrics["mae_y_target_light"],
                "delta_mae_y_target_sound": perturbed_metrics["mae_y_target_sound"] - baseline_metrics["mae_y_target_sound"],
            }
        )
    importance_df = pd.DataFrame(rows).sort_values("delta_composite_loss", ascending=False, kind="stable")
    return importance_df, baseline_metrics


def ablation_study_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    full_reg_model, full_airflow_model = train_baseline_models(train_df)
    baseline_pred = predict_with_baseline(full_reg_model, full_airflow_model, test_df)
    baseline_metrics = compute_metrics(test_df, baseline_pred)

    rows: list[dict[str, float | str]] = []
    for group_name, columns in FEATURE_GROUPS.items():
        reduced_train = train_df.drop(columns=columns)
        reduced_test = test_df.drop(columns=columns)
        reduced_feature_cols = [col for col in FEATURE_COLUMNS if col not in columns]

        def reduced_vector(df: pd.DataFrame) -> np.ndarray:
            return df[reduced_feature_cols].to_numpy(dtype=np.float64)

        reg_model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))),
            ]
        )
        airflow_model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
            ]
        )
        x_train = reduced_vector(reduced_train)
        x_test = reduced_vector(reduced_test)
        reg_model.fit(x_train, reduced_train[REGRESSION_TARGET_COLUMNS].to_numpy(dtype=float))
        airflow_model.fit(x_train, reduced_train["y_target_airflow"].round().astype(int).to_numpy())

        reg_pred = reg_model.predict(x_test)
        airflow_pred = airflow_model.predict(x_test)
        pred_df = pd.DataFrame(reg_pred, columns=REGRESSION_TARGET_COLUMNS, index=reduced_test.index)
        pred_df["y_target_airflow"] = airflow_pred
        metrics = compute_metrics(reduced_test, pred_df)
        rows.append(
            {
                "group": group_name,
                "delta_composite_loss": metrics["composite_loss"] - baseline_metrics["composite_loss"],
                "delta_airflow_f1": baseline_metrics["airflow_f1"] - metrics["airflow_f1"],
                "delta_mae_y_target_temp_main": metrics["mae_y_target_temp_main"] - baseline_metrics["mae_y_target_temp_main"],
                "delta_mae_y_target_temp_toilet": metrics["mae_y_target_temp_toilet"] - baseline_metrics["mae_y_target_temp_toilet"],
                "delta_mae_y_target_light": metrics["mae_y_target_light"] - baseline_metrics["mae_y_target_light"],
                "delta_mae_y_target_sound": metrics["mae_y_target_sound"] - baseline_metrics["mae_y_target_sound"],
            }
        )
    return pd.DataFrame(rows).sort_values("delta_composite_loss", ascending=False, kind="stable"), baseline_metrics


def ablation_study_fl(model, test_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    baseline_pred = predict_with_fl(model, test_df)
    baseline_metrics = compute_metrics(test_df, baseline_pred)
    rows: list[dict[str, float | str]] = []
    for group_name, columns in FEATURE_GROUPS.items():
        ablated = test_df.copy()
        for column in columns:
            if column in ablated.columns:
                ablated[column] = 0.0
        pred_df = predict_with_fl(model, ablated)
        metrics = compute_metrics(ablated, pred_df)
        rows.append(
            {
                "group": group_name,
                "delta_composite_loss": metrics["composite_loss"] - baseline_metrics["composite_loss"],
                "delta_airflow_f1": baseline_metrics["airflow_f1"] - metrics["airflow_f1"],
                "delta_mae_y_target_temp_main": metrics["mae_y_target_temp_main"] - baseline_metrics["mae_y_target_temp_main"],
                "delta_mae_y_target_temp_toilet": metrics["mae_y_target_temp_toilet"] - baseline_metrics["mae_y_target_temp_toilet"],
                "delta_mae_y_target_light": metrics["mae_y_target_light"] - baseline_metrics["mae_y_target_light"],
                "delta_mae_y_target_sound": metrics["mae_y_target_sound"] - baseline_metrics["mae_y_target_sound"],
            }
        )
    return pd.DataFrame(rows).sort_values("delta_composite_loss", ascending=False, kind="stable"), baseline_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task #2 permutation importance and ablation analysis.")
    parser.add_argument("--model-type", choices=["baseline", "fl"], default="baseline", help="Which model family to analyze")
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR, help="Directory with Task #2 train/test splits")
    parser.add_argument("--weights-path", default=DEFAULT_WEIGHTS_PATH, help="Required for model-type=fl")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory for feature importance outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for permutation shuffling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    split_dir = resolve_path(args.split_dir, script_dir)
    out_dir = resolve_path(args.out_dir, script_dir)
    train_path = os.path.join(split_dir, "state_to_outcome_train.csv")
    test_path = os.path.join(split_dir, "state_to_outcome_test.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing train split: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing test split: {test_path}")

    train_df = load_rows(train_path)
    test_df = load_rows(test_path)
    os.makedirs(out_dir, exist_ok=True)

    if args.model_type == "baseline":
        reg_model, airflow_model = train_baseline_models(train_df)
        importance_df, baseline_metrics = permutation_importance(
            test_df,
            predict_fn=lambda df: predict_with_baseline(reg_model, airflow_model, df),
            random_state=args.seed,
        )
        ablation_df, ablation_baseline = ablation_study_baseline(train_df, test_df)
    else:
        weights_path = resolve_path(args.weights_path, script_dir)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Missing weights file: {weights_path}")
        model = load_fl_model(weights_path)
        importance_df, baseline_metrics = permutation_importance(
            test_df,
            predict_fn=lambda df: predict_with_fl(model, df),
            random_state=args.seed,
        )
        ablation_df, ablation_baseline = ablation_study_fl(model, test_df)

    importance_df.to_csv(os.path.join(out_dir, f"permutation_importance_{args.model_type}.csv"), index=False)
    ablation_df.to_csv(os.path.join(out_dir, f"ablation_{args.model_type}.csv"), index=False)
    pd.DataFrame([baseline_metrics]).to_csv(os.path.join(out_dir, f"baseline_metrics_{args.model_type}.csv"), index=False)
    pd.DataFrame([ablation_baseline]).to_csv(os.path.join(out_dir, f"ablation_baseline_metrics_{args.model_type}.csv"), index=False)

    print("feature_importance.py complete")
    print(f"model_type={args.model_type}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
