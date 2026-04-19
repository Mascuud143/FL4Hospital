from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


_CONFIG_PATH = Path(__file__).resolve().with_name("ai_runtime_config.json")
_DEFAULT_CONFIG = {
    "selected_source": "k_hours",
    "k_hours_model_type": "mlp",
    "state_to_outcome_model_type": "mlp",
    "k_hours_weights_path": "k_hours_based/fl_weights_dashboard_mlp/latest_global_weights.npz",
    "state_to_outcome_weights_path": "ai_state_to_outcome/fl_weights_dashboard_mlp/latest_global_weights.npz",
    "room_modes": {},
}


def _normalize(config: dict | None) -> dict:
    normalized = deepcopy(_DEFAULT_CONFIG)
    if isinstance(config, dict):
        normalized.update(
            {
                "selected_source": str(config.get("selected_source", normalized["selected_source"]) or normalized["selected_source"]),
                "k_hours_model_type": str(config.get("k_hours_model_type", normalized["k_hours_model_type"]) or normalized["k_hours_model_type"]),
                "state_to_outcome_model_type": str(
                    config.get("state_to_outcome_model_type", normalized["state_to_outcome_model_type"])
                    or normalized["state_to_outcome_model_type"]
                ),
                "k_hours_weights_path": str(config.get("k_hours_weights_path", normalized["k_hours_weights_path"]) or normalized["k_hours_weights_path"]),
                "state_to_outcome_weights_path": str(
                    config.get("state_to_outcome_weights_path", normalized["state_to_outcome_weights_path"])
                    or normalized["state_to_outcome_weights_path"]
                ),
            }
        )
        room_modes = config.get("room_modes", {})
        if isinstance(room_modes, dict):
            normalized["room_modes"] = {str(key): bool(value) for key, value in room_modes.items()}
    if normalized["selected_source"] not in {"k_hours", "state_to_outcome"}:
        normalized["selected_source"] = _DEFAULT_CONFIG["selected_source"]
    if normalized["k_hours_model_type"] not in {"mlp", "lstm", "mlp_lstm"}:
        normalized["k_hours_model_type"] = _DEFAULT_CONFIG["k_hours_model_type"]
    normalized["state_to_outcome_model_type"] = "mlp"
    return normalized


def load_ai_config() -> dict:
    if not _CONFIG_PATH.exists():
        return _normalize(None)
    try:
        raw = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return _normalize(None)
    return _normalize(raw)


def save_ai_config(config: dict) -> dict:
    normalized = _normalize(config)
    _CONFIG_PATH.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
    return normalized


def update_ai_config(
    *,
    selected_source: str,
    k_hours_model_type: str,
    state_to_outcome_model_type: str,
    k_hours_weights_path: str,
    state_to_outcome_weights_path: str,
) -> dict:
    config = load_ai_config()
    config["selected_source"] = selected_source
    config["k_hours_model_type"] = k_hours_model_type
    config["state_to_outcome_model_type"] = state_to_outcome_model_type
    config["k_hours_weights_path"] = k_hours_weights_path
    config["state_to_outcome_weights_path"] = state_to_outcome_weights_path
    return save_ai_config(config)


def is_room_ai_enabled(room_id: int) -> bool:
    config = load_ai_config()
    return bool(config.get("room_modes", {}).get(str(room_id), False))


def set_room_ai_enabled(room_id: int, enabled: bool) -> dict:
    config = load_ai_config()
    room_modes = dict(config.get("room_modes", {}))
    room_modes[str(room_id)] = bool(enabled)
    config["room_modes"] = room_modes
    return save_ai_config(config)
