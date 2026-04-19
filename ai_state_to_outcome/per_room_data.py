import os
import re

import pandas as pd


def _normalize_room_id(room_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(room_id).strip()).strip("._-")
    return normalized or "unknown"


def room_file_name(room_id: str) -> str:
    return f"room_{_normalize_room_id(room_id)}.csv"


def room_subset_dir(split_dir: str, subset: str) -> str:
    return os.path.join(split_dir, subset)


def room_file_path(split_dir: str, subset: str, room_id: str) -> str:
    return os.path.join(room_subset_dir(split_dir, subset), room_file_name(room_id))


def list_room_ids(split_dir: str, subset: str) -> list[str]:
    subset_dir = room_subset_dir(split_dir, subset)
    if not os.path.isdir(subset_dir):
        return []
    room_ids: list[str] = []
    prefix = "room_"
    for name in os.listdir(subset_dir):
        if not name.endswith(".csv") or not name.startswith(prefix):
            continue
        room_ids.append(name[len(prefix) : -4])
    return sorted(room_ids, key=lambda x: int(x) if x.isdigit() else x)


def load_room_df(split_dir: str, subset: str, room_id: str, usecols: list[str] | None = None) -> pd.DataFrame:
    path = room_file_path(split_dir, subset, room_id)
    if not os.path.exists(path):
        return pd.DataFrame(columns=usecols or [])
    if usecols:
        return pd.read_csv(path, usecols=lambda col: col in set(usecols))
    return pd.read_csv(path)
