from __future__ import annotations

# Write rows to CSV files.
# - Writes one model row with write_model_row()
# - Writes many model rows with write_model_rows()
# - Adds utility rows with insert_utility_usage()
# - Flushes utility rows with flush_utility_usage_writes()

import csv
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from persistence.database import session_scope
from persistence.models.utility_usage import UtilityUsage


_BASE_DIR = Path(__file__).resolve().parents[1] / "filestorage"
_COUNTERS_PATH = _BASE_DIR / "_counters.json"


def _ensure_dir() -> None:
    _BASE_DIR.mkdir(parents=True, exist_ok=True)
    if not _COUNTERS_PATH.exists():
        _COUNTERS_PATH.write_text("{}", encoding="utf-8")


def _load_counters() -> dict:
    try:
        return json.loads(_COUNTERS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_counters(counters: dict) -> None:
    _COUNTERS_PATH.write_text(json.dumps(counters, ensure_ascii=True, indent=2), encoding="utf-8")


def _table_for(model_or_class: Any) -> Any:
    table = getattr(model_or_class, "__table__", None)
    if table is None:
        table = getattr(getattr(model_or_class, "__class__", None), "__table__", None)
    return table


def _assign_autoincrement_rows(table: Any, rows: list[dict]) -> None:
    if table is None:
        return
    pk_cols = list(table.primary_key.columns)
    if len(pk_cols) != 1:
        return
    pk_name = pk_cols[0].name
    missing_rows = [row for row in rows if row.get(pk_name) in (None, "", 0)]
    if not missing_rows:
        return
    _ensure_dir()
    counters = _load_counters()
    next_id = int(counters.get(table.name, 0)) + 1
    for row in missing_rows:
        row[pk_name] = next_id
        next_id += 1
    counters[table.name] = next_id - 1
    _save_counters(counters)


def _serialize_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _append_rows(*, table_name: str, fieldnames: Iterable[str], rows: list[dict]) -> None:
    if not rows:
        return
    _ensure_dir()
    path = _BASE_DIR / f"{table_name}.csv"
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def write_model_row(model: Any) -> None:
    table = _table_for(model)
    if table is None:
        return
    try:
        fieldnames = [column.name for column in table.columns]
        row = {name: _serialize_value(getattr(model, name, None)) for name in fieldnames}
        _assign_autoincrement_rows(table, [row])
        _append_rows(table_name=table.name, fieldnames=fieldnames, rows=[row])
    except Exception as exc:
        print(f"[csv_storage] Failed to write {table}: {exc}")


def write_model_rows(model_class: Any, rows: list[dict]) -> None:
    table = _table_for(model_class)
    if table is None or not rows:
        return
    try:
        fieldnames = [column.name for column in table.columns]
        serialized_rows = [
            {name: _serialize_value(row.get(name)) for name in fieldnames}
            for row in rows
        ]
        _assign_autoincrement_rows(table, serialized_rows)
        _append_rows(table_name=table.name, fieldnames=fieldnames, rows=serialized_rows)
    except Exception as exc:
        print(f"[csv_storage] Failed to write {table}: {exc}")


_UTILITY_USAGE_BUFFER: list[tuple[int, str, datetime, datetime, Optional[float], Optional[float], Optional[int]]] = []
_UTILITY_WRITE_BATCH_SIZE = 50000


def _flush_utility_buffer() -> None:
    global _UTILITY_USAGE_BUFFER
    if not _UTILITY_USAGE_BUFFER:
        return
    batch = _UTILITY_USAGE_BUFFER
    _UTILITY_USAGE_BUFFER = []
    serialized_batch = []
    for room_id, category, start_time, end_time, power_kwh, water_liters, device_id in batch:
        row = {
            "room_id": room_id,
            "category": category,
            "start_time": start_time,
            "end_time": end_time,
            "power_consumption": power_kwh,
            "water_consumption": water_liters,
        }
        if device_id is not None and hasattr(UtilityUsage, "device_id"):
            row["device_id"] = int(device_id)
        serialized_batch.append(row)
    write_model_rows(UtilityUsage, serialized_batch)
    with session_scope() as session:
        session.bulk_insert_mappings(UtilityUsage, serialized_batch)


def flush_utility_usage_writes() -> None:
    _flush_utility_buffer()


def insert_utility_usage(
    *,
    room_id: int,
    category: str,
    start_time: datetime,
    end_time: datetime,
    power_kwh: Optional[float] = None,
    water_liters: Optional[float] = None,
    device_id: Optional[int] = None,
) -> None:
    start_time_utc = start_time.astimezone(timezone.utc)
    end_time_utc = end_time.astimezone(timezone.utc)
    _UTILITY_USAGE_BUFFER.append(
        (
            int(room_id),
            str(category),
            start_time_utc,
            end_time_utc,
            power_kwh,
            water_liters,
            int(device_id) if device_id is not None else None,
        )
    )
    if len(_UTILITY_USAGE_BUFFER) >= _UTILITY_WRITE_BATCH_SIZE:
        _flush_utility_buffer()


__all__ = [
    "flush_utility_usage_writes",
    "insert_utility_usage",
    "write_model_row",
    "write_model_rows",
]
