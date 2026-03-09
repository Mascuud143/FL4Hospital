from __future__ import annotations

import csv
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable


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


def _maybe_assign_autoincrement(model: Any, row: dict) -> None:
    table = getattr(model, "__table__", None)
    if table is None:
        return

    pk_cols = list(table.primary_key.columns)
    if len(pk_cols) != 1:
        return

    pk_name = pk_cols[0].name
    cur_val = row.get(pk_name)
    if cur_val not in (None, "", 0):
        return

    _ensure_dir()
    counters = _load_counters()
    next_id = int(counters.get(table.name, 0)) + 1
    counters[table.name] = next_id
    _save_counters(counters)
    row[pk_name] = next_id


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


def _append_row(
    *,
    table_name: str,
    fieldnames: Iterable[str],
    row: dict,
) -> None:
    _ensure_dir()
    path = _BASE_DIR / f"{table_name}.csv"
    write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_model_row(model: Any) -> None:
    """
    Append a CSV row for a SQLAlchemy model instance.
    Safe to call before the row is committed (autoincrement IDs may be empty).
    """
    table = getattr(model, "__table__", None)
    if table is None:
        return

    try:
        fieldnames = [c.name for c in table.columns]
        row = {name: _serialize_value(getattr(model, name, None)) for name in fieldnames}
        _maybe_assign_autoincrement(model, row)
        _append_row(table_name=table.name, fieldnames=fieldnames, row=row)
    except Exception as exc:
        # Never block simulation if CSV logging fails
        print(f"[csv_filestorage] Failed to write {table}: {exc}")
