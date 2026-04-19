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


def _append_row(
    *,
    table_name: str,
    fieldnames: Iterable[str],
    row: dict,
) -> None:
    _append_rows(table_name=table_name, fieldnames=fieldnames, rows=[row])


def _append_rows(
    *,
    table_name: str,
    fieldnames: Iterable[str],
    rows: list[dict],
) -> None:
    if not rows:
        return

    _ensure_dir()
    path = _BASE_DIR / f"{table_name}.csv"
    write_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def write_model_row(model: Any) -> None:
    """
    Append a CSV row for a SQLAlchemy model instance.
    Safe to call before the row is committed (autoincrement IDs may be empty).
    """
    table = _table_for(model)
    if table is None:
        return

    try:
        fieldnames = [c.name for c in table.columns]
        row = {name: _serialize_value(getattr(model, name, None)) for name in fieldnames}
        _assign_autoincrement_rows(table, [row])
        _append_row(table_name=table.name, fieldnames=fieldnames, row=row)
    except Exception as exc:
        # Never block simulation if CSV logging fails
        print(f"[csv_filestorage] Failed to write {table}: {exc}")


def write_model_rows(model_class: Any, rows: list[dict]) -> None:
    """
    Append multiple CSV rows for one SQLAlchemy model/table in a single file write.
    """
    table = _table_for(model_class)
    if table is None or not rows:
        return

    try:
        fieldnames = [c.name for c in table.columns]
        serialized_rows = [
            {name: _serialize_value(row.get(name)) for name in fieldnames}
            for row in rows
        ]
        _assign_autoincrement_rows(table, serialized_rows)
        _append_rows(table_name=table.name, fieldnames=fieldnames, rows=serialized_rows)
    except Exception as exc:
        print(f"[csv_filestorage] Failed to write {table}: {exc}")
