from __future__ import annotations

import csv
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def resolve_db_path(database_root: str | Path, db_id: str) -> Path:
    root = Path(database_root)
    candidates = (
        root / db_id / f"{db_id}.sqlite",
        root / db_id / f"{db_id}.db",
        root / f"{db_id}.sqlite",
        root / f"{db_id}.db",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"cannot find SQLite database for {db_id!r} under {root}")


def _description_rows(db_dir: Path) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    description_dir = db_dir / "database_description"
    if not description_dir.is_dir():
        return result
    for path in sorted(description_dir.glob("*.csv")):
        table = path.stem
        try:
            with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
                for row in csv.DictReader(handle):
                    column = (row.get("original_column_name") or row.get("column_name") or "").strip()
                    description = (row.get("column_description") or "").strip()
                    value_description = (row.get("value_description") or "").strip()
                    combined = "; ".join(part for part in (description, value_description) if part)
                    if column and combined:
                        result[(table.casefold(), column.casefold())] = combined
        except (OSError, csv.Error):
            continue
    return result


def _display_cell(value: Any, max_chars: int) -> str:
    text = "NULL" if value is None else str(value).replace("\n", " ")
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"


@lru_cache(maxsize=512)
def render_schema(
    db_path: str,
    sample_rows: int = 3,
    max_description_chars: int = 320,
    max_cell_chars: int = 96,
    description_db_dir: str | None = None,
) -> str:
    path = Path(db_path).resolve()
    metadata_dir = path.parent if description_db_dir is None else Path(description_db_dir).resolve()
    descriptions = _description_rows(metadata_dir)
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro&immutable=1", uri=True)
    try:
        table_rows = connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        sections: list[str] = []
        for (table,) in table_rows:
            q_table = quote_identifier(table)
            columns = connection.execute(f"PRAGMA table_info({q_table})").fetchall()
            foreign_keys = connection.execute(f"PRAGMA foreign_key_list({q_table})").fetchall()
            definitions: list[str] = []
            notes: list[str] = []
            for _, name, col_type, not_null, default, primary_key in columns:
                pieces = [quote_identifier(name), col_type or "TEXT"]
                if primary_key:
                    pieces.append("PRIMARY KEY")
                if not_null:
                    pieces.append("NOT NULL")
                if default is not None:
                    pieces.append(f"DEFAULT {default}")
                definitions.append("  " + " ".join(pieces))
                description = descriptions.get((table.casefold(), name.casefold()))
                if description:
                    description = description.replace("\n", " ")[:max_description_chars]
                    notes.append(f"-- {table}.{name}: {description}")
            for _, _, ref_table, from_col, to_col, *_ in foreign_keys:
                target = quote_identifier(ref_table)
                if to_col is not None:
                    target += f" ({quote_identifier(to_col)})"
                definitions.append(
                    f"  FOREIGN KEY ({quote_identifier(from_col)}) REFERENCES {target}"
                )
            ddl = f"CREATE TABLE {q_table} (\n" + ",\n".join(definitions) + "\n);"
            samples: list[str] = []
            if sample_rows > 0 and columns:
                try:
                    rows = connection.execute(f"SELECT * FROM {q_table} LIMIT ?", (sample_rows,)).fetchall()
                    names = [column[1] for column in columns]
                    samples.append("-- sample columns: " + " | ".join(names))
                    samples.extend(
                        "-- sample row: " + " | ".join(_display_cell(cell, max_cell_chars) for cell in row)
                        for row in rows
                    )
                except sqlite3.Error:
                    pass
            sections.append("\n".join([ddl, *notes, *samples]))
        return "\n\n".join(sections)
    finally:
        connection.close()
