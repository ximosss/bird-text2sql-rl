from __future__ import annotations

import math
import json
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp


class UnsafeSQLError(ValueError):
    pass


@dataclass(frozen=True)
class ExecutionResult:
    ok: bool
    rows: frozenset[tuple[Any, ...]]
    error: str | None = None
    timed_out: bool = False
    ordered_rows: tuple[tuple[Any, ...], ...] = ()


_FORBIDDEN_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Command,
    exp.Transaction,
)


def validate_read_only_sql(sql: str) -> None:
    try:
        statements = [statement for statement in sqlglot.parse(sql, read="sqlite") if statement is not None]
    # Deeply nested model output can overflow sqlglot's recursive parser before
    # it raises SqlglotError. Both cases are ordinary invalid model outputs.
    except (sqlglot.errors.SqlglotError, RecursionError) as exc:
        raise UnsafeSQLError(f"parse_error: {exc}") from exc
    if len(statements) != 1:
        raise UnsafeSQLError("exactly one SQL statement is required")
    statement = statements[0]
    if any(statement.find(node_type) is not None for node_type in _FORBIDDEN_NODES):
        raise UnsafeSQLError("write or control statements are forbidden")
    if statement.find(exp.Select) is None:
        raise UnsafeSQLError("a SELECT query is required")


def _normalize_cell(value: Any, float_digits: int) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return ("float", "nan")
        if math.isinf(value):
            return ("float", "inf" if value > 0 else "-inf")
        return round(value, float_digits)
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    return value


def normalize_rows(rows: list[tuple[Any, ...]], float_digits: int = 10) -> frozenset[tuple[Any, ...]]:
    """BIRD-style comparison: ignore row order/duplicates, preserve value case."""
    return frozenset(tuple(_normalize_cell(cell, float_digits) for cell in row) for row in rows)


def normalize_ordered_rows(
    rows: list[tuple[Any, ...]], float_digits: int = 10
) -> tuple[tuple[Any, ...], ...]:
    return tuple(tuple(_normalize_cell(cell, float_digits) for cell in row) for row in rows)


def _encode_cell(value: Any) -> Any:
    if isinstance(value, tuple):
        return {"__bird_tuple__": [_encode_cell(item) for item in value]}
    return value


def _decode_cell(value: Any) -> Any:
    if isinstance(value, dict) and "__bird_tuple__" in value:
        return tuple(_decode_cell(item) for item in value["__bird_tuple__"])
    return value


def serialize_rows(rows: frozenset[tuple[Any, ...]]) -> str:
    encoded = [[_encode_cell(cell) for cell in row] for row in rows]
    encoded.sort(key=lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True, default=str))
    return json.dumps(encoded, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def deserialize_rows(value: str) -> frozenset[tuple[Any, ...]]:
    rows = json.loads(value)
    return frozenset(tuple(_decode_cell(cell) for cell in row) for row in rows)


def serialize_ordered_rows(rows: tuple[tuple[Any, ...], ...]) -> str:
    encoded = [[_encode_cell(cell) for cell in row] for row in rows]
    return json.dumps(encoded, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def deserialize_ordered_rows(value: str) -> tuple[tuple[Any, ...], ...]:
    rows = json.loads(value)
    return tuple(tuple(_decode_cell(cell) for cell in row) for row in rows)


def _nonempty_rows(rows: tuple[tuple[Any, ...], ...]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        row
        for row in rows
        if any(cell is not None and str(cell).strip() for cell in row)
    )


def _row_key(row: tuple[Any, ...]) -> tuple[str, ...]:
    # Match the ReViSQL/BIRD-Platinum contract: column order inside a result row
    # is ignored, while list grading still preserves row order.
    return tuple(sorted(str(cell) for cell in row))


def compare_result_rows(
    gold_rows: tuple[tuple[Any, ...], ...],
    predicted_rows: tuple[tuple[Any, ...], ...],
    grading_method: str = "set",
) -> bool:
    gold = _nonempty_rows(gold_rows)
    predicted = _nonempty_rows(predicted_rows)
    if not gold or not predicted:
        return gold == predicted
    if len(gold[0]) != len(predicted[0]):
        return False

    if len(gold) == len(predicted) == 1 and len(gold[0]) == len(predicted[0]) == 1:
        try:
            expected = float(gold[0][0])
            actual = float(predicted[0][0])
        except (TypeError, ValueError):
            pass
        else:
            scale = max(abs(expected), 1.0)
            if abs(expected - actual) / scale < 1e-2:
                return True

    method = grading_method.strip().casefold()
    gold_keys = tuple(_row_key(row) for row in gold)
    predicted_keys = tuple(_row_key(row) for row in predicted)
    if method == "multiset":
        return Counter(gold_keys) == Counter(predicted_keys)
    if method == "list":
        return gold_keys == predicted_keys
    if method == "set":
        return set(gold_keys) == set(predicted_keys)
    if method == "subset":
        return bool(predicted_keys) and set(predicted_keys).issubset(set(gold_keys))
    if method.startswith("subset,"):
        parts = [part.strip() for part in method.split(",")]
        if len(parts) != 3 or parts[1] not in {"=", ">="}:
            raise ValueError(f"invalid grading method: {grading_method!r}")
        required = int(parts[2])
        distinct_predicted = set(predicted_keys)
        size_ok = len(distinct_predicted) == required if parts[1] == "=" else len(distinct_predicted) >= required
        return size_ok and distinct_predicted.issubset(set(gold_keys))
    raise ValueError(f"unknown grading method: {grading_method!r}")


def execute_sql(
    db_path: str | Path,
    sql: str,
    *,
    timeout_seconds: float = 5.0,
    float_digits: int = 10,
) -> ExecutionResult:
    try:
        validate_read_only_sql(sql)
    except UnsafeSQLError as exc:
        return ExecutionResult(False, frozenset(), str(exc), False)

    path = Path(db_path).resolve()
    if not path.is_file():
        return ExecutionResult(False, frozenset(), f"database not found: {path}", False)

    deadline = time.monotonic() + timeout_seconds
    connection: sqlite3.Connection | None = None
    try:
        uri = f"file:{path.as_posix()}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True, timeout=timeout_seconds)
        connection.execute("PRAGMA query_only = ON")

        def progress_handler() -> int:
            return 1 if time.monotonic() > deadline else 0

        connection.set_progress_handler(progress_handler, 1000)
        rows = connection.execute(sql).fetchall()
        ordered_rows = normalize_ordered_rows(rows, float_digits)
        return ExecutionResult(True, frozenset(ordered_rows), None, False, ordered_rows)
    except sqlite3.Error as exc:
        timed_out = time.monotonic() > deadline or "interrupted" in str(exc).lower()
        return ExecutionResult(False, frozenset(), str(exc), timed_out)
    finally:
        if connection is not None:
            connection.close()


@lru_cache(maxsize=16_384)
def execute_gold_cached(
    db_path: str,
    sql: str,
    timeout_seconds: float = 5.0,
    float_digits: int = 10,
) -> ExecutionResult:
    return execute_sql(db_path, sql, timeout_seconds=timeout_seconds, float_digits=float_digits)


def compare_execution(
    db_path: str | Path,
    predicted_sql: str,
    gold_sql: str,
    *,
    timeout_seconds: float = 5.0,
    gold_timeout_seconds: float | None = None,
    float_digits: int = 10,
    grading_method: str = "set",
) -> tuple[bool, ExecutionResult, ExecutionResult]:
    gold_timeout = timeout_seconds if gold_timeout_seconds is None else gold_timeout_seconds
    gold = execute_gold_cached(str(Path(db_path).resolve()), gold_sql, gold_timeout, float_digits)
    predicted = execute_sql(db_path, predicted_sql, timeout_seconds=timeout_seconds, float_digits=float_digits)
    # A failed prediction never matches a failed/empty gold execution.
    exact = bool(
        gold.ok
        and predicted.ok
        and compare_result_rows(gold.ordered_rows, predicted.ordered_rows, grading_method)
    )
    return exact, predicted, gold
