from __future__ import annotations

import sqlite3
from pathlib import Path

import bird_text2sql.executor as executor
from bird_text2sql.executor import (
    compare_execution,
    compare_result_rows,
    deserialize_ordered_rows,
    deserialize_rows,
    execute_sql,
    serialize_ordered_rows,
    serialize_rows,
)


def make_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "fixture.sqlite"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT, score REAL);
        INSERT INTO people VALUES (1, 'Alice', 1.123456789012);
        INSERT INTO people VALUES (2, 'alice', 2.0);
        """
    )
    connection.close()
    return db_path


def test_alias_names_do_not_affect_execution_match(tmp_path: Path) -> None:
    db_path = make_db(tmp_path)
    exact, predicted, gold = compare_execution(
        db_path,
        "SELECT name AS predicted_alias FROM people WHERE id = 1",
        "SELECT name AS gold_alias FROM people WHERE id = 1",
    )
    assert exact and predicted.ok and gold.ok


def test_database_value_case_is_preserved(tmp_path: Path) -> None:
    db_path = make_db(tmp_path)
    exact, _, _ = compare_execution(
        db_path,
        "SELECT name FROM people WHERE id = 2",
        "SELECT name FROM people WHERE id = 1",
    )
    assert not exact


def test_failed_prediction_never_matches_failed_gold(tmp_path: Path) -> None:
    db_path = make_db(tmp_path)
    exact, predicted, gold = compare_execution(
        db_path,
        "SELECT x FROM missing_a",
        "SELECT y FROM missing_b",
    )
    assert not exact
    assert not predicted.ok
    assert not gold.ok


def test_rejects_write_control_and_multiple_statements(tmp_path: Path) -> None:
    db_path = make_db(tmp_path)
    for sql in (
        "DELETE FROM people",
        "PRAGMA table_info(people)",
        "SELECT 1; SELECT 2",
    ):
        result = execute_sql(db_path, sql)
        assert not result.ok, sql


def test_tokenizer_errors_are_scored_as_invalid_sql(tmp_path: Path) -> None:
    db_path = make_db(tmp_path)
    result = execute_sql(db_path, "SELECT 'unterminated")
    assert not result.ok
    assert result.error is not None
    assert result.error.startswith("parse_error:")


def test_parser_recursion_errors_are_scored_as_invalid_sql(tmp_path: Path, monkeypatch) -> None:
    db_path = make_db(tmp_path)

    def raise_recursion_error(*args, **kwargs):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(executor.sqlglot, "parse", raise_recursion_error)
    result = execute_sql(db_path, "SELECT 1")
    assert not result.ok
    assert result.error == "parse_error: maximum recursion depth exceeded"


def test_float_results_are_rounded_for_stable_comparison(tmp_path: Path) -> None:
    db_path = make_db(tmp_path)
    exact, _, _ = compare_execution(
        db_path,
        "SELECT score + 0.000000000001 FROM people WHERE id = 1",
        "SELECT score FROM people WHERE id = 1",
        float_digits=10,
    )
    assert exact


def test_normalized_rows_have_deterministic_json_roundtrip() -> None:
    rows = frozenset({("Alice", 1.25, None, ("bytes", "ff")), ("alice", 2, None, ("float", "nan"))})
    encoded = serialize_rows(rows)
    assert serialize_rows(frozenset(reversed(tuple(rows)))) == encoded
    assert deserialize_rows(encoded) == rows


def test_ordered_rows_roundtrip_preserves_duplicates_and_order() -> None:
    rows = (("b", 2), ("a", 1), ("a", 1))
    assert deserialize_ordered_rows(serialize_ordered_rows(rows)) == rows


def test_verified_grading_methods() -> None:
    gold = ((1, "a"), (2, "b"), (2, "b"))
    reordered = (("b", 2), ("a", 1), ("b", 2))
    assert compare_result_rows(gold, reordered, "multiset")
    assert compare_result_rows(gold, reordered, "set")
    assert not compare_result_rows(gold, tuple(reversed(reordered)), "list")
    assert compare_result_rows(gold, (("a", 1),), "subset,=,1")
    assert not compare_result_rows(gold, (("a", 1),), "subset,=,2")
