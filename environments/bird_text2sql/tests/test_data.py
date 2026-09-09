from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from bird_text2sql.data import iter_dataset_rows
from bird_text2sql.executor import serialize_ordered_rows


def make_fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    database_dir = tmp_path / "databases" / "demo"
    database_dir.mkdir(parents=True)
    database_path = database_dir / "demo.sqlite"
    connection = sqlite3.connect(database_path)
    connection.executescript("CREATE TABLE t(v INTEGER); INSERT INTO t VALUES (1), (2);")
    connection.close()

    sql = "SELECT v FROM t"
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "question_id": 7,
                    "db_id": "demo",
                    "question": "values?",
                    "evidence": "",
                    "SQL": sql,
                }
            ]
        ),
        encoding="utf-8",
    )
    return dataset_path, tmp_path / "databases", sql


def test_gold_cache_injects_precomputed_rows_and_uses_question_id(tmp_path: Path) -> None:
    dataset_path, database_root, sql = make_fixture(tmp_path)
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "7": {
                    "gold_sql_sha256": hashlib.sha256(sql.encode()).hexdigest(),
                    "gold_rows_json": serialize_ordered_rows(((1,), (2,))),
                    "grading_method": "set",
                }
            }
        ),
        encoding="utf-8",
    )

    row = next(iter(iter_dataset_rows(dataset_path, database_root, gold_cache_path=cache_path)))
    assert row["info"]["example_id"] == "7"
    assert row["info"]["gold_rows_json"] == "[[1],[2]]"


def test_gold_cache_rejects_stale_sql(tmp_path: Path) -> None:
    dataset_path, database_root, _ = make_fixture(tmp_path)
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "7": {
                    "gold_sql_sha256": "stale",
                    "gold_rows_json": "[[1],[2]]",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="gold cache SQL hash mismatch"):
        list(iter_dataset_rows(dataset_path, database_root, gold_cache_path=cache_path))
