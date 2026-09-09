from __future__ import annotations

import importlib.util
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

from bird_text2sql.executor import deserialize_ordered_rows


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_rlvr_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_rlvr_data", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_verified_builder_preserves_grading_and_duplicate_rows(tmp_path: Path) -> None:
    database_root = tmp_path / "databases"
    db_dir = database_root / "db"
    db_dir.mkdir(parents=True)
    connection = sqlite3.connect(db_dir / "db.sqlite")
    connection.executescript("CREATE TABLE t(v INTEGER); INSERT INTO t VALUES (1), (1), (2);")
    connection.close()
    rows = [
        {
            "question_id": 7,
            "db_id": "db",
            "question": "List values.",
            "evidence": "Return duplicate rows;",
            "SQL": "SELECT v FROM t ORDER BY rowid",
            "grading_method": "multiset",
        }
    ]

    accepted, rejected = MODULE.prepare_split(
        rows,
        database_root,
        split="train",
        timeout_seconds=5.0,
        workers=1,
    )

    assert not rejected
    assert accepted[0]["grading_method"] == "multiset"
    assert deserialize_ordered_rows(accepted[0]["gold_rows_json"]) == ((1,), (1,), (2,))


def test_cli_spawn_workers_build_train_and_validation(tmp_path: Path) -> None:
    database_root = tmp_path / "databases"
    db_dir = database_root / "db"
    db_dir.mkdir(parents=True)
    connection = sqlite3.connect(db_dir / "db.sqlite")
    connection.executescript("CREATE TABLE t(v INTEGER); INSERT INTO t VALUES (1), (2);")
    connection.close()
    train_path = tmp_path / "train.json"
    validation_path = tmp_path / "validation.json"
    train_path.write_text(
        json.dumps([{"question_id": 1, "db_id": "db", "question": "sum", "SQL": "SELECT SUM(v) FROM t"}]),
        encoding="utf-8",
    )
    validation_path.write_text(
        json.dumps([{"question_id": 2, "db_id": "db", "question": "count", "SQL": "SELECT COUNT(*) FROM t"}]),
        encoding="utf-8",
    )
    output = tmp_path / "output"

    subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--verified-train",
            str(train_path),
            "--verified-validation",
            str(validation_path),
            "--database-root",
            str(database_root),
            "--output-dir",
            str(output),
            "--workers",
            "2",
        ],
        check=True,
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["train"] == 1
    assert manifest["counts"]["validation"] == 1


def test_oversized_gold_result_is_rejected(tmp_path: Path) -> None:
    database_root = tmp_path / "databases"
    db_dir = database_root / "db"
    db_dir.mkdir(parents=True)
    connection = sqlite3.connect(db_dir / "db.sqlite")
    connection.executescript("CREATE TABLE t(v INTEGER); INSERT INTO t VALUES (1), (2);")
    connection.close()

    accepted, rejected = MODULE.prepare_split(
        [{"question_id": 1, "db_id": "db", "question": "values", "SQL": "SELECT v FROM t"}],
        database_root,
        split="train",
        timeout_seconds=5.0,
        workers=1,
        max_gold_rows=1,
    )

    assert not accepted
    assert rejected[0]["reason"] == "gold_result_too_large"
