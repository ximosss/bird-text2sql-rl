from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_db(root: Path, db_id: str) -> None:
    db_dir = root / db_id
    db_dir.mkdir(parents=True)
    connection = sqlite3.connect(db_dir / f"{db_id}.sqlite")
    connection.executescript("CREATE TABLE nums (v INTEGER); INSERT INTO nums VALUES (1), (2), (3);")
    connection.close()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def run_builder(repo: Path, source: Path, database_root: Path, output: Path) -> dict:
    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "prepare_rl_data.py"),
            "--source",
            str(source),
            "--database-root",
            str(database_root),
            "--output-dir",
            str(output),
            "--ood-db-count",
            "1",
            "--eval-id-size",
            "2",
            "--eval-ood-size",
            "2",
            "--workers",
            "2",
        ],
        cwd=repo,
        check=True,
    )
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))


def test_prepare_rl_data_is_deterministic_and_leakage_safe(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    database_root = tmp_path / "databases"
    rows: list[dict] = []
    queries = [
        ("List values.", "SELECT v FROM nums"),
        ("What is the sum?", "SELECT SUM(v) FROM nums"),
        ("How many rows?", "SELECT COUNT(*) FROM nums"),
    ]
    for db_id in ("db_a", "db_b", "db_c"):
        make_db(database_root, db_id)
        rows.extend(
            {"db_id": db_id, "question": question, "evidence": "", "SQL": sql}
            for question, sql in queries
        )

    source = tmp_path / "bird.jsonl"
    write_jsonl(source, rows)
    output_a = tmp_path / "output_a"
    output_b = tmp_path / "output_b"
    manifest_a = run_builder(repo, source, database_root, output_a)
    manifest_b = run_builder(repo, source, database_root, output_b)

    assert manifest_a["contract_version"] == "bird-rl-v1"
    assert manifest_a["counts"] == {
        "eval_id": 2,
        "eval_ood": 2,
        "source": 9,
        "train": 4,
        "valid_nonempty": 9,
    }
    assert set(manifest_a["db_sets"]["train"]).isdisjoint(manifest_a["db_sets"]["ood"])

    train_rows = read_jsonl(output_a / "train.jsonl")
    eval_id_rows = read_jsonl(output_a / "eval_id.jsonl")
    eval_ood_rows = read_jsonl(output_a / "eval_ood.jsonl")
    assert {row["id"] for row in train_rows}.isdisjoint(row["id"] for row in eval_id_rows)
    assert {row["db_id"] for row in train_rows}.isdisjoint(row["db_id"] for row in eval_ood_rows)
    assert all(row["gold_result_json"] for row in train_rows + eval_id_rows + eval_ood_rows)
    assert all(row["difficulty"] == "single_table" for row in train_rows + eval_id_rows + eval_ood_rows)

    for name in ("train", "eval_id", "eval_ood", "rejections"):
        assert manifest_a["files"][name]["sha256"] == manifest_b["files"][name]["sha256"]
