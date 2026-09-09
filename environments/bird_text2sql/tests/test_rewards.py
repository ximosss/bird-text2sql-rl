from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from bird_text2sql.rewards import build_rubric
from bird_text2sql.executor import serialize_rows
from bird_text2sql.prompts import structured_answer


@pytest.fixture(autouse=True)
def run_sql_inline(monkeypatch):
    """Keep unit tests deterministic in sandboxes that restrict threaded SQLite."""

    async def inline(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("bird_text2sql.rewards._run_sql", inline)


def make_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "reward.sqlite"
    connection = sqlite3.connect(db_path)
    connection.executescript("CREATE TABLE t (v INTEGER); INSERT INTO t VALUES (1), (2);")
    connection.close()
    return db_path


def score(tmp_path: Path, completion: str) -> dict:
    db_path = make_db(tmp_path)
    rubric = build_rubric()
    state = {
        "prompt": [{"role": "user", "content": "q"}],
        "completion": [{"role": "assistant", "content": completion}],
        "answer": "SELECT SUM(v) FROM t",
        "info": {"db_path": str(db_path)},
    }
    asyncio.run(rubric.score_rollout(state))
    return state


def test_binary_reward_exact(tmp_path: Path) -> None:
    state = score(
        tmp_path,
        structured_answer("SELECT SUM(v) FROM t", reasoning="Sum the requested values."),
    )
    assert state["reward"] == 1.0
    assert state["metrics"]["exact_execution"] == 1.0
    assert state["metrics"]["format_valid"] == 0.0


def test_binary_reward_executable_but_wrong(tmp_path: Path) -> None:
    state = score(
        tmp_path,
        structured_answer("SELECT COUNT(*) FROM t", reasoning="Count rows."),
    )
    assert state["reward"] == 0.0
    assert state["metrics"]["executable_sql"] == 1.0
    assert state["metrics"]["exact_execution"] == 0.0


def test_binary_reward_invalid(tmp_path: Path) -> None:
    state = score(tmp_path, structured_answer("DELETE FROM t", reasoning="Delete rows."))
    assert state["reward"] == 0.0
    assert state["metrics"]["executable_sql"] == 0.0


def test_precomputed_gold_result_avoids_reexecuting_bad_answer_sql(tmp_path: Path) -> None:
    db_path = make_db(tmp_path)
    rubric = build_rubric()
    state = {
        "prompt": [{"role": "user", "content": "q"}],
        "completion": [
            {
                "role": "assistant",
                "content": structured_answer(
                    "SELECT SUM(v) FROM t", reasoning="Sum the requested values."
                ),
            }
        ],
        "answer": "SELECT missing FROM missing_table",
        "info": {"db_path": str(db_path), "gold_result_json": serialize_rows(frozenset({(3,)}))},
    }
    asyncio.run(rubric.score_rollout(state))
    assert state["reward"] == 1.0
