from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from bird_text2sql.prompts import structured_answer
from bird_text2sql.taskset import (
    BirdText2SQLData,
    BirdText2SQLTask,
    BirdText2SQLTaskConfig,
)


def make_db(tmp_path: Path) -> Path:
    path = tmp_path / "task.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript("CREATE TABLE t(v INTEGER); INSERT INTO t VALUES (1), (2);")
    connection.close()
    return path


def test_direct_rl_reward_shaping(monkeypatch, tmp_path: Path) -> None:
    async def inline(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    async def refuted(*args, **kwargs):
        return False, None

    monkeypatch.setattr("bird_text2sql.taskset._run_sql", inline)
    monkeypatch.setattr("bird_text2sql.taskset.grade_equivalence", refuted)
    db_path = make_db(tmp_path)
    task = BirdText2SQLTask(
        BirdText2SQLData(
            idx=0,
            prompt=[{"role": "user", "content": "q"}],
            system_prompt="s",
            answer="SELECT SUM(v) FROM t",
            example_id="one",
            db_id="db",
            db_path=str(db_path),
            question="sum",
            question_fingerprint="sum",
            evidence="sum means SUM(v);",
        ),
        BirdText2SQLTaskConfig(shape_reward=True, use_verieql=True),
    )
    trace = SimpleNamespace(
        last_reply=structured_answer(
            "SELECT SUM(v) FROM t",
            reasoning="Compute the sum.",
        ),
        info={},
    )

    async def score_all() -> tuple[float, dict]:
        total = sum(
            [
                await task.execution_reward(trace),
                await task.semantic_equivalence_penalty(trace),
                await task.evidence_process_penalty(trace),
                await task.format_contract_penalty(trace),
            ]
        )
        return total, await task._score(trace)

    total, score = asyncio.run(score_all())
    assert total == pytest.approx(0.65)
    assert score["exact"]
    assert score["semantic_equivalent"] is False
    assert not score["evidence_process_valid"]


def test_verieql_unsupported_is_unknown_not_fatal(monkeypatch, tmp_path: Path) -> None:
    async def inline(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    async def unsupported(*args, **kwargs):
        return None, "NotImplementedError: unsupported query"

    monkeypatch.setattr("bird_text2sql.taskset._run_sql", inline)
    monkeypatch.setattr("bird_text2sql.taskset.grade_equivalence", unsupported)
    db_path = make_db(tmp_path)
    task = BirdText2SQLTask(
        BirdText2SQLData(
            idx=0,
            prompt=[{"role": "user", "content": "q"}],
            system_prompt="s",
            answer="SELECT SUM(v) FROM t",
            example_id="unsupported",
            db_id="db",
            db_path=str(db_path),
            question="sum",
            question_fingerprint="sum",
        ),
        BirdText2SQLTaskConfig(
            shape_reward=True,
            use_verieql=True,
            verieql_required=True,
        ),
    )
    trace = SimpleNamespace(
        last_reply=structured_answer(
            "SELECT SUM(v) FROM t",
            reasoning="Compute the sum.",
        ),
        info={},
    )

    assert asyncio.run(task.execution_reward(trace)) == 1.0
    assert asyncio.run(task.semantic_equivalence_penalty(trace)) == 0.0


def test_strict_format_reward_gates_fallback_exact_match(monkeypatch, tmp_path: Path) -> None:
    async def inline(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("bird_text2sql.taskset._run_sql", inline)
    db_path = make_db(tmp_path)
    task = BirdText2SQLTask(
        BirdText2SQLData(
            idx=0,
            prompt=[{"role": "user", "content": "q"}],
            system_prompt="s",
            answer="SELECT SUM(v) FROM t",
            example_id="strict-format",
            db_id="db",
            db_path=str(db_path),
            question="sum",
            question_fingerprint="sum",
        ),
        BirdText2SQLTaskConfig(
            strict_format_reward=True,
            format_penalty=0.2,
        ),
    )
    trace = SimpleNamespace(
        last_reply="junk before SQL SELECT SUM(v) FROM t",
        info={},
    )

    reward = asyncio.run(task.execution_reward(trace))
    metrics = asyncio.run(task.execution_metrics(trace))

    assert reward == pytest.approx(-0.2)
    assert metrics["exact_execution"] == 1.0
    assert metrics["format_valid"] == 0.0
    assert metrics["exact_but_format_invalid"] == 1.0
    assert metrics["exact_and_format_valid"] == 0.0
    assert metrics["parser_fallback"] == 1.0


def test_strict_format_reward_preserves_valid_exact_match(monkeypatch, tmp_path: Path) -> None:
    async def inline(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("bird_text2sql.taskset._run_sql", inline)
    db_path = make_db(tmp_path)
    task = BirdText2SQLTask(
        BirdText2SQLData(
            idx=0,
            prompt=[{"role": "user", "content": "q"}],
            system_prompt="s",
            answer="SELECT SUM(v) FROM t",
            example_id="valid-format",
            db_id="db",
            db_path=str(db_path),
            question="sum",
            question_fingerprint="sum",
        ),
        BirdText2SQLTaskConfig(
            strict_format_reward=True,
            format_penalty=0.2,
        ),
    )
    trace = SimpleNamespace(
        last_reply="SELECT SUM(v) FROM t",
        assistant_messages=[SimpleNamespace(reasoning_content="</tool_call> reasoning")],
        info={},
    )

    reward = asyncio.run(task.execution_reward(trace))
    metrics = asyncio.run(task.execution_metrics(trace))

    assert reward == 1.0
    assert metrics["exact_and_format_valid"] == 1.0
    assert metrics["exact_but_format_invalid"] == 0.0
    assert metrics["parser_sql_only"] == 1.0
    assert metrics["empty_content"] == 0.0
    assert metrics["content_control_artifact"] == 0.0
    assert metrics["reasoning_present"] == 1.0
    assert metrics["reasoning_control_artifact"] == 1.0
