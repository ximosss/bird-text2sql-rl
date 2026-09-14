from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from bird_text2sql.prompts import REVISQL_SYSTEM_PROMPT
from bird_text2sql.taskset import BirdText2SQLData, BirdText2SQLTask, BirdText2SQLTaskConfig


def make_task(tmp_path: Path) -> BirdText2SQLTask:
    path = tmp_path / "reward.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript("CREATE TABLE t(v INTEGER); INSERT INTO t VALUES (1), (2);")
    connection.close()
    return BirdText2SQLTask(
        BirdText2SQLData(
            idx=0,
            prompt="q",
            system_prompt="s",
            answer="SELECT SUM(v) FROM t",
            example_id="revisql",
            db_id="db",
            db_path=str(path),
            question="sum",
            question_fingerprint="sum",
            evidence="sum means SUM(v); use table t",
        ),
        BirdText2SQLTaskConfig(protocol="revisql", shape_reward=True),
    )


def trace(*texts: str):
    return SimpleNamespace(
        last_reply=texts[-1],
        assistant_messages=[SimpleNamespace(content=text) for text in texts],
        info={},
    )


def test_revisql_prompt_makes_terminal_order_explicit() -> None:
    assert "verification text before opening the final solution block" in REVISQL_SYSTEM_PROMPT
    assert "final SQL inside <solution>...</solution>" in REVISQL_SYSTEM_PROMPT
    assert "do not repeat its closing delimiter or append any other text" in REVISQL_SYSTEM_PROMPT
    assert "on the fifth turn, finalize without calling a tool" in REVISQL_SYSTEM_PROMPT


def test_revisql_strict_format_reward_penalizes_trailing_text(tmp_path: Path) -> None:
    task = make_task(tmp_path)
    task.config.strict_format_reward = True
    task.config.format_penalty = 0.05
    rollout = trace("<solution>SELECT SUM(v) FROM t</solution> trailing")
    assert asyncio.run(task.execution_reward(rollout)) == pytest.approx(-0.05)


def test_revisql_evidence_deadlines_and_reward_arithmetic(monkeypatch, tmp_path: Path) -> None:
    async def inline(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("bird_text2sql.taskset._run_sql", inline)
    task = make_task(tmp_path)
    rollout = trace(
        "I will inspect the schema.",
        "Still drafting.",
        "Requirement of external knowledge 1 (sum): x\n"
        "Requirement of external knowledge 2 (table): y\n"
        "Verification of external knowledge 1 (sum): x\n"
        "Verification of external knowledge 2 (table): y\n"
        "<solution>SELECT SUM(v) FROM t</solution>",
    )

    async def rewards():
        return (
            await task.execution_reward(rollout),
            await task.evidence_process_penalty(rollout),
            await task.format_contract_penalty(rollout),
            await task.execution_metrics(rollout),
        )

    outcome, evidence, fmt, metrics = asyncio.run(rewards())
    assert outcome == 1.0
    assert evidence == pytest.approx(-0.1)  # requirements arrived after turn two
    assert fmt == 0.0  # ReViSQL has no invented format penalty
    assert metrics["evidence_requirement_penalty"] == 1.0
    assert metrics["evidence_verification_penalty"] == 0.0


def test_revisql_one_turn_easy_rule_charges_only_one_penalty(monkeypatch, tmp_path: Path) -> None:
    async def inline(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("bird_text2sql.taskset._run_sql", inline)
    task = make_task(tmp_path)
    rollout = trace("<solution>SELECT SUM(v) FROM t</solution>")
    assert asyncio.run(task.evidence_process_penalty(rollout)) == pytest.approx(-0.1)
    metrics = asyncio.run(task.execution_metrics(rollout))
    assert metrics["evidence_one_turn_penalty"] == 1.0
