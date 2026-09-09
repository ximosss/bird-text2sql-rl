from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_corrective_sft_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_corrective_sft_data", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def completion(sql: str, reasoning: str = "teacher reason") -> str:
    return (
        '<requirements>\n<requirement index="1">teacher requirement</requirement>\n</requirements>\n'
        f"<reasoning>{reasoning}</reasoning>\n"
        '<verification>\n<check index="1">teacher check</check>\n</verification>\n'
        f"<sql>{sql}</sql>"
    )


def row(question: str, sql: str) -> dict:
    return {
        "db_id": "db",
        "question_fingerprint": question,
        "messages": [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": completion(sql)},
        ],
    }


def test_hybrid_target_preserves_teacher_cot_and_base_sql() -> None:
    result = MODULE.hybrid_target(completion("SELECT teacher"), "```sql\nSELECT base\n```")

    assert "teacher reason" in result
    assert "teacher requirement" in result
    assert "<sql>SELECT base</sql>" in result
    assert "SELECT teacher" not in result


def test_build_rows_replays_exact_and_corrects_wrong_or_unmatched() -> None:
    rows = [row("q1", "SELECT gold1"), row("q2", "SELECT gold2"), row("q3", "SELECT gold3")]
    rollouts = {
        ("db", "q1"): {
            "exact": True,
            "format_valid": True,
            "completion": completion("SELECT base1"),
            "trace_id": "t1",
        },
        ("db", "q2"): {
            "exact": False,
            "format_valid": True,
            "completion": completion("SELECT bad2"),
            "trace_id": "t2",
        },
    }

    result, counts = MODULE.build_train_rows(rows, set(), rollouts)

    assert "<sql>SELECT base1</sql>" in result[0]["messages"][-1]["content"]
    assert "<sql>SELECT gold2</sql>" in result[1]["messages"][-1]["content"]
    assert "<sql>SELECT gold3</sql>" in result[2]["messages"][-1]["content"]
    assert counts == {
        "base_completion_replay": 0,
        "teacher_cot_base_sql": 1,
        "teacher_sql_correction": 1,
        "teacher_format_repair": 0,
        "teacher_unmatched": 1,
        "teacher_target_copies": 3,
    }


def test_base_completion_replay_and_repeated_teacher_corrections() -> None:
    rows = [row("q1", "SELECT gold1"), row("q2", "SELECT gold2"), row("q3", "SELECT gold3")]
    base_completion = completion("SELECT base1", reasoning="base reason")
    rollouts = {
        ("db", "q1"): {
            "exact": True,
            "format_valid": True,
            "completion": base_completion,
            "trace_id": "t1",
        },
        ("db", "q2"): {
            "exact": False,
            "format_valid": True,
            "completion": completion("SELECT bad2"),
            "trace_id": "t2",
        },
    }

    result, counts = MODULE.build_train_rows(
        rows,
        set(),
        rollouts,
        exact_target="base-completion",
        correction_repeat=2,
    )

    assert len(result) == 5
    assert result[0]["messages"][-1]["content"] == base_completion
    assert result[0]["target_policy"] == "base_exact_completion_replay"
    assert [item["target_repeat_index"] for item in result] == [0, 0, 1, 0, 1]
    assert counts == {
        "base_completion_replay": 1,
        "teacher_cot_base_sql": 0,
        "teacher_sql_correction": 1,
        "teacher_format_repair": 0,
        "teacher_unmatched": 1,
        "teacher_target_copies": 4,
    }


def test_invalid_exact_base_completion_uses_teacher_format_repair() -> None:
    rows = [row("q1", "SELECT gold1")]
    rollouts = {
        ("db", "q1"): {
            "exact": True,
            "format_valid": False,
            "completion": "```sql\nSELECT base1\n```",
            "trace_id": "t1",
        }
    }

    result, counts = MODULE.build_train_rows(
        rows,
        set(),
        rollouts,
        exact_target="base-completion",
        correction_repeat=2,
    )

    assert len(result) == 2
    assert all("<sql>SELECT gold1</sql>" in item["messages"][-1]["content"] for item in result)
    assert counts["teacher_format_repair"] == 1
    assert counts["teacher_target_copies"] == 2


def test_build_rows_rejects_validation_leakage() -> None:
    with pytest.raises(ValueError, match="leakage"):
        MODULE.build_train_rows([row("q1", "SELECT 1")], {("db", "q1")}, {})
