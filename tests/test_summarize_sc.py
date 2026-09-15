from __future__ import annotations

from bird_text2sql.self_consistency import summarize


def candidate(trace_id: str, result_hash: str | None, exact: bool) -> dict:
    return {
        "trace_id": trace_id,
        "exact": exact,
        "executable": result_hash is not None,
        "format_valid": result_hash is not None,
        "result_hash": result_hash,
    }


def test_execution_majority_is_not_pass_at_k() -> None:
    groups = {
        "one": [
            candidate("a", "wrong", False),
            candidate("b", "wrong", False),
            candidate("c", "right", True),
        ],
        "two": [
            candidate("d", "right", True),
            candidate("e", "right", True),
            candidate("f", "wrong", False),
        ],
    }

    result = summarize(groups, expected_rollouts=3, seed=0)

    assert result["self_consistency"]["exact"] == 1
    assert result["pass_at_k"]["exact"] == 2


def test_invalid_candidates_do_not_vote_as_one_result() -> None:
    groups = {
        "one": [
            candidate("a", None, False),
            candidate("b", None, False),
            candidate("c", "right", True),
        ]
    }

    result = summarize(groups, expected_rollouts=3, seed=0)

    assert result["self_consistency"]["mean_winning_votes"] == 1
    assert result["self_consistency"]["ties"] == 1
