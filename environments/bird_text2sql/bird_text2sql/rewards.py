from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

import verifiers as vf

from .executor import compare_execution, deserialize_rows, execute_sql
from .parser import parse_completion


_SQL_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="bird-sql")


async def _run_sql(func, /, *args, **kwargs):
    return await asyncio.get_running_loop().run_in_executor(
        _SQL_EXECUTOR, partial(func, *args, **kwargs)
    )


def _info_dict(info: Any) -> dict[str, Any]:
    if isinstance(info, dict):
        return info
    if isinstance(info, str):
        parsed = json.loads(info)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("info must contain db_path metadata")


async def _compute_score(
    *,
    completion: Any,
    answer: str,
    info: Any,
    state: vf.State,
    timeout_seconds: float,
    gold_timeout_seconds: float,
    float_digits: int,
) -> dict[str, Any]:
    parsed = parse_completion(completion)
    result: dict[str, Any] = {
        "format_valid": parsed.format_valid,
        "sql_present": parsed.sql is not None,
        "executable": False,
        "exact": False,
        "timeout": False,
        "gold_valid": False,
        "parser_source": parsed.source,
    }
    if parsed.sql is not None:
        metadata = _info_dict(info)
        expected_json = metadata.get("gold_result_json")
        if isinstance(expected_json, str) and expected_json:
            predicted = await _run_sql(
                execute_sql,
                metadata["db_path"],
                parsed.sql,
                timeout_seconds=timeout_seconds,
                float_digits=float_digits,
            )
            exact = bool(predicted.ok and predicted.rows == deserialize_rows(expected_json))
            gold_ok, gold_error = True, None
        else:
            exact, predicted, gold = await _run_sql(
                compare_execution,
                metadata["db_path"],
                parsed.sql,
                answer,
                timeout_seconds=timeout_seconds,
                gold_timeout_seconds=gold_timeout_seconds,
                float_digits=float_digits,
            )
            gold_ok, gold_error = gold.ok, gold.error
        result.update(
            executable=predicted.ok,
            exact=exact,
            timeout=predicted.timed_out,
            gold_valid=gold_ok,
            predicted_error=predicted.error,
            gold_error=gold_error,
        )
    return result


async def _score_once(
    *,
    completion: Any,
    answer: str,
    info: Any,
    state: vf.State,
    timeout_seconds: float,
    gold_timeout_seconds: float,
    float_digits: int,
) -> dict[str, Any]:
    cached = state.get("_bird_score")
    if isinstance(cached, dict):
        return cached
    pending = state.get("_bird_score_task")
    if not isinstance(pending, asyncio.Task):
        pending = asyncio.create_task(
            _compute_score(
                completion=completion,
                answer=answer,
                info=info,
                state=state,
                timeout_seconds=timeout_seconds,
                gold_timeout_seconds=gold_timeout_seconds,
                float_digits=float_digits,
            )
        )
        state["_bird_score_task"] = pending
    result = await pending
    state.pop("_bird_score_task", None)
    state["_bird_score"] = result
    return result


def build_rubric(
    *, timeout_seconds: float = 5.0, gold_timeout_seconds: float = 30.0, float_digits: int = 10
) -> vf.Rubric:
    async def execution_reward(completion, answer, info, state, **kwargs) -> float:
        """Binary execution reward: exact=1, every other outcome=0."""
        score = await _score_once(
            completion=completion,
            answer=answer,
            info=info,
            state=state,
            timeout_seconds=timeout_seconds,
            gold_timeout_seconds=gold_timeout_seconds,
            float_digits=float_digits,
        )
        return float(score["exact"])

    async def exact_execution(completion, answer, info, state, **kwargs) -> float:
        score = await _score_once(
            completion=completion,
            answer=answer,
            info=info,
            state=state,
            timeout_seconds=timeout_seconds,
            gold_timeout_seconds=gold_timeout_seconds,
            float_digits=float_digits,
        )
        return float(score["exact"])

    async def executable_sql(completion, answer, info, state, **kwargs) -> float:
        score = await _score_once(
            completion=completion,
            answer=answer,
            info=info,
            state=state,
            timeout_seconds=timeout_seconds,
            gold_timeout_seconds=gold_timeout_seconds,
            float_digits=float_digits,
        )
        return float(score["executable"])

    async def format_valid(completion, answer, info, state, **kwargs) -> float:
        score = await _score_once(
            completion=completion,
            answer=answer,
            info=info,
            state=state,
            timeout_seconds=timeout_seconds,
            gold_timeout_seconds=gold_timeout_seconds,
            float_digits=float_digits,
        )
        return float(score["format_valid"])

    async def execution_timeout(completion, answer, info, state, **kwargs) -> float:
        score = await _score_once(
            completion=completion,
            answer=answer,
            info=info,
            state=state,
            timeout_seconds=timeout_seconds,
            gold_timeout_seconds=gold_timeout_seconds,
            float_digits=float_digits,
        )
        return float(score["timeout"])

    rubric = vf.Rubric(funcs=[execution_reward], weights=[1.0])
    rubric.add_metric(exact_execution)
    rubric.add_metric(executable_sql)
    rubric.add_metric(format_valid)
    rubric.add_metric(execution_timeout)
    return rubric
