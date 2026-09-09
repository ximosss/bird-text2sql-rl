from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from pydantic import Field

from .data import iter_dataset_rows
from .executor import (
    compare_execution,
    compare_result_rows,
    deserialize_ordered_rows,
    deserialize_rows,
    execute_sql,
)
from .parser import parse_completion
from .prompts import SYSTEM_PROMPT, evidence_items
from .verieql import grade_equivalence


_SQL_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="bird-v1-sql")


async def _run_sql(func, /, *args, **kwargs):
    return await asyncio.get_running_loop().run_in_executor(
        _SQL_EXECUTOR, partial(func, *args, **kwargs)
    )


class BirdText2SQLData(vf.TaskData):
    answer: str
    example_id: str
    db_id: str
    db_path: str
    question: str
    question_fingerprint: str
    evidence: str = ""
    gold_result_json: str = ""
    gold_rows_json: str = ""
    grading_method: str = "set"


class BirdText2SQLTaskConfig(vf.TaskConfig):
    timeout_seconds: float = Field(5.0, gt=0)
    gold_timeout_seconds: float = Field(30.0, gt=0)
    float_digits: int = Field(10, ge=0)
    shape_reward: bool = False
    use_verieql: bool = False
    verieql_required: bool = False
    verieql_timeout_seconds: float = Field(30.0, gt=0)
    verieql_bound_size: int = Field(2, ge=1)
    equivalence_penalty: float = Field(0.2, ge=0.0, le=1.0)
    evidence_penalty: float = Field(0.1, ge=0.0, le=1.0)
    format_penalty: float = Field(0.05, ge=0.0, le=1.0)


class BirdText2SQLTask(vf.Task[BirdText2SQLData, vf.State, BirdText2SQLTaskConfig]):
    @property
    def key(self) -> str:
        return self.data.example_id

    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    async def _compute_score(self, trace: vf.Trace) -> dict[str, Any]:
        parsed = parse_completion(trace.last_reply)
        result: dict[str, Any] = {
            "format_valid": parsed.format_valid,
            "sql_present": parsed.sql is not None,
            "evidence_items": len(evidence_items(self.data.evidence)),
            "evidence_process_valid": parsed.evidence_process_valid(
                len(evidence_items(self.data.evidence))
            ),
            "executable": False,
            "exact": False,
            "timeout": False,
            "gold_valid": False,
            "semantic_equivalent": None,
            "equivalence_error": None,
            "parser_source": parsed.source,
        }
        if parsed.sql is None:
            return result

        if self.data.gold_rows_json:
            predicted = await _run_sql(
                execute_sql,
                self.data.db_path,
                parsed.sql,
                timeout_seconds=self.config.timeout_seconds,
                float_digits=self.config.float_digits,
            )
            exact = bool(
                predicted.ok
                and compare_result_rows(
                    deserialize_ordered_rows(self.data.gold_rows_json),
                    predicted.ordered_rows,
                    self.data.grading_method,
                )
            )
            gold_ok, gold_error = True, None
        elif self.data.gold_result_json:
            predicted = await _run_sql(
                execute_sql,
                self.data.db_path,
                parsed.sql,
                timeout_seconds=self.config.timeout_seconds,
                float_digits=self.config.float_digits,
            )
            exact = bool(predicted.ok and predicted.rows == deserialize_rows(self.data.gold_result_json))
            gold_ok, gold_error = True, None
        else:
            exact, predicted, gold = await _run_sql(
                compare_execution,
                self.data.db_path,
                parsed.sql,
                self.data.answer,
                timeout_seconds=self.config.timeout_seconds,
                gold_timeout_seconds=self.config.gold_timeout_seconds,
                float_digits=self.config.float_digits,
                grading_method=self.data.grading_method,
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
        if exact and self.config.use_verieql:
            equivalent, equivalence_error = await grade_equivalence(
                self.data.db_path,
                parsed.sql,
                self.data.answer,
                timeout_seconds=self.config.verieql_timeout_seconds,
                bound_size=self.config.verieql_bound_size,
            )
            verifier_unavailable = bool(
                equivalence_error
                and (
                    equivalence_error.startswith("verieql_unavailable:")
                    or equivalence_error.startswith("worker_exit_")
                    or equivalence_error.startswith("invalid_worker_output:")
                )
            )
            if equivalent is None and self.config.verieql_required and verifier_unavailable:
                raise RuntimeError(
                    "VeriEQL was required but its worker is unavailable: "
                    f"{equivalence_error or 'no diagnostic'}"
                )
            result.update(
                semantic_equivalent=equivalent,
                equivalence_error=equivalence_error,
            )
        return result

    async def _score(self, trace: vf.Trace) -> dict[str, Any]:
        cached = trace.info.get("bird_score")
        if isinstance(cached, dict):
            return cached

        pending = trace.info.get("_bird_score_task")
        if not isinstance(pending, asyncio.Task):
            pending = asyncio.create_task(self._compute_score(trace))
            trace.info["_bird_score_task"] = pending
        try:
            score = await pending
        finally:
            trace.info.pop("_bird_score_task", None)
        trace.info["bird_score"] = score
        return score

    @vf.reward(weight=1.0)
    async def execution_reward(self, trace: vf.Trace) -> float:
        return float((await self._score(trace))["exact"])

    @vf.reward(weight=1.0)
    async def semantic_equivalence_penalty(self, trace: vf.Trace) -> float:
        score = await self._score(trace)
        if not self.config.shape_reward:
            return 0.0
        return -self.config.equivalence_penalty if score["semantic_equivalent"] is False else 0.0

    @vf.reward(weight=1.0)
    async def evidence_process_penalty(self, trace: vf.Trace) -> float:
        score = await self._score(trace)
        if not self.config.shape_reward or score["evidence_items"] == 0:
            return 0.0
        return -self.config.evidence_penalty if not score["evidence_process_valid"] else 0.0

    @vf.reward(weight=1.0)
    async def format_contract_penalty(self, trace: vf.Trace) -> float:
        score = await self._score(trace)
        if not self.config.shape_reward:
            return 0.0
        return -self.config.format_penalty if not score["format_valid"] else 0.0

    @vf.metric
    async def execution_metrics(self, trace: vf.Trace) -> dict[str, float]:
        score = await self._score(trace)
        return {
            "exact_execution": float(score["exact"]),
            "executable_sql": float(score["executable"]),
            "format_valid": float(score["format_valid"]),
            "evidence_process_valid": float(score["evidence_process_valid"]),
            "semantic_equivalence_refuted": float(score["semantic_equivalent"] is False),
            "semantic_equivalence_unknown": float(
                self.config.use_verieql and score["exact"] and score["semantic_equivalent"] is None
            ),
            "execution_timeout": float(score["timeout"]),
        }


class BirdText2SQLConfig(vf.TasksetConfig):
    path: str
    database_root: str
    schema_root: str | None = None
    gold_cache_path: str | None = None
    limit: int | None = Field(None, ge=1)
    shuffle_seed: int | None = None
    sample_rows: int = Field(3, ge=0)
    task: BirdText2SQLTaskConfig = BirdText2SQLTaskConfig()


class BirdText2SQLTaskset(vf.Taskset[BirdText2SQLTask, BirdText2SQLConfig]):
    def load(self) -> list[BirdText2SQLTask]:
        tasks: list[BirdText2SQLTask] = []
        for index, row in enumerate(
            iter_dataset_rows(
                Path(self.config.path),
                Path(self.config.database_root),
                schema_root=Path(self.config.schema_root) if self.config.schema_root else None,
                gold_cache_path=(
                    Path(self.config.gold_cache_path) if self.config.gold_cache_path else None
                ),
                limit=self.config.limit,
                shuffle_seed=self.config.shuffle_seed,
                sample_rows=self.config.sample_rows,
            )
        ):
            info = row["info"]
            tasks.append(
                BirdText2SQLTask(
                    BirdText2SQLData(
                        idx=index,
                        prompt=row["prompt"],
                        system_prompt=SYSTEM_PROMPT,
                        answer=row["answer"],
                        example_id=info["example_id"],
                        db_id=info["db_id"],
                        db_path=info["db_path"],
                        question=info["question"],
                        question_fingerprint=info["question_fingerprint"],
                        evidence=info["evidence"],
                        gold_result_json=info["gold_result_json"],
                        gold_rows_json=info["gold_rows_json"],
                        grading_method=info["grading_method"],
                    ),
                    self.config.task,
                )
            )
        return tasks
