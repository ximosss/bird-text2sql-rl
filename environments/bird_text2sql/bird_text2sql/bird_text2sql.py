from __future__ import annotations

from pathlib import Path

import verifiers as vf

from .data import build_dataset
from .prompts import SYSTEM_PROMPT
from .rewards import build_rubric


DEFAULT_TRAIN_PATH = Path("/data/ximo/sql-training/bird23_train_filtered.jsonl")
DEFAULT_TRAIN_DB_ROOT = Path("/data/ximo/sql-training/train_databases")
DEFAULT_EVAL_PATH = Path("/data/ximo/minidev/MINIDEV/mini_dev_sqlite.json")
DEFAULT_EVAL_DB_ROOT = Path("/data/ximo/minidev/MINIDEV/dev_databases")


def load_environment(
    train_path: str = str(DEFAULT_TRAIN_PATH),
    train_database_root: str = str(DEFAULT_TRAIN_DB_ROOT),
    eval_path: str = str(DEFAULT_EVAL_PATH),
    eval_database_root: str = str(DEFAULT_EVAL_DB_ROOT),
    eval_schema_root: str | None = None,
    eval_gold_cache_path: str | None = None,
    train_limit: int | None = None,
    eval_limit: int | None = None,
    train_shuffle_seed: int | None = 17,
    sample_rows: int = 3,
    timeout_seconds: float = 5.0,
    gold_timeout_seconds: float = 30.0,
    float_digits: int = 10,
    max_seq_len: int = 36_864,
    **kwargs,
) -> vf.Environment:
    """Load the no-tools, single-response BIRD Text-to-SQL environment."""
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"unsupported environment arguments: {unexpected}")
    dataset = build_dataset(
        train_path,
        train_database_root,
        limit=train_limit,
        shuffle_seed=train_shuffle_seed,
        sample_rows=sample_rows,
    )
    eval_dataset = build_dataset(
        eval_path,
        eval_database_root,
        schema_root=eval_schema_root,
        gold_cache_path=eval_gold_cache_path,
        limit=eval_limit,
        sample_rows=sample_rows,
    )
    return vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=SYSTEM_PROMPT,
        rubric=build_rubric(
            timeout_seconds=timeout_seconds,
            gold_timeout_seconds=gold_timeout_seconds,
            float_digits=float_digits,
        ),
        max_seq_len=max_seq_len,
    )
