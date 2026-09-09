from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

from .prompts import build_user_prompt
from .schema import render_schema, resolve_db_path


def read_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        if source.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"expected a JSON list in {source}")
    return value


def normalized_question(value: str) -> str:
    return " ".join(value.casefold().split())


def gold_sql(row: dict[str, Any]) -> str:
    value = row.get("SQL") or row.get("sql") or row.get("query") or row.get("gold_sql")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("row has no gold SQL in SQL/sql/query/gold_sql")
    return value.strip()


def read_gold_cache(path: str | Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in gold cache {source}")
    return {str(key): item for key, item in value.items()}


def iter_dataset_rows(
    path: str | Path,
    database_root: str | Path,
    *,
    schema_root: str | Path | None = None,
    gold_cache_path: str | Path | None = None,
    limit: int | None = None,
    shuffle_seed: int | None = None,
    sample_rows: int = 3,
) -> Iterable[dict[str, Any]]:
    records = read_records(path)
    gold_cache = read_gold_cache(gold_cache_path)
    if shuffle_seed is not None:
        records = list(records)
        random.Random(shuffle_seed).shuffle(records)
    if limit is not None:
        records = records[:limit]
    for index, row in enumerate(records):
        db_id = str(row["db_id"])
        example_id = str(row.get("question_id", row.get("id", f"{db_id}:{index}")))
        question = str(row["question"])
        evidence = row.get("evidence")
        answer = gold_sql(row)
        cached_gold = gold_cache.get(example_id, {})
        if cached_gold:
            observed_hash = hashlib.sha256(answer.encode()).hexdigest()
            expected_hash = cached_gold.get("gold_sql_sha256")
            if observed_hash != expected_hash:
                raise ValueError(
                    f"gold cache SQL hash mismatch for example {example_id!r}: "
                    f"expected {expected_hash}, observed {observed_hash}"
                )
        db_path = resolve_db_path(database_root, db_id)
        description_db_dir: Path | None = None
        if schema_root is not None:
            description_db_dir = Path(schema_root) / db_id
            if not (description_db_dir / "database_description").is_dir():
                raise FileNotFoundError(
                    f"cannot find schema descriptions for {db_id!r} under {schema_root}"
                )
        schema = render_schema(
            str(db_path),
            sample_rows=sample_rows,
            description_db_dir=str(description_db_dir) if description_db_dir else None,
        )
        yield {
            "prompt": [
                {
                    "role": "user",
                    "content": build_user_prompt(
                        db_id=db_id,
                        question=question,
                        schema=schema,
                        evidence=str(evidence) if evidence else None,
                    ),
                }
            ],
            "answer": answer,
            "info": {
                "example_id": example_id,
                "db_id": db_id,
                "db_path": str(db_path),
                "question": question,
                "question_fingerprint": normalized_question(question),
                "evidence": str(evidence) if evidence else "",
                "gold_result_json": row.get("gold_result_json", ""),
                "gold_rows_json": cached_gold.get(
                    "gold_rows_json", row.get("gold_rows_json", "")
                ),
                "grading_method": str(
                    cached_gold.get("grading_method")
                    or row.get("grading_method")
                    or "set"
                ).strip(),
            },
        }


def build_dataset(
    path: str | Path,
    database_root: str | Path,
    *,
    schema_root: str | Path | None = None,
    gold_cache_path: str | Path | None = None,
    limit: int | None = None,
    shuffle_seed: int | None = None,
    sample_rows: int = 3,
):
    from datasets import Dataset

    return Dataset.from_list(
        list(
            iter_dataset_rows(
                path,
                database_root,
                schema_root=schema_root,
                gold_cache_path=gold_cache_path,
                limit=limit,
                shuffle_seed=shuffle_seed,
                sample_rows=sample_rows,
            )
        )
    )
