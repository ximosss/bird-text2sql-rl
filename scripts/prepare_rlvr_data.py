#!/usr/bin/env python3
"""Build the verified BIRD-Platinum task data used by direct RLVR."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from sqlglot import exp, parse_one

from bird_text2sql.data import gold_sql, normalized_question, read_records
from bird_text2sql.executor import execute_sql, serialize_ordered_rows
from bird_text2sql.schema import resolve_db_path


CONTRACT_VERSION = "bird-rlvr-v2"


def validate_gold(payload: tuple[dict[str, Any], str, float]):
    row, database_root, timeout_seconds = payload
    db_path = resolve_db_path(Path(database_root), str(row["db_id"]))
    return execute_sql(db_path, gold_sql(row), timeout_seconds=timeout_seconds)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sql_difficulty(sql: str) -> str:
    statement = parse_one(sql, read="sqlite")
    joins = statement.find(exp.Join) is not None
    nested = sum(1 for _ in statement.find_all(exp.Select)) > 1
    set_op = any(statement.find(kind) is not None for kind in (exp.Union, exp.Intersect, exp.Except))
    if joins and nested:
        return "join_and_nested"
    if set_op:
        return "set_operation"
    if nested:
        return "nested"
    if joins:
        return "join"
    return "single_table"


def stable_id(row: dict[str, Any]) -> str:
    source_id = row.get("question_id", row.get("id", ""))
    payload = f"{row['db_id']}\t{source_id}\t{normalized_question(str(row['question']))}"
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_split(
    rows: list[dict[str, Any]],
    database_root: Path,
    *,
    split: str,
    timeout_seconds: float,
    workers: int,
    max_gold_rows: int = 100_000,
    max_gold_bytes: int = 2_000_000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payloads = ((row, str(database_root), timeout_seconds) for row in rows)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def consume(results) -> None:
        for position, (row, result) in enumerate(zip(rows, results, strict=True), start=1):
            if position % 100 == 0 or position == len(rows):
                print(f"[{split}] validated {position}/{len(rows)}", file=sys.stderr, flush=True)
            example_id = f"bird-platinum-{split}:{stable_id(row)}"
            if example_id in seen:
                rejected.append({"split": split, "reason": "duplicate", "id": example_id})
                continue
            seen.add(example_id)
            if not result.ok:
                rejected.append(
                    {
                        "split": split,
                        "reason": "gold_invalid_or_timeout",
                        "id": example_id,
                        "error": result.error,
                    }
                )
                continue
            if not result.ordered_rows:
                rejected.append({"split": split, "reason": "empty_gold_result", "id": example_id})
                continue
            gold_rows_json = serialize_ordered_rows(result.ordered_rows)
            encoded_bytes = len(gold_rows_json.encode("utf-8"))
            if len(result.ordered_rows) > max_gold_rows or encoded_bytes > max_gold_bytes:
                rejected.append(
                    {
                        "split": split,
                        "reason": "gold_result_too_large",
                        "id": example_id,
                        "rows": len(result.ordered_rows),
                        "bytes": encoded_bytes,
                    }
                )
                continue
            grading_method = str(row.get("grading_method") or "set").strip()
            accepted.append(
                {
                    "id": example_id,
                    "question_id": row.get("question_id"),
                    "db_id": str(row["db_id"]),
                    "question": str(row["question"]),
                    "evidence": str(row.get("evidence") or ""),
                    "SQL": gold_sql(row),
                    "grading_method": grading_method,
                    "gold_rows_json": gold_rows_json,
                    "difficulty": sql_difficulty(gold_sql(row)),
                }
            )
    if workers == 1:
        consume(map(validate_gold, payloads))
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            consume(executor.map(validate_gold, payloads, chunksize=1))
    accepted.sort(key=lambda row: row["id"])
    rejected.sort(key=lambda row: row["id"])
    return accepted, rejected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verified-train", type=Path, required=True)
    parser.add_argument("--verified-validation", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gold-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-gold-rows", type=int, default=100_000)
    parser.add_argument("--max-gold-bytes", type=int, default=2_000_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.max_gold_rows < 1 or args.max_gold_bytes < 1:
        raise ValueError("worker and gold-result limits must be positive")
    paths = {
        "train": args.output_dir / "train.jsonl",
        "validation": args.output_dir / "validation.jsonl",
        "rejections": args.output_dir / "rejections.jsonl",
    }
    existing = [path for path in (*paths.values(), args.output_dir / "manifest.json") if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"output already exists; pass --overwrite: {existing[0]}")

    train_source = read_records(args.verified_train)
    validation_source = read_records(args.verified_validation)
    train, train_rejected = prepare_split(
        train_source,
        args.database_root,
        split="train",
        timeout_seconds=args.gold_timeout_seconds,
        workers=args.workers,
        max_gold_rows=args.max_gold_rows,
        max_gold_bytes=args.max_gold_bytes,
    )
    validation, validation_rejected = prepare_split(
        validation_source,
        args.database_root,
        split="validation",
        timeout_seconds=args.gold_timeout_seconds,
        workers=args.workers,
        max_gold_rows=args.max_gold_rows,
        max_gold_bytes=args.max_gold_bytes,
    )
    train_fingerprints = {(row["db_id"], normalized_question(row["question"])) for row in train}
    validation_fingerprints = {(row["db_id"], normalized_question(row["question"])) for row in validation}
    overlap = train_fingerprints & validation_fingerprints
    if overlap:
        raise ValueError(f"train/validation leakage detected for {len(overlap)} questions")

    rejections = train_rejected + validation_rejected
    write_jsonl(paths["train"], train)
    write_jsonl(paths["validation"], validation)
    write_jsonl(paths["rejections"], rejections)
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "sources": {
            "verified_train": {"path": str(args.verified_train.resolve()), "sha256": file_sha256(args.verified_train)},
            "verified_validation": {
                "path": str(args.verified_validation.resolve()),
                "sha256": file_sha256(args.verified_validation),
            },
        },
        "database_root": str(args.database_root.resolve()),
        "execution_policy": {
            "gold_timeout_seconds": args.gold_timeout_seconds,
            "empty_gold_results_excluded": True,
            "ordered_rows_and_duplicates_preserved": True,
            "grading_method_preserved": True,
            "max_gold_rows": args.max_gold_rows,
            "max_gold_bytes": args.max_gold_bytes,
        },
        "counts": {
            "train_source": len(train_source),
            "validation_source": len(validation_source),
            "train": len(train),
            "validation": len(validation),
            "rejected": len(rejections),
        },
        "train_difficulty_counts": dict(sorted(Counter(row["difficulty"] for row in train).items())),
        "rejection_counts": dict(sorted(Counter(row["reason"] for row in rejections).items())),
        "files": {
            name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
