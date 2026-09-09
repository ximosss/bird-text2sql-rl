#!/usr/bin/env python3
"""Build deterministic, leakage-safe BIRD data for direct execution RL."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from sqlglot import exp, parse_one

from bird_text2sql.data import gold_sql, normalized_question, read_records
from bird_text2sql.executor import execute_sql, serialize_rows
from bird_text2sql.schema import resolve_db_path


CONTRACT_VERSION = "bird-rl-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/data/ximo/sql-training/bird23_train_filtered.jsonl"),
    )
    parser.add_argument(
        "--database-root",
        type=Path,
        default=Path("/data/ximo/sql-training/train_databases"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/ximo/bird-text2sql-rl/data/rl-v1"),
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--ood-db-count", type=int, default=14)
    parser.add_argument("--eval-id-size", type=int, default=150)
    parser.add_argument("--eval-ood-size", type=int, default=150)
    parser.add_argument("--gold-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(row: dict[str, Any]) -> str:
    return f"{row['db_id']}\t{normalized_question(str(row['question']))}"


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sql_difficulty(sql: str) -> str:
    statement = parse_one(sql, read="sqlite")
    has_join = statement.find(exp.Join) is not None
    has_nested_select = sum(1 for _ in statement.find_all(exp.Select)) > 1
    has_set = any(statement.find(node) is not None for node in (exp.Union, exp.Intersect, exp.Except))
    if has_join and has_nested_select:
        return "join_and_nested"
    if has_set:
        return "set_operation"
    if has_nested_select:
        return "nested"
    if has_join:
        return "join"
    return "single_table"


def balanced_holdout(rows: list[dict[str, Any]], size: int, seed: int, label: str) -> list[dict[str, Any]]:
    by_db: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_db[str(row["db_id"])].append(row)
    for db_id, db_rows in by_db.items():
        db_rows.sort(key=lambda row: stable_hash(f"{CONTRACT_VERSION}:{seed}:{label}:{fingerprint(row)}"))

    selected: list[dict[str, Any]] = []
    db_ids = sorted(by_db)
    while len(selected) < size:
        progressed = False
        for db_id in db_ids:
            if by_db[db_id]:
                selected.append(by_db[db_id].pop(0))
                progressed = True
                if len(selected) == size:
                    break
        if not progressed:
            raise ValueError(f"requested {size} {label} examples, but only {len(selected)} are available")
    return selected


def main() -> None:
    args = parse_args()
    source_rows = read_records(args.source)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        grouped[fingerprint(row)].append(row)

    rejections: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()

    def reject(reason: str, row: dict[str, Any]) -> None:
        rejection_counts[reason] += 1
        rejections.append(
            {"reason": reason, "db_id": row.get("db_id"), "question": row.get("question")}
        )

    candidates: list[dict[str, Any]] = []
    for rows in grouped.values():
        sql_values = {gold_sql(row) for row in rows}
        if len(sql_values) != 1:
            for row in rows:
                reject("duplicate_conflict", row)
            continue
        candidates.append(rows[0])
        for duplicate in rows[1:]:
            reject("duplicate_repeat", duplicate)

    def validate(row: dict[str, Any]):
        db_path = resolve_db_path(args.database_root, str(row["db_id"]))
        return execute_sql(db_path, gold_sql(row), timeout_seconds=args.gold_timeout_seconds)

    valid_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for index, (row, result) in enumerate(zip(candidates, executor.map(validate, candidates)), start=1):
            if not result.ok:
                reject("gold_invalid_or_timeout", row)
                continue
            if not result.rows:
                reject("gold_empty_result", row)
                continue
            key = fingerprint(row)
            valid_rows.append(
                {
                    "id": f"bird-train:{stable_hash(key)[:16]}",
                    "db_id": str(row["db_id"]),
                    "question": str(row["question"]),
                    "evidence": str(row.get("evidence") or ""),
                    "SQL": gold_sql(row),
                    "gold_result_json": serialize_rows(result.rows),
                    "difficulty": sql_difficulty(gold_sql(row)),
                }
            )
            if index % 500 == 0 or index == len(candidates):
                print(f"validated={index}/{len(candidates)} accepted={len(valid_rows)}", flush=True)

    all_db_ids = sorted({str(row["db_id"]) for row in valid_rows})
    if not 0 < args.ood_db_count < len(all_db_ids):
        raise ValueError("ood-db-count must leave at least one train DB and one OOD DB")
    ood_db_ids = set(
        sorted(
            all_db_ids,
            key=lambda db_id: stable_hash(f"{CONTRACT_VERSION}:{args.seed}:db:{db_id}"),
        )[: args.ood_db_count]
    )
    train_db_ids = set(all_db_ids) - ood_db_ids

    id_pool = [row for row in valid_rows if row["db_id"] in train_db_ids]
    ood_pool = [row for row in valid_rows if row["db_id"] in ood_db_ids]
    eval_id = balanced_holdout(id_pool, args.eval_id_size, args.seed, "eval-id")
    eval_ood = balanced_holdout(ood_pool, args.eval_ood_size, args.seed, "eval-ood")
    eval_id_keys = {row["id"] for row in eval_id}
    train_rows = [row for row in id_pool if row["id"] not in eval_id_keys]

    train_rows.sort(key=lambda row: stable_hash(f"{CONTRACT_VERSION}:{args.seed}:train:{row['id']}"))
    eval_id.sort(key=lambda row: row["id"])
    eval_ood.sort(key=lambda row: row["id"])
    rejections.sort(key=lambda row: (str(row.get("db_id")), str(row.get("question")), row["reason"]))

    paths = {
        "train": args.output_dir / "train.jsonl",
        "eval_id": args.output_dir / "eval_id.jsonl",
        "eval_ood": args.output_dir / "eval_ood.jsonl",
        "rejections": args.output_dir / "rejections.jsonl",
    }
    write_jsonl(paths["train"], train_rows)
    write_jsonl(paths["eval_id"], eval_id)
    write_jsonl(paths["eval_ood"], eval_ood)
    write_jsonl(paths["rejections"], rejections)

    manifest = {
        "contract_version": CONTRACT_VERSION,
        "seed": args.seed,
        "source": {"path": str(args.source.resolve()), "sha256": file_sha256(args.source)},
        "database_root": str(args.database_root.resolve()),
        "split_policy": {
            "ood_db_count": args.ood_db_count,
            "ood_db_selection": "lowest SHA256(bird-rl-v1:seed:db:db_id)",
            "eval_sampling": "deterministic balanced round-robin across DBs",
            "validation_fingerprints_excluded_from_train": True,
        },
        "execution_policy": {
            "gold_timeout_seconds": args.gold_timeout_seconds,
            "prediction_timeout_seconds": 5.0,
            "empty_gold_results_excluded": True,
            "gold_results_precomputed": True,
        },
        "db_sets": {"train": sorted(train_db_ids), "ood": sorted(ood_db_ids)},
        "counts": {
            "source": len(source_rows),
            "valid_nonempty": len(valid_rows),
            "train": len(train_rows),
            "eval_id": len(eval_id),
            "eval_ood": len(eval_ood),
        },
        "train_difficulty_counts": dict(sorted(Counter(row["difficulty"] for row in train_rows).items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
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
