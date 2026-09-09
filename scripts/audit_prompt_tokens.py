#!/usr/bin/env python3
"""Audit rendered prompt/SFT lengths before selecting seq_len."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from bird_text2sql.data import iter_dataset_rows, read_records
from bird_text2sql.prompts import SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data/qwen3-4b-instruct-2507")
    parser.add_argument("--mode", choices=("sft", "bird"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--database-root", type=Path)
    parser.add_argument("--seq-len", type=int, default=36_864)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    index = min(len(values) - 1, math.ceil(q * len(values)) - 1)
    return sorted(values)[index]


def token_count(rendered: object) -> int:
    if isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    elif hasattr(rendered, "input_ids"):
        rendered = getattr(rendered, "input_ids")
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if isinstance(rendered, list) and rendered and isinstance(rendered[0], list):
        if len(rendered) != 1:
            raise ValueError("expected one rendered conversation")
        rendered = rendered[0]
    if not isinstance(rendered, list):
        raise TypeError(f"unsupported tokenizer output type: {type(rendered)!r}")
    return len(rendered)


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.mode == "sft":
        records = read_records(args.data)
        selected = records[: args.limit]
        examples = [
            (row["messages"], {"db_id": row.get("db_id", ""), "question": row.get("question_fingerprint", "")})
            for row in selected
        ]
    else:
        if args.database_root is None:
            raise ValueError("--database-root is required in bird mode")
        rows = iter_dataset_rows(args.data, args.database_root, limit=args.limit)
        examples = [
            (
                [{"role": "system", "content": SYSTEM_PROMPT}, *row["prompt"]],
                {"db_id": row["info"]["db_id"], "question": row["info"]["question"]},
            )
            for row in rows
        ]

    lengths: list[int] = []
    details: list[dict[str, object]] = []
    for conversation, metadata in examples:
        token_ids = tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=args.mode == "bird",
            enable_thinking=False,
        )
        length = token_count(token_ids)
        lengths.append(length)
        details.append({**metadata, "tokens": length})
    if lengths and max(lengths) < 32:
        raise ValueError("token audit produced implausibly short conversations; check tokenizer output handling")
    overflow_details = [detail for detail in details if int(detail["tokens"]) > args.seq_len]
    overflow_by_db = Counter(str(detail["db_id"]) for detail in overflow_details)
    report = {
        "model": args.model,
        "mode": args.mode,
        "examples": len(lengths),
        "seq_len": args.seq_len,
        "p50": percentile(lengths, 0.50),
        "p90": percentile(lengths, 0.90),
        "p95": percentile(lengths, 0.95),
        "p99": percentile(lengths, 0.99),
        "max": max(lengths, default=0),
        "overflow_count": sum(length > args.seq_len for length in lengths),
        "overflow_by_db": dict(sorted(overflow_by_db.items(), key=lambda item: (-item[1], item[0]))),
        "longest_examples": sorted(details, key=lambda item: int(item["tokens"]), reverse=True)[:20],
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
