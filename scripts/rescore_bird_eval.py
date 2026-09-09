#!/usr/bin/env python3
"""Recompute BIRD reward metrics without changing saved model completions."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from bird_text2sql.rewards import _score_once


METRIC_NAMES = (
    "execution_reward",
    "exact_execution",
    "executable_sql",
    "format_valid",
    "execution_timeout",
)


async def rescore_row(row: dict, *, timeout_seconds: float, gold_timeout_seconds: float) -> dict:
    score = await _score_once(
        completion=row["completion"],
        answer=row["answer"],
        info=row["info"],
        state={},
        timeout_seconds=timeout_seconds,
        gold_timeout_seconds=gold_timeout_seconds,
        float_digits=10,
    )
    reward = 1.0 if score["exact"] else 0.1 if score["executable"] else 0.0
    values = {
        "execution_reward": reward,
        "exact_execution": float(score["exact"]),
        "executable_sql": float(score["executable"]),
        "format_valid": float(score["format_valid"]),
        "execution_timeout": float(score["timeout"]),
    }
    updated = dict(row)
    updated["reward"] = reward
    updated_metrics = dict(updated.get("metrics") or {})
    updated_metrics.update(values)
    updated["metrics"] = updated_metrics
    updated.update(values)
    updated["local_rescore"] = {
        "score": score,
        "source": "scripts/rescore_bird_eval.py",
    }
    return updated


async def main_async() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--gold-timeout-seconds", type=float, default=30.0)
    args = parser.parse_args()

    for source in args.results:
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        rescored = [
            await rescore_row(
                row,
                timeout_seconds=args.timeout_seconds,
                gold_timeout_seconds=args.gold_timeout_seconds,
            )
            for row in rows
        ]
        output = source.with_name("results.rescored.jsonl")
        output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rescored),
            encoding="utf-8",
        )
        changed = sum(
            any(float(before.get("metrics", {}).get(name, 0.0)) != float(after["metrics"][name]) for name in METRIC_NAMES)
            for before, after in zip(rows, rescored)
        )
        print(json.dumps({"source": str(source), "output": str(output), "rows": len(rows), "changed": changed}))


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
