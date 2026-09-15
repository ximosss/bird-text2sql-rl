"""Aggregate execution-result self-consistency from Verifiers v1 traces."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_groups(path: Path) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            example_id = str(record["task"]["data"]["example_id"])
            traces = record.get("traces") or []
            if not traces:
                raise ValueError(f"{path}:{line_number}: record has no traces")
            for trace in traces:
                score = trace.get("info", {}).get("bird_score")
                if not isinstance(score, dict):
                    raise ValueError(f"{path}:{line_number}: trace has no bird_score")
                groups[example_id].append(
                    {
                        "trace_id": str(trace["id"]),
                        "exact": bool(score.get("exact")),
                        "executable": bool(score.get("executable")),
                        "format_valid": bool(score.get("format_valid")),
                        "result_hash": score.get("predicted_result_hash"),
                    }
                )
    return dict(groups)


def summarize(
    groups: dict[str, list[dict[str, Any]]],
    *,
    expected_rollouts: int,
    seed: int,
) -> dict[str, Any]:
    bad_counts = {
        example_id: len(candidates)
        for example_id, candidates in groups.items()
        if len(candidates) != expected_rollouts
    }
    if bad_counts:
        preview = ", ".join(f"{key}={value}" for key, value in sorted(bad_counts.items())[:5])
        raise ValueError(
            f"expected {expected_rollouts} candidates per example; mismatches: {preview}"
        )

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    all_candidates = [candidate for candidates in groups.values() for candidate in candidates]
    tie_count = 0

    for example_id in sorted(groups):
        candidates = groups[example_id]
        signatures: list[str] = []
        members: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in candidates:
            result_hash = candidate["result_hash"]
            # Failed executions do not share an execution result and therefore
            # must not form an artificial majority bloc.
            signature = (
                f"result:{result_hash}"
                if candidate["executable"] and result_hash
                else f"invalid:{candidate['trace_id']}"
            )
            signatures.append(signature)
            members[signature].append(candidate)

        counts = Counter(signatures)
        winning_votes = max(counts.values())
        winners = sorted(key for key, count in counts.items() if count == winning_votes)
        if len(winners) > 1:
            tie_count += 1
        winning_signature = rng.choice(winners)
        representative = rng.choice(members[winning_signature])
        selected.append(
            {
                "example_id": example_id,
                "trace_id": representative["trace_id"],
                "exact": representative["exact"],
                "executable": representative["executable"],
                "format_valid": representative["format_valid"],
                "winning_votes": winning_votes,
                "num_tied_results": len(winners),
            }
        )

    num_tasks = len(selected)
    sc_exact = sum(item["exact"] for item in selected)
    pass_at_k = sum(any(candidate["exact"] for candidate in groups[key]) for key in groups)
    return {
        "protocol": "execution-result-majority",
        "temperature": 1.0,
        "num_tasks": num_tasks,
        "num_rollouts": expected_rollouts,
        "total_candidates": len(all_candidates),
        "seed": seed,
        "self_consistency": {
            "exact": sc_exact,
            "accuracy": sc_exact / num_tasks if num_tasks else 0.0,
            "executable": sum(item["executable"] for item in selected),
            "format_valid": sum(item["format_valid"] for item in selected),
            "ties": tie_count,
            "mean_winning_votes": (
                sum(item["winning_votes"] for item in selected) / num_tasks
                if num_tasks
                else 0.0
            ),
        },
        "candidate_mean": {
            "exact": sum(item["exact"] for item in all_candidates) / len(all_candidates),
            "executable": sum(item["executable"] for item in all_candidates)
            / len(all_candidates),
            "format_valid": sum(item["format_valid"] for item in all_candidates)
            / len(all_candidates),
        },
        "pass_at_k": {
            "exact": pass_at_k,
            "accuracy": pass_at_k / num_tasks if num_tasks else 0.0,
        },
        "selections": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", type=Path)
    parser.add_argument("--expected-rollouts", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.expected_rollouts < 1:
        parser.error("--expected-rollouts must be positive")

    summary = summarize(
        load_groups(args.traces),
        expected_rollouts=args.expected_rollouts,
        seed=args.seed,
    )
    output = args.output or args.traces.with_name("sc-summary.json")
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    result = summary["self_consistency"]
    print(
        f"SC-{summary['num_rollouts']}: {result['exact']}/{summary['num_tasks']} "
        f"= {result['accuracy']:.2%}"
    )
    print(f"Summary: {output}")
