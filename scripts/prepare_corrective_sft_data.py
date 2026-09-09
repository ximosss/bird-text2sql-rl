#!/usr/bin/env python3
"""Build teacher-CoT error-correction SFT data from fixed Base rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from bird_text2sql.parser import parse_completion
from bird_text2sql.prompts import structured_answer


CONTRACT_VERSION = "bird-sft-corrective-v4"
EXACT_TARGETS = ("teacher-cot-base-sql", "base-completion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--base-traces", type=Path, required=True)
    parser.add_argument("--expected-rollouts", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exact-target", choices=EXACT_TARGETS, default="teacher-cot-base-sql")
    parser.add_argument("--correction-repeat", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.correction_repeat < 1:
        parser.error("--correction-repeat must be at least 1")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            yield row


def task_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["db_id"]), str(row["question_fingerprint"])


def assistant_text(trace: dict[str, Any]) -> str:
    trajectories = trace.get("traces")
    if not isinstance(trajectories, list) or len(trajectories) != 1:
        raise ValueError(f"expected exactly one trajectory for trace {trace.get('id')}")
    nodes = trajectories[0].get("nodes")
    if not isinstance(nodes, list):
        raise ValueError(f"missing nodes for trace {trace.get('id')}")
    for node in reversed(nodes):
        message = node.get("message") if isinstance(node, dict) else None
        if isinstance(message, dict) and message.get("role") == "assistant" and node.get("sampled"):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
    raise ValueError(f"missing sampled assistant content for trace {trace.get('id')}")


def load_base_rollouts(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rollouts: dict[tuple[str, str], dict[str, Any]] = {}
    for trace in read_jsonl(path):
        if not trace.get("ok"):
            raise ValueError(f"failed trace cannot enter corrective SFT: {trace.get('id')}")
        trajectories = trace.get("traces")
        if not isinstance(trajectories, list) or len(trajectories) != 1:
            raise ValueError(f"expected one trajectory for trace {trace.get('id')}")
        trajectory = trajectories[0]
        task_data = trace.get("task", {}).get("data", {})
        key = task_key(task_data)
        if key in rollouts:
            raise ValueError(f"duplicate Base rollout key: {key}")
        metrics = trajectory.get("metrics", {})
        exact = metrics.get("exact_execution")
        if exact not in (0, 0.0, 1, 1.0):
            raise ValueError(f"missing binary exact_execution for {key}")
        format_valid = metrics.get("format_valid")
        if format_valid not in (0, 0.0, 1, 1.0):
            raise ValueError(f"missing binary format_valid for {key}")
        rollouts[key] = {
            "exact": bool(exact),
            "format_valid": bool(format_valid),
            "completion": assistant_text(trace),
            "trace_id": trace.get("id"),
        }
    return rollouts


def hybrid_target(teacher_completion: str, base_completion: str) -> str:
    teacher = parse_completion(teacher_completion)
    base = parse_completion(base_completion)
    if (
        teacher.source != "legacy_contract"
        or teacher.sql is None
        or teacher.reasoning is None
    ):
        raise ValueError("teacher target does not satisfy the four-block contract")
    if base.sql is None:
        raise ValueError("exact Base rollout has no parseable SQL")
    return structured_answer(
        base.sql,
        reasoning=teacher.reasoning,
        requirements=[text for _, text in teacher.requirements],
        verification=[text for _, text in teacher.verification],
    )


def build_train_rows(
    teacher_rows: Iterable[dict[str, Any]],
    validation_keys: set[tuple[str, str]],
    rollouts: dict[tuple[str, str], dict[str, Any]],
    *,
    exact_target: str = "teacher-cot-base-sql",
    correction_repeat: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if exact_target not in EXACT_TARGETS:
        raise ValueError(f"unsupported exact target: {exact_target}")
    if correction_repeat < 1:
        raise ValueError("correction_repeat must be at least 1")
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    counts = {
        "base_completion_replay": 0,
        "teacher_cot_base_sql": 0,
        "teacher_sql_correction": 0,
        "teacher_format_repair": 0,
        "teacher_unmatched": 0,
        "teacher_target_copies": 0,
    }
    for row in teacher_rows:
        key = task_key(row)
        if key in seen:
            raise ValueError(f"duplicate SFT train key: {key}")
        if key in validation_keys:
            raise ValueError(f"train/validation leakage: {key}")
        seen.add(key)

        messages = row.get("messages")
        if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
            raise ValueError(f"malformed SFT messages for {key}")
        teacher_completion = str(messages[-1].get("content") or "")
        rollout = rollouts.get(key)
        updated = json.loads(json.dumps(row, ensure_ascii=False))
        repeats = correction_repeat
        if rollout is None:
            counts["teacher_unmatched"] += 1
            updated["target_policy"] = "teacher_verified_sql_unmatched"
        elif rollout["exact"]:
            if exact_target == "base-completion" and rollout["format_valid"]:
                parsed_base = parse_completion(rollout["completion"])
                if parsed_base.sql is None or parsed_base.source not in {
                    "sql_only",
                    "legacy_contract",
                }:
                    raise ValueError(f"format-valid Base rollout does not parse for {key}")
                updated["messages"][-1]["content"] = rollout["completion"]
                updated["target_policy"] = "base_exact_completion_replay"
                counts["base_completion_replay"] += 1
                repeats = 1
            elif exact_target == "base-completion":
                updated["target_policy"] = "teacher_cot_format_repair"
                counts["teacher_format_repair"] += 1
            else:
                updated["messages"][-1]["content"] = hybrid_target(teacher_completion, rollout["completion"])
                updated["target_policy"] = "teacher_cot_base_exact_sql"
                counts["teacher_cot_base_sql"] += 1
                repeats = 1
            updated["base_trace_id"] = rollout["trace_id"]
        else:
            updated["target_policy"] = "teacher_cot_verified_sql_correction"
            updated["base_trace_id"] = rollout["trace_id"]
            counts["teacher_sql_correction"] += 1
        for repeat_index in range(repeats):
            repeated = json.loads(json.dumps(updated, ensure_ascii=False))
            repeated["target_repeat_index"] = repeat_index
            output.append(repeated)
            if repeated["target_policy"].startswith("teacher_"):
                counts["teacher_target_copies"] += 1

    extra_rollouts = set(rollouts) - seen
    if extra_rollouts:
        raise ValueError(f"Base rollout contains {len(extra_rollouts)} keys outside SFT train")
    return output, counts


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    traces_path = args.base_traces.resolve()
    output_dir = args.output_dir.resolve()
    train_source = source_dir / "train.jsonl"
    validation_source = source_dir / "validation.jsonl"
    manifest_source = source_dir / "manifest.json"
    for required in (train_source, validation_source, manifest_source, traces_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_rows = list(read_jsonl(validation_source))
    validation_keys = {task_key(row) for row in validation_rows}
    if len(validation_keys) != len(validation_rows):
        raise ValueError("duplicate SFT validation key")
    rollouts = load_base_rollouts(traces_path)
    if len(rollouts) != args.expected_rollouts:
        raise ValueError(f"expected {args.expected_rollouts} Base rollouts, found {len(rollouts)}")
    train_rows, policy_counts = build_train_rows(
        read_jsonl(train_source),
        validation_keys,
        rollouts,
        exact_target=args.exact_target,
        correction_repeat=args.correction_repeat,
    )

    train_output = output_dir / "train.jsonl"
    validation_output = output_dir / "validation.jsonl"
    with train_output.open("w", encoding="utf-8") as handle:
        for row in train_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    shutil.copy2(validation_source, validation_output)

    source_manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "strategy": {
            "exact_target": args.exact_target,
            "correction_repeat": args.correction_repeat,
            "description": (
                "replay the selected exact-target policy on Base-solved tasks; use full teacher CoT "
                "and ReViSQL verified SQL for Base errors, unmatched tasks, and Base format repairs"
            ),
        },
        "source_contract_version": source_manifest.get("contract_version"),
        "counts": {
            "train": len(train_rows),
            "unique_train": len(set(task_key(row) for row in train_rows)),
            "validation": len(validation_rows),
            "base_rollouts": len(rollouts),
            **policy_counts,
        },
        "sources": {
            "train": {"path": str(train_source), "sha256": sha256(train_source)},
            "validation": {"path": str(validation_source), "sha256": sha256(validation_source)},
            "base_traces": {"path": str(traces_path), "sha256": sha256(traces_path)},
        },
        "files": {
            "train": {"path": str(train_output), "sha256": sha256(train_output)},
            "validation": {"path": str(validation_output), "sha256": sha256(validation_output)},
        },
        "validation_copied_byte_for_byte": sha256(validation_output) == sha256(validation_source),
        "validation_keys_excluded_from_train": True,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
