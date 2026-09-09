from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "summarize_bird_sft.py"
SPEC = importlib.util.spec_from_file_location("summarize_bird_sft", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_load_steps_merges_trainer_and_eval_records(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    rows = [
        {"step": 0, "val/loss": 1.2},
        {
            "step": 0,
            "eval/bird-platinum-validation/all/agent/metrics/exact_execution/mean": 0.55,
            "eval/bird-platinum-validation/all/agent/metrics/executable_sql/mean": 0.91,
        },
        {"step": 10, "val/loss": 0.8},
        {
            "step": 10.0,
            "eval/bird-platinum-validation/all/agent/metrics/exact_execution/mean": 0.60,
            "eval/bird-platinum-validation/all/agent/metrics/format_valid/mean": 0.99,
            "eval/bird-platinum-validation/all/agent/metrics/evidence_process_valid/mean": 0.72,
            "eval/bird-platinum-validation/all/agent/is_truncated/mean": 0.01,
        },
    ]
    metrics.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    steps = MODULE.load_steps(metrics)

    assert steps[0] == {"exact": 0.55, "executable": 0.91, "val_loss": 1.2}
    assert steps[10] == {
        "exact": 0.60,
        "format": 0.99,
        "evidence": 0.72,
        "truncated": 0.01,
        "val_loss": 0.8,
    }


def test_load_task_exact_and_paired_counts(tmp_path: Path) -> None:
    trace_dir = tmp_path / "rollouts" / "step_0" / "eval" / "all"
    trace_dir.mkdir(parents=True)
    rows = [
        {
            "task": {"data": {"example_id": "a"}},
            "traces": [
                {"metrics": {"exact_execution": 1.0}},
                {"metrics": {"exact_execution": 0.0}},
            ],
        },
        {
            "task": {"data": {"example_id": "b"}},
            "traces": [{"metrics": {"exact_execution": 0.0}}],
        },
    ]
    (trace_dir / "traces.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    baseline = MODULE.load_task_exact(tmp_path, 0)

    assert baseline == {"a": 0.5, "b": 0.0}
    assert MODULE.paired_counts(baseline, {"a": 1.0, "b": 0.0}) == (1, 0)
    assert MODULE.paired_counts(baseline, {"a": 0.0}) is None
