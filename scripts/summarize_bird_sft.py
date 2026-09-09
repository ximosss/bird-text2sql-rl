#!/usr/bin/env python3
"""Summarize fixed-validation BIRD SFT checkpoints from Prime-RL metrics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


SOURCE = "bird-platinum-validation"
PREFIX = f"eval/{SOURCE}/all/agent/"
METRICS = {
    "exact": PREFIX + "metrics/exact_execution/mean",
    "executable": PREFIX + "metrics/executable_sql/mean",
    "format": PREFIX + "metrics/format_valid/mean",
    "evidence": PREFIX + "metrics/evidence_process_valid/mean",
    "truncated": PREFIX + "is_truncated/mean",
    "val_loss": "val/loss",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--goal-pp", type=float, default=4.0)
    return parser.parse_args()


def load_steps(metrics_path: Path) -> dict[int, dict[str, float]]:
    steps: dict[int, dict[str, float]] = {}
    with metrics_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {metrics_path}:{line_number}") from exc
            raw_step = row.get("step")
            if raw_step is None:
                continue
            step = int(raw_step)
            values = steps.setdefault(step, {})
            for name, key in METRICS.items():
                value = row.get(key)
                if isinstance(value, int | float):
                    values[name] = float(value)
    return steps


def percent(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.2f}"


def load_task_exact(run_dir: Path, step: int) -> dict[str, float]:
    path = run_dir / "rollouts" / f"step_{step}" / "eval" / "all" / "traces.jsonl"
    if not path.is_file():
        return {}
    values: dict[str, list[float]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = str(row["task"]["data"]["example_id"])
            for trace in row["traces"]:
                values[key].append(float(trace["metrics"]["exact_execution"]))
    return {key: sum(scores) / len(scores) for key, scores in values.items()}


def paired_counts(baseline: dict[str, float], candidate: dict[str, float]) -> tuple[int, int] | None:
    if not baseline or baseline.keys() != candidate.keys():
        return None
    gains = sum(candidate[key] > baseline[key] for key in baseline)
    losses = sum(candidate[key] < baseline[key] for key in baseline)
    return gains, losses


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    steps = load_steps(run_dir / "metrics.jsonl")
    baseline = steps.get(0, {}).get("exact")
    if baseline is None:
        raise SystemExit("step-0 execution baseline is not complete")

    # Prime-RL's fresh trainer progress starts at step 1, while the online
    # evaluator names the startup policy step 0. The step-1 eval_on_start loss
    # is therefore the teacher-forced Base loss, before the first optimizer update.
    if "val_loss" not in steps[0] and "val_loss" in steps.get(1, {}):
        steps[0]["val_loss"] = steps[1]["val_loss"]

    target = baseline + args.goal_pp / 100
    base_executable = steps[0].get("executable")
    baseline_tasks = load_task_exact(run_dir, 0)
    print(f"baseline={100 * baseline:.2f}% target={100 * target:.2f}% (+{args.goal_pp:.2f} pp)")
    print("step  exact%  delta_pp  paired(g/l)  exec%   format%  evid%   trunc%  val_loss  checkpoint  gate")

    eligible: list[tuple[int, dict[str, float]]] = []
    evaluated: list[tuple[int, dict[str, float]]] = []
    for step, values in sorted(steps.items()):
        exact = values.get("exact")
        if exact is None:
            continue
        evaluated.append((step, values))
        checkpoint = (run_dir / "checkpoints" / f"step_{step}" / "trainer").is_dir() if step else False
        paired = paired_counts(baseline_tasks, load_task_exact(run_dir, step))
        paired_text = "-" if step == 0 or paired is None else f"{paired[0]}/{paired[1]}"
        gate = step > 0 and checkpoint and exact >= target
        gate &= values.get("format", 0.0) >= 0.98
        gate &= values.get("truncated", 1.0) <= 0.02
        if base_executable is not None:
            gate &= values.get("executable", 0.0) >= base_executable - 0.01
        if gate:
            eligible.append((step, values))
        print(
            f"{step:>4}  {percent(exact):>6}  {100 * (exact - baseline):>+8.2f}  {paired_text:>11}  "
            f"{percent(values.get('executable')):>6}  {percent(values.get('format')):>7}  "
            f"{percent(values.get('evidence')):>6}  "
            f"{percent(values.get('truncated')):>6}  "
            f"{values.get('val_loss', float('nan')):>8.4f}  "
            f"{'yes' if checkpoint else 'no':>10}  {'PASS' if gate else '-'}"
        )

    if eligible:
        # Accuracy is primary; an exact tie selects the earlier checkpoint.
        best_step, best_values = max(eligible, key=lambda item: (item[1]["exact"], -item[0]))
        print(f"selected=step_{best_step} exact={100 * best_values['exact']:.2f}% delta={100 * (best_values['exact'] - baseline):+.2f}pp")
    else:
        print("selected=none (goal and safety gates not yet satisfied)")

    best_so_far = -1.0
    declines = 0
    overfit = False
    for step, values in evaluated:
        if step == 0:
            continue
        exact = values["exact"]
        if exact > best_so_far:
            best_so_far = exact
            declines = 0
        elif best_so_far - exact >= 0.02:
            declines += 1
            overfit |= declines >= 2
        else:
            declines = 0
    print(f"overfit_warning={'yes' if overfit else 'no'}")


if __name__ == "__main__":
    main()
