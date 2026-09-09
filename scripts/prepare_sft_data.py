#!/usr/bin/env python3
"""Prepare broad and BIRD-alignment CoT datasets for Prime-RL native SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from bird_text2sql.data import normalized_question, read_records
from bird_text2sql.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
    distillation_answer,
)
from bird_text2sql.schema import render_schema, resolve_db_path


CONTRACT_VERSION = "bird-cot-sql-v1"
_THINK_RE = re.compile(r"<think>(?P<body>.*?)</think>", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<(?:answer|sql)>(?P<body>.*?)</(?:answer|sql)>", re.IGNORECASE | re.DOTALL)
_SQL_LINE_RE = re.compile(r"(?ms)^#SQL:.*\Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fraction(value: str) -> float:
    numerator = int(hashlib.sha256(value.encode()).hexdigest()[:16], 16)
    return numerator / float(16**16)


def iter_json_records(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[dict[str, Any]]:
    if path.suffix.casefold() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"expected objects in {path}")
                    yield value
        return

    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer = ""
        position = 0
        started = False
        finished = False
        while not finished:
            chunk = handle.read(chunk_size)
            if chunk:
                buffer = buffer[position:] + chunk
                position = 0
            elif position >= len(buffer):
                break
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if not started:
                    if position >= len(buffer):
                        break
                    if buffer[position] != "[":
                        raise ValueError(f"expected a JSON array in {path}")
                    started = True
                    position += 1
                    continue
                while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
                    position += 1
                if position < len(buffer) and buffer[position] == "]":
                    finished = True
                    position += 1
                    break
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if chunk:
                        break
                    raise
                if not isinstance(value, dict):
                    raise ValueError(f"expected objects in {path}")
                yield value
                position = end
        if not finished:
            raise ValueError(f"unterminated JSON array in {path}")


def _safe_process_text(value: str) -> str:
    return value.strip()


def wide_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    prompt = row.get("input_seq") or row.get("prompt") or row.get("question")
    output = row.get("output_seq") or row.get("completion") or row.get("answer")
    if not isinstance(prompt, str) or not isinstance(output, str):
        raise ValueError("wide row requires string input_seq/prompt and output_seq/completion")
    thinking = _THINK_RE.search(output)
    answer = _ANSWER_RE.search(output)
    if thinking is None or answer is None:
        raise ValueError("wide completion requires <think> and <answer>/<sql> blocks")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt.strip()},
        distillation_answer(
            answer.group("body"),
            reasoning=_safe_process_text(thinking.group("body")),
        ),
    ]


def teacher_reasoning(row: dict[str, Any]) -> str:
    parsed = row.get("parsed")
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except json.JSONDecodeError:
            parsed = None
    if isinstance(parsed, dict):
        labels = (
            ("Analysis", "reason"),
            ("Relevant columns", "columns"),
            ("Question-to-SQL mapping", "select"),
            ("Values", "values"),
            ("Query plan", "sql_like"),
        )
        parts = [
            f"{label}: {str(parsed[key]).strip()}"
            for label, key in labels
            if parsed.get(key) and str(parsed[key]).strip().casefold() != "none"
        ]
        if parts:
            return _safe_process_text("\n".join(parts))
    raw = str(row.get("reasoning") or "")
    raw = _SQL_LINE_RE.sub("", raw).strip()
    if not raw:
        raise ValueError("BIRD CoT row has no teacher reasoning")
    return _safe_process_text(raw)


def bird_messages(
    row: dict[str, Any], database_root: Path, *, sample_rows: int
) -> list[dict[str, str]]:
    evidence = str(row.get("evidence") or "")
    sql = str(row.get("SQL") or row.get("sql") or "").strip()
    if not sql:
        raise ValueError("BIRD row has no SQL")
    schema = render_schema(
        str(resolve_db_path(database_root, str(row["db_id"]))), sample_rows=sample_rows
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(
                db_id=str(row["db_id"]),
                question=str(row["question"]),
                schema=schema,
                evidence=evidence,
            ),
        },
        distillation_answer(sql, reasoning=teacher_reasoning(row)),
    ]


class TokenFilter:
    def __init__(self, model: str | None, max_tokens: int | None):
        self.max_tokens = max_tokens
        self.tokenizer = None
        if max_tokens is not None:
            if model is None:
                raise ValueError("--model is required when --max-tokens is set")
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    def length(self, messages: list[dict[str, str]]) -> int | None:
        if self.tokenizer is None:
            return None
        rendered_messages = []
        for message in messages:
            rendered = dict(message)
            reasoning = rendered.pop("reasoning_content", None)
            if reasoning:
                rendered["content"] = (
                    f"<think>\n{reasoning.strip()}\n</think>\n\n"
                    f"{rendered.get('content', '').lstrip()}"
                )
            rendered_messages.append(rendered)
        tokens = self.tokenizer.apply_chat_template(
            rendered_messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        if isinstance(tokens, Mapping):
            tokens = tokens["input_ids"]
        elif hasattr(tokens, "input_ids"):
            tokens = getattr(tokens, "input_ids")
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        if isinstance(tokens, list) and tokens and isinstance(tokens[0], list):
            if len(tokens) != 1:
                raise ValueError("expected one rendered conversation")
            tokens = tokens[0]
        if not isinstance(tokens, list):
            raise TypeError(f"unsupported tokenizer output type: {type(tokens)!r}")
        return len(tokens)

    def accepts(self, messages: list[dict[str, str]]) -> tuple[bool, int | None]:
        length = self.length(messages)
        return self.max_tokens is None or (length is not None and length <= self.max_tokens), length


def _write_row(handle, messages: list[dict[str, str]], metadata: dict[str, Any]) -> None:
    handle.write(json.dumps({"messages": messages, **metadata}, ensure_ascii=False) + "\n")


def _prepare_output(output_dir: Path, overwrite: bool) -> tuple[Any, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (output_dir / "train.jsonl", output_dir / "validation.jsonl")
    existing = [path for path in (*paths, output_dir / "manifest.json") if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"output already exists; pass --overwrite: {existing[0]}")
    return paths[0].open("w", encoding="utf-8"), paths[1].open("w", encoding="utf-8")


def prepare_wide(args: argparse.Namespace) -> dict[str, Any]:
    token_filter = TokenFilter(args.model, args.max_tokens)
    train_handle, validation_handle = _prepare_output(args.output_dir, args.overwrite)
    counts = {"source": 0, "train": 0, "validation": 0, "invalid": 0, "overlong": 0}
    longest = 0
    total_tokens = 0
    try:
        for index, row in enumerate(iter_json_records(args.source)):
            if args.limit is not None and counts["source"] >= args.limit:
                break
            counts["source"] += 1
            try:
                messages = wide_messages(row)
            except ValueError:
                counts["invalid"] += 1
                continue
            accepted, length = token_filter.accepts(messages)
            longest = max(longest, length or 0)
            if not accepted:
                counts["overlong"] += 1
                continue
            total_tokens += length or 0
            key = str(row.get("id") or row.get("question_id") or f"row:{index}")
            split = "validation" if stable_fraction(f"{CONTRACT_VERSION}:{args.seed}:{key}") < args.validation_fraction else "train"
            _write_row(
                validation_handle if split == "validation" else train_handle,
                messages,
                {"source": "synsql-think", "source_id": key},
            )
            counts[split] += 1
    finally:
        train_handle.close()
        validation_handle.close()
    if counts["train"] == 0 or counts["validation"] == 0:
        raise ValueError("wide split produced an empty train or validation file")
    return {
        "contract_version": CONTRACT_VERSION,
        "stage": "wide",
        "seed": args.seed,
        "source": {"path": str(args.source.resolve()), "sha256": file_sha256(args.source)},
        "validation_fraction": args.validation_fraction,
        "max_tokens": args.max_tokens,
        "longest_tokens": longest or None,
        "accepted_tokens": total_tokens or None,
        "counts": counts,
    }


def _cot_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["db_id"]), str(row["question_id"])


def _load_cot(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if path.suffix.casefold() in {".json", ".jsonl"}:
        rows: Iterable[dict[str, Any]] = iter_json_records(path)
    else:
        from datasets import load_dataset

        rows = load_dataset("parquet", data_files=str(path), split="train")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = _cot_key(row)
        if key in result:
            raise ValueError(f"duplicate CoT key: {key}")
        result[key] = dict(row)
    return result


def prepare_bird(args: argparse.Namespace) -> dict[str, Any]:
    token_filter = TokenFilter(args.model, args.max_tokens)
    cot = _load_cot(args.cot)
    train_handle, validation_handle = _prepare_output(args.output_dir, args.overwrite)
    counts = {"train": 0, "validation": 0, "missing_cot": 0, "overlong": 0}
    used: set[tuple[str, str]] = set()
    longest = 0
    total_tokens = 0
    try:
        for split, path, handle in (
            ("train", args.verified_train, train_handle),
            ("validation", args.verified_validation, validation_handle),
        ):
            for verified in read_records(path):
                key = _cot_key(verified)
                teacher = cot.get(key)
                if teacher is None:
                    counts["missing_cot"] += 1
                    continue
                # The verified split owns labels; the community file supplies CoT only.
                row = {**teacher, **verified, "reasoning": teacher.get("reasoning"), "parsed": teacher.get("parsed")}
                messages = bird_messages(row, args.database_root, sample_rows=args.sample_rows)
                accepted, length = token_filter.accepts(messages)
                longest = max(longest, length or 0)
                if not accepted:
                    counts["overlong"] += 1
                    continue
                total_tokens += length or 0
                _write_row(
                    handle,
                    messages,
                    {
                        "source": "bird-platinum-gpt-cot",
                        "source_id": f"{key[0]}:{key[1]}",
                        "db_id": key[0],
                        "question_fingerprint": normalized_question(str(verified["question"])),
                    },
                )
                used.add(key)
                counts[split] += 1
    finally:
        train_handle.close()
        validation_handle.close()
    if counts["missing_cot"] or len(used) + counts["overlong"] != len(cot):
        raise ValueError(
            "BIRD CoT coverage mismatch: "
            f"cot={len(cot)} used={len(used)} overlong={counts['overlong']} missing={counts['missing_cot']}"
        )
    return {
        "contract_version": CONTRACT_VERSION,
        "stage": "bird",
        "sources": {
            "verified_train": {"path": str(args.verified_train.resolve()), "sha256": file_sha256(args.verified_train)},
            "verified_validation": {"path": str(args.verified_validation.resolve()), "sha256": file_sha256(args.verified_validation)},
            "cot": {"path": str(args.cot.resolve()), "sha256": file_sha256(args.cot)},
        },
        "database_root": str(args.database_root.resolve()),
        "sample_rows": args.sample_rows,
        "max_tokens": args.max_tokens,
        "longest_tokens": longest or None,
        "accepted_tokens": total_tokens or None,
        "counts": counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="stage", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", type=Path, required=True)
    common.add_argument("--model", default="/data/qwen3-4b-instruct-2507")
    common.add_argument("--max-tokens", type=int, default=16_384)
    common.add_argument("--overwrite", action="store_true")

    wide = subparsers.add_parser("wide", parents=[common])
    wide.add_argument("--source", type=Path, required=True)
    wide.add_argument("--seed", type=int, default=17)
    wide.add_argument("--validation-fraction", type=float, default=0.002)
    wide.add_argument("--limit", type=int)

    bird = subparsers.add_parser("bird", parents=[common])
    bird.add_argument("--verified-train", type=Path, required=True)
    bird.add_argument("--verified-validation", type=Path, required=True)
    bird.add_argument("--cot", type=Path, required=True)
    bird.add_argument("--database-root", type=Path, required=True)
    bird.add_argument("--sample-rows", type=int, default=3)
    args = parser.parse_args()
    if args.stage == "wide" and not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    manifest = prepare_wide(args) if args.stage == "wide" else prepare_bird(args)
    manifest_path = args.output_dir / "manifest.json"
    manifest["files"] = {
        name: {
            "path": str((args.output_dir / f"{name}.jsonl").resolve()),
            "sha256": file_sha256(args.output_dir / f"{name}.jsonl"),
        }
        for name in ("train", "validation")
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
