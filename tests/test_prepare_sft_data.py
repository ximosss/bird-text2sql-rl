from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prepare_sft_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_sft_data", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_streams_json_array_and_converts_wide_cot(tmp_path: Path) -> None:
    source = tmp_path / "wide.json"
    source.write_text(
        json.dumps(
            [
                {
                    "input_seq": "Schema: t(v). Question: sum v",
                    "output_seq": "<think>Use SUM over v.</think><answer>SELECT SUM(v) FROM t</answer>",
                },
                {
                    "input_seq": "Schema: t(v). Question: count",
                    "output_seq": "<think>Count rows.</think><answer>SELECT COUNT(*) FROM t</answer>",
                },
            ]
        ),
        encoding="utf-8",
    )
    rows = list(MODULE.iter_json_records(source, chunk_size=7))
    assert len(rows) == 2
    messages = MODULE.wide_messages(rows[0])
    assert messages[-1]["content"] == "SELECT SUM(v) FROM t"
    assert messages[-1]["reasoning_content"] == "Use SUM over v."


def test_teacher_reasoning_uses_structured_fields_without_teacher_sql() -> None:
    reasoning = MODULE.teacher_reasoning(
        {
            "parsed": json.dumps(
                {
                    "reason": "Count matching rows.",
                    "columns": "t.v",
                    "select": "count t.v",
                    "values": "None",
                    "sql_like": "Filter then count.",
                    "sql": "SELECT COUNT(*) FROM t",
                }
            )
        }
    )
    assert "Count matching rows" in reasoning
    assert "SELECT COUNT" not in reasoning


def test_token_filter_counts_batch_encoding_input_ids() -> None:
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]}

    token_filter = MODULE.TokenFilter(None, None)
    token_filter.tokenizer = Tokenizer()
    assert token_filter.length([{"role": "user", "content": "q"}]) == 4
