#!/usr/bin/env python3
from __future__ import annotations

import argparse

from renderers.base import ToolCallParseStatus
from renderers.configs import DefaultRendererConfig
from renderers.default import DefaultRenderer
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check the canonical non-thinking Qwen3 tool-call contract."
    )
    parser.add_argument("model")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = DefaultRendererConfig(
        tool_parser="qwen3",
        enable_thinking=False,
    )
    renderer = DefaultRenderer(tokenizer, config)
    messages = [
        {"role": "system", "content": "Use the SQL tool when useful."},
        {"role": "user", "content": "Count the rows."},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_sql_query",
                "description": "Execute read-only SQL.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]

    rendered = renderer.render_ids(
        messages, tools=tools, add_generation_prompt=True
    )
    canonical = list(
        tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )
    assert rendered == canonical
    prompt_tail = tokenizer.decode(rendered[-16:], skip_special_tokens=False)
    assert "<think>" not in prompt_tail and "</think>" not in prompt_tail

    completion = tokenizer.encode(
        '<tool_call>\n{"name":"execute_sql_query",'
        '"arguments":{"query":"SELECT COUNT(*) FROM t"}}\n</tool_call>',
        add_special_tokens=False,
    )
    parsed = renderer.parse_response(completion, tools=tools)
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert call.status is ToolCallParseStatus.OK
    assert call.name == "execute_sql_query"
    assert call.arguments == {"query": "SELECT COUNT(*) FROM t"}
    print("canonical_prompt=true")
    print("thinking_prefill=false")
    print("tool_call_parse=true")


if __name__ == "__main__":
    main()
