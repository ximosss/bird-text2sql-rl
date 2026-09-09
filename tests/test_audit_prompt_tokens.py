from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_prompt_tokens.py"
SPEC = importlib.util.spec_from_file_location("audit_prompt_tokens", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_token_count_accepts_list_mapping_and_single_batch() -> None:
    assert MODULE.token_count([1, 2, 3]) == 3
    assert MODULE.token_count({"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]}) == 4
    assert MODULE.token_count({"input_ids": [[1, 2, 3]]}) == 3
