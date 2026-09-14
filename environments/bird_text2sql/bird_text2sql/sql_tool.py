from __future__ import annotations

import json
from typing import Any

import verifiers.v1 as vf
from pydantic import Field

from .executor import ExecutionResult, execute_sql


class BirdSQLState(vf.State):
    db_path: str = ""


class BirdSQLToolConfig(vf.SharedToolsetConfig):
    timeout_seconds: float = Field(8.0, gt=0)
    float_digits: int = Field(10, ge=0)
    max_rows: int = Field(50, ge=1)
    max_cell_chars: int = Field(200, ge=16)
    max_output_chars: int = Field(12_000, ge=256)


def _bounded_cell(value: Any, max_chars: int) -> Any:
    text = str(value)
    if len(text) <= max_chars:
        return value
    return text[:max_chars] + "...(truncated)"


def format_tool_result(
    result: ExecutionResult,
    *,
    max_rows: int,
    max_cell_chars: int,
    max_output_chars: int,
) -> str:
    if not result.ok:
        kind = "timeout" if result.timed_out else "error"
        return f"SQL execution {kind}: {result.error or 'unknown error'}"
    rows = [
        [_bounded_cell(cell, max_cell_chars) for cell in row]
        for row in result.ordered_rows[:max_rows]
    ]
    payload = {
        "rows": rows,
        "returned_rows": len(rows),
        "truncated": len(result.ordered_rows) > max_rows,
    }
    text = "SQL execution results: " + json.dumps(payload, ensure_ascii=False, default=str)
    if len(text) > max_output_chars:
        text = text[:max_output_chars] + "...(output truncated)"
    return text


class BirdSQLToolset(vf.Toolset[BirdSQLToolConfig, BirdSQLState]):
    TOOL_PREFIX = None

    @vf.tool
    def execute_sql_query(self, query: str) -> str:
        """Execute one read-only SQLite SELECT/WITH query and return bounded rows or an error."""

        if not self.state.db_path:
            return "SQL execution error: rollout database was not initialized"
        result = execute_sql(
            self.state.db_path,
            query,
            timeout_seconds=self.config.timeout_seconds,
            float_digits=self.config.float_digits,
        )
        return format_tool_result(
            result,
            max_rows=self.config.max_rows,
            max_cell_chars=self.config.max_cell_chars,
            max_output_chars=self.config.max_output_chars,
        )


if __name__ == "__main__":
    BirdSQLToolset.run()
