from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
from functools import lru_cache
from pathlib import Path


_WORKER = Path(__file__).with_name("_verieql_worker.py")


def _verieql_type(sqlite_type: str) -> str:
    value = (sqlite_type or "").upper()
    if any(token in value for token in ("INT", "BOOL", "REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL")):
        return "INT"
    return "VARCHAR"


@lru_cache(maxsize=512)
def sqlite_schema(db_path: str) -> dict[str, dict[str, str]]:
    connection = sqlite3.connect(f"file:{Path(db_path).resolve().as_posix()}?mode=ro", uri=True)
    try:
        tables = [
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        schema: dict[str, dict[str, str]] = {}
        for table in tables:
            escaped = table.replace("'", "''")
            columns = {
                str(row[1]): _verieql_type(str(row[2]))
                for row in connection.execute(f"PRAGMA table_info('{escaped}')")
            }
            if columns:
                schema[table] = columns
        return schema
    finally:
        connection.close()


async def _run_worker(payload: dict, timeout_seconds: float) -> dict:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(_WORKER),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(json.dumps(payload).encode()), timeout=timeout_seconds
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        return {"result": None, "error": "timeout"}
    if process.returncode != 0:
        return {
            "result": None,
            "error": f"worker_exit_{process.returncode}: {stderr.decode(errors='replace')[:500]}",
        }
    try:
        parsed = json.loads(stdout.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"result": None, "error": f"invalid_worker_output: {exc}"}
    return parsed if isinstance(parsed, dict) else {"result": None, "error": "invalid_worker_output"}


async def check_available(timeout_seconds: float = 15.0) -> tuple[bool, str | None]:
    result = await _run_worker({"probe": True}, timeout_seconds)
    return result.get("result") is True, result.get("error")


async def grade_equivalence(
    db_path: str,
    predicted_sql: str,
    gold_sql: str,
    *,
    timeout_seconds: float = 30.0,
    bound_size: int = 2,
) -> tuple[bool | None, str | None]:
    schema = sqlite_schema(str(Path(db_path).resolve()))
    if not schema:
        return None, "empty_schema"
    response = await _run_worker(
        {
            "schema": schema,
            "predicted_sql": predicted_sql,
            "gold_sql": gold_sql,
            "bound_size": bound_size,
        },
        timeout_seconds,
    )
    value = response.get("result")
    return (value if isinstance(value, bool) else None), response.get("error")
