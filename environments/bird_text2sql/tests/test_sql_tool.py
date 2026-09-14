from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from bird_text2sql.sql_tool import BirdSQLToolConfig, BirdSQLToolset


def make_db(tmp_path: Path) -> Path:
    path = tmp_path / "tool.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE t(v TEXT);"
        "INSERT INTO t VALUES ('abcdefghijklmnopqrstuvwxyz'), ('second'), ('third');"
    )
    connection.close()
    return path


def test_sql_tool_is_read_only_and_bounded(tmp_path: Path) -> None:
    toolset = BirdSQLToolset(
        BirdSQLToolConfig(max_rows=2, max_cell_chars=16, max_output_chars=512)
    )
    toolset._inert_state.db_path = str(make_db(tmp_path))

    result = toolset.execute_sql_query("SELECT v FROM t ORDER BY rowid")
    assert '"returned_rows": 2' in result
    assert '"truncated": true' in result
    assert "...(truncated)" in result

    rejected = toolset.execute_sql_query("DELETE FROM t")
    assert "write or control statements are forbidden" in rejected


def test_sql_tool_rejects_multiple_statements(tmp_path: Path) -> None:
    toolset = BirdSQLToolset(BirdSQLToolConfig())
    toolset._inert_state.db_path = str(make_db(tmp_path))
    result = toolset.execute_sql_query("SELECT 1; SELECT 2")
    assert "exactly one SQL statement" in result


def test_sql_tool_module_reports_subprocess_port(tmp_path: Path) -> None:
    port_file = tmp_path / "mcp-port"
    env = os.environ | {
        "MCP_PORT_FILE": str(port_file),
        "VF_CONFIG": BirdSQLToolConfig().model_dump_json(),
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "bird_text2sql.sql_tool"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not port_file.exists():
            if process.poll() is not None:
                stderr = process.stderr.read()
                if "PermissionError: [Errno 1] Operation not permitted" in stderr:
                    pytest.skip("sandbox forbids binding a local MCP test socket")
                raise AssertionError(stderr)
            time.sleep(0.05)
        assert port_file.read_text().strip().isdigit()
    finally:
        process.terminate()
        process.wait(timeout=5)
