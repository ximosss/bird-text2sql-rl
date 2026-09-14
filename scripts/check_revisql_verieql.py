#!/usr/bin/env python3
"""Preflight the four required VeriEQL result classes."""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
from pathlib import Path

from bird_text2sql.verieql import check_available, grade_equivalence


async def main() -> None:
    available, error = await check_available()
    if not available:
        raise RuntimeError(f"VeriEQL unavailable: {error}")
    with tempfile.TemporaryDirectory(prefix="bird-revisql-verieql-") as directory:
        db_path = Path(directory) / "probe.sqlite"
        connection = sqlite3.connect(db_path)
        connection.executescript("CREATE TABLE t(v INT); INSERT INTO t VALUES (1), (2);")
        connection.close()
        probes = {
            "supported": await grade_equivalence(
                str(db_path), "SELECT v FROM t", "SELECT v FROM t", timeout_seconds=30
            ),
            "refuted": await grade_equivalence(
                str(db_path), "SELECT v FROM t", "SELECT v + 1 FROM t", timeout_seconds=30
            ),
            "unsupported": await grade_equivalence(
                str(db_path), "SELECT random() FROM t", "SELECT v FROM t", timeout_seconds=30
            ),
            "timeout": await grade_equivalence(
                str(db_path), "SELECT v FROM t", "SELECT v FROM t", timeout_seconds=1e-6
            ),
        }
    assert probes["supported"] == (True, None), probes
    assert probes["refuted"] == (False, None), probes
    assert probes["unsupported"][0] is None, probes
    assert probes["timeout"] == (None, "timeout"), probes
    for name, value in probes.items():
        print(f"{name}={value!r}")


if __name__ == "__main__":
    asyncio.run(main())
