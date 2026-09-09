#!/usr/bin/env python3
"""Fail-fast probe for the pinned VeriEQL worker used by RLVR."""

from __future__ import annotations

import asyncio

from bird_text2sql.verieql import check_available


def main() -> None:
    available, error = asyncio.run(check_available())
    if not available:
        raise SystemExit(f"VeriEQL preflight failed: {error or 'unknown error'}")
    print("VeriEQL preflight passed")


if __name__ == "__main__":
    main()
