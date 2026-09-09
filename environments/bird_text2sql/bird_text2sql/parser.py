from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedCompletion:
    sql: str | None
    format_valid: bool
    source: str
    reasoning: str | None = None
    requirements: tuple[tuple[int, str], ...] = ()
    verification: tuple[tuple[int, str], ...] = ()

    def evidence_process_valid(self, expected_items: int) -> bool:
        expected = tuple(range(1, expected_items + 1))
        requirement_indices = tuple(index for index, _ in self.requirements)
        verification_indices = tuple(index for index, _ in self.verification)
        return requirement_indices == expected and verification_indices == expected


_STRICT_CONTRACT_RE = re.compile(
    r"^\s*(?:<requirements>(?P<requirements>.*?)</requirements>|<requirements\s*/>)\s*"
    r"<reasoning>(?P<reasoning>.*?)</reasoning>\s*"
    r"(?:<verification>(?P<verification>.*?)</verification>|<verification\s*/>)\s*"
    r"<sql>(?P<sql>.*?)</sql>\s*$",
    re.IGNORECASE | re.DOTALL,
)
_REQUIREMENT_RE = re.compile(
    r'<requirement\s+index=["\'](?P<index>\d+)["\']\s*>(?P<text>.*?)</requirement>',
    re.IGNORECASE | re.DOTALL,
)
_CHECK_RE = re.compile(
    r'<check\s+index=["\'](?P<index>\d+)["\']\s*>(?P<text>.*?)</check>',
    re.IGNORECASE | re.DOTALL,
)
_SQL_TAG_RE = re.compile(r"<sql>(?P<sql>.*?)</sql>", re.IGNORECASE | re.DOTALL)
_FENCE_RE = re.compile(r"```(?:sql)?\s*(?P<sql>.*?)```", re.IGNORECASE | re.DOTALL)
_HEADING_RE = re.compile(
    r"(?:^|\n)\s*(?:#{1,4}\s*)?SQL(?:\s+Query)?\s*:\s*(?P<sql>.*)",
    re.IGNORECASE | re.DOTALL,
)
_SELECT_RE = re.compile(r"\b(?:WITH|SELECT)\b.*", re.IGNORECASE | re.DOTALL)
_SQL_ONLY_RE = re.compile(r"^\s*(?P<sql>(?:WITH|SELECT)\b.*)\s*$", re.IGNORECASE | re.DOTALL)


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if not isinstance(completion, list):
        return ""
    for message in reversed(completion):
        if isinstance(message, dict):
            role, content = message.get("role"), message.get("content", "")
        else:
            role, content = getattr(message, "role", None), getattr(message, "content", "")
        if role != "assistant":
            continue
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ).strip()
    return ""


def _clean_sql(sql: str) -> str | None:
    value = sql.strip()
    return value if value else None


def _indexed_blocks(pattern: re.Pattern[str], text: str) -> tuple[tuple[int, str], ...] | None:
    blocks = tuple(
        (int(match.group("index")), match.group("text").strip())
        for match in pattern.finditer(text)
    )
    remainder = pattern.sub("", text).strip()
    if remainder or any(not value for _, value in blocks):
        return None
    if tuple(index for index, _ in blocks) != tuple(range(1, len(blocks) + 1)):
        return None
    return blocks


def parse_completion(completion: Any) -> ParsedCompletion:
    text = completion_text(completion)
    sql_only = _SQL_ONLY_RE.match(text)
    if sql_only:
        return ParsedCompletion(_clean_sql(sql_only.group("sql")), True, "sql_only")
    strict = _STRICT_CONTRACT_RE.match(text)
    if strict:
        requirements = _indexed_blocks(_REQUIREMENT_RE, strict.group("requirements") or "")
        verification = _indexed_blocks(_CHECK_RE, strict.group("verification") or "")
        reasoning = strict.group("reasoning").strip()
        sql = _clean_sql(strict.group("sql"))
        parseable = requirements is not None and verification is not None and bool(reasoning) and sql is not None
        return ParsedCompletion(
            sql=sql,
            # Historical four-block traces remain extractable, but SQL-only is
            # the sole valid format for current runs.
            format_valid=False,
            source="legacy_contract" if parseable else "malformed_legacy_contract",
            reasoning=reasoning or None,
            requirements=requirements or (),
            verification=verification or (),
        )

    sql_tag = _SQL_TAG_RE.search(text)
    if sql_tag:
        return ParsedCompletion(_clean_sql(sql_tag.group("sql")), False, "sql_tag_only")

    fence = _FENCE_RE.search(text)
    if fence:
        return ParsedCompletion(_clean_sql(fence.group("sql")), False, "fence")

    heading = _HEADING_RE.search(text)
    if heading:
        return ParsedCompletion(_clean_sql(heading.group("sql")), False, "heading")

    select = _SELECT_RE.search(text)
    if select:
        return ParsedCompletion(_clean_sql(select.group(0)), False, "fallback")
    return ParsedCompletion(None, False, "missing")
