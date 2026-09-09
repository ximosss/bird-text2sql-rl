from __future__ import annotations

from collections.abc import Sequence

SYSTEM_PROMPT = """You are an expert SQLite text-to-SQL system. Return exactly one read-only SQLite SELECT or WITH statement that answers the question from the supplied schema. Do not invent tables or columns. Return SQL only: no explanation, XML, Markdown fence, or other text."""


def build_user_prompt(*, db_id: str, question: str, schema: str, evidence: str | None = None) -> str:
    evidence_text = evidence.strip() if evidence and evidence.strip() else "(none)"
    return f"""Database: {db_id}

Schema (includes optional descriptions and example values):
{schema}

External evidence:
{evidence_text}

Question:
{question.strip()}"""


def evidence_items(evidence: str | None) -> list[str]:
    if not evidence:
        return []
    return [item.strip() for item in evidence.split(";") if item.strip()]


def distillation_answer(sql: str, *, reasoning: str) -> dict[str, str]:
    """Keep supervised reasoning separate from the SQL response content."""
    return {
        "role": "assistant",
        "reasoning_content": reasoning.strip(),
        "content": sql.strip(),
    }


def structured_answer(
    sql: str,
    *,
    reasoning: str,
    requirements: Sequence[str] = (),
    verification: Sequence[str] = (),
) -> str:
    """Format historical four-block artifacts for offline compatibility."""
    requirement_text = "\n".join(
        f'<requirement index="{index}">{text.strip()}</requirement>'
        for index, text in enumerate(requirements, start=1)
    )
    verification_text = "\n".join(
        f'<check index="{index}">{text.strip()}</check>'
        for index, text in enumerate(verification, start=1)
    )
    requirements_block = f"\n{requirement_text}\n" if requirement_text else ""
    verification_block = f"\n{verification_text}\n" if verification_text else ""
    return (
        f"<requirements>{requirements_block}</requirements>\n"
        f"<reasoning>{reasoning.strip()}</reasoning>\n"
        f"<verification>{verification_block}</verification>\n"
        f"<sql>{sql.strip()}</sql>"
    )
