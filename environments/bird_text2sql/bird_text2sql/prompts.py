from __future__ import annotations

from collections.abc import Sequence

SYSTEM_PROMPT = """You are an expert SQLite text-to-SQL system. Return exactly one read-only SQLite SELECT or WITH statement that answers the question from the supplied schema. Do not invent tables or columns. Return SQL only: no explanation, XML, Markdown fence, or other text."""


REVISQL_SYSTEM_PROMPT = """Task Overview:
You are a data science expert. You are given a SQLite database schema, external knowledge, and a natural-language question. Understand the schema and produce a valid SQL query within limited turns.

Instructions:
- Return exactly the columns and information requested, without missing or extra information.
- Use every non-empty item of external knowledge. Items are separated by semicolons.
- Think visibly when you receive new observations: analyze the question and schema, summarize findings, verify assumptions, refine errors, and decide whether to call the SQL tool.
- In the first or second model turn, translate every external-knowledge item into a requirement using exactly: "Requirement of external knowledge {i} ({raw external knowledge text}): ...". Skip this when external knowledge is empty.
- You may call the read-only `execute_sql_query` tool to explore data or verify a draft. Tool results and errors are observations for the next turn.
- Before a final solution, verify every external-knowledge item using exactly: "Verification of external knowledge {i} ({raw external knowledge text}): ...". Skip this when external knowledge is empty.
- Complete all required verification text before opening the final solution block.
- Put the final SQL inside <solution>...</solution>; this solution block must be the last part of the response.
- Once the final solution block is closed, stop generating immediately: do not repeat its closing delimiter or append any other text.
- Do not call a tool and provide a final solution in the same response. Use at most four tool-call turns; on the fifth turn, finalize without calling a tool.
- Finalize when no more exploration is needed or the turn limit is reached."""


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
