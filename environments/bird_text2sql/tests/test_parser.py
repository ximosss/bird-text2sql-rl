from bird_text2sql.parser import parse_completion, parse_revisql_completion
from bird_text2sql.prompts import structured_answer


def test_legacy_contract_is_extractable_but_not_current_valid_format() -> None:
    parsed = parse_completion(
        [
            {
                "role": "assistant",
                "content": structured_answer(
                    "SELECT name FROM people",
                    reasoning="Read the requested column.",
                    requirements=["Use score = 100."],
                    verification=["The WHERE clause uses score = 100."],
                ),
            }
        ]
    )
    assert not parsed.format_valid
    assert parsed.sql == "SELECT name FROM people"
    assert parsed.evidence_process_valid(1)
    assert parsed.source == "legacy_contract"

    legacy = parse_completion("<sql>SELECT name FROM people</sql>")
    assert not legacy.format_valid
    assert legacy.sql == "SELECT name FROM people"


def test_evidence_contract_requires_contiguous_matching_indices() -> None:
    parsed = parse_completion(
        """<requirements><requirement index="1">a</requirement></requirements>
<reasoning>r</reasoning>
<verification><check index="1">a</check></verification>
<sql>SELECT 1</sql>"""
    )
    assert not parsed.format_valid
    assert parsed.evidence_process_valid(1)
    assert not parsed.evidence_process_valid(2)

    malformed = parse_completion(
        """<requirements><requirement index="2">a</requirement></requirements>
<reasoning>r</reasoning><verification></verification><sql>SELECT 1</sql>"""
    )
    assert not malformed.format_valid


def test_self_closing_empty_evidence_blocks_are_legacy_only() -> None:
    parsed = parse_completion(
        "<requirements/><reasoning>r</reasoning><verification/><sql>SELECT 1</sql>"
    )

    assert not parsed.format_valid
    assert parsed.sql == "SELECT 1"
    assert parsed.evidence_process_valid(0)


def test_lenient_sql_extraction_does_not_count_as_valid_format() -> None:
    parsed = parse_completion("Here it is:\n```sql\nSELECT 1;\n```")
    assert not parsed.format_valid
    assert parsed.sql == "SELECT 1;"


def test_sql_only_is_the_current_valid_format() -> None:
    parsed = parse_completion("SELECT name FROM people")

    assert parsed.format_valid
    assert parsed.sql == "SELECT name FROM people"
    assert parsed.source == "sql_only"


def test_missing_sql() -> None:
    parsed = parse_completion("I do not know")
    assert parsed.sql is None
    assert not parsed.format_valid


def test_revisql_extracts_only_terminal_solution() -> None:
    parsed = parse_revisql_completion(
        "Visible analysis.\n<solution>SELECT name FROM people</solution>"
    )
    assert parsed.format_valid
    assert parsed.sql == "SELECT name FROM people"
    assert parsed.source == "solution"

    assert not parse_revisql_completion("<solution>SELECT 1</solution> trailing").format_valid
    assert not parse_revisql_completion("SELECT 1").format_valid


def test_revisql_rejects_nested_reasoning_tags_inside_solution() -> None:
    parsed = parse_revisql_completion(
        "<solution><think>hidden</think> SELECT 1</solution>"
    )
    assert not parsed.format_valid
    assert parsed.sql is None
