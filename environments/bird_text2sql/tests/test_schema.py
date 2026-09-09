from __future__ import annotations

import sqlite3
from pathlib import Path

from bird_text2sql.schema import render_schema, resolve_db_path


def test_schema_contains_types_case_preserved_samples_and_descriptions(tmp_path: Path) -> None:
    db_dir = tmp_path / "demo"
    db_dir.mkdir()
    db_path = db_dir / "demo.sqlite"
    connection = sqlite3.connect(db_path)
    connection.executescript("CREATE TABLE people (name TEXT); INSERT INTO people VALUES ('Alice');")
    connection.close()
    descriptions = db_dir / "database_description"
    descriptions.mkdir()
    (descriptions / "people.csv").write_text(
        "original_column_name,column_name,column_description,data_format,value_description\n"
        "name,name,Person display name,text,Case-sensitive value\n",
        encoding="utf-8",
    )

    assert resolve_db_path(tmp_path, "demo") == db_path.resolve()
    rendered = render_schema(str(db_path.resolve()))
    assert 'CREATE TABLE "people"' in rendered
    assert "Person display name" in rendered
    assert "Alice" in rendered


def test_schema_supports_implicit_foreign_key_target(tmp_path: Path) -> None:
    db_path = tmp_path / "implicit.sqlite"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        "CREATE TABLE parent (id INTEGER PRIMARY KEY);"
        "CREATE TABLE child (parent_id INTEGER REFERENCES parent);"
    )
    connection.close()
    rendered = render_schema(str(db_path.resolve()))
    assert 'FOREIGN KEY ("parent_id") REFERENCES "parent"' in rendered


def test_schema_can_overlay_corrected_descriptions(tmp_path: Path) -> None:
    db_dir = tmp_path / "databases" / "demo"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "demo.sqlite"
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE people (name TEXT)")
    connection.close()

    corrected_db_dir = tmp_path / "corrected-schemas" / "demo"
    descriptions = corrected_db_dir / "database_description"
    descriptions.mkdir(parents=True)
    (descriptions / "people.csv").write_text(
        "original_column_name,column_description,value_description\n"
        "name,Corrected person name,Corrected value meaning\n",
        encoding="utf-8",
    )

    rendered = render_schema(
        str(db_path),
        description_db_dir=str(corrected_db_dir),
    )
    assert "Corrected person name" in rendered
    assert "Corrected value meaning" in rendered
