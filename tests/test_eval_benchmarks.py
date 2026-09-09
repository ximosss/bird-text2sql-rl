from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data/eval"
CONFIG_ROOT = ROOT / "configs/prime-rl/eval"
REQUIRED_COLUMNS = {"question_id", "question", "evidence", "SQL", "db_id"}


def load_rows(relative_path: str) -> list[dict]:
    return json.loads((DATA_ROOT / relative_path).read_text(encoding="utf-8"))


def keyed(rows: list[dict]) -> dict[str, dict]:
    return {str(row["question_id"]): row for row in rows}


def test_final_generalization_datasets_have_expected_contract() -> None:
    arcwise = load_rows("arcwise/arcwise_plat_full_with_diff.json")
    arcwise_sql = load_rows("arcwise/arcwise_plat_sql_only_with_diff.json")
    mini = load_rows("bird/mini_dev_sqlite.json")
    full = load_rows("bird/dev_20240627.json")

    assert [len(rows) for rows in (arcwise, arcwise_sql, mini, full)] == [498, 498, 500, 1534]
    for rows in (arcwise, arcwise_sql, mini, full):
        assert all(REQUIRED_COLUMNS <= row.keys() for row in rows)
        assert len(keyed(rows)) == len(rows)

    arcwise_ids = set(keyed(arcwise))
    assert arcwise_ids == set(keyed(arcwise_sql))
    assert set(keyed(mini)) - arcwise_ids == {"119", "120"}
    assert set(keyed(full)) == {str(index) for index in range(1534)}

    # The full correction must actually differ from the SQL-only correction.
    full_by_id = keyed(arcwise)
    sql_by_id = keyed(arcwise_sql)
    assert any(full_by_id[key]["question"] != sql_by_id[key]["question"] for key in arcwise_ids)
    assert any(full_by_id[key]["evidence"] != sql_by_id[key]["evidence"] for key in arcwise_ids)
    assert any(full_by_id[key]["SQL"] != sql_by_id[key]["SQL"] for key in arcwise_ids)


def test_generalization_configs_are_local_reproducible_and_correctly_sized() -> None:
    expected = {
        "arcwise-plat": (498, "arcwise_plat_full_with_diff.json"),
        "arcwise-plat-sql": (498, "arcwise_plat_sql_only_with_diff.json"),
        "bird-mini-dev": (500, "mini_dev_sqlite.json"),
        "bird-full-dev": (1534, "dev_20240627.json"),
    }
    for name, (size, filename) in expected.items():
        config = tomllib.loads((CONFIG_ROOT / f"{name}.toml").read_text())
        assert config["num_tasks"] == size
        assert config["push"] is False
        assert config["output_dir"].endswith("/outputs/prime-rl")
        assert config["env"]["taskset"]["path"].endswith(filename)
        assert config["env"]["taskset"]["database_root"].endswith(
            "/minidev/MINIDEV/dev_databases"
        )
        assert config["env"]["taskset"]["gold_cache_path"].endswith(
            "/data/eval/slow_gold_cache.json"
        )
        assert config["sampling"]["temperature"] == 0.0
        assert config["num_rollouts"] == 1
        assert config["env"]["taskset"]["task"]["timeout_seconds"] == 30.0

    arcwise = tomllib.loads((CONFIG_ROOT / "arcwise-plat.toml").read_text())
    assert arcwise["env"]["taskset"]["schema_root"].endswith("/data/eval/arcwise/schemas")
    for name in ("arcwise-plat-sql", "bird-mini-dev", "bird-full-dev"):
        config = tomllib.loads((CONFIG_ROOT / f"{name}.toml").read_text())
        assert "schema_root" not in config["env"]["taskset"]


def test_arcwise_schema_overlay_is_complete_for_its_database_ids() -> None:
    rows = load_rows("arcwise/arcwise_plat_full_with_diff.json")
    schema_root = DATA_ROOT / "arcwise/schemas"
    observed = {path.parent.parent.name for path in schema_root.glob("*/database_description/*.csv")}
    assert observed == {str(row["db_id"]) for row in rows}


def test_slow_gold_cache_matches_all_four_annotation_sets() -> None:
    cache = json.loads((DATA_ROOT / "slow_gold_cache.json").read_text())
    assert set(cache) == {"518", "701"}
    for relative_path in (
        "arcwise/arcwise_plat_full_with_diff.json",
        "arcwise/arcwise_plat_sql_only_with_diff.json",
        "bird/mini_dev_sqlite.json",
        "bird/dev_20240627.json",
    ):
        rows = keyed(load_rows(relative_path))
        for question_id, cached in cache.items():
            observed = hashlib.sha256(rows[question_id]["SQL"].strip().encode()).hexdigest()
            assert observed == cached["gold_sql_sha256"]


def test_launcher_exposes_the_frozen_generalization_suite() -> None:
    launcher = (ROOT / "scripts/launch_prime_rl_eval.sh").read_text()
    assert "generalization" in launcher
    for name in ("arcwise-plat", "arcwise-plat-sql", "bird-mini-dev", "bird-full-dev"):
        assert name in launcher
