# Final generalization benchmarks

This directory vendors the small, immutable annotations needed by the final
Text-to-SQL generalization suite. SQLite databases remain under
`/data/ximo/minidev/MINIDEV/dev_databases` and are not duplicated here.

As of 2026-09-06 these assets and their Prime-RL configs have passed local
configuration checks, but no formal Base/SFT/RLVR generalization batch has been
run. A semantic preflight blocker remains: the shared eval server does not yet
enable the Qwen3 reasoning parser, while the suite currently applies the same
thinking-on config to Base and trained candidates. They are frozen final-test
assets and must not be used to select a checkpoint or launched as a formal
batch until that protocol is aligned.

| Benchmark | Rows | Annotation contract | File |
| --- | ---: | --- | --- |
| Arcwise-Plat | 498 | Corrected SQL, questions/evidence, and schema descriptions | `arcwise/arcwise_plat_full_with_diff.json` |
| Arcwise-Plat-SQL | 498 | Corrected SQL; original ambiguous questions/evidence/schema | `arcwise/arcwise_plat_sql_only_with_diff.json` |
| BIRD Mini-Dev | 500 | Original noisy BIRD Mini-Dev SQLite annotations | `bird/mini_dev_sqlite.json` |
| BIRD Full Dev | 1,534 | Original noisy BIRD Dev annotations | `bird/dev_20240627.json` |

The two Arcwise variants measure performance when task expertise has repaired
some or all of the benchmark environment. The original Mini-Dev and Full Dev
are intentionally retained to measure robustness to the annotation and schema
noise encountered in the official release. Scores across these contracts must
be reported side by side rather than averaged or treated as the same test set.

Arcwise deliberately omits BIRD Mini-Dev question IDs `119` and `120`, so its
498 rows must not be reported with a denominator of 500. The two Arcwise files
share exactly the same 498 IDs. `Arcwise-Plat` uses the corrected schema
descriptions in `arcwise/schemas`; the other three suites read the original
descriptions beside the SQLite databases.

`slow_gold_cache.json` stores the official gold execution rows for IDs 518 and
701. Their unchanged gold SQL takes about 60 seconds and more than 300 seconds,
respectively, on the local 250/460 MB SQLite files and becomes unstable under
concurrent evaluation. The cache prevents gold-side I/O timeouts; predicted SQL
is still executed normally. Each cache entry is bound to the exact gold SQL by
SHA-256, and loading fails if an annotation changes.

## Provenance

- Arcwise data and 75 schema-description CSVs: commit
  `fe766045c55b6875a43b30e9ac7683df5582f8cf` of
  <https://github.com/uiuc-kang-lab/text_to_sql_benchmarks>.
- Original Full Dev: `dev_20240627/dev.json` extracted from the BIRD official
  `https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip`, last modified
  2024-06-29, archive ETag `04B4AF221C9186361F09B16ABFD917EC`.
- Original Mini-Dev: the SQLite annotation file distributed by the official
  BIRD Mini-Dev release: <https://github.com/bird-bench/mini_dev>.

SHA-256:

```text
baefe2ca4fbab86c000df72aad9eaa563f7d422e425af2cbff071f656ea4eea8  arcwise/arcwise_plat_full_with_diff.json
e7bf76408f99266506ea558d84982c69422e5db3dfecab5f08a3a8d3900e8395  arcwise/arcwise_plat_sql_only_with_diff.json
630272f2b1c44d8cef2c3b246f623355cf0bbc1e832c81061df895530dfc2f06  bird/dev_20240627.json
def4b2b43a9b06955193418f24c9be170eb6d83763ab702311790e0cdda8c791  bird/mini_dev_sqlite.json
3153cff228c470419eda534315d3d56fe4464bcbe4f42a7bf83df536f6370fe8  slow_gold_cache.json
```

Both upstream datasets are distributed under CC BY-SA 4.0. A copy of the
license is included in each source subdirectory.
