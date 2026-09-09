# bird-text2sql environment

BIRD SQLite 的单轮 Verifiers v1 Taskset。模型输入 question、evidence、schema 和样例行，最终 content 只能是一条只读 SQLite `SELECT`/`WITH`。

```text
SELECT ...
```

不要求 XML、requirements、verification、Markdown 或工具调用。只有 SQL-only
content 会被标记为 format-valid；旧 XML/tag/fence parser 仅保留用于读取和重打分
历史 trace，提取出的 SQL 可以执行但格式记为无效。

环境在 scorer 侧用只读 SQLite 执行预测 SQL，按题目
`set`/`multiset`/`list`/`subset` grading method 与缓存 gold rows 比较。SQLite
工作在线程池中运行，同一 trace 的 reward/metrics 共享一次 score task。当前正式
配置设置 `shape_reward=false`、`use_verieql=false`，唯一训练信号是 0/1 exact
execution；`executable_sql`、`format_valid` 和 `execution_timeout` 只作诊断。

## 安装与测试

```bash
prime env install bird-text2sql --path environments --plain
uv run pytest environments/bird_text2sql/tests -q
```

## 最小配置

```toml
[env.taskset]
id = "bird-text2sql"
path = "/path/to/validation.jsonl"
database_root = "/path/to/train_databases"
# Optional corrected descriptions: <schema_root>/<db_id>/database_description/*.csv
schema_root = "/path/to/corrected-schemas"
# Optional precomputed rows for known slow gold queries, guarded by SQL SHA-256.
gold_cache_path = "/path/to/gold-cache.json"
sample_rows = 3

[env.taskset.task]
shape_reward = false
use_verieql = false

[env.agent.harness]
id = "null"

[env.agent.runtime]
type = "subprocess"
```

当前推荐 SFT、direct RLVR、post-training selection eval 与最终四基准见仓库根目录的
[README](../../README.md)；完整实现合同见
[docs/ENVIRONMENT.md](../../docs/ENVIRONMENT.md)。
