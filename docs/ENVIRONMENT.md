# Text-to-SQL 环境

环境位于 `environments/bird_text2sql/`，暴露 `load_environment(...)` 和 Verifiers
v1 Taskset。它用于 RLVR rollout、checkpoint execution eval 和最终测试。

## 1. 单轮输入输出

模型输入包含：

- database ID；
- 自然语言问题；
- external evidence（若有）；
- SQLite `CREATE TABLE` schema、列描述；
- 每张表最多 `sample_rows` 行真实样例值。

模型没有 SQL 执行工具、数据库反馈或第二轮修正机会。最终 content 必须只有一条
只读 SQLite 语句：

```sql
SELECT name FROM singer ORDER BY age DESC LIMIT 1
```

禁止解释、XML、Markdown fence 和额外文本。SFT 的 `reasoning_content` 是训练渠道，
不改变最终 content 的 SQL-only 合同。

## 2. Parser

当前 parser 只有 SQL-only `SELECT`/`WITH` 会标为 `format_valid=true`。为了重打分
旧 trace，它仍能从旧 XML、`<sql>`、Markdown fence 等格式提取 SQL 并尝试执行，
但这些输出一律格式无效。格式有效不代表 SQL 可执行或语义正确。

## 3. 执行

- SQLite 连接以只读模式打开。
- 只允许一条 `SELECT` 或 `WITH`；写操作、多语句和危险 pragma 被拒绝。
- prediction 和 gold 分别设置 timeout。
- 同一 trace 的 reward 与诊断 metrics 共享一次执行结果。
- gold rows 可来自 taskset 的 `gold_rows_json` 或 SQL SHA-256 绑定的 sidecar cache。

## 4. 结果比较

环境根据每题 `grading_method` 比较预测 rows 与 gold rows：

| 方法 | 行顺序 | 重复行 | 含义 |
| --- | --- | --- | --- |
| `set` | 忽略 | 忽略 | 无序集合相等 |
| `multiset` | 忽略 | 保留 | 无序多重集合相等 |
| `list` | 保留 | 保留 | 有序结果完全相等 |
| `subset` | 忽略 | 忽略 | 非空预测结果必须是 gold 结果的子集 |

行内列顺序按环境实现归一化，字符串大小写保持，浮点值按配置精度处理。

正式配置使用 `shape_reward=false`、`use_verieql=false`，所以主 reward 是二值 exact
execution。`format_valid`、`executable_sql`、`execution_timeout` 仅作诊断。

## 5. 最小 Taskset 配置

```toml
[env.taskset]
id = "bird-text2sql"
path = "/path/to/tasks.jsonl"
database_root = "/path/to/databases"
sample_rows = 3

# 只有 Arcwise-Plat 需要校正后的 schema description overlay。
schema_root = "/path/to/corrected-schemas"

# 可选：已知慢 gold 查询的结果缓存。
gold_cache_path = "/path/to/gold-cache.json"

[env.taskset.task]
timeout_seconds = 8.0
gold_timeout_seconds = 60.0
float_digits = 10
shape_reward = false
use_verieql = false

[env.agent]
max_turns = 1

[env.agent.harness]
id = "null"

[env.agent.runtime]
type = "subprocess"
```

## 6. 安装与测试

```bash
prime env install bird-text2sql --path environments --plain
uv run pytest environments/bird_text2sql/tests -q
```

数据格式与 split 见 [`DATASET.md`](DATASET.md)，完整评测协议见
[`EVALUATION.md`](EVALUATION.md)。
