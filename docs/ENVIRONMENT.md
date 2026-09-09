# RL Environment 与输出合同

模型输入包含问题、external evidence、SQLite schema 和少量样例行。模型没有数据库工具、执行反馈或第二轮修正。

当前 system prompt 的要求是：只返回一条只读 SQLite `SELECT` 或 `WITH`；禁止解释、XML、Markdown fence 和其他文字。`prompts.distillation_answer()` 只负责把 SFT 的 teacher reasoning 与 gold SQL 分别写入 `reasoning_content` / `content`，不会改变线上最终答案合同。

合法最终 content 示例：

```sql
SELECT name FROM singer ORDER BY age DESC LIMIT 1
```

`BirdText2SQLTaskset` 的 parser 只把 SQL-only content 标记为 `format_valid=true`。为了让旧 traces 仍可重打分，parser 还能从旧四块 XML、`<sql>`、Markdown fence、标题或裸 `SELECT` fallback 中提取 SQL；这些路径一律是“可尝试执行但格式无效”，不再出现在当前 prompt、数据构建或训练 target 中。

## 环境入口与单轮语义

环境包公开 `BirdText2SQLTaskset` 和兼容入口 `load_environment(...) -> vf.Environment`。生产配置使用 Taskset：每个 task 生成一轮 user message，模型只回复一次，没有数据库工具、执行反馈、自我修正轮次或 harness 状态。选择 `Taskset` 而不是 `StatefulToolEnv` 的原因正是 rollout 没有跨轮状态；SQLite 只存在于 scorer 侧。

`data.py` 接受 JSON/JSONL，兼容 `SQL`、`sql`、`query`、`gold_sql` 等 gold 字段；题目 ID 优先取 `question_id`，其次取 `id`。配置可选 `schema_root` 覆盖 descriptions、`gold_cache_path` 提供慢 gold sidecar，并支持 `limit` / `shuffle` 作为 smoke 或诊断手段。

## 评测流程与代码思路

环境包是 Verifiers v1 Taskset（`environments/bird_text2sql/bird_text2sql/`），
单轮、无 harness（`env.agent.harness.id = "null"`、runtime `subprocess`）。
评分链路：

```text
completion
  → parser.parse_completion()        # 格式合同
  → executor.execute_sql()           # 只读执行预测 SQL
  → compare_result_rows()            # 按该题 grading method 对比缓存 gold 行
  →（可选）verieql.grade_equivalence() # 仅在 exact 后调用
```

关键实现决策：

- **线程池隔离 SQLite**（`taskset.py`）：同步的 sqlite3 调用通过
  `ThreadPoolExecutor`（8 workers）包成 async，避免阻塞事件循环；同一 trace
  的评分用 `asyncio.Task` 缓存在 `trace.info` 里，多个 reward/metric 共享一次
  执行结果（`_score` / `_compute_score`），一条 SQL 只执行一次。
- **gold 结果优先级**（`prepare_rlvr_data.py` + `data.py`）：先用任务行的
  `gold_rows_json`，再兼容 `gold_result_json`，最后才现场执行 gold。最终泛化
  基准中已知慢 gold 由 `data/eval/slow_gold_cache.json` 兜底，并按 gold SQL
  SHA-256 绑定，防止标注变化后误用旧缓存。
- **只读安全**（`executor.py`）：先用 sqlglot 按 SQLite 方言解析，拒绝多语句、
  写/控制语句和非 SELECT；再用 `mode=ro&immutable=1` URI +
  `PRAGMA query_only` + progress-handler deadline 三层防护执行。parser/
  token 错误一律计为不可执行，不会变成 scorer error。
- **Platinum grading methods**（`compare_result_rows`）：按题目标注的
  `set`/`multiset`/`list`/`subset`/`subset,=,N` 比较；行内列序忽略、行序仅在
  `list` 下保留，`set` 与 `multiset` 对重复行的处理不同；空/全 NULL 行剔除；
  单值数值结果使用小于 1% 的相对误差容差。浮点在执行结果归一化时保留 10 位。
- **schema 渲染与缓存**（`schema.py` + `data.py`）：每张表输出
  `CREATE TABLE` DDL、列描述注释（来自数据库旁 CSV 或 `schema_root` 校正版）
  和 3 行样例；`lru_cache` 缓存渲染结果，同一数据库只读一次。
- **reward 设计**：`shape_reward=false` 时唯一的非零项就是
  `execution_reward`（exact 的 0/1）；shaped 惩罚（equivalence/evidence/format）
  保留在代码里但当前全部关闭，VeriEQL 也只在 `use_verieql=true` 且 exact 后
  才调用。这样使训练信号与最终 execution 指标一致，并避免把旧 XML process
  约束误当成当前目标。

评测时预测 SQL 会先按 SQLite 方言解析，只允许一条只读查询，再用只读 SQLite connection 执行。主指标 `exact_execution` 根据任务的 BIRD Platinum grading method 比较结果。当前推荐 RL 配置同样关闭 XML/process shaping，只使用 execution correctness。

关键指标：

| 指标 | 含义 |
| --- | --- |
| `exact_execution` | 预测结果是否符合该题 grading method |
| `executable_sql` | SQL 是否安全且成功执行 |
| `format_valid` | 最终 content 是否为 SQL-only |
| `execution_timeout` | 查询是否超时 |

另外仍会记录 `evidence_process_valid`、`semantic_equivalence_refuted` 与
`semantic_equivalence_unknown` 供历史 shaped-reward 兼容。SQL-only 主路线没有
编号 requirements/checks，所以 `evidence_process_valid` 不具备当前质量含义；
`shape_reward=false`、`use_verieql=false` 时三者也不参与总 reward。

任何预测侧解析失败、unsafe SQL、执行错误或 timeout 都是普通 0 分，不应升级为
scorer crash。只有基础设施/环境自身异常才属于 scorer error。
