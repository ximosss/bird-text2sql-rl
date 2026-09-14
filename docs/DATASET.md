# 数据集

## 1. 总表

| 阶段 | Split | 上游数据 | 处理后数量 | 作用 |
| --- | --- | --- | ---: | --- |
| SFT | train | ReViSQL `bird-verified-train` + GPT-5.4 CoT | 2,064 | 参数训练 |
| SFT | validation | ReViSQL `bird-verified-val` + GPT-5.4 CoT | 398 | teacher-forced loss；只作拟合诊断 |
| RLVR | train | ReViSQL `bird-verified-train` | 2,050 | 在线 rollout 与 execution reward |
| RLVR | validation | ReViSQL `bird-verified-val` | 396 | SFT/RLVR checkpoint 的 execution 选择 |
| 最终测试 | Arcwise-Plat | 校正后的 BIRD Mini-Dev 子集 | 498 | 全校正条件下的泛化 |
| 最终测试 | Arcwise-Plat-SQL | 仅校正 SQL 的 BIRD Mini-Dev 子集 | 498 | 原始问题/schema 噪声下的泛化 |
| 最终测试 | BIRD Mini-Dev | 官方 Mini-Dev | 500 | 原始标注上的小规模测试 |
| 最终测试 | BIRD Full Dev | 官方 2024-06-27 Dev | 1,534 | 原始标注上的完整测试 |



## 2. 共同依赖：BIRD SQLite 数据库

JSON/JSONL 只保存题目和 SQL，不包含数据库。需要另外准备两套数据库：

```text
<bird-train-root>/
└── <db_id>/
    ├── <db_id>.sqlite
    └── database_description/*.csv

<bird-dev-root>/
└── <db_id>/
    ├── <db_id>.sqlite
    └── database_description/*.csv
```

- SFT 和 RLVR 使用 BIRD train databases。
- 四个最终测试使用 BIRD dev databases。
- `sample_rows=3` 时，预处理或环境会把每张表最多三行真实值写进 schema prompt。

从 [BIRD 官方站点](https://bird-bench.github.io/) 获取.

## 3. SFT 数据

### 3.1 来源

SFT 逐题连接两个来源，连接键是 `(db_id, question_id)`：

1. [ReViSQL](https://github.com/uiuc-kang-lab/ReViSQL) 的
   `data/bird-verified-train.json`（2,064）和 `data/bird-verified-val.json`
   （398）。它拥有最终的 `question`、`evidence`、`SQL` 和 `grading_method`；
   gold SQL 以这里为准。
2. [wenyupapa/BIRD-Verified-CoT-2462-GPT5.4](https://huggingface.co/datasets/wenyupapa/BIRD-Verified-CoT-2462-GPT5.4)
   的 `bird-verified-cot-2462.parquet`（2,462）。它只提供 teacher reasoning，
   不覆盖或改写 ReViSQL 的题目与 SQL。

CoT 数据集是 ReViSQL 2,462 道 verified 题的 GPT-5.4 structured reasoning 版本，


### 3.2 用法

- `train.jsonl` 用于 `configs/prime-rl/sft-bird.toml` 的 SFT train。
- `validation.jsonl` 只计算 teacher-forced validation loss，不执行 SQL，也不作为
  最终泛化分数。
- assistant 的 reasoning 和最终答案分开存储：

```json
{
  "role": "assistant",
  "reasoning_content": "teacher reasoning",
  "content": "SELECT ..."
}
```

训练 renderer 将它渲染为模型原生 thinking tokens 后接 SQL。`content` 始终只有
verified gold SQL，不含 CoT、XML、Markdown 或解释。

### 3.3 预处理

`scripts/prepare_sft_data.py bird` 执行以下步骤：

1. 读取 ReViSQL 的原始 train/validation split，不重新随机切分。
2. 按 `(db_id, question_id)` 一一连接 CoT；重复 key、缺失 CoT 或未消费的 CoT
   都会使构建失败。
3. 从对应 SQLite 和 description CSV 渲染 schema，并为每张表加入最多 3 行样例。
4. 从 CoT 的 `parsed` 字段提取 reasoning；若无结构化字段，回退到 `reasoning`
   并删除其中重复的 `#SQL`。
5. 把 ReViSQL gold SQL 写入 `assistant.content`。
6. 按 Qwen3 的真实训练模板计算完整序列长度；超过 32,768 tokens 的样本过滤掉。
7. 输出数据和 manifest，记录输入/输出 SHA-256、数量、token 数和最长样本。

构建命令：

```bash
uv run scripts/prepare_sft_data.py bird \
  --verified-train /path/to/ReViSQL/data/bird-verified-train.json \
  --verified-validation /path/to/ReViSQL/data/bird-verified-val.json \
  --cot /path/to/bird-verified-cot-2462.parquet \
  --database-root /path/to/train_databases \
  --output-dir data/processed/sft-bird-cot-sql-v1 \
  --model /path/to/Qwen3-4B-Instruct-2507 \
  --max-tokens 32768 \
  --sample-rows 3
```

已有生成物：

```text
data/processed/sft-bird-cot-sql-v1/
├── train.jsonl          # 2,064
├── validation.jsonl     #   398
└── manifest.json        # provenance、哈希、计数和 token 统计
```

当前 manifest：`missing_cot=0`、`overlong=0`、最长 32,093 tokens，总接受
10,852,629 tokens。manifest 内的绝对源路径只是生成时记录，不是新的目录要求。

## 4. RLVR 数据

### 4.1 来源与用途

RLVR 同样从 ReViSQL 的 `bird-verified-train.json` 和 `bird-verified-val.json`
开始，但不使用 GPT-5.4 CoT：

- `rlvr-v2/train.jsonl`：RLVR 在线采样的训练 taskset，2,050 题。
- `rlvr-v2/validation.jsonl`：固定 execution eval，396 题；同时用于 SFT 后评测
  和 RLVR checkpoint 选择。

每条任务保留 `question`、`evidence`、`db_id`、verified `SQL`、
`grading_method`，并新增稳定 ID、SQL 难度标签和预执行的 `gold_rows_json`。运行时
环境根据题目和数据库重新构造与 SFT 一致的 schema prompt；模型看不到 gold SQL
或 gold rows。

### 4.2 预处理

`scripts/prepare_rlvr_data.py` 对每条 gold SQL 做只读 SQLite 执行，并执行以下规则：

- 60 秒内无法成功执行：拒绝；
- gold 结果为空：拒绝，因为 execution equality 会产生大量无信息真阳性；
- 超过 100,000 行或序列化后超过 2 MB：拒绝，避免训练热路径失控；
- 保留原始行序和重复行，评分时再按该题的 `set`、`multiset`、`list` 或 `subset`
  合同比较；
- 检查 train/validation 的 `(db_id, normalized question)` 不重叠；
- 所有拒绝项写入 `rejections.jsonl`，所有输入输出哈希写入 `manifest.json`。

构建命令：

```bash
uv run scripts/prepare_rlvr_data.py \
  --verified-train /path/to/ReViSQL/data/bird-verified-train.json \
  --verified-validation /path/to/ReViSQL/data/bird-verified-val.json \
  --database-root /path/to/train_databases \
  --output-dir /path/to/rlvr-v2
```

期望输出：

```text
rlvr-v2/
├── train.jsonl          # 2,050 / 2,064
├── validation.jsonl     #   396 /   398
├── rejections.jsonl     #    16
└── manifest.json
```

16 条拒绝由 3 条空结果、1 条执行失败/超时和 12 条结果过大组成。SFT validation
是 398 而 execution validation 是 396，正是因为 validation 侧有 2 条被上述规则
拒绝，并非 split 不一致。

RLVR 的主 reward 是预测 SQL 与预缓存 gold rows 的 exact execution equality。
`format_valid`、`executable_sql`、`execution_timeout` 是诊断指标.

## 5. 最终测试数据

### 5.0 为什么主报告采用 Arcwise

[Thinking Machines/ReViSQL 的数据审计](https://thinkingmachines.ai/news/putting-task-expertise-into-rl/)
在抽查约 2,500 条 BIRD Train 后发现 52.1% 的 gold SQL 不正确。对 BIRD Mini-Dev
的两轮清洗最终检测到 52.8% 样本存在 annotation 错误；第一轮 Arcwise 清洗已经
修正 32.3%，第二轮又发现更多问题。因此：

- Arcwise-Plat-SQL 是文章正式使用、可跨项目比较的主指标；
- Arcwise-Plat 继续修正 question、evidence 和 schema descriptions，是本项目的
  干净输入能力指标；
- 原始 Mini-Dev 和 Full Dev 保留为带噪鲁棒性测试，不能单独用于判断训练是否提升。

本项目 step 1,300 在共享 498 个 ID 的两套 Arcwise 数据上从 75.50% 提升到
81.12%，paired 净增 28 题（50 gain / 22 loss，`p=0.00129`），进一步验证输入侧
annotation 噪声会显著影响测量结果。完整结果见 [`RESULTS.md`](RESULTS.md#1-revisql-bird-qwen3-4b)。

### 5.1 四套固定基准

| 基准 | 本仓库 annotation | DB/schema | 测什么 |
| --- | --- | --- | --- |
| Arcwise-Plat | `data/eval/arcwise/arcwise_plat_full_with_diff.json` | BIRD dev DB + Arcwise 校正 descriptions | SQL、问题/evidence、schema 全校正后的能力 |
| Arcwise-Plat-SQL | `data/eval/arcwise/arcwise_plat_sql_only_with_diff.json` | BIRD dev DB + 原始 descriptions | 只修 SQL label、保留输入噪声时的能力 |
| BIRD Mini-Dev | `data/eval/bird/mini_dev_sqlite.json` | BIRD dev DB + 原始 descriptions | 官方 500 题原始 Mini-Dev |
| BIRD Full Dev | `data/eval/bird/dev_20240627.json` | BIRD dev DB + 原始 descriptions | 官方 1,534 题完整 Dev |

Arcwise 两套数据来自
[uiuc-kang-lab/text_to_sql_benchmarks](https://github.com/uiuc-kang-lab/text_to_sql_benchmarks)
固定 commit `fe766045c55b6875a43b30e9ac7683df5582f8cf`。它们共享 498 个 ID，均缺少
Mini-Dev 的 ID 119 和 120，所以不能把分母写成 500。

BIRD Mini-Dev 来自 [官方 Mini-Dev 仓库](https://github.com/bird-bench/mini_dev)；
Full Dev 来自 BIRD 官方 `dev_20240627` 发布。annotation、校正 schema overlay、
许可证和 SHA-256 已 vendored；详见 [`data/eval/README.md`](../data/eval/README.md)。

### 5.2 测试时处理

最终测试不再做训练式过滤或重新切分。环境加载 annotation 后：

1. 按 `db_id` 找到 dev SQLite；Arcwise-Plat 额外覆盖校正 description CSV。
2. 构造相同的 system/user prompt，每表最多加入 3 行样例。
3. 现场执行预测 SQL；gold SQL 通常也现场执行。
4. ID 518 和 701 的 gold 查询极慢，因此从 `data/eval/slow_gold_cache.json`
   读取 gold rows；缓存由 gold SQL SHA-256 绑定，annotation 改变会直接报错。

四个配置均为 `temperature=0`、单 rollout、`push=false`，结果写入
`outputs/prime-rl/`。Base、SFT、RLVR 候选必须在同一基准上采用预先固定的生成与
thinking 协议，并做逐题 paired 比较。

## 6. 数据隔离规则

- SFT train 和 RLVR train 可以同源；两者都不能读取最终测试 annotation 做训练。
- 398 条 SFT validation 只看 loss；396 条 RLVR validation 可用于选择 checkpoint。
- 四个最终测试只在 checkpoint 和协议冻结后运行，不能据其结果返回调参。
- 不跨 split 移动题目，也不把最终测试失败样本加入 corrective SFT/RLVR。
- 报告结果时必须写清数据集名称、实际分母、annotation 版本、数据库快照、prompt、
  thinking 设置、采样参数和 timeout。
