# 数据集与数据合同

## 1. 当前数据资产总览

| 用途 | 数据集 | 数量 | 是否用于选模型 |
| --- | --- | ---: | --- |
| SFT train / validation | ReViSQL verified + BIRD-Verified-CoT-2462-GPT5.4 | 2,064 / 398 | validation 是 teacher-forced 诊断 |
| SFT post-eval taskset | `rlvr-v2` validation | 396 | 是；用于确认 SFT v10 execution 效果 |
| 最终干净侧泛化 | Arcwise-Plat / Arcwise-Plat-SQL | 498 / 498 | 否；冻结后只跑一次 |
| 最终噪声侧泛化 | 原始 BIRD Mini-Dev / Full Dev | 500 / 1,534 | 否；冻结后只跑一次 |

这里的边界是刻意的：verified validation 用来判断训练有没有工作；四个最终基准不参与调参或 checkpoint 选择，避免把最终泛化集变成开发集。

## 2. 当前 SFT 数据

当前 BIRD SFT 数据由两部分按题目一一 join：ReViSQL verified train/validation 提供问题、external evidence 与 gold SQL；BIRD-Verified-CoT-2462-GPT5.4 只提供 teacher reasoning。

```text
verified train       2,064
verified validation    398
teacher CoT           2,462
```

每条 assistant 消息为：

```json
{
  "role": "assistant",
  "reasoning_content": "teacher chain of thought",
  "content": "SELECT ..."
}
```

`content` 必须是 verified gold SQL，不含 XML、解释或 Markdown；`reasoning_content` 只含 teacher CoT。换言之，监督数据学习的是“在 reasoning channel 思考、在最终 content 只交 SQL”，而不是把 CoT 泄露到最终答案。

当前生成物：

```text
data/processed/sft-bird-cot-sql-v1/
├── train.jsonl          # 2,064
├── validation.jsonl     # 398
└── manifest.json
```

现存 manifest 记录 `contract_version=bird-cot-sql-v1`、`missing_cot=0`、`overlong=0`、最长序列 32,093 tokens、总接受 token 数 10,852,629。长度按实际 Qwen3 训练序列，即 `<think>CoT</think> + SQL` 计算；Hugging Face 原生 chat template 不读取自定义 `reasoning_content`，所以转换器不能只对 `content` 做长度审计。

重新构建：

```bash
uv run scripts/prepare_sft_data.py bird \
  --verified-train /path/to/bird_verified_train.json \
  --verified-validation /path/to/bird_verified_val.json \
  --cot /path/to/data.parquet \
  --database-root /data/ximo/sql-training/train_databases \
  --output-dir data/processed/sft-bird-cot-sql-v1 \
  --model /data/qwen3-4b-instruct-2507 \
  --max-tokens 32768 \
  --sample-rows 3
```

转换器会验证 join 唯一且完整，并把源文件与输出文件 SHA-256 写入 manifest。

## 3. SFT post-eval 任务数据

`scripts/prepare_rlvr_data.py` 从同一份 verified JSON 构建 post-SFT execution eval
使用的 Taskset。它预先只读执行 gold SQL，把保留行序和重复行的
`gold_rows_json` 写进任务；评测热路径只需执行预测 SQL。这既降低 gold 侧 I/O，
也避免并发评测时慢 gold 查询造成不稳定。

过滤规则如下，所有拒绝项都写入 `rejections.jsonl`：

- gold 无法执行或在 60 秒内超时；
- gold 结果为空；
- gold 结果超过 100,000 行或序列化后超过 2 MB。

当前本地生成物在 `/data/ximo/bird-text2sql-rl/data/rlvr-v2/`：

```text
train        2,050 / 源 2,064
validation     396 / 源   398
rejected       16
  empty_gold_result          3
  gold_invalid_or_timeout    1
  gold_result_too_large     12
```

每行保留题目 `grading_method`；环境优先使用行内 `gold_rows_json`，也兼容由题目
ID 与 SQL SHA-256 保护的 sidecar gold cache。

构建命令：

```bash
uv run scripts/prepare_rlvr_data.py \
  --verified-train /path/to/bird_verified_train.json \
  --verified-validation /path/to/bird_verified_val.json \
  --database-root /data/ximo/sql-training/train_databases \
  --output-dir /data/ximo/bird-text2sql-rl/data/rlvr-v2
```

SFT validation 是 398 题而 selection eval 是 396 题，差异正是上述 2 条 validation 拒绝项，不是漏数据。

## 4. 最终泛化基准

冻结 checkpoint 后，对 Base 和 SFT 模型统一运行：

| 基准 | 题数 | 标注环境 | 目的 |
| --- | ---: | --- | --- |
| Arcwise-Plat | 498 | SQL、问题/evidence、schema descriptions 均校正 | 良好环境下的能力上限 |
| Arcwise-Plat-SQL | 498 | SQL 校正，原始问题/evidence/schema 保留 | 隔离 SQL label 修正的影响 |
| BIRD Mini-Dev | 500 | 官方原始噪声标注 | 小规模混乱环境鲁棒性 |
| BIRD Full Dev | 1,534 | 官方 2024-06 Full Dev 原始标注 | 更广覆盖的混乱环境鲁棒性 |

Arcwise 两套数据来自 `uiuc-kang-lab/text_to_sql_benchmarks` 固定 commit `fe766045c55b6875a43b30e9ac7683df5582f8cf`。Arcwise 明确缺少原始 Mini-Dev 的 question ID 119、120，因此分母始终是 498。只有 Arcwise-Plat 使用校正后的 schema descriptions；其他三套使用数据库旁的原始 descriptions。

原始标注、license、来源与逐文件 SHA-256 见 [`data/eval/README.md`](../data/eval/README.md)。`slow_gold_cache.json` 只缓存已知极慢的 gold ID 518、701，并由 gold SQL SHA-256 绑定；预测 SQL仍然现场执行。

## 5. 历史数据资产

这些数据解释历史 run，但不进入当前 v10 → direct RLVR 主路线：

| 本地目录 | 数量 | 历史用途 |
| --- | ---: | --- |
| `rl-v1` | source 6,601；valid nonempty 6,496；train 4,904；ID/OOD 各 150 | 旧 SQL-only online GRPO；14 个 schema-OOD DB |
| `sft-v2-wide` | train 914,320；validation 1,836 | SynSQL Think wide CoT SFT |
| `sft-v2-bird` | train 2,064；validation 398 | 旧 SQL-only direct SFT target |
| `sft-v3-bird-corrective` | train 2,064；validation 398 | 1,407 条 Base 正确回放 + 643 条 teacher SQL 修复 + 14 条未匹配补齐 |
| `sft-v4-bird-error-correction` | train 2,729；unique 2,064 | 错题/teacher target 加权重复 |
| `sft-v5-bird-base-replay` | train 2,064；unique 2,064 | 去掉额外重复后的 Base replay 混合 |
| `sft-v6-bird-sql-emphasis` | train 2,729；unique 2,064 | 再加入 665 条 SQL-emphasis copy |

构建 corrective 系列所用的有效 Base rollout 是 `sft-train-base-v3--20260904-201932`：2,050/2,050 trace 成功，1,407 exact、643 non-exact。对应训练与结果解释见 [RESULTS.md](RESULTS.md)。这些目录在本地数据盘上，源数据、数据库和模型均不随仓库发布；复现时以各目录 `manifest.json` 的哈希为准。
