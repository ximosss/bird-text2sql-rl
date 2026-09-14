# 评测

本项目有三种不同的“评测”。混用它们会导致错误结论。

| 类型 | 数据 | 指标 | 用途 |
| --- | --- | --- | --- |
| SFT validation | 398 条带 teacher target 的 SFT validation | teacher-forced loss | 诊断拟合/过拟合 |
| checkpoint selection | 396 条 `rlvr-v2` validation | exact execution | 选择 SFT/RLVR checkpoint |
| final test | 498/498/500/1,534 四套冻结基准 | exact execution + paired analysis | Arcwise 测干净泛化；原始 BIRD 测带噪鲁棒性 |

## 1. SFT 后 execution eval

SFT 完成并导出 adapter 后，使用现有 tmux launcher：

```bash
./scripts/launch_sft_post_eval.sh bird-sft-cot-sql-v10-20260906-0454 1965
```

launcher 启动带 `--reasoning-parser qwen3` 的本地 vLLM，并顺序运行：

- Base：`configs/prime-rl/eval/platinum-base-v2.toml`，`enable_thinking=false`；
- SFT：`configs/prime-rl/eval/platinum-v2.toml`，`enable_thinking=true`。

两者读取同一个 `rlvr-v2/validation.jsonl`，使用 `temperature=0`、
`max_tokens=2048`、单 rollout、相同数据库与 execution scorer。Base-off / SFT-on
是已完成 v10 评测采用的部署协议；比较其他模型时必须先固定并记录自己的协议。

正式 v10 batch：

| 模型 | exact execution | executable | format valid | timeout |
| --- | ---: | ---: | ---: | ---: |
| Base | 254/396 = 64.14% | 353/396 | 395/396 | 27/396 |
| SFT step 1,965 | 286/396 = 72.22% | 370/396 | 396/396 | 3/396 |

paired flips 为 65 gain / 33 loss，净增 32 题，McNemar exact `p=0.0016`。

## 2. RLVR validation

RLVR 必须继续使用同一份 396 题 validation，并在 step 0 与每个候选 checkpoint
上跑完整 taskset。不要用 train reward 均值代替 held-out execution accuracy，也
不要根据最终四基准选择 RLVR step。

至少报告：

- exact execution 的分子、分母和百分比；
- 相对初始化 checkpoint 的百分点变化；
- paired gain/loss 与净变化；
- paired 95% CI 和 McNemar exact p-value；
- format valid、executable SQL、timeout；
- run name、checkpoint step、配置文件与模型 ID。

## 3. 最终测试

冻结模型、prompt、thinking、采样和 timeout 后，分别运行：

| 配置 | 数据量 | 数据合同 |
| --- | ---: | --- |
| `eval/arcwise-plat.toml` | 498 | SQL、问题/evidence、schema descriptions 校正 |
| `eval/arcwise-plat-sql.toml` | 498 | 只校正 SQL |
| `eval/bird-mini-dev.toml` | 500 | 官方原始 Mini-Dev |
| `eval/bird-full-dev.toml` | 1,534 | 官方原始 Full Dev |

最终结论以两个 Arcwise split 为主。ReViSQL/Thinking Machines 的审计发现 BIRD
Train 有 52.1% 的 gold SQL 错误，BIRD Mini-Dev 的总检测错误率为 52.8%；文章
也以 Arcwise-Plat-SQL 作为正式主榜。原始 Mini-Dev/Full Dev 仍完整报告，但只解释为
对噪声 annotation 的鲁棒性。依据见
[原文](https://thinkingmachines.ai/news/putting-task-expertise-into-rl/)和
[`RESULTS.md`](RESULTS.md#12-为什么以-arcwise-为主)。

项目规定的单次 eval 命令本体是：

```bash
cd prime-rl
uv run --no-sync eval @ ../configs/prime-rl/eval/arcwise-plat.toml
```

正式长任务必须由仓库 tmux launcher 启动。Base 的四套测试使用
`scripts/launch_base_final_eval.sh` 以及对应的 `*-base.toml`，统一关闭 thinking，
并由带 Qwen3 reasoning parser 的 vLLM 提供服务：

```bash
./scripts/launch_base_final_eval.sh
```

SFT v10 的四套测试使用对应的候选配置，由 launcher 显式覆盖模型 ID；它加载
step-1,965 LoRA、开启 thinking，并检查 reasoning parser：

```bash
./scripts/launch_sft_v10_final_eval.sh
```

ReViSQL checkpoint 必须使用自己的模型合同，不能套用旧 SFT 的 thinking/SQL-only
配置。step 1,300 的正式四测由以下 launcher 顺序执行：

```bash
./scripts/launch_revisql_final_eval.sh
```

对应配置为 `eval/revisql-{bird-mini-dev,bird-full-dev,arcwise-plat-sql,arcwise-plat}.toml`；
它们固定 `enable_thinking=false`、`temperature=0`、单 rollout、最多五轮、每轮
3,072 output tokens，并启用只读 SQL tool。

四套结果逐项报告，不取简单平均。首要外部可比指标是 Arcwise-Plat-SQL；进一步
全校正后的能力指标是 Arcwise-Plat。Arcwise 两套的分母是 498，不是 500；它们与
BIRD Mini-Dev 的标注合同不同。完整来源见 [`DATASET.md`](DATASET.md)。

## 4. 执行评分口径

- 只接受单条只读 SQLite `SELECT`/`WITH` 作为格式有效输出。
- 预测 SQL 在只读 SQLite 中现场执行。
- gold rows 来自 taskset 内缓存或 SHA-256 绑定的 sidecar cache。
- 按每题 `grading_method` 使用 `set`、`multiset`、`list` 或 `subset` 比较。
- `exact execution` 是主指标；可执行但结果错误仍记 0。
- scorer timeout 与模型输出截断分开记录。

所有结果写入 `outputs/prime-rl/`，`push=false`，通过同一个 Prime-RL dashboard
检查 trace 与聚合指标。

## 5. 评测前检查的复用

正式 eval 不要求每次先创建一个 check run。代码 commit、resolved eval config、
taskset manifest、数据库快照和模型 ID 相同，就复用最近一次成功的 preflight；
重新启动或改 run name 不触发新检查。只有这些输入发生变化，或模型服务/reasoning
parser 有改动时，才先跑少量样本验证链路。少量样本结果只叫 smoke，不计入正式
结果。
