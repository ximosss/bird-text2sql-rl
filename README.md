# BIRD Text-to-SQL SFT / RLVR

## 1. 项目介绍

本项目在 `Qwen3-4B-Instruct-2507` 上研究 BIRD Text-to-SQL 的 SFT、传统 RLVR 与
ReViSQL-BIRD 训练：

1. **SFT**：使用专家校正的题目、SQL 和 GPT-5.4 teacher reasoning 训练模型。
2. **RLVR**：执行模型生成的 SQL，根据查询结果是否正确提供 reward。
3. **ReViSQL-BIRD**：从原始 Instruct checkpoint 启动，以 BIRD-Platinum、CISPO、
   多轮数据库交互、VeriEQL 和 evidence process reward 训练。

模型输入是自然语言问题、external evidence、SQLite schema、字段描述和少量样例
行。历史 SQL-only 路线直接输出一条只读 SQL；ReViSQL 路线可在最多五轮内调用只读
SQL 工具，并以 `<solution>` 提交最终 SQL。环境负责解析、执行并比较结果集。

主流程如下：

```text
ReViSQL verified data + teacher CoT
                 │
                 ▼
              SFT 训练
                 │
                 ▼
       verified validation 选择模型
                 │
                 ▼
         execution-reward RLVR
                 │
                 ▼
 Arcwise / BIRD frozen benchmarks 最终评测
```

当前状态：ReViSQL-BIRD 已训练到 step 1,322，step 1,300 是 validation 最佳权重。
其最终 Arcwise-Plat-SQL/Arcwise-Plat 分数为 **75.50%/81.12%**。根据上游数据审计，
原始 BIRD Train/Mini-Dev 超过一半 annotation 存在错误，因此最终结论以两个
Arcwise split 为主，Mini-Dev/Full Dev 只作为带噪鲁棒性诊断。完整结果与
Thinking Machines 文章对比见 [`docs/RESULTS.md`](docs/RESULTS.md#1-revisql-bird-qwen3-4b)。

### 目录结构

```text
.
├── configs/prime-rl/
│   ├── sft-bird.toml             # 当前 SFT 配置
│   ├── rl-base-grpo-v1.toml      # 历史 RL 配置，不是当前主线
│   ├── rl-revisql-bird-qwen3-4b-v1.toml # 当前 ReViSQL-BIRD 配置
│   └── eval/                     # selection/final eval 配置
├── data/
│   ├── processed/                # 生成后的 SFT 数据，git ignored
│   └── eval/                     # 仓库内固定的最终评测 annotation
├── environments/bird_text2sql/   # Verifiers v1 Taskset、parser、executor、reward
├── scripts/                      # 数据准备、训练、评测、dashboard launcher
├── outputs/prime-rl/             # 所有训练和评测结果，git ignored
├── prime-rl/                     # 本地 Prime-RL checkout，git ignored
└── docs/                         # 数据、训练、评测和结果细节
```

## 2. 配置、安装与启动

### 2.1 运行要求

- Linux、Python 3.12、`uv`、`git`、`tmux`、`curl`、`jq`、Hugging Face CLI（`hf`）
- NVIDIA GPU 和可用的 CUDA 驱动
- SFT：1 张训练 GPU
- standalone eval：1 张推理 GPU
- 历史 RL 配置：1 张训练 GPU + 1 张推理 GPU
- BIRD train/dev SQLite databases
- 本地 Qwen3-4B-Instruct-2507 权重

本项目使用本地 Prime-RL/Verifiers v1，不上传结果，不需要 Prime API key，也不使用
W&B。所有长任务都由 `scripts/` 下的 tmux launcher 启动。

### 2.2 安装项目依赖

克隆本仓库后，在仓库根目录安装数据处理、环境和测试依赖：

```bash
uv sync --frozen
```

`prime-rl/` 不受本仓库 Git 管理。首次安装时创建固定版本的本地 checkout：

```bash
git clone https://github.com/PrimeIntellect-ai/prime-rl.git prime-rl
git -C prime-rl checkout ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1
git -C prime-rl submodule update --init --recursive
cd prime-rl
uv sync --all-extras --frozen
cd ..
```

ReViSQL 的 VeriEQL shaping 依赖官方 VeriEQL v1.0。第三方源码不 vendored 到本
仓库；把固定 tag 克隆到 launcher 约定的位置：

```bash
git clone --depth 1 --branch v1.0 \
  https://github.com/VeriEQL/VeriEQL.git third_party/VeriEQL
git -C third_party/VeriEQL rev-parse HEAD
# 期望：3b99928bf976c11cb59c9db6a177adba21f75798
```

本项目 environment 已声明运行 VeriEQL worker 所需的 Python 依赖，并对官方
z3py helper 的 context 参数差异做了兼容，不要用 VeriEQL 自带文件覆盖环境中的
`z3` 包。

安装本项目的 Verifiers environment：

```bash
prime env install bird-text2sql --path environments --plain
```

验证安装：

```bash
UV_CACHE_DIR=/tmp/bird-text2sql-rl-uv-cache uv run pytest -q
```

这项验证只需在代码、依赖或环境实现发生变化后执行，不是每个正式 run 的固定前置
步骤。

### 2.3 配置本地路径

下载基座模型；路径可以自定义，但随后必须同步修改配置和 launcher：

```bash
hf download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir /data/qwen3-4b-instruct-2507
```

BIRD SQLite 数据库从 [BIRD 官方站点](https://bird-bench.github.io/) 下载。解压后应
分别得到 train 和 dev database 目录，每个数据库目录至少包含
`<db_id>.sqlite`；字段描述位于 `database_description/*.csv`。

当前配置使用以下路径：

| 资源 | 默认路径 | 用途 |
| --- | --- | --- |
| 基座模型 | `/data/qwen3-4b-instruct-2507` | SFT、模型服务、LoRA merge |
| BIRD train databases | `/data/ximo/sql-training/train_databases` | SFT 数据构建、RLVR、selection eval |
| BIRD dev databases | `/data/ximo/minidev/MINIDEV/dev_databases` | 最终评测 |
| SFT 数据 | `data/processed/sft-bird-cot-sql-v1` | SFT train/validation |
| RLVR taskset | `/data/ximo/bird-text2sql-rl/data/rlvr-v2` | RLVR train/validation、post-SFT eval |
| 输出目录 | `outputs/prime-rl` | 训练、评测、日志、trace |

如果本机路径不同，修改对应的 TOML 和 launcher。下面的命令可以找到全部硬编码路径：

```bash
rg -n '/data/|/home/ubuntu/bird-text2sql-rl' configs/prime-rl scripts
```

路径修改原则：

- `configs/prime-rl/sft-bird.toml`：基座模型、SFT 数据和输出目录。
- `configs/prime-rl/eval/*.toml`：annotation、数据库、schema overlay 和输出目录。
- `scripts/launch_*.sh`：基座模型、adapter 和 run 目录。
- 不要修改 `data/processed/*/manifest.json` 中记录的旧绝对路径；它们只是 provenance。

### 2.4 启动 dashboard

训练和评测共用 Prime-RL dashboard：

```bash
./scripts/launch_prime_rl_dashboard.sh start
./scripts/launch_prime_rl_dashboard.sh check
```

默认地址是 <http://127.0.0.1:7788>，默认 tmux session 是
`bird-prime-dashboard`。查看后台任务：

```bash
tmux list-sessions
tmux attach -t bird-prime-dashboard
```

## 3. 数据

### 3.1 SFT 训练与 validation 数据

| Split | 来源 | 数量 | 用途 |
| --- | --- | ---: | --- |
| train | ReViSQL verified train + GPT-5.4 CoT | 2,064 | SFT 参数训练 |
| validation | ReViSQL verified validation + GPT-5.4 CoT | 398 | teacher-forced validation loss |

数据来源：

- [ReViSQL](https://github.com/uiuc-kang-lab/ReViSQL)：提供校正后的 question、
  evidence、gold SQL 和原始 train/validation split。
- [BIRD-Verified-CoT-2462-GPT5.4](https://huggingface.co/datasets/wenyupapa/BIRD-Verified-CoT-2462-GPT5.4)：
  提供与 2,462 道 verified 题逐题对应的 teacher reasoning。
- BIRD train databases：提供 SQLite、schema descriptions 和 prompt 中的样例行。

下载前两个 annotation 来源：

```bash
git clone https://github.com/uiuc-kang-lab/ReViSQL.git /path/to/ReViSQL
hf download wenyupapa/BIRD-Verified-CoT-2462-GPT5.4 \
  bird-verified-cot-2462.parquet \
  --repo-type dataset \
  --local-dir /path/to/bird-cot
```

准备原始 annotation 和 CoT 后运行：

```bash
uv run scripts/prepare_sft_data.py bird \
  --verified-train /path/to/ReViSQL/data/bird-verified-train.json \
  --verified-validation /path/to/ReViSQL/data/bird-verified-val.json \
  --cot /path/to/bird-cot/bird-verified-cot-2462.parquet \
  --database-root /data/ximo/sql-training/train_databases \
  --output-dir data/processed/sft-bird-cot-sql-v1 \
  --model /data/qwen3-4b-instruct-2507 \
  --max-tokens 32768 \
  --sample-rows 3
```

输出：

```text
data/processed/sft-bird-cot-sql-v1/
├── train.jsonl          # 2,064
├── validation.jsonl     # 398
└── manifest.json        # 输入/输出哈希、数量和 token 统计
```

预处理按 `(db_id, question_id)` 连接 ReViSQL 与 CoT，渲染 schema 和每表 3 行样例，
把 reasoning 写入 `assistant.reasoning_content`、verified SQL 写入
`assistant.content`，最后按 Qwen3 模板过滤超过 32,768 tokens 的样本。

### 3.2 RLVR 训练与 validation 数据

| Split | 来源 | 数量 | 用途 |
| --- | --- | ---: | --- |
| train | 可执行化的 ReViSQL verified train | 2,050 | RLVR rollout 和 execution reward |
| validation | 可执行化的 ReViSQL verified validation | 396 | SFT/RLVR checkpoint selection |

RLVR 不使用 teacher CoT。预处理会执行每条 gold SQL，并缓存 `gold_rows_json`；无法
执行、结果为空、超过 100,000 行或超过 2 MB 的任务会写入 `rejections.jsonl`。

```bash
uv run scripts/prepare_rlvr_data.py \
  --verified-train /path/to/ReViSQL/data/bird-verified-train.json \
  --verified-validation /path/to/ReViSQL/data/bird-verified-val.json \
  --database-root /data/ximo/sql-training/train_databases \
  --output-dir /data/ximo/bird-text2sql-rl/data/rlvr-v2
```

输出应为：

```text
rlvr-v2/
├── train.jsonl          # 2,050
├── validation.jsonl     # 396
├── rejections.jsonl     # 16
└── manifest.json
```

### 3.3 最终评测数据

最终评测 annotation 已提交在 `data/eval/`，数据库仍使用 BIRD dev databases：

| 数据集 | 数量 | 内容 |
| --- | ---: | --- |
| Arcwise-Plat | 498 | question、evidence、SQL 和 schema descriptions 全部校正 |
| Arcwise-Plat-SQL | 498 | 只校正 SQL，保留原始输入和 schema descriptions |
| BIRD Mini-Dev | 500 | 官方原始 Mini-Dev |
| BIRD Full Dev | 1,534 | 官方 2024-06-27 Dev |

四套数据只用于冻结后的最终评测，不参与训练和 checkpoint 选择。来源、文件哈希和
许可证见 [`data/eval/README.md`](data/eval/README.md)，完整预处理合同见
[`docs/DATASET.md`](docs/DATASET.md)。

## 4. 训练

### 4.1 启动 SFT

当前 SFT 配置是 `configs/prime-rl/sft-bird.toml`：Qwen3-4B、LoRA rank 16、
sequence length 32,768、1,965 optimizer steps、1 张 GPU。

确认第 2 节中的模型和数据路径已经配置后启动：

```bash
./scripts/launch_sft_training.sh bird-sft-cot-sql-v10
```

launcher 会创建同名 tmux session。查看训练：

```bash
tmux attach -t bird-sft-cot-sql-v10
```

训练输出位于：

```text
outputs/prime-rl/bird-sft-cot-sql-v10/
├── checkpoints/
├── configs/
├── logs/
└── metrics.jsonl
```

### 4.2 导出 LoRA adapter

训练完成后，把选定 checkpoint 导出为 PEFT adapter。以下示例导出最终 step 1,965：

```bash
uv run scripts/export_lora_checkpoint.py \
  --checkpoint outputs/prime-rl/bird-sft-cot-sql-v10/checkpoints/step_1965 \
  --output outputs/prime-rl/bird-sft-cot-sql-v10/adapter \
  --training-config configs/prime-rl/sft-bird.toml
```

可选：把 LoRA 合并为独立 Hugging Face 模型：

```bash
uv run scripts/merge_sft_lora.py \
  --base-model /data/qwen3-4b-instruct-2507 \
  --adapter outputs/prime-rl/bird-sft-cot-sql-v10/adapter \
  --output-dir /path/to/qwen3-4b-bird-sft-v10
```

### 4.3 RLVR

环境和 `rlvr-v2` 数据构建器已经存在，但当前仓库没有可直接启动的 SFT v10 → RLVR
配置和 launcher。不要使用 `rl-base-grpo-v1.toml` 代替：它从原始 Base 模型启动，
读取历史 `rl-v1/train.jsonl`，与当前数据合同不同。

## 5. 评测

### 5.1 SFT checkpoint selection

先按 4.2 导出 adapter，然后同时评测 Base 和 SFT：

```bash
./scripts/launch_sft_post_eval.sh bird-sft-cot-sql-v10 1965
```

该 launcher 会自动：

1. 启动本地 vLLM，同时加载 Base 和 LoRA；
2. 安装 `bird-text2sql` environment；
3. 用同一份 `rlvr-v2/validation.jsonl` 顺序评测 Base 和 SFT；
4. 把结果写入 `outputs/prime-rl/`；
5. 评测结束后关闭模型服务。

默认 tmux session 是 `bird-sft-post-eval-<run-name>`：

```bash
tmux attach -t bird-sft-post-eval-bird-sft-cot-sql-v10
```

### 5.2 Base 最终评测

一次运行全部四个冻结基准：

```bash
./scripts/launch_base_final_eval.sh
```

也可以只运行指定基准：

```bash
./scripts/launch_base_final_eval.sh bird-mini-dev arcwise-plat
```

支持的名称：`bird-mini-dev`、`bird-full-dev`、`arcwise-plat-sql`、
`arcwise-plat`。

### 5.3 SFT v10 最终评测

默认读取现有 adapter：
`outputs/prime-rl/bird-sft-cot-sql-v10-20260906-0454/adapter`。

```bash
./scripts/launch_sft_v10_final_eval.sh
```

只运行部分基准：

```bash
./scripts/launch_sft_v10_final_eval.sh bird-full-dev arcwise-plat
```

Base 配置关闭 thinking，SFT 配置开启 thinking；两类 launcher 都使用 Qwen3
reasoning parser、`temperature=0`、单 rollout 和 30 秒 prediction timeout。

### 5.4 ReViSQL step 1,300 最终评测

默认读取已导出的 step 1,300 adapter，并按 ReViSQL non-thinking、多轮 SQL tool
合同顺序运行四套冻结测试：

```bash
./scripts/launch_revisql_final_eval.sh
```

正式 run batch 为
`bird-revisql-step1300-{bird-mini-dev,bird-full-dev,arcwise-plat-sql,arcwise-plat}--20260915-004207`。
主报告采用 Arcwise-Plat-SQL 75.50% 和 Arcwise-Plat 81.12%；原始 Mini-Dev/Full
Dev 的 59.60%/62.65% 只解释为带噪 annotation 下的鲁棒性。

### 5.5 查看运行状态和结果

```bash
tmux list-sessions
tmux attach -t <session-name>
./scripts/launch_prime_rl_dashboard.sh start
```

所有结果都在 `outputs/prime-rl/<run-name>/`：

```text
configs/eval.toml      # resolved eval config
traces.jsonl           # 每题 prompt、completion、reward 和 metrics
result-summary.json    # 聚合结果（成功结束时生成）
logs/                  # 运行日志
```

主指标是 `exact_execution`。同时检查 `executable_sql`、`format_valid`、
`execution_timeout`，并在比较模型时使用相同题目的 paired gain/loss。

## 6. 更多文档

- [`docs/DATASET.md`](docs/DATASET.md)：数据来源、预处理和 split 合同
- [`docs/RL.md`](docs/RL.md)：SFT/RLVR 配置说明
- [`docs/EVALUATION.md`](docs/EVALUATION.md)：评测协议
- [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)：prompt、parser、executor 和 reward
- [`docs/RESULTS.md`](docs/RESULTS.md)：已完成实验结果
