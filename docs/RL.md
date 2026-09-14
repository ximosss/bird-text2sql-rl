# 训练

仓库同时保留历史 SFT → RLVR 路线和当前 ReViSQL-BIRD 路线。当前最佳路线从原始
Qwen3-4B-Instruct-2507 直接启动，在 BIRD-Platinum 上用 CISPO、数据库交互、
VeriEQL 和 evidence process shaping 训练，不依赖 SFT-v10。

## 1. SFT

### 输入

- 基座：`/data/qwen3-4b-instruct-2507`
- 配置：`configs/prime-rl/sft-bird.toml`
- train：`data/processed/sft-bird-cot-sql-v1/train.jsonl`，2,064 条
- validation：同目录 `validation.jsonl`，398 条

每个 target 包含 teacher `reasoning_content` 和 verified SQL `content`。只对
assistant tokens 计算 loss；validation 是 teacher-forced loss，不是 execution
accuracy。完整数据配方见 [`DATASET.md`](DATASET.md)。

### 训练配置

| 项目 | 值 |
| --- | --- |
| 模型 | Qwen3-4B-Instruct-2507 |
| 方法 | LoRA，rank 16、alpha 32、dropout 0.05 |
| sequence length | 32,768 |
| batch / micro batch | 1 / 1 |
| optimizer | AdamW，lr `3e-5`，cosine decay |
| warmup / min lr | 100 steps / `3e-6` |
| steps | 1,965 |
| checkpoint interval | 327 |
| inference GPU | 0；SFT 期间不做 rollout |

长任务按项目约定从 tmux launcher 启动，输出写到 `outputs/prime-rl/<run-name>/`，
由 file monitor/dashboard 读取：

```bash
./scripts/launch_sft_training.sh <run-name>
```

训练完成后，把最终 DCP checkpoint 导出为标准 PEFT adapter；已有 run 的产物位于：

```text
outputs/prime-rl/bird-sft-cot-sql-v10-20260906-0454/adapter/
```

如需合并为独立 Hugging Face 模型：

```bash
uv run scripts/merge_sft_lora.py \
  --base-model /path/to/Qwen3-4B-Instruct-2507 \
  --adapter outputs/prime-rl/bird-sft-cot-sql-v10-20260906-0454/adapter \
  --output-dir /path/to/qwen3-4b-bird-sft-v10
```

## 2. ReViSQL-BIRD RLVR

### 当前主线

- 初始化模型：原始 `/data/qwen3-4b-instruct-2507`。
- train taskset：`rlvr-v2/train.jsonl`，2,050 条。
- validation taskset：`rlvr-v2/validation.jsonl`，396 条。
- rollout：最多五轮，模型可调用只读 `execute_sql_query`，最终输出
  `<solution>SQL</solution>`。
- reward：execution equality；仅 VeriEQL 明确 refute 时下调 0.2，并对未满足
  requirement/verification evidence 合同分别施加 0.1 shaping。
- 优化：CISPO、group 16、batch 64、LoRA rank 32、learning rate `1e-5`。
- renderer：canonical Jinja、`enable_thinking=false`，无 reasoning parser/prefill。

RLVR 数据不含 teacher CoT。`gold_rows_json` 只供 scorer 使用，不会进入模型 prompt。
训练与 validation 必须使用同一个 environment/parser/executor 合同。配置与 launcher：

```text
configs/prime-rl/rl-revisql-bird-qwen3-4b-v1.toml
scripts/launch_revisql_bird.sh
```

### 仓库状态

`configs/prime-rl/rl-base-grpo-v1.toml` 是旧的 base → SQL-only GRPO 实验，读取旧
`rl-v1` 数据；它不是当前主线。历史 SFT-v10 → RLVR 入口为
`configs/prime-rl/rl-sft-v10-platinum-g16-v1.toml` 与
`scripts/launch_rl_sft_v10_platinum_g16.sh`。首个 100-step pilot 已完成，结果与
限制见 [RESULTS.md](RESULTS.md#4-sft-v10--rlvr-100-step-pilot)。

当前 ReViSQL run 为
`bird-revisql-qwen3-4b-promptfmt-v4-pilot40-20260913`。训练完成到 step 1,322，
validation 最佳点是 step 1,300：319/396 = 80.56%。其导出 adapter 已完成四套
冻结测试，Arcwise-Plat-SQL/Arcwise-Plat 分别为 75.50%/81.12%；详见
[RESULTS.md](RESULTS.md#1-revisql-bird-qwen3-4b)。

所有正式配置固定：

- merged SFT 模型路径；
- `rlvr-v2/train.jsonl` 与 `rlvr-v2/validation.jsonl`；
- group size、batch size、采样温度、completion 长度、KL 系数；
- SQL timeout、并发度、checkpoint/eval interval；
- renderer/tool parser 与 thinking 设置；
- `push=false`、file monitor、`outputs/prime-rl/`。

## 3. 模型选择

SFT 和 RLVR 都在相同的 396 题 verified execution validation 上比较 Base/前一阶段
模型与候选 checkpoint。主指标是 exact execution；同时报告 paired gain/loss、
置信区间和 McNemar exact p-value。398 条 SFT validation loss 只用于诊断是否拟合
或过拟合。

最终的 Arcwise/BIRD 四基准只在模型和推理协议冻结后运行一次，不参与模型选择。

## 4. 监控

```bash
./scripts/launch_prime_rl_dashboard.sh
```

训练与 eval 均写 `outputs/prime-rl/`。保持 file monitor 为唯一事实来源，不启用
W&B，也不把新 run 写到历史 `outputs/evals/`。

## 5. Check 与正式 run

check 绑定实验版本，不绑定每次启动。用下面四项共同标识一个已经验证过的版本：

- code commit；
- resolved config SHA-256；
- dataset manifest SHA-256；
- base model 或初始化 checkpoint ID。

四项都没有变化时，重复运行、断点恢复、机器重启后续跑或只改 run name，都不需要
重新 check。正式 run 的目录只记录其复用的 preflight 结果，不额外创建
`*-config-check` run。

只有以下情况需要重新检查：

- 修改配置、环境代码、renderer、prompt、scorer 或数据；
- 更换基座模型或初始化 checkpoint；
- 依赖、训练后端、模型服务或 GPU 拓扑变化；
- 上一次 check 失败，且相关问题已经修复。

其中静态配置/路径检查只需验证解析和输入完整性；1–2 step smoke 只用于训练栈、
模型或硬件发生变化的情况。check 和 smoke 都不是正式实验，不进入结果表，也不能
代替完整 validation。
