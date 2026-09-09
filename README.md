# BIRD Text-to-SQL SFT / RLVR

当前主路线是：从 `Qwen3-4B-Instruct-2507` 开始，用 teacher CoT + verified gold SQL 做完整 BIRD SFT；用独立 execution eval 选择模型；再从合并后的 SFT 权重做 direct RLVR；最后在四个冻结基准上评测泛化。SFT 已完成：`bird-sft-cot-sql-v10-20260906-0454` 在 verified validation（396 题）上相对 Base 取得 +8.08 pp（64.14% → 72.22%，paired p=0.0016）。首轮 direct RLVR `bird-direct-rlvr-v3-20260906-2331` 已完成，但所有 checkpoint 都未取得显著 paired 正收益，正式选择仍是 merged SFT v10；signal-dense boundary RLVR v4 尚待 run card 批准。四基准最终评测仍未运行，不能提前宣称最终泛化结果。完整记录与分析见 [`docs/RESULTS.md`](docs/RESULTS.md)。

旧的 `bird-sqlonly-online-*`、v2–v9、base replay、corrective、XML 四块协议和边训边测配置只用于追溯，不是新实验的入口。

## 当前合同

- system prompt 要求最终回答只有一条只读 SQLite `SELECT` 或 `WITH`。
- SFT 样本把 teacher CoT 放在 `assistant.reasoning_content`，把 gold SQL 单独放在 `assistant.content`。
- Qwen renderer 训练 `<think>teacher CoT</think>` 后的 SQL，但推理 API 用 reasoning parser 将思考与最终 content 分开；最终 content 只有 SQL。
- 不训练 `requirements`、`verification`、XML、Markdown 或工具调用。
- SFT 不启动 inference server，不做 rollout/online eval；只保留低成本 teacher-forced validation loss。
- 完成最终 checkpoint 后导出 LoRA，再单独运行 Base 与候选的 execution eval。
- 只启用 Prime-RL file monitor/dashboard；不启用 W&B。
- 已完成主 SFT 为 1,965 packed optimizer steps，约六次完整数据遍历；首轮 direct RLVR 已跑满 240 steps 但未建立显著增益，下一轮 v4 见 [`docs/RUN_CARD_RLVR_V4.md`](docs/RUN_CARD_RLVR_V4.md)。

## 数据

当前本地数据为 `data/processed/sft-bird-cot-sql-v1`：

```text
train       2,064
validation    398
missing CoT     0
overlong        0
```

重新构建：

```bash
uv run scripts/prepare_sft_data.py bird \
  --verified-train /path/to/ReViSQL/data/bird_verified_train.json \
  --verified-validation /path/to/ReViSQL/data/bird_verified_val.json \
  --cot /path/to/BIRD-Verified-CoT-2462-GPT5.4/data.parquet \
  --database-root /data/ximo/sql-training/train_databases \
  --output-dir data/processed/sft-bird-cot-sql-v1 \
  --model /data/qwen3-4b-instruct-2507 \
  --max-tokens 32768 \
  --sample-rows 3
```

评测与 RLVR 不直接用这份 SFT 数据，而是用同一 verified 源构建的 RLVR-v2 任务集
（validation 396 / train 2,050，gold 结果预缓存），见
[`docs/DATASET.md`](docs/DATASET.md)。

## 训练

配置是 [`configs/prime-rl/sft-bird.toml`](configs/prime-rl/sft-bird.toml)。所有长任务通过 tmux launcher 启动：

```bash
./scripts/launch_training_v2.sh bird start
./scripts/launch_training_v2.sh bird check
```

训练成功后 launcher 自动把最终 DCP checkpoint 导出到：

```text
outputs/prime-rl/<run-name>/adapter/
```

## 训练后 execution eval

只有 adapter 已导出后，下面的命令才会启动 Base/候选模型服务并顺序评测：

```bash
./scripts/launch_sft_post_eval.sh <run-name> 1965
```

评测使用项目规定的 Verifiers v1 路径 `cd prime-rl && uv run --no-sync eval @ <config>`，结果仍写入 `outputs/prime-rl/` 并由同一个 dashboard 查看。v10 的有效 paired batch 是 `bird-sft-post-{base,step1965}--20260906-125601`（Base 用 `platinum-base-v2.toml` 关闭 thinking，候选用 `platinum-v2.toml` 开启 thinking），结果见 [`docs/RESULTS.md`](docs/RESULTS.md)。

上面的 ReViSQL verified validation 只用于模型选择。确定 checkpoint 后，应使用
四个冻结基准做一次最终泛化评测。数据与 TOML 已准备，但截至 2026-09-07 还不能
把现有通用 launcher 直接用于 v10：`launch_eval_server.sh` 未启用 Qwen3 reasoning
parser，且 generalization 配置会让 Base 与 thinking 候选共用
`enable_thinking=true`。这与有效 post-SFT paired eval 的 Base-off / candidate-on
协议不一致；在脚本和配置分离前不要启动正式 batch。

它包含良好/已校正环境侧的 Arcwise-Plat、Arcwise-Plat-SQL，以及混乱环境侧的原始 BIRD
Mini-Dev（500）和原始 BIRD Full Dev（1,534）。数据来源、哈希和 498/500 的
ID 差异见 [`data/eval/README.md`](data/eval/README.md)，评测解释见
[`docs/EVALUATION.md`](docs/EVALUATION.md)。

## 验证

```bash
uv run pytest -q
cd prime-rl
uv run --no-sync sft @ ../configs/prime-rl/sft-bird.toml \
  --run.name sft-bird-config-check --dry-run
```

更多细节见：

- [数据格式](docs/DATASET.md)
- [输出与评分](docs/ENVIRONMENT.md)
- [训练配置](docs/RL.md)
- [RLVR v4 run card](docs/RUN_CARD_RLVR_V4.md)
- [训练后评测](docs/EVALUATION.md)
- [正式 run、基线、结果与分析](docs/RESULTS.md)
- [开发与代码地图](docs/DEVELOPMENT.md)
