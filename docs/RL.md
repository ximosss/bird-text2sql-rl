# BIRD SFT v10 训练合同

当前唯一保留的主线是 direct BIRD CoT+SQL SFT：

- 配置：[`sft-bird.toml`](../configs/prime-rl/sft-bird.toml)
- run：`bird-sft-cot-sql-v10-20260906-0454`
- 基座：Qwen3-4B-Instruct-2507
- LoRA：rank 16 / alpha 32 / dropout 0.05
- sequence length：32,768
- optimizer steps：1,965
- batch / micro batch：1 / 1
- optimizer：AdamW，lr `3e-5`，cosine，100-step warmup，min lr `3e-6`
- monitor：本地 file monitor

训练只对 assistant token 计算 loss。`reasoning_content` 使用模型原生 thinking
token，最终 `content` 只包含 verified gold SQL。训练期间不启动 rollout eval。

最终 PEFT adapter 位于：

```text
outputs/prime-rl/bird-sft-cot-sql-v10-20260906-0454/adapter/
```

训练后的正式 paired execution eval 为 Base `64.14%`、SFT v10 `72.22%`
（+8.08 pp，McNemar exact p=0.0016）。teacher-forced validation loss 只用于
诊断拟合，不替代 execution eval。

如需生成独立 merged HF 模型：

```bash
uv run scripts/merge_sft_lora.py \
  --adapter outputs/prime-rl/bird-sft-cot-sql-v10-20260906-0454/adapter \
  --output-dir /data/ximo/bird-text2sql-rl/models/qwen3-4b-bird-sft-v10
```

旧 SQL-only SFT/GRPO 文件仅作为历史资产，不是当前入口。
