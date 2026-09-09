# SFT v10 训练后 execution eval

SFT 期间只计算 teacher-forced validation loss。execution eval 在最终 adapter
导出后独立运行。

## 已验证结果

有效批次为 `bird-sft-post-{base,step1965}--20260906-125601`，使用相同的396道
verified validation 题：

| 模型 | exact execution | executable | format valid | timeout |
| --- | ---: | ---: | ---: | ---: |
| Base | 254/396 = 64.14% | 353/396 | 395/396 | 27/396 |
| SFT v10 step 1,965 | 286/396 = 72.22% | 370/396 | 396/396 | 3/396 |

逐题配对为65 gain / 33 loss，净增32题，McNemar exact p=0.0016。

## 运行合同

- Base 使用 `eval/platinum-base-v2.toml`，`enable_thinking=false`。
- SFT 候选使用 `eval/platinum-v2.toml`，`enable_thinking=true`。
- 两者使用相同 taskset、`temperature=0`、`max_tokens=2048` 和 execution scorer。
- 结果写入 `outputs/prime-rl/`，`push=false`，不启用 W&B。
- 本项目通过 `prime-rl/` 环境运行 Verifiers v1 eval，不使用 `prime eval run`。

启动入口：

```bash
./scripts/launch_sft_post_eval.sh <run-name> 1965
```

四个冻结泛化基准尚未正式运行，不能用来反向选择 checkpoint。
