# 开发说明

## 1. 本地验证

```bash
uv sync
prime env install bird-text2sql --path environments --plain
uv run pytest -q
```

当前测试覆盖：SFT 流式转换、RLVR gold cache、SQL-only 与 legacy parser、evidence 编号、SQL 安全、BIRD Platinum grading methods、legacy/shaped reward、unknown verifier 行为、schema 渲染和 token 审计。

只跑环境或数据脚本：

```bash
uv run pytest environments/bird_text2sql/tests -q
uv run pytest tests -q
```

在某些受限 sandbox 中，`asyncio` thread executor 内执行 SQLite 会卡住；环境测试通过 monkeypatch 内联 SQLite 来隔离该平台限制，executor 本身另有同步数据库测试。生产 Taskset 仍在线程池中执行 SQLite。Prime-RL smoke eval 必须作为最终异步集成验证。

## 2. 代码地图

| 文件 | 责任 |
| --- | --- |
| `prompts.py` | 当前 SQL-only system/user prompt、SFT reasoning/content 分离；四块 helper 仅兼容历史 |
| `parser.py` | SQL-only 才标记 format-valid；旧 XML/tag/fence/裸 SQL fallback 可执行但格式无效 |
| `schema.py` | SQLite 定位、描述 CSV、schema 和样例行 |
| `executor.py` | 只读执行、ordered gold rows、Platinum grading methods |
| `verieql.py` / `_verieql_worker.py` | async subprocess bridge、schema 映射、timeout、z3py 兼容 |
| `data.py` | JSON/JSONL 规范化、schema overlay、行内/sidecar gold cache |
| `taskset.py` | 生产 Taskset、线程池 SQLite、共享 score task、reward 与 metrics |
| `rewards.py` / `bird_text2sql.py` | v1 legacy compatibility，不进入 v2 |
| `prepare_sft_data.py` | BIRD SFT 数据与 manifest；wide 仅供历史复现 |
| `prepare_rlvr_data.py` | verified task 校验、gold result cache 与 manifest |
| `merge_sft_lora.py` | 把选定 SFT LoRA safe-merge 为 direct RLVR 的独立 HF 起点 |
| `launch_training_v2.sh` | direct BIRD SFT / RLVR tmux launcher |

环境包必须继续公开：

```python
from bird_text2sql import BirdText2SQLTaskset, load_environment
```

新代码只应依赖 `BirdText2SQLTaskset`；`load_environment` 是历史入口。

## 3. 修改合同的检查项

### Prompt / parser

- 同时修改 SFT target builder 与 RL parser；
- 为合法 SQL-only、XML/tag/fence fallback、非 SQL 文本补测试；若修改 legacy 四块兼容，再覆盖缺块与 evidence 编号；
- 重新做 prompt/token 审计；
- 重跑同一 verified validation 上的 step-0 Base、SFT 和 RLVR 对照，因为 prompt 变化会改变基线。

### Executor / grading

- 覆盖 set、multiset、list、subset 与数量约束；
- 明确是否保留行顺序、重复、列顺序和浮点容差；
- gold cache contract 变化时提高 `CONTRACT_VERSION` 并重新构建全部 RLVR data；
- parser recursion、SQL timeout 和 unsafe SQL 必须返回 0，而非 scorer error。

### Reward / VeriEQL

- 正式 eval source 必须保持 `shape_reward=false`；当前 direct RLVR 也保持关闭；
- VeriEQL 只在 exact execution 后调用，避免为明显错误浪费 Z3 成本；
- `False`、`unknown` 和 infrastructure unavailable 不能混为一类；
- 变更 bound、penalty 或 evidence split 后同时改文档、测试和 run tags；
- 不要把 reasoning 长度或 LLM-as-judge 分数加入 reward，除非先定义可审计的失败模式。

### 数据

- SFT label/split 以 ReViSQL verified JSON 为准，CoT 源只提供 reasoning；
- 所有 source/output 保存 SHA-256；
- 不提交下载的数据、数据库、模型或 VeriEQL checkout；
- license 与数据卡快照属于实验记录的一部分。

## 4. 验证顺序

```text
unit tests
  →（仅 use_verieql=true 时）setup/check VeriEQL
  → prime env install
  → SFT/RL config model_validate + --dry-run
  → 5-task v2 eval smoke（tmux）
  → 2–5 step GPU train smoke（tmux）
  → fixed probe
  → full stage run
  → verified validation paired selection eval
  → 冻结 checkpoint
  → Arcwise-Plat / Arcwise-Plat-SQL / BIRD Mini-Dev / Full Dev 最终 paired eval
```

项目覆盖 Prime 通用默认：standalone eval 使用 `prime-rl/` 环境中的 `uv run --no-sync eval @ <config>`，由 tmux launcher 启动；`push=false`，新产物写 `outputs/prime-rl/`。不要使用 `prime eval run`、`configs/endpoints.toml` 或 `outputs/evals/`。

## 5. 脚本状态

| 脚本 | 状态 | 说明 |
| --- | --- | --- |
| `prepare_sft_data.py` | 主流程 | direct BIRD SFT；wide 子命令仅供历史复现 |
| `prepare_rlvr_data.py` | v2 主流程 | verified RLVR tasks |
| `setup_verieql.sh` | 可选 shaped 路径 | 代理下载、固定官方 commit；当前正式配置未启用 |
| `check_verieql.py` | 可选 shaped 路径 | `use_verieql=true` 时的 fail-fast probe |
| `merge_sft_lora.py` | 主流程 | 把选定的 BIRD SFT LoRA 合并为独立 HF 模型（RLVR 起点） |
| `launch_training_v2.sh` | 主流程 | direct BIRD SFT / direct RLVR tmux launcher |
| `launch_sft_post_eval.sh` | 主流程 | 训练后 Base/候选 paired execution eval |
| `launch_eval_server.sh` | legacy/待对齐 | Base + 两个 LoRA endpoint；默认指向旧 run，且尚未启用 Qwen3 reasoning parser |
| `launch_prime_rl_eval.sh` | 数据路径已就绪、协议待对齐 | `generalization` suite 已接四基准，但当前 Base/候选共用 thinking-on TOML；`direct`/`full-dev`/`v2`/`sft-base*` 为历史 |
| `launch_prime_rl_dashboard.sh` | 共用 | 统一本地 viewer |
| `export_lora_checkpoint.py` | 按需 | 从 trainer checkpoint 导出 adapter |
| `prepare_rl_data.py` / `launch_rl_base_grpo_v1.sh` | legacy | SQL-only v1 复现 |
| `rescore_bird_eval.py` / `sync_remote_experiment.sh` | legacy/机器相关 | 不进入 v2 |

## 6. 配置与实验记录

不要覆盖已有基线 TOML。每个正式 run 至少保存：

- 源码 commit/dirty diff；
- Prime-RL、Verifiers、VeriEQL commit 和 dependency lock；
- resolved config；
- 三份 data manifest 及 source/output SHA-256；
- base model 路径和 fingerprint；
- 父 checkpoint、selection 原因和 paired eval run names；
- unknown/refuted/timeout/format 指标。

`outputs/`、`data/`、`prime-rl/` 和 `third_party/VeriEQL/` 不随源码发布。源码 checkout 本身不足以复现实验。
