# 开发说明

## 本地验证

```bash
uv sync
prime env install bird-text2sql --path environments --plain
uv run pytest -q
```

当前根级测试覆盖 SFT 数据转换、prompt token 审计、post-eval taskset/gold cache，
以及冻结泛化数据和配置合同。环境包测试覆盖 parser、schema、只读执行、reward 和
Taskset。

## 当前代码地图

| 文件 | 责任 |
| --- | --- |
| `environments/bird_text2sql/bird_text2sql/` | Text-to-SQL Taskset、prompt、parser、schema、只读执行与评分 |
| `scripts/prepare_sft_data.py` | 构建 SFT v10 的 reasoning/content 分离数据和 manifest |
| `scripts/audit_prompt_tokens.py` | 训练前审计渲染后的序列长度 |
| `scripts/export_lora_checkpoint.py` | 从 Prime-RL trainer checkpoint 导出 PEFT adapter |
| `scripts/merge_sft_lora.py` | 将 SFT v10 adapter 合并为独立 HF 模型 |
| `scripts/prepare_rlvr_data.py` | 构建 SFT post-eval 使用的 verified taskset 和 gold cache |
| `scripts/launch_sft_post_eval.sh` | 在 tmux 中运行 Base/SFT paired execution eval |
| `scripts/launch_prime_rl_dashboard.sh` | 启动统一的本地 Prime-RL viewer |

环境包公开：

```python
from bird_text2sql import BirdText2SQLTaskset, load_environment
```

生产路径使用 `BirdText2SQLTaskset`；`load_environment` 保留生态兼容性。

## 修改合同

- SFT label/split 以 verified JSON 为准，teacher CoT 只提供 reasoning。
- `content` 始终只有一条只读 SQLite 查询。
- parser、executor 或 grading 变化必须补相应单元测试。
- gold cache contract 变化时提高版本并重建 taskset。
- 所有数据生成物保存 source/output SHA-256。
- standalone eval 使用 `prime-rl/` 环境中的 `uv run --no-sync eval @ <config>`，
  `push=false`，输出写入 `outputs/prime-rl/`。
- 长期任务和 dashboard 通过 tmux launcher 启动。

VeriEQL 接口仍作为环境层的可选兼容代码存在，但当前所有保留配置均设置
`use_verieql=false`；仓库不携带第三方 VeriEQL checkout。

下载的数据、数据库、模型、`outputs/` 和 `prime-rl/` 均不提交到根 Git。
