# 实验结果与历史基线

## 0. 当前状态与正式 run 索引

direct BIRD SFT（teacher CoT + verified gold SQL）已完成，并通过独立 paired
execution eval 验证：SFT step 1965 相对 Base 在 ReViSQL verified validation
（RLVR-v2 口径，396 题）上 exact execution 从 64.14% 升到 72.22%
（+8.08 pp，paired 65 gain / 33 loss，McNemar exact p = 0.0016）。详见 §1。

这里把“正式 run”定义为产生了可分析的完整训练阶段或完整评测 taskset；
config-check、dry-run、1–2 题/步 smoke、启动失败和被中断的探索 run 单独保留记录，
但不混入主结果。按时间顺序索引如下：

| Run | 类型 | 状态 | 摘要 |
| --- | --- | --- | --- |
| `bird-sqlonly-online-20260829-191739-24054` | SQL-only online GRPO | 完成 120 steps | 旧协议；online step 80 最佳，见 §3 |
| `direct-{id,ood,full-dev}-{base,step80,step120}--20260830-163442` | 独立评测 batch | 完成 | 见 §3 |
| `bird-sft-wide-v2-20260903-182552-21766` | wide CoT SFT | step 1,001/16,000 时停止 | 明显负迁移，见 §2 |
| `sft-train-base-v3--20260904-201932` | Base 全训练集 rollout/eval | 完成 2,050 题 | corrective 数据来源，见 §1.8 |
| v3–v7 系列（20260904–20260905） | 旧 SQL-only SFT 数据混合探索 | 中断 | 各 run 12–49 steps，见 §1.8 |
| `bird-sft-clean-v8-20260905-015311-27261` | SQL-only SFT | 完成 90 steps | 最佳 online 点仍无稳定优势，见 §1.8 |
| `bird-sft-clean-v9-20260905-123445-10941` | SQL-only SFT | step 170/510 失败 | broadcast receiver timeout，见 §1.8 |
| `bird-sft-cot-sql-v10-20260906-0454` | direct BIRD CoT+SQL SFT（当前主路线） | 完成 | 1,965 steps，见 §1.1 |
| `bird-sft-post-{base,step1965}--20260906-125601` | 训练后 paired execution eval | 完成 | +8.08 pp，见 §1.2–1.5 |

另有 2026-08 历史 `outputs/evals/` direct-GRPO batch，因运行栈与 scorer 更旧，
独立列在 §4。最终 Arcwise/BIRD 四基准截至 2026-09-07 尚未正式运行，而且通用
launcher 的 reasoning parser 与 Base/候选 thinking 分流仍待对齐，见
[EVALUATION.md](EVALUATION.md)。

以下结果均来自本项目环境，不应直接冒充官方 BIRD leaderboard 分数；口径限制统一见 §5。

## 1. direct BIRD CoT+SQL SFT（当前主路线）

### 1.1 训练记录：`bird-sft-cot-sql-v10-20260906-0454`

- 配置：[`sft-bird.toml`](../configs/prime-rl/sft-bird.toml)，resolved config 与
  仓库版本逐字节一致（run 目录 `configs/sft.toml`）。
- 数据：`data/processed/sft-bird-cot-sql-v1`（train 2,064；validation 398；
  manifest 共 10,852,629 accepted tokens；teacher CoT 放
  `reasoning_content`，gold SQL 单独放 `content`）。
- 训练：1 GPU，1,965 packed optimizer steps（约 6 个 epoch），2026-09-06
  04:55–12:24，约 7.5 小时；吞吐约 591 tok/s/GPU，峰值显存 24.4 GiB，MFU 约 10%。
- teacher-forced train loss：step 1 的 1.956（ppl 7.07）降到 step 1,965 的
  0.194（ppl 1.21），无 NaN。
- teacher-forced validation loss（每 327 步）：

| step | 327 | 654 | 981 | 1,308 | 1,635 | 1,962 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| val loss | 0.338 | 0.302 | 0.295 | **0.292** | 0.299 | 0.304 |

val loss 在 step 1,308 附近最低，最后约两个 epoch 轻微回升——即对 2,064 条
训练集出现轻度过拟合。本次只导出并评测了最终 step 1,965 的 adapter；中间
checkpoint（327…1,962）保留在 run 目录 `checkpoints/`，必要时可补充
step-1,308 的 paired 对比。
- 最终 DCP checkpoint 已导出为标准 PEFT adapter：`outputs/prime-rl/bird-sft-cot-sql-v10-20260906-0454/adapter/`。

启动前的 2-step GPU smoke 为 `bird-sft-smoke-preflight2-20260906`，
config/dry-run 检查为 `sft-bird-{preflight,final}-check`、`sft-bird-config-check*`。

### 1.2 训练后 paired execution eval 设置

`./scripts/launch_sft_post_eval.sh bird-sft-cot-sql-v10-20260906-0454 1965`
启动单台 vLLM（`--reasoning-parser qwen3`，Base 与 LoRA 候选同服），然后顺序评测：

- taskset：`/data/ximo/bird-text2sql-rl/data/rlvr-v2/validation.jsonl`
  （ReViSQL verified validation 398 题减去 2 条拒绝项，剩 396 题；数据位置与
  构建方式见 [DATASET.md](DATASET.md)）。
- **Base**：`bird-text2sql-base` = 原始 Qwen3-4B-Instruct-2507，
  `platinum-base-v2.toml`，`enable_thinking=false`（Instruct-2507 原生非思考模式，
  最终 content 直接是 SQL）。
- **候选**：`bird-text2sql-sft` = Base + step-1,965 LoRA，`platinum-v2.toml`，
  `enable_thinking=true`（与训练时的 qwen3 renderer 一致；`<think>…</think>`
  由 reasoning parser 分离，最终 content 只有 SQL；实测 396/396 条都有非空
  reasoning，平均约 770 字符）。
- 两者共用 `temperature=0`、`max_tokens=2048`、每表 3 行样例、prediction
  timeout 8 秒、`shape_reward=false`、`push=false`；结果写入
  `outputs/prime-rl/bird-sft-post-{base,step1965}--20260906-125601`。

Base 关闭 thinking、候选开启 thinking 是刻意的不对称：它分别匹配两个模型的原生
/训练后行为。副作用是两者的可用 token 预算不同（候选的 reasoning 占用
completion 预算），但这正是部署形态，且 2048 预算下候选没有出现截断性失败。

### 1.3 主结果

主指标是二值 exact execution（括号内为相对 Base 的变化）：

| 模型 | exact execution | executable SQL | format valid | timeout |
| --- | ---: | ---: | ---: | ---: |
| Base（step 0） | 254 / 396 = **64.14%** | 353 / 396 = 89.14% | 395 / 396 = 99.75% | 27 / 396 = 6.82% |
| SFT step 1,965 | 286 / 396 = **72.22%**（+8.08 pp） | 370 / 396 = 93.43%（+4.29 pp） | 396 / 396 = 100%（+0.25 pp） | 3 / 396 = 0.76%（-6.06 pp） |

"可执行但语义错误"从 99 题降到 84 题：提升不只来自把报错 SQL 修成可执行，也来自语义修正。Base 唯一一条 format invalid 是模型无视 SQL-only 约束、在
content 里输出了带 Markdown fence 的英文推理（`parser_source=fence`）；
候选 396/396 全部 SQL-only。

### 1.4 Paired flips 与显著性

gain 表示 Base 错、SFT 对；loss 表示 Base 对、SFT 错。同一批 396 题逐题配对：

| 对比 | gain / loss | 净变化 | delta 95% CI | McNemar exact p |
| --- | ---: | ---: | ---: | ---: |
| Base → SFT step 1,965 | 65 / 33 | +32 题 | [+3.2, +13.0] pp | 0.0016 |

+8.08 pp 的配对区间远离 0，是本项目至今第一个统计上稳健的正收益
（对比 §3 旧 SQL-only RL 的 +0.85 pp、p=0.072）。

### 1.5 收益与回退的结构分析

65 个 gain 中，17 个来自修复 Base 的 8 秒 timeout（Base 的 27 个 timeout 集中在
`donor`、`retails`、`movie_platform`、`language_corpus`、`codebase_comments`、
`coinmarketcap`、`sales_in_weather`、`world_development_indicators` 8 个库，
是 Base 在这些库上生成病态慢查询），其余 48 个是纯语义修正；33 个 loss 中
没有任何一个由新 timeout 造成。也就是说约 1/4 的 gain（17/65）来自"学会
不写慢查询"，其余来自语义正确性本身。

数据库级变化（节选，按 delta 排序）：

| 数据库 | 题数 | Base | SFT | delta | gain / loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `donor` | 12 | 16.7% | 83.3% | +66.7 pp | 8 / 0 |
| `retails` | 8 | 37.5% | 87.5% | +50.0 pp | 4 / 0 |
| `movie_platform` | 8 | 37.5% | 75.0% | +37.5 pp | 3 / 0 |
| `app_store` | 5 | 0.0% | 40.0% | +40.0 pp | 2 / 0 |
| `codebase_comments` | 11 | 63.6% | 81.8% | +18.2 pp | 3 / 1 |
| `works_cycles` | 24 | 70.8% | 58.3% | -12.5 pp | 3 / 6 |
| `superstore` | 5 | 60.0% | 20.0% | -40.0 pp | 0 / 2 |
| `cookbook` | 5 | 100.0% | 60.0% | -40.0 pp | 0 / 2 |
| `talkingdata` | 3 | 66.7% | 33.3% | -33.3 pp | 1 / 2 |

收益最大的库几乎全是 Base timeout 重灾区；回退集中在个别小库。抽查
`works_cycles`（AdventureWorks，loss 最多）的 6 个 loss，典型模式是 SFT 模型
把简单问题过度复杂化：例如"793 号产品的型号名"本是
`Product ⋈ ProductModel` 两表直连，SFT 却绕道
`ProductModelProductDescriptionCulture` 三表连接；"夜班最多的部门"被改写成
不必要的子查询。这类回退与 teacher CoT 偏好"先分解再组合"的风格一致，
是 distillation 的已知副作用。

### 1.6 作废的评测尝试（记录以免复用）

- `bird-sft-post-*--20260906-124449`（第一批）：当时 Base 与候选共用
  `platinum-v2.toml`，配置里没有 `chat_template_kwargs`。qwen3 chat template
  默认开启 thinking，而 eval server 又开了 `--reasoning-parser qwen3`，导致
  非思考的 Base 把全部输出送进 reasoning 字段、content 为空——396/396
  format invalid、全 0 分（SQL 本身大多正确但无法计分）；候选 run 的全部模型
  调用 16 秒内即报空 ProviderError、0 条 trace（紧接 Base eval 结束 3 秒后
  启动，LoRA 模块首次实际加载疑似未就绪/失败，原因未深究）。整批作废。
- `bird-sft-post-*--20260906-125338`（第二批）：修复配置（新建
  `platinum-base-v2.toml` 给 Base 显式 `enable_thinking=false`）后的 1 题
  smoke，验证通过后随即启动完整批次。
- `bird-sft-post-*--20260906-125601`（第三批）：完整 396 题配对结果，即 §1.3
  的数据来源。

教训：Base 与 thinking-SFT 候选同服评测时，两者的 chat template kwargs 必须
分别显式固定，且要等 vLLM 报告 LoRA 模块就绪后再发请求。

### 1.7 结论与下一步

- SFT 在 verified validation 上带来稳健、分布较广的提升（65 gain / 33 loss，
  p=0.0016），且 format、executable、timeout 全部改善，没有发现系统性代价。
- 尚未完成的验证：四基准最终泛化评测（Arcwise-Plat、Arcwise-Plat-SQL、
  Mini-Dev、Full Dev，见 [EVALUATION.md](EVALUATION.md)）还没跑；verified
  validation 是模型选择集，最终泛化结论要等四基准的 Base/SFT paired 结果。
- 可选补充：step-1,308（val loss 最低）与 step-1,965 的 paired 对比，确认
  1,965 没有因轻度过拟合损失泛化。

### 1.8 Base rollout 与 v3–v9 SQL-only SFT 迭代（已被 v10 取代）

corrective 数据先由 `sft-train-base-v3--20260904-201932` 对 RLVR-v2 train 的
2,050 题做一次完整 Base rollout：2,050/2,050 trace 成功，1,407 exact
（68.63%）、1,966 executable（95.90%）、2,038 format valid（99.41%）、24 timeout
（1.17%）；643 条 non-exact 进入 teacher correction 分支。此前
`sft-train-base-v3--20260904-195513` 因配置问题 2,050/2,050 trace 均报错，不能
作为数据来源；`201606`、`201801` 是修复过程中的小 smoke。

随后在 2026-09-04–05 用 SQL-only target（`enable_thinking=false`）快速迭代多种
数据混合。表中“eval”均是各 run 自己保存的 396 题 Platinum validation online
结果；不同 run 的 step-0 会因当时服务/模板状态而波动，因此只看同一行内部变化，
不能拿这些 Base 绝对值与 v10 post-eval 的 64.14% 横向相减。

| Run | planned / reached | 数据混合 | online exact execution | 终态 |
| --- | ---: | --- | --- | --- |
| `bird-sft-direct-v3-20260904-185424-12817` | 60 / 21 | `sft-v2-bird` direct | step 0/10/20：67.93/68.94/62.88% | interrupt |
| `bird-sft-corrective-v4-20260904-203633-9736` | 80 / 49 | `sft-v3-bird-corrective` | step 0/10/20/30/40：69.19/68.43/70.45/68.94/68.18% | interrupt |
| `bird-sft-error-correction-v5-20260904-214532-9601` | 60 / 15 | `sft-v4-bird-error-correction` | step 0/10：70.45/67.93% | interrupt |
| `bird-sft-base-replay-v6-20260904-221013-26931` | 80 / 28 | `sft-v5-bird-base-replay` | step 0/10/20：69.19/70.20/68.43% | interrupt |
| `bird-sft-sql-emphasis-v7-20260905-005420-11017` | 80 / 12 | `sft-v6-bird-sql-emphasis` | 仅完整 step 0：69.70%；step 10 eval 未完成 | interrupt |
| `bird-sft-clean-v8-20260905-015311-27261` | 90 / 90 | `sft-v5-bird-base-replay` 清理版 | step 0/30/60/90：69.95/70.71/69.78/70.20% | 完成 |
| `bird-sft-clean-v9-20260905-123445-10941` | 510 / 170 | 同上，更长 | 仅完整 step 0：69.11%；step 85 未产出聚合结果 | step 170 等待 `.receiver_ready` 1,200 秒后失败 |

v3–v7 的日志都以 KeyboardInterrupt/SIGTERM 结束，不应误写成完整训练；v8 是这条
线唯一按计划跑完的 SFT，但最佳 step 30 只比同 run Base 高 0.76 pp，step 90 只高
0.25 pp，且未做独立 paired 显著性验证。v9 的 teacher-forced val loss 已从 1.614
降到 step 170 的 0.298，却因 eval/broadcast 断链无法得到对应 execution 结论；这也
再次说明 loss 不能替代独立 execution eval。SQL-only 短 SFT 没有形成稳定收益，
因此主路线最终改为 v10 的 teacher CoT + verified SQL 完整 SFT。

## 2. 已停止的 wide CoT run

`bird-sft-wide-v2-20260903-182552-21766` 原计划 16,000 steps，在 step 1,001
人工停止。它使用 `sft-v2-wide`（914,320 train / 1,836 validation）的 SynSQL Think
CoT，teacher-forced train loss 从 0.991 降到 0.336，validation loss 从 step 1 的
0.999 降到 step 1,000 的 0.328；但它不能作为 BIRD warm start：固定 BIRD Dev
probe 的 exact execution 从 step 0 的 58.59% 降到 step 500 的 53.12%、step 1000
的 51.17%；format valid 从 99.61% 降到 0%，executable SQL 从 91.80% 降到
80.47%。这说明 wide teacher loss 的改善伴随明显 domain/输出协议负迁移。当前主
方案因此改为从原始 Qwen3-4B 直接做 BIRD CoT SFT（即 §1 的 v10）。

## 3. 历史 SQL-only online curriculum 实验

以下全部结果属于旧 SQL-only 方案，仅用于说明为什么主路线改为 verified data +
SFT warm start + RLVR。旧 run 来自 `outputs/prime-rl/` 或历史 `outputs/evals/`；
它们使用 `<sql>`-only prompt、旧 result-set evaluator 和不同数据，不能与当前
CoT+SQL 协议的绝对分数直接相减。

### 3.0 训练 run 与启动恢复记录

唯一完成 120/120 steps 的正式训练是
`bird-sqlonly-online-20260829-191739-24054`。固定 150 题 online eval 中，ID 从
step 0 的 72.00% 到 step 80 的 72.67%（online 最佳），step 120 回到 72.00%；
schema-OOD 从 64.00% 到 step 80 的 65.33%、step 120 的 64.67%。online 差异只有
1–2 题，因而后续没有只凭该曲线下结论，而是另跑 §3.2 的完整独立 batch。

同日前六个同前缀目录是启动/恢复过程，不是六次额外正式训练：`184806`、`185111`
为空目录；`185327`、`185558` 因找不到 `vllm-router` 停止；`190411` 在训练前被
终止；`190853` 完成 step-0 eval/broadcast 后中断，没有 optimizer step。保留这些
目录是为了说明运行历史，不应把其空 metrics 计成 0 分实验。

### 3.1 模型与评测完整性

正式 batch 为 `20260830-163442`，比较：

- **Base**：`/data/qwen3-4b-instruct-2507`；
- **step 80**：训练中 online eval 最好的 retained trainer checkpoint，恢复为评测 LoRA；
- **step 120**：最终 broadcast LoRA。

同一 suite 的三个 run 除 model ID 和 run name/dir 外配置一致：`temperature=0`、
`max_tokens=1024`、`num_rollouts=1`、SQL timeout 5 秒、每表 3 条样例行、
`push=false`。vLLM 使用 `generation_config=vllm`，不继承模型目录中的隐藏采样默认值。

| Suite | 每个模型的 trace | Run 名 |
| --- | ---: | --- |
| Local ID | 150 / 150 | `direct-id-{base,step80,step120}--20260830-163442` |
| schema-OOD | 150 / 150 | `direct-ood-{base,step80,step120}--20260830-163442` |
| BIRD Dev | 1,534 / 1,534 | `full-dev-{base,step80,step120}--20260830-163442` |

三个模型使用完全相同的 task key；每条模型调用记录的 model ID 也与 run 名一致。
全部 5,502 条 completion 中没有 `<plan>` 输出。

BIRD Dev 的 `thrombosis_prediction` 第 1,257 题在三个模型上都生成了 1,024-token
的嵌套 `CAST` 退化输出，随后 `sqlglot` 触发 `RecursionError`，导致 reward 为
`None`。本文保守地把该题对三个模型都计为 0；因此 paired delta 不受影响。若
viewer 忽略空 reward，其显示的绝对均值会比本文高约 0.04 pp。

### 3.2 主结果

主指标是二值 execution reward。括号内是相对 Base 的变化。

| 数据集 | Base | step 80 | step 120 |
| --- | ---: | ---: | ---: |
| Local ID，150 题 | 108 / 150 = **72.00%** | 107 / 150 = **71.33%** (-0.67 pp) | 110 / 150 = **73.33%** (+1.33 pp) |
| schema-OOD，150 题 | 97 / 150 = **64.67%** | 97 / 150 = **64.67%** (+0.00 pp) | 97 / 150 = **64.67%** (+0.00 pp) |
| BIRD Dev，1,534 题 | 831 / 1,534 = **54.17%** | 841 / 1,534 = **54.82%** (+0.65 pp) | 844 / 1,534 = **55.02%** (+0.85 pp) |

step 120 是这次独立评测中最好的 checkpoint：ID 净增 2 题，Full Dev 净增 13 题；
step 80 则没有复现 online eval 中的领先。

### 3.3 Paired flips 与不确定性

gain 表示 Base 错、LoRA 对；loss 表示 Base 对、LoRA 错。95% 区间按同题 reward
差计算，`p` 是双侧 exact McNemar 检验。

| 数据集 | 对比 | gain / loss | 净变化 | delta 的 95% CI | p |
| --- | --- | ---: | ---: | ---: | ---: |
| Local ID | Base → step 80 | 2 / 3 | -1 题 | [-3.60, +2.26] pp | 1.000 |
| Local ID | Base → step 120 | 4 / 2 | +2 题 | [-1.87, +4.54] pp | 0.688 |
| schema-OOD | Base → step 80 | 1 / 1 | 0 题 | [-1.85, +1.85] pp | 1.000 |
| schema-OOD | Base → step 120 | 1 / 1 | 0 题 | [-1.85, +1.85] pp | 1.000 |
| BIRD Dev | Base → step 80 | 23 / 13 | +10 题 | [-0.11, +1.42] pp | 0.132 |
| BIRD Dev | Base → step 120 | 29 / 16 | +13 题 | [-0.01, +1.70] pp | 0.072 |

step 120 相对 step 80 在 Full Dev 上是 9 gain / 6 loss，净增 3 题（+0.20 pp，
`p=0.607`），两者差异很小。

schema-OOD 三个均值虽然完全相同，但不是同一模型行为：Base 与 step 120 只有
70.0% 的 completion 文本完全相同，并发生了 1 gain / 1 loss。Full Dev 上 Base 与
step 120 的 completion 文本相同率为 71.45%。这说明 LoRA 确实改变了生成，只是
OOD gain/loss 正好抵消。

### 3.4 格式、可执行性与退化

| 模型 | Full Dev format valid | executable SQL | execution timeout | 非 `stop` 结束 | scorer error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 1,529 / 1,534 = 99.67% | 1,394 / 1,534 = 90.87% | 19 / 1,534 = 1.24% | 3 | 1 |
| step 80 | 1,531 / 1,534 = 99.80% | 1,409 / 1,534 = 91.85% | 22 / 1,534 = 1.43% | 2 | 1 |
| step 120 | 1,531 / 1,534 = 99.80% | 1,409 / 1,534 = 91.85% | 20 / 1,534 = 1.30% | 2 | 1 |

SQL-only prompt 约束工作正常：没有 `<plan>`，格式遵循率接近 100%。step 80/120
都比 Base 多 15 条可执行 SQL，但可执行不等于语义正确；step 80 的 timeout 还略多。

### 3.5 Full Dev 的收益分布

step 120 的数据库级变化并不均匀：

| 数据库 | 题数 | Base | step 120 | delta | gain / loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| `debit_card_specializing` | 64 | 50.00% | 54.69% | +4.69 pp | 3 / 0 |
| `formula_1` | 174 | 44.25% | 47.13% | +2.87 pp | 5 / 0 |
| `california_schools` | 89 | 16.85% | 19.10% | +2.25 pp | 3 / 1 |
| `student_club` | 158 | 72.78% | 74.68% | +1.90 pp | 5 / 2 |
| `superhero` | 129 | 81.40% | 82.17% | +0.78 pp | 2 / 1 |
| `card_games` | 191 | 52.36% | 52.88% | +0.52 pp | 3 / 2 |
| `codebase_community` | 186 | 66.13% | 66.13% | +0.00 pp | 3 / 3 |
| `european_football_2` | 129 | 65.89% | 65.89% | +0.00 pp | 1 / 1 |
| `thrombosis_prediction` | 163 | 49.69% | 49.69% | +0.00 pp | 1 / 1 |
| `toxicology` | 145 | 55.86% | 55.17% | -0.69 pp | 3 / 4 |
| `financial` | 106 | 16.04% | 15.09% | -0.94 pp | 0 / 1 |

按 gold SQL 的简单、可重叠结构标签看，step 120 在 `GROUP BY`（+2.19 pp）、
`ORDER BY/LIMIT`（+1.90 pp）和嵌套查询（+1.72 pp）上增幅较大；普通 JOIN 题只
增加 0.61 pp。CTE 题从 3.92% 升到 4.90%，绝对正确率仍很低。这些标签只用于
描述，不能当作独立或因果分组。

抽查 paired flips 后，收益主要来自：

- 把错误的行值输出改为 `COUNT`/聚合；
- 删除多余过滤条件；
- 修复相关子查询、分组和连接目标；
- 修复 Base 中的明显语法错误。

回退主要来自：

- 新增题目未要求的过滤条件；
- 使用错误 join key 或漏掉必要的表；
- 把 `COUNT(DISTINCT entity)` 退化为 `COUNT(*)`；
- 把“等于最大值”等语义错误改成“大于最大值”。

### 3.6 当时结论（已被 §1 取代）

当时把 step 120 作为暂定候选：ID 和 Full Dev 都是最高分，格式与可执行性没有
退化。但 schema-OOD 净提升为 0、Full Dev 的配对区间包含 0（p=0.072）、收益集中
在部分数据库，因此不足以证明“训练稳定提升 BIRD 泛化能力”。这正是转向
verified data + 完整 CoT SFT（§1）的直接原因；§1 的 +8.08 pp（p=0.0016）
远超该线所有 run。

## 4. 历史 direct-GRPO 结果

以下结果来自旧 `outputs/evals/` 归档，比较 Base、历史 step 50 和 step 70。它们
使用旧评测流程，只用于说明历史趋势。

### 4.1 Local ID / schema-OOD

完整 run 各 150 题：

| 模型 | ID exact execution | 相对 Base | OOD exact execution | 相对 Base |
| --- | ---: | ---: | ---: | ---: |
| Base | 72.00% | — | 66.00% | — |
| step 50 | 72.67% | +0.67 pp | 65.33% | -0.67 pp |
| step 70 | 73.33% | +1.33 pp | 65.33% | -0.67 pp |

Paired flips：

| 对比 | ID gain / loss | OOD gain / loss |
| --- | ---: | ---: |
| Base → step 50 | 3 / 2 | 3 / 4 |
| Base → step 70 | 4 / 2 | 4 / 5 |

对应完整 run ID：

```text
Base ID       be379c8c    Base OOD       4c2a0206
step 50 ID    0b9470f0    step 50 OOD    bff1d513
step 70 ID    bb37e34d    step 70 OOD    d4432fda
```

### 4.2 完整 BIRD Dev

历史流程做过两次 1,534 题重复：

| 重复 | Base | step 50 | step 70 |
| --- | ---: | ---: | ---: |
| A | 56.65% | 57.37% | 56.91% |
| B | 57.24% | 57.24% | 57.17% |

相对各次 Base：

| 重复 | step 50 | step 70 |
| --- | ---: | ---: |
| A | +0.72 pp | +0.26 pp |
| B | +0.00 pp | -0.07 pp |

Paired flips：

| 重复 | Base → step 50 | Base → step 70 |
| --- | ---: | ---: |
| A | 28 gain / 17 loss | 35 gain / 31 loss |
| B | 23 gain / 23 loss | 33 gain / 34 loss |

对应 run ID：

```text
重复 A: Base 4e4a1084, step 50 e4eb090f, step 70 1ffbad95
重复 B: Base f9ced85f, step 50 7d589ca4, step 70 24bcc441
```

历史结果同样只表现出小幅、不可稳定复现的变化：ID 略升，schema-OOD 未升，
Full Dev 没有跨重复稳定收益。

## 5. 结果口径限制

当前与历史分数都属于本项目环境，不应直接冒充官方 BIRD leaderboard 分数：

- schema prompt 包含每表最多 3 条真实样例行；
- 执行比较按每题 `set`/`multiset`/`list`/`subset` grading method 决定是否保留
  行序与重复行；行内列序忽略，细节见 [ENVIRONMENT.md](ENVIRONMENT.md)；
- 预测 SQL timeout：§1 为 8 秒，§3 为 5 秒；
- 使用项目指定的 BIRD Dev JSON 与 SQLite 快照；
- 没有做官方 Test 提交。

当前流程还改变了 prompt、环境实现、Verifiers/Prime-RL 路径和 vLLM generation
config，因此 §1 的 Base 64.14% 与 §3 的 Base 54.17%、历史 Base 56.65%–57.24%
的差距不能解释为模型本身变化（taskset 也从原始 BIRD Dev 换成了 verified
validation）。跨实验比较前必须确认 prompt、数据库版本、生成参数和 evaluator
完全一致。
