# 实验结果

## 当前结论

当前主结果是从原始 Qwen3-4B-Instruct-2507 启动、使用 BIRD-Platinum、CISPO 和
ReViSQL 奖励/交互合同训练得到的 step 1,300。该 checkpoint 在完整 396 题
validation 上达到 319/396 = 80.56%，并在最终 Arcwise-Plat-SQL 与 Arcwise-Plat
上分别达到 376/498 = 75.50% 和 404/498 = 81.12%。

最终能力判断以两个 Arcwise split 为主。Thinking Machines/ReViSQL 的数据审计发现
BIRD Train 有 52.1% 的 gold SQL 错误，BIRD Mini-Dev 的总检测错误率为 52.8%；原始
Mini-Dev 和 Full Dev 因而只作为带噪鲁棒性诊断，不作为干净能力主结论。历史
Base/SFT 使用 SQL-only 协议，ReViSQL step 1,300 使用非 thinking、多轮工具协议；
跨协议差值是端到端系统差值，不能全部归因于权重。

## 1. ReViSQL-BIRD Qwen3-4B

训练 run：
`bird-revisql-qwen3-4b-promptfmt-v4-pilot40-20260913`。

- 初始化：`/data/qwen3-4b-instruct-2507`，不是 SFT-v10。
- 数据：BIRD-Platinum executable split，2,050 train / 396 validation。
- 优化：CISPO、group 16、batch 64、LoRA rank 32/alpha 64、learning rate `1e-5`。
- 协议：canonical non-thinking Jinja、最多五轮、SQL execution tool、每轮最多
  3,072 output tokens、总 context 32,768。
- 选择：step 1,300 的 validation exact 为 **319/396 = 80.56%**，高于最终
  step 1,322 的 317/396；step 1,300 因而是当前最佳 checkpoint。
- 导出 adapter：
  `outputs/prime-rl/bird-revisql-qwen3-4b-promptfmt-v4-pilot40-20260913/adapter-step1300/`。
  原始 step 1,300 DCP checkpoint 与受保护的 step 960 均未被覆盖。

### 1.1 四套最终测试

正式 batch：`bird-revisql-step1300-{bird-mini-dev,bird-full-dev,arcwise-plat-sql,arcwise-plat}--20260915-004207`。
四套均为 `temperature=0`、单 rollout、`enable_thinking=false`、最多五轮、每轮
3,072 tokens、相同 execution scorer；模型 ID 为 `bird-revisql-step1300`。

| 数据集 | exact execution | executable | format valid | length finish | timeout | rollout error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BIRD Mini-Dev | **298/500 = 59.60%** | 468/500 = 93.60% | 471/500 = 94.20% | 25/500 | 0/500 | 0/500 |
| BIRD Full Dev | **961/1,534 = 62.65%** | 1,466/1,534 = 95.57% | 1,480/1,534 = 96.48% | 48/1,534 | 0/1,534 | 0/1,534 |
| Arcwise-Plat-SQL | **376/498 = 75.50%** | 467/498 = 93.78% | 475/498 = 95.38% | 21/498 | 1/498 | 0/498 |
| Arcwise-Plat | **404/498 = 81.12%** | 479/498 = 96.18% | 485/498 = 97.39% | 11/498 | 2/498 | 0/498 |

四个 run 共 3,030/3,030 条完整结束，tmux eval/server 均以 status 0 退出；没有
空输出、content/reasoning control artifact 或 rollout error。

### 1.2 为什么以 Arcwise 为主

[Thinking Machines 的 ReViSQL 报告](https://thinkingmachines.ai/news/putting-task-expertise-into-rl/)
给出的审计结果是：BIRD Train 中 52.1% 的 gold SQL 不正确；对 BIRD Mini-Dev
两轮清洗后，检测到的总错误率达到 52.8%。文章正式主榜使用只修正 gold SQL 的
Arcwise-Plat-SQL，而不是原始 Mini-Dev/Full Dev。

本项目相应采用以下报告层级：

1. `Arcwise-Plat-SQL` 是与文章直接对齐的首要外部指标；
2. `Arcwise-Plat` 是进一步修正 question/evidence/schema 后的干净能力主指标；
3. 原始 Mini-Dev/Full Dev 只报告为带噪鲁棒性，不据此否定干净集收益。

两个 Arcwise split 共享同一批 498 个 ID。step 1,300 从只修 SQL 的 376 题提升到
全校正的 404 题，逐题为 50 gain / 22 loss，净增 28 题，即 **+5.62 pp**；paired
95% CI 为 `[+2.32, +8.93]` pp，McNemar exact `p=0.00129`。这说明 gold SQL
之外的 question、evidence 和 schema 噪声仍会显著压低同一模型的结果。

### 1.3 与文章结果对比

下表使用共同的 Arcwise-Plat-SQL 498 题。除明确标出的 SC-16 外，模型结果都是
greedy 单样本、`temperature=0`；SC-16 是文章额外报告的 `temperature=1` 多样本
多数表决，不能与 greedy 成本/解码口径混为一谈。文章模型为 Kimi-K2.6，本项目
模型为 Qwen3-4B，因此这是方法/系统位置对比，不是同规模模型对比。

| 系统 | 解码 | Arcwise-Plat-SQL | 相对本项目 step 1,300 |
| --- | --- | ---: | ---: |
| 原始 BIRD Train + reward shaping（文章消融） | greedy | 74.30% | 本项目 **+1.20 pp** |
| **Qwen3-4B ReViSQL step 1,300（本项目）** | **greedy** | **376/498 = 75.50%** | — |
| BIRD-Platinum verified data only（Kimi-K2.6） | greedy | 88.55% | 本项目 -13.05 pp |
| ReViSQL-K2.6 完整方法 | greedy | 91.37% | 本项目 -15.87 pp |
| ReViSQL-K2.6 | SC-16，temperature=1 | 92.97% | 本项目 -17.47 pp |
| 文章 human proxy | — | 92.96% | 本项目 -17.46 pp |

本项目 4B 模型已经达到并略高于文章“原始脏 BIRD Train + reward shaping”消融线，
但与 Kimi-K2.6 的 verified-data/full-method 结果仍有 13–16 pp 差距。文章没有报告
全校正 Arcwise-Plat，因此本项目的 81.12% 不能直接与文章的 91.37% 排名比较。

### 1.4 跨协议历史对比

历史 Base/SFT 采用单轮 SQL-only 合同；本节只衡量部署系统整体变化。逐题配对键为
`(example_id, db_id, question_fingerprint)`，不是包含 system prompt 的 task hash。

| 数据集 | 历史基线 → ReViSQL | gain / loss | delta 95% CI | McNemar exact p |
| --- | ---: | ---: | ---: | ---: |
| Mini，Base | 53.60% → 59.60%（+6.00 pp） | 80 / 50 | [+1.56, +10.44] pp | 0.0107 |
| Mini，SFT-v10 | 50.80% → 59.60%（+8.80 pp） | 84 / 40 | [+4.50, +13.10] pp | 9.65e-5 |
| Full，Base | 58.67% → 62.65%（+3.98 pp） | 196 / 135 | [+1.66, +6.29] pp | 0.000946 |
| Full，SFT-v10 | 56.00% → 62.65%（+6.65 pp） | 200 / 98 | [+4.47, +8.83] pp | 3.47e-9 |
| Arcwise，Base | 58.84% → 81.12%（+22.29 pp） | 134 / 23 | [+17.76, +26.82] pp | 3.00e-20 |
| Arcwise，SFT-v10 | 61.65% → 81.12%（+19.48 pp） | 124 / 27 | [+14.95, +24.00] pp | 4.69e-16 |

这些显著差值证明最终 ReViSQL 系统优于历史部署合同；若要把提升严格归因于 RL
权重，仍需让原始 Base 使用完全相同的 ReViSQL 四测配置做 paired baseline。

### 1.5 失败结构与下一步

- 3,030 题中 841 题是“SQL 可执行但结果错误”，占全部 991 个非 exact 的 84.86%；
  主要剩余瓶颈仍是语义正确性，而不是执行链路。
- 共 105 次 `finish_reason=length`，全部非 exact，并解释了 119 个 format-invalid
  样本中的 105 个。Arcwise-Plat-SQL/Arcwise 分别有 21/11 次长度截断。
- 两个 Arcwise run 都没有 SQL tool message；四套 3,030 题总共只有 Full Dev 的
  2 次工具调用。模型学习了 requirement/verification/final-solution 形式，但几乎
  没有利用数据库反馈进行迭代修正。
- 后续训练应优先控制冗长输出并提高真实工具使用率。不过 Arcwise-Plat-SQL 即使
  假设 21 个截断全部修复，上限也只有 79.72%，长度并不能单独解释与文章结果的差距。

## 2. SFT v10

Run：`bird-sft-cot-sql-v10-20260906-0454`

- 基座：Qwen3-4B-Instruct-2507。
- 数据：2,064 train / 398 validation；teacher CoT + ReViSQL verified SQL。
- 训练：1 GPU，1,965 optimizer steps，约 6 次数据遍历，约 7.5 小时。
- 最终 adapter：`outputs/prime-rl/bird-sft-cot-sql-v10-20260906-0454/adapter/`。
- selection eval：`rlvr-v2/validation.jsonl`，396 题。

Teacher-forced validation loss：

| step | 327 | 654 | 981 | 1,308 | 1,635 | 1,962 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| loss | 0.338 | 0.302 | 0.295 | **0.292** | 0.299 | 0.304 |

最后两个 epoch 有轻度回升。此次只导出并正式评测了最终 step 1,965，不能根据 loss
直接声称 step 1,308 的 execution accuracy 更高。

正式 paired execution batch：
`bird-sft-post-{base,step1965}--20260906-125601`。

| 模型 | exact execution | executable | format valid | timeout |
| --- | ---: | ---: | ---: | ---: |
| Base | 254/396 = **64.14%** | 353/396 = 89.14% | 395/396 | 27/396 |
| SFT step 1,965 | 286/396 = **72.22%** | 370/396 = 93.43% | 396/396 | 3/396 |

SFT 相对 Base 提升 32 题，即 8.08 个百分点。逐题为 65 gain / 33 loss，paired
delta 95% CI 为 `[+3.2, +13.0]` pp，McNemar exact `p=0.0016`。

这个结果只证明模型在 ReViSQL verified validation 的当前 prompt/executor 协议上
提升；它不是官方 BIRD leaderboard 分数，也不是最终泛化结果。


## 3. 冻结测试：Base vs SFT v10

正式 batch：`bird-base-{bird-mini-dev,bird-full-dev,arcwise-plat}--20260909-220548`。

| 数据集 | exact execution | executable | format valid | timeout | rollout error |
| --- | ---: | ---: | ---: | ---: | ---: |
| BIRD Mini-Dev | **268/500 = 53.60%** | 461/500 = 92.20% | 500/500 = 100% | 0/500 | 0/500 |
| BIRD Full Dev | **900/1,534 = 58.67%** | 1,434/1,534 = 93.48% | 1,533/1,534 = 99.93% | 4/1,534 = 0.26% | 0/1,534 |
| Arcwise-Plat | **293/498 = 58.84%** | 465/498 = 93.37% | 498/498 = 100% | 0/498 | 0/498 |

模型为 `/data/qwen3-4b-instruct-2507`（served model ID `bird-text2sql-base`）。三套
均使用 SQL-only system prompt、`enable_thinking=false`、`temperature=0`、
`max_tokens=2048`、单 rollout、每表 3 行样例、30 秒 prediction timeout，数据库为
`/data/ximo/minidev/MINIDEV/dev_databases`。完整 resolved config、trace、日志和
版本哈希保存在各 run 目录。

错误结构以“SQL 可执行但结果错误”为主：Mini-Dev 193/232、Full Dev 534/634、
Arcwise-Plat 172/205。不可执行预测最常见的是引用不存在的列，其次是解析错误和
不存在的函数。错误较集中的数据库包括 `formula_1`、`card_games`、
`thrombosis_prediction`、`codebase_community`、`california_schools` 和
`financial`；这只是样本量与难度共同作用下的失败主题，不应当作归一化数据库排名。

首批 `...--20260909-215805` 在 Mini-Dev 和 Full Dev 各暴露 1 条病态嵌套 SQL，
使 `sqlglot` 递归溢出并留下 rollout error。scorer 已把该异常改为普通不可执行 SQL，
首批因此不列为正式结果，以上表格只采用修复后 0 error 的完整重跑。

SFT v10 正式 batch：
`bird-sft-v10-{bird-mini-dev,bird-full-dev,arcwise-plat}--20260909-222408`。

| 数据集 | exact execution | executable | format valid | timeout | rollout error |
| --- | ---: | ---: | ---: | ---: | ---: |
| BIRD Mini-Dev | **254/500 = 50.80%** | 449/500 = 89.80% | 500/500 = 100% | 1/500 = 0.20% | 0/500 |
| BIRD Full Dev | **859/1,534 = 56.00%** | 1,392/1,534 = 90.74% | 1,534/1,534 = 100% | 2/1,534 = 0.13% | 0/1,534 |
| Arcwise-Plat | **307/498 = 61.65%** | 449/498 = 90.16% | 498/498 = 100% | 0/498 | 0/498 |

SFT 使用 v10 step 1,965 adapter（served model ID `bird-text2sql-sft-v10`），开启
`enable_thinking=true`；其余 taskset、system prompt、temperature、token 上限、
rollout 数、数据库、sample rows 和 scorer 与 Base 相同。三套 2,532 条输出均有
非空 reasoning，平均 reasoning 长度分别为 816、744 和 837 字符，没有
`finish_reason=length`。

逐题通过 task hash 与 Base 完整配对：

| 数据集 | Base → SFT | gain / loss | delta 95% CI | McNemar exact p |
| --- | ---: | ---: | ---: | ---: |
| BIRD Mini-Dev | 53.60% → 50.80%（-2.80 pp） | 52 / 66 | [-7.06, +1.46] pp | 0.2313 |
| BIRD Full Dev | 58.67% → 56.00%（-2.67 pp） | 158 / 199 | [-5.08, -0.26] pp | 0.0341 |
| Arcwise-Plat | 58.84% → 61.65%（+2.81 pp） | 71 / 57 | [-1.64, +7.26] pp | 0.2504 |

Full Dev 的下降在单项 0.05 水平显著，但没有通过三基准 Bonferroni 阈值
`0.05/3`；另外两项区间均跨 0。失败结构显示两个原始 BIRD split 的回退主要来自
不可执行 SQL 增加：Mini-Dev 为 50 次普通执行/解析失败加 1 次 timeout，Full Dev
为 140 次普通执行/解析失败加 2 次 timeout；最常见错误都是引用不存在的列。
Arcwise-Plat 虽然可执行率从 93.37% 降至 90.16%，但“可执行而语义错误”从 172
降至 142，净效果为 +14 题。这说明 v10 对校正后的 Arcwise 合同有语义收益，但对
原始 BIRD 噪声合同的 schema grounding 更脆弱。

### 3.1 OpenRouter DeepSeek R1 外部基线

正式 run：`bird-deepseek-r1-arcwise-plat-formal--20260911-110457`。模型为
OpenRouter `deepseek/deepseek-r1`，只评测 Arcwise-Plat 498 题。除移除 Qwen 专属的
`chat_template_kwargs.enable_thinking` 外，沿用同一 SQL-only system prompt、taskset、
schema/sample rows、execution scorer、`temperature=0`、`max_tokens=2048`、单 rollout
和 30 秒 prediction timeout。R1 的 reasoning 由 OpenRouter 原生字段单独返回。
请求通过本地 `localhost:8081` HTTP(S) 代理；API key 只从运行时环境读取，没有写入
resolved config 或 trace。正式批次为 16 并发，耗时 44m 14s，498/498 rollout 成功，
无 API error。

| 模型 | exact execution | executable | format valid | empty content | length finish | timeout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B Base | 293/498 = **58.84%** | 465/498 = 93.37% | 498/498 | 0/498 | 0/498 | 0/498 |
| SFT v10 | **307/498 = 61.65%** | 449/498 = 90.16% | 498/498 | 0/498 | 0/498 | 0/498 |
| DeepSeek R1 | 293/498 = **58.84%** | 341/498 = 68.47% | 0/498 | 145/498 = 29.12% | 153/498 = 30.72% | 4/498 = 0.80% |

R1 的主要瓶颈是协议预算而非已产出 SQL 的语义质量。153 条 `finish_reason=length`
中，145 条没有 final content；剩余 8 条只有截断的可见内容。非空的 353 条中，
341 条可执行（96.60%），293 条 exact（83.00%）。正常停止的 345 条全部返回 Markdown
SQL fence，因此严格 `format_valid` 为 0；当前兼容 parser 能抽取 fence 内 SQL 并按既定
execution contract 评分。总计 205 条失败可分为 145 条空答案、48 条可执行但结果错误，
以及 12 条非空但不可执行；最后一类包括 5 条解析错误、3 条不存在列和 4 条执行超时。

逐题 task hash 完整配对：

| 对比 | exact 变化 | gain / loss | delta 95% CI | McNemar exact p |
| --- | ---: | ---: | ---: | ---: |
| Base → DeepSeek R1 | 58.84% → 58.84%（0.00 pp） | 95 / 95 | [-5.43, +5.43] pp | 1.0000 |
| SFT v10 → DeepSeek R1 | 61.65% → 58.84%（-2.81 pp） | 87 / 101 | [-8.20, +2.58] pp | 0.3431 |

R1 与 Base 虽然总分相同，但各有 95 道独占正确题，不能解释为能力相同。当前结果只回答
“R1 在项目冻结的 2048-token 协议下表现如何”；它不能回答 R1 在足够 reasoning budget
下的上限。OpenRouter 报告本批次输入 212,370 uncached + 1,690,688 cached tokens、
816,235 completion tokens（其中 815,444 reasoning tokens），费用 `$3.3727281`。
完整 resolved config、trace、日志和摘要保存在正式 run 目录。

## 4. SFT-v10 → RLVR 100-step pilot

正式 run：`bird-rl-sft-v10-plat-full-g16-v2-20260910-0257`，完成 100/100 steps，
耗时 5h 51m 37s。初始化模型为 merged SFT v10；训练集为完整 Platinum train
2,050 题，validation 为冻结的 396 题。配置使用 group size 16、batch size 128、
temperature 0.8、LoRA rank 32、learning rate `3e-6`、`kl_tau=1e-3`，并以
zero-signal gate 回填全对/全错组。配置、数据和初始化模型 SHA-256 均记录在 run
目录的 `preflight.json`。

Online validation（temperature 0，单 rollout）：

| step | 0 | 20 | 40 | 60 | 80 | 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| exact execution | 282/396<br>71.21% | 279/396<br>70.45% | **288/396<br>72.73%** | 278/396<br>70.20% | 285/396<br>71.97% | 282/396<br>71.21% |

step 40 是 pilot 内最佳点，但相对 step 0 只有 11 gain / 5 loss，净增 6 题，
McNemar exact `p=0.2101`；step 100 为 10 gain / 10 loss，净变化 0，`p=1.0`。
因此当前曲线只支持“训练稳定可继续”，不支持“validation 已随训练持续提高”。

Advantage 与覆盖率诊断：

- 共尝试 935 个 group，其中 805 mixed（86.10%）、128 全错（13.69%）、2 全对
  （0.21%）；少量不完整 mixed group 被丢弃，最终训练 800 个完整 mixed group，
  即 12,800 条 effective rollout。
- 800 个 effective group 对应 800 道互不重复的训练题，只覆盖 2,050 题池的
  39.02%。这次主要瓶颈已不是缺少 advantage，而是 100 steps 尚未覆盖大部分题。
- 每 20-step block 的 all-rollout reward 从 0.2894、0.3146、0.3789、0.3948 升到
  0.4290；effective reward 从 0.3289 升到 0.4965。该趋势证明 train distribution
  上在学习，但不能替代冻结 validation。
- 优化保持稳定：grad norm 中位数 0.0497、最大 0.3537；mismatch KL 中位数
  `5.66e-4`、最大 `1.19e-3`；entropy 首尾约 0.200/0.218，没有崩塌迹象。

第一次正式启动遗漏 Qwen3 reasoning parser，step-0 的 396 条输出全部 format
invalid；它在第一个 optimizer step 前即停止，不计为实验。上述 v2 run 已显式固定
`inference.vllm.reasoning_parser="qwen3"`，step 0 format valid 为 396/396。

## 5. 尚未完成

- 用原始 Base 在完全相同的 ReViSQL non-thinking、多轮合同下跑四套 paired baseline，
  以分离权重收益与协议收益。
- 针对长度截断和工具调用稀少设计新的训练改动；最终 Arcwise 已经使用一次，不能
  根据其失败样本回流训练或用于选择 checkpoint。

## 6. 报告口径

每个结果必须同时给出数据集、分母、模型/checkpoint、prompt、thinking 设置、生成
参数、数据库版本、prediction timeout 和 scorer 版本。不同 taskset 或协议下的
Base 分数不是同一个基线。

本项目 schema prompt 含每表最多 3 行真实样例；execution comparison 受每题
`grading_method` 控制。以上分数均为本地环境结果，没有提交 BIRD 官方隐藏 test。
