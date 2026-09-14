# 最终测试数据

本目录保存四套冻结测试所需的小型 annotation 和 schema overlay。SQLite 数据库
体积较大，不随仓库发布；四套测试都需要另外准备 BIRD dev databases。

| 基准 | 数量 | 文件 | 标注合同 |
| --- | ---: | --- | --- |
| Arcwise-Plat | 498 | `arcwise/arcwise_plat_full_with_diff.json` | SQL、问题/evidence、schema descriptions 校正 |
| Arcwise-Plat-SQL | 498 | `arcwise/arcwise_plat_sql_only_with_diff.json` | 只校正 SQL；其余保持原始 BIRD |
| BIRD Mini-Dev | 500 | `bird/mini_dev_sqlite.json` | 官方原始 Mini-Dev |
| BIRD Full Dev | 1,534 | `bird/dev_20240627.json` | 官方原始 Full Dev |

这些数据只用于模型与协议冻结后的最终测试，不参与训练、调参或 checkpoint 选择。
“最终测试”是本项目的实验角色，不表示 BIRD 官方隐藏 test。

## 来源

- Arcwise annotation 和 75 个 schema-description CSV：
  [uiuc-kang-lab/text_to_sql_benchmarks](https://github.com/uiuc-kang-lab/text_to_sql_benchmarks)，
  commit `fe766045c55b6875a43b30e9ac7683df5582f8cf`。
- BIRD Full Dev：BIRD 官方 `dev.zip` 中的 `dev_20240627/dev.json`；下载包最后修改
  时间为 2024-06-29，记录的 ETag 为 `04B4AF221C9186361F09B16ABFD917EC`。
- BIRD Mini-Dev：[bird-bench/mini_dev](https://github.com/bird-bench/mini_dev)
  发布的 SQLite annotation。

Arcwise 两个文件共享相同的 498 个唯一 question ID。它们有意省略 Mini-Dev ID
119 和 120，所以分母必须报告为 498。只有 Arcwise-Plat 使用
`arcwise/schemas/` 中的校正 descriptions；另外三套使用 dev database 旁的原始
descriptions。

## 慢 gold cache

`slow_gold_cache.json` 保存 ID 518 和 701 的 gold execution rows。它只避免极慢
gold 查询在并发评测中超时，预测 SQL 仍然现场执行。每项缓存由 gold SQL 的
SHA-256 绑定；annotation 中的 SQL 改变时加载会失败。

## SHA-256

```text
baefe2ca4fbab86c000df72aad9eaa563f7d422e425af2cbff071f656ea4eea8  arcwise/arcwise_plat_full_with_diff.json
e7bf76408f99266506ea558d84982c69422e5db3dfecab5f08a3a8d3900e8395  arcwise/arcwise_plat_sql_only_with_diff.json
630272f2b1c44d8cef2c3b246f623355cf0bbc1e832c81061df895530dfc2f06  bird/dev_20240627.json
def4b2b43a9b06955193418f24c9be170eb6d83763ab702311790e0cdda8c791  bird/mini_dev_sqlite.json
3153cff228c470419eda534315d3d56fe4464bcbe4f42a7bf83df536f6370fe8  slow_gold_cache.json
```

两类上游数据均按 CC BY-SA 4.0 发布；许可证副本分别位于 `arcwise/LICENSE` 和
`bird/LICENSE`。
