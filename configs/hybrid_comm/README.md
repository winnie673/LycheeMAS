# Hybrid communication configs

该目录只保存可复用配置与协议产物；逐题输出、诊断记录和指标保存在
`runs/hybrid_comm/`。

```text
hybrid_comm/
├── experiment.yaml          # 实验默认参数
├── models.yaml              # 模型映射
├── protocol_schema.yaml     # 冻结的任务无关 Protocol Schema
├── prompts/                 # Planner、Solver、Verifier 提示模板
├── tasks/                   # 任务特定答案表示、推理风险和诊断标签
├── protocols/               # 人工设置的任务基线协议
└── generated/
    ├── gsm8k/
    │   ├── initial/
    │   ├── evolution/
    │   └── final/
    └── math500/
        ├── initial/         # Card_0 及初始化阶段保留版本
        ├── evolution/       # 各轮 candidate 与 winner
        └── final/           # 验证集选择后冻结的最终 Card

独立复现实验可在 `generated/` 下建立同构的版本目录，例如 `math500_v2/`，
使用 `--generated-root` 指定，避免覆盖先前的 Card。
```

约定：

- `protocol_schema.yaml` 的字段集合、语义、顺序和控制值域在进化期间保持冻结。
- 自 `SCHEMA-3` 起，`R` 仅表示任务目标的精确候选答案；具体答案格式由任务
  Card 的 `solver_to_verifier.field_rules.R` 规定。GSM8K 的有限数值约束因此保留
  在 GSM8K Card 中，而不属于通用 Schema。
- `protocols/` 中的文件是人工基线，不视为 LLM 生成的正式 Card。
- `generated/` 仅保存可复用 Card YAML，不保存运行日志或中间 JSON。
- Schema、生成、诊断与进化约束保持任务无关；`--task-profile` 将 `tasks/`
  中的任务要求注入 `initialize_protocol_card.py` 和 `evolve_protocol_card.py`。
- GSM8K 正式运行优先使用
  `generated/gsm8k/final/card_gsm8k_evolved.yaml`；需要复现实验过程时再使用
  `initial/` 和 `evolution/` 中的阶段版本。
