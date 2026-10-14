# MATH-500 Card 生成、进化与评测

## 实验划分与统计口径

`scripts/generate_math500_manifest.py` 使用随机种子 42 对 MATH-500 的 500 道题进行互斥划分：D0 为 20 题，用于采集 Free-NL 通信轨迹并初始化 Card；G1、G2、G3 各为 50 题，用于三轮反馈驱动进化；V 为固定的 100 题验证集，用于候选选择；T 为 200 题，用于协议冻结后的对比评测；其余 30 题为 reserve。划分文件为 `data/splits/math500_protocol_v2_seed42.json`，哈希为 `sha256:7d34d428de8f6398`。

实验后端为 DeepSeek API `deepseek-chat`，通信 token 使用 `deepseek-ai/DeepSeek-V3` tokenizer 计算。答案采用任务专用 `math500` 数学等价评分器。运行失败样例的效用记为 0；其余样例的效用定义为：

```text
utility = isCorrect + 1 / communication_tokens
```

候选 Card 只在固定 V 集上选择，T 集不参与初始化、进化或选择。本项目此前已多次查看该 T 集，因此本次 T 集结果属于固定划分上的复现实验，不能表述为严格未见数据上的泛化结果。

## 执行流程

生成约束作用于 Card Generator 与独立语义审查器，不直接改写已有 Card 或运行时字段规则。新 Card 的任务特定规则仍由 LLM 归纳。生成与审查应共同要求：IR 支撑从关键关系到最终答案的决定性操作及必要分支；G 与 R 对应；P 不引入无依据假设；符号与简洁自然语言按语义需要混合；完整示例的证据足以支持答案；Verifier 区分局部错误与证据不足。表达长度和关系数量不设固定上限，压缩服从语义充分性。

1. 在 D0 上运行 Free-NL Planner–Solver–Verifier，保存 20 组完整通信轨迹。
2. Card Generator 接收冻结 Schema、MATH-500 任务配置、D0 题目及对应 Free-NL 轨迹，只生成 `field_rules`、完整消息示例和 `value_policy`。
3. 生成结果经过确定性结构校验和独立语义审查，得到 Card0。
4. 每轮分别在 Gi 上运行同题配对的 Pure-NL 与 Hybrid，构造错误和高成本反馈集合。
5. Diagnosis LLM 对每个选中案例判断通信缺失、歧义、冗余或基础推理错误，并将诊断条目绑定到原始案例 ID。
6. Evolver 只能根据被判定为通信相关的诊断证据修改最多三个既有路径；不得改变 Schema、字段集合、字段语义、通信边或消息语法，也可以返回 `KEEP`。
7. 对 `UPDATE` 产生的单一候选 Card，在固定 V 集上与父 Card 比较平均效用。胜者成为下一轮父 Card；`KEEP` 时直接沿用父 Card。
8. 三轮后冻结胜出 Card，并在 T 集分别运行 Pure-NL 与 Hybrid。

完整流水线通过 `scripts/run_math500_card_pipeline.py` 执行，本轮检查点保存在 `runs/hybrid_comm/math500_constraints_v2/pipeline_state.json`。Card0 生成时增加了一项机器可校验的 JSON 形状约束：字段级 `examples` 必须是只含一个字符串的数组，避免模型返回标量字符串导致初始化失败。Evolver 同时接收精确的可引用诊断案例 ID 列表，防止修改引用不存在或非通信相关的案例。

## Card 初始化与进化结果

采用新增证据充分性、目标一致性、变换保真和反馈分类约束重新生成 Card0。Card0 位于 `configs/hybrid_comm/generated/math500_constraints_v2/initial/card_0_math500.yaml`，已通过冻结 Schema 校验和独立语义审查。

| 阶段 | 动作 | 修改路径 | V 正确率 | 平均通信 token | 平均效用 | 选择结果 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Card0 | 初始化 | — | 83.0% | 178.73 | 0.838009 | 初始父 Card |
| Round 1 | UPDATE | `planner_to_solver.field_rules.P` | 81.0% | 177.16 | 0.818149 | 拒绝，保留 Card0 |
| Round 2 | KEEP | — | 83.0% | 178.73 | 0.838009 | 沿用 Card0 |
| Round 3 | UPDATE | `planner_to_solver.field_rules.P` | 82.0% | 179.95 | 0.827649 | 拒绝，保留 Card0 |

第一轮和第三轮均根据通信相关诊断修改 Planner 的 `P` 规则，但在固定 V 集分别下降 2 个和 1 个百分点，因此均被拒绝。第二轮没有形成证据充分的最小修改，Evolver 返回 `KEEP`。最终冻结的是重新生成的 Card0：`configs/hybrid_comm/generated/math500_constraints_v2/final/card_math500_evolved.yaml`。相较上一轮旧约束生成的 Card0，新的 Card0 在相同 V 集上由 73.0% 提高到 83.0%；由于两次运行均重新采样，该差异同时包含协议变化和模型随机性，不能直接解释为 10 个百分点的确定因果效应。

## T 集复现结果

| 指标 | Pure-NL | Hybrid Card0 |
| --- | ---: | ---: |
| 正确题数/200 | 177 | 160 |
| Accuracy | 88.5% | 80.0% |
| 平均通信 token | 245.5 | 170.1 |
| 平均输出 token | 351.1 | 212.6 |
| 平均通信与输出 token | 596.6 | 382.7 |
| 平均模型调用次数 | 3.13 | 3.28 |
| 平均延迟（秒） | 13.095 | 13.544 |
| 运行失败数 | 8 | 6 |
| 触发修订数 | 1 | 29 |
| 平均效用 | 0.890013 | 0.807857 |

两组输出的 200 个题目 ID 与标准答案逐项一致，使用当前评分器独立复算后仍分别为 177 题和 160 题正确。配对结果为：两者都正确 152 题，仅 Pure-NL 正确 25 题，仅 Hybrid 正确 8 题，两者都错误 15 题。

Hybrid 将平均通信 token 降低 30.7%，将平均输出 token 降低 39.4%，准确率低于 Pure-NL 8.5 个百分点。29 次 Hybrid 修订中，2 次由错改对、16 次错后仍错、10 次对后仍对，另有 1 次未形成可比较的第二条 Solver 消息，没有发生可观测的对→错修订。错误类型包括 20 次 `ARITH` 和 9 次 `INSUFFICIENT_INFO`；新增生成约束使证据不足不再全部被误标为答案错误，但修订的净纠错收益仍然有限。

相较上一轮复现，Hybrid 从 147/200 提高到 160/200，平均通信 token 从 145.5 增加到 170.1。逐题比较显示两轮 Hybrid 均正确 136 题，新卡片单独正确 24 题，旧卡片单独正确 11 题。结果与“增加决定性证据可改善正确率”的方向一致，但两轮重新生成了全部消息，不能把净增 13 题全部归因于生成约束。当前协议仍未完全达到准确率与通信效率的平衡：它保留明显的通信节省，但相对本轮 Pure-NL 仍低 8.5 个百分点。后续应在新的未见划分上重复运行，并把相对 NL 的准确率差、修订净收益和格式失败率加入候选门控。

## 产物位置

- 流程状态：`runs/hybrid_comm/math500_constraints_v2/pipeline_state.json`
- 复用的 D0 Free-NL 轨迹：`runs/hybrid_comm/math500/initialization/`
- Card 生成记录：`runs/hybrid_comm/math500_constraints_v2/card_generation/`
- 三轮反馈、诊断和选择：`runs/hybrid_comm/math500_constraints_v2/evolution/`
- V 集运行：`runs/hybrid_comm/math500_constraints_v2/validation/`
- T 集逐题结果与汇总：`runs/hybrid_comm/math500_constraints_v2/final_test/`
- 最终冻结 Card：`configs/hybrid_comm/generated/math500_constraints_v2/final/card_math500_evolved.yaml`

上一轮结果保留在 `runs/hybrid_comm/math500/`，用于同一固定划分上的复现对比。
