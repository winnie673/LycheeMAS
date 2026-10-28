# `memory` — 记忆层（运行时组件，顶层包）

## 功能

回答「**智能体之间记住什么、以什么表征传递**」。两个对称的可替换接缝：

- **方法接缝（manager，类别 `memory_manager`）**：记忆如何存储/召回/物化——`managers/`。
- **触发接缝（router，类别 `memory_router`）**：当前 agent 本轮用哪个通道——`routing/`。

文本/隐空间策略由现有 manager 与 router 组合使用。原文提到的
`runtime/injection.py` 已被移除；通用 `attach_memory` 在 origin 中仍未实现。
当前通过建图方已有的 `node_factory` 注入通道内容，不新增任务专用执行框架。

## Hybrid 文本协议策略

### 核心机制与实验规格解耦（memory-compare-v3-contract）

核心 `protocol/base.py` 定义 `ProtocolArchitecture`、`ProtocolContract`、字段/条件
约束、`EvolutionSchedule` 及通用 Schema 转换，不提供具体任务、角色、字段常量。
`channels/hybrid_protocol.py` 只执行声明式规则并存取合法文本；
`protocol/evolution.py` 只消费契约、数据日程及生成/执行/评分回调，不认识 GSM8K 或 PSV。
字段规则支持枚举、禁用值、正则、条件必需字段及互斥输出，只有固定的安全操作，
不从卡片执行 Python、加载代码或动态导入验证函数。

字段基础定义的唯一来源是 `configs/memory/gsm8k_protocol_schema.yaml`。
`scripts/hybrid_gsm8k_spec.py` 加载一次，经通用 `build_contract_from_schema` 派生
字段接口、逐边语义、枚举、占位符禁用及条件约束；只补充 PSV 角色、数值答案要求、
接收端兼容别名和 D0/G1-G3/V/T 日程。生成/审查复用同一 Schema，不含兼容别名。
同名字段的逐边语义会全部保留，不以最后一条覆盖；Schema 的自然语言政策用于
设计审查，不冒称已经全部转成可执行校验。核心 src 不导入 scripts。
具体构图、NL 答案抽取、格式重试/fallback 和语义修订仍由实验执行层负责。
因此机制可复用，但现有 CLI 和执行图仍仅实例化 GSM8K + PSV，不自动适配任意图。

无内嵌契约的历史格式加载时仍必须显式提供 `protocol_contract`，缺少即报错，
绝不静默丢掉语义校验。配置目录不再保留旧卡，兼容性回归使用临时合成卡片。
新生成卡片包含完整 `validation_contract`，并记录架构/契约 SHA256，可独立加载；
若同时传入外部契约，两者必须一致。指纹用于兼容性/可复现性检查，不是数字签名。
默认已发布卡片的旧契约使用显式历史指纹白名单和运行时谓词等价检查，保留其
内嵌契约，不用新契约覆盖它。未知指纹、架构差异或校验强弱变化均报错。
`protocol_contract_for_card(path)` 负责该实验侧选择；核心通道仍严格核对外部/内嵌契约。
新生成卡片使用 Schema 派生的新契约，指纹可以不同。旧契约中空的新约束属性不
参与序列化，保证旧指纹稳定。不改卡片内容或数学评分，不重新解释历史结果；
新版本标为 `memory-compare-v5-schema-derived` / `memory-evolution-v5-schema-derived`。

其他架构接入需由调用方提供契约和执行回调，不需要改核心通道或进化算法：

```python
from lychee_mas.methods.memory.protocol import ProtocolEvolution

# contract、schedule、generate、execute、score 由实验提供。
# execute(case_without_gold, mode, card_path, audit_path) 必须真正通过 memory 通道运行。
algorithm = ProtocolEvolution(
    contract=contract, schedule=schedule, generate=generate, execute=execute,
    score=score, executor_capability="Description of the target executor",
)
result = algorithm.run(development_splits, new_output_directory)
```

核心当前选择目标是二值正确性（score=0/1），再比较精确通信 token；
不同任务的答案解析、评分与构图不属于通道。测试还覆盖两角色、颜色标签答案、
非 PSV 字段和自定义分区，另将字段重命名为 C/U/SAY，证明没有隐藏的字段绑定。

### GSM8K 离线生成、进化与在线选卡（2026-10-27）

离线方法位于 `protocol/base.py` 和 `protocol/evolution.py`；入口为
`scripts/run_memory_protocol_evolution.py`。不新增 registry 类别，不修改 plugins，
运行时仍为 CDM → NLMemory → HybridProtocolMemory。交付的实验入口只处理 GSM8K。

`psv_v1` 契约统一角色提示、三条通信边、字段接口及至多一次修订。
示例执行前显式检查卡片任务/角色/边/字段；新生成卡片另核对架构 ID 和 SHA256。
旧冻结卡片未修改，由实验显式注入完整校验契约并做结构兼容检查；
不能由此声称旧卡片本身已经携带架构指纹。
`protocol_path` 选择任意兼容 GSM8K 卡片；未知架构明确报错，不猜测角色映射。

流程为 D0 的 NL 通信轨迹 → 模型归纳 Card-0 → G1/G2/G3 上获取反馈、
生成 UPDATE/KEEP 候选 → 每轮在相同 V 子集分别运行父代/候选 → 冻结胜出卡片。
初始卡片不是手工编写或从旧 Card-2 复制；预设的只有字段接口与高层语义。
执行模型能力通过 `--executor-capability` 显式告知生成模型；两者可以使用不同模型、
端点和密钥环境变量。模型生成最多三次格式尝试，原文/错误/用量逐次留档。
节点生成仍复用现有三次尝试和一轮语义修订，不额外维护执行循环。

V 筛选先比较准确率，再比较完整通信 token；并列或 token 不完整时保留父代。
KEEP/只改版本描述等非语义内容不晋级。输出 token 单独报告，输入只留审计。
父代先、候选后为固定执行顺序，未消除 API 随时间变化的影响。
新产物在 `runs/memory_protocol_evolution/<唯一运行名>/final/gsm8k_card.yaml`，
不会自动覆盖 `configs/memory/task_cards/` 中的默认卡。中间各阶段保存 metrics、
evolution、selection、原始调用和失败信息；不自动访问 T 或启动最终评测。

在仓库根目录、LycheeMAS 环境中使用：

```powershell
$env:PYTHONPATH = "$PWD\src"
$env:PYTHONUTF8 = "1"
# 纯离线贯通检查：每个开发分区仅取一题，生成的是明确标记的假卡片。
python scripts/run_memory_protocol_evolution.py `
  --manifest ..\LycheeMAS\data\splits\gsm8k_protocol_nl_field_v1_seed42.json `
  --backend fake --limit 1 --out runs/memory_protocol_evolution/offline_001

# 真实实验需另行启动；密钥预先由安全方式提供为环境变量。
# 不传 --limit 时使用旧划分完整 D0/G1/G2/G3/V，不读取 T 作为生成素材。
python scripts/run_memory_protocol_evolution.py `
  --manifest ..\LycheeMAS\data\splits\gsm8k_protocol_nl_field_v1_seed42.json `
  --backend api --model deepseek-chat --proposer-model deepseek-chat `
  --executor-capability "Target executor: limited reasoning and strict-format reliability; prefer short explicit rules." `
  --tokenizer deepseek-ai/DeepSeek-V3 --out runs/memory_protocol_evolution/gsm8k_api_001

# 在线单题：卡片格式、架构检查和 memory 传递；默认假后端，不代表题目效果。
python examples/03_gsm8k_memory_protocol.py --benchmark gsm8k `
  --protocol-path runs/memory_protocol_evolution/gsm8k_api_001/final/gsm8k_card.yaml

# 独立启动最终 T 比较；--protocol-path 选新卡，不修改项目默认配置。
python scripts/run_memory_protocol_compare.py --benchmark gsm8k `
  --manifest ..\LycheeMAS\data\splits\gsm8k_protocol_nl_field_v1_seed42.json --split T `
  --backend api --model deepseek-chat --tokenizer deepseek-ai/DeepSeek-V3 `
  --protocol-path runs/memory_protocol_evolution/gsm8k_api_001/final/gsm8k_card.yaml `
  --out runs/memory_protocol_compare/gsm8k_new_card_001
```

API 默认关闭 SDK 隐式重试；`--proposer-base-url` 和 `--proposer-api-key-env`
可将生成模型与执行服务分离。API 必须有匹配执行模型的 tokenizer。
`--limit > 0` 单独标为 pilot，fake 标为 effect_valid=false，不作为实际效果证据。
原划分含历史 T→D0 的调整，沿用不意味着这是从未接触过的盲测集。
当前未实现自动给任意 MAS 图挂载；调用者须显式消费下面的通道接口。

### 按 benchmark 选择卡片

当前 `hybrid_protocol` 策略仅保留 GSM8K 冻结任务卡，不需要新增 channel 或修改 plugins：

| benchmark/task | 配置（仓库根目录相对路径） | 任务卡 |
| --- | --- | --- |
| `gsm8k` | `configs/memory/gsm8k_hybrid_protocol.yaml` | `configs/memory/task_cards/gsm8k_psv_hybrid_card_20261028.yaml` |

实验入口根据已知的 benchmark 标识选择上述配置，再按现有方式将 `task`、
`protocol_path` 交给 CDM。不要从题目文本猜任务，不要将未知 benchmark 静默映射到 GSM8K。
每个 benchmark 使用独立的 manager 实例；每题 reset，不在一条通信链中途换卡。

```python
config_paths = {
    "gsm8k": "configs/memory/gsm8k_hybrid_protocol.yaml",
}
if benchmark not in config_paths:
    raise ValueError(f"No hybrid task card configured for {benchmark!r}")
# Load config_paths[benchmark], construct the existing fixed router and CDM
# as in examples/03_gsm8k_memory_protocol.py. No channel implementation changes.
```

`task_card.task` 与传入的 `task` 不一致会显式报错，避免错用协议。当前卡片要求
Planner/Solver/Verifier 的命名及对应通信边，不会自动兼容任意角色图。

通道负责 `wire-v2` 风格的字段边界及 NL 占位符校验；
示例节点演示有界重试/fallback 和调用日志。
原离线协议实验的准确率不能直接视为本集成的实测结果。新增框架内 GSM8K 对照入口见下节。

### 框架内 NL/Hybrid 评测

`scripts/run_memory_protocol_compare.py` 只负责固定清单、逐题配对、评分和汇总，
直接复用现有示例的 `run_demo`，不另写节点工厂、运行器、模型后端或构图算法。
此入口只接受 GSM8K，默认串行、同题两模式交替先后顺序。

- NL 使用原有 `prev_output` 文本策略，生成自然语言解释，Solver 最后输出一个
  `\boxed{数值}`；Verifier 首行输出 ACCEPT 或 REVISE，REVISE 后附自然语言修改意见。
  不使用 JSON 或 Hybrid 字段。完整消息（包括这些控制标记）均计入通信 token。
- Hybrid 使用冻结 Card-2，仍由通道检验字段。两者均经 CDM observe/recall 传递，
  首节点无上游内容；prev_output 的 recall 参数是已交付消息而不是原题。
- 两模式使用同一个 static 图/node_factory。静态链展开为 P-S-V-S1-V1，首轮 ACCEPT
  时后两个节点无调用跳过；REVISE 时最多一次修订及复核，不改变框架图构建器。
- 修订 Solver 使用此前通过 recall 得到的规划、旧解答和反馈文本快照；保存这些
  MemoryBundle 文本快照是为适配“只保留最新消息”的通道，不直接拼接未经通道读取的输出。
  重复注入的上下文属于输入成本，不重复计作新通信。
- 每次角色激活最多三次生成尝试，两模式上限一致；最多五次角色激活、十五次模型调用。
  末次仍 REVISE 时标为 unverified，标准评分仍检查最后候选答案；调用耗尽的题目算运行失败，
  纳入总题数并记零分，所有有用量信息的失败尝试仍计费。
- GSM8K 采用 `strict_numeric_decimal_v1` 裸数值比较，不使用旧 `exact` 的子串兜底；
  保留负号，规整数值写法，不从完整推理中查找 gold。两模式使用相同评分函数。
- 首次 GSM8K 实测版本为 `memory-compare-v1`。历史 `memory-compare-v2` 修正 NL 最终
  boxed 答案后合法 LaTeX 闭合符的处理；只放行与盒子前开符匹配的闭合符及末尾句号，
  不允许额外推理、多个答案盒子或不配对的结束符。不修改数学答案，不按 gold 修复输出。
  不回写 GSM8K 历史成绩；新旧版本不能无说明合并。

输出目录约定：`runs/memory_protocol_compare/<唯一运行名>/`，目录必须不存在。
根目录保存 config.json（卡片/配置快照、源代码哈希、git、依赖版本及预算）、
selected_cases.json 和 comparison.json；各模式目录保存 outputs.jsonl、metrics.json、
逐题 attempts.jsonl 与原始 task.json。失败中断保存 failure.json，不产生伪造的完整 comparison。
不覆盖旧结果，不自动续跑，也不在评测过程中修改卡片或划分。

通信 token 用 backend.tok 对成功投递的完整 payload 精确编码，不含输入来源标题。
API 评测必须显式指定与服务模型匹配的 tokenizer（匹配关系由实验配置者核实）；
缺少调用 usage 时成本标为不完整，同时保留已知用量，不把未知用量当零。
记录任务墙钟耗时（含初始化/构图、调用、校验、逐次审计，不含最终写盘和评分）
以及模型报告的调用耗时之和，两者不混用。
假后端只验证流程，comparison 的 effect_valid=false，不代表 benchmark 效果。

```powershell
$env:PYTHONPATH = "$PWD\src"
# 本地只验证入口，不调用 API；示例清单为旧实验目录中的已冻结划分。
conda run --no-capture-output -n LycheeMAS python scripts/run_memory_protocol_compare.py `
  --benchmark gsm8k --manifest ..\LycheeMAS\data\splits\gsm8k_protocol_nl_field_v1_seed42.json `
  --split T --limit 2 --backend fake --out runs/memory_protocol_compare/offline_check_001

# 正式评测需另行启动；密钥由环境读取，不放命令行。
conda run --no-capture-output -n LycheeMAS python scripts/run_memory_protocol_compare.py `
  --benchmark gsm8k --manifest ..\LycheeMAS\data\splits\gsm8k_protocol_nl_field_v1_seed42.json `
  --split T --backend api --model deepseek-chat --tokenizer deepseek-ai/DeepSeek-V3 `
  --max-revision-rounds 1 --out runs/memory_protocol_compare/gsm8k_api_001
```

`--mode nl|hybrid|both`，默认 both；`--max-revision-rounds 0|1`，默认 1。
API 调用默认温度 0、seed 42、各角色最大输出 512 token，关闭 SDK 隐式重试。
控制流的最大三次尝试也包含暂态连接失败；不可恢复配置/代码/编码错误留档后中止整批。
不保证不同时间的 API 模型返回完全一致，使用固定清单不等于测试数据从未被接触。

本次离线验证覆盖了最多十五次调用的硬上限、两模式完整修订、失败留档、
状态隔离和 gold 不进入模型提示。另使用冻结 T 的两题做假后端 CLI 检查，
只确认结果文件可生成，未启动正式 API 批量评测。

成本主指标仍只比较通信和输出 token，输入用量留作审计，不推断总费用收益。

### 校验、重试与冒烟约定

`examples/03_gsm8k_memory_protocol.py` 保留原 static graph/node_factory，支持
`--benchmark gsm8k` 和 `--backend fake|api`。默认使用无网络假后端。
每节点最多三次尝试：正常生成、带失败原文及校验错误的格式纠正、保持字段语义的
简化提示 fallback。无效消息不进入 memory；三次仍失败则显式终止。这里只演示
格式恢复之外，现支持至多一轮 Verifier REVISE 引发的修订及复核；输出保留实际决定。

通道只抛出 `ProtocolValidationError` 标记消息格式错误，不调用模型或修复答案。
节点只重试这种异常及明确声明的临时传输异常；配置、编码、程序错误不得吞掉。
API 默认复用 OpenAICompatibleBackend，关闭 SDK 自带重试，避免叠加隐形调用。
该后端不暴露 API finish_reason 时记录为 unavailable，并用输出 token 达到上限
作为“疑似截断”保守拒绝信号，不声称拿到了真实终止原因。

运行记录约定：`runs/memory_protocol_smoke/<run_id>/` 下放每题尝试 JSONL 和
结果 JSON，运行前指定唯一的新文件，禁止覆盖既有日志。记录失败原文、具体错误、
输入消息、卡片版本/指纹、调用用量和成功交付消息，密钥脱敏。正式轨迹只放合法
消息，但其 token 记账包含本节点所有有用量记录的尝试；通信 token 只计成功交付
的完整 payload。无 tokenizer 时为 None，不使用近似统计。失败任务的调用证据
仍保留在 JSONL 中，不以一条伪造成功消息掩盖失败。

```powershell
$env:PYTHONPATH = "$PWD\src"
conda run --no-capture-output -n LycheeMAS python -m pytest tests/test_memory_hybrid_protocol.py -q
conda run --no-capture-output -n LycheeMAS python examples/03_gsm8k_memory_protocol.py --benchmark gsm8k
# 实际问题通过 --question 提供，--task-id 记录数据来源；不将 gold 传入节点。
conda run --no-capture-output -n LycheeMAS python examples/03_gsm8k_memory_protocol.py --benchmark gsm8k --backend api --model deepseek-chat --output runs/memory_protocol_smoke/my_run/gsm8k.json
```

API 密钥从 DEEPSEEK_API_KEY 环境变量读取，不能出现在命令行或配置中。正常的真实
调用不保证触发重试；边界、fallback、耗尽、超时和截断由合成脚本输出确定性测试。

历史 GSM8K API 冒烟记录在 `runs/memory_protocol_smoke/20261025_memory_wire_v2/`。
GSM8K 原 D0 第一题三次调用完成合法消息交付，输出 108（正确）。这是通信冒烟，
不是准确率结论，也不证明格式校验能识别数学错误。历史产物保持不变。

配置：`configs/memory/gsm8k_hybrid_protocol.yaml`。与现有 cdm.yaml 一致，
使用 `name: cdm` 和嵌套 router；`nl_strategy: hybrid_protocol` 选择 NLMemory
的内部策略。通道主体为 channels/hybrid_protocol.py，离线工具在 protocol/；
不需要新注册类别或修改插件接口。

```python
from lychee_mas import REGISTRY
from lychee_mas.core.types import Message
from lychee_mas.methods.memory.routing.base import RouteDecision
from scripts.hybrid_gsm8k_spec import protocol_contract_for_card

card_path = "configs/memory/task_cards/gsm8k_psv_hybrid_card_20261028.yaml"
contract = protocol_contract_for_card(card_path)

memory = REGISTRY.create(
    "memory_manager", "cdm", backend=backend,
    nl_strategy="hybrid_protocol",
    protocol_path=card_path, task="gsm8k",
    protocol_contract=contract,
)
# 发送节点将约束附加到自己的 system prompt，再使用原有 backend.generate_chat。
memory.nl.check_architecture(contract.architecture)  # 调用方仍须创建相符的图。
rules = memory.nl.output_instructions("planner", "solver")
# raw_output 是该次原有模型调用的结果，不增加转码调用。
memory.observe([Message(sender="planner", receiver="solver", content=raw_output)])
bundle = memory.recall(RouteDecision(channel="nl"), question)
# 接收节点只注入 bundle.NL_Channel；fields 元数据不再重复拼入 prompt。
```

- 默认任务卡片：v5-schema-derived 流程中 V 集最终保留的 GSM8K PSV Hybrid Card-0。
  来源为 `runs/memory_protocol_evolution/gsm8k_schema_derived_api_20261027_212336/final/gsm8k_card.yaml`；
  发布名为 `gsm8k_psv_hybrid_card_20261028.yaml`，保留源卡片的 `CARD-0-gsm8k` wire 版本、
  全部规则、示例及内嵌校验契约，只增加名称和发布来源路径/SHA256 元数据。
  两张旧配置卡已按用户确认清理。生成器仍只写 runs/，不会自动发布。
- observe 校验最新消息的 sender/receiver、允许/必需字段及 CHK/E/REQ；无效消息报错且
  不覆盖上一条合法消息。core.Message 与带 sender/receiver/content 的字典均支持。
- recall 与 prev_output 使用同一个 PREV_OUTPUT_HEADER；只转发最新已观察的 payload。
  query 保持为任务问题，不将问题误当协议消息。reset 清空本次消息。
- metadata 包含 fields、协议版本/指纹、收发角色、message_id 和通信 token；
  不改变 MemoryBundle 类型。通信 token 只对完整 payload（含字段名和分隔符）计数，
  标题属于输入开销。无 backend.tok 时明确为 None；编码异常显式传播。
- 原有 none/nl/latent/both 路由保留；混合协议只改变文本分量，latent 策略仍归 LatentMemory。
- 协议不决定发言顺序，不执行模型调用、重试或评分；这些职责保留在原有节点/调用方。
  示例 examples/03_gsm8k_memory_protocol.py 使用现有 static graph 和注入式生成回调，
  演示 P-S-V 与至多一轮修订，调度由共享示例节点完成，通道只支持 V-S 反馈消息。
- 本次没有新增通用 attach_memory 实现，不能声称给任意图调用该入口即可生效。

## 接口

### 原方案行为恢复（2026-10-27）

审查边界版 `memory-evolution-v4-review-boundary`：程序先验证保存卡片的运行时契约，
语义审查仅接收协议正文，排除 `task_card` 和 `validation_contract`，不将兼容别名
当成可由生成模型修改的字段定义。审查结果须满足布尔决策、真实字段路径及
阻断违规/可选建议分离的一致性检查；不一致时对同一卡片最多复核一次，分别留档。
持续不一致则明确失败，既不自动批准，也不把错误审查当成改卡依据。六次生成
上限和严格语义通过条件保留；已有失败产物仅用于独立定向检查，不直接冻结。

示例约束版 `memory-evolution-v3-example-grounded` 在 GSM8K 设计适配器中区分
失败轨迹与正确教学示例：从失败中归纳通信需求，再构造自洽的小例子；不能修复时
替换整例，不编造数值。审查建议不得违反数值答案约束，并区分消息证据缺失与
题目条件不足。初始重试增加替换示例提示；模型、温度、六次上限、严格审查、
Schema、数据划分及在线执行不变。新运行单独保存，不覆盖历史失败记录。

审查澄清版 `memory-evolution-v3-review-clarified` 仅对生成/审查提示明确：
`INSUFFICIENT_INFO` 属于 E，缺少证据使用 `CHK:REVISE;E:INSUFFICIENT_INFO;REQ:…`；
不增加 CHK 状态，不降低严格审查、六次上限或示例证据充分性要求。
该修改依据开发阶段审查记录，不使用 T 集结果；失败运行保留，新运行另存。

- `memory-compare-v4-restored`：实验规格保留原 Planner/Solver/Verifier 完整行为正文，
  NL 与 Hybrid 共用角色约束；接受 `CHK:FAIL/INSUFFICIENT` 作为修订别名，原消息不改写。
- `memory-evolution-v3`：GSM8K 入口注入 `scripts/hybrid_gsm8k_design.py`，迁入原设计
  纯函数及固定 schema（不是预设初始卡）。D0 最多六次生成并独立语义审查；每轮 G
  同时执行 NL/Hybrid 构造反馈，独立诊断后限定最多三处补丁、最多三次提案尝试。
  修改必须引用真实且被标为通信相关的案例；无此诊断直接 KEEP，不重写整卡。
- 节点默认正常生成、带原始输出和错误原因重试、最小合法格式 fallback 共最多三次。
  仅格式错误触发格式 fallback，传输错误不自动切换格式。fallback 保留答案字段约束。
  日志保留原始输出、错误、尝试类别、实际 finish_reason 与用量；仅凭 token 达预算
  不推断实际截断，实际 length 结尾若语法合法也按原实现允许投递。
- API 实验入口显式设置 SDK `max_retries=0`，避免与节点重试叠加；未照搬旧实现未传递
  SDK 重试参数的问题。不可恢复的程序错误仍直接失败，不伪装成模型格式错误。
- 有意保留当前差异：NL 为自然文本而非 JSON；修订后再次验证；V 集仍先比较正确率，
  再比较精确通信 token，不恢复旧效用公式。故不能称为旧实验结果的完全等价复现。
- 完整角色正文改变架构指纹，旧的内嵌指纹产物可能明确拒绝加载；历史冻结 Card-2
  不改写，由调用方显式注入契约。新实验必须写新目录，不能混合历史版本指标。
- 离线假后端只验证工程流程，不代表协议效果；实际 API 生成与评测需另行启动。

### `base.py` — 方法接缝契约

```python
@dataclass
class MemoryBundle:                     # 本轮要注入的内容（每通道一对「内容 + 产出方法」）
    NL_Channel: Optional[str]           # 文本通道注入内容
    NL_strategy: Optional[str]          # 产出它的 NL 方法名
    Latent_Channel: Optional[Any]       # 隐空间载荷（张量 / projector 栈）
    Latent_strategy: Optional[str]      # 产出它的 latent 方法名（决定注入方式）
    meta: dict

class MemoryManager(ABC):
    name: str
    def observe(self, messages: list[Any]) -> None                        # 更新记忆库
    def recall(self, decision: RouteDecision, query: str) -> MemoryBundle # 物化选定通道
    def reset(self) -> None                                               # 清空单次对话状态
```

### `routing/base.py` — 触发接缝契约

```python
Channel = Literal["none", "nl", "latent", "both"]

@dataclass
class RouterInputs:      # 路由器可条件化的信息
    role / task / turn / sender / receiver / query / availability / same_model_pair

@dataclass
class RouteDecision:
    channel: Channel; reason: str      # uses_latent() / uses_nl()

class MemoryRouter(ABC):
    def decide(self, x: RouterInputs) -> RouteDecision: ...
```

### 其它模块

- **`context.py` — `RoutingContext`**：跨 agent 共享的路由/记账状态（turn 计数、role→model 表、决策日志 `decisions`、可选 TraceStore / JsonlSpanLogger）。接口：`turn_of / bump_turn / same_model_pair / log_decision / set_case / log_span / reset`。
- **`store.py` — `MemoryStore`**：key→value 缓存接缝（可选容量 + FIFO）：`put/get/has/clear/__len__`。
- **`channels/`**：表征通道实现——`nl.py`（文本通道，多种 strategy）与 `latent.py`（隐空间通道统一接口，多种物化策略）；`c2c_*.py` 是 latent 的内部子实现，manager 不直接依赖。

## 已注册组件

| 类别 | 已实现 | 桩 |
|---|---|---|
| `memory_manager` | `cdm`（双通道） | `mem0`, `ama` |
| `memory_router` | `static`（(role×task)→channel 策略表）, `fixed`（恒定通道；工厂 `fixed_channel_router`） | `learned`, `soft_gate` |

## 约定

- import 本包触发全部注册，不触发 torch（惰性导入）。**`channels/c2c_projector.py` 顶层 import torch，不得出现在注册链上**（只被惰性加载函数引用）。
- 新增记忆方法/路由算法：继承对应 ABC + 注册 + 子包 `__init__` 触发 + config + test（六步配方）。
