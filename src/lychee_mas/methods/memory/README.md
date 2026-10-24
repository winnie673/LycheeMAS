# `memory` — 记忆层（运行时组件，顶层包）

## 功能

回答「**智能体之间记住什么、以什么表征传递**」。两个对称的可替换接缝：

- **方法接缝（manager，类别 `memory_manager`）**：记忆如何存储/召回/物化——`managers/`。
- **触发接缝（router，类别 `memory_router`）**：当前 agent 本轮用哪个通道——`routing/`。

文本/隐空间策略由现有 manager 与 router 组合使用。原文提到的
`runtime/injection.py` 已被移除；通用 `attach_memory` 在 origin 中仍未实现。
当前通过建图方已有的 `node_factory` 注入通道内容，不新增任务专用执行框架。

## Hybrid 文本协议策略

配置：`configs/memory/gsm8k_hybrid_protocol.yaml`。与现有 cdm.yaml 一致，
使用 `name: cdm` 和嵌套 router；`nl_strategy: hybrid_protocol` 选择 NLMemory
的内部策略。新增实现只有 channels/hybrid_protocol.py，不需要新注册类别或改插件接口。

```python
from lychee_mas import REGISTRY
from lychee_mas.core.types import Message
from lychee_mas.methods.memory.routing.base import RouteDecision

memory = REGISTRY.create(
    "memory_manager", "cdm", backend=backend,
    nl_strategy="hybrid_protocol",
    protocol_path="configs/memory/task_cards/gsm8k_card2.yaml", task="gsm8k",
)
# 发送节点将约束附加到自己的 system prompt，再使用原有 backend.generate_chat。
rules = memory.nl.output_instructions("planner", "solver")
# raw_output 是该次原有模型调用的结果，不增加转码调用。
memory.observe([Message(sender="planner", receiver="solver", content=raw_output)])
bundle = memory.recall(RouteDecision(channel="nl"), question)
# 接收节点只注入 bundle.NL_Channel；fields 元数据不再重复拼入 prompt。
```

- 任务卡片：V 集筛选的 GSM8K Card-2，字段和规则不变，增加 task/roles/edges 来源元信息。
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
  演示 P-S-V 三次通信。可选修订应由调用方安排，通道支持 V-S 反馈消息。
- 本次没有新增通用 attach_memory 实现，不能声称给任意图调用该入口即可生效。

## 接口

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
