"""hybrid_comm 内部公共类型（纯标准库，零重依赖）。"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class CommunicationMode(str, Enum):
    """实验条件：纯自然语言或混合消息。"""

    NL = "nl"
    HYBRID = "hybrid"


class Decision(str, Enum):
    """Verifier 的语义判定。"""

    ACCEPT = "ACCEPT"
    REVISE = "REVISE"


class ErrorType(str, Enum):
    """Verifier 反馈的错误类型。"""

    ARITH = "ARITH"
    RELATION = "RELATION"
    MISSING_FACT = "MISSING_FACT"
    MISSING_UNIT = "MISSING_UNIT"
    WRONG_TARGET = "WRONG_TARGET"
    INSUFFICIENT_INFO = "INSUFFICIENT_INFO"
    OTHER = "OTHER"


@dataclass
class ModelCall:
    """一次模型请求的尝试记录。"""

    role: str
    attempt: int
    raw_output: str
    parsed: dict[str, Any] = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    fallback: bool = False
    transport_failed: bool = False
    parse_failed: bool = False
    error_message: str = ""
    ts: float = field(default_factory=time.time)


@dataclass
class MessageRecord:
    """一条实际传输的 envelope + payload。"""

    edge: str
    mode: CommunicationMode
    payload_format: str
    fallback: bool
    payload: str
    raw_output: str
    protocol_version: str
    protocol_hash: str
    sender: str
    receiver: str
    call: ModelCall = field(default_factory=ModelCall)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseResult:
    """单题运行结果。"""

    task_id: str
    question: str
    gold: str
    mode: CommunicationMode
    protocol_version: str
    final_answer: str = ""
    normalized_final_answer: str = ""
    score: float = 0.0
    run_status: str = "completed"
    failed_stage: str = ""
    failure_reason: str = ""
    decision: Optional[Decision] = None
    error_type: Optional[ErrorType] = None
    messages: list[MessageRecord] = field(default_factory=list)
    model_calls: list[ModelCall] = field(default_factory=list)
    revision_triggered: bool = False
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_s: float = 0.0
    static_input_tokens: int = 0
    dynamic_input_tokens: int = 0
    communication_tokens: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_new_id)


@dataclass
class GenResult:
    """统一生成原语返回结构。"""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    usage_unknown: bool = False
