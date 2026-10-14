"""将原始模型输出解析为控制字段与实际 payload。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..protocol.schema import ProtocolCard
from ..types import CommunicationMode
from .serializer import PayloadSerializer
from .validator import PayloadValidator


@dataclass
class ParsedPayload:
    """一次解析后的结果。"""

    mode: CommunicationMode
    edge: str
    payload_format: str
    control: dict[str, Any]
    payload: str
    final_answer: str = ""
    decision: str = ""


class PayloadParser:
    """将 LLM 原始输出转换为程序内部可消费的 payload。"""

    def __init__(self, mode: CommunicationMode) -> None:
        self.mode = mode

    def parse(
        self,
        raw: str,
        edge: str,
        card: ProtocolCard,
        *,
        role: str,
    ) -> ParsedPayload:
        """解析并校验一次角色输出。"""
        if self.mode == CommunicationMode.HYBRID:
            return self._parse_fields_only(raw, edge, card, role=role)
        data = PayloadValidator.parse_json(raw)

        if self.mode == CommunicationMode.NL:
            validated = PayloadValidator.validate_nl(data, role)
            payload = PayloadSerializer.serialize_nl(validated.get("message", ""))
            final_answer = validated.get("answer", "")
            decision_value = validated.get("decision", "")
            decision = getattr(decision_value, "value", decision_value)
            payload_format = "nl"
        else:
            validated = PayloadValidator.validate_hybrid(data, card, edge=edge)
            payload = PayloadSerializer.serialize_hybrid(validated["fields"], card)
            final_answer = validated["fields"].get("R", "")
            decision = data.get("decision", "")
            payload_format = "hybrid"

        return ParsedPayload(
            mode=self.mode,
            edge=edge,
            payload_format=payload_format,
            control={**data, **validated},
            payload=payload,
            final_answer=str(final_answer),
            decision=str(decision),
        )

    def _parse_fields_only(
        self, raw: str, edge: str, card: ProtocolCard, *, role: str
    ) -> ParsedPayload:
        """解析无 JSON envelope 的 fields-only wire syntax。"""
        payload = raw.strip()
        fields = PayloadSerializer.deserialize_hybrid_strict(payload, card)
        data: dict[str, Any] = {"fields": fields}
        if role == "verifier":
            data["decision"] = "ACCEPT" if fields.get("CHK") == "ACCEPT" else "REVISE"
            data["error_type"] = fields.get("E")
        validated = PayloadValidator.validate_hybrid(data, card, edge=edge)
        return ParsedPayload(
            mode=self.mode,
            edge=edge,
            payload_format="fields_only",
            control={**data, **validated},
            payload=payload,
            final_answer=fields.get("R", ""),
            decision=str(data.get("decision", "")),
        )

    def parse_fallback(
        self,
        raw: str,
        edge: str,
        card: ProtocolCard,
        *,
        role: str,
    ) -> ParsedPayload:
        """回退调用使用统一简化 envelope：{'message': ..., 'answer'|...}。"""
        if self.mode == CommunicationMode.HYBRID:
            return self._parse_fields_only(raw, edge, card, role=role)
        data = PayloadValidator.parse_json(raw)
        if not isinstance(data.get("message"), str):
            raise ValueError("fallback payload must contain a string 'message'")

        validated = PayloadValidator.validate_nl(
            {
                "message": data["message"],
                "answer": data.get("answer", ""),
                "decision": data.get("decision"),
                "error_type": data.get("error_type"),
            },
            role,
        )
        payload = PayloadSerializer.serialize_nl(validated["message"])
        final_answer = validated.get("answer", "")
        decision = validated.get("decision", "")

        return ParsedPayload(
            mode=self.mode,
            edge=edge,
            payload_format="fallback",
            control={**data, **validated},
            payload=payload,
            final_answer=str(final_answer),
            decision=str(decision),
        )
