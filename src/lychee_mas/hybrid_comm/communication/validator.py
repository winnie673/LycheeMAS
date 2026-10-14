"""模型输出与 payload 的校验。"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from ..protocol.schema import ProtocolCard
from ..types import Decision, ErrorType


class PayloadValidator:
    """校验 JSON 控制输出、结构化字段与数值合法性。"""

    _DECIMAL_RE = re.compile(r"^-?\d+(\.\d+)?$")

    @classmethod
    def parse_json(cls, raw: str) -> dict[str, Any]:
        """解析外层 JSON；失败时抛出 ValueError。"""
        if not isinstance(raw, str):
            raise ValueError("model output must be a string")
        text = raw.strip()
        if not text:
            raise ValueError("empty model output")
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError("model output JSON must be an object")
        return obj

    @classmethod
    def validate_hybrid(
        cls, data: dict[str, Any], card: ProtocolCard, *, edge: str
    ) -> dict[str, Any]:
        """校验 Hybrid 字段并返回规范化字典。"""
        is_acceptor = edge == "verifier_to_solver" and data.get("decision") == Decision.ACCEPT.value
        if "fields" not in data or not isinstance(data["fields"], dict):
            raise ValueError(f"edge={edge}: hybrid payload must contain a 'fields' object")

        fields = {str(k): str(v) for k, v in data.get("fields", {}).items()}
        unknown = [k for k in fields if k not in card.allowed_fields]
        if unknown:
            raise ValueError(f"edge={edge}: unknown fields {unknown}")

        effective_required = card.required_fields
        if is_acceptor:
            effective_required = ["CHK"]
        elif edge == "verifier_to_solver":
            effective_required = ["CHK", "E", "REQ"]

        missing = [k for k in effective_required if k not in fields or fields[k] == ""]
        if missing:
            raise ValueError(f"edge={edge}: missing required fields {missing}")

        # The task Card may opt into decimal-only answers; the frozen Schema does not.
        answer_format = str(card.field_rules.get("answer_format", "flexible"))
        if "R" in fields and fields["R"] and answer_format == "decimal":
            cls._require_decimal(fields["R"], f"edge={edge}: field R")

        # Verifier 反馈字段一致性。
        if edge == "verifier_to_solver":
            cls._validate_verifier_feedback(data, fields)

        return {
            "fields": fields,
        }

    @classmethod
    def validate_nl(cls, data: dict[str, Any], role: str) -> dict[str, Any]:
        """校验 NL 控制字段并返回规范化字典。"""
        if role in ("planner", "solver"):
            message = data.get("message")
            if not isinstance(message, str) or message.strip() == "":
                raise ValueError(f"role={role}: NL message must be a non-empty string")

        if role == "solver":
            answer = data.get("answer")
            if not isinstance(answer, str) or answer.strip() == "":
                raise ValueError("solver NL answer must be a non-empty string")
            return {"message": message.strip(), "answer": answer.strip()}

        if role == "verifier":
            decision = cls._require_decision(data.get("decision"))
            error_type = None
            feedback = None
            if decision == Decision.REVISE:
                # NL wire schema names the human-readable feedback field "message".
                # Keep accepting the legacy "feedback" key for old result fixtures.
                feedback = data.get("message", data.get("feedback"))
                if not isinstance(feedback, str) or feedback.strip() == "":
                    raise ValueError("verifier REVISE feedback must be non-empty")
                error_type = cls._require_error_type(data.get("error_type"))
            return {
                "decision": decision,
                "error_type": error_type,
                "feedback": feedback,
                "message": str(data.get("message", feedback or "")),
            }

        # planner
        return {"message": str(data.get("message", "")).strip()}

    @classmethod
    def _validate_verifier_feedback(
        cls, data: dict[str, Any], fields: dict[str, str]
    ) -> None:
        decision = data.get("decision")
        if decision not in (Decision.ACCEPT.value, Decision.REVISE.value):
            raise ValueError(f"verifier decision must be ACCEPT or REVISE, got {decision!r}")

        chks = fields.get("CHK", "")
        es = fields.get("E", "")
        req = fields.get("REQ", "")

        if decision == Decision.ACCEPT.value:
            if chks != "ACCEPT" or es or req:
                raise ValueError("verifier ACCEPT must be CHK:ACCEPT without E or REQ")
            return

        if chks not in ("REVISE", "FAIL", "INSUFFICIENT"):
            raise ValueError(
                f"verifier CHK must be REVISE, FAIL or INSUFFICIENT, got {chks!r}"
            )
        if es not in {e.value for e in ErrorType}:
            raise ValueError(f"verifier E must be a valid ErrorType, got {es!r}")
        if data.get("error_type") != es:
            raise ValueError("verifier error_type must match feedback field E")
        if not req.strip():
            raise ValueError("verifier REVISE must provide a non-empty REQ")

    @classmethod
    def _require_decimal(cls, value: str, context: str) -> None:
        if not cls._DECIMAL_RE.match(value.strip()):
            raise ValueError(f"{context}: invalid decimal value {value!r}")
        try:
            num = float(value)
        except ValueError as exc:
            raise ValueError(f"{context}: invalid decimal value {value!r}") from exc
        if not math.isfinite(num):
            raise ValueError(f"{context}: value must be finite, got {value!r}")

    @classmethod
    def _require_decision(cls, value: Any) -> Decision:
        try:
            return Decision(value)
        except ValueError as exc:
            raise ValueError(f"invalid decision {value!r}") from exc

    @classmethod
    def _require_error_type(cls, value: Any) -> ErrorType:
        try:
            return ErrorType(value)
        except ValueError as exc:
            raise ValueError(f"invalid error_type {value!r}") from exc


class ValidationError(ValueError):
    """校验失败的显式异常。"""
