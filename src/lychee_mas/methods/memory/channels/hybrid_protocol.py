"""Internal frozen-card strategy for NLMemory; no graph, backend or LLM calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class HybridProtocolMemory:
    """Validate and retain one previous-agent message, like the prev_output strategy."""

    ERROR_TYPES = frozenset({
        "ARITH", "RELATION", "MISSING_FACT", "MISSING_UNIT",
        "WRONG_TARGET", "INSUFFICIENT_INFO", "OTHER",
    })

    def __init__(self, protocol_path: str, task: str, tokenizer: Any = None) -> None:
        import yaml  # Config dependency, loaded only when this strategy is used.

        raw = yaml.safe_load(Path(protocol_path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("task card must be a mapping")
        scope = raw.get("task_card", {})
        if not isinstance(scope, dict) or scope.get("task") != task:
            raise ValueError(f"task card does not support task {task!r}")
        roles, edges = scope.get("roles", []), scope.get("edges", [])
        if not roles or not edges:
            raise ValueError("task card must declare roles and edges")
        self._cards: dict[str, dict] = {}
        self._pairs: dict[tuple[str, str], str] = {}
        for edge in edges:
            if not isinstance(edge, str) or edge.count("_to_") != 1:
                raise ValueError("task card has an invalid edge")
            sender, receiver = edge.split("_to_")
            if sender not in roles or receiver not in roles:
                raise ValueError(f"edge {edge!r} refers to undeclared roles")
            card = raw.get(edge)
            if not isinstance(card, dict):
                raise ValueError(f"missing card for edge {edge!r}")
            allowed, required = card.get("allowed_fields", []), card.get("required_fields", [])
            if not allowed or not set(required).issubset(allowed):
                raise ValueError(f"invalid fields for edge {edge!r}")
            if "NL" in allowed and raw.get("allow_nl_field") is not True:
                raise ValueError("NL field requires allow_nl_field: true")
            self._cards[edge] = card
            self._pairs[sender, receiver] = edge
        if raw.get("message_format", {}).get("wire_syntax") != "direct":
            raise ValueError("hybrid_protocol requires direct wire syntax")
        if raw.get("message_format", {}).get("structure") != "fields_only":
            raise ValueError("hybrid_protocol requires fields_only structure")
        self.version = str(raw["version"])
        material = {key: value for key, value in raw.items() if key != "task_card"}
        self.protocol_hash = hashlib.sha256(json.dumps(
            material, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        ).encode()).hexdigest()
        self.task = task
        self._value_policy = dict(raw.get("value_policy", {}))
        self.tokenizer = tokenizer
        self._last: dict[str, Any] | None = None

    def _edge(self, sender: str, receiver: str | None) -> str:
        try:
            return self._pairs[sender, receiver]
        except KeyError as exc:
            raise ValueError(f"unsupported task-card edge: {sender!r} -> {receiver!r}") from exc

    def output_instructions(self, sender: str, receiver: str) -> str:
        """The sending node appends these instructions before its existing model call."""
        edge = self._edge(sender, receiver)
        card = self._cards[edge]
        lines = [
            f"Task: {self.task}; edge: {edge}; protocol: {self.version}",
            "Output only FIELD:value segments separated by semicolons, without JSON or Markdown.",
            "Values may contain concise natural language, symbols or a mixture.",
            "Do not use semicolons inside values. Omit unused optional fields.",
            f"Allowed fields: {', '.join(card['allowed_fields'])}",
            f"Required fields: {', '.join(card['required_fields'])}",
        ]
        for name, rule in card.get("field_rules", {}).items():
            if name != "canonical_order":
                lines.append(f"{name}: {rule}")
        for name, rule in self._value_policy.items():
            lines.append(f"Value policy {name}: {rule}")
        if card.get("output_schema"):
            lines.append(card["output_schema"])
        if "CHK" in card["allowed_fields"]:
            lines.append(f"Error categories for E: {', '.join(sorted(self.ERROR_TYPES))}")
        lines.extend(f"Example: {example}" for example in card.get("examples", []))
        return "\n".join(lines)

    def validate(self, payload: str, sender: str, receiver: str | None) -> dict[str, str]:
        edge = self._edge(sender, receiver)
        card = self._cards[edge]
        if not isinstance(payload, str) or not payload.strip():
            raise ValueError("protocol payload must be a nonempty string")
        fields: dict[str, str] = {}
        for segment in payload.strip().split(";"):
            name, separator, value = segment.strip().partition(":")
            name, value = name.strip(), value.strip()
            if not separator or not value:
                raise ValueError(f"invalid field segment: {segment!r}")
            if name not in card["allowed_fields"]:
                raise ValueError(f"unknown field {name!r} on {edge}")
            if name in fields:
                raise ValueError(f"duplicate field {name!r}")
            fields[name] = value
        missing = set(card["required_fields"]) - fields.keys()
        if missing:
            raise ValueError(f"missing required fields: {sorted(missing)}")
        if "CHK" in card["allowed_fields"]:
            if fields.get("CHK") == "ACCEPT":
                if set(fields) != {"CHK"}:
                    raise ValueError("ACCEPT must contain only CHK:ACCEPT")
            elif fields.get("CHK") == "REVISE":
                if not fields.get("REQ") or fields.get("E") not in self.ERROR_TYPES:
                    raise ValueError("REVISE requires a valid E and a nonempty REQ")
            else:
                raise ValueError("CHK must be ACCEPT or REVISE")
        return fields

    def observe(self, sender: str, payload: str, *, receiver: str | None,
                message_id: str | None = None) -> None:
        fields = self.validate(payload, sender, receiver)
        payload = payload.strip()
        tokens = None
        if self.tokenizer is not None:
            tokens = len(self.tokenizer.encode(payload, add_special_tokens=False))
        # Commit only after all validation/counting succeeds.
        self._last = {
            "sender": sender, "receiver": receiver, "message_id": message_id,
            "protocol_edge": self._edge(sender, receiver),
            "protocol_version": self.version, "protocol_hash": self.protocol_hash,
            "fields": fields, "communication_tokens": tokens,
            "communication_tokens_exact": tokens is not None, "payload": payload,
        }

    def recall(self) -> str:
        return self._last["payload"] if self._last else ""

    def metadata(self) -> dict[str, Any]:
        if self._last is None:
            return {}
        result = {key: value for key, value in self._last.items() if key != "payload"}
        result["fields"] = dict(result["fields"])
        return result

    def reset(self) -> None:
        self._last = None
