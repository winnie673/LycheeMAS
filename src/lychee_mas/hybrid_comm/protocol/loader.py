"""从 YAML 加载协议卡片。"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from .schema import ProtocolCard, ProtocolSuite


def _load_yaml(path: str) -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - config extra
        raise ImportError(
            "loading protocol YAML requires PyYAML; install with `uv pip install -e '.[config]'`"
        ) from exc
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _dict_to_card(
    edge: str,
    data: dict[str, Any],
    version: str,
    message_format: dict[str, Any],
    value_policy: dict[str, Any],
) -> ProtocolCard:
    examples = data.get("examples", [])
    if isinstance(examples, str):
        examples = [examples]
    return ProtocolCard(
        edge=edge,
        allowed_fields=list(data.get("allowed_fields", [])),
        required_fields=list(data.get("required_fields", [])),
        field_rules=dict(data.get("field_rules", {})),
        message_format=dict(data.get("message_format", message_format)),
        description=str(data.get("description", "")),
        version=version,
        examples=[str(e) for e in examples],
        output_schema=str(data.get("output_schema", "")),
        value_policy=dict(data.get("value_policy", value_policy)),
    )


def load_protocol_from_yaml(path: str) -> ProtocolSuite:
    """加载单文件协议套件。

    YAML 结构示例：
        version: P0
        planner_to_solver:
          allowed_fields: [K, G, P, C, U, UNIT]
          required_fields: [G, P]
          ...
        solver_to_verifier: ...
        verifier_to_solver: ...
        message_format:
          structure: fields_only
          wire_syntax: direct
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"protocol file not found: {path}")

    raw = _load_yaml(path)
    if not isinstance(raw, dict):
        raise ValueError(f"protocol YAML root must be a mapping: {path}")
    if "nl_policy" in raw:
        raise ValueError("nl_policy is no longer supported; use flexible values inside fields")
    for edge in ("planner_to_solver", "solver_to_verifier", "verifier_to_solver"):
        card_data = raw.get(edge, {})
        if "nl_policy" in card_data:
            raise ValueError(f"{edge}: nl_policy is no longer supported")
        if "NL" in card_data.get("allowed_fields", []):
            raise ValueError(f"{edge}: a separate NL field is not supported")

    version = str(raw.get("version", "P0"))
    message_format = dict(
        raw.get(
            "message_format",
            {"structure": "fields_only", "wire_syntax": "direct"},
        )
    )
    if message_format.get("structure") != "fields_only":
        raise ValueError("message_format.structure must be fields_only")
    if message_format.get("wire_syntax") != "direct":
        raise ValueError("message_format.wire_syntax must be direct")
    suite = ProtocolSuite(
        version=version,
        planner_to_solver=_dict_to_card(
            "planner_to_solver", raw.get("planner_to_solver", {}), version, message_format,
            dict(raw.get("value_policy", {}))
        ),
        solver_to_verifier=_dict_to_card(
            "solver_to_verifier", raw.get("solver_to_verifier", {}), version, message_format,
            dict(raw.get("value_policy", {}))
        ),
        verifier_to_solver=_dict_to_card(
            "verifier_to_solver", raw.get("verifier_to_solver", {}), version, message_format,
            dict(raw.get("value_policy", {}))
        ),
        message_format=message_format,
        value_policy=dict(raw.get("value_policy", {})),
        meta={"source": path},
    )
    suite.meta["protocol_hash"] = _hash_protocol(raw)
    return suite


def _hash_protocol(raw: dict[str, Any]) -> str:
    material = json.dumps(raw, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(material.encode('utf-8')).hexdigest()[:16]}"


def protocol_suite_to_dict(suite: ProtocolSuite) -> dict[str, Any]:
    """将协议套件序列化为可落盘的字典。"""

    def _card_dict(card: ProtocolCard) -> dict[str, Any]:
        return {
            "edge": card.edge,
            "allowed_fields": card.allowed_fields,
            "required_fields": card.required_fields,
            "field_rules": card.field_rules,
            "message_format": card.message_format,
            "description": card.description,
            "examples": card.examples,
            "output_schema": card.output_schema,
        }

    return {
        "version": suite.version,
        "planner_to_solver": _card_dict(suite.planner_to_solver),
        "solver_to_verifier": _card_dict(suite.solver_to_verifier),
        "verifier_to_solver": _card_dict(suite.verifier_to_solver),
        "message_format": suite.message_format,
        "value_policy": suite.value_policy,
        "meta": suite.meta,
    }
