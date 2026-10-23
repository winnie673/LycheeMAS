"""协议 YAML 的稳定项目格式化。"""

from __future__ import annotations

import json
import textwrap
from typing import Any

EDGES = ("planner_to_solver", "solver_to_verifier", "verifier_to_solver")


def dump_protocol_yaml(data: dict[str, Any]) -> str:
    """以稳定格式输出协议，避免 safe_dump 制造无意义的整文件 diff。"""
    lines = [f"version: {data.get('version', 'P0')}"]
    if data.get("allow_nl_field") is True:
        lines.append("allow_nl_field: true")
    description = " ".join(str(data.get("description", "")).split())
    if description:
        lines.append("description: >")
        sentences = description.replace(". ", ".\n").splitlines()
        for sentence in sentences:
            lines.extend(f"  {line}" for line in textwrap.wrap(sentence, width=88))

    for edge in EDGES:
        card = data.get(edge, {})
        lines.extend(["", f"{edge}:"])
        lines.append(f"  allowed_fields: {_flow_list(card.get('allowed_fields', []))}")
        lines.append(f"  required_fields: {_flow_list(card.get('required_fields', []))}")
        rules = card.get("field_rules", {})
        lines.append("  field_rules:")
        if "canonical_order" in rules:
            lines.append(f"    canonical_order: {_flow_list(rules['canonical_order'])}")
        for key, value in rules.items():
            if key != "canonical_order":
                if isinstance(value, dict):
                    lines.append(f"    {key}:")
                    lines.append(f"      usage: {_quoted(value.get('usage', ''))}")
                    lines.append(
                        f"      examples: {_quoted_flow_list(value.get('examples', []))}"
                    )
                else:
                    lines.append(f"    {key}: {_quoted(value)}")
        examples = card.get("examples", [])
        if examples:
            lines.append("  examples:")
            lines.extend(f"    - {_quoted(example)}" for example in examples)
        output_schema = card.get("output_schema")
        if output_schema:
            lines.append(f"  output_schema: {_quoted(output_schema)}")
    lines.extend(["", "message_format:"])
    message_format = data.get(
        "message_format", {"structure": "fields_only", "wire_syntax": "direct"}
    )
    for key, value in message_format.items():
        lines.append(f"  {key}: {_scalar(value)}")
    value_policy = data.get("value_policy", {})
    if value_policy:
        lines.extend(["", "value_policy:"])
        for key, value in value_policy.items():
            lines.append(f"  {key}: {_quoted(value)}")
    return "\n".join(lines) + "\n"


def _flow_list(values: list[Any]) -> str:
    return "[" + ", ".join(str(value) for value in values) + "]"


def _quoted_flow_list(values: list[Any]) -> str:
    return "[" + ", ".join(_quoted(value) for value in values) + "]"


def _quoted(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    unsafe = set(":#{}[],&*?|>!%@`")
    if text and not any(char in unsafe for char in text):
        return text
    return _quoted(text)
