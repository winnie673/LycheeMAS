"""协议卡片的内存表示与基础约束。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProtocolCard:
    """单条边级协议卡片。"""

    edge: str
    allowed_fields: list[str] = field(default_factory=list)
    required_fields: list[str] = field(default_factory=list)
    field_rules: dict[str, Any] = field(default_factory=dict)
    message_format: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    version: str = "P0"
    examples: list[str] = field(default_factory=list)
    output_schema: str = ""
    value_policy: dict[str, Any] = field(default_factory=dict)

    def canonical_order(self) -> list[str]:
        """保留卡片定义的顺序；未声明的字段按字母序追加。"""
        declared = list(self.field_rules.get("canonical_order", []))
        tail = [f for f in self.allowed_fields if f not in declared]
        return declared + sorted(tail)

    def validate_field_name(self, name: str) -> None:
        if name not in self.allowed_fields:
            raise ValueError(f"edge={self.edge}: field {name!r} not allowed")


@dataclass
class ProtocolSuite:
    """三张卡片组成的一套协议。"""

    version: str = "P0"
    planner_to_solver: ProtocolCard = field(
        default_factory=lambda: ProtocolCard("planner_to_solver")
    )
    solver_to_verifier: ProtocolCard = field(
        default_factory=lambda: ProtocolCard("solver_to_verifier")
    )
    verifier_to_solver: ProtocolCard = field(
        default_factory=lambda: ProtocolCard("verifier_to_solver")
    )
    message_format: dict[str, Any] = field(
        default_factory=lambda: {"structure": "fields_only", "wire_syntax": "direct"}
    )
    value_policy: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def card_for(self, edge: str) -> ProtocolCard:
        mapping = {
            "planner_to_solver": self.planner_to_solver,
            "solver_to_verifier": self.solver_to_verifier,
            "verifier_to_solver": self.verifier_to_solver,
        }
        if edge not in mapping:
            raise ValueError(f"unknown edge {edge!r}")
        return mapping[edge]

    def hash_material(self) -> str:
        """返回用于计算 protocol_hash 的稳定文本（不追求密码学强度）。"""
        parts = [self.version]
        for card in (self.planner_to_solver, self.solver_to_verifier, self.verifier_to_solver):
            parts.append(card.edge)
            parts.append(",".join(card.canonical_order()))
            parts.append(",".join(sorted(card.required_fields)))
        return "|".join(parts)
