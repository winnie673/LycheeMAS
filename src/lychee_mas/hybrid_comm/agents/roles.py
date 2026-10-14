"""构造 Planner、Solver、Verifier 三角色的输入消息。"""

from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Any

from ..communication import hybrid, nl
from ..protocol.schema import ProtocolCard, ProtocolSuite
from ..types import CommunicationMode


class RolePromptBuilder:
    """根据当前角色、通信模式和协议卡片组装 prompt。"""

    _ROLE_ORDER = {
        "planner": ("planner_to_solver", "solver"),
        "solver": ("solver_to_verifier", "verifier"),
        "verifier": ("verifier_to_solver", "solver"),
    }

    def __init__(self, prompt_dir: str | None = None, task_name: str = "gsm8k"):
        if prompt_dir is None:
            prompt_dir = str(_default_prompt_dir())
        self.prompt_dir = Path(prompt_dir)
        self.task_name = task_name
        self._cache: dict[str, str] = {}
        self._static_token_counts: dict[tuple[str, str, str, bool], int] = {}

    def build_messages(
        self,
        role: str,
        mode: CommunicationMode,
        suite: ProtocolSuite,
        question: str,
        incoming: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """返回 OpenAI 风格的 messages 列表。"""
        system = self._system_prompt(role, mode, suite)
        user = self._user_prompt(role, mode, suite, question, incoming)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _system_prompt(self, role: str, mode: CommunicationMode, suite: ProtocolSuite) -> str:
        template = self._load(f"{role}_system.txt")
        card_edge, _ = self._ROLE_ORDER[role]
        card = suite.card_for(card_edge) if card_edge else None
        card_text = format_card_text(card, mode) if card else "No protocol card for this role."
        if self.task_name == "math500":
            template = template.replace("final numeric answer", "final mathematical answer")
        result = Template(template).substitute(
            base_rules=self._load("system_rules.txt").strip(),
            mode_rules=_mode_rules(mode, card),
            card_text=card_text,
            output_schema=_output_schema(role, mode, card, self.task_name),
            examples=_examples(role, mode, card),
        )
        if self.task_name == "math500":
            result += (
                "\nFor MATH-500, preserve exact mathematical forms, domains, and branches. "
                "Do not force a decimal approximation."
            )
        return result

    def _user_prompt(
        self,
        role: str,
        mode: CommunicationMode,
        suite: ProtocolSuite,
        question: str,
        incoming: list[dict[str, Any]],
    ) -> str:
        template = self._load("user.txt")
        incoming_text = _format_incoming(role, mode, incoming)
        return Template(template).substitute(
            question=question.strip(),
            incoming_text=incoming_text,
        )

    def _load(self, name: str) -> str:
        if name not in self._cache:
            path = self.prompt_dir / name
            if not path.exists():
                raise FileNotFoundError(f"prompt template missing: {path}")
            self._cache[name] = path.read_text(encoding="utf-8")
        return self._cache[name]

    def static_system_token_count(
        self,
        role: str,
        mode: CommunicationMode,
        suite: ProtocolSuite,
        counter: Any,
        *,
        fallback: bool = False,
    ) -> int:
        """返回某角色静态 system prompt 的 token 数（含回退模式）。"""
        cache_key = (role, str(mode), suite.version, fallback)
        if cache_key not in self._static_token_counts:
            system = (
                self.fallback_system_prompt(role)
                if fallback
                else self._system_prompt(role, mode, suite)
            )
            self._static_token_counts[cache_key] = counter.count(system)
        return self._static_token_counts[cache_key]

    def fallback_system_prompt(self, role: str) -> str:
        """回退调用使用的最简 system prompt。"""
        answer_description = (
            "final mathematical answer" if self.task_name == "math500" else "final numeric answer"
        )
        if role == "verifier":
            return (
                "role=verifier\n\nPrevious output failed validation. "
                "Output only a JSON object with 'decision' (ACCEPT or REVISE), "
                "'error_type' (null for ACCEPT, a valid error enum for REVISE), "
                "and 'message' (empty for ACCEPT, actionable feedback for REVISE)."
            )
        if role == "planner":
            return (
                "role=planner\n\nPrevious output failed validation. "
                "Output only a JSON object with a non-empty 'message' string."
            )
        return (
            f"role={role}\n\n"
            "Previous outputs failed validation. Fallback mode: produce only a valid JSON object "
            f"with fields 'message' (concise text) and 'answer' ({answer_description}). "
            "No extra text outside JSON."
        )


def format_card_text(card: ProtocolCard, mode: CommunicationMode) -> str:
    """把协议卡片渲染为可插入 prompt 的文本。"""
    if mode == CommunicationMode.NL:
        return (
            f"Edge: {card.edge}\n"
            "Payload format: concise natural language. "
            "Do not use structured field tags."
        )
    lines = [f"Edge: {card.edge}", f"Version: {card.version}"]
    if card.allowed_fields:
        lines.append(f"Allowed fields: {', '.join(card.allowed_fields)}")
    if card.required_fields:
        lines.append(f"Required fields: {', '.join(card.required_fields)}")
    for key, rule in (card.field_rules or {}).items():
        if key == "canonical_order":
            continue
        if isinstance(rule, dict):
            lines.append(f"  {key}: {rule.get('usage', '')}")
            examples = rule.get("examples", [])
            if examples:
                lines.append(f"    Value examples: {', '.join(map(str, examples))}")
        else:
            lines.append(f"  {key}: {rule}")
    if card.message_format:
        lines.append("Message format:")
        for key, rule in card.message_format.items():
            lines.append(f"  {key}: {rule}")
    if card.value_policy:
        lines.append("Value policy:")
        for key, rule in card.value_policy.items():
            lines.append(f"  {key}: {rule}")
    return "\n".join(lines)


def _mode_rules(mode: CommunicationMode, card: ProtocolCard | None = None) -> str:
    if mode == CommunicationMode.NL:
        return "Use concise natural language. Do not use structured field tags."
    return (
        "Use structured fields only. Structure is fixed; representation inside fields is "
        "flexible. Field values may use concise natural language, symbols, LaTeX, or a "
        "mixture of them. Prefer the shortest unambiguous representation. Do not add a "
        "separate natural-language channel."
    )


def _output_schema(
    role: str, mode: CommunicationMode, card: ProtocolCard | None, task_name: str = "gsm8k"
) -> str:
    if mode == CommunicationMode.HYBRID:
        if card and card.output_schema:
            return card.output_schema
        schema_map = {
            "planner": hybrid.planner_output_schema,
            "solver": hybrid.solver_output_schema,
            "verifier": hybrid.verifier_output_schema,
        }
    else:
        schema_map = {
            "planner": nl.planner_output_schema,
            "solver": nl.solver_output_schema,
            "verifier": nl.verifier_output_schema,
        }
    value = schema_map[role]()
    if mode == CommunicationMode.NL and role == "solver" and task_name == "math500":
        return value.replace("最终数值答案", "最终精确数学答案")
    return value


def _examples(role: str, mode: CommunicationMode, card: ProtocolCard | None) -> str:
    """返回适合当前角色和模式的合法输出示例；优先使用协议卡片自带示例。"""
    if mode == CommunicationMode.HYBRID and card and card.examples:
        return "\n".join(str(e) for e in card.examples)
    if mode == CommunicationMode.HYBRID:
        examples = {
            "planner": "K:eggs=16,eat=3,bake=4;G:daily_revenue;P:remain=eggs-eat-bake",
            "solver": "IR:remaining=50-18=32;R:32;UNIT:meters",
            "verifier": "CHK:ACCEPT\nCHK:REVISE;E:ARITH;REQ:recompute 50-18",
        }
    else:
        examples = {
            "planner": '{"message": "Known: eggs=16, used=7. Plan: remaining=9, revenue=18."}',
            "solver": '{"message": "50 - 18 = 32.", "answer": "32"}',
            "verifier": (
                'ACCEPT: {"decision": "ACCEPT", "error_type": null, "message": ""}\n'
                'REVISE: {"decision": "REVISE", "error_type": "ARITH", '
                '"message": "Recheck the subtraction."}'
            ),
        }
    return examples[role]


def _format_incoming(
    role: str, mode: CommunicationMode, incoming: list[dict[str, Any]]
) -> str:
    if not incoming:
        return "No incoming message."
    parts: list[str] = []
    for msg in incoming:
        edge = msg.get("edge", "unknown")
        payload = msg.get("payload", "")
        parts.append(f"Incoming message ({edge}):\n{payload}")
    return "\n\n".join(parts)


def _default_prompt_dir() -> Path:
    """默认 prompt 目录：相对于本文件上溯到仓库根后的 configs/hybrid_comm/prompts。"""
    here = Path(__file__).resolve().parent
    repo_root = here.parents[3]  # src/lychee_mas/hybrid_comm/agents -> repo root
    return repo_root / "configs" / "hybrid_comm" / "prompts"
