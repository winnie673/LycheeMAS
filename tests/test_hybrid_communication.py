"""序列化、反序列化与校验测试。"""

from __future__ import annotations

import pytest
from lychee_mas.hybrid_comm.communication import (
    PayloadParser,
    PayloadSerializer,
)
from lychee_mas.hybrid_comm.protocol import load_protocol_from_yaml
from lychee_mas.hybrid_comm.types import CommunicationMode


@pytest.fixture
def suite():
    return load_protocol_from_yaml("configs/hybrid_comm/protocols/protocol_p0_fields_only.yaml")


def test_hybrid_round_trip(suite) -> None:
    card = suite.card_for("planner_to_solver")
    fields = {"G": "daily_revenue", "P": "used=3+4,remain=16-used"}
    payload = PayloadSerializer.serialize_hybrid(fields, card)
    recovered = PayloadSerializer.deserialize_hybrid(payload, card)
    assert recovered == fields


def test_escape_round_trip(suite) -> None:
    card = suite.card_for("planner_to_solver")
    fields = {"G": "a>b\\c\nd", "P": "x<y"}
    payload = PayloadSerializer.serialize_hybrid(fields, card)
    recovered = PayloadSerializer.deserialize_hybrid(payload, card)
    assert recovered == fields


def test_hybrid_parser_accepts_valid_solver(suite) -> None:
    raw = "IR:step1;R:18"
    parser = PayloadParser(CommunicationMode.HYBRID)
    card = suite.card_for("solver_to_verifier")
    parsed = parser.parse(raw, "solver_to_verifier", card, role="solver")
    assert parsed.final_answer == "18"
    assert parsed.payload_format == "fields_only"
    assert "IR:step1;R:18" in parsed.payload


def test_hybrid_parser_rejects_missing_required(suite) -> None:
    raw = "IR:step1"
    parser = PayloadParser(CommunicationMode.HYBRID)
    card = suite.card_for("solver_to_verifier")
    with pytest.raises(ValueError):
        parser.parse(raw, "solver_to_verifier", card, role="solver")


def test_hybrid_parser_rejects_json_envelope(suite) -> None:
    raw = '{"fields": {"R": "18"}}'
    parser = PayloadParser(CommunicationMode.HYBRID)
    card = suite.card_for("solver_to_verifier")
    with pytest.raises(ValueError):
        parser.parse(raw, "solver_to_verifier", card, role="solver")


def test_nl_parser_solver() -> None:
    raw = '{"message": "The answer is 18.", "answer": "18"}'
    parser = PayloadParser(CommunicationMode.NL)
    parsed = parser.parse(raw, "solver_to_verifier", None, role="solver")
    assert parsed.final_answer == "18"
    assert parsed.payload == "The answer is 18."


def test_nl_prompt_does_not_use_hybrid_card_examples() -> None:
    from lychee_mas.hybrid_comm.agents.roles import RolePromptBuilder

    suite = load_protocol_from_yaml("configs/hybrid_comm/protocols/protocol_p0_fields_only.yaml")
    system = RolePromptBuilder()._system_prompt("planner", CommunicationMode.NL, suite)
    assert '"message": "Known: eggs=16' in system
    assert '"fields": {"K": "eggs=16' not in system


def test_nl_prompt_ignores_fields_only_card_schema() -> None:
    from lychee_mas.hybrid_comm.agents.roles import RolePromptBuilder

    suite = load_protocol_from_yaml(
        "configs/hybrid_comm/protocols/protocol_math500_fields_only.yaml"
    )
    system = RolePromptBuilder()._system_prompt("planner", CommunicationMode.NL, suite)
    assert 'Output JSON: {"message"' in system
    assert "Output only K:facts" not in system


def test_prompt_formats_do_not_conflict_between_nl_and_hybrid(suite) -> None:
    from lychee_mas.hybrid_comm.agents.roles import RolePromptBuilder

    builder = RolePromptBuilder(task_name="math500")
    nl = builder._system_prompt("verifier", CommunicationMode.NL, suite)
    hybrid = builder._system_prompt("verifier", CommunicationMode.HYBRID, suite)
    assert 'Output JSON: {"decision"' in nl
    assert "otherwise output CHK:ACCEPT" not in nl
    assert "output valid JSON" not in hybrid
    assert "CHK:ACCEPT" in hybrid
    fallback = builder.fallback_system_prompt("verifier")
    assert "'decision'" in fallback
    assert "'answer'" not in fallback


def test_nl_verifier_uses_message_as_revision_feedback(suite) -> None:
    raw = (
        '{"decision": "REVISE", "error_type": "ARITH", '
        '"message": "Recheck the subtraction."}'
    )
    parser = PayloadParser(CommunicationMode.NL)
    parsed = parser.parse(
        raw, "verifier_to_solver", suite.verifier_to_solver, role="verifier"
    )
    assert parsed.decision == "REVISE"
    assert parsed.control["feedback"] == "Recheck the subtraction."


def test_fields_only_protocol_rejects_separate_nl() -> None:
    suite = load_protocol_from_yaml(
        "configs/hybrid_comm/protocols/protocol_p0_fields_only.yaml"
    )
    parser = PayloadParser(CommunicationMode.HYBRID)
    raw = "G:answer;P:x=1;NL:extra explanation"
    with pytest.raises(ValueError, match="unknown field"):
        parser.parse(raw, "planner_to_solver", suite.planner_to_solver, role="planner")


def test_fields_only_prompt_allows_mixed_field_values() -> None:
    from lychee_mas.hybrid_comm.agents.roles import RolePromptBuilder

    suite = load_protocol_from_yaml(
        "configs/hybrid_comm/protocols/protocol_p0_fields_only.yaml"
    )
    system = RolePromptBuilder()._system_prompt(
        "planner", CommunicationMode.HYBRID, suite
    )
    assert "Use structured fields only" in system
    assert "symbols, LaTeX, or a mixture" in system


def test_fields_only_parser_converts_wire_payload_to_fields() -> None:
    suite = load_protocol_from_yaml(
        "configs/hybrid_comm/protocols/protocol_p0_fields_only.yaml"
    )
    parser = PayloadParser(CommunicationMode.HYBRID)
    raw = "K:x=3,y=4;G:z;P:z=x+y"
    parsed = parser.parse(raw, "planner_to_solver", suite.planner_to_solver, role="planner")
    assert parsed.payload == raw
    assert parsed.payload_format == "fields_only"
    assert parsed.control["fields"] == {"K": "x=3,y=4", "G": "z", "P": "z=x+y"}


def test_fields_only_parser_rejects_duplicate_field() -> None:
    suite = load_protocol_from_yaml(
        "configs/hybrid_comm/protocols/protocol_p0_fields_only.yaml"
    )
    parser = PayloadParser(CommunicationMode.HYBRID)
    with pytest.raises(ValueError, match="duplicate field"):
        parser.parse(
            "G:x;G:y;P:z=1",
            "planner_to_solver",
            suite.planner_to_solver,
            role="planner",
        )


def test_verifier_feedback_validation(suite) -> None:
    raw = "CHK:FAIL;E:ARITH;REQ:recheck addition"
    parser = PayloadParser(CommunicationMode.HYBRID)
    card = suite.card_for("verifier_to_solver")
    parsed = parser.parse(raw, "verifier_to_solver", card, role="verifier")
    assert parsed.decision == "REVISE"


def test_verifier_accept_uses_chk_field(suite) -> None:
    parser = PayloadParser(CommunicationMode.HYBRID)
    card = suite.card_for("verifier_to_solver")
    parsed = parser.parse("CHK:ACCEPT", "verifier_to_solver", card, role="verifier")
    assert parsed.decision == "ACCEPT"
    assert parsed.control["fields"] == {"CHK": "ACCEPT"}


def test_verifier_rejects_bare_accept(suite) -> None:
    parser = PayloadParser(CommunicationMode.HYBRID)
    card = suite.card_for("verifier_to_solver")
    with pytest.raises(ValueError):
        parser.parse("ACCEPT", "verifier_to_solver", card, role="verifier")
