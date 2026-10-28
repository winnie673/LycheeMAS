"""GSM8K experiment specification; core memory code must never import this module."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from lychee_mas.methods.memory.protocol.base import (
    EvolutionSchedule,
    ProtocolContract,
    assert_runtime_compatible,
    build_contract_from_schema,
    load_protocol_schema,
)

ROLE_BEHAVIOR = {
    "planner": (
        "You are part of a multi-agent math problem solving system. Each agent "
        "has a distinct role and sees only the information it is allowed to "
        "see. Output only the format specified below, without surrounding text. "
        "Do not reveal or reference the gold answer or standard solution.\n\nYour "
        "role: Planner.\nYou read the original problem and produce a concise "
        "message to the Solver. You must not give the final numeric answer. "
        "Include only the facts, goal, plan, and constraints that the Solver "
        "needs."
    ),
    "solver": (
        "You are part of a multi-agent math problem solving system. Each agent "
        "has a distinct role and sees only the information it is allowed to "
        "see. Output only the format specified below, without surrounding text. "
        "Do not reveal or reference the gold answer or standard solution.\n\nYour "
        "role: Solver.\nYou read the original problem and the Planner's message, "
        "then solve the problem. Output a candidate final answer and the "
        "information required by the protocol card."
    ),
    "verifier": (
        "You are part of a multi-agent math problem solving system. Each agent "
        "has a distinct role and sees only the information it is allowed to "
        "see. Output only the format specified below, without surrounding text. "
        "Do not reveal or reference the gold answer or standard solution.\n\nYour "
        "role: Verifier.\nInspect the original task and the Solver's message. Do "
        "not access the gold answer.\n\nVerify in this order before deciding:\n1. "
        "Identify the exact target and any domain, branch, unit, or format "
        "constraints in the task. Check that the candidate answer addresses "
        "that target.\n2. Independently check the key relations and calculations "
        "in the Solver's explanation against the task. Recompute decisive steps "
        "when possible; do not assume that the explanation or your own first "
        "calculation is correct.\n3. Check that the available evidence supports "
        "the candidate answer and that relevant conditions or cases are "
        "respected.\n\nRequest revision only in one of two situations:\n- A "
        "specific, checkable contradiction is established in the explanation, "
        "candidate answer, or their relation to the task. Identify the exact "
        "expression, value, missing case, or violated condition and request a "
        "targeted correction. Verify the disputed step locally against the task "
        "or an explicit relation in the Solver's message before asserting an "
        "error. An alternative answer from your own calculation alone is not "
        "sufficient.\n- A specific essential step is absent and you cannot "
        "verify the candidate answer from the task and available message. "
        "Select the insufficient-information error category and request only "
        "the missing intermediate evidence; do not assert that the answer is "
        "wrong or request a different answer.\n\nIf an alleged defect requires "
        "solving a substantial new subproblem rather than checking an explicit "
        "relation or calculation, do not propose a replacement answer. Request "
        "only the particular missing derivation when it is essential for "
        "verification. Your inability to independently solve the entire task is "
        "not evidence that the Solver is wrong. Before asking for a different "
        "answer, verify why the current answer cannot satisfy the task; "
        "otherwise do not assert an answer error.\n\nIf the answer is supported "
        "and you have no established defect, accept it. Do not request a full "
        "derivation, generic rechecking, or extra detail merely because the "
        "explanation is concise. Follow the mode-specific output format below. "
        "These decision rules take precedence if a card example or field rule "
        "suggests a broader revision trigger."
    ),
}

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "configs/memory/gsm8k_protocol_schema.yaml"
GSM8K_SCHEMA = load_protocol_schema(SCHEMA_PATH)
# Topology/roles are experiment choices; field lists and rules come only from YAML.
PSV_EDGES = ("planner_to_solver", "solver_to_verifier", "verifier_to_solver")
RUNTIME_ALIASES = {"FAIL": "REVISE", "INSUFFICIENT": "REVISE"}


def build_gsm8k_contract(schema: dict) -> ProtocolContract:
    contract = build_contract_from_schema(
        schema,
        task="gsm8k",
        architecture_name="psv_v1",
        roles=tuple(ROLE_BEHAVIOR.items()),
        edges=PSV_EDGES,
        control_fields={"verifier_to_solver": "CHK"},
        runtime_aliases={"verifier_to_solver": RUNTIME_ALIASES},
        task_field_meanings={"R": "A bare final numeric answer, no unit or reasoning."},
        reserved_fields=("C", "U"),
        fixed_card_options=(("allow_nl_field", schema["allow_nl_field"]),),
        generation_guidance=(
            "Follow the frozen schema, preserve answer-changing meaning and omit unused fields. "
            "For this task R is numeric only. Include optional NL examples for planner and solver."
        ),
    )
    # Error wording and user-facing instructions are presentation, not duplicated predicates.
    errors = {
        "NL": "omit unused NL instead of sending a placeholder",
        "CHK": "verifier CHK must be REVISE, FAIL or INSUFFICIENT, got {value}",
        "E": "verifier E must be a valid ErrorType, got {value}",
    }
    enum = schema["verifier_to_solver"]["value_constraints"]["E"]["allowed_values"]
    return replace(
        contract,
        edge_validation=tuple(
            replace(
                edge,
                fields=tuple(replace(rule, error=errors.get(rule.name)) for rule in edge.fields),
                conditions=tuple(
                    replace(
                        condition,
                        error="verifier ACCEPT must be CHK:ACCEPT without E or REQ"
                        if condition.equals == "ACCEPT"
                        else "edge=verifier_to_solver: missing required fields {missing}",
                    )
                    for condition in edge.conditions
                ),
                instruction_lines=(f"Error categories for E: {', '.join(sorted(enum))}",)
                if edge.edge == "verifier_to_solver"
                else (),
            )
            for edge in contract.edge_validation
        ),
    )


GSM8K_PROTOCOL = build_gsm8k_contract(GSM8K_SCHEMA)
PSV_V1 = GSM8K_PROTOCOL.architecture
FIELD_MEANINGS = dict(GSM8K_PROTOCOL.field_meanings)
ERROR_TYPES = tuple(GSM8K_SCHEMA["verifier_to_solver"]["value_constraints"]["E"]["allowed_values"])

# Explicitly recognized frozen lineage, not an unrestricted fingerprint bypass.
HISTORICAL_CONTRACT_SHA256 = frozenset(
    {
        "ac09c03455d8af3a35273b605bb457591de11f0e22bded3325bbb626ee774caf",
    }
)


def protocol_contract_for_card(path: str | Path) -> ProtocolContract:
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("task card must be a mapping")
    if "validation_contract" not in raw:
        return GSM8K_PROTOCOL  # Legacy cards still require explicit injection by the caller.
    embedded = ProtocolContract.from_dict(raw["validation_contract"])
    embedded.check_card(raw, require_identity=True)
    if embedded.fingerprint != GSM8K_PROTOCOL.fingerprint:
        if embedded.fingerprint not in HISTORICAL_CONTRACT_SHA256:
            raise ValueError("unrecognized historical contract fingerprint")
        assert_runtime_compatible(GSM8K_PROTOCOL, embedded)
    return embedded


GSM8K_SCHEDULE = EvolutionSchedule("D0", ("G1", "G2", "G3"), "V", heldout=("T",))


def verification_decision(value: str) -> str:
    """Keep the raw wire alias, map only the control decision like the source parser."""
    canonical = RUNTIME_ALIASES.get(value, value)
    allowed = GSM8K_SCHEMA["verifier_to_solver"]["value_constraints"]["CHK"]["allowed_values"]
    if canonical not in allowed:
        raise ValueError(f"unknown verification decision: {value}")
    return canonical
