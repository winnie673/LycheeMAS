"""GSM8K experiment specification; core memory code must never import this module."""

from __future__ import annotations

from lychee_mas.methods.memory.protocol.base import (
    ConditionalConstraint,
    EdgeValidation,
    EvolutionSchedule,
    FieldConstraint,
    ProtocolArchitecture,
    ProtocolContract,
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

PSV_V1 = ProtocolArchitecture(
    name="psv_v1",
    roles=tuple(ROLE_BEHAVIOR.items()),
    edges=(
        ("planner_to_solver", ("K", "G", "P", "NL"), ("G",)),
        ("solver_to_verifier", ("IR", "R", "NL"), ("R",)),
        ("verifier_to_solver", ("CHK", "E", "REQ"), ("CHK",)),
    ),
)
FIELD_MEANINGS = {
    "K": "Selected problem givens, not computed results.",
    "G": "Exact target quantity and its intended unit.",
    "P": "Plan and relations, not a completed final answer.",
    "IR": "Decisive intermediate evidence supporting the answer.",
    "R": "A bare final numeric answer, no unit or reasoning.",
    "NL": (
        "Optional short natural-language clarification of answer-changing meaning; omit if unused."
    ),
    "CHK": "ACCEPT or REVISE; ACCEPT sends CHK only.",
    "E": "Error category on REVISE.",
    "REQ": "Actionable revision request on REVISE, not a replacement answer.",
}
ERROR_TYPES = (
    "ARITH",
    "RELATION",
    "MISSING_FACT",
    "MISSING_UNIT",
    "WRONG_TARGET",
    "INSUFFICIENT_INFO",
    "OTHER",
)
NL_GUARD = FieldConstraint(
    "NL",
    forbidden_values=("none", "n/a", "no ambiguity", "unknown", "not applicable"),
    error="omit unused NL instead of sending a placeholder",
)
GSM8K_PROTOCOL = ProtocolContract(
    task="gsm8k",
    architecture=PSV_V1,
    field_meanings=tuple(FIELD_MEANINGS.items()),
    reserved_fields=("C", "U"),
    fixed_card_options=(("allow_nl_field", True),),
    edge_validation=(
        EdgeValidation("planner_to_solver", fields=(NL_GUARD,)),
        EdgeValidation(
            "solver_to_verifier",
            fields=(NL_GUARD,),
        ),
        EdgeValidation(
            "verifier_to_solver",
            fields=(
                FieldConstraint(
                    "CHK",
                    choices=("ACCEPT", "REVISE", "FAIL", "INSUFFICIENT"),
                    error="verifier CHK must be REVISE, FAIL or INSUFFICIENT, got {value}",
                ),
                FieldConstraint(
                    "E",
                    choices=ERROR_TYPES,
                    error="verifier E must be a valid ErrorType, got {value}",
                ),
            ),
            conditions=(
                ConditionalConstraint(
                    "CHK",
                    "ACCEPT",
                    only=("CHK",),
                    error="verifier ACCEPT must be CHK:ACCEPT without E or REQ",
                ),
                ConditionalConstraint(
                    "CHK",
                    "REVISE",
                    required=("E", "REQ"),
                    error="edge=verifier_to_solver: missing required fields {missing}",
                ),
                ConditionalConstraint(
                    "CHK",
                    "FAIL",
                    required=("E", "REQ"),
                    error="edge=verifier_to_solver: missing required fields {missing}",
                ),
                ConditionalConstraint(
                    "CHK",
                    "INSUFFICIENT",
                    required=("E", "REQ"),
                    error="edge=verifier_to_solver: missing required fields {missing}",
                ),
            ),
            instruction_lines=(f"Error categories for E: {', '.join(sorted(ERROR_TYPES))}",),
        ),
    ),
    generation_guidance=(
        "Use optional NL to preserve answer-changing meaning unsafe to symbolize; omit unused NL. "
        "R is numeric only. CHK:ACCEPT alone; REVISE requires E and REQ. "
        "Do not add legacy C/U or rename fields. "
        "Include optional NL examples for planner and solver."
    ),
)
GSM8K_SCHEDULE = EvolutionSchedule("D0", ("G1", "G2", "G3"), "V", heldout=("T",))


def verification_decision(value: str) -> str:
    """Keep the raw wire alias, map only the control decision like the source parser."""
    return "ACCEPT" if value == "ACCEPT" else "REVISE"
