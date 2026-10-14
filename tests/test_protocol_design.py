from __future__ import annotations

import pytest
from lychee_mas.hybrid_comm.protocol import (
    apply_evolution_changes,
    classify_feedback,
    load_protocol_from_yaml,
    load_schema,
    load_task_profile,
    materialize_card,
    write_card,
)
from lychee_mas.hybrid_comm.protocol.design import (
    card_review_prompt,
    initialization_prompt,
)
from lychee_mas.hybrid_comm.protocol.loader import _load_yaml


def _generated(schema):
    edges = {"planner_to_solver", "solver_to_verifier", "verifier_to_solver"}
    return {
        "cards": {
            edge: {
                "field_rules": {
                    field: {
                        "usage": f"Rule for {field}",
                        "examples": (
                            ["ACCEPT"]
                            if field == "CHK"
                            else (
                                ["ARITH"]
                                if field == "E"
                                else (["1"] if field == "R" else [f"{field.lower()}=1"])
                            )
                        ),
                    }
                    for field in spec["allowed_fields"]
                },
                "examples": (
                    [
                        "CHK:ACCEPT",
                        "CHK:REVISE;E:ARITH;REQ:recompute x",
                        "CHK:REVISE;E:RELATION;REQ:use correct relation",
                    ]
                    if edge == "verifier_to_solver"
                    else (
                        [
                            "G:x;P:x=1",
                            "K:a=1;G:x;P:x=a+1",
                            "K:a=1;G:x;P:x=a;C:x must be positive",
                        ]
                        if edge == "planner_to_solver"
                        else [
                            "IR:x=1;R:1",
                            "IR:x=1;R:1;C:x must be positive",
                            "IR:x=1;R:1;U:relation is ambiguous",
                        ]
                    )
                ),
                "value_policy": {},
            }
            for edge, spec in schema.items()
            if edge in edges
        },
        "value_policy": {"preference": "shortest unambiguous value"},
    }


def test_materialized_card_preserves_schema(tmp_path):
    schema = load_schema("configs/hybrid_comm/protocol_schema.yaml")
    assert schema["schema_version"] == "SCHEMA-3"
    assert "value_constraints" not in schema["solver_to_verifier"]
    assert "task-specific representation" in schema["solver_to_verifier"][
        "field_semantics"
    ]["R"]
    card = materialize_card(schema, _generated(schema), version="CARD-0-GSM8K")
    path = tmp_path / "card.yaml"
    write_card(path, card)
    suite = load_protocol_from_yaml(str(path))
    assert suite.version == "CARD-0-GSM8K"
    assert suite.planner_to_solver.allowed_fields == ["K", "G", "P", "C", "U"]
    assert "Example:" in suite.planner_to_solver.field_rules["K"]
    assert suite.value_policy == {"preference": "shortest unambiguous value"}


def test_gsm8k_answer_format_is_kept_in_task_card():
    schema = load_schema("configs/hybrid_comm/protocol_schema.yaml")
    card = _load_yaml(
        "configs/hybrid_comm/generated/gsm8k/final/card_gsm8k_evolved.yaml"
    )
    assert "finite decimal without a unit" in card["solver_to_verifier"][
        "field_rules"
    ]["R"]
    assert "finite decimal" not in schema["solver_to_verifier"]["field_semantics"]["R"]


def test_math500_task_profile_keeps_exact_symbolic_answers():
    profile = load_task_profile("configs/hybrid_comm/tasks/math500.yaml")
    assert profile["task"] == "math500"
    assert profile["scorer_kind"] == "math500"
    assert "radical" in profile["answer_policy"]["allowed_forms"]
    assert any(
        "Do not force a decimal" in item
        for item in profile["answer_policy"]["requirements"]
    )


def test_materialized_description_uses_task_profile():
    schema = load_schema("configs/hybrid_comm/protocol_schema.yaml")
    profile = load_task_profile("configs/hybrid_comm/tasks/math500.yaml")
    card = materialize_card(
        schema,
        _generated(schema),
        version="CARD-0-MATH500",
        task_profile=profile,
    )
    assert "MATH-500" in card["description"]
    assert card["value_policy"] == {"preference": "shortest unambiguous value"}


def test_math500_prompts_are_task_specific_without_gsm8k_constraints():
    schema = load_schema("configs/hybrid_comm/protocol_schema.yaml")
    profile = load_task_profile("configs/hybrid_comm/tasks/math500.yaml")
    generated = _generated(schema)
    card = materialize_card(
        schema,
        generated,
        version="CARD-0-MATH500",
        task_profile=profile,
    )
    init_prompt = initialization_prompt(schema, [], profile)
    review_prompt = card_review_prompt(schema, card, profile)
    assert "MATH-500" in init_prompt
    assert "radical" in init_prompt
    assert '`"examples":["x=1"]`' in init_prompt
    assert 'return a scalar string' in init_prompt
    assert "MATH-500" in review_prompt
    assert "final numeric answer without a unit" not in init_prompt


def test_card_rejects_missing_frozen_field():
    schema = load_schema("configs/hybrid_comm/protocol_schema.yaml")
    generated = _generated(schema)
    del generated["cards"]["planner_to_solver"]["field_rules"]["U"]
    with pytest.raises(ValueError, match="exactly the frozen fields"):
        materialize_card(schema, generated, version="CARD-0-GSM8K")


def test_feedback_is_paired_and_classified():
    nl = [{"task_id": "a", "is_correct": True, "communication_tokens": 20}]
    hybrid = [{"task_id": "a", "is_correct": False, "communication_tokens": 10}]
    groups = classify_feedback(nl, hybrid)
    assert groups["hybrid_incorrect_nl_correct"][0]["nl"]["task_id"] == "a"


def test_failed_hybrid_case_has_separate_feedback_category():
    nl = [{"task_id": "a", "is_correct": True, "run_status": "completed"}]
    hybrid = [
        {
            "task_id": "a",
            "is_correct": False,
            "run_status": "failed",
            "communication_tokens": 0,
        }
    ]
    groups = classify_feedback(nl, hybrid)
    assert groups["run_failed"][0]["hybrid"]["task_id"] == "a"


def test_evolution_applies_only_evidence_bound_rule_patch():
    schema = load_schema("configs/hybrid_comm/protocol_schema.yaml")
    current = _load_yaml(
        "configs/hybrid_comm/generated/gsm8k/initial/card_0_gsm8k_v6.yaml"
    )
    decision = {
        "action": "UPDATE",
        "changes": [
            {
                "path": "planner_to_solver.field_rules.P",
                "value": "Preserve multiplicity explicitly. Example: total=n*each",
                "diagnosis_case_ids": ["case-1"],
            }
        ],
    }
    candidate, paths = apply_evolution_changes(
        schema,
        current,
        decision,
        version="CARD-1-GSM8K",
        diagnosed_case_ids={"case-1"},
    )
    assert paths == ["planner_to_solver.field_rules.P"]
    assert candidate["planner_to_solver"]["field_rules"]["K"] == current[
        "planner_to_solver"
    ]["field_rules"]["K"]
    assert candidate["version"] == "CARD-1-GSM8K"


def test_evolution_rejects_unknown_diagnosis_case():
    schema = load_schema("configs/hybrid_comm/protocol_schema.yaml")
    current = _load_yaml(
        "configs/hybrid_comm/generated/gsm8k/initial/card_0_gsm8k_v6.yaml"
    )
    decision = {
        "action": "UPDATE",
        "changes": [
            {
                "path": "planner_to_solver.field_rules.P",
                "value": "Rule. Example: x=1",
                "diagnosis_case_ids": ["invented"],
            }
        ],
    }
    with pytest.raises(ValueError, match="unknown diagnosis"):
        apply_evolution_changes(
            schema,
            current,
            decision,
            version="CARD-1-GSM8K",
            diagnosed_case_ids={"case-1"},
        )
