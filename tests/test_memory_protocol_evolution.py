"""Offline induction/selection and online compatibility; no network calls."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from lychee_mas.methods.memory.channels.hybrid_protocol import HybridProtocolMemory
from lychee_mas.methods.memory.protocol import (
    ConditionalConstraint,
    EdgeValidation,
    EvolutionSchedule,
    FieldConstraint,
    ProtocolArchitecture,
    ProtocolContract,
    ProtocolEvolution,
    assert_runtime_compatible,
    build_contract_from_schema,
)
from scripts.hybrid_gsm8k_spec import (
    GSM8K_PROTOCOL,
    GSM8K_SCHEDULE,
    GSM8K_SCHEMA,
    PSV_V1,
    build_gsm8k_contract,
    protocol_contract_for_card,
)


@pytest.fixture
def entry(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    return importlib.import_module("run_memory_protocol_evolution")


def splits():
    return {
        name: [{"task_id": name, "question": f"{name}: Add nine and nine.", "gold": "18"}]
        for name in ("D0", "G1", "G2", "G3", "V")
    }


def numeric(pred, gold):
    return float(pred == gold)


def fixture_card(entry):
    result = entry.fake_proposal([{}, {"content": json.dumps({"parent_card": None})}], 8192)
    card = json.loads(result.text)["card"]
    card["task_card"] = {
        "task": "gsm8k",
        "roles": list(dict(PSV_V1.roles)),
        "edges": [edge for edge, _, _ in PSV_V1.edges],
        "architecture": PSV_V1.name,
        "architecture_sha256": PSV_V1.fingerprint,
    }
    return card


def write_card(path, card):
    import yaml

    path.write_text(yaml.safe_dump(card), encoding="utf-8")
    return path


def test_schema_is_shared_by_design_and_runtime_without_receive_alias_leakage():
    from scripts.hybrid_gsm8k_design import GSM8KDesign

    design = GSM8KDesign()
    assert design.schema == GSM8K_SCHEMA
    design.schema["description"] = "local copy"
    assert GSM8K_SCHEMA["description"] != "local copy"
    for edge, allowed, required in PSV_V1.edges:
        assert allowed == tuple(GSM8K_SCHEMA[edge]["allowed_fields"])
        assert required == tuple(GSM8K_SCHEMA[edge]["required_fields"])
    enum = GSM8K_SCHEMA["verifier_to_solver"]["value_constraints"]["CHK"]["allowed_values"]
    assert enum == ["ACCEPT", "REVISE"]
    check = GSM8K_PROTOCOL.validation_for("verifier_to_solver").fields[0]
    assert set(check.choices) == {*enum, "FAIL", "INSUFFICIENT"}


def test_schema_changes_drive_enum_guards_meanings_and_required_fields():
    from scripts.hybrid_gsm8k_design import GSM8KDesign

    schema = copy.deepcopy(GSM8K_SCHEMA)
    schema["verifier_to_solver"]["value_constraints"]["E"]["allowed_values"].append("NEW_ERROR")
    schema["planner_to_solver"]["required_fields"].append("P")
    schema["planner_to_solver"]["optional_field_policy"]["NL"][
        "forbidden_placeholder_values"
    ].append("unused")
    schema["planner_to_solver"]["field_semantics"]["NL"] = "Planning-only clarification"
    before = copy.deepcopy(schema)
    contract = build_gsm8k_contract(schema)
    assert "P" in contract.architecture.edges[0][2]
    edge = contract.validation_for("planner_to_solver")
    with pytest.raises(ValueError):
        next(r for r in edge.fields if r.name == "NL").check({"NL": "unused"})
    errors = contract.validation_for("verifier_to_solver").fields[1]
    assert "NEW_ERROR" in errors.choices
    prompt = GSM8KDesign(schema=schema).initialize([], "test executor")
    assert "NEW_ERROR" in prompt.split("The only valid E values are ", 1)[1].split(".\n", 1)[0]
    meaning = dict(contract.field_meanings)["NL"]
    assert "planner_to_solver: Planning-only clarification" in meaning
    assert GSM8K_SCHEMA["solver_to_verifier"]["field_semantics"]["NL"] in meaning
    assert schema == before  # Alias expansion/compilation never changes design data.


def test_schema_forbidden_is_not_an_exclusive_field_set_and_roundtrips():
    schema = {
        "message_format": {
            "structure": "fields_only",
            "wire_syntax": "direct",
            "field_separator": ";",
        },
        "author_to_editor": {
            "allowed_fields": ["STATE", "TEXT", "NOTE"],
            "required_fields": ["STATE"],
            "field_semantics": {"STATE": "Status", "TEXT": "A color", "NOTE": "Explanation"},
            "value_constraints": {"STATE": {"allowed_values": ["PASS", "EDIT"]}},
            "optional_field_policy": {"NOTE": {"forbidden_placeholder_values": ["none"]}},
            "conditional_required": {"EDIT": ["TEXT"]},
            "conditional_forbidden": {"PASS": ["NOTE"]},
        },
    }
    contract = build_contract_from_schema(
        schema,
        task="toy_colors",
        architecture_name="pair",
        roles=(("author", "Write a color"), ("editor", "Check the color")),
        edges=("author_to_editor",),
        control_fields={"author_to_editor": "STATE"},
        runtime_aliases={"author_to_editor": {"RETRY": "EDIT"}},
    )
    restored = ProtocolContract.from_dict(json.loads(json.dumps(contract.as_dict())))
    assert restored.fingerprint == contract.fingerprint
    rules = restored.validation_for("author_to_editor")
    for condition in rules.conditions:
        condition.check({"STATE": "PASS", "TEXT": "blue"})  # TEXT is not forbidden.
    with pytest.raises(ValueError):
        for condition in rules.conditions:
            condition.check({"STATE": "PASS", "NOTE": "explanation"})
    with pytest.raises(ValueError):
        for condition in rules.conditions:
            condition.check({"STATE": "RETRY"})


@pytest.mark.parametrize("change", ["unknown_field", "bad_enum", "unknown_operation", "conflict"])
def test_schema_conversion_rejects_unsupported_or_inconsistent_rules(change):
    schema = copy.deepcopy(GSM8K_SCHEMA)
    edge = schema["verifier_to_solver"]
    if change == "unknown_field":
        edge["conditional_required"]["REVISE"].append("UNKNOWN")
    elif change == "bad_enum":
        edge["value_constraints"]["E"]["allowed_values"] = []
    elif change == "unknown_operation":
        edge["value_constraints"]["E"]["execute_python"] = "not allowed"
    else:
        edge["conditional_forbidden"]["REVISE"] = ["E"]
    with pytest.raises(ValueError):
        build_gsm8k_contract(schema)


def synthetic_legacy_contract():
    """Old serialization shape, without depending on a removed historical card."""
    allowed = {edge: fields for edge, fields, _ in GSM8K_PROTOCOL.architecture.edges}
    return replace(
        GSM8K_PROTOCOL,
        edge_validation=tuple(
            replace(
                rule,
                conditions=tuple(
                    replace(
                        condition,
                        forbidden=(),
                        only=tuple(
                            field
                            for field in allowed[rule.edge]
                            if field not in condition.forbidden
                        ),
                    )
                    if condition.forbidden
                    else condition
                    for condition in rule.conditions
                ),
            )
            for rule in GSM8K_PROTOCOL.edge_validation
        ),
    )


def synthetic_legacy_card(entry):
    raw = fixture_card(entry)
    raw.pop("validation_contract", None)
    raw["task_card"] = {
        "task": "gsm8k",
        "roles": list(dict(PSV_V1.roles)),
        "edges": [edge for edge, _, _ in PSV_V1.edges],
    }
    return raw


def test_frozen_contract_fingerprint_and_validation_are_preserved():
    path = Path("configs/memory/task_cards/gsm8k_psv_hybrid_card_20261028.yaml")
    before = path.read_bytes()
    frozen = protocol_contract_for_card(path)
    assert frozen.fingerprint == GSM8K_PROTOCOL.fingerprint
    assert_runtime_compatible(GSM8K_PROTOCOL, frozen)
    legacy = synthetic_legacy_contract()
    assert_runtime_compatible(GSM8K_PROTOCOL, legacy)
    assert ProtocolContract.from_dict(legacy.as_dict()).fingerprint == legacy.fingerprint
    assert all(
        "forbidden" not in c
        for rule in legacy.as_dict()["edge_validation"]
        for c in rule["conditions"]
    )
    assert path.read_bytes() == before


def test_frozen_compatibility_rejects_guard_drift_and_unknown_fingerprints(monkeypatch, tmp_path):
    import yaml
    from scripts import hybrid_gsm8k_spec as spec

    source = Path("configs/memory/task_cards/gsm8k_psv_hybrid_card_20261028.yaml")
    legacy = synthetic_legacy_contract()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["validation_contract"] = legacy.as_dict()
    raw["task_card"]["contract_sha256"] = legacy.fingerprint
    path = write_card(tmp_path / "historical.yaml", raw)
    monkeypatch.setattr(spec, "HISTORICAL_CONTRACT_SHA256", frozenset({legacy.fingerprint}))
    assert spec.protocol_contract_for_card(path).fingerprint == legacy.fingerprint
    schema = copy.deepcopy(GSM8K_SCHEMA)
    schema["planner_to_solver"]["optional_field_policy"]["NL"][
        "forbidden_placeholder_values"
    ].append("unused")
    with monkeypatch.context() as patch:
        patch.setattr(spec, "GSM8K_PROTOCOL", build_gsm8k_contract(schema))
        with pytest.raises(ValueError, match="incompatible"):
            spec.protocol_contract_for_card(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["validation_contract"]["generation_guidance"] += " changed"
    changed = ProtocolContract.from_dict(raw["validation_contract"])
    raw["task_card"]["contract_sha256"] = changed.fingerprint
    candidate = write_card(tmp_path / "unrecognized.yaml", raw)
    with pytest.raises(ValueError, match="unrecognized historical"):
        spec.protocol_contract_for_card(candidate)


def test_legacy_frozen_card_compatible_and_no_mutation(entry, tmp_path):
    path = write_card(tmp_path / "legacy.yaml", synthetic_legacy_card(entry))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    channel = HybridProtocolMemory(str(path), "gsm8k", contract=GSM8K_PROTOCOL)
    channel.check_architecture(PSV_V1)
    with pytest.raises(ValueError, match="identity"):
        PSV_V1.check_card(channel._raw, "gsm8k", require_identity=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize(
    "mutation", ["task", "roles", "edges", "identity", "fields", "fingerprint"]
)
def test_incompatible_card_rejected_before_model_call(entry, tmp_path, mutation):
    card = fixture_card(entry)
    if mutation == "task":
        card["task_card"]["task"] = "unsupported_task"
    elif mutation in ("roles", "edges"):
        card["task_card"][mutation] = []
    elif mutation == "identity":
        card["task_card"]["architecture"] = "voting"
    elif mutation == "fingerprint":
        card["task_card"]["architecture_sha256"] = "wrong"
    else:
        card["planner_to_solver"]["allowed_fields"] = ["G"]
    path = write_card(tmp_path / "card.yaml", card)
    calls = []
    with pytest.raises(ValueError):
        entry.executor().run_demo(lambda *a, **kw: calls.append(a), protocol_path=path)
    assert calls == []


def test_explicit_card_reuses_existing_cdm_executor(entry, tmp_path):
    path = write_card(tmp_path / "card.yaml", fixture_card(entry))
    module = entry.executor()
    trajectory = module.run_demo(protocol_path=path)
    assert trajectory.final_answer.content == "18"
    assert len(trajectory.messages) == 3
    assert trajectory.meta["architecture_sha256"] == PSV_V1.fingerprint
    assert trajectory.meta["protocol_path"] == str(path)
    with pytest.raises(ValueError, match="only implements"):
        module.run_demo(architecture="voting")
    with pytest.raises(ValueError, match="GSM8K Hybrid"):
        module.run_demo(mode="nl", protocol_path=path)


def test_full_lineage_promotes_rejects_and_keeps(entry, tmp_path):
    calls, proposals = [], []

    def execute(case, mode, card, log):
        assert set(case) == {"task_id", "question"}  # No gold at execution boundary.
        calls.append((case, mode, card))
        correct = not (
            log.parent.parent.name == "round1" and log.parent.name == "validation_parent"
        )
        correct &= not (
            log.parent.parent.name == "round2" and log.parent.name == "validation_candidate"
        )
        return {
            "status": "completed",
            "prediction": "18" if correct else "17",
            "communication_tokens": 10,
            "model_output_tokens": 12,
            "wall_time_s": 0.01,
        }

    def generate(messages, max_new_tokens):
        request = json.loads(messages[1]["content"])
        proposals.append(request)
        if len(proposals) in (1, 4):
            return entry.fake_proposal(messages, max_new_tokens)
        card = copy.deepcopy(request["parent_card"])
        card["version"] = f"candidate-{len(proposals)}"
        card["planner_to_solver"]["field_rules"]["P"] += " Keep relations explicit."
        return SimpleNamespace(
            text=json.dumps({"decision": "UPDATE", "reason": "G feedback.", "card": card})
        )

    algorithm = ProtocolEvolution(
        generate=generate,
        execute=execute,
        contract=GSM8K_PROTOCOL,
        schedule=GSM8K_SCHEDULE,
        score=numeric,
        executor_capability="Limited small reasoning model",
    )
    result = algorithm.run(splits(), tmp_path / "run")
    assert [r["promoted"] for r in result["history"]] == [True, False, False]
    assert result["heldout_evaluated"] is False and result["effect_valid"] is False
    assert len(calls) == 10  # D0 + three G + six V arms.
    assert [p["parent_card"]["version"] for p in proposals[2:]] == ["candidate-2", "candidate-2"]
    assert proposals[0]["parent_card"] is None
    assert all(p["executor_capability"] == "Limited small reasoning model" for p in proposals)
    assert [p["development_evidence"][0]["task_id"] for p in proposals] == ["D0", "G1", "G2", "G3"]
    assert "V-selected" not in Path(result["selected_card"]).read_text(encoding="utf-8")
    channel = HybridProtocolMemory(result["selected_card"], "gsm8k")
    channel.check_architecture(PSV_V1)
    assert channel.version == "candidate-2"
    assert json.loads((tmp_path / "run/round1/evolution.json").read_text(encoding="utf-8"))[
        "changed_paths"
    ]
    with pytest.raises(FileExistsError):
        algorithm.run(splits(), tmp_path / "run")


@pytest.mark.parametrize(
    "old,new,expected,reason",
    [
        ((0.8, 10), (0.9, 100), True, "accuracy"),
        ((0.9, 100), (0.8, 10), False, "accuracy"),
        ((0.9, 100), (0.9, 90), True, "exact_communication_tokens"),
        ((0.9, 90), (0.9, 100), False, "exact_communication_tokens"),
        ((0.9, None), (0.9, 90), False, "tie_or_unknown_cost_keep_parent"),
        ((0.9, 90), (0.9, 90), False, "tie_or_unknown_cost_keep_parent"),
    ],
)
def test_selection(old, new, expected, reason):
    def metrics(pair):
        return {"accuracy": pair[0], "avg_communication_tokens": pair[1]}

    assert ProtocolEvolution.select(metrics(old), metrics(new)) == (expected, reason)


def test_t_and_overlap_forbidden_before_any_call(entry, tmp_path):
    algorithm = ProtocolEvolution(
        contract=GSM8K_PROTOCOL,
        schedule=GSM8K_SCHEDULE,
        generate=None,
        execute=None,
        score=numeric,
        executor_capability="small",
    )
    data = splits()
    data["T"] = data["V"]
    with pytest.raises(ValueError, match="forbidden"):
        algorithm.run(data, tmp_path / "bad")
    data = splits()
    data["V"][0]["question"] = data["D0"][0]["question"]
    with pytest.raises(ValueError, match="duplicate"):
        algorithm.run(data, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_invalid_proposals_bounded_and_raw_output_saved(entry, tmp_path):
    count = []

    def generate(messages, **kwargs):
        count.append(messages)
        return SimpleNamespace(text='{"decision":"KEEP","reason":"invalid initial KEEP"}')

    algorithm = ProtocolEvolution(
        generate=generate,
        contract=GSM8K_PROTOCOL,
        schedule=GSM8K_SCHEDULE,
        execute=lambda *a: {
            "status": "failed",
            "prediction": "",
            "communication_tokens": None,
        },
        score=numeric,
        executor_capability="small",
    )
    out = tmp_path / "bad"
    with pytest.raises(ValueError, match="exhausted"):
        algorithm.run(splits(), out)
    assert len(count) == 3
    assert "Correct the proposal" in count[1][-1]["content"]
    assert (out / "failure.json").exists() and not (out / "result.json").exists()
    assert "raw_output" in json.loads((out / "card0/proposal_2.json").read_text(encoding="utf-8"))


def test_changed_field_interface_is_not_evolved(entry, tmp_path):
    def generate(messages, **kwargs):
        result = json.loads(entry.fake_proposal(messages, 8192).text)
        result["card"]["planner_to_solver"]["allowed_fields"].append("C")
        return SimpleNamespace(text=json.dumps(result))

    algorithm = ProtocolEvolution(
        generate=generate,
        contract=GSM8K_PROTOCOL,
        schedule=GSM8K_SCHEDULE,
        execute=lambda *a: {"status": "completed", "prediction": "18"},
        score=numeric,
        executor_capability="small",
    )
    with pytest.raises(ValueError, match="exhausted"):
        algorithm.run(splits(), tmp_path / "bad")


def test_script_adapter_and_evolution_full_smoke(entry, tmp_path):
    module = entry.executor()
    seen = []

    def generate(messages, **kwargs):
        seen.append(messages)
        assert "918273" not in json.dumps(messages)
        return module.scripted_generate(messages, **kwargs)

    data = splits()
    data["D0"][0]["gold"] = "918273"
    algorithm = ProtocolEvolution(
        generate=entry.fake_proposal,
        contract=GSM8K_PROTOCOL,
        schedule=GSM8K_SCHEDULE,
        execute=entry.make_execute(module, generate, None, 512, (TimeoutError,)),
        score=lambda pred, gold: entry.score_answer("gsm8k", pred, gold),
        executor_capability="fake",
        sanitize=module.redact,
    )
    result = algorithm.run(data, tmp_path / "run")
    assert len(seen) == 30
    assert all(not row["promoted"] for row in result["history"])
    assert len(list((tmp_path / "run").rglob("*.attempts.jsonl"))) == 10
    # The frozen artifact is selectable by the existing paired evaluator.
    evaluator = importlib.import_module("run_memory_protocol_compare")
    comparison = evaluator.run_evaluation(
        [splits()["V"][0]],
        tmp_path / "compare",
        benchmark="gsm8k",
        generate=module.scripted_generate,
        protocol_path=Path(result["selected_card"]),
    )
    assert comparison["paired"] == {"both_correct": 1}


def test_integrity_failure_never_freezes_card(entry, tmp_path):
    def verify():
        raise RuntimeError("source changed")

    algorithm = ProtocolEvolution(
        generate=entry.fake_proposal,
        contract=GSM8K_PROTOCOL,
        schedule=GSM8K_SCHEDULE,
        execute=lambda *a: {"status": "completed", "prediction": "18"},
        score=numeric,
        executor_capability="fake",
    )
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="source changed"):
        algorithm.run(splits(), out, verify_sources=verify)
    assert not (out / "final").exists() and not (out / "result.json").exists()
    assert (out / "failure.json").exists()


def test_proposer_program_error_not_retried(entry, tmp_path):
    calls = []

    def broken(*args, **kwargs):
        calls.append(1)
        raise TypeError("broken backend contract")

    algorithm = ProtocolEvolution(
        generate=broken,
        contract=GSM8K_PROTOCOL,
        schedule=GSM8K_SCHEDULE,
        execute=lambda *a: {"status": "completed", "prediction": "18"},
        score=numeric,
        executor_capability="fake",
    )
    with pytest.raises(TypeError, match="broken backend"):
        algorithm.run(splits(), tmp_path / "run")
    assert len(calls) == 1
    record = json.loads((tmp_path / "run/card0/proposal_0.json").read_text(encoding="utf-8"))
    assert record["status"] == "fatal_error"


@pytest.mark.parametrize(
    "broken", ["json", "section", "examples", "format", "field_rule", "truncated"]
)
def test_proposal_repair(entry, tmp_path, broken):
    calls = []

    def generate(messages, max_new_tokens):
        request = json.loads(messages[1]["content"])
        calls.append(request)
        result = json.loads(entry.fake_proposal(messages, max_new_tokens).text)
        if len(calls) == 1:
            card = result["card"]
            if broken == "json":
                return SimpleNamespace(text="invalid json")
            if broken == "section":
                card["planner_to_solver"] = []
            elif broken == "examples":
                card["solver_to_verifier"]["examples"] = ["R:18\nIR:missing separator"]
            elif broken == "format":
                card["message_format"] = []
            elif broken == "field_rule":
                card["planner_to_solver"]["field_rules"]["C"] = "Forbidden field"
            elif broken == "truncated":
                return SimpleNamespace(text=json.dumps(result), n_gen_tokens=max_new_tokens)
        return SimpleNamespace(text=json.dumps(result))

    algorithm = ProtocolEvolution(
        generate=generate,
        contract=GSM8K_PROTOCOL,
        schedule=GSM8K_SCHEDULE,
        execute=lambda *a: {"status": "completed", "prediction": "18"},
        score=numeric,
        executor_capability="fake",
    )
    result = algorithm.run(splits(), tmp_path / "run")
    assert result["status"] == "completed" and len(calls) == 5
    assert (tmp_path / "run/card0/proposal_1.yaml").exists()
    first = json.loads((tmp_path / "run/card0/proposal_0.json").read_text(encoding="utf-8"))
    assert first["status"] == "invalid" and first["error"]


def generic_contract(names=("DATA", "RESULT", "NOTE")):
    data, result, note = names
    architecture = ProtocolArchitecture(
        "author_editor_v1",
        (("author", "Describe the selected color."), ("editor", "Review the description.")),
        (
            ("author_to_editor", names, (result,)),
            ("editor_to_author", ("STATE", "FEEDBACK"), ("STATE",)),
        ),
        max_semantic_revisions=0,
    )
    return ProtocolContract(
        "toy_colors",
        architecture,
        (
            (data, "Selected object."),
            (result, "A color label, not a numeric answer."),
            (note, "An optional free-text clarification."),
            ("STATE", "PASS or EDIT."),
            ("FEEDBACK", "Requested edit."),
        ),
        edge_validation=(
            EdgeValidation(
                "author_to_editor",
                fields=(
                    FieldConstraint(result, choices=("blue", "green")),
                    FieldConstraint(note, forbidden_values=("none",)),
                ),
                required_example_fields=(note,),
            ),
            EdgeValidation(
                "editor_to_author",
                fields=(FieldConstraint("STATE", choices=("PASS", "EDIT")),),
                conditions=(
                    ConditionalConstraint("STATE", "PASS", only=("STATE",)),
                    ConditionalConstraint("STATE", "EDIT", required=("FEEDBACK",)),
                ),
            ),
        ),
        fixed_card_options=(("clarifications_enabled", True),),
    )


@pytest.mark.parametrize("names", [("DATA", "RESULT", "NOTE"), ("C", "U", "SAY")])
def test_non_psv_non_numeric_lineage_and_cdm_without_core_changes(tmp_path, names):
    from lychee_mas import REGISTRY
    from lychee_mas.core.types import Message
    from lychee_mas.methods.memory.routing.base import RouteDecision

    contract = generic_contract(names)
    schedule = EvolutionSchedule("seed", ("cycle_a", "cycle_b"), "check", ("never_use",))
    data = {
        key: [{"task_id": key, "question": f"{key}: Name the ocean color.", "gold": "blue"}]
        for key in (schedule.induction, *schedule.evolution, schedule.validation)
    }
    requests, executions = [], []

    def generate(messages, **kwargs):
        request = json.loads(messages[1]["content"])
        requests.append(request)
        assert request["task"] == "toy_colors"
        assert set(request["architecture"]["roles"]) == {"author", "editor"}
        if request["parent_card"] is not None:
            return SimpleNamespace(text=json.dumps({"decision": "KEEP", "reason": "No change."}))
        card = {
            "version": "toy-0",
            "description": "Synthetic renamed-field fixture.",
            "clarifications_enabled": True,
            "message_format": {"wire_syntax": "direct", "structure": "fields_only"},
        }
        examples = {
            "author_to_editor": [
                f"{names[0]}:ocean;{names[1]}:blue;{names[2]}:Use the usual label."
            ],
            "editor_to_author": ["STATE:PASS"],
        }
        for edge, allowed, required in contract.architecture.edges:
            card[edge] = {
                "allowed_fields": list(allowed),
                "required_fields": list(required),
                "field_rules": {
                    "canonical_order": list(allowed),
                    **{k: dict(contract.field_meanings)[k] for k in allowed},
                },
                "examples": examples[edge],
            }
        return SimpleNamespace(
            text=json.dumps({"decision": "UPDATE", "reason": "Induced.", "card": card})
        )

    class TestOnlyTokenizer:
        def encode(self, text, add_special_tokens=False):
            return list(text)

    def execute(case, mode, card, log):
        assert set(case) == {"task_id", "question"}
        executions.append(mode)
        # No external contract in Hybrid: the newly generated artifact carries it.
        memory = REGISTRY.create(
            "memory_manager",
            "cdm",
            backend=SimpleNamespace(tok=TestOnlyTokenizer()),
            task=contract.task,
            nl_strategy="hybrid_protocol" if mode == "hybrid" else "prev_output",
            protocol_path=str(card) if card else None,
        )
        payload = (
            f"{names[1]}:blue;{names[2]}:Use the usual label." if card else "The color is blue."
        )
        memory.observe([Message(sender="author", receiver="editor", content=payload)])
        bundle = memory.recall(
            RouteDecision(channel="nl"), payload if not card else case["question"]
        )
        assert bundle.NL_Channel.endswith(payload)
        if card:
            assert bundle.meta["fields"][names[1]] == "blue"
            assert bundle.meta["contract_hash"] == contract.fingerprint
        return {"status": "completed", "prediction": "blue", "communication_tokens": len(payload)}

    algorithm = ProtocolEvolution(
        contract=contract,
        schedule=schedule,
        generate=generate,
        execute=execute,
        score=lambda pred, gold: float(pred == gold),
        executor_capability="Synthetic color model.",
    )
    result = algorithm.run(data, tmp_path / "generic")
    assert len(result["history"]) == 2 and len(executions) == 7
    assert len(requests) == 3 and result["heldout_evaluated"] is False
    assert Path(result["selected_card"]).name == "toy_colors_card.yaml"
    channel = HybridProtocolMemory(result["selected_card"], "toy_colors")
    assert channel.contract.fingerprint == contract.fingerprint
    with pytest.raises(ValueError, match="invalid value"):
        channel.validate(f"{names[1]}:123", "author", "editor")
    with pytest.raises(ValueError, match="conditional"):
        channel.validate("STATE:EDIT", "editor", "author")
    with pytest.raises(ValueError, match="conditional"):
        channel.validate("STATE:PASS;FEEDBACK:extra", "editor", "author")
    assert channel.validate("STATE:EDIT;FEEDBACK:Check hue", "editor", "author")["STATE"] == "EDIT"


def test_legacy_card_requires_explicit_contract_and_never_silently_weakens(entry, tmp_path):
    card = str(write_card(tmp_path / "legacy.yaml", synthetic_legacy_card(entry)))
    with pytest.raises(ValueError, match="explicit protocol contract"):
        HybridProtocolMemory(card, "gsm8k")
    channel = HybridProtocolMemory(card, "gsm8k", contract=GSM8K_PROTOCOL)
    with pytest.raises(ValueError, match="valid ErrorType"):
        channel.validate("CHK:REVISE;E:BAD;REQ:Check arithmetic", "verifier", "solver")


def test_contract_serialization_and_mismatch_fail_closed(entry, tmp_path):
    serialized = json.loads(json.dumps(GSM8K_PROTOCOL.as_dict()))
    assert ProtocolContract.from_dict(serialized).fingerprint == GSM8K_PROTOCOL.fingerprint
    card = fixture_card(entry)
    card["validation_contract"] = serialized
    card["task_card"]["contract_sha256"] = GSM8K_PROTOCOL.fingerprint
    serialized["edge_validation"][0]["fields"][0]["forbidden_values"] = []
    path = write_card(tmp_path / "tampered.yaml", card)
    with pytest.raises(ValueError, match="contract"):
        HybridProtocolMemory(str(path), "gsm8k", contract=GSM8K_PROTOCOL)
    with pytest.raises(ValueError, match="fingerprint"):
        HybridProtocolMemory(str(path), "gsm8k")


def test_unknown_declarative_operations_are_not_executed():
    value = json.loads(json.dumps(GSM8K_PROTOCOL.as_dict()))
    value["edge_validation"][0]["fields"][0]["execute"] = "do_not_execute_this"
    with pytest.raises(ValueError, match="serialized"):
        ProtocolContract.from_dict(value)


def test_core_contains_no_experiment_imports_or_named_roles():
    import ast

    paths = [
        Path("src/lychee_mas/methods/memory/channels/hybrid_protocol.py"),
        *Path("src/lychee_mas/methods/memory/protocol").glob("*.py"),
    ]
    forbidden = {"gsm8k", "psv_v1", "planner", "solver", "verifier", "CHK", "IR", "NL"}
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(("scripts", "hybrid_gsm8k_spec"))
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value not in forbidden, (path, node.value)
