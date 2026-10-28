"""Self-contained restoration checks, plus optional parity against trusted local source."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from lychee_mas.backends.openai_api_backend import APIGenResult, OpenAICompatibleBackend
from lychee_mas.methods.memory.channels.hybrid_protocol import HybridProtocolMemory
from lychee_mas.methods.memory.protocol import ProtocolEvolution
from scripts.hybrid_gsm8k_design import GSM8KDesign
from scripts.hybrid_gsm8k_spec import GSM8K_PROTOCOL, GSM8K_SCHEDULE, ROLE_BEHAVIOR


@pytest.fixture
def entry(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    return importlib.import_module("run_memory_protocol_evolution")


def data():
    return {
        key: [{"task_id": key, "question": f"{key}: Add nine and nine.", "gold": "18"}]
        for key in ("D0", "G1", "G2", "G3", "V")
    }


def algorithm(entry, generate=None):
    return ProtocolEvolution(
        contract=GSM8K_PROTOCOL,
        schedule=GSM8K_SCHEDULE,
        design=GSM8KDesign(),
        generate=generate or entry.fake_reviewed_design,
        execute=lambda *args: {
            "status": "completed",
            "prediction": "18",
            "communication_tokens": 30,
            "messages": [],
        },
        score=lambda pred, gold: float(pred == gold),
        executor_capability="Synthetic limited executor",
        proposal_tokens=4096,
    )


def initial_fixture(entry):
    raw = entry.fake_reviewed_design([{}, {"content": "protocol-card designer"}], 4096).text
    design = GSM8KDesign()
    return design, design.materialize(json.loads(raw), "CARD-0")


def test_full_review_and_diagnosis_pipeline(entry, tmp_path):
    result = algorithm(entry).run(data(), tmp_path / "run")
    out = tmp_path / "run"
    assert result["effect_valid"] is False and result["heldout_evaluated"] is False
    assert all(not row["promoted"] for row in result["history"])
    assert len(list(out.rglob("metrics.json"))) == 13  # Added paired NL G evidence.
    assert len(list(out.rglob("review_0.json"))) == 1
    assert len(list(out.rglob("diagnosis_0.json"))) == 3
    assert not list(out.rglob("evolution_0.json"))  # No communication diagnosis => no update call.
    review = json.loads((out / "card0/initial_validation_0.json").read_text(encoding="utf-8"))
    assert review["status"] == "approved"
    assert (out / "round1/feedback.json").exists()
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["design_strategy"] == "evidence_bound_reviewed"


def test_semantic_review_rejects_then_regenerates(entry, tmp_path):
    reviews = []

    def generate(messages, max_new_tokens):
        prompt = messages[1]["content"]
        if "strict, read-only reviewer" in prompt:
            reviews.append(1)
            assert max_new_tokens == 2048
            if len(reviews) == 1:
                return APIGenResult(
                    json.dumps(
                        {
                            "approved": False,
                            "findings": [
                                {
                                    "path": "solver_to_verifier.examples[0]",
                                    "evidence": "IR insufficient",
                                    "fix": "Supply decisive evidence",
                                }
                            ],
                        }
                    ),
                    10,
                    20,
                    0,
                )
        if "PREVIOUS_JSON" in prompt:
            assert "IR insufficient" in prompt
            assert "replace the entire example with a different coherent" in prompt
            assert "suggestion conflicting with" in prompt
        return entry.fake_reviewed_design(messages, max_new_tokens)

    algorithm(entry, generate).run(data(), tmp_path / "run")
    assert len(reviews) == 2
    record = json.loads(
        (tmp_path / "run/card0/initial_validation_0.json").read_text(encoding="utf-8")
    )
    assert "IR insufficient" in record["error"] and record["review_response"]


@pytest.mark.parametrize("kind", ["invalid_json", "review_rejects"])
def test_six_initial_attempts_are_bounded_and_audited(entry, tmp_path, kind):
    calls = []

    def generate(messages, max_new_tokens):
        calls.append(messages)
        if kind == "invalid_json":
            return APIGenResult("not JSON", 10, 20, 0)
        if "strict, read-only reviewer" in messages[1]["content"]:
            return APIGenResult(
                json.dumps(
                    {
                        "approved": False,
                        "findings": [
                            {
                                "path": "solver_to_verifier.examples[0]",
                                "evidence": "missing evidence",
                                "fix": "Supply decisive evidence",
                            }
                        ],
                    }
                ),
                10,
                20,
                0,
            )
        return entry.fake_reviewed_design(messages, max_new_tokens)

    out = tmp_path / "run"
    with pytest.raises(ValueError, match="six initial"):
        algorithm(entry, generate).run(data(), out)
    assert len(calls) == (6 if kind == "invalid_json" else 12)
    assert len(list((out / "card0").glob("initial_validation_*.json"))) == 6
    assert (out / "failure.json").exists() and not (out / "result.json").exists()


def test_evidence_patch_retry_rejects_unknown_case_then_applies(entry, tmp_path):
    eligible, attempts = [], []

    def generate(messages, max_new_tokens):
        prompt = messages[1]["content"]
        if "DIAGNOSIS ONLY" in prompt:
            feedback = json.loads(prompt.split("FEEDBACK:\n", 1)[1])
            pair = next(iter(feedback["cases"].values()))[0]
            case_id = pair["hybrid"]["task_id"]
            if case_id == "G1":
                eligible.append(case_id)
                return APIGenResult(
                    json.dumps(
                        {
                            "case_analysis": [
                                {
                                    "case_id": case_id,
                                    "communication_related": True,
                                    "issue_type": "OPERATION_DIRECTION_ERROR",
                                    "evidence": "Synthetic relation gap",
                                }
                            ]
                        }
                    ),
                    10,
                    20,
                    0,
                )
        if "protocol-card evolver" in prompt:
            attempts.append(prompt)
            cite = "not_in_feedback" if len(attempts) == 1 else eligible[0]
            value = {
                "action": "UPDATE",
                "reason": "Preserve direction.",
                "changes": [
                    {
                        "path": "planner_to_solver.field_rules.P",
                        "value": "Preserve operation direction.",
                        "diagnosis_case_ids": [cite],
                        "evidence": "Synthetic relation gap",
                        "expected_benefit": "Clearer direction",
                        "risk": "None in this fixture",
                    }
                ],
            }
            return APIGenResult(json.dumps(value), 10, 20, 0)
        return entry.fake_reviewed_design(messages, max_new_tokens)

    result = algorithm(entry, generate).run(data(), tmp_path / "run")
    assert len(attempts) == 2
    record = json.loads(
        (tmp_path / "run/round1/evolution_validation_0.json").read_text(encoding="utf-8")
    )
    assert "unknown diagnosis case" in record["error"]
    evolution = json.loads((tmp_path / "run/round1/evolution.json").read_text(encoding="utf-8"))
    assert evolution["decision"] == "UPDATE"
    assert evolution["changed_paths"] == ["planner_to_solver.field_rules.P"]
    assert result["heldout_evaluated"] is False


@pytest.mark.parametrize("related,case_id", [(False, "G1"), (True, "invented")])
def test_no_evidence_bound_communication_diagnosis_forces_keep(entry, tmp_path, related, case_id):
    def generate(messages, max_new_tokens):
        prompt = messages[1]["content"]
        assert "protocol-card evolver" not in prompt
        if "DIAGNOSIS ONLY" in prompt:
            return APIGenResult(
                json.dumps(
                    {
                        "case_analysis": [
                            {
                                "case_id": case_id,
                                "communication_related": related,
                                "issue_type": "BASE_REASONING_ERROR",
                            }
                        ]
                    }
                ),
                10,
                20,
                0,
            )
        return entry.fake_reviewed_design(messages, max_new_tokens)

    result = algorithm(entry, generate).run(data(), tmp_path / "run")
    assert all(row["reason"] == "no_semantic_change_keep_parent" for row in result["history"])


@pytest.mark.parametrize(
    "problem", ["too_many", "duplicate", "unknown_case", "wrong_path", "whole_card"]
)
def test_patch_bounds_are_source_equivalent(entry, problem):
    design, card = initial_fixture(entry)
    item = {
        "path": "planner_to_solver.field_rules.P",
        "value": "Preserve direction",
        "diagnosis_case_ids": ["G1"],
    }
    changes = [item]
    if problem == "too_many":
        changes = [copy.deepcopy(item) for _ in range(4)]
    elif problem == "duplicate":
        changes = [item, item]
    elif problem == "unknown_case":
        item["diagnosis_case_ids"] = ["unseen"]
    elif problem == "wrong_path":
        item["path"] = "solver_to_verifier.allowed_fields"
    decision = {"action": "UPDATE", "changes": changes}
    if problem == "whole_card":
        decision = {"action": "UPDATE", "card": card}
    original = copy.deepcopy(card)
    with pytest.raises(ValueError):
        design.apply(card, decision, {"G1"}, "CARD-1")
    assert card == original


@pytest.mark.parametrize("alias", ["FAIL", "INSUFFICIENT"])
def test_feedback_alias_preserves_wire_and_triggers_revision(entry, alias):
    module = entry.executor()
    verifier_calls = []

    def generate(messages, max_new_tokens):
        if messages[0]["content"].startswith("You are the verifier"):
            verifier_calls.append(1)
            if len(verifier_calls) == 1:
                return APIGenResult(f"CHK:{alias};E:ARITH;REQ:Recompute the sum", 10, 20, 0)
        return module.scripted_generate(messages, max_new_tokens)

    result = module.run_demo(generate)
    assert len(result.messages) == 5
    assert result.messages[2].meta["fields"]["CHK"] == alias
    assert result.meta["revision_rounds"] == 1 and result.meta["verified"] is True


def test_two_transport_errors_do_not_force_format_fallback(entry):
    module = entry.executor()
    seen = []

    def generate(messages, max_new_tokens):
        seen.append(messages)
        if len(seen) <= 2:
            raise TimeoutError("synthetic temporary failure")
        return module.scripted_generate(messages, max_new_tokens)

    result = module.run_demo(generate)
    assert len(seen) == 5 and all(not row["fallback"] for row in result.meta["attempts"])
    assert seen[2][0] == seen[0][0]


@pytest.mark.parametrize("usage", [None, SimpleNamespace(prompt_tokens=10, completion_tokens=2)])
def test_api_metadata_is_preserved_without_network(usage):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="R:18"), finish_reason="length")],
        usage=usage,
    )
    backend = object.__new__(OpenAICompatibleBackend)
    backend.model_name = "test"
    backend.temperature = backend.top_p = backend.seed = None
    backend.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response))
    )
    result = backend.generate_chat([{"role": "user", "content": "test"}])
    assert result.finish_reason == "length" and result.usage_unknown is (usage is None)


def test_role_bodies_are_shared_and_complete(entry):
    module = entry.executor()
    for mode in ("nl", "hybrid"):
        prompts = []

        def generate(messages, max_new_tokens):
            prompts.append(messages[0]["content"])
            return module.scripted_generate(messages, max_new_tokens)

        module.run_demo(generate, mode=mode)
        for role, prompt in zip(("planner", "solver", "verifier"), prompts):
            assert ROLE_BEHAVIOR[role] in prompt
        assert (
            "An alternative answer from your own calculation alone is not sufficient." in prompts[2]
        )
        assert "Your inability to independently solve the entire task is not evidence" in prompts[2]


def test_generation_and_review_distinguish_check_status_from_error_category(entry):
    design, card = initial_fixture(entry)
    for prompt in (
        design.initialize([], "test executor"),
        design.review(card, [], "test executor"),
    ):
        assert "CHK:REVISE;E:INSUFFICIENT_INFO;REQ:the missing relation" in prompt
        assert "INSUFFICIENT_INFO is an E value, NEVER a CHK status." in prompt
        assert "CHK has only ACCEPT and REVISE." in prompt
        assert "distinguish them through E" in prompt
    review = design.review(card, [], "test executor")
    assert "Do not reject REVISE merely because essential evidence is" in review
    assert "Reject ungrounded answer-changing assumptions" in review
    assert design.schema["verifier_to_solver"]["value_constraints"]["CHK"]["allowed_values"] == [
        "ACCEPT",
        "REVISE",
    ]
    assert (
        "INSUFFICIENT_INFO"
        in (design.schema["verifier_to_solver"]["value_constraints"]["E"]["allowed_values"])
    )


def test_example_guidance_respects_task_policy_and_failed_trace_boundaries(entry):
    design, card = initial_fixture(entry)
    initial = design.initialize([], "test executor")
    assert "NOT authoritative worked solutions" in initial
    assert "NEW small, self-contained teaching" in initial
    assert "REPLACE the entire example" in initial
    assert "not a task with no determinable answer" in initial
    review = design.review(card, [], "test executor")
    assert "R remains an integer or finite decimal" in review
    assert "MUST NOT recommend a non-numeric R" in review
    assert "Missing MESSAGE evidence is different from missing TASK premises" in review
    assert "Edge examples are independent unless" in review
    assert "REPLACING that example" in review
    assert "never list 'no violation here'" in review
    assert "REQ prescribes an unverified replacement answer" in review


def test_review_receives_design_only_but_saved_card_keeps_runtime_contract(entry, tmp_path):
    prompts = []

    def generate(messages, max_new_tokens):
        prompt = messages[1]["content"]
        if "strict, read-only reviewer" in prompt:
            material = json.loads(prompt.split("CARD:", 1)[1].split("\nD0_TRACES:", 1)[0])
            assert "task_card" not in material and "validation_contract" not in material
            assert '"FAIL"' not in prompt and '"INSUFFICIENT"' not in prompt
            prompts.append(prompt)
        return entry.fake_reviewed_design(messages, max_new_tokens)

    result = algorithm(entry, generate).run(data(), tmp_path / "run")
    saved = algorithm(entry)._card(Path(result["selected_card"]))
    assert saved["validation_contract"] == json.loads(json.dumps(GSM8K_PROTOCOL.as_dict()))
    assert saved["task_card"]["contract_sha256"] == GSM8K_PROTOCOL.fingerprint
    assert len(prompts) == 1


@pytest.mark.parametrize("issue", ["outside_path", "nonviolation", "optional", "contradiction"])
def test_inconsistent_review_rechecks_same_candidate_without_automatic_approval(
    entry, tmp_path, issue
):
    _, card = initial_fixture(entry)
    snapshots = []

    def generate(messages, max_new_tokens):
        prompt = messages[1]["content"]
        snapshots.append(prompt.split("CARD:", 1)[1].split("\nD0_TRACES:", 1)[0])
        finding = {"path": "solver_to_verifier.examples[0]", "evidence": "missing evidence"}
        response = {"approved": False, "findings": [finding]}
        if len(snapshots) == 1:
            if issue == "outside_path":
                finding["path"] = "validation_contract.edge_validation[2].fields[0].choices"
            elif issue == "nonviolation":
                finding["evidence"] = "The example is correct. No violation here."
            elif issue == "optional":
                finding["fix"] = "Optionally add another example"
            else:
                response["approved"] = True
        return APIGenResult(json.dumps(response), 10, 20, 0)

    _, review = algorithm(entry, generate)._review_card(tmp_path, card, [], 0)
    assert review["approved"] is False  # Rechecked genuine rejection is NOT a pass.
    assert snapshots[0] == snapshots[1]
    assert (tmp_path / "review_recheck_0.json").exists()
    assert json.loads((tmp_path / "review_validation_0_0.json").read_text())["errors"]


def test_persistent_reviewer_inconsistency_stops_without_regenerating_card(entry, tmp_path):
    calls = []

    def generate(messages, max_new_tokens):
        calls.append(messages[1]["content"])
        if "strict, read-only reviewer" in messages[1]["content"]:
            return APIGenResult('{"approved":false,"findings":[]}', 10, 20, 0)
        return entry.fake_reviewed_design(messages, max_new_tokens)

    with pytest.raises(RuntimeError, match="after one recheck"):
        algorithm(entry, generate).run(data(), tmp_path / "run")
    assert len(calls) == 3  # One generation, two reviews; no silent pass or regeneration.
    assert not (tmp_path / "run/result.json").exists()
    assert (tmp_path / "run/failure.json").exists()


def test_approved_review_may_include_nonblocking_recommendations(entry, tmp_path):
    _, card = initial_fixture(entry)

    def generate(messages, max_new_tokens):
        return APIGenResult(
            json.dumps(
                {
                    "approved": True,
                    "findings": [],
                    "recommendations": [
                        {"path": "value_policy", "suggestion": "Optional improvement"}
                    ],
                }
            ),
            10,
            20,
            0,
        )

    _, review = algorithm(entry, generate)._review_card(tmp_path, card, [], 0)
    assert review["approved"] is True and review["recommendations"]
    assert not (tmp_path / "review_recheck_0.json").exists()


def test_default_promoted_card_preserves_source_and_loads_embedded_contract(entry):
    import yaml

    module = entry.executor()
    config = module.load_memory_config("gsm8k", "hybrid")
    path = Path(config["protocol_path"])
    assert path.name == "gsm8k_psv_hybrid_card_20261028.yaml"
    assert sorted(p.name for p in path.parent.glob("*.yaml")) == [path.name]
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    metadata = raw["task_card"]
    assert metadata["name"] == path.stem
    assert metadata["promoted_source_sha256"] == (
        "7f15911ee23835cae50f21617115981ca97ff599e8a4c9fb11d1c38065504233"
    )
    source = Path(metadata["promoted_from"])
    if source.exists():
        assert hashlib.sha256(source.read_bytes()).hexdigest() == metadata["promoted_source_sha256"]
        original = yaml.safe_load(source.read_text(encoding="utf-8"))
        assert {k: v for k, v in raw.items() if k != "task_card"} == {
            k: v for k, v in original.items() if k != "task_card"
        }
    channel = HybridProtocolMemory(str(path), "gsm8k")  # No external contract required.
    assert channel.contract.fingerprint == config["protocol_contract"].fingerprint
    assert channel.contract.fingerprint == metadata["contract_sha256"]
    assert channel.version == "CARD-0-gsm8k"
    assert channel.protocol_hash == (
        "4bcdae885cc464e0a51977368e4bc06650c0326dbdcb182ce0327477c1d32daa"
    )
    result = module.run_demo(module.scripted_generate)
    assert result.final_answer.content == "18"
    assert all(m.meta["protocol_hash"] == channel.protocol_hash for m in result.messages)


def test_optional_original_function_and_role_parity():
    original = Path(__file__).resolve().parents[2] / "LycheeMAS"
    source = original / "src/lychee_mas/hybrid_comm/protocol/design.py"
    if not source.exists():
        pytest.skip("Original experiment checkout not present; self-contained checks still run")
    current = Path("scripts/hybrid_gsm8k_design.py")
    old = {
        n.name: ast.dump(n, include_attributes=False)
        for n in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef)
    }
    current_source = current.read_text(encoding="utf-8")
    # Approved deviations: field-boundary clarifications and schema-derived enum text.
    clarifications = [
        (
            '    error_values = schema["verifier_to_solver"]'
            '["value_constraints"]["E"]["allowed_values"]\n'
            '    error_example, error_list = error_values[0], ", ".join(error_values)\n',
            "",
        ),
        ("such as `{error_example}`.", "such as `ARITH`."),
        (
            "The only valid E values are {error_list}. Diagnostic issue labels",
            "The only valid E values are ARITH, RELATION, MISSING_FACT, "
            "MISSING_UNIT, WRONG_TARGET,\n"
            "INSUFFICIENT_INFO, and OTHER. Diagnostic issue labels",
        ),
        (
            "   For missing essential evidence use "
            "CHK:REVISE;E:INSUFFICIENT_INFO;REQ:the missing relation.\n"
            "   INSUFFICIENT_INFO is an E value, NEVER a CHK status. "
            "CHK has only ACCEPT and REVISE.\n"
            "   Both proven errors and missing evidence request revision; "
            "distinguish them through E,\n"
            "   not through a new CHK value. Request only the particular "
            "missing relation or operation.\n"
            "   Do not prescribe an unverified replacement answer in REQ. Accept when sufficient",
            "   Use INSUFFICIENT_INFO for the latter and request only "
            "the particular missing relation or\n"
            "   operation. Do not prescribe an unverified replacement answer in REQ. "
            "Accept when sufficient",
        ),
        (
            "missing evidence using CHK:REVISE;E:INSUFFICIENT_INFO;REQ:the missing relation.\n"
            "INSUFFICIENT_INFO is an E value, NEVER a CHK status. CHK has only ACCEPT and REVISE.\n"
            "Both proven errors and missing evidence request revision; "
            "distinguish them through E,\n"
            "not through a new CHK value. Do not reject REVISE merely "
            "because essential evidence is\n"
            "missing. Reject when missing evidence is asserted to prove a concrete error, when an\n"
            "inappropriate E category is used, or when REQ prescribes "
            "an unverified replacement answer.",
            "missing evidence using INSUFFICIENT_INFO. Reject rules "
            "that conflate lack of evidence with\n"
            "an established error or prescribe an unverified replacement answer.",
        ),
    ]
    for clarified, original_text in clarifications:
        assert current_source.count(clarified) == 1
        current_source = current_source.replace(clarified, original_text)
    migrated = {
        n.name: ast.dump(n, include_attributes=False)
        for n in ast.parse(current_source).body
        if isinstance(n, ast.FunctionDef)
    }
    assert len(migrated) == 16
    assert all(old[name] == value for name, value in migrated.items())
    prompt_dir = original / "configs/hybrid_comm/prompts"
    base = (prompt_dir / "system_rules.txt").read_text(encoding="utf-8").strip()
    for role, body in ROLE_BEHAVIOR.items():
        text = (prompt_dir / f"{role}_system.txt").read_text(encoding="utf-8")
        expected = text.split("\n", 1)[1].split("Allowed communication mode:")[0]
        assert expected.replace("$base_rules", base).strip() == body


def test_sdk_retries_explicitly_disabled_in_experiment_entries():
    for path in (
        "scripts/run_memory_protocol_compare.py",
        "scripts/run_memory_protocol_evolution.py",
        "examples/03_gsm8k_memory_protocol.py",
    ):
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "with_options"
        ]
        assert calls and all(
            any(
                k.arg == "max_retries" and isinstance(k.value, ast.Constant) and k.value.value == 0
                for k in call.keywords
            )
            for call in calls
        )
