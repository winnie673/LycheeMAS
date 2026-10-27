"""Offline paired evaluation, revision, accounting and failure provenance."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def evaluation():
    spec = importlib.util.spec_from_file_location(
        "memory_protocol_evaluation_test",
        "scripts/run_memory_protocol_compare.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CharacterTokenizer:
    """Test-only accounting oracle, explicitly NOT a real model tokenizer."""

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(text)


def prompt_role(messages):
    system = messages[0]["content"]
    return (system.splitlines()[0].removeprefix("role=") if system.startswith("role=")
            else system.split("You are the ", 1)[1].split(" ", 1)[0])


def scripted(module, mode, *, decisions=("REVISE", "ACCEPT"), fail_role=None):
    calls = []
    counts = {}

    def generate(messages, max_new_tokens):
        assert max_new_tokens == 512
        role = prompt_role(messages)
        count = counts.get(role, 0)
        counts[role] = count + 1
        calls.append((role, messages))
        if role == fail_role:
            payload = "invalid"
        elif role == "planner":
            payload = "G:sum;P:sum=9+9" if mode == "hybrid" else "Find the sum by adding 9 and 9."
        elif role == "solver":
            answer = "17" if count == 0 else "18"
            payload = (
                "IR:addition;R:" + answer
                if mode == "hybrid"
                else "Adding the two numbers gives \\boxed{" + answer + "}"
            )
        else:
            decision = decisions[min(count, len(decisions) - 1)]
            payload = "CHK:" + decision if mode == "hybrid" else decision
            if decision == "REVISE":
                payload += (
                    ";E:ARITH;REQ:Recompute the addition"
                    if mode == "hybrid"
                    else "\nRecompute the addition."
                )
        return module.APIGenResult(payload, 10, 20, 0.01)

    return generate, calls


@pytest.mark.parametrize("mode", ["nl", "hybrid"])
def test_semantic_revision_rechecks_and_receives_memory_snapshots(evaluation, mode):
    module = evaluation.executor()
    generate, calls = scripted(module, mode)
    trajectory = module.run_demo(
        generate,
        mode=mode,
        backend=SimpleNamespace(tok=CharacterTokenizer()),
    )
    assert [role for role, _ in calls] == ["planner", "solver", "verifier", "solver", "verifier"]
    assert trajectory.final_answer.content == "18"
    assert trajectory.meta["verified"] is True
    assert trajectory.meta["revision_rounds"] == 1
    assert trajectory.total_tokens == 150
    assert [message.round for message in trajectory.messages] == [0, 0, 0, 1, 1]
    revision = trajectory.messages[3]
    assert revision.meta["incoming_message_ids"] == [m.id for m in trajectory.messages[:3]]
    revised_prompt = calls[3][1][1]["content"]
    for message in trajectory.messages[:3]:
        assert revised_prompt.count(message.content) == 1
    assert calls[4][1][1]["content"].endswith(trajectory.messages[3].content)
    assert sum(m.meta["communication_tokens"] for m in trajectory.messages) == sum(
        len(m.content) for m in trajectory.messages
    )
    if mode == "nl":
        assert "Allowed fields" not in calls[1][1][0]["content"]
        assert "IR:" not in calls[1][1][1]["content"]


@pytest.mark.parametrize("mode", ["nl", "hybrid"])
@pytest.mark.parametrize(
    "limit,decisions,calls_expected,verified",
    [
        (1, ("ACCEPT",), 3, True),
        (0, ("REVISE",), 3, False),
        (1, ("REVISE", "REVISE"), 5, False),
    ],
)
def test_revision_stop_rules(evaluation, mode, limit, decisions, calls_expected, verified):
    module = evaluation.executor()
    generate, calls = scripted(module, mode, decisions=decisions)
    result = module.run_demo(generate, mode=mode, max_revision_rounds=limit)
    assert len(calls) == calls_expected
    assert result.meta["verified"] is verified
    assert result.final_answer.content == ("18" if calls_expected == 5 else "17")


@pytest.mark.parametrize(
    "text,role,answer",
    [
        (r"The positive root is \boxed{\sqrt{7}}", "solver", r"\sqrt{7}"),
        (r"The set is \boxed{\{1,2\}}", "solver", r"\{1,2\}"),
        ("ACCEPT", "verifier", "ACCEPT"),
        ("REVISE\nFix the sign.", "verifier", "REVISE"),
    ],
)
def test_nl_control_extraction_preserves_math(evaluation, text, role, answer):
    fields = evaluation.executor().nl_fields(text, role)
    assert fields["R" if role == "solver" else "CHK"] == answer


@pytest.mark.parametrize(
    "text,role",
    [
        (r"Answer is 18", "solver"),
        (r"\boxed{18} then more", "solver"),
        (r"\boxed{18", "solver"),
        (r"\boxed{}", "solver"),
        (r"\boxed{1} \boxed{18}", "solver"),
        ("Probably ACCEPT", "verifier"),
        ("REVISE", "verifier"),
    ],
)
def test_nl_malformed_answers_and_decisions_are_not_silently_inferred(evaluation, text, role):
    with pytest.raises(ValueError):
        evaluation.executor().nl_fields(text, role)


@pytest.mark.parametrize(
    "pred,gold,expected",
    [
        ("18", "18", 1),
        ("18.0", "18", 1),
        ("1,000", "1000", 1),
        ("1,8", "18", 0),
        ("-18", "18", 0),
        ("118", "18", 0),
        ("18/2", "18", 0),
        ("the answer is 18", "18", 0),
        ("NaN", "18", 0),
    ],
)
def test_gsm8k_scoring_is_bare_numeric_not_substring(evaluation, pred, gold, expected):
    assert evaluation.score_answer("gsm8k", pred, gold) == expected


def test_nl_prev_output_receives_payload_not_original_question(evaluation, monkeypatch):
    from lychee_mas.methods.memory.channels.nl import NLMemory

    observed = []
    original = NLMemory.recall

    def spy(memory, text):
        if memory.strategy == "prev_output":
            observed.append(text)
        return original(memory, text)

    monkeypatch.setattr(NLMemory, "recall", spy)
    module = evaluation.executor()
    result = module.run_demo(mode="nl", question="Unique original question, 9+9?")
    assert observed == [message.content for message in result.messages]
    assert all("Unique original question" not in text for text in observed)


def test_nl_format_retry_and_fallback_are_bounded(evaluation):
    module = evaluation.executor()
    counts = {}

    def generate(messages, max_new_tokens):
        role = prompt_role(messages)
        counts[role] = counts.get(role, 0) + 1
        if role == "solver" and counts[role] < 3:
            return module.APIGenResult("I think it is 18", 10, 20, 0)
        return module.scripted_generate(messages, max_new_tokens)

    result = module.run_demo(generate, mode="nl")
    assert result.final_answer.content == "18"
    assert len(result.meta["attempts"]) == 5
    assert result.messages[1].meta["fallback"] is True
    assert result.total_tokens == 150


@pytest.mark.parametrize("mode", ["nl", "hybrid"])
def test_failed_task_keeps_partial_communications_and_all_attempts(evaluation, mode, tmp_path):
    module = evaluation.executor()
    generate, calls = scripted(module, mode, fail_role="solver")
    row = evaluation.evaluate_case(
        {"task_id": "test-1", "question": "9+9?", "gold": "18"},
        mode=mode,
        benchmark="gsm8k",
        generate=generate,
        backend=SimpleNamespace(tok=CharacterTokenizer()),
        max_new_tokens=512,
        max_revision_rounds=1,
        log_path=tmp_path / "attempts.jsonl",
    )
    assert row["status"] == "failed" and row["score"] == 0
    assert row["calls"] == 4 and row["model_output_tokens"] == 80
    assert len(row["messages"]) == 1
    assert row["communication_tokens"] == len(row["messages"][0]["content"])
    assert [role for role, _ in calls] == ["planner", "solver", "solver", "solver"]
    records = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    assert len(records) == 4 and records[-1]["fallback"] is True


def test_unverified_candidate_is_scored_independently(evaluation, tmp_path):
    module = evaluation.executor()
    generate, _ = scripted(module, "hybrid", decisions=("REVISE", "REVISE"))
    row = evaluation.evaluate_case(
        {"task_id": "one", "question": "9+9?", "gold": "18"},
        mode="hybrid",
        benchmark="gsm8k",
        generate=generate,
        backend=None,
        max_new_tokens=512,
        max_revision_rounds=1,
        log_path=tmp_path / "attempts.jsonl",
    )
    assert row["status"] == "completed" and row["verified"] is False
    assert row["prediction"] == "18" and row["is_correct"] is True


def test_missing_usage_not_reported_as_zero_cost(evaluation):
    usage = evaluation.attempt_usage(
        [
            {"status": "transport_error", "usage_available": False, "fallback": False},
            {
                "status": "delivered",
                "usage_available": True,
                "prompt_tokens": 10,
                "output_tokens": 20,
                "fallback": False,
                "latency_s": 1,
            },
        ]
    )
    assert usage["model_output_tokens"] is None and usage["known_output_tokens"] == 20


def test_batch_shared_executor_pairing_gold_isolation_and_no_overwrite(evaluation, tmp_path):
    module = evaluation.executor()
    prompts = []

    def generate(messages, max_new_tokens):
        prompts.append(messages)
        return module.scripted_generate(messages, max_new_tokens)

    cases = [
        {"task_id": str(i), "question": "Compute nine plus nine.", "gold": "918273"}
        for i in range(2)
    ]
    out = tmp_path / "paired"
    comparison = evaluation.run_evaluation(
        cases, out, benchmark="gsm8k", generate=generate, metadata={"backend": "fake"}
    )
    assert comparison["effect_valid"] is False
    assert comparison["paired"] == {"both_incorrect": 2}
    for mode in ("nl", "hybrid"):
        metrics = json.loads((out / mode / "metrics.json").read_text())
        assert metrics["n"] == 2 and metrics["avg_calls"] == 3
        assert metrics["avg_communication_tokens"] is None
        assert metrics["avg_model_output_tokens"] == 60
    assert all("918273" not in json.dumps(prompt) for prompt in prompts)
    assert json.loads((out / "config.json").read_text())["max_revision_rounds"] == 1
    with pytest.raises(FileExistsError):
        evaluation.run_evaluation(cases, out, benchmark="gsm8k", generate=generate)


def test_manifest_scope_ids_and_limit(evaluation, tmp_path):
    manifest = tmp_path / "split.json"
    row = {"task_id": "one", "question": "9+9?", "gold": "18"}
    manifest.write_text(json.dumps({"dataset": "openai/gsm8k", "splits": {"T": [row]}}))
    assert evaluation.load_cases(manifest, "gsm8k", "T", 1) == [row]
    with pytest.raises(ValueError, match="benchmark"):
        evaluation.load_cases(manifest, "unsupported_task", "T")
    manifest.write_text(json.dumps({"task": "gsm8k", "splits": {"T": [row, row]}}))
    with pytest.raises(ValueError, match="duplicate"):
        evaluation.load_cases(manifest, "gsm8k", "T")


def test_fatal_error_aborts_batch_but_persists_attempt_and_task(evaluation, tmp_path):
    def broken(*args, **kwargs):
        raise ValueError("backend misconfigured")

    out = tmp_path / "failed"
    with pytest.raises(ValueError, match="backend misconfigured"):
        evaluation.run_evaluation(
            [{"task_id": "one", "question": "9+9?", "gold": "18"}],
            out,
            benchmark="gsm8k",
            generate=broken,
        )
    assert not (out / "comparison.json").exists()
    assert json.loads((out / "failure.json").read_text())["status"] == "aborted"
    raw = json.loads(next((out / "nl").glob("*.task.json")).read_text())
    assert raw["status"] == "fatal_error" and raw["calls"] == 1


def test_batch_rejects_unapproved_benchmark_before_writes(evaluation, tmp_path):
    out = tmp_path / "unsupported"
    with pytest.raises(ValueError, match="unsupported"):
        evaluation.run_evaluation([{"task_id": "one"}], out, benchmark="unknown", generate=None)
    assert not out.exists()


@pytest.mark.parametrize("mode", ["nl", "hybrid"])
def test_revision_and_format_budget_has_fifteen_call_hard_bound(evaluation, mode):
    module = evaluation.executor()
    counters = {}

    def generate(messages, max_new_tokens):
        role = prompt_role(messages)
        index = counters.get(role, 0)
        counters[role] = index + 1
        if index % 3 < 2:
            text = ""  # Recoverable invalid output for either representation.
        elif role == "verifier":
            text = (
                "CHK:REVISE;E:ARITH;REQ:Recompute the sum"
                if mode == "hybrid"
                else "REVISE\nRecompute the sum."
            )
        else:
            return module.scripted_generate(messages, max_new_tokens)
        return module.APIGenResult(text, 10, 20, 0.0)

    result = module.run_demo(generate, mode=mode)
    assert counters == {"planner": 3, "solver": 6, "verifier": 6}
    assert len(result.meta["attempts"]) == 15 and len(result.messages) == 5
    assert result.meta["revision_rounds"] == 1 and result.meta["verified"] is False
    assert result.total_tokens == 450


def test_bad_backend_contract_is_audited_without_retry(evaluation, tmp_path):
    module = evaluation.executor()
    attempts = []
    with pytest.raises(TypeError, match="contract"):
        module.run_demo(
            lambda *args, **kwargs: None, attempts=attempts, log_path=tmp_path / "attempts.jsonl"
        )
    assert len(attempts) == 1 and attempts[0]["status"] == "fatal_error"
    assert "contract" in attempts[0]["error"]


def test_unknown_backend_usage_is_not_assumed_free(evaluation, tmp_path):
    module = evaluation.executor()

    def generate(messages, max_new_tokens):
        response = module.scripted_generate(messages, max_new_tokens)
        response.n_prompt_pos = response.n_gen_tokens = 0
        return response

    row = evaluation.evaluate_case(
        {"task_id": "one", "question": "9+9?", "gold": "18"},
        mode="nl",
        benchmark="gsm8k",
        generate=generate,
        backend=None,
        max_new_tokens=512,
        max_revision_rounds=1,
        log_path=tmp_path / "attempts.jsonl",
    )
    assert row["status"] == "completed" and row["is_correct"] is True
    assert row["usage_complete"] is False and row["model_output_tokens"] is None


def test_batch_detects_source_change_without_writing_success(evaluation, tmp_path, monkeypatch):
    module = evaluation.executor()
    original = evaluation.digest
    counts = {}
    default_card_name = Path(module.load_memory_config("gsm8k", "hybrid")["protocol_path"]).name

    def changed_digest(path):
        name = str(path)
        counts[name] = counts.get(name, 0) + 1
        return "changed" if path.name == default_card_name and counts[name] > 1 else original(path)

    monkeypatch.setattr(evaluation, "digest", changed_digest)
    out = tmp_path / "changed"
    with pytest.raises(RuntimeError, match="not frozen"):
        evaluation.run_evaluation(
            [{"task_id": "one", "question": "9+9?", "gold": "18"}],
            out,
            benchmark="gsm8k",
            generate=module.scripted_generate,
        )
    assert (out / "failure.json").exists() and not (out / "comparison.json").exists()
