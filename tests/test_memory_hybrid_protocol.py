"""Channel contract tests using registered CDM and the existing core.Message."""
from __future__ import annotations

import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from lychee_mas import REGISTRY
from lychee_mas.core.types import Message
from lychee_mas.methods.memory.channels.hybrid_protocol import (
    HybridProtocolMemory,
    ProtocolValidationError,
)
from lychee_mas.methods.memory.channels.nl import PREV_OUTPUT_HEADER, NLMemory
from lychee_mas.methods.memory.routing.base import RouteDecision
from scripts.hybrid_gsm8k_spec import GSM8K_PROTOCOL

CARD = "configs/memory/task_cards/gsm8k_psv_hybrid_card_20261028.yaml"
PAYLOAD = "G:sum;P:sum=9+9;NL:Count both groups."


class Tokenizer:
    def encode(self, text: str, add_special_tokens: bool) -> list[str]:
        assert add_special_tokens is False
        return list(text)


def manager(tokenizer=None):
    return REGISTRY.create(
        "memory_manager", "cdm", backend=SimpleNamespace(tok=tokenizer),
        nl_strategy="hybrid_protocol", protocol_path=CARD, task="gsm8k",
        protocol_contract=GSM8K_PROTOCOL,
    )


def deliver(memory, text=PAYLOAD, sender="planner", receiver="solver", message_id=None):
    message = Message(sender=sender, receiver=receiver, content=text)
    if message_id is not None:
        message.id = message_id
    memory.observe([message])
    return message


def test_card_rules_and_fingerprint():
    memory = manager()
    instructions = memory.nl.output_instructions("planner", "solver")
    assert "Allowed fields: K, G, P, NL" in instructions
    assert "Explicit givens or relations selected from the shared problem" in instructions
    assert "Concise symbolic relations" in instructions
    assert "Example:" in instructions
    deliver(memory)
    meta = memory.nl.metadata()
    assert meta["protocol_version"] == "CARD-0-gsm8k"
    assert meta["protocol_hash"] == (
        "4bcdae885cc464e0a51977368e4bc06650c0326dbdcb182ce0327477c1d32daa"
    )


def test_observe_recall_uses_existing_message_bundle_and_header():
    memory = manager(Tokenizer())
    message = deliver(memory)
    bundle = memory.recall(RouteDecision(channel="nl"), "Original question, not wire content")
    assert bundle.NL_Channel == f"{PREV_OUTPUT_HEADER}\n{PAYLOAD}"
    assert bundle.NL_strategy == "hybrid_protocol"
    assert bundle.meta["fields"]["NL"] == "Count both groups."
    assert bundle.meta["communication_tokens"] == len(PAYLOAD)
    assert bundle.meta["message_id"] == message.id
    assert bundle.NL_Channel.count("Count both groups.") == 1
    assert bundle.Latent_Channel is None


@pytest.mark.parametrize("payload,sender,receiver", [
    ("G:x;C:old", "planner", "solver"),
    ("G:x;U:old", "planner", "solver"),
    ("G:x;G:y", "planner", "solver"),
    ("P:x", "planner", "solver"),
    ("G:x;NL:", "planner", "solver"),
    ('{"G":"x"}', "planner", "solver"),
    ("IR:x", "solver", "verifier"),
    ("CHK:ACCEPT;REQ:x", "verifier", "solver"),
    ("CHK:REVISE;E:ARITH", "verifier", "solver"),
    ("CHK:REVISE;E:UNKNOWN;REQ:x", "verifier", "solver"),
    ("CHK:UNKNOWN;E:ARITH;REQ:x", "verifier", "solver"),
    ("G:x", "critic", "solver"),
    ("G:x", "planner", "verifier"),
    ("G:x", "planner", None),
    ("", "planner", "solver"),
])
def test_invalid_messages_fail_without_destroying_previous_delivery(payload, sender, receiver):
    memory = manager()
    deliver(memory)
    bad = Message(sender=sender, receiver=receiver, content=payload)
    for _ in range(2):
        with pytest.raises(ValueError):
            memory.observe([bad])
    assert memory.recall(RouteDecision(channel="nl"), "").NL_Channel.endswith(PAYLOAD)


def test_feedback_and_reset_follow_channel_lifecycle():
    memory = manager()
    deliver(memory, "CHK:REVISE;E:ARITH;REQ:Recompute 9+9", "verifier", "solver")
    assert memory.nl.metadata()["fields"]["CHK"] == "REVISE"
    deliver(memory, "IR:9+9=18;R:18", "solver", "verifier")
    assert memory.nl.metadata()["fields"]["R"] == "18"
    deliver(memory, "CHK:ACCEPT", "verifier", "solver")
    assert memory.nl.metadata()["fields"] == {"CHK": "ACCEPT"}
    memory.reset()
    assert memory.recall(RouteDecision(channel="nl"), "").NL_Channel == ""
    assert memory.nl.metadata() == {}


def test_dict_and_core_messages_have_same_metadata():
    memory = manager()
    memory.observe([{"sender": "planner", "receiver": "solver", "content": "G:x", "id": "one"}])
    assert memory.nl.metadata()["sender"] == "planner"
    assert memory.nl.metadata()["receiver"] == "solver"
    assert memory.nl.metadata()["message_id"] == "one"


def test_duplicate_delivery_does_not_duplicate_transcript_but_new_message_id_does():
    memory = manager()
    first = deliver(memory, message_id="one")
    memory.observe([first])
    deliver(memory, message_id="two")
    assert len(memory._transcript) == 2
    assert memory.nl.metadata()["message_id"] == "two"


def test_no_tokenizer_is_explicitly_unavailable_and_encoding_failure_is_not_hidden():
    memory = manager()
    deliver(memory)
    assert memory.nl.metadata()["communication_tokens"] is None
    assert memory.nl.metadata()["communication_tokens_exact"] is False

    class BrokenTokenizer:
        def encode(self, *args, **kwargs):
            raise RuntimeError("encoding failed")

    with pytest.raises(RuntimeError, match="encoding failed"):
        deliver(manager(BrokenTokenizer()))


def test_metadata_is_not_mutable_protocol_storage():
    memory = manager()
    deliver(memory)
    meta = memory.nl.metadata()
    meta["fields"]["G"] = "wrong"
    assert memory.nl.metadata()["fields"]["G"] == "sum"


def test_router_channels_keep_the_existing_manager_behavior(monkeypatch):
    memory = manager()
    deliver(memory)
    assert memory.recall(RouteDecision(channel="none"), "").NL_Channel is None

    def materialize(bundle, source):
        bundle.Latent_Channel = source

    monkeypatch.setattr(memory.latent, "materialize", materialize)
    both = memory.recall(RouteDecision(channel="both"), "")
    assert both.NL_Channel.endswith(PAYLOAD)
    assert both.Latent_Channel
    latent = memory.recall(RouteDecision(channel="latent"), "")
    assert latent.NL_Channel is None
    assert latent.Latent_Channel


def test_wrong_task_and_role_rejected_before_generation():
    with pytest.raises(ValueError, match="protocol_path"):
        NLMemory(strategy="hybrid_protocol")
    wrong_task = NLMemory(strategy="hybrid_protocol", protocol_path=CARD, task="unsupported_task")
    with pytest.raises(ValueError, match="does not support task"):
        wrong_task.output_instructions("planner", "solver")
    with pytest.raises(ValueError, match="unsupported"):
        manager().nl.output_instructions("planner", "critic")
    assert NLMemory().output_instructions("any", "other") == ""


def test_independent_managers_do_not_share_memory():
    def work(label):
        memory = manager()
        deliver(memory, f"G:{label}")
        return memory.recall(RouteDecision(channel="nl"), "").NL_Channel
    with ThreadPoolExecutor(max_workers=2) as executor:
        assert list(executor.map(work, ["alpha", "beta"])) == [
            f"{PREV_OUTPUT_HEADER}\nG:alpha", f"{PREV_OUTPUT_HEADER}\nG:beta",
        ]


def test_standard_graph_and_existing_backend_callback_contract(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "memory_channel_demo", "examples/03_gsm8k_memory_protocol.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    calls = []

    def generate(messages, max_new_tokens):
        calls.append(messages)
        return module.scripted_generate(messages, max_new_tokens=max_new_tokens)

    result = module.run_demo(generate)
    assert result.final_answer.content == "18"
    assert result.total_tokens == 90
    assert len(result.messages) == len(calls) == 3
    assert calls[1][1]["content"].endswith(
        f"{PREV_OUTPUT_HEADER}\nG:sum;P:sum=9+9",
    )
    assert all(isinstance(message, Message) for message in result.messages)


def demo_module():
    spec = importlib.util.spec_from_file_location(
        "memory_protocol_recovery_demo", "examples/03_gsm8k_memory_protocol.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def scripted_attempts(module, scripts):
    calls = []
    counts = {}

    def generate(messages, max_new_tokens):
        assert max_new_tokens == 512
        system = messages[0]["content"]
        role = (system.splitlines()[0].removeprefix("role=") if system.startswith("role=")
                else system.split("You are the ", 1)[1].split(" ", 1)[0])
        index = counts.get(role, 0)
        counts[role] = index + 1
        calls.append((role, messages))
        value = scripts.get(role, {}).get(index)
        if isinstance(value, Exception):
            raise value
        if value is None:
            return module.scripted_generate(messages, max_new_tokens)
        if not isinstance(value, str):
            return value
        return module.APIGenResult(value, 10, 20, 0.0)
    return generate, calls


@pytest.mark.parametrize("payload", [
    "G:x\nP:y", "G:x\rP:y", "G:x\nNL:y", "G:x\nC:legacy",
    "G:x;NL:none", "G:x;NL:n/a",
])
def test_wire_v2_rejects_boundaries_and_placeholders(payload):
    protocol = HybridProtocolMemory(CARD, "gsm8k", contract=GSM8K_PROTOCOL)
    with pytest.raises(ProtocolValidationError):
        protocol.validate(payload, "planner", "solver")
    assert protocol.recall() == ""


def test_node_retry_sees_rejected_text_and_keeps_all_usage():
    module = demo_module()
    generate, calls = scripted_attempts(module, {"planner": {0: "G:sum\nP:add"}})
    result = module.run_demo(generate)
    assert len(calls) == 4
    assert result.total_tokens == 120
    assert calls[1][1][-2] == {"role": "assistant", "content": "G:sum\nP:add"}
    assert "missing semicolon" in calls[1][1][-1]["content"]
    assert result.messages[0].meta["attempt_count"] == 2
    assert result.messages[0].meta["fallback"] is False
    assert len(result.messages) == 3


def test_node_fallback_uses_same_card_and_redacts_failed_outputs(tmp_path, monkeypatch):
    module = demo_module()
    secret = "test-only-credential-not-real"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    generate, calls = scripted_attempts(module, {"solver": {
        0: "R:18\nIR:" + secret, 1: "R:18\nIR:9+9=18", 2: "IR:9+9=18;R:18",
    }})
    log = tmp_path / "attempts.jsonl"
    result = module.run_demo(generate, benchmark="gsm8k", log_path=log)
    assert result.final_answer.content == "18"
    assert result.total_tokens == 150
    fallback = calls[3][1][0]["content"]
    assert fallback.startswith("role=solver") and "Allowed fields: IR, R, NL" in fallback
    assert "R contains only the final answer" in fallback
    assert "never swap R with IR or NL" in fallback
    assert result.messages[1].meta["fallback"] is True
    records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 5
    assert records[1]["status"] == "validation_error"
    assert "[REDACTED]" in records[1]["raw_output"]
    assert secret not in log.read_text(encoding="utf-8")


def test_node_exhaustion_persists_evidence_and_never_calls_verifier(tmp_path):
    module = demo_module()
    generate, calls = scripted_attempts(module, {"solver": dict.fromkeys(range(3), "invalid")})
    attempts = []
    log = tmp_path / "attempts.jsonl"
    with pytest.raises(RuntimeError, match="solver exhausted 3 attempts"):
        module.run_demo(generate, attempts=attempts, log_path=log)
    assert [role for role, _ in calls] == ["planner", "solver", "solver", "solver"]
    assert len(attempts) == 4
    assert all(row["status"] == "validation_error" for row in attempts[1:])
    assert all(row["raw_output"] == "invalid" and row["error"] for row in attempts[1:])


def test_transport_error_is_separate_from_format_and_can_recover():
    module = demo_module()
    generate, calls = scripted_attempts(module, {"solver": {0: TimeoutError("scripted timeout")}})
    result = module.run_demo(generate)
    assert len(calls) == 4 and result.total_tokens == 90
    failed = result.meta["attempts"][1]
    assert failed["status"] == "transport_error" and failed["usage_available"] is False


@pytest.mark.parametrize("finish,tokens", [("length", 20), ("", 512)])
def test_length_metadata_is_audited_but_syntax_controls_delivery(finish, tokens):
    module = demo_module()
    truncated = SimpleNamespace(text="R:18", n_prompt_pos=10, n_gen_tokens=tokens,
                                finish_reason=finish, latency_s=0.0)
    generate, calls = scripted_attempts(module, {"solver": {0: truncated}})
    result = module.run_demo(generate)
    assert len(calls) == 3 and result.final_answer.content == "18"
    assert result.meta["attempts"][1]["suspected_truncation"] is (finish == "length")
    assert result.meta["attempts"][1]["length_budget_reached"] is (tokens >= 512)
    assert result.meta["attempts"][1]["status"] == "delivered"


def test_unexpected_backend_errors_and_tokenizer_errors_are_not_retried():
    module = demo_module()
    generate, calls = scripted_attempts(module, {"planner": {0: ValueError("configuration bug")}})
    attempts = []
    with pytest.raises(ValueError, match="configuration bug"):
        module.run_demo(generate, attempts=attempts)
    assert len(calls) == 1 and attempts[0]["status"] == "fatal_error"

    class BrokenTokenizer:
        def encode(self, *args, **kwargs):
            raise RuntimeError("tokenizer bug")

    generate, calls = scripted_attempts(module, {})
    with pytest.raises(RuntimeError, match="tokenizer bug"):
        module.run_demo(generate, backend=SimpleNamespace(tok=BrokenTokenizer()))
    assert len(calls) == 1


def test_demo_unknown_benchmark_and_existing_log_fail_before_generation(tmp_path):
    module = demo_module()
    generate, calls = scripted_attempts(module, {})
    with pytest.raises(ValueError, match="No hybrid task card"):
        module.run_demo(generate, benchmark="unknown")
    path = tmp_path / "attempts.jsonl"
    path.touch()
    with pytest.raises(FileExistsError):
        module.run_demo(generate, log_path=path)
    assert not calls
