"""Channel contract tests using registered CDM and the existing core.Message."""
from __future__ import annotations

import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from lychee_mas import REGISTRY
from lychee_mas.core.types import Message
from lychee_mas.methods.memory.channels.nl import PREV_OUTPUT_HEADER, NLMemory
from lychee_mas.methods.memory.routing.base import RouteDecision

CARD = "configs/memory/task_cards/gsm8k_card2.yaml"
PAYLOAD = "G:sum;P:sum=9+9;NL:Count both groups."


class Tokenizer:
    def encode(self, text: str, add_special_tokens: bool) -> list[str]:
        assert add_special_tokens is False
        return list(text)


def manager(tokenizer=None):
    return REGISTRY.create(
        "memory_manager", "cdm", backend=SimpleNamespace(tok=tokenizer),
        nl_strategy="hybrid_protocol", protocol_path=CARD,
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
    assert "Rate or ratio values must carry their per-unit semantics" in instructions
    assert "Every answer-changing relation" in instructions
    assert "Example:" in instructions
    deliver(memory)
    meta = memory.nl.metadata()
    assert meta["protocol_version"] == "CARD-2-GSM8K-NL-FIELD-V1"
    assert meta["protocol_hash"] == (
        "e7aac3350d7a898d74ff30f82dbbfeeeb7bc22e3e053545cdfe9543969c33b17"
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
    ("CHK:FAIL;E:ARITH;REQ:x", "verifier", "solver"),
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
    wrong_task = NLMemory(strategy="hybrid_protocol", protocol_path=CARD, task="math500")
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
