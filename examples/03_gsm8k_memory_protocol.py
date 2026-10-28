"""Use the registered CDM strategy with the existing static graph/node_factory."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypedDict

from lychee_mas import REGISTRY
from lychee_mas.backends.openai_api_backend import APIGenResult
from lychee_mas.core.types import AgentSpec, Answer, Message, Trajectory
from lychee_mas.methods.memory.channels.hybrid_protocol import ProtocolValidationError
from lychee_mas.plugins import build_langgraph

CONFIGS = {"gsm8k": "configs/memory/gsm8k_hybrid_protocol.yaml"}
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
QUESTIONS = {"gsm8k": "What is 9+9?"}
RUNTIME_VERSION = "memory-compare-v5-schema-derived"


class TaskExecutionError(RuntimeError):
    """A node exhausted its bounded recoverable attempts; not a programming error."""


def nl_instructions(role: str) -> str:
    common = "Communicate in concise natural language. Do not use JSON or Hybrid field labels. "
    if role == "planner":
        return common + "State the goal and a useful plan, without a final answer."
    if role == "solver":
        return common + (
            r"Explain decisive evidence, then end with exactly one \boxed{answer}. "
            "The box contains only the exact requested mathematical answer, never reasoning."
        )
    return common + (
        "Start with ACCEPT on its own line if correct, or REVISE on its own line followed "
        "by an actionable explanation of the error. Do not supply a replacement answer."
    )


def nl_fields(payload: str, role: str) -> dict[str, str]:
    """Control/answer extraction only; natural-language wire text stays unchanged."""
    if not payload.strip():
        raise ProtocolValidationError("empty natural-language message")
    if role == "planner":
        return {}
    if role == "verifier":
        lines = payload.strip().splitlines()
        if lines[0] not in {"ACCEPT", "REVISE"}:
            raise ProtocolValidationError(
                "verifier must start with ACCEPT or REVISE on its own line"
            )
        if lines[0] == "REVISE" and not "\n".join(lines[1:]).strip():
            raise ProtocolValidationError("REVISE requires actionable feedback")
        return {"CHK": lines[0]}
    markers = list(re.finditer(r"\\boxed\s*\{", payload))
    if len(markers) != 1:
        raise ProtocolValidationError("solver must end with exactly one boxed final answer")
    start = markers[0].end()
    depth = 1
    for end in range(start, len(payload)):
        if payload[end] in "{}" and (end == 0 or payload[end - 1] != "\\"):
            depth += 1 if payload[end] == "{" else -1
        if depth == 0:
            answer = payload[start:end].strip()
            suffix = payload[end + 1 :].strip().removesuffix(".").removesuffix("。").strip()
            prefix = payload[: markers[0].start()].rstrip()
            paired_math_close = any(
                prefix.endswith(opening)
                and suffix == closing
                and (opening != "$" or not prefix.endswith(("$$", r"\$")))
                for opening, closing in ((r"\[", r"\]"), (r"\(", r"\)"), ("$$", "$$"), ("$", "$"))
            )
            if not answer or (suffix and not paired_math_close):
                break
            return {"R": answer}
    raise ProtocolValidationError("boxed answer must be nonempty, balanced and at the end")


def redact(value: Any) -> Any:
    """Redact credentials before persisting records, never modify live model output."""
    if isinstance(value, str):
        for name, secret in os.environ.items():
            if len(secret) >= 8 and any(
                k in name.upper() for k in ("KEY", "TOKEN", "PASSWORD", "SECRET")
            ):
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def fallback_instructions(instructions: str, role: str) -> str:
    """Preserve the original Hybrid runner's minimal system prompt verbatim."""
    headers = dict(
        line.split(": ", 1)
        for line in instructions.splitlines()
        if line.startswith(("Allowed fields: ", "Required fields: "))
    )
    allowed, required = headers["Allowed fields"], headers["Required fields"]
    result = (
        f"role={role}\n\nOutput only FIELD:value segments separated by semicolons. "
        f"Allowed fields: {allowed}. Required fields: {required}. "
        "Do not output JSON, markdown, or separate natural language. "
        "Verifier must output CHK:ACCEPT when no correction is needed."
    )
    if role == "solver":
        result += (
            " R contains only the final answer. Put derivation in IR when available, "
            "never swap R with IR or NL."
        )
    if "NL" in allowed.split(", "):
        result += (
            " NL is only an optional one-sentence interpretation or ambiguity, "
            "not an answer slot. Omit NL when unused, never output NL:none."
        )
    return result


class DemoState(TypedDict):
    question: str
    outputs: dict[str, str]
    memory: Any
    trajectory: Trajectory


def load_memory_config(benchmark: str, mode: str, protocol_path: Path | None = None) -> dict:
    import yaml
    from hybrid_gsm8k_spec import GSM8K_PROTOCOL, protocol_contract_for_card

    if benchmark not in CONFIGS or mode not in {"nl", "hybrid"}:
        raise ValueError("unknown benchmark or communication mode")
    with (ROOT / CONFIGS[benchmark]).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["protocol_path"] = str(ROOT / config["protocol_path"])
    if protocol_path is not None:
        if benchmark != "gsm8k" or mode != "hybrid":
            raise ValueError("explicit card selection is supported for GSM8K Hybrid only")
        config["protocol_path"] = str(Path(protocol_path).resolve())
    config["protocol_contract"] = (
        protocol_contract_for_card(config["protocol_path"]) if mode == "hybrid" else GSM8K_PROTOCOL
    )
    if mode == "nl":
        config["nl_strategy"] = "prev_output"
    return config


def scripted_generate(messages: list[dict[str, str]], max_new_tokens: int = 256) -> APIGenResult:
    """Offline callback with the same result contract as the existing API backend."""
    system = messages[0]["content"]
    if "Communication mode: nl;" in system:
        role = system.split("You are the ", 1)[1].split(" ", 1)[0]
        answer = "18"
        text = {
            "planner": "Use the stated equation to compute the requested quantity.",
            "solver": "Using the equation gives the requested value. \\boxed{" + answer + "}",
            "verifier": "ACCEPT",
        }[role]
        return APIGenResult(text, 10, 20, 0.0)
    payloads = {
        "planner_to_solver": "G:sum;P:sum=9+9",
        "solver_to_verifier": "IR:9+9=18;R:18",
        "verifier_to_solver": "CHK:ACCEPT",
    }
    for edge, payload in payloads.items():
        if f"edge: {edge};" in system:
            return APIGenResult(payload, 10, 20, 0.0)
    if system.startswith("role="):
        role = system.splitlines()[0].split("=", 1)[1]
        edge = dict(
            planner="planner_to_solver", solver="solver_to_verifier", verifier="verifier_to_solver"
        )[role]
        return APIGenResult(payloads[edge], 10, 20, 0.0)
    raise ValueError("missing role output instructions")


def run_demo(
    generate: Callable[..., APIGenResult] = scripted_generate,
    *,
    benchmark: str = "gsm8k",
    question: str | None = None,
    task_id: str = "demo",
    backend: Any = None,
    max_new_tokens: int = 512,
    attempts: list[dict] | None = None,
    log_path: Path | None = None,
    transport_errors: tuple[type[Exception], ...] = (TimeoutError, ConnectionError),
    mode: str = "hybrid",
    max_revision_rounds: int = 1,
    trajectory: Trajectory | None = None,
    protocol_path: Path | None = None,
    architecture: str = "psv_v1",
) -> Trajectory:
    from hybrid_gsm8k_spec import PSV_V1

    if architecture != PSV_V1.name:
        raise ValueError("this executor only implements psv_v1")
    if benchmark not in CONFIGS:
        raise ValueError(f"No hybrid task card configured for {benchmark!r}")
    if mode not in {"nl", "hybrid"} or max_revision_rounds not in (0, 1):
        raise ValueError("mode must be nl/hybrid and max_revision_rounds must be 0 or 1")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("x", encoding="utf-8"):
            pass  # Refuse accidental overwrite before any model call.
    config = load_memory_config(benchmark, mode, protocol_path)
    router_config = dict(config.pop("router"))
    router = REGISTRY.create("memory_router", router_config.pop("name"), **router_config)
    manager_name = config.pop("name")
    # Model loading and generation remain in the caller/backend, never in the channel.
    memory = REGISTRY.create("memory_manager", manager_name, backend=backend, **config)
    memory.nl.check_architecture(PSV_V1)
    memory.reset()
    records = attempts if attempts is not None else []
    trajectory = trajectory if trajectory is not None else Trajectory(task_id=task_id)
    if trajectory.messages or trajectory.final_answer is not None:
        raise ValueError("trajectory must be empty for a new task")
    trajectory.meta.update(
        {
            "benchmark": benchmark,
            "mode": mode,
            "runtime_version": RUNTIME_VERSION,
            "max_new_tokens": max_new_tokens,
            "attempt_limit_per_node": 3,
            "attempts": records,
            "max_revision_rounds": max_revision_rounds,
            "revision_rounds": 0,
            "architecture": architecture,
            "architecture_sha256": PSV_V1.fingerprint,
            "protocol_path": config["protocol_path"] if mode == "hybrid" else None,
        }
    )
    # Snapshots are obtained through CDM recall after delivery, never from raw model output.
    # Keep historical rendered bundles so revision can revisit a plan with a latest-only channel.
    snapshots: dict[str, dict[str, str]] = {}

    def node_factory(spec: AgentSpec, is_terminal: bool):
        def node(state: DemoState) -> dict:
            round_index = spec.meta["round"]
            if round_index and trajectory.meta.get("verification_decision") != "REVISE":
                return {}
            if spec.role == "solver" and round_index:
                trajectory.meta["revision_rounds"] = round_index
            manager = state["memory"]
            from lychee_mas.methods.memory.routing.base import RouterInputs

            decision = router.decide(RouterInputs(role=spec.role, task=benchmark, turn=round_index))
            roles = (
                []
                if spec.role == "planner"
                else ["solver"]
                if spec.role == "verifier"
                else ["planner", "solver", "verifier"]
                if round_index
                else ["planner"]
            )
            incoming = [snapshots[role] for role in roles]
            context = "\n\n".join(item["text"] for item in incoming)
            instructions = (
                manager.nl.output_instructions(spec.role, spec.meta["receiver"])
                if mode == "hybrid"
                else nl_instructions(spec.role)
            )
            if round_index:
                instructions += "\nThis is semantic revision/reverification. Address the feedback."
            messages = [
                {
                    "role": "system",
                    "content": spec.system_prompt
                    + f"\nCommunication mode: {mode};\n"
                    + instructions,
                },
                {"role": "user", "content": state["question"] + "\n" + context},
            ]
            prompt_tokens = output_tokens = 0
            last_error = ""
            used_fallback = False
            for attempt in range(3):
                if used_fallback:
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                fallback_instructions(instructions, spec.role)
                                if mode == "hybrid"
                                else spec.system_prompt
                                + f"\nCommunication mode: {mode};\n"
                                + nl_instructions(spec.role)
                            ),
                        },
                        {
                            "role": "user",
                            "content": state["question"]
                            + "\n"
                            + context
                            + "\nFallback reason: "
                            + last_error,
                        },
                    ]
                record = {
                    "role": spec.role,
                    "attempt": attempt,
                    "fallback": used_fallback,
                    "input_messages": messages,
                    "raw_output": "",
                    "error": "",
                    "status": "pending",
                    "prompt_tokens": 0,
                    "output_tokens": 0,
                    "finish_reason": "unavailable",
                    "suspected_truncation": False,
                    "round": round_index,
                    "mode": mode,
                    "incoming_message_id": incoming[-1]["id"] if incoming else None,
                    "incoming_message_ids": [item["id"] for item in incoming],
                }
                fatal_error = None
                call_started = time.perf_counter()
                try:
                    response = generate(messages, max_new_tokens=max_new_tokens)
                    record["raw_output"] = getattr(response, "text", "")
                    if (
                        not isinstance(getattr(response, "text", None), str)
                        or not all(
                            isinstance(getattr(response, name, None), int)
                            and getattr(response, name) >= 0
                            for name in ("n_prompt_pos", "n_gen_tokens")
                        )
                        or not isinstance(getattr(response, "latency_s", None), (int, float))
                    ):
                        raise TypeError("backend result violates the text/usage/latency contract")
                except transport_errors as exc:
                    record.update(status="transport_error", error=str(exc), usage_available=False)
                except Exception as exc:
                    # Audit but never retry unexpected backend/programming errors.
                    record.update(status="fatal_error", error=str(exc), usage_available=False)
                    fatal_error = exc
                else:
                    prompt_tokens += response.n_prompt_pos
                    output_tokens += response.n_gen_tokens
                    finish = getattr(response, "finish_reason", "") or "unavailable"
                    record.update(
                        raw_output=response.text,
                        prompt_tokens=response.n_prompt_pos,
                        output_tokens=response.n_gen_tokens,
                        finish_reason=finish,
                        usage_available=(
                            not getattr(response, "usage_unknown", False)
                            and response.n_prompt_pos > 0
                            and response.n_gen_tokens > 0
                        ),
                        latency_s=response.latency_s,
                    )
                    try:
                        # Source behavior: finish_reason is audited, syntax decides validity.
                        record["suspected_truncation"] = finish == "length"
                        record["length_budget_reached"] = response.n_gen_tokens >= max_new_tokens
                        if not isinstance(response.text, str) or not response.text.strip():
                            raise ProtocolValidationError("empty model output")
                        message = Message(
                            sender=spec.role,
                            receiver=spec.meta["receiver"],
                            content=response.text.strip(),
                            prompt_tokens=prompt_tokens,
                            completion_tokens=output_tokens,
                            round=round_index,
                        )
                        if mode == "nl":
                            fields = nl_fields(response.text, spec.role)
                            tok = getattr(backend, "tok", None)
                            tokens = (
                                len(tok.encode(response.text.strip(), add_special_tokens=False))
                                if tok is not None
                                else None
                            )
                            message.meta.update(
                                fields=fields,
                                communication_tokens=tokens,
                                communication_tokens_exact=tokens is not None,
                            )
                        manager.observe([message])
                    except ProtocolValidationError as exc:
                        record.update(status="validation_error", error=str(exc))
                    except Exception as exc:
                        # In particular, tokenizer failures are not wire-format failures.
                        record.update(status="fatal_error", error=str(exc))
                        fatal_error = exc
                    else:
                        record["status"] = "delivered"
                record["call_wall_time_s"] = time.perf_counter() - call_started
                safe = redact(record)
                records.append(safe)
                if log_path is not None:
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(safe, ensure_ascii=False) + "\n")
                if fatal_error is not None:
                    raise fatal_error
                if record["status"] == "delivered":
                    break
                last_error = safe["error"]
                if record["status"] == "validation_error" and attempt == 1:
                    used_fallback = True
                elif attempt < 2:
                    messages = [*messages]
                    if record["status"] == "validation_error":
                        messages.append({"role": "assistant", "content": record["raw_output"]})
                        target = (
                            "valid fields-only syntax"
                            if mode == "hybrid"
                            else "valid natural-language format"
                        )
                        error_text = (
                            f"Output did not pass validation: {last_error}. "
                            f"Please fix and output {target}."
                        )
                    else:
                        error_text = last_error
                    messages.append(
                        {
                            "role": "user",
                            "content": f"[format retry {attempt + 1}] {error_text}",
                        }
                    )
            else:
                raise TaskExecutionError(f"{spec.role} exhausted 3 attempts: {last_error}")
            message.meta.update(manager.nl.metadata())
            # prev_output expects the previous payload as recall input, not the problem text.
            bundle = manager.recall(
                decision, message.content if mode == "nl" else state["question"]
            )
            snapshots[spec.role] = {"text": bundle.NL_Channel or "", "id": message.id}
            message.meta.update(
                attempt_count=attempt + 1,
                fallback=used_fallback,
                tokens_include_all_attempts=True,
                incoming_message_ids=[item["id"] for item in incoming],
            )
            state["trajectory"].add(message)
            if "R" in message.meta["fields"]:
                state["trajectory"].final_answer = Answer(
                    content=message.meta["fields"]["R"],
                    source=spec.name,
                )
            if "CHK" in message.meta["fields"]:
                from hybrid_gsm8k_spec import verification_decision

                state["trajectory"].meta["verification_decision"] = verification_decision(
                    message.meta["fields"]["CHK"]
                )
            return {"outputs": {**state["outputs"], spec.name: message.content}}

        return node

    agents = [
        AgentSpec(
            name=role,
            role=role,
            system_prompt=f"You are the {role} for {benchmark}. " + dict(PSV_V1.roles)[role],
            meta={"receiver": receiver, "round": round_index},
        )
        for role, receiver, round_index in [
            ("planner", "solver", 0),
            ("solver", "verifier", 0),
            ("verifier", "solver", 0),
            *(
                [("solver", "verifier", 1), ("verifier", "solver", 1)]
                if max_revision_rounds
                else []
            ),
        ]
    ]
    for spec in agents:
        if spec.meta["round"]:
            spec.name += "_revision"
    graph = build_langgraph(
        method="static", node_factory=node_factory, state_schema=DemoState, agents=agents
    )
    started = time.perf_counter()
    try:
        graph.compile().invoke(
            {
                "question": question or QUESTIONS[benchmark],
                "outputs": {},
                "memory": memory,
                "trajectory": trajectory,
            }
        )
    finally:
        trajectory.meta["execution_wall_time_s"] = time.perf_counter() - started
    trajectory.meta["verified"] = trajectory.meta.get("verification_decision") == "ACCEPT"
    return trajectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=sorted(CONFIGS), default="gsm8k")
    parser.add_argument("--backend", choices=["fake", "api"], default="fake")
    parser.add_argument("--mode", choices=["nl", "hybrid"], default="hybrid")
    parser.add_argument("--protocol-path", type=Path, default=None)
    parser.add_argument("--architecture", choices=["psv_v1"], default="psv_v1")
    parser.add_argument("--max-revision-rounds", choices=[0, 1], type=int, default=1)
    parser.add_argument("--question", default=None)
    parser.add_argument("--task-id", default="synthetic-smoke")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--tokenizer", default=None, help="Exact tokenizer; no approximation")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    backend = None
    generate = scripted_generate
    errors = (TimeoutError, ConnectionError)
    if args.output and (
        args.output.exists() or args.output.with_suffix(".attempts.jsonl").exists()
    ):
        raise FileExistsError("refusing to overwrite existing smoke artifacts")
    if args.backend == "api":
        from lychee_mas.backends.openai_api_backend import OpenAICompatibleBackend
        from openai import APIConnectionError, InternalServerError, RateLimitError

        backend = OpenAICompatibleBackend(
            args.model,
            api_key_env="DEEPSEEK_API_KEY",
            temperature=0,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        backend.client = backend.client.with_options(max_retries=0)
        generate = backend.generate_chat
        errors += (APIConnectionError, InternalServerError, RateLimitError)
    if args.tokenizer:
        if backend is None:
            raise ValueError("--tokenizer requires --backend api")
        from transformers import AutoTokenizer

        backend.tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    records: list[dict] = []
    try:
        result = run_demo(
            generate,
            benchmark=args.benchmark,
            question=args.question,
            task_id=args.task_id,
            backend=backend,
            max_new_tokens=args.max_new_tokens,
            attempts=records,
            log_path=args.output.with_suffix(".attempts.jsonl") if args.output else None,
            transport_errors=errors,
            mode=args.mode,
            max_revision_rounds=args.max_revision_rounds,
            protocol_path=args.protocol_path,
            architecture=args.architecture,
        )
    except (RuntimeError, ProtocolValidationError, *errors) as exc:
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                json.dump(
                    redact({"status": "failed", "error": str(exc), "attempts": records}),
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
        raise
    if args.output:
        report = {
            "status": "completed",
            "backend": args.backend,
            "model": args.model,
            "question": args.question or QUESTIONS[args.benchmark],
            "trajectory": asdict(result),
            "model_usage_tokens": result.total_tokens,
            "communication_tokens": (
                sum(m.meta["communication_tokens"] for m in result.messages)
                if all(m.meta["communication_tokens_exact"] for m in result.messages)
                else None
            ),
        }
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(redact(report), handle, ensure_ascii=False, indent=2)
    print(f"answer={result.final_answer.content}, messages={len(result.messages)}")
    for message in result.messages:
        print(redact(f"{message.sender}->{message.receiver}: {message.content}"))
    print("Model usage:", result.total_tokens)
    print("Verification:", result.meta["verification_decision"])
    print("Communication tokens:", [m.meta["communication_tokens"] for m in result.messages])


if __name__ == "__main__":
    main()
