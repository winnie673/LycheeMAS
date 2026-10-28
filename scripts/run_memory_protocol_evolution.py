"""GSM8K D0/G1-G3/V protocol evolution through the existing CDM/static executor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from hybrid_gsm8k_design import GSM8KDesign
from hybrid_gsm8k_spec import FIELD_MEANINGS, GSM8K_PROTOCOL, GSM8K_SCHEDULE, GSM8K_SCHEMA, PSV_V1
from lychee_mas.methods.memory.protocol import ProtocolEvolution
from run_memory_protocol_compare import ROOT, attempt_usage, executor, load_cases, score_answer


def fake_proposal(messages: list[dict], max_new_tokens: int):
    """Synthetic fixture ONLY; never a model-generated or publishable initial card."""
    from lychee_mas.backends.openai_api_backend import APIGenResult

    request = json.loads(messages[1]["content"])
    if request["parent_card"] is not None:
        return APIGenResult(
            json.dumps({"decision": "KEEP", "reason": "Synthetic plumbing check."}), 0, 0, 0
        )
    card = {
        "version": "FAKE-ONLY",
        "description": "Synthetic test fixture, not induced knowledge.",
        "allow_nl_field": True,
        "message_format": {"wire_syntax": "direct", "structure": "fields_only"},
    }
    examples = {
        "planner_to_solver": ["G:sum;P:9+9;NL:Add both groups."],
        "solver_to_verifier": ["IR:9+9=18;R:18;NL:Both groups are included."],
        "verifier_to_solver": ["CHK:ACCEPT"],
    }
    for edge, allowed, required in PSV_V1.edges:
        card[edge] = {
            "allowed_fields": list(allowed),
            "required_fields": list(required),
            "field_rules": {
                "canonical_order": list(allowed),
                **{key: FIELD_MEANINGS[key] for key in allowed},
            },
            "examples": examples[edge],
        }
    return APIGenResult(
        json.dumps({"decision": "UPDATE", "reason": "Synthetic fixture.", "card": card}), 0, 0, 0
    )


def fake_reviewed_design(messages: list[dict], max_new_tokens: int):
    """Synthetic fixtures for the legacy-shaped multi-stage designer, NEVER real induction."""
    from lychee_mas.backends.openai_api_backend import APIGenResult

    prompt = messages[1]["content"]
    if "strict, read-only reviewer" in prompt:
        value = {"approved": True, "findings": []}
    elif "DIAGNOSIS ONLY" in prompt:
        value = {"case_analysis": [], "global_findings": []}
    elif "protocol-card designer" in prompt:
        examples = {
            "planner_to_solver": [
                "G:sum;P:sum=a+b",
                "K:price=2,count=3;G:cost;P:cost=price*count",
                "K:start=0,end=3;G:years;P:years=end-start;NL:Count completed years only.",
            ],
            "solver_to_verifier": [
                "IR:2+3=5;R:5",
                "IR:2*3=6;R:6",
                "IR:years=3-0=3;R:3;NL:Count completed years only.",
            ],
            "verifier_to_solver": [
                "CHK:ACCEPT",
                "CHK:REVISE;E:ARITH;REQ:Recheck addition",
                "CHK:REVISE;E:INSUFFICIENT_INFO;REQ:Provide the decisive multiplication",
            ],
        }
        values = {
            "K": "price=2,count=3",
            "G": "cost",
            "P": "cost=price*count",
            "NL": "Count full years.",
            "IR": "2*3=6",
            "R": "6",
            "CHK": "ACCEPT",
            "E": "ARITH",
            "REQ": "Recheck addition",
        }
        value = {
            "cards": {
                edge: {
                    "field_rules": {
                        field: {"usage": FIELD_MEANINGS[field], "examples": [values[field]]}
                        for field in allowed
                    },
                    "examples": examples[edge],
                }
                for edge, allowed, _ in PSV_V1.edges
            },
            "value_policy": {},
        }
    else:
        raise ValueError("unknown synthetic design phase")
    return APIGenResult(json.dumps(value), 10, 20, 0)


def make_execute(module, generate, backend, max_tokens, errors):
    """Adapt the SAME run_demo; no duplicate graph or node execution logic."""
    from lychee_mas.core.types import Trajectory

    def execute(case, mode, card, log_path):
        import time

        trajectory = Trajectory(task_id=case["task_id"])
        attempts, status, error = [], "completed", None
        started = time.perf_counter()
        try:
            module.run_demo(
                generate,
                benchmark="gsm8k",
                question=case["question"],
                task_id=case["task_id"],
                mode=mode,
                protocol_path=card,
                backend=backend,
                max_new_tokens=max_tokens,
                max_revision_rounds=1,
                attempts=attempts,
                trajectory=trajectory,
                log_path=log_path,
                transport_errors=errors,
            )
        except module.TaskExecutionError as exc:
            status, error = "failed", str(exc)
        except Exception as exc:
            status, error = "fatal_error", str(exc)
            raise
        finally:
            row = {
                "task_id": case["task_id"],
                "status": status,
                "error": error,
                "prediction": trajectory.final_answer.content if trajectory.final_answer else "",
                "messages": [asdict(message) for message in trajectory.messages],
                "communication_tokens": sum(
                    m.meta["communication_tokens"] for m in trajectory.messages
                )
                if getattr(backend, "tok", None) is not None
                else None,
                "wall_time_s": time.perf_counter() - started,
                "verified": trajectory.meta.get("verified", False),
                "revision_rounds": trajectory.meta.get("revision_rounds", 0),
                "model_calls": [
                    {
                        k: a.get(k)
                        for k in (
                            "role",
                            "attempt",
                            "status",
                            "error",
                            "raw_output",
                            "finish_reason",
                            "output_tokens",
                            "usage_available",
                        )
                    }
                    for a in attempts
                ],
                "failed_attempts": [
                    {k: a.get(k) for k in ("status", "error", "raw_output", "attempt", "role")}
                    for a in attempts
                    if a["status"] != "delivered"
                ],
                **attempt_usage(attempts),
            }
            with log_path.with_suffix(".task.json").open("x", encoding="utf-8") as handle:
                json.dump(module.redact(row), handle, ensure_ascii=False, indent=2)
        print(f"{log_path.parent.name}: {case['task_id']} {status}", flush=True)
        return module.redact(row)

    return execute


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["gsm8k"], default="gsm8k")
    parser.add_argument("--architecture", choices=["psv_v1"], default="psv_v1")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=["fake", "api"], default="fake")
    parser.add_argument("--model", default="deepseek-chat", help="Execution model")
    parser.add_argument("--proposer-model", default=None, help="Defaults to execution model")
    parser.add_argument("--proposer-base-url", default=None, help="Defaults to execution endpoint")
    parser.add_argument(
        "--proposer-api-key-env", default=None, help="Defaults to execution key env"
    )
    parser.add_argument("--executor-capability", default=None, help="Required for API induction")
    parser.add_argument(
        "--tokenizer", default=None, help="Matching execution model tokenizer, required for API"
    )
    parser.add_argument(
        "--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0=all development cases; >0 is a separately marked pilot",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--proposal-tokens", type=int, default=4096)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("--out must be a new directory")
    if min(args.max_new_tokens, args.proposal_tokens) < 1:
        parser.error("token budgets must be positive")
    # Read ONLY development partitions; no T rows are selected or sent to the algorithm.
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    splits = {
        name: load_cases(args.manifest, "gsm8k", name, args.limit)
        for name in (GSM8K_SCHEDULE.induction, *GSM8K_SCHEDULE.evolution, GSM8K_SCHEDULE.validation)
    }
    module = executor()
    backend, generate, propose = None, module.scripted_generate, fake_reviewed_design
    errors = (TimeoutError, ConnectionError)
    if args.backend == "api":
        if not args.tokenizer or not args.executor_capability:
            parser.error("API requires --tokenizer and --executor-capability")
        from lychee_mas.backends.openai_api_backend import OpenAICompatibleBackend
        from openai import APIConnectionError, InternalServerError, RateLimitError
        from transformers import AutoTokenizer

        common = dict(
            base_url=args.base_url, api_key_env=args.api_key_env, temperature=0, seed=args.seed
        )
        backend = OpenAICompatibleBackend(args.model, **common)
        backend.client = backend.client.with_options(max_retries=0)
        backend.tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
        proposer_options = {
            **common,
            "base_url": args.proposer_base_url or args.base_url,
            "api_key_env": args.proposer_api_key_env or args.api_key_env,
        }
        proposer = OpenAICompatibleBackend(args.proposer_model or args.model, **proposer_options)
        proposer.client = proposer.client.with_options(max_retries=0)
        generate, propose = backend.generate_chat, proposer.generate_chat
        errors += (APIConnectionError, InternalServerError, RateLimitError)
    else:
        print(
            "FAKE: synthetic cards and outputs; NOT protocol quality or benchmark evidence.",
            flush=True,
        )
    sources = [
        Path(__file__),
        ROOT / "examples/03_gsm8k_memory_protocol.py",
        ROOT / "scripts/run_memory_protocol_compare.py",
        ROOT / "scripts/hybrid_gsm8k_spec.py",
        ROOT / "scripts/hybrid_gsm8k_design.py",
        ROOT / "configs/memory/gsm8k_protocol_schema.yaml",
        *sorted((ROOT / "src/lychee_mas/methods/memory").rglob("*.py")),
        ROOT / "src/lychee_mas/backends/openai_api_backend.py",
        ROOT / "src/lychee_mas/methods/build/lg_builder.py",
        ROOT / "configs/memory/gsm8k_hybrid_protocol.yaml",
    ]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    packages = {}
    for name in ("PyYAML", "langgraph", "openai", "transformers"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    )
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(
        manifest_sha256=manifest_hash,
        source_sha256=hashes,
        python=platform.python_version(),
        packages=packages,
        git_sha=git.stdout.strip(),
        git_dirty=bool(dirty.stdout.strip()),
        pilot=args.limit > 0,
        gold_passed_to_executor=False,
    )
    algorithm = ProtocolEvolution(
        design=GSM8KDesign(schema=GSM8K_SCHEMA),
        contract=GSM8K_PROTOCOL,
        schedule=GSM8K_SCHEDULE,
        generate=propose,
        execute=make_execute(module, generate, backend, args.max_new_tokens, errors),
        score=lambda prediction, gold: score_answer("gsm8k", prediction, gold),
        executor_capability=args.executor_capability
        or "Synthetic scripted backend; no reasoning ability.",
        effect_valid=args.backend == "api",
        sanitize=module.redact,
        proposal_tokens=args.proposal_tokens,
        transport_errors=errors,
    )

    def verify_sources():
        if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != manifest_hash or any(
            hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected
            for path, expected in hashes.items()
        ):
            raise RuntimeError("source/manifest changed; result is invalid")

    result = algorithm.run(splits, args.out, metadata=metadata, verify_sources=verify_sources)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
