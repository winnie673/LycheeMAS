"""Paired NL/Hybrid evaluation through the example's shared CDM/static graph executor."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RUNTIME_VERSION = "memory-compare-v4-restored"
SCORERS = {"gsm8k": "strict_numeric_decimal_v1"}


def executor():
    """Import the existing example, not a second graph/node implementation."""
    name = "lychee_memory_protocol_example"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name,
            ROOT / "examples/03_gsm8k_memory_protocol.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(path: Path, benchmark: str, split: str, limit: int = 0) -> list[dict]:
    if benchmark not in SCORERS:
        raise ValueError("unsupported batch evaluation benchmark")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    task = manifest.get("task")
    dataset = manifest.get("dataset", "")
    if task != benchmark and not (
        task is None and benchmark == "gsm8k" and dataset == "openai/gsm8k"
    ):
        raise ValueError("manifest benchmark does not match --benchmark")
    rows = manifest.get("splits", {}).get(split)
    if not isinstance(rows, list) or not rows or limit < 0:
        raise ValueError("split must be a nonempty list and limit must be nonnegative")
    ids = set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("task_id"), str)
            or not row["task_id"]
            or not isinstance(row.get("question"), str)
            or not row["question"].strip()
            or not isinstance(row.get("gold"), str)
            or not row["gold"].strip()
        ):
            raise ValueError("each case needs nonempty string task_id, question and gold")
        if row["task_id"] in ids:
            raise ValueError("duplicate task_id in split")
        ids.add(row["task_id"])
        if benchmark == "gsm8k":
            score_answer(benchmark, "0", row["gold"])  # Validate numeric gold before API calls.
    return rows[:limit] if limit else rows


def score_answer(benchmark: str, prediction: str, gold: str) -> float:
    if benchmark != "gsm8k":
        raise ValueError("unsupported batch evaluation benchmark")
    # Unlike the old framework `exact` fallback, do not strip signs or match substrings.
    numeric = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    values = [value.strip() for value in (prediction, gold)]
    grouped = r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d*)?(?:[eE][+-]?\d+)?"
    if "," in values[1] and not re.fullmatch(grouped, values[1]):
        raise ValueError("GSM8K gold has invalid thousands separators")
    if "," in values[0] and not re.fullmatch(grouped, values[0]):
        return 0.0
    values = [value.replace(",", "") for value in values]
    if not re.fullmatch(numeric, values[1]):
        raise ValueError("GSM8K gold must be a bare numeric answer")
    if not re.fullmatch(numeric, values[0]):
        return 0.0
    try:
        return float(Decimal(values[0]) == Decimal(values[1]))
    except InvalidOperation:
        return 0.0


def attempt_usage(attempts: list[dict]) -> dict:
    known = [
        row
        for row in attempts
        if row.get("usage_available", True)
        and row["status"] not in {"transport_error", "fatal_error"}
    ]
    # Fatal errors after generation (e.g. tokenizer failure) still have recorded usage.
    known += [
        row
        for row in attempts
        if row["status"] == "fatal_error" and row.get("usage_available") is True
    ]
    complete = len(known) == len(attempts)
    inputs = sum(row["prompt_tokens"] for row in known)
    outputs = sum(row["output_tokens"] for row in known)
    return {
        "calls": len(attempts),
        "usage_complete": complete,
        "known_input_tokens": inputs,
        "known_output_tokens": outputs,
        "model_input_tokens": inputs if complete else None,
        "model_output_tokens": outputs if complete else None,
        "format_failed_calls": sum(row["status"] == "validation_error" for row in attempts),
        "transport_failed_calls": sum(row["status"] == "transport_error" for row in attempts),
        "fallback_calls": sum(row["fallback"] for row in attempts),
        "sum_model_latency_s": sum(row.get("latency_s", 0) for row in attempts),
    }


def evaluate_case(
    case: dict,
    *,
    mode: str,
    benchmark: str,
    generate: Any,
    backend: Any,
    max_new_tokens: int,
    max_revision_rounds: int,
    log_path: Path,
    transport_errors: tuple = (TimeoutError, ConnectionError),
    protocol_path: Path | None = None,
) -> dict:
    from lychee_mas.core.types import Trajectory

    module = executor()
    trajectory = Trajectory(task_id=case["task_id"])
    attempts: list[dict] = []
    error = None
    status = "completed"
    started = time.perf_counter()
    try:
        module.run_demo(
            generate,
            benchmark=benchmark,
            question=case["question"],
            task_id=case["task_id"],
            mode=mode,
            max_new_tokens=max_new_tokens,
            max_revision_rounds=max_revision_rounds,
            backend=backend,
            log_path=log_path,
            attempts=attempts,
            trajectory=trajectory,
            transport_errors=transport_errors,
            protocol_path=protocol_path if mode == "hybrid" else None,
        )
    except module.TaskExecutionError as exc:
        status, error = "failed", str(exc)
    except Exception as exc:
        # Persist the partial task before stopping on configuration/programming errors.
        status, error = "fatal_error", str(exc)
        raise
    finally:
        wall_time = time.perf_counter() - started
        prediction = trajectory.final_answer.content if trajectory.final_answer else ""
        tokenizer_available = getattr(backend, "tok", None) is not None
        comm = (
            sum(message.meta["communication_tokens"] for message in trajectory.messages)
            if tokenizer_available
            else None
        )
        row = {
            "task_id": case["task_id"],
            "mode": mode,
            "benchmark": benchmark,
            "status": status,
            "error": error,
            "prediction": prediction,
            "gold": case["gold"],
            "wall_time_s": wall_time,
            "communication_tokens": comm,
            "communication_tokens_exact": tokenizer_available,
            "revision_rounds": trajectory.meta.get("revision_rounds", 0),
            "verification_decision": trajectory.meta.get("verification_decision"),
            "verified": status == "completed" and trajectory.meta.get("verified", False),
            "messages": [asdict(message) for message in trajectory.messages],
            **attempt_usage(attempts),
        }
        # Even fatal/scorer errors leave a raw task record; no fabricated success.
        with log_path.with_suffix(".task.json").open("x", encoding="utf-8") as handle:
            json.dump(module.redact(row), handle, ensure_ascii=False, indent=2)
    row["score"] = (
        score_answer(benchmark, prediction, case["gold"]) if status == "completed" else 0.0
    )
    row["is_correct"] = row["score"] == 1.0
    return module.redact(row)


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot summarize empty results")
    count = len(rows)

    def mean(key):
        values = [row[key] for row in rows]
        return sum(values) / count if all(value is not None for value in values) else None

    return {
        "n": count,
        "correct": sum(row["is_correct"] for row in rows),
        "accuracy": sum(row["is_correct"] for row in rows) / count,
        "failed_cases": sum(row["status"] != "completed" for row in rows),
        "failure_rate": sum(row["status"] != "completed" for row in rows) / count,
        "unverified_cases": sum(
            row["status"] == "completed" and not row["verified"] for row in rows
        ),
        "revision_rate": sum(row["revision_rounds"] > 0 for row in rows) / count,
        "avg_communication_tokens": mean("communication_tokens"),
        "avg_model_output_tokens": mean("model_output_tokens"),
        "avg_model_input_tokens": mean("model_input_tokens"),
        "known_output_tokens_total": sum(row["known_output_tokens"] for row in rows),
        "usage_incomplete_cases": sum(not row["usage_complete"] for row in rows),
        "avg_calls": mean("calls"),
        "avg_wall_time_s": mean("wall_time_s"),
        "avg_sum_model_latency_s": mean("sum_model_latency_s"),
        "format_failed_calls": sum(row["format_failed_calls"] for row in rows),
        "fallback_calls": sum(row["fallback_calls"] for row in rows),
        "transport_failed_calls": sum(row["transport_failed_calls"] for row in rows),
    }


def run_evaluation(
    cases: list[dict],
    out: Path,
    *,
    benchmark: str,
    generate: Any,
    backend: Any = None,
    modes: tuple = ("nl", "hybrid"),
    max_new_tokens: int = 512,
    max_revision_rounds: int = 1,
    metadata: dict | None = None,
    transport_errors: tuple = (
        TimeoutError,
        ConnectionError,
    ),
    protocol_path: Path | None = None,
) -> dict:
    if not cases or modes not in (("nl",), ("hybrid",), ("nl", "hybrid")):
        raise ValueError("need nonempty cases and nl/hybrid/both modes")
    if benchmark not in SCORERS:
        raise ValueError("unsupported batch evaluation benchmark")
    if max_new_tokens < 1 or max_revision_rounds not in (0, 1):
        raise ValueError("invalid output or revision budget")
    ids = [case["task_id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task_id")
    for case in cases:
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError("case needs nonempty question")
        if benchmark == "gsm8k":
            score_answer(benchmark, "0", case["gold"])
    module = executor()
    config_path = ROOT / module.CONFIGS[benchmark]
    memory_config = module.load_memory_config(benchmark, "hybrid", protocol_path)
    if (
        memory_config["task"] != benchmark
        or memory_config["name"] != "cdm"
        or memory_config["nl_strategy"] != "hybrid_protocol"
        or memory_config["router"] != {"name": "fixed", "channel": "nl"}
    ):
        raise ValueError("evaluation requires the configured CDM/fixed-NL task strategy")
    card_path = Path(memory_config["protocol_path"])
    from lychee_mas.methods.memory.channels.hybrid_protocol import HybridProtocolMemory

    contract = memory_config["protocol_contract"]
    card = HybridProtocolMemory(str(card_path), benchmark, contract=contract)
    for edge, _, _ in contract.architecture.edges:
        sender, receiver = edge.split("_to_")
        card.output_instructions(sender, receiver)  # Fail before any paid calls.
    out.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__),
        ROOT / "examples/03_gsm8k_memory_protocol.py",
        ROOT / "src/lychee_mas/methods/memory/channels/hybrid_protocol.py",
        ROOT / "src/lychee_mas/methods/memory/channels/nl.py",
        ROOT / "src/lychee_mas/methods/memory/managers/DualChannelMemory.py",
        ROOT / "src/lychee_mas/methods/build/lg_builder.py",
        ROOT / "src/lychee_mas/backends/openai_api_backend.py",
        ROOT / "src/lychee_mas/methods/memory/protocol/base.py",
        ROOT / "scripts/hybrid_gsm8k_spec.py",
    ]
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
    )
    packages = {}
    for package in (
        "openai",
        "PyYAML",
        "langgraph",
        "transformers",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    snapshot = {
        **(metadata or {}),
        "runtime_version": RUNTIME_VERSION,
        "benchmark": benchmark,
        "scorer": SCORERS[benchmark],
        "contract_sha256": contract.fingerprint,
        "contract": contract.as_dict(),
        "modes": modes,
        "n": len(cases),
        "task_ids": [case["task_id"] for case in cases],
        "max_new_tokens": max_new_tokens,
        "max_revision_rounds": max_revision_rounds,
        "attempt_limit_per_node": 3,
        "pair_order": "alternates_by_case_index",
        "git_sha": git.stdout.strip() if git.returncode == 0 else None,
        "git_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "packages": packages,
        "python": platform.python_version(),
        "config_text": config_path.read_text(encoding="utf-8"),
        "card_sha256": digest(card_path),
        "selected_card_path": str(card_path.resolve()),
        "config_sha256": digest(config_path),
        "card_text": card_path.read_text(encoding="utf-8"),
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in sources},
        "gold_passed_to_model": False,
        "cost_policy": "Delivered full payload incl feedback; all known model attempts; "
        "wall time includes setup, graph, model and audit; excludes scoring/final writes.",
    }
    (out / "config.json").write_text(
        json.dumps(module.redact(snapshot), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "selected_cases.json").write_text(
        json.dumps(module.redact(cases), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    results: dict[str, list[dict]] = {mode: [] for mode in modes}
    try:
        for index, case in enumerate(cases):
            order = modes if index % 2 == 0 else modes[::-1]
            for mode in order:
                mode_dir = out / mode
                mode_dir.mkdir(exist_ok=True)
                log_path = mode_dir / f"{index:05d}.attempts.jsonl"
                row = evaluate_case(
                    case,
                    mode=mode,
                    benchmark=benchmark,
                    generate=generate,
                    backend=backend,
                    max_new_tokens=max_new_tokens,
                    max_revision_rounds=max_revision_rounds,
                    log_path=log_path,
                    transport_errors=transport_errors,
                    protocol_path=protocol_path,
                )
                results[mode].append(row)
                with (mode_dir / "outputs.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(
                    f"{mode}: {index + 1}/{len(cases)} completed, status={row['status']}, "
                    f"correct={row['is_correct']}",
                    flush=True,
                )
        if (
            digest(card_path) != snapshot["card_sha256"]
            or digest(config_path) != snapshot["config_sha256"]
            or any(
                digest(ROOT / path) != expected
                for path, expected in snapshot["source_sha256"].items()
            )
        ):
            raise RuntimeError("source/config/card changed during evaluation; run is not frozen")
        comparison = {
            "status": "completed",
            "runtime_version": RUNTIME_VERSION,
            "benchmark": benchmark,
            "effect_valid": (metadata or {}).get("backend") == "api",
            "summaries": {mode: summarize(rows) for mode, rows in results.items()},
        }
        if len(modes) == 2:
            pairs: dict[str, int] = {}
            for nl, hybrid in zip(results["nl"], results["hybrid"]):
                if nl["task_id"] != hybrid["task_id"]:
                    raise ValueError("paired task order mismatch")
                key = (
                    "both_correct"
                    if nl["is_correct"] and hybrid["is_correct"]
                    else "nl_only_correct"
                    if nl["is_correct"]
                    else "hybrid_only_correct"
                    if hybrid["is_correct"]
                    else "both_incorrect"
                )
                pairs[key] = pairs.get(key, 0) + 1
            comparison["paired"] = pairs
        for mode, rows in results.items():
            (out / mode / "metrics.json").write_text(
                json.dumps(summarize(rows), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        (out / "comparison.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return comparison
    except Exception as exc:
        (out / "failure.json").write_text(
            json.dumps(
                module.redact(
                    {
                        "status": "aborted",
                        "error": str(exc),
                        "completed_by_mode": {mode: len(rows) for mode, rows in results.items()},
                    }
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=sorted(SCORERS), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="T")
    parser.add_argument("--limit", type=int, default=0, help="0 = full selected split")
    parser.add_argument("--mode", choices=["nl", "hybrid", "both"], default="both")
    parser.add_argument("--protocol-path", type=Path, default=None,
                        help="Explicit GSM8K Hybrid card; default comes from its memory config")
    parser.add_argument("--backend", choices=["fake", "api"], default="fake")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument(
        "--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument(
        "--tokenizer", default=None, help="Required for API; matching HF ID/local path"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-revision-rounds", choices=[0, 1], type=int, default=1)
    parser.add_argument("--out", type=Path, required=True, help="New directory only")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("--out must not already exist")
    cases = load_cases(args.manifest, args.benchmark, args.split, args.limit)
    # Dependency/scorer preflight before paying for any model call.
    if score_answer(args.benchmark, "1", "1") != 1.0:
        raise RuntimeError("scorer selfcheck failed")
    module = executor()
    backend, generate = None, module.scripted_generate
    errors = (TimeoutError, ConnectionError)
    if args.backend == "api":
        if not args.tokenizer:
            parser.error("API evaluation requires --tokenizer; no approximate communication counts")
        from lychee_mas.backends.openai_api_backend import OpenAICompatibleBackend
        from openai import APIConnectionError, InternalServerError, RateLimitError
        from transformers import AutoTokenizer

        backend = OpenAICompatibleBackend(
            args.model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            temperature=0,
            seed=args.seed,
        )
        backend.client = backend.client.with_options(max_retries=0)
        backend.tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
        generate = backend.generate_chat
        errors += (APIConnectionError, InternalServerError, RateLimitError)
    else:
        print("FAKE BACKEND: validates execution only, NOT benchmark effectiveness.", flush=True)
    metadata = vars(args).copy()
    metadata = {
        key: str(value) if isinstance(value, Path) else value for key, value in metadata.items()
    }
    metadata["manifest_sha256"] = digest(args.manifest)
    run_evaluation(
        cases,
        args.out,
        benchmark=args.benchmark,
        generate=generate,
        backend=backend,
        modes=("nl", "hybrid") if args.mode == "both" else (args.mode,),
        max_new_tokens=args.max_new_tokens,
        max_revision_rounds=args.max_revision_rounds,
        metadata=metadata,
        transport_errors=errors,
        protocol_path=args.protocol_path,
    )


if __name__ == "__main__":
    main()
