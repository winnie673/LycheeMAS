"""hybrid_comm 单题/批量运行入口（fake 或 API 后端）。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _load_manifest(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _make_backend(backend_name: str, model: str | None = None) -> Any:
    from lychee_mas.hybrid_comm.backends import DeepSeekBackend, FakeBackend

    if backend_name == "fake":
        return FakeBackend()
    if backend_name == "api":
        return DeepSeekBackend(model=model)
    raise ValueError(f"unknown backend {backend_name!r}; choose fake or api")


def _make_run_dir(root: str, backend: str, mode: str, protocol_id: str, seed: int) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    d = Path(root) / f"{stamp}_{backend}_{mode}_{protocol_id}_seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hybrid_comm on GSM8K manifest.")
    parser.add_argument("--backend", choices=["fake", "api"], default="fake")
    parser.add_argument("--mode", choices=["nl", "hybrid"], default="hybrid")
    parser.add_argument(
        "--protocol", default="configs/hybrid_comm/protocols/protocol_p0_fields_only.yaml"
    )
    parser.add_argument("--manifest", default="data/splits/gsm8k_protocol_seed42.json")
    parser.add_argument("--split", default="D0", help="Which manifest split to run.")
    parser.add_argument("--n", type=int, default=None, help="Number of cases to run.")
    parser.add_argument("--run-root", default="runs/hybrid_comm")
    parser.add_argument("--model", default=None, help="API model identifier override.")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "Tokenizer name or local path for accurate token counting "
            "(default: whitespace approximate)."
        ),
    )
    args = parser.parse_args()

    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from lychee_mas.eval.metrics import aggregate_samples, write_results
    from lychee_mas.hybrid_comm.agents import RolePromptBuilder
    from lychee_mas.hybrid_comm.metrics import TokenCounter
    from lychee_mas.hybrid_comm.pipeline import HybridRunner
    from lychee_mas.hybrid_comm.protocol import load_protocol_from_yaml
    from lychee_mas.hybrid_comm.types import CommunicationMode

    suite = load_protocol_from_yaml(args.protocol)
    mode = CommunicationMode(args.mode)
    backend = _make_backend(args.backend, model=args.model)
    prompt_builder = RolePromptBuilder()
    tokenizer_name = args.tokenizer
    if tokenizer_name is None and args.backend == "api":
        tokenizer_name = "deepseek-ai/DeepSeek-V3"
    token_counter = TokenCounter(tokenizer_name=tokenizer_name)

    manifest = _load_manifest(args.manifest)
    cases = manifest["splits"][args.split]
    if args.n:
        cases = cases[: args.n]

    run_dir = _make_run_dir(
        args.run_root, args.backend, args.mode, suite.version, manifest.get("seed", 42)
    )
    print(f"Run directory: {run_dir}")

    runner = HybridRunner(
        backend=backend,
        prompt_builder=prompt_builder,
        token_counter=token_counter,
        suite=suite,
        mode=mode,
    )

    samples: list[dict[str, Any]] = []
    for i, case in enumerate(cases):
        print(f"[{i + 1}/{len(cases)}] {case['task_id']}")
        result = runner.run_one(
            task_id=case["task_id"],
            question=case["question"],
            gold=case["gold"],
        )
        samples.append(_result_to_sample(result))

    metrics = aggregate_samples(
        samples,
        run_info={
            "model": args.model or args.backend,
            "backend_provider": args.backend,
            "method": f"hybrid_comm_{args.mode}",
            "task": "gsm8k",
            "scorer_kind": "exact",
            "protocol_version": suite.version,
            "protocol_hash": suite.meta.get("protocol_hash", ""),
        },
    )

    config = {
        "backend": args.backend,
        "mode": args.mode,
        "protocol": args.protocol,
        "protocol_version": suite.version,
        "protocol_hash": suite.meta.get("protocol_hash", ""),
        "manifest": args.manifest,
        "split": args.split,
        "n": len(cases),
        "git_sha": _git_sha(),
        "tokenizer": tokenizer_name,
        "tokenizer_approximate": token_counter.is_approximate(),
    }

    write_results(run_dir, samples, metrics, config)
    print(f"Wrote {len(samples)} samples to {run_dir}")
    summary_keys = ("num_cases", "accuracy", "mean_model_calls_per_case")
    print(json.dumps({k: metrics[k] for k in summary_keys if k in metrics}, indent=2))


def _result_to_sample(result: Any) -> dict[str, Any]:
    """把 CaseResult 转成 outputs.jsonl 的一行。"""
    return {
        "task_id": result.task_id,
        "task": "gsm8k",
        "mode": str(result.mode),
        "protocol_version": result.protocol_version,
        "question": result.question,
        "gold": result.gold,
        "pred": result.final_answer,
        "normalized_pred": result.normalized_final_answer,
        "score": result.score,
        "correct": result.score == 1.0,
        "is_correct": result.score == 1.0,
        "run_status": result.run_status,
        "failed_stage": result.failed_stage,
        "failure_reason": result.failure_reason,
        "decision": str(result.decision) if result.decision else None,
        "error_type": str(result.error_type) if result.error_type else None,
        "revision_triggered": result.revision_triggered,
        "model_call_count": len(result.model_calls),
        "num_model_calls": len(result.model_calls),
        "sum_input_text_tokens": result.total_input_tokens,
        "sum_output_text_tokens": result.total_output_tokens,
        "sum_model_latency_s": result.total_latency_s,
        "communication_tokens": result.communication_tokens,
        "messages": [
            {
                "edge": m.edge,
                "payload_format": m.payload_format,
                "fallback": m.fallback,
                "payload": m.payload,
                "raw_payload": m.payload,
                "fields": (m.call.parsed or {}).get("fields", {}),
                "sender": m.sender,
                "receiver": m.receiver,
            }
            for m in result.messages
        ],
    }


def _git_sha() -> str | None:
    import subprocess

    try:
        repo_root = Path(__file__).resolve().parents[1]
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root)
            .decode()
            .strip()
        )
    except Exception:
        return None


if __name__ == "__main__":
    main()
