"""一键对比 NL 与 Hybrid 在相同数据上的指标。"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def _load_manifest(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _run_mode(
    mode: str,
    backend_name: str,
    model: str | None,
    manifest: dict[str, Any],
    split: str,
    n: int | None,
    run_root: str,
    protocol_path: str,
    tokenizer_name: str | None,
    workers: int = 1,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """跑一个模式，返回 (run_dir, samples, metrics)。"""
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))

    from lychee_mas.eval.metrics import aggregate_samples, write_results
    from lychee_mas.hybrid_comm.agents import RolePromptBuilder
    from lychee_mas.hybrid_comm.backends import DeepSeekBackend, FakeBackend
    from lychee_mas.hybrid_comm.metrics import TokenCounter
    from lychee_mas.hybrid_comm.pipeline import HybridRunner
    from lychee_mas.hybrid_comm.protocol import load_protocol_from_yaml
    from lychee_mas.hybrid_comm.types import CommunicationMode

    suite = load_protocol_from_yaml(protocol_path)
    task_name = manifest.get("task", "gsm8k")
    scorer_kind = manifest.get("scorer_kind", "exact")
    mode_enum = CommunicationMode(mode)
    backend: Any = FakeBackend() if backend_name == "fake" else DeepSeekBackend(model=model)
    prompt_builder = RolePromptBuilder(task_name=task_name)
    token_counter = TokenCounter(tokenizer_name=tokenizer_name)
    token_counter.count("warmup")
    if token_counter.is_approximate():
        print(f"[{mode}] 警告：使用近似 token 计数（tokenizer={tokenizer_name}）")
    else:
        print(f"[{mode}] 使用 tokenizer：{tokenizer_name}")

    cases = manifest["splits"][split]
    if n:
        cases = cases[:n]

    stamp = time.strftime("%Y%m%d_%H%M%S")
    seed = manifest.get("seed", 42)
    run_dir = Path(run_root) / f"{stamp}_{backend_name}_{mode}_{suite.version}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[{mode}] Run directory: {run_dir}")

    runner = HybridRunner(
        backend=backend,
        prompt_builder=prompt_builder,
        token_counter=token_counter,
        suite=suite,
        mode=mode_enum,
        scorer_kind=scorer_kind,
        role_output_limits={"planner": 512, "solver": 512, "verifier": 512},
    )

    if workers < 1:
        raise ValueError("workers must be >= 1")
    if workers > 1 and backend_name == "api":
        backend._get_client()

    def run_case(case: dict[str, Any]) -> dict[str, Any]:
        result = runner.run_one(
            task_id=case["task_id"],
            question=case["question"],
            gold=case["gold"],
        )
        return _result_to_sample(result, task_name)

    samples: list[dict[str, Any]] = []
    output_path = run_dir / "outputs.jsonl"
    with output_path.open("w", encoding="utf-8") as output_file:
        if workers == 1:
            rows = map(run_case, cases)
        else:
            pool = ThreadPoolExecutor(max_workers=workers)
            rows = pool.map(run_case, cases)
        try:
            for i, sample in enumerate(rows):
                samples.append(sample)
                output_file.write(json.dumps(sample, ensure_ascii=False) + "\n")
                output_file.flush()
                print(f"[{mode}] [{i + 1}/{len(cases)}] {sample['task_id']}", flush=True)
        finally:
            if workers > 1:
                pool.shutdown(wait=True)

    metrics = aggregate_samples(
        samples,
        run_info={
            "model": model or backend_name,
            "backend_provider": backend_name,
            "method": f"hybrid_comm_{mode}",
            "task": task_name,
            "scorer_kind": scorer_kind,
            "protocol_version": suite.version,
            "protocol_hash": suite.meta.get("protocol_hash", ""),
        },
    )

    config = {
        "backend": backend_name,
        "mode": mode,
        "protocol": protocol_path,
        "protocol_version": suite.version,
        "protocol_hash": suite.meta.get("protocol_hash", ""),
        "manifest": str(Path(manifest.get("_path", "unknown"))),
        "split": split,
        "n": len(cases),
        "tokenizer": tokenizer_name,
        "tokenizer_approximate": token_counter.is_approximate(),
        "task": task_name,
        "scorer_kind": scorer_kind,
        "workers": workers,
    }

    write_results(str(run_dir), samples, metrics, config)
    print(f"[{mode}] Wrote {len(samples)} samples to {run_dir}")
    return str(run_dir), samples, metrics


def _result_to_sample(result: Any, task_name: str = "gsm8k") -> dict[str, Any]:
    return {
        "task_id": result.task_id,
        "task": task_name,
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
        "sum_static_input_tokens": result.static_input_tokens,
        "sum_dynamic_input_tokens": result.dynamic_input_tokens,
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


def _summarize(samples: list[dict[str, Any]]) -> dict[str, float]:
    n = len(samples)
    avg_out = round(sum(s["sum_output_text_tokens"] for s in samples) / n, 1)
    avg_comm = round(sum(s["communication_tokens"] for s in samples) / n, 1)
    correct = sum(1 for s in samples if s.get("score", 0) == 1.0)
    return {
        "n": n,
        "correct": correct,
        "accuracy": round(correct / n, 6),
        "avg_calls": round(sum(s["num_model_calls"] for s in samples) / n, 2),
        "avg_out": avg_out,
        "avg_comm": avg_comm,
        "avg_comm_out": round(avg_comm + avg_out, 1),
        "avg_latency": round(sum(s["sum_model_latency_s"] for s in samples) / n, 3),
        "failed_cases": sum(1 for s in samples if s.get("run_status") == "failed"),
        "revision_cases": sum(1 for s in samples if s.get("revision_triggered")),
        "mean_utility": round(sum(_task_utility(s) for s in samples) / n, 6),
    }


def _task_utility(sample: dict[str, Any]) -> float:
    communication_tokens = int(sample.get("communication_tokens", 0))
    if sample.get("run_status") != "completed" or communication_tokens <= 0:
        return 0.0
    return float(sample.get("score", 0)) + 1 / communication_tokens


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run NL and Hybrid on the same split and compare metrics."
    )
    parser.add_argument("--backend", choices=["fake", "api"], default="api")
    parser.add_argument(
        "--protocol", default="configs/hybrid_comm/protocols/protocol_p0_fields_only.yaml"
    )
    parser.add_argument("--manifest", default="data/splits/synthetic_seed42.json")
    parser.add_argument("--split", default="D0")
    parser.add_argument("--n", "--num-samples", type=int, default=None, dest="n")
    parser.add_argument("--run-root", default="runs/hybrid_comm")
    parser.add_argument("--model", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["nl", "hybrid"],
        default=["nl", "hybrid"],
        help="需要运行的通信模式；默认依次运行 nl 和 hybrid。",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer 名称或本地路径（默认 api 后端使用 deepseek-ai/DeepSeek-V3）",
    )
    args = parser.parse_args()

    tokenizer_name = args.tokenizer
    if tokenizer_name is None and args.backend == "api":
        tokenizer_name = "deepseek-ai/DeepSeek-V3"

    manifest = _load_manifest(args.manifest)
    manifest["_path"] = args.manifest  # for config snapshot

    results: dict[str, tuple[str, list[dict[str, Any]], dict[str, Any]]] = {}
    for mode in args.modes:
        run_dir, samples, metrics = _run_mode(
            mode=mode,
            backend_name=args.backend,
            model=args.model,
            manifest=manifest,
            split=args.split,
            n=args.n,
            run_root=args.run_root,
            protocol_path=args.protocol,
            tokenizer_name=tokenizer_name,
            workers=args.workers,
        )
        results[mode] = (run_dir, samples, metrics)

    if set(results) != {"nl", "hybrid"}:
        return

    nl_summary = _summarize(results["nl"][1])
    hy_summary = _summarize(results["hybrid"][1])

    print("\n" + "=" * 60)
    print("NL vs Hybrid 对比")
    print("=" * 60)
    print(f"{'指标':<20} {'NL':>12} {'Hybrid':>12} {'变化':>12}")
    print("-" * 60)
    for key, label in (
        ("correct", "正确题数"),
        ("accuracy", "准确率"),
        ("avg_calls", "平均调用次数"),
        ("avg_out", "平均输出 tokens"),
        ("avg_comm", "平均通信 tokens"),
        ("avg_comm_out", "平均通信+输出 tokens"),
        ("avg_latency", "平均 latency(s)"),
        ("failed_cases", "失败案例数"),
        ("revision_cases", "触发修订数"),
        ("mean_utility", "平均 acc+1/token"),
    ):
        nl_v = nl_summary[key]
        hy_v = hy_summary[key]
        if isinstance(nl_v, float):
            delta = f"{hy_v - nl_v:+.1f}"
            nl_s = f"{nl_v:.1f}"
            hy_s = f"{hy_v:.1f}"
        else:
            delta = f"{hy_v - nl_v:+d}"
            nl_s = str(nl_v)
            hy_s = str(hy_v)
        print(f"{label:<20} {nl_s:>12} {hy_s:>12} {delta:>12}")

    # 保存对比结果
    comparison = {
        "nl_run_dir": results["nl"][0],
        "hybrid_run_dir": results["hybrid"][0],
        "nl_summary": nl_summary,
        "hybrid_summary": hy_summary,
    }
    parent_dir = os.path.commonpath([results["nl"][0], results["hybrid"][0]])
    comp_path = Path(parent_dir) / "comparison.json"
    with open(comp_path, "w", encoding="utf-8") as fh:
        json.dump(comparison, fh, ensure_ascii=False, indent=2)
    print(f"\n对比结果已保存: {comp_path}")


if __name__ == "__main__":
    main()
