"""Evaluate a protocol on a split and return a compact summary."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

_MANIFEST_PATH = "data/splits/gsm8k_protocol_v54_seed42.json"


def _load_manifest(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _result_to_sample(result: Any) -> dict[str, Any]:
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


class ProtocolEvaluator:
    """Run a protocol on a data split and produce a summary."""

    def __init__(self, manifest_path: str | Path | None = None):
        self.manifest_path = Path(manifest_path or _MANIFEST_PATH)

    def evaluate(
        self,
        protocol_path: str | Path,
        split: str,
        backend_name: str = "fake",
        model: str | None = None,
        n: int | None = None,
        run_root: str | Path = "runs/hybrid_comm",
        mode: str = "hybrid",
    ) -> dict[str, Any]:
        """Run ``protocol_path`` on ``split`` in the given mode and return a summary dict.

        Summary keys: ``n``, ``correct``, ``accuracy``, ``avg_communication_tokens``,
        ``avg_output_tokens``, ``avg_model_calls``, ``run_dir``.
        """
        from lychee_mas.eval.metrics import aggregate_samples, write_results
        from lychee_mas.hybrid_comm.agents import RolePromptBuilder
        from lychee_mas.hybrid_comm.backends import DeepSeekBackend, FakeBackend
        from lychee_mas.hybrid_comm.metrics import TokenCounter
        from lychee_mas.hybrid_comm.pipeline import HybridRunner
        from lychee_mas.hybrid_comm.protocol import load_protocol_from_yaml
        from lychee_mas.hybrid_comm.types import CommunicationMode

        protocol_path = Path(protocol_path)
        if not protocol_path.exists():
            raise FileNotFoundError(f"protocol not found: {protocol_path}")
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {self.manifest_path}")

        manifest = _load_manifest(self.manifest_path)
        manifest["_path"] = str(self.manifest_path)

        suite = load_protocol_from_yaml(str(protocol_path))
        tokenizer_name = (
            "deepseek-ai/DeepSeek-V3"
            if backend_name == "api"
            else None
        )
        backend: Any = FakeBackend() if backend_name == "fake" else DeepSeekBackend(model=model)
        prompt_builder = RolePromptBuilder()
        token_counter = TokenCounter(tokenizer_name=tokenizer_name)

        cases = manifest["splits"][split]
        if n:
            cases = cases[:n]

        stamp = time.strftime("%Y%m%d_%H%M%S")
        seed = manifest.get("seed", 42)
        run_dir = (
            Path(run_root)
            / f"{stamp}_{backend_name}_{mode}_{suite.version}_seed{seed}_{split}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        runner = HybridRunner(
            backend=backend,
            prompt_builder=prompt_builder,
            token_counter=token_counter,
            suite=suite,
            mode=CommunicationMode(mode),
        )

        samples: list[dict[str, Any]] = []
        for i, case in enumerate(cases):
            result = runner.run_one(
                task_id=case["task_id"],
                question=case["question"],
                gold=case["gold"],
            )
            samples.append(_result_to_sample(result))

        metrics = aggregate_samples(
            samples,
            run_info={
                "model": model or backend_name,
                "backend_provider": backend_name,
                "method": f"hybrid_comm_{mode}",
                "task": "gsm8k",
                "protocol_version": suite.version,
                "protocol_hash": suite.meta.get("protocol_hash", ""),
            },
        )

        config = {
            "backend": backend_name,
            "mode": mode,
            "protocol": str(protocol_path),
            "protocol_version": suite.version,
            "protocol_hash": suite.meta.get("protocol_hash", ""),
            "manifest": str(manifest.get("_path", "unknown")),
            "split": split,
            "n": len(cases),
            "tokenizer": tokenizer_name,
            "tokenizer_approximate": token_counter.is_approximate(),
        }
        write_results(str(run_dir), samples, metrics, config)

        return self._summarize(samples, str(run_dir), str(protocol_path))

    @staticmethod
    def _summarize(
        samples: list[dict[str, Any]], run_dir: str, protocol_path: str
    ) -> dict[str, Any]:
        n = len(samples)
        correct = sum(1 for s in samples if s.get("score", 0) == 1.0)
        return {
            "n": n,
            "correct": correct,
            "accuracy": round(correct / n, 4) if n else 0.0,
            "avg_communication_tokens": round(
                sum(s["communication_tokens"] for s in samples) / n, 1
            )
            if n
            else 0.0,
            "avg_output_tokens": round(
                sum(s["sum_output_text_tokens"] for s in samples) / n, 1
            )
            if n
            else 0.0,
            "avg_model_calls": round(
                sum(s["num_model_calls"] for s in samples) / n, 2
            )
            if n
            else 0.0,
            "run_dir": run_dir,
            "protocol_path": protocol_path,
        }
