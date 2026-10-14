"""兼容入口：在同一冻结 split 上比较 NL 与 P0-FIELDS-ONLY。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from run_hybrid_compare import _load_manifest, _run_mode


def _summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(samples)
    correct = sum(bool(item.get("correct")) for item in samples)
    total_comm = sum(int(item.get("communication_tokens", 0)) for item in samples)
    total_out = sum(int(item.get("sum_output_text_tokens", 0)) for item in samples)
    total_calls = sum(int(item.get("num_model_calls", 0)) for item in samples)
    failures = sum(item.get("run_status") != "completed" for item in samples)
    revisions = sum(bool(item.get("revision_triggered")) for item in samples)
    mean_utility = (
        sum(
            float(bool(item.get("correct")))
            + (1.0 / item["communication_tokens"] if item["communication_tokens"] > 0 else -1)
            for item in samples
        )
        / n
        if n
        else 0.0
    )
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "total_communication_tokens": total_comm,
        "avg_communication_tokens": total_comm / n if n else 0.0,
        "total_output_tokens": total_out,
        "avg_output_tokens": total_out / n if n else 0.0,
        "avg_model_calls": total_calls / n if n else 0.0,
        "run_failures": failures,
        "revision_count": revisions,
        "mean_utility": mean_utility,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/splits/gsm8k_protocol_v54_seed42.json")
    parser.add_argument("--split", default="T")
    parser.add_argument("--backend", default="api", choices=["api", "fake"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--tokenizer", default="deepseek-ai/DeepSeek-V3")
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    manifest["_path"] = args.manifest
    root = Path("runs/hybrid_comm") / f"{time.strftime('%Y%m%d_%H%M%S')}_T_nl_fields_only"
    root.mkdir(parents=True, exist_ok=False)
    conditions = (
        ("NL", "nl", "configs/hybrid_comm/protocols/protocol_p0_fields_only.yaml"),
        (
            "P0_FIELDS_ONLY",
            "hybrid",
            "configs/hybrid_comm/protocols/protocol_p0_fields_only.yaml",
        ),
    )
    report: dict[str, Any] = {
        "manifest": args.manifest,
        "split": args.split,
        "backend": args.backend,
        "model": args.model or "deepseek-chat",
        "tokenizer": args.tokenizer,
        "conditions": {},
    }
    for label, mode, protocol in conditions:
        run_dir, samples, _ = _run_mode(
            mode,
            args.backend,
            args.model,
            manifest,
            args.split,
            args.n,
            str(root),
            protocol,
            args.tokenizer,
        )
        report["conditions"][label] = {
            "protocol": protocol,
            "run_dir": run_dir,
            **_summary(samples),
        }
        (root / "comparison.partial.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (root / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
