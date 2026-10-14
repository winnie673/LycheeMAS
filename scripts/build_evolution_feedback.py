"""Build deterministic categorized feedback from paired NL/Hybrid outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lychee_mas.hybrid_comm.protocol import classify_feedback  # noqa: E402
from lychee_mas.hybrid_comm.protocol.design import load_jsonl  # noqa: E402

LIMITS = {
    "run_failed": 4,
    "hybrid_incorrect_nl_correct": 6,
    "both_incorrect": 2,
    "hybrid_correct_nl_incorrect": 2,
    "hybrid_correct_high_cost": 4,
    "revision_success": 2,
}


def _utility(sample: dict) -> float:
    tokens = int(sample.get("communication_tokens", 0))
    if sample.get("run_status") != "completed" or tokens <= 0:
        return 0.0
    return float(bool(sample.get("is_correct"))) + 1 / tokens


def _summary(samples: list[dict]) -> dict:
    n = len(samples)
    return {
        "n": n,
        "correct": sum(bool(item.get("is_correct")) for item in samples),
        "accuracy": sum(bool(item.get("is_correct")) for item in samples) / n if n else 0.0,
        "failed": sum(item.get("run_status") != "completed" for item in samples),
        "mean_communication_tokens": (
            sum(int(item.get("communication_tokens", 0)) for item in samples) / n if n else 0.0
        ),
        "mean_utility": sum(_utility(item) for item in samples) / n if n else 0.0,
    }


def _select(groups: dict[str, list[dict]]) -> dict[str, list[dict]]:
    selected = {}
    for label, cases in groups.items():
        if label == "hybrid_correct_high_cost":
            cases = sorted(
                cases,
                key=lambda pair: int(pair["hybrid"].get("communication_tokens", 0)),
                reverse=True,
            )
        selected[label] = cases[: LIMITS.get(label, 6)]
    return {label: cases for label, cases in selected.items() if cases}


def main() -> None:
    parser = argparse.ArgumentParser(description="Categorize paired evolution feedback.")
    parser.add_argument("--nl", required=True)
    parser.add_argument("--hybrid", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    nl = load_jsonl(args.nl)
    hybrid = load_jsonl(args.hybrid)
    groups = classify_feedback(nl, hybrid)
    selected = _select(groups)
    payload = {
        "batch_metrics": {"nl": _summary(nl), "hybrid": _summary(hybrid)},
        "category_counts": {key: len(value) for key, value in groups.items()},
        "selection_limits": LIMITS,
        "cases": selected,
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: len(value) for key, value in selected.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
