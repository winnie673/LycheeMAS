"""Summarize paired MATH-500 NL/Hybrid outputs after symbolic-equivalence rescoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_hybrid_compare import _summarize  # noqa: E402


def _read(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nl", required=True)
    parser.add_argument("--hybrid", required=True)
    parser.add_argument("--card", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    nl = _read(args.nl)
    hybrid = _read(args.hybrid)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    expected = manifest["splits"]["T"]
    ids = [case["task_id"] for case in expected]
    if [row["task_id"] for row in nl] != ids:
        raise ValueError("NL outputs do not match frozen T task order")
    if [row["task_id"] for row in hybrid] != ids:
        raise ValueError("Hybrid outputs do not match frozen T task order")
    if any(a["gold"] != b["gold"] for a, b in zip(nl, hybrid)):
        raise ValueError("paired runs contain different gold answers")

    paired = {
        "both_correct": sum(a["is_correct"] and b["is_correct"] for a, b in zip(nl, hybrid)),
        "nl_only_correct": sum(a["is_correct"] and not b["is_correct"] for a, b in zip(nl, hybrid)),
        "hybrid_only_correct": sum(
            not a["is_correct"] and b["is_correct"] for a, b in zip(nl, hybrid)
        ),
        "both_incorrect": sum(
            not a["is_correct"] and not b["is_correct"] for a, b in zip(nl, hybrid)
        ),
    }
    comparison = {
        "task": "math500",
        "split": "T",
        "n": len(ids),
        "manifest_hash": manifest["manifest_hash"],
        "scorer_kind": "math500",
        "scoring_note": (
            "Saved predictions rescored using the task-specific strict "
            "MATH-500 equivalence scorer."
        ),
        "final_card": args.card,
        "nl_outputs": args.nl,
        "hybrid_outputs": args.hybrid,
        "nl_summary": _summarize(nl),
        "hybrid_summary": _summarize(hybrid),
        "paired_outcomes": paired,
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
