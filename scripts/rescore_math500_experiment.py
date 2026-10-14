"""Rescore saved MATH-500 runs and audit frozen-card selection without inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rescore_math500_outputs import rescore_rows
from run_hybrid_compare import _summarize
from select_evolved_card import candidate_wins, summarize


def _read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="runs/hybrid_comm/math500_v2")
    parser.add_argument(
        "--initialization-run",
        default=(
            "runs/hybrid_comm/math500_v2/initialization/"
            "20261007_213432_api_nl_MATH500-P0-FIELDS-ONLY_seed42"
        ),
    )
    args = parser.parse_args()
    root = Path(args.run_root)
    state = json.loads((root / "pipeline_state.json").read_text(encoding="utf-8"))
    runs = {"D0_nl": args.initialization_run, **state["runs"]}
    report: dict = {
        "scorer": "math500_v2",
        "manifest_hash": state["manifest_hash"],
        "runs": {},
        "validation_selections": {},
    }
    for key, directory in runs.items():
        source = Path(directory) / "outputs.jsonl"
        original = _read_rows(source)
        rescored, changes = rescore_rows(original)
        target = source.with_name("outputs_math500_rescored.jsonl")
        with target.open("w", encoding="utf-8") as handle:
            for row in rescored:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        flips = [
            {
                "task_id": before["task_id"],
                "gold": before["gold"],
                "pred": before["pred"],
                "old_score": before["score"],
                "new_score": after["score"],
            }
            for before, after in zip(original, rescored)
            if bool(before["is_correct"]) != bool(after["is_correct"])
        ]
        assert len(flips) == changes
        report["runs"][key] = {
            "source": str(source),
            "rescored": str(target),
            "n": len(rescored),
            "old_correct": sum(bool(row["is_correct"]) for row in original),
            "new_correct": sum(bool(row["is_correct"]) for row in rescored),
            "score_changes": changes,
            "flips": flips,
        }

    for round_no in (2, 3):
        incumbent_key = "card0_v" if round_no == 2 else "card2_candidate_v"
        candidate_key = f"card{round_no}_candidate_v"
        incumbent = summarize(report["runs"][incumbent_key]["rescored"])
        candidate = summarize(report["runs"][candidate_key]["rescored"])
        wins, criterion = candidate_wins(incumbent, candidate)
        report["validation_selections"][f"round_{round_no}"] = {
            "winner": "candidate" if wins else "incumbent",
            "winner_card": (
                state["rounds"][f"round_{round_no}"]["candidate_card"]
                if wins else state["rounds"][f"round_{round_no}"]["input_card"]
            ),
            "criterion": criterion,
            "incumbent": {k: incumbent[k] for k in ("correct", "accuracy", "mean_utility")},
            "candidate": {k: candidate[k] for k in ("correct", "accuracy", "mean_utility")},
        }

    nl = _read_rows(Path(report["runs"]["test_nl"]["rescored"]))
    hybrid = _read_rows(Path(report["runs"]["test_hybrid"]["rescored"]))
    if len(nl) != len(hybrid) or any(
        a["task_id"] != b["task_id"] or a["gold"] != b["gold"]
        for a, b in zip(nl, hybrid)
    ):
        raise ValueError("final-test NL and Hybrid outputs are not paired")
    comparison = {
        "task": "math500",
        "split": "T",
        "scorer": "math500_v2",
        "manifest_hash": state["manifest_hash"],
        "final_card": state["final_card"],
        "selection_note": (
            "These saved T outputs belong to the originally frozen Card_3. "
            "The corrected V selection would retain Card_2; its T results were not saved."
        ),
        "nl_summary": _summarize(nl),
        "hybrid_summary": _summarize(hybrid),
        "paired_outcomes": {
            "both_correct": sum(a["is_correct"] and b["is_correct"] for a, b in zip(nl, hybrid)),
            "nl_only_correct": sum(
                a["is_correct"] and not b["is_correct"] for a, b in zip(nl, hybrid)
            ),
            "hybrid_only_correct": sum(
                not a["is_correct"] and b["is_correct"] for a, b in zip(nl, hybrid)
            ),
            "both_incorrect": sum(
                not a["is_correct"] and not b["is_correct"] for a, b in zip(nl, hybrid)
            ),
        },
    }
    comparison_path = root / "final_test" / "comparison_math500_rescored.json"
    _write_json(comparison_path, comparison)
    report["comparison"] = str(comparison_path)
    report_path = root / "rescoring" / "scorer_v2_report.json"
    _write_json(report_path, report)
    print(json.dumps({
        "report": str(report_path),
        "comparison": str(comparison_path),
        "score_changes": sum(run["score_changes"] for run in report["runs"].values()),
        "validation_selections": report["validation_selections"],
        "nl_summary": comparison["nl_summary"],
        "hybrid_summary": comparison["hybrid_summary"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
