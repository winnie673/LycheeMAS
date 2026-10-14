"""Select an evolved Card on a fixed validation set using acc + 1/token."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def task_utility(sample: dict) -> float:
    tokens = int(sample.get("communication_tokens", 0))
    if sample.get("run_status") != "completed" or tokens <= 0:
        return 0.0
    return float(bool(sample.get("is_correct"))) + 1 / tokens


def summarize(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"empty validation outputs: {path}")
    for row in rows:
        if row.get("task") == "math500" and row.get("run_status") == "completed":
            from lychee_mas.eval.metrics import score

            corrected = score(
                "math500", str(row.get("normalized_pred") or row.get("pred") or ""),
                str(row.get("gold") or ""),
            )
            row["score"] = corrected
            row["is_correct"] = corrected == 1.0
    n = len(rows)
    correct = sum(bool(row.get("is_correct")) for row in rows)
    comm = sum(int(row.get("communication_tokens", 0)) for row in rows)
    return {
        "outputs": path,
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "mean_communication_tokens": comm / n,
        "mean_utility": sum(task_utility(row) for row in rows) / n,
        "failed": sum(row.get("run_status") != "completed" for row in rows),
        "task_ids": [str(row.get("task_id", "")) for row in rows],
    }


def candidate_wins(incumbent: dict, candidate: dict) -> tuple[bool, str]:
    if candidate["n"] != incumbent["n"]:
        raise ValueError("validation runs have different case counts")
    if "task_ids" in candidate and "task_ids" in incumbent:
        if candidate["task_ids"] != incumbent["task_ids"]:
            raise ValueError("validation runs contain different or reordered task IDs")
    if candidate["mean_utility"] != incumbent["mean_utility"]:
        return candidate["mean_utility"] > incumbent["mean_utility"], "mean_utility"
    if candidate["accuracy"] != incumbent["accuracy"]:
        return candidate["accuracy"] > incumbent["accuracy"], "accuracy"
    if candidate["mean_communication_tokens"] != incumbent["mean_communication_tokens"]:
        return (
            candidate["mean_communication_tokens"]
            < incumbent["mean_communication_tokens"],
            "mean_communication_tokens",
        )
    return False, "stable_incumbent_tiebreak"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incumbent-card", required=True)
    parser.add_argument("--incumbent-outputs", required=True)
    parser.add_argument("--candidate-card", required=True)
    parser.add_argument("--candidate-outputs", required=True)
    parser.add_argument("--winner-card", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    incumbent = summarize(args.incumbent_outputs)
    candidate = summarize(args.candidate_outputs)
    wins, criterion = candidate_wins(incumbent, candidate)
    source = args.candidate_card if wins else args.incumbent_card
    target = Path(args.winner_card)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    record = {
        "winner": "candidate" if wins else "incumbent",
        "criterion": criterion,
        "winner_source": source,
        "winner_card": args.winner_card,
        "incumbent": incumbent,
        "candidate": candidate,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = (
        "n", "correct", "accuracy", "mean_communication_tokens", "mean_utility", "failed"
    )
    concise = {
        "winner": record["winner"],
        "criterion": criterion,
        "incumbent": {key: incumbent[key] for key in fields},
        "candidate": {key: candidate[key] for key in fields},
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
