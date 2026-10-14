"""Recompute MATH-500 symbolic-equivalence scores from saved answers without API calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lychee_mas.eval.metrics import score  # noqa: E402


def rescore_rows(rows: list[dict]) -> tuple[list[dict], int]:
    corrected = []
    changed = 0
    for source in rows:
        row = dict(source)
        if row.get("task") != "math500":
            raise ValueError("rescore_math500_outputs requires math500 rows")
        if row.get("run_status") == "completed":
            pred = str(row.get("pred") or row.get("normalized_pred") or "")
            value = score("math500", pred, str(row.get("gold") or ""))
            new_correct = value == 1.0
            if new_correct != bool(row.get("is_correct")):
                changed += 1
            row["score"] = value
            row["correct"] = new_correct
            row["is_correct"] = new_correct
        corrected.append(row)
    return corrected, changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source = Path(args.input)
    with source.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    corrected, changed = rescore_rows(rows)
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in corrected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"input": str(source), "output": str(target), "n": len(rows),
                      "score_changes": changed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
