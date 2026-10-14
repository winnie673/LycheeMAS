"""Run one evidence-bound diagnosis/update round for a GSM8K Card."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lychee_mas.hybrid_comm.backends import DeepSeekBackend  # noqa: E402
from lychee_mas.hybrid_comm.protocol import (  # noqa: E402
    apply_evolution_changes,
    diagnosis_prompt,
    evolution_prompt,
    load_schema,
    write_card,
)
from lychee_mas.hybrid_comm.protocol.design import extract_json_object  # noqa: E402
from lychee_mas.hybrid_comm.protocol.loader import _load_yaml  # noqa: E402


def _call(backend, prompt):
    return backend.generate_chat(
        [{"role": "system", "content": "Return valid JSON only."},
         {"role": "user", "content": prompt}],
        max_new_tokens=4096,
    )


def _feedback_case_ids(feedback: dict) -> set[str]:
    case_ids: set[str] = set()
    for cases in feedback.get("cases", feedback).values():
        if not isinstance(cases, list):
            continue
        for case in cases:
            if not isinstance(case, dict):
                continue
            task_id = case.get("task_id")
            if task_id is None and isinstance(case.get("hybrid"), dict):
                task_id = case["hybrid"].get("task_id")
            if task_id is not None:
                case_ids.add(str(task_id))
    return case_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose and evolve one GSM8K Card.")
    parser.add_argument("--card", required=True)
    parser.add_argument("--feedback", required=True)
    parser.add_argument("--version", required=True, help="Candidate version, e.g. CARD-1-GSM8K")
    parser.add_argument("--out", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--schema", default="configs/hybrid_comm/protocol_schema.yaml")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    schema = load_schema(args.schema)
    current = _load_yaml(args.card)
    feedback = json.loads(Path(args.feedback).read_text(encoding="utf-8"))
    backend = DeepSeekBackend(model=args.model)
    diagnosed = _call(backend, diagnosis_prompt(current, feedback))
    diagnosis = extract_json_object(diagnosed.text)
    known_case_ids = _feedback_case_ids(feedback)
    diagnosed_case_ids = {
        str(item.get("case_id"))
        for item in diagnosis.get("case_analysis", [])
        if isinstance(item, dict)
        and item.get("communication_related") is True
        and str(item.get("case_id")) in known_case_ids
    }
    if not diagnosed_case_ids and diagnosis.get("case_analysis"):
        raise ValueError("diagnosis has no evidence-bound communication-related case")

    attempts = []
    prompt = evolution_prompt(schema, current, diagnosis, feedback)
    candidate = None
    decision = None
    changed_paths: list[str] = []
    revised_tokens = 0
    for attempt in range(3):
        revised = _call(backend, prompt)
        revised_tokens += revised.completion_tokens
        try:
            decision = extract_json_object(revised.text)
            action = decision.get("action")
            if action == "UPDATE":
                candidate, changed_paths = apply_evolution_changes(
                    schema,
                    current,
                    decision,
                    version=args.version,
                    diagnosed_case_ids=diagnosed_case_ids,
                )
            elif action != "KEEP":
                raise ValueError("evolver action must be KEEP or UPDATE")
            attempts.append({"raw_response": revised.text, "error": None})
            break
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            attempts.append({"raw_response": revised.text, "error": str(exc)})
            if attempt == 2:
                raise
            prompt += (
                f"\n\nYour previous output failed validation: {exc}\n"
                "Return the complete decision again without changing the frozen Schema."
            )

    assert decision is not None
    action = decision["action"]
    if action == "UPDATE":
        assert candidate is not None
        write_card(args.out, candidate)
    else:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.card, args.out)
    artifact = {
        "action": action,
        "diagnosis": diagnosis,
        "diagnosis_raw": diagnosed.text,
        "decision": decision,
        "decision_attempts": attempts,
        "changed_paths": changed_paths,
        "input_card": args.card,
        "feedback": args.feedback,
        "output_card": args.out,
        "usage": {
            "diagnosis_prompt_tokens": diagnosed.prompt_tokens,
            "diagnosis_completion_tokens": diagnosed.completion_tokens,
            "evolution_completion_tokens": revised_tokens,
        },
    }
    target = Path(args.artifact)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{action}: {args.out}")


if __name__ == "__main__":
    main()
