"""Generate an initial task-adaptive Card from a frozen Schema and Free-NL traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lychee_mas.hybrid_comm.backends import DeepSeekBackend  # noqa: E402
from lychee_mas.hybrid_comm.protocol import (  # noqa: E402
    card_review_prompt,
    initialization_prompt,
    load_schema,
    load_task_profile,
    materialize_card,
    write_card,
)
from lychee_mas.hybrid_comm.protocol.design import extract_json_object, load_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an initial task-adaptive Card.")
    parser.add_argument("--task-profile", required=True)
    parser.add_argument("--traces", required=True, help="Free-NL outputs.jsonl")
    parser.add_argument("--schema", default="configs/hybrid_comm/protocol_schema.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--version", required=True)
    parser.add_argument("--resume-artifact", default=None)
    parser.add_argument(
        "--review-policy", choices=["strict", "advisory"], default="strict"
    )
    args = parser.parse_args()

    schema = load_schema(args.schema)
    task_profile = load_task_profile(args.task_profile)
    traces = load_jsonl(args.traces)[: args.n]
    if len(traces) != args.n:
        raise SystemExit(f"expected {args.n} NL traces, found {len(traces)}")

    base_prompt = initialization_prompt(schema, traces, task_profile)
    prompt = base_prompt
    if args.resume_artifact:
        prior = json.loads(Path(args.resume_artifact).read_text(encoding="utf-8"))
        last = prior["attempts"][-1]
        prompt = (
            base_prompt
            + "\n\nRevise this previously generated JSON to fix its validation error. "
            "Return the complete JSON object.\nERROR:\n"
            + str(last["error"])
            + "\nPREVIOUS_JSON:\n"
            + str(last["raw_response"])
        )
    backend = DeepSeekBackend(model=args.model)
    attempts: list[dict] = []
    prompt_tokens = 0
    completion_tokens = 0
    card = None
    generated = None
    semantic_review_approved = False
    for attempt in range(6):
        review_raw = ""
        result = backend.generate_chat(
            [
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_new_tokens=4096,
        )
        prompt_tokens += result.prompt_tokens
        completion_tokens += result.completion_tokens
        try:
            generated = extract_json_object(result.text)
            card = materialize_card(
                schema,
                generated,
                version=args.version,
                task_profile=task_profile,
            )
            review_result = backend.generate_chat(
                [
                    {"role": "system", "content": "Return valid JSON only."},
                    {
                        "role": "user",
                        "content": card_review_prompt(schema, card, task_profile, traces),
                    },
                ],
                max_new_tokens=2048,
            )
            prompt_tokens += review_result.prompt_tokens
            completion_tokens += review_result.completion_tokens
            review = extract_json_object(review_result.text)
            review_raw = review_result.text
            semantic_review_approved = review.get("approved") is True
            if not semantic_review_approved and args.review_policy == "strict":
                raise ValueError(
                    "semantic reviewer rejected card: "
                    + json.dumps(review.get("findings", []), ensure_ascii=False)
                )
            attempts.append(
                {"raw_response": result.text, "review_response": review_raw, "error": None}
            )
            break
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            attempts.append(
                {"raw_response": result.text, "review_response": review_raw, "error": str(exc)}
            )
            if attempt == 5:
                failure = {
                    "task_profile": args.task_profile,
                    "task": task_profile["task"],
                    "schema": args.schema,
                    "traces": args.traces,
                    "trace_count": len(traces),
                    "output": args.out,
                    "version": args.version,
                    "model": args.model or "deepseek-chat",
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    },
                    "attempt_count": len(attempts),
                    "attempts": attempts,
                    "status": "failed",
                }
                target = Path(args.artifact)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                raise
            prompt = base_prompt + (
                "\n\nYour previous output failed validation:\n"
                f"{exc}\nRegenerate the complete JSON object and address every finding; "
                "do not change the frozen schema. Semicolons separate fields and cannot "
                "occur inside a field value.\nPREVIOUS_JSON:\n"
                f"{result.text}"
            )

    assert card is not None and generated is not None
    write_card(args.out, card)
    artifact = {
        "task_profile": args.task_profile,
        "task": task_profile["task"],
        "schema": args.schema,
        "traces": args.traces,
        "trace_count": len(traces),
        "output": args.out,
        "version": args.version,
        "model": args.model or "deepseek-chat",
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
        "attempt_count": len(attempts),
        "attempts": attempts,
        "review_policy": args.review_policy,
        "semantic_review_approved": semantic_review_approved,
    }
    target = Path(args.artifact)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

