"""Generate Card_0 from the frozen Schema and Free-NL GSM8K traces."""

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
    materialize_card,
    write_card,
)
from lychee_mas.hybrid_comm.protocol.design import extract_json_object, load_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GSM8K Card_0 from NL traces.")
    parser.add_argument("--traces", required=True, help="Free-NL outputs.jsonl")
    parser.add_argument("--schema", default="configs/hybrid_comm/protocol_schema.yaml")
    parser.add_argument(
        "--out",
        default="configs/hybrid_comm/generated/gsm8k/initial/card_0_gsm8k.yaml",
    )
    parser.add_argument("--artifact", default="runs/hybrid_comm/card_0_generation.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--version", default="CARD-0-GSM8K")
    args = parser.parse_args()

    schema = load_schema(args.schema)
    traces = load_jsonl(args.traces)[: args.n]
    if len(traces) != args.n:
        raise SystemExit(f"expected {args.n} NL traces, found {len(traces)}")
    prompt = initialization_prompt(schema, traces)
    backend = DeepSeekBackend(model=args.model)
    attempts = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    card = None
    generated = None
    for attempt in range(3):
        review_raw = ""
        result = backend.generate_chat(
            [
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_new_tokens=4096,
        )
        total_prompt_tokens += result.prompt_tokens
        total_completion_tokens += result.completion_tokens
        try:
            generated = extract_json_object(result.text)
            card = materialize_card(schema, generated, version=args.version)
            review_result = backend.generate_chat(
                [
                    {"role": "system", "content": "Return valid JSON only."},
                    {"role": "user", "content": card_review_prompt(schema, card)},
                ],
                max_new_tokens=2048,
            )
            total_prompt_tokens += review_result.prompt_tokens
            total_completion_tokens += review_result.completion_tokens
            review = extract_json_object(review_result.text)
            review_raw = review_result.text
            if review.get("approved") is not True:
                raise ValueError(
                    "semantic reviewer rejected card: "
                    + json.dumps(review.get("findings", []), ensure_ascii=False)
                )
            attempts.append(
                {
                    "raw_response": result.text,
                    "review_response": review_result.text,
                    "error": None,
                }
            )
            break
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            attempts.append(
                {
                    "raw_response": result.text,
                    "review_response": review_raw,
                    "error": str(exc),
                }
            )
            if attempt == 2:
                raise
            prompt += (
                "\n\nYour previous output failed deterministic validation:\n"
                f"{exc}\nRegenerate the complete JSON object and address every finding; "
                "do not change the frozen schema. Remember that semicolons separate fields "
                "and can never occur inside a field value."
            )
    assert card is not None and generated is not None
    write_card(args.out, card)
    artifact = {
        "schema": args.schema,
        "traces": args.traces,
        "trace_count": len(traces),
        "output": args.out,
        "version": args.version,
        "model": args.model or "deepseek-chat",
        "usage": {"prompt_tokens": total_prompt_tokens,
                  "completion_tokens": total_completion_tokens},
        "attempt_count": len(attempts),
        "attempts": attempts,
        "raw_response": result.text,
    }
    target = Path(args.artifact)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
