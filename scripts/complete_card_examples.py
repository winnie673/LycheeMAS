"""Complete full-message examples without changing an existing Card's rules."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lychee_mas.hybrid_comm.backends import DeepSeekBackend  # noqa: E402
from lychee_mas.hybrid_comm.protocol import (  # noqa: E402
    load_protocol_from_yaml,
    load_schema,
    write_card,
)
from lychee_mas.hybrid_comm.protocol.design import (  # noqa: E402
    EDGES,
    _parse_example,
    _validate_example,
    extract_json_object,
    load_jsonl,
)
from lychee_mas.hybrid_comm.protocol.loader import _load_yaml  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use an LLM to complete full examples while preserving Card rules."
    )
    parser.add_argument("--base-card", required=True)
    parser.add_argument("--traces", required=True)
    parser.add_argument("--schema", default="configs/hybrid_comm/protocol_schema.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()

    schema = load_schema(args.schema)
    base = _load_yaml(args.base_card)
    traces = load_jsonl(args.traces)[: args.n]
    missing_counts = {edge: 3 - len(base[edge].get("examples", [])) for edge in EDGES}
    prompt = f"""You add missing examples to an existing GSM8K protocol Card.

The Card rules, fields, field order, required fields, and wire syntax are frozen. Do not
rewrite them. Preserve every existing example verbatim. Generate ONLY the missing additional
examples: planner_to_solver={missing_counts['planner_to_solver']},
solver_to_verifier={missing_counts['solver_to_verifier']}, and
verifier_to_solver={missing_counts['verifier_to_solver']}.
Use the Free-NL traces as evidence of communication functions, not text to imitate.

Every example must be internally coherent: all fields in one message must refer to the same
problem and calculation. A C or U value must directly qualify the K/G/P or IR/R in that same
message. Never splice fields from unrelated cases. Every addition must differ from every
existing example and from every other addition; do not repeat one scenario with an optional
field appended. The new verifier REVISE example must use a different E category and failure
mode from the existing ARITH example.

Planner examples must cover: G/P only with shared facts omitted; necessary selected K; and a
meaningful C or U case. K contains explicit givens only; all derivation belongs in symbolic P.
Solver examples must cover minimal symbolic IR/R and a meaningful C or U case. R is a finite
decimal without a unit. At least one planner and solver example omits C and U. Verifier
examples must be exactly one CHK:ACCEPT plus two distinct valid REVISE messages using the
frozen E enum and an actionable REQ. Optional fields are omitted when unused.

Return JSON only in this exact shape, containing additions only:
{{"examples": {{"planner_to_solver": ["...", "..."],
"solver_to_verifier": ["...", "..."],
"verifier_to_solver": ["..."]}}}}

FROZEN_SCHEMA:
{json.dumps(schema, ensure_ascii=False)}

BASE_CARD:
{json.dumps(base, ensure_ascii=False)}

FREE_NL_TRACES:
{json.dumps(traces, ensure_ascii=False)}
"""
    backend = DeepSeekBackend(model=args.model)
    attempts: list[dict[str, object]] = []
    completed = None
    review = None
    for attempt in range(6):
        result = backend.generate_chat(
            [
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_new_tokens=2048,
        )
        review_raw = ""
        try:
            generated = extract_json_object(result.text)
            examples = generated.get("examples")
            if not isinstance(examples, dict) or set(examples) != set(EDGES):
                raise ValueError("examples must contain exactly the three frozen edges")
            candidate = copy.deepcopy(base)
            candidate["version"] = args.version
            candidate["description"] = (
                f"LLM-generated GSM8K protocol card {args.version}, based on "
                f"{base['version']} with completed full-message examples."
            )
            for edge in EDGES:
                values = examples[edge]
                if not isinstance(values, list) or len(values) != missing_counts[edge]:
                    raise ValueError(
                        f"{edge}: exactly {missing_counts[edge]} additions are required"
                    )
                allowed = set(schema[edge]["allowed_fields"])
                for value in values:
                    _validate_example(edge, str(value), allowed)
                combined = [*base[edge].get("examples", []), *(str(value) for value in values)]
                if len(set(combined)) != len(combined):
                    raise ValueError(f"{edge}: full examples must be distinct")
                parsed = [_parse_example(str(value)) for value in combined]
                if edge in {"planner_to_solver", "solver_to_verifier"}:
                    signatures = {
                        tuple((key, value) for key, value in item.items() if key not in {"C", "U"})
                        for item in parsed
                    }
                    if len(signatures) != len(parsed):
                        raise ValueError(
                            f"{edge}: examples must use distinct scenarios, "
                            "not optional-field variants"
                        )
                if edge == "verifier_to_solver":
                    revise_errors = [
                        item.get("E") for item in parsed if item.get("CHK") == "REVISE"
                    ]
                    if len(revise_errors) != len(set(revise_errors)):
                        raise ValueError(
                            "verifier_to_solver: REVISE examples need distinct E categories"
                        )
                if edge == "solver_to_verifier" and any(
                    not item.get("R", "").replace("-", "", 1).replace(".", "", 1).isdigit()
                    for item in parsed
                ):
                    raise ValueError("solver R examples must be finite decimals without units")
                candidate[edge]["examples"] = combined
            review_prompt = f"""You are a strict reviewer of newly generated full-message
examples for a frozen GSM8K protocol Card. Review ONLY the examples. The field rules are
frozen context and must never be a finding.

Reject an example if its fields describe different problems; C/U is unrelated to the same
message's K/G/P or IR/R; K contains a derived computation; P is procedural prose instead of
compact symbolic relations; fields duplicate one another; optional fields are placeholders;
R is not a finite decimal without a unit; or verifier control fields violate the Schema.
Also reject an edge set that does not cover the distinct situations requested by the rules.
All agents see the original problem, so G/P-only planner examples are valid even when numeric
givens occur directly in P. A P expression may reference variables introduced by K; that is
the purpose of K and is not duplication. Duplication means restating the same binding or
semantic assertion in two fields.

Return JSON only as {{"approved":true,"findings":[]}} or
{{"approved":false,"findings":[{{"path":"...","evidence":"...","fix":"..."}}]}}.

FROZEN_SCHEMA:{json.dumps(schema, ensure_ascii=False)}
FROZEN_RULES:{json.dumps({edge: base[edge]['field_rules'] for edge in EDGES}, ensure_ascii=False)}
ALL_EXAMPLES:{json.dumps({edge: candidate[edge]['examples'] for edge in EDGES}, ensure_ascii=False)}
"""
            review_result = backend.generate_chat(
                [
                    {"role": "system", "content": "Return valid JSON only."},
                    {"role": "user", "content": review_prompt},
                ],
                max_new_tokens=2048,
            )
            review_raw = review_result.text
            review = extract_json_object(review_raw)
            findings = review.get("findings", [])
            only_non_findings = (
                bool(findings)
                and all(isinstance(item, dict) for item in findings)
                and all(
                "no change needed" in str(item.get("fix", "")).lower()
                and "valid" in str(item.get("fix", "")).lower()
                for item in findings
                )
            )
            if review.get("approved") is not True and not only_non_findings:
                raise ValueError(
                    "semantic reviewer rejected examples: "
                    + json.dumps(findings, ensure_ascii=False)
                )
            if only_non_findings:
                review = {
                    "approved": True,
                    "findings": [],
                    "reviewer_note": "Discarded self-contradictory valid/no-change findings.",
                }
            completed = candidate
            attempts.append({"response": result.text, "review": review_raw, "error": None})
            break
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            attempts.append({"response": result.text, "review": review_raw, "error": str(exc)})
            if attempt == 5:
                raise
            prompt += f"\n\nPrevious output was rejected: {exc}\nRegenerate all examples."

    assert completed is not None
    write_card(args.out, completed)
    load_protocol_from_yaml(args.out)
    artifact = {
        "base_card": args.base_card,
        "schema": args.schema,
        "traces": args.traces,
        "trace_count": len(traces),
        "output": args.out,
        "version": args.version,
        "model": args.model or "deepseek-chat",
        "attempts": attempts,
        "review": review,
    }
    target = Path(args.artifact)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
