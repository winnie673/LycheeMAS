"""Task-adaptive protocol-card initialization and evolution primitives."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from .formatting import dump_protocol_yaml
from .loader import _load_yaml

EDGES = ("planner_to_solver", "solver_to_verifier", "verifier_to_solver")
MUTABLE_CARD_KEYS = {"field_rules", "examples", "value_policy"}
FORBIDDEN_PLACEHOLDERS = {"none", "n/a", "no ambiguity", "unknown", "not applicable"}
SEMANTIC_FALLBACK_FIELDS = ("C", "U", "NL")


def _fallback_fields(spec: dict[str, Any]) -> tuple[str, ...]:
    allowed = set(spec.get("allowed_fields", []))
    return tuple(field for field in SEMANTIC_FALLBACK_FIELDS if field in allowed)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, accepting a surrounding markdown fence."""
    value = text.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("designer output must be a JSON object")
    return parsed


def validate_card(schema: dict[str, Any], card: dict[str, Any]) -> None:
    """Reject any Card that changes the frozen Schema."""
    if set(card.get("cards", {})) != set(EDGES):
        raise ValueError("card must contain exactly the three schema edges")
    for edge in EDGES:
        spec = schema[edge]
        generated = card["cards"][edge]
        if set(generated) - MUTABLE_CARD_KEYS:
            raise ValueError(f"{edge}: only field_rules, examples and value_policy are mutable")
        rules = generated.get("field_rules")
        if not isinstance(rules, dict) or set(rules) != set(spec["allowed_fields"]):
            raise ValueError(f"{edge}: field_rules must cover exactly the frozen fields")
        for field, rule_spec in rules.items():
            if not isinstance(rule_spec, dict) or set(rule_spec) != {"usage", "examples"}:
                raise ValueError(f"{edge}.{field}: rule must contain usage and examples")
            if not str(rule_spec["usage"]).strip():
                raise ValueError(f"{edge}.{field}: usage must be non-empty")
            values = rule_spec["examples"]
            if not isinstance(values, list) or not 1 <= len(values) <= 3:
                raise ValueError(f"{edge}.{field}: provide 1-3 short value examples")
            if any(";" in str(value) for value in values):
                raise ValueError(f"{edge}.{field}: value examples must not contain semicolons")
            if field == "K" and any(
                any(part.count("=") > 1 for part in str(value).split(","))
                for value in values
            ):
                raise ValueError(f"{edge}.K: derived computations belong in P")
            allowed_values = spec.get("value_constraints", {}).get(field, {}).get(
                "allowed_values"
            )
            if allowed_values and any(str(value) not in allowed_values for value in values):
                raise ValueError(f"{edge}.{field}: examples must use frozen allowed values")
        examples = generated.get("examples", [])
        if not isinstance(examples, list):
            raise ValueError(f"{edge}: examples must be a list")
        if len(examples) != 3:
            raise ValueError(f"{edge}: exactly 3 diverse full examples are required")
        for example in examples:
            _validate_example(edge, str(example), set(spec["allowed_fields"]))
        parsed_examples = [_parse_example(str(example)) for example in examples]
        if edge == "planner_to_solver":
            if any(
                any(part.count("=") > 1 for part in fields.get("K", "").split(","))
                for fields in parsed_examples
            ):
                raise ValueError("planner K examples must not contain derived computations")
            for fields in parsed_examples:
                k_items = {part.strip() for part in fields.get("K", "").split(",") if part.strip()}
                p_items = {part.strip() for part in fields.get("P", "").split(",") if part.strip()}
                if k_items & p_items:
                    raise ValueError("planner examples must not repeat relations across K and P")
        fallback_fields = _fallback_fields(spec)
        if edge in {"planner_to_solver", "solver_to_verifier"} and not any(
            not any(field in fields for field in fallback_fields)
            for fields in parsed_examples
        ):
            names = " and ".join(fallback_fields)
            raise ValueError(f"{edge}: one full example must omit optional {names}")
        if edge == "planner_to_solver" and not any(
            "K" not in fields and not any(field in fields for field in fallback_fields)
            for fields in parsed_examples
        ):
            raise ValueError(
                "planner examples must show that shared facts need not be retransmitted"
            )
        if edge == "planner_to_solver" and not any("K" in fields for fields in parsed_examples):
            raise ValueError("planner examples must include a case where selected K is necessary")
        if edge == "planner_to_solver" and not any(
            any(field in fields for field in fallback_fields)
            for fields in parsed_examples
        ):
            raise ValueError("planner examples must include a semantic fallback case")
        if edge == "solver_to_verifier" and not any(
            any(field in fields for field in fallback_fields)
            for fields in parsed_examples
        ):
            raise ValueError("solver examples must include a semantic fallback case")
        if edge == "verifier_to_solver":
            if not any(fields == {"CHK": "ACCEPT"} for fields in parsed_examples):
                raise ValueError("verifier examples must include CHK:ACCEPT")
            if not any(fields.get("CHK") == "REVISE" for fields in parsed_examples):
                raise ValueError("verifier examples must include a valid REVISE message")
            allowed_errors = set(spec["value_constraints"]["E"]["allowed_values"])
            if any(
                fields.get("CHK") == "REVISE" and fields.get("E") not in allowed_errors
                for fields in parsed_examples
            ):
                raise ValueError("verifier REVISE examples must use a frozen E value")
        for field, rule in rules.items():
            lowered = str(rule["usage"]).lower()
            if "joined by ';'" in lowered or "separated by ';'" in lowered:
                raise ValueError(f"{edge}.{field}: semicolon cannot occur inside a field value")
    if not isinstance(card.get("value_policy", {}), dict):
        raise ValueError("value_policy must be an object")


def materialize_card(
    schema: dict[str, Any],
    generated: dict[str, Any],
    *,
    version: str,
    task_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_card(schema, generated)
    result: dict[str, Any] = {
        "version": version,
        "description": (
            f"LLM-generated {_task_name(task_profile)} protocol card {version}."
        ),
        "message_format": schema["message_format"],
        "value_policy": generated.get("value_policy", {}),
    }
    if schema.get("allow_nl_field") is True:
        result["allow_nl_field"] = True
    for edge in EDGES:
        spec = schema[edge]
        content = generated["cards"][edge]
        result[edge] = {
            "allowed_fields": spec["allowed_fields"],
            "required_fields": spec["required_fields"],
            "field_rules": {
                "canonical_order": spec["allowed_fields"],
                "value_representation": (
                    "Field values may use concise natural language, symbols, LaTeX, or a "
                    "mixture of them. Prefer the shortest unambiguous representation. "
                    "Do not use semicolons inside field values; use commas instead."
                ),
                **{
                    field: f"{rule['usage']} Example: {rule['examples'][0]}"
                    for field, rule in content["field_rules"].items()
                },
            },
            "examples": content.get("examples", []),
            "output_schema": _output_schema(edge, spec),
        }
    return result


def write_card(path: str | Path, card: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dump_protocol_yaml(card), encoding="utf-8")


def load_task_profile(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate task-specific Card-generation requirements."""
    profile = _load_yaml(str(path))
    if not isinstance(profile, dict):
        raise ValueError("task profile root must be a mapping")
    required = {"task", "display_name", "scorer_kind", "answer_policy"}
    missing = required - set(profile)
    if missing:
        raise ValueError(f"task profile missing keys: {sorted(missing)}")
    return profile


def _task_name(task_profile: dict[str, Any] | None) -> str:
    if not task_profile:
        return "GSM8K"
    return str(task_profile.get("display_name") or task_profile.get("task") or "task")


def _generation_profile(task_profile: dict[str, Any]) -> dict[str, Any]:
    """Keep diagnosis-only labels out of Card generation and control-field examples."""
    return {key: value for key, value in task_profile.items() if key != "diagnosis_issue_types"}


def initialization_prompt(
    schema: dict[str, Any],
    traces: list[dict[str, Any]],
    task_profile: dict[str, Any] | None = None,
) -> str:
    task_name = _task_name(task_profile)
    profile = task_profile or {
        "task": "gsm8k",
        "display_name": "GSM8K",
        "answer_policy": {
            "rule": "R is the exact final numeric answer without a unit."
        },
    }
    profile = _generation_profile(profile)
    planner_fallback = "/".join(_fallback_fields(schema["planner_to_solver"]))
    solver_fallback = "/".join(_fallback_fields(schema["solver_to_verifier"]))
    if planner_fallback == "NL" and solver_fallback == "NL":
        fallback_guidance = (
            "NL is the single optional one-sentence natural-language fallback. Use it only "
            "when symbolic fields would lose answer-changing meaning or when an unresolved "
            "interpretation must be exposed."
        )
    else:
        fallback_guidance = (
            "C states an additional answer-changing validity condition. U states a genuine "
            "unresolved ambiguity that could change the answer."
        )
    return f"""You are a protocol-card designer for {task_name}.

The communication schema is frozen. You may NOT add, delete, rename, or redefine fields,
edges, required fields, or message syntax.

Infer task-specific field rules, expression preferences, and examples from the target-task
questions and their complete Free-NL Planner-Solver-Verifier communication traces.
Distinguish communication needs from base-model reasoning errors.

The NL traces are evidence about communication needs. They are NOT templates to imitate,
summarize, translate, or transcribe. All agents already receive the original problem.
Do not transmit a fact merely because it appears in the problem. Preserve it only when it
is derived, resolves an ambiguous relation, identifies information essential to the chosen
plan, or cannot be safely recovered by the receiver.

Apply a deletion test to every proposed information item: if the receiver can still recover
the sender's intended target, operation, relation, or verification evidence from the shared
problem and remaining fields, omit the item. Each semantic item should normally occur in
one field only. K contains only explicit givens or relations selected from the shared problem,
never a computed or derived state. G says what to obtain; P contains every derivation and
operation direction. {fallback_guidance} IR is minimal verification evidence, not a full
solution.

Optimize for semantic sufficiency, task correctness, fewer communication tokens, no
repetition between fields, symbolic expressions when shorter, and concise natural language
when symbols are insufficient.
The wire syntax is FIELD:value;FIELD:value. Every example must use semicolons between
fields, never periods. A field value must never contain a semicolon. Use commas or arrows
inside a value. Verifier acceptance must be exactly CHK:ACCEPT.

FROZEN_SCHEMA:
{json.dumps(schema, ensure_ascii=False)}

TASK_PROFILE:
{json.dumps(profile, ensure_ascii=False)}

TRACES:
{json.dumps(traces, ensure_ascii=False)}

Return JSON only with this shape:
{{"cards": {{"planner_to_solver": {{"field_rules": {{FIELD: RULE}}, "examples": [],
"value_policy": {{}}}}, "solver_to_verifier": {{...}}, "verifier_to_solver": {{...}}}},
"value_policy": {{}}}}
Every frozen field must occur exactly once in its edge's field_rules.
Provide exactly three diverse complete examples for every edge. Planner examples must cover:
(1) G/P only with shared facts omitted, (2) necessary selected K, and (3) a meaningful
{planner_fallback} fallback case. Solver examples must cover sufficient compact IR/R evidence
and a meaningful {solver_fallback} fallback case. Verifier examples must cover CHK:ACCEPT,
a locally proven error with a targeted REVISE,
and missing essential evidence with CHK:REVISE;E:INSUFFICIENT_INFO;REQ:specific evidence.
Each field rule must be an object with `usage` and `examples`. `examples` contains exactly
one short example of that field's VALUE only, never a complete message. The `examples` value
MUST be a JSON array containing exactly one string, for example `"examples":["x=1"]`; never
return a scalar string such as `"examples":"x=1"`. Optional fields MUST
be omitted when unused. Never emit none, N/A, no ambiguity, unknown, or not applicable.
Control-field value examples must be individual frozen values: CHK examples must be exactly
`ACCEPT` or `REVISE`; E examples must be exactly one frozen E enum value such as `ARITH`.
Do not put explanatory text, alternatives, or multiple values in these example entries.
The only valid E values are ARITH, RELATION, MISSING_FACT, MISSING_UNIT, WRONG_TARGET,
INSUFFICIENT_INFO, and OTHER. Diagnostic issue labels are not E values and must never
appear in verifier_to_solver.field_rules.E examples or full verifier messages.
At least one full planner and solver example must omit the optional semantic fallback field(s).
Verifier full examples
must include CHK:ACCEPT and a valid CHK:REVISE;E:ENUM;REQ:action. Use only the frozen
uppercase E enum. Keep all frozen field responsibilities non-overlapping.
Infer the answer representation from the task profile, samples, and evaluator needs, and
state it explicitly in solver_to_verifier.field_rules.R. Task-specific answer requirements
belong to the Card, not the frozen Schema. A semantic fallback must not merely restate output
formatting. At least one planner example must omit K and every semantic fallback field,
demonstrating that facts
already visible in the shared problem are not retransmitted. Construct new minimal examples;
use small, mathematically checkable examples rather than transcribing NL messages or using
long worked solutions from D0. Every full example must describe one coherent problem.
Verify every displayed equation and answer. Planner P is a plan, not the completed solution:
write the necessary relations that point to the method, without carrying the algebra
through to the final value of G. For example, with G:solve x^2=a, P:x=±sqrt(a) is a plan;
P:x=±sqrt(4)=±2 is a completed answer and must not appear in a planner example.
Do not convert long natural-language plans into shorter natural-language plans. Compile them
into compact symbolic relations. Prefer `P:sub=10+4*20,total=1.2*sub` over
`P:compute 10+4*20, then add 20%`. Prefer `K:eggs=16,eaten=3,baked=4,price=2` for explicit
facts; put `remain=eggs-eaten-baked,revenue=remain*price` in P. Prefer symbolic relations when
shorter and unambiguous, but allow concise natural language for correspondences, enumeration,
case coverage, or proof obligations that symbols cannot safely express. Do not impose a fixed
number of relations or a pure-symbolic requirement.

Infer field rules and examples that satisfy these communication functions:
1. Distinguish a plan from verification evidence. IR must support the decisive operations
   connecting key relations to R. Initial conditions, generic formulas, or method names alone
   do not support a concrete answer. Preserve necessary operands, operations, and results;
   do not demand every routine step or a full derivation.
2. For counts, all solutions, minima, or maxima, preserve the essential case coverage,
   validity filtering, and final aggregation or optimality evidence. Infer task-specific
   evidence needs from the traces rather than adding a separate template for every task.
3. R directly answers G. Extraction or combination of intermediate results belongs in IR;
   R contains one final answer object, not a derivation chain. An equation is allowed when
   the requested answer itself is an equation. Semantic fallback fields contain preserved
   meaning or ambiguity, not answers.
4. P preserves the original target, conditions, transformation direction, and object
   correspondence. Do not present an unverified guess or answer-changing assumption as an
   established fact; represent unresolved constructions as quantities to solve or verify.
5. Every full Solver example must be mathematically coherent and supply enough context and
   decisive evidence to support R. Do not invent a result from unspecified parameters.
6. Verifier rules distinguish a locally established conflict from missing essential evidence.
   Use INSUFFICIENT_INFO for the latter and request only the particular missing relation or
   operation. Do not prescribe an unverified replacement answer in REQ. Accept when sufficient
   evidence supports R and no concrete defect is established.
"""


def card_review_prompt(
    schema: dict[str, Any],
    card: dict[str, Any],
    task_profile: dict[str, Any] | None = None,
    traces: list[dict[str, Any]] | None = None,
) -> str:
    task_name = _task_name(task_profile)
    profile = _generation_profile(
        task_profile or {"task": "gsm8k", "display_name": "GSM8K"}
    )
    nl_fallback = "NL" in schema["planner_to_solver"].get("allowed_fields", [])
    fallback_review = (
        "use NL as repeated prose or make it effectively mandatory"
        if nl_fallback
        else "use C for target, units, or output format; use U as a placeholder"
    )
    fallback_name = "NL" if nl_fallback else "C/U"
    return f"""You are a strict, read-only reviewer of a {task_name} communication Card.
Do not rewrite the Card. Decide whether it extracts minimal communication functions rather
than transcribing shared problem text or Free-NL messages.

Reject when rules or examples systematically: unnecessarily copy shared givens;
put computed or derived states in K; replace necessary relations with vague method names;
repeat the same facts/equations across K and P; {fallback_review}; expand IR beyond
verification-critical evidence; or make optional fields effectively mandatory. A selected
explicit given in K is allowed when it is needed
to disambiguate the chosen plan; the frozen K semantics expressly permit this. Symbolic
equation chains are not procedural natural language. Reject only concrete violations,
not missing repetitions of prohibitions already guaranteed by the frozen Schema.
Reject if the R rule does not explicitly define the
task-specific answer representation supported by the task profile, samples, and evaluator,
or replaces an exact accepted form with an unsupported approximation.
Also reject any full example whose fields describe different problems or whose
{fallback_name} does not preserve meaning that changes the IR, R, K, G, or P in that same
example. Preserve
necessary derived relations and operation direction.
Reject planner P examples only when they explicitly state the final value of G. A plan may
contain relations from which the Solver can readily infer the answer; this is expected and
must not be rejected. Check mathematical validity of full examples using the D0 cases when
applicable. If the evidence does not establish a violation, approve instead of speculating.

Check evidence sufficiency as well as brevity: reject a Solver example that supplies only
initial conditions or a generic method but cannot support its specific R. Decisive operations
from key relations to R and necessary counting, branch, filtering, or optimality evidence
must be supported. Do not require a full solution or fixed number of equations.
Check that R directly answers the target and contains a final answer object rather than a
derivation chain; equations are allowed when the answer itself is an equation. The semantic
fallback field must not carry the answer. Reject ungrounded answer-changing assumptions or
invalid transformations.
Allow concise natural language when it safely expresses correspondences, case coverage, or
verification obligations; do not reject based on procedural keywords or absence of symbols.
Check that Verifier examples cover ACCEPT, a concrete locally proven error, and a request for
missing evidence using INSUFFICIENT_INFO. Reject rules that conflate lack of evidence with
an established error or prescribe an unverified replacement answer.

Return JSON only:
{{"approved": true, "findings": []}}
or
{{"approved": false, "findings": [{{"path":"...","evidence":"...","fix":"..."}}]}}

FROZEN_SCHEMA:{json.dumps(schema, ensure_ascii=False)}
TASK_PROFILE:{json.dumps(profile, ensure_ascii=False)}
CARD:{json.dumps(card, ensure_ascii=False)}
D0_TRACES:{json.dumps(traces or [], ensure_ascii=False)}
"""


def diagnosis_prompt(
    current_card: dict[str, Any],
    feedback: dict[str, Any],
    task_profile: dict[str, Any] | None = None,
) -> str:
    task_name = _task_name(task_profile)
    profile = task_profile or {"task": "gsm8k", "display_name": "GSM8K"}
    return f"""You diagnose a {task_name} protocol card. DIAGNOSIS ONLY; do not modify it.
Do not treat every wrong answer as a communication failure. For each selected paired case,
compare sender intent, encoded meaning, semantic gaps, receiver sufficiency, redundancy,
and whether the cause is communication or base reasoning. Preserve successful compact
patterns. Return JSON only: {{"case_analysis": [{{"case_id": "...",
"issue_type": "...", "communication_related": true, "protocol_path": "...",
"evidence": "..."}}], "global_findings": []}}.

Use task-specific issue types from TASK_PROFILE when applicable, while retaining general
communication issue types. A task-specific issue label does not change the frozen verifier
error enum.

TASK_PROFILE:
{json.dumps(profile, ensure_ascii=False)}

CURRENT_CARD:
{json.dumps(current_card, ensure_ascii=False)}
FEEDBACK:
{json.dumps(feedback, ensure_ascii=False)}
"""


def evolution_prompt(
    schema: dict[str, Any],
    current_card: dict[str, Any],
    diagnosis: dict[str, Any],
    feedback: dict[str, Any],
    task_profile: dict[str, Any] | None = None,
) -> str:
    profile = task_profile or {"task": "gsm8k", "display_name": "GSM8K"}
    return f"""You are a protocol-card evolver. Use ONLY diagnosed evidence.
The schema is frozen. You may modify only an existing field rule, one edge's complete
examples list, or the root value_policy. You may not add, delete, rename, or redefine fields
or change edges, required fields, control enums, or syntax. Prefer simplifying an existing
rule instead of adding clauses. Preserve verifier sufficiency. Make at most 3 path changes.

Every change must cite one or more communication-related diagnosis case_ids. The paired raw
cases are supplied so you can verify the diagnosis against the exact NL and Hybrid messages.
Do not modify the Card for BASE_REASONING_ERROR. If evidence does not justify a minimal
change, return KEEP.

Return JSON only. Either:
{{"action":"KEEP","reason":"..."}}
or:
{{"action":"UPDATE","reason":"...","changes":[{{"path":"planner_to_solver.field_rules.P",
"value":"replacement rule string","diagnosis_case_ids":["..."],"evidence":"...",
"expected_benefit":"...","risk":"..."}}]}}

Allowed paths are exactly an existing EDGE.field_rules.FIELD, EDGE.examples, or value_policy.
For EDGE.examples, value must be the complete replacement list, not one example. Do not
return a complete rewritten Card.

FROZEN_SCHEMA:{json.dumps(schema, ensure_ascii=False)}
TASK_PROFILE:{json.dumps(profile, ensure_ascii=False)}
CURRENT_CARD:{json.dumps(current_card, ensure_ascii=False)}
DIAGNOSIS:{json.dumps(diagnosis, ensure_ascii=False)}
PAIRED_EVIDENCE:{json.dumps(feedback, ensure_ascii=False)}
"""


def apply_evolution_changes(
    schema: dict[str, Any],
    current_card: dict[str, Any],
    decision: dict[str, Any],
    *,
    version: str,
    diagnosed_case_ids: set[str],
    task_profile: dict[str, Any] | None = None,
    max_changes: int = 3,
) -> tuple[dict[str, Any], list[str]]:
    """Apply an evidence-bound patch while keeping the Schema structurally frozen."""
    if decision.get("action") != "UPDATE":
        raise ValueError("decision action must be UPDATE")
    changes = decision.get("changes")
    if not isinstance(changes, list) or not 1 <= len(changes) <= max_changes:
        raise ValueError(f"UPDATE must contain 1-{max_changes} changes")
    candidate = deepcopy(current_card)
    changed_paths: list[str] = []
    for change in changes:
        if not isinstance(change, dict):
            raise ValueError("each change must be an object")
        path = str(change.get("path", ""))
        if path in changed_paths:
            raise ValueError(f"duplicate change path: {path}")
        cited = change.get("diagnosis_case_ids")
        if not isinstance(cited, list) or not cited:
            raise ValueError(f"{path}: diagnosis_case_ids must be non-empty")
        if not set(map(str, cited)).issubset(diagnosed_case_ids):
            raise ValueError(f"{path}: cites an unknown diagnosis case")
        parts = path.split(".")
        if path == "value_policy":
            if not isinstance(change.get("value"), dict):
                raise ValueError("value_policy replacement must be an object")
            candidate["value_policy"] = change["value"]
        elif len(parts) == 3 and parts[0] in EDGES and parts[1] == "field_rules":
            edge, _, field = parts
            if field not in schema[edge]["allowed_fields"]:
                raise ValueError(f"unknown frozen field in path: {path}")
            value = change.get("value")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{path}: replacement rule must be a non-empty string")
            candidate[edge]["field_rules"][field] = value.strip()
        elif len(parts) == 2 and parts[0] in EDGES and parts[1] == "examples":
            edge = parts[0]
            values = change.get("value")
            if not isinstance(values, list) or not values:
                raise ValueError(f"{path}: replacement must be a non-empty list")
            allowed = set(schema[edge]["allowed_fields"])
            for example in values:
                _validate_example(edge, str(example), allowed)
            candidate[edge]["examples"] = [str(example) for example in values]
        else:
            raise ValueError(f"illegal evolution path: {path}")
        changed_paths.append(path)

    candidate["version"] = version
    candidate["description"] = (
        f"Evidence-evolved {_task_name(task_profile)} protocol card {version}, derived from "
        f"{current_card.get('version', 'unknown')}."
    )
    _validate_materialized_schema(schema, candidate)
    return candidate, changed_paths


def _validate_materialized_schema(schema: dict[str, Any], card: dict[str, Any]) -> None:
    if schema.get("allow_nl_field") is True and card.get("allow_nl_field") is not True:
        raise ValueError("allow_nl_field opt-in changed from frozen Schema")
    if card.get("message_format") != schema.get("message_format"):
        raise ValueError("message_format changed from frozen Schema")
    for edge in EDGES:
        spec = schema[edge]
        content = card.get(edge)
        if not isinstance(content, dict):
            raise ValueError(f"missing edge: {edge}")
        if content.get("allowed_fields") != spec["allowed_fields"]:
            raise ValueError(f"{edge}: allowed_fields changed")
        if content.get("required_fields") != spec["required_fields"]:
            raise ValueError(f"{edge}: required_fields changed")
        rules = content.get("field_rules", {})
        rule_fields = set(rules) - {"canonical_order", "value_representation"}
        if rule_fields != set(spec["allowed_fields"]):
            raise ValueError(f"{edge}: field_rules do not match frozen fields")
        if rules.get("canonical_order") != spec["allowed_fields"]:
            raise ValueError(f"{edge}: canonical_order changed")


def classify_feedback(
    nl_cases: list[dict[str, Any]], hybrid_cases: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Create deterministic paired feedback categories for one evolution batch."""
    nl = {str(case["task_id"]): case for case in nl_cases}
    groups: dict[str, list[dict[str, Any]]] = {}
    correct_costs = sorted(
        int(c.get("communication_tokens", 0))
        for c in hybrid_cases
        if c.get("is_correct")
    )
    threshold = correct_costs[max(0, int(len(correct_costs) * 0.8) - 1)] if correct_costs else 0
    for hybrid in hybrid_cases:
        baseline = nl.get(str(hybrid["task_id"]))
        if baseline is None:
            raise ValueError(f"missing paired NL case: {hybrid['task_id']}")
        hc, nc = bool(hybrid.get("is_correct")), bool(baseline.get("is_correct"))
        if hybrid.get("run_status") not in {None, "completed"}:
            label = "run_failed"
        elif not hc and nc:
            label = "hybrid_incorrect_nl_correct"
        elif not hc and not nc:
            label = "both_incorrect"
        elif hc and not nc:
            label = "hybrid_correct_nl_incorrect"
        elif int(hybrid.get("communication_tokens", 0)) >= threshold:
            label = "hybrid_correct_high_cost"
        elif hybrid.get("revision_triggered"):
            label = "revision_success"
        else:
            continue
        groups.setdefault(label, []).append({"nl": baseline, "hybrid": hybrid})
    return groups


def _output_schema(edge: str, spec: dict[str, Any]) -> str:
    fields = ",".join(spec["allowed_fields"])
    required = ",".join(spec["required_fields"])
    if edge == "verifier_to_solver":
        return "Output CHK:ACCEPT, or direct fields; REVISE requires CHK, E and REQ."
    return f"Output direct fields from [{fields}]; required: [{required}]."


def _validate_example(edge: str, example: str, allowed: set[str]) -> None:
    if edge == "verifier_to_solver" and example.strip() == "ACCEPT":
        raise ValueError("verifier acceptance example must be CHK:ACCEPT")
    raw_segments = example.split(";")
    if any(not part.strip() for part in raw_segments):
        raise ValueError(f"{edge}: empty field segment is forbidden")
    segments = [part.strip() for part in raw_segments]
    labels = []
    for segment in segments:
        if ":" not in segment:
            raise ValueError(f"{edge}: invalid example segment {segment!r}")
        labels.append(segment.split(":", 1)[0].strip())
    embedded = re.findall(r"(?:^|[.;]\s*)\b([A-Z]{1,4}):", example)
    if labels != embedded or any(label not in allowed for label in labels):
        raise ValueError(f"{edge}: example does not use valid semicolon-delimited fields")
    for field, value in _parse_example(example).items():
        if field in SEMANTIC_FALLBACK_FIELDS and value.lower() in FORBIDDEN_PLACEHOLDERS:
            raise ValueError(f"{edge}.{field}: placeholder values are forbidden")


def _parse_example(example: str) -> dict[str, str]:
    return {
        segment.split(":", 1)[0].strip(): segment.split(":", 1)[1].strip()
        for segment in example.split(";")
        if ":" in segment
    }


def load_schema(path: str | Path) -> dict[str, Any]:
    return _load_yaml(str(path))
