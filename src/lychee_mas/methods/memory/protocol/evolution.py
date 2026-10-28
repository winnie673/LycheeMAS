"""Injected offline protocol evolution; runtime remains the existing CDM executor."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Callable

from .base import EvidenceDesign, EvolutionSchedule, Execute, Generate, ProtocolContract, Score


def changed_paths(left: object, right: object, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        return [
            path
            for key in sorted(left.keys() | right.keys())
            for path in changed_paths(left.get(key), right.get(key), f"{prefix}/{key}")
        ]
    return [] if left == right else [prefix or "/"]


class ProtocolEvolution:
    """Induce, propose and select with experiment-injected contracts and split schedule."""

    def __init__(
        self,
        *,
        generate: Generate,
        execute: Execute,
        score: Score,
        contract: ProtocolContract,
        schedule: EvolutionSchedule,
        executor_capability: str,
        effect_valid: bool = False,
        sanitize: Callable = lambda value: value,
        proposal_tokens: int = 8192,
        design: EvidenceDesign | None = None,
        transport_errors: tuple[type[Exception], ...] = (TimeoutError, ConnectionError),
    ):
        if not executor_capability.strip() or proposal_tokens < 1:
            raise ValueError("need executor capability and positive proposal token budget")
        self.generate, self.execute, self.score = generate, execute, score
        self.contract, self.schedule = contract, schedule
        self.architecture, self.capability = contract.architecture, executor_capability
        self.effect_valid, self.sanitize = effect_valid, sanitize
        self.proposal_tokens, self.transport_errors = proposal_tokens, transport_errors
        self.design = design

    def _write(self, path: Path, value: object) -> None:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(self.sanitize(value), handle, ensure_ascii=False, indent=2, allow_nan=False)

    def _card(self, path: Path) -> dict:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def _validate(self, raw: dict, path: Path) -> None:
        from ..channels.hybrid_protocol import HybridProtocolMemory

        self.contract.check_card(raw, require_identity=True)
        if not isinstance(raw.get("message_format"), dict):
            raise ValueError("message_format must be a mapping")
        if not isinstance(raw.get("value_policy", {}), dict):
            raise ValueError("value_policy must be a mapping")
        channel = HybridProtocolMemory(str(path), self.contract.task, contract=self.contract)
        for edge, allowed, _ in self.architecture.edges:
            section = raw[edge]
            rules, examples = section.get("field_rules", {}), section.get("examples", [])
            if (
                not isinstance(rules, dict)
                or any(not isinstance(rules.get(k), str) or not rules[k].strip() for k in allowed)
                or rules.get("canonical_order") != list(allowed)
                or set(rules) - set(allowed) - {"canonical_order", "value_representation"}
            ):
                raise ValueError(f"missing rules or changed canonical order on {edge}")
            if not isinstance(examples, list) or not examples:
                raise ValueError(f"need model-generated examples on {edge}")
            sender, receiver = edge.split("_to_")
            seen_fields = set()
            validation = self.contract.validation_for(edge)
            for example in examples:
                fields = channel.validate(example, sender, receiver)
                seen_fields.update(fields)
                for rule in validation.example_only:
                    rule.check(fields)
            if not set(validation.required_example_fields) <= seen_fields:
                raise ValueError(f"missing required example field coverage on {edge}")

    def _propose(self, out: Path, evidence: list[dict], parent: Path | None) -> tuple[Path, dict]:
        if self.design is not None:
            return self._evidence_proposal(out, evidence, parent)
        import yaml

        request = {
            "task": self.contract.task,
            "architecture": self.architecture.as_dict(),
            "executor_capability": self.capability,
            "fixed_field_meanings": dict(self.contract.field_meanings),
            "contract": self.contract.as_dict(),
            "parent_card": self._card(parent) if parent else None,
            "development_evidence": evidence,
            "instructions": (
                "Induce a usable card from induction natural-language traces when parent is null "
                "(UPDATE required). Otherwise evolve ONLY the parent using development evidence; "
                "choose UPDATE or KEEP. Respect the stated executor capability. "
                "Preserve every field's high-level meaning. Optimize correctness then "
                "communication "
                "tokens, not input tokens. "
                "Return one JSON object: decision UPDATE/KEEP, reason string, card object "
                "(UPDATE only). No Markdown. Card needs version, description, fixed_card_options "
                "from the contract, "
                "message_format={wire_syntax:direct,structure:fields_only}, and each edge section "
                "with the EXACT allowed_fields/required_fields from architecture, field_rules "
                "(canonical_order equals allowed_fields, and a nonempty rule for every field), "
                "and nonempty examples of direct FIELD:value segments separated by semicolons. "
                "No semicolons in values. Obey the contract's declarative constraints, example "
                "field coverage and generation_guidance. Do not rename fields or copy heldout "
                "examples. task_card and validation_contract are assigned by the tool."
            ),
        }
        messages = [
            {"role": "system", "content": "Design a reusable communication protocol, not answers."},
            {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
        ]
        for attempt in range(3):
            record = {"attempt": attempt, "messages": copy.deepcopy(messages)}
            generated = False
            try:
                result = self.generate(messages, max_new_tokens=self.proposal_tokens)
                generated = True
                record.update(
                    raw_output=result.text,
                    **{
                        key: getattr(result, key, None)
                        for key in ("n_prompt_pos", "n_gen_tokens", "latency_s")
                    },
                )
                record["usage_available"] = all(
                    isinstance(record.get(k), int) and record[k] > 0
                    for k in ("n_prompt_pos", "n_gen_tokens")
                )
                if getattr(result, "n_gen_tokens", 0) >= self.proposal_tokens:
                    raise ValueError("proposal may be truncated at output budget")
                value = json.loads(result.text)
                if not isinstance(value, dict) or not isinstance(value.get("reason"), str):
                    raise ValueError("proposal needs a decision and reason")
                decision = value.get("decision")
                if decision == "KEEP":
                    if parent is None or "card" in value:
                        raise ValueError("KEEP requires a parent and no replacement card")
                    card = self._card(parent)
                elif decision == "UPDATE" and isinstance(value.get("card"), dict):
                    card = copy.deepcopy(value["card"])
                    card["task_card"] = {
                        "task": self.contract.task,
                        "roles": list(dict(self.architecture.roles)),
                        "edges": [edge for edge, _, _ in self.architecture.edges],
                        "architecture": self.architecture.name,
                        "architecture_sha256": self.architecture.fingerprint,
                        "contract_sha256": self.contract.fingerprint,
                    }
                    if (
                        "validation_contract" in card
                        and ProtocolContract.from_dict(card["validation_contract"]).fingerprint
                        != self.contract.fingerprint
                    ):
                        raise ValueError("proposal changed the validation contract")
                    card["validation_contract"] = self.contract.as_dict()
                    if not isinstance(card.get("version"), str) or not card["version"].strip():
                        raise ValueError("card needs a version")
                else:
                    raise ValueError("expected UPDATE with card or KEEP")
                path = out / f"proposal_{attempt}.yaml"
                if self.sanitize(card) != card:
                    raise ValueError("card contains a credential; refusing to persist it")
                with path.open("x", encoding="utf-8") as handle:
                    yaml.safe_dump(card, handle, allow_unicode=True, sort_keys=False)
                self._validate(card, path)
                record["status"] = "valid"
                self._write(out / f"proposal_{attempt}.json", record)
                before = self._card(parent) if parent else {}

                # Metadata/version changes alone do not count as semantic evolution.
                def clean(c):
                    return {
                        k: v
                        for k, v in c.items()
                        if k not in {"task_card", "version", "description"}
                    }

                changes = changed_paths(clean(before), clean(card))
                return path, {
                    "decision": decision,
                    "reason": value["reason"],
                    "changed_paths": changes,
                }
            except (ValueError, TypeError, KeyError, *self.transport_errors) as exc:
                if not generated and not isinstance(exc, self.transport_errors):
                    record.update(status="fatal_error", error=f"{type(exc).__name__}: {exc}")
                    self._write(out / f"proposal_{attempt}.json", record)
                    raise
                record.update(status="invalid", error=f"{type(exc).__name__}: {exc}")
                self._write(out / f"proposal_{attempt}.json", record)
                if attempt == 2:
                    raise ValueError("three protocol proposal attempts exhausted") from exc
                messages.extend(
                    [
                        {"role": "assistant", "content": record.get("raw_output", "")},
                        {
                            "role": "user",
                            "content": f"Correct the proposal: {exc}. "
                            "Preserve the fixed interface.",
                        },
                    ]
                )
            except Exception as exc:
                record.update(status="fatal_error", error=f"{type(exc).__name__}: {exc}")
                self._write(out / f"proposal_{attempt}.json", record)
                raise
        raise AssertionError("unreachable")

    def _design_call(
        self, out: Path, operation: str, attempt: int, prompt: str, budget: int
    ) -> str:
        messages = [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ]
        record = {
            "operation": operation,
            "attempt": attempt,
            "input_messages": messages,
            "raw_output": "",
            "error": None,
        }
        try:
            result = self.generate(messages, max_new_tokens=budget)
            record.update(
                raw_output=result.text,
                status="returned",
                **{
                    key: getattr(result, key, None)
                    for key in (
                        "n_prompt_pos",
                        "n_gen_tokens",
                        "latency_s",
                        "finish_reason",
                        "usage_unknown",
                    )
                },
            )
            return result.text
        except Exception as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self._write(out / f"{operation}_{attempt}.json", record)

    def _save_designed_card(self, path: Path, card: dict) -> None:
        import yaml

        if self.sanitize(card) != card:
            raise ValueError("card contains a credential; refusing to persist it")
        with path.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(card, handle, allow_unicode=True, sort_keys=False)
        self._validate(card, path)

    def _stamp(self, card: dict) -> dict:
        card = copy.deepcopy(card)
        card["task_card"] = {
            "task": self.contract.task,
            "roles": list(dict(self.architecture.roles)),
            "edges": [edge for edge, _, _ in self.architecture.edges],
            "architecture": self.architecture.name,
            "architecture_sha256": self.architecture.fingerprint,
            "contract_sha256": self.contract.fingerprint,
            "effect_valid": self.effect_valid,
        }
        if not self.effect_valid:
            card["description"] = "FAKE plumbing artifact; not a learned protocol."
        card["validation_contract"] = self.contract.as_dict()
        return card

    @staticmethod
    def _review_errors(review: dict, material: dict) -> list[str]:
        """Reject contradictory reviewer output, not the card it was reviewing."""
        if not isinstance(review, dict) or type(review.get("approved")) is not bool:
            return ["review requires a boolean approved"]
        findings = review.get("findings")
        if not isinstance(findings, list):
            return ["review requires a findings list"]
        errors = []
        if review["approved"] == bool(findings):
            errors.append("approved must be true exactly when blocking findings is empty")
        if not isinstance(review.get("recommendations", []), list):
            errors.append("optional recommendations must be a separate list")
        for index, finding in enumerate(findings):
            if not isinstance(finding, dict) or any(
                not isinstance(finding.get(key), str) or not finding[key].strip()
                for key in ("path", "evidence")
            ):
                errors.append(f"finding {index} requires concrete path and evidence")
                continue
            path = finding["path"]
            value = material
            try:
                if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*|\[\d+\])*", path):
                    raise KeyError(path)
                for part in path.replace("[", ".").replace("]", "").split("."):
                    value = value[int(part)] if isinstance(value, list) else value[part]
            except (KeyError, IndexError, TypeError, ValueError):
                errors.append(f"finding {index} points outside supplied design material: {path}")
            text = finding["evidence"] + " " + str(finding.get("fix", ""))
            if finding.get("severity", "error") not in ("error", "blocking") or re.search(
                r"\bno (?:concrete )?violation\b|\bnot (?:a |an )?(?:hard |concrete )?violation\b"
                r"|\bweak coverage gap\b|\barguably covered\b|\boptionally\b",
                text,
                re.IGNORECASE,
            ):
                errors.append(
                    f"finding {index} mixes nonviolations or optional advice into rejection"
                )
        return errors

    def _review_card(self, out: Path, card: dict, evidence: list, attempt: int) -> tuple[str, dict]:
        # Runtime identity/compatibility are checked by _validate, not redesigned by an LLM.
        material = copy.deepcopy(
            {k: v for k, v in card.items() if k not in {"task_card", "validation_contract"}}
        )
        base = self.design.review(material, evidence, self.capability)
        prompt = base + (
            "\n\nREVIEW_OUTPUT_CONTRACT:\n"
            "Return approved:boolean, findings:list and optional recommendations:list. "
            "findings contains ONLY concrete blocking violations with an existing dotted path "
            "(use [index] for array elements), evidence and a compatible fix. "
            "If findings is empty, approved MUST be true; otherwise approved MUST be false. "
            "Nonviolations and optional improvements belong outside findings and must not "
            "cause rejection. Do not critique runtime metadata not present in CARD."
        )
        for check in range(2):
            raw = self._design_call(
                out, "review" if check == 0 else "review_recheck", attempt, prompt, 2048
            )
            try:
                review = self.design.parse(raw)
                errors = self._review_errors(review, material)
            except (ValueError, KeyError) as exc:
                errors = [f"invalid review JSON: {exc}"]
            self._write(
                out / f"review_validation_{attempt}_{check}.json",
                {"status": "inconsistent" if errors else "valid", "errors": errors},
            )
            if not errors:
                return raw, review
            prompt = base + (
                "\n\nREVIEW_RECHECK:\nYour previous review was inconsistent:\n"
                + json.dumps(errors, ensure_ascii=False)
                + "\nRe-evaluate the SAME card independently. Do not rewrite or waive its "
                "requirements. Report only proven blocking violations in findings with "
                "existing dotted paths, evidence and fixes. Optional suggestions belong in "
                "recommendations. approved must be true exactly when findings is empty. "
                "Do not critique runtime metadata not present in CARD.\nPREVIOUS_REVIEW:\n" + raw
            )
        raise RuntimeError("review consistency check failed after one recheck; card not approved")

    def _evidence_proposal(
        self, out: Path, evidence: list | dict, parent: Path | None
    ) -> tuple[Path, dict]:
        if parent is None:
            base = self.design.initialize(evidence, self.capability)
            prompt = base
            for attempt in range(6):
                raw = self._design_call(out, "initialize", attempt, prompt, self.proposal_tokens)
                validation = {"attempt": attempt, "raw_response": raw, "error": None}
                try:
                    card = self._stamp(
                        self.design.materialize(
                            self.design.parse(raw), f"CARD-0-{self.contract.task}"
                        )
                    )
                    path = out / f"proposal_{attempt}.yaml"
                    self._save_designed_card(path, card)
                    review_raw, review = self._review_card(out, card, evidence, attempt)
                    validation["review_response"] = review_raw
                    if review.get("approved") is not True:
                        raise ValueError(
                            "semantic reviewer rejected card: "
                            + json.dumps(review.get("findings", []), ensure_ascii=False)
                        )
                    validation["status"] = "approved"
                    self._write(out / f"initial_validation_{attempt}.json", validation)
                    return path, {
                        "decision": "UPDATE",
                        "reason": "Initial semantic review passed.",
                        "changed_paths": changed_paths({}, card),
                        "semantic_review_approved": True,
                    }
                except (ValueError, KeyError) as exc:
                    validation.update(status="invalid", error=str(exc))
                    self._write(out / f"initial_validation_{attempt}.json", validation)
                    if attempt == 5:
                        raise ValueError(
                            "six initial generation/review attempts exhausted"
                        ) from exc
                    prompt = base + (
                        f"\n\nYour previous output failed validation:\n{exc}\n"
                        "Regenerate the complete JSON object and address every finding; "
                        "do not change the frozen schema. Semicolons separate fields and cannot "
                        "occur inside a field value. If an example cannot satisfy the fixed "
                        "contract, replace the entire example with a different coherent "
                        "miniature problem; do not preserve its unsupported conclusion or "
                        "invent missing premises. Any reviewer suggestion conflicting with "
                        "the frozen schema or task answer policy must not be followed; "
                        "repair the concrete violation while preserving those constraints. "
                        f"\nPREVIOUS_JSON:\n{raw}"
                    )
        current = self._card(parent)
        raw = self._design_call(
            out,
            "diagnosis",
            0,
            self.design.diagnose(current, evidence, self.capability),
            self.proposal_tokens,
        )
        try:
            diagnosis = self.design.parse(raw)
            eligible = self.design.eligible(diagnosis, evidence)
        except (ValueError, KeyError) as exc:
            self._write(out / "diagnosis_validation.json", {"error": str(exc), "raw_response": raw})
            raise
        self._write(
            out / "diagnosis.json", {"diagnosis": diagnosis, "eligible_case_ids": sorted(eligible)}
        )
        if not eligible:
            path = out / "proposal_0.yaml"
            self._save_designed_card(path, current)
            return path, {
                "decision": "KEEP",
                "changed_paths": [],
                "reason": "No evidence-bound communication-related diagnosis case was found.",
            }
        prompt = self.design.evolve(current, diagnosis, evidence, eligible, self.capability)
        for attempt in range(3):
            raw = self._design_call(out, "evolution", attempt, prompt, self.proposal_tokens)
            validation = {"attempt": attempt, "raw_response": raw, "error": None}
            try:
                decision = self.design.parse(raw)
                action = decision.get("action")
                if action == "UPDATE":
                    card, changes = self.design.apply(
                        current,
                        decision,
                        eligible,
                        f"CARD-{out.name.removeprefix('round')}-{self.contract.task}",
                    )
                    card = self._stamp(card)
                elif action == "KEEP":
                    card, changes = current, []
                else:
                    raise ValueError("evolver action must be KEEP or UPDATE")
                path = out / f"proposal_{attempt}.yaml"
                self._save_designed_card(path, card)
                validation["status"] = "valid"
                self._write(out / f"evolution_validation_{attempt}.json", validation)
                return path, {
                    "decision": action,
                    "reason": decision.get("reason", ""),
                    "changed_paths": changes,
                    "eligible_case_ids": sorted(eligible),
                }
            except (ValueError, KeyError) as exc:
                validation.update(status="invalid", error=str(exc))
                self._write(out / f"evolution_validation_{attempt}.json", validation)
                if attempt == 2:
                    raise ValueError("three evidence-bound evolution attempts exhausted") from exc
                prompt += (
                    f"\n\nYour previous output failed validation: {exc}\n"
                    "Return the complete decision again without changing the frozen Schema. "
                    "Use only these exact diagnosis case IDs: "
                    + json.dumps(sorted(eligible), ensure_ascii=False)
                )
        raise AssertionError("unreachable")

    def _batch(
        self, cases: list[dict], mode: str, card: Path | None, out: Path
    ) -> tuple[list, dict]:
        out.mkdir()
        original = hashlib.sha256(card.read_bytes()).hexdigest() if card else None
        rows = []
        for index, case in enumerate(cases):
            # Deliberately strip gold and other manifest metadata at the execution boundary.
            row = self.execute(
                {"task_id": case["task_id"], "question": case["question"]},
                mode,
                card,
                out / f"case_{index:05d}.attempts.jsonl",
            )
            score = (
                self.score(row.get("prediction", ""), case["gold"])
                if row["status"] == "completed"
                else 0.0
            )
            if score not in (0.0, 1.0):
                raise ValueError("exact-match score must be 0 or 1")
            row.update(
                task_id=case["task_id"], gold=case["gold"], score=score, question=case["question"]
            )
            rows.append(row)
            self._write(out / f"case_{index:05d}.json", row)
        if card and hashlib.sha256(card.read_bytes()).hexdigest() != original:
            raise RuntimeError("card changed during execution")
        metrics = {
            "n": len(rows),
            "correct": sum(row["score"] for row in rows),
            "accuracy": sum(row["score"] for row in rows) / len(rows),
            "failed": sum(row["status"] != "completed" for row in rows),
        }
        for key in ("communication_tokens", "model_output_tokens", "wall_time_s"):
            values = [row.get(key) for row in rows]
            if any(
                v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0)
                for v in values
            ):
                raise ValueError(f"invalid {key} usage")
            metrics[f"avg_{key}"] = (
                sum(values) / len(rows) if all(v is not None for v in values) else None
            )
        self._write(out / "metrics.json", metrics)
        return rows, metrics

    @staticmethod
    def select(parent: dict, candidate: dict) -> tuple[bool, str]:
        if candidate["accuracy"] != parent["accuracy"]:
            return candidate["accuracy"] > parent["accuracy"], "accuracy"
        old, new = (m["avg_communication_tokens"] for m in (parent, candidate))
        if old is not None and new is not None and new != old:
            return new < old, "exact_communication_tokens"
        return False, "tie_or_unknown_cost_keep_parent"

    def run(
        self,
        splits: dict[str, list[dict]],
        out: Path,
        *,
        metadata: dict | None = None,
        verify_sources: Callable[[], None] | None = None,
    ) -> dict:
        required = {self.schedule.induction, *self.schedule.evolution, self.schedule.validation}
        if set(splits) != required or any(not splits[name] for name in required):
            raise ValueError(
                "supply exactly the nonempty development splits; other splits forbidden"
            )
        ids, questions = set(), set()
        for rows in splits.values():
            for row in rows:
                if any(
                    not isinstance(row.get(k), str) or not row[k].strip()
                    for k in ("task_id", "question", "gold")
                ):
                    raise ValueError("cases need nonempty task_id/question/gold")
                if row["task_id"] in ids or row["question"].strip() in questions:
                    raise ValueError("duplicate case or question across development splits")
                ids.add(row["task_id"])
                questions.add(row["question"].strip())
        out = Path(out)
        out.mkdir(parents=True, exist_ok=False)
        self._write(
            out / "config.json",
            {
                "version": "memory-evolution-v5-schema-derived",
                "task": self.contract.task,
                "architecture": self.architecture.as_dict(),
                "contract": self.contract.as_dict(),
                "contract_sha256": self.contract.fingerprint,
                "schedule": {
                    "induction": self.schedule.induction,
                    "evolution": self.schedule.evolution,
                    "validation": self.schedule.validation,
                    "heldout": self.schedule.heldout,
                },
                "effect_valid": self.effect_valid,
                "executor_capability": self.capability,
                "proposal_tokens": self.proposal_tokens,
                "design_strategy": "evidence_bound_reviewed"
                if self.design is not None
                else "direct",
                "selection": "accuracy_then_exact_communication_tokens_else_parent",
                "metadata": metadata or {},
                "counts": {k: len(v) for k, v in splits.items()},
            },
        )
        self._write(out / "development_cases.json", splits)
        try:
            evidence, _ = self._batch(
                splits[self.schedule.induction], "nl", None, out / f"{self.schedule.induction}_nl"
            )
            stage = out / "card0"
            stage.mkdir()
            parent, evolution = self._propose(stage, evidence, None)
            self._write(stage / "evolution.json", evolution)
            history = []
            for number, split in enumerate(self.schedule.evolution, 1):
                stage = out / f"round{number}"
                stage.mkdir()
                evidence, _ = self._batch(
                    splits[split], "hybrid", parent, stage / "evolution_parent"
                )
                if self.design is not None:
                    baseline, _ = self._batch(splits[split], "nl", None, stage / "evolution_nl")
                    evidence = self.design.feedback(baseline, evidence)
                    self._write(stage / "feedback.json", evidence)
                candidate, evolution = self._propose(stage, evidence, parent)
                self._write(stage / "evolution.json", evolution)
                # Always run paired V arms, including KEEP, to expose execution variability.
                _, pm = self._batch(
                    splits[self.schedule.validation], "hybrid", parent, stage / "validation_parent"
                )
                _, cm = self._batch(
                    splits[self.schedule.validation],
                    "hybrid",
                    candidate,
                    stage / "validation_candidate",
                )
                promoted, reason = self.select(pm, cm)
                if not evolution["changed_paths"]:
                    promoted, reason = False, "no_semantic_change_keep_parent"
                selection = {
                    "round": number,
                    "parent": str(parent),
                    "candidate": str(candidate),
                    "parent_metrics": pm,
                    "candidate_metrics": cm,
                    "promoted": promoted,
                    "reason": reason,
                }
                self._write(stage / "selection.json", selection)
                history.append(selection)
                if promoted:
                    parent = candidate
            final = out / "final"
            if verify_sources is not None:
                verify_sources()
            final.mkdir()
            card = self._card(parent)
            card["task_card"].update(
                selection="validation-selected"
                if self.effect_valid
                else "FAKE-plumbing-only-NOT-efficacy",
                source_path=str(parent),
                source_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
            )
            import yaml

            frozen = final / f"{self.contract.task}_card.yaml"
            with frozen.open("x", encoding="utf-8") as handle:
                yaml.safe_dump(card, handle, allow_unicode=True, sort_keys=False)
            result = {
                "status": "completed",
                "effect_valid": self.effect_valid,
                "selected_card": str(frozen),
                "history": history,
                "heldout_evaluated": False,
            }
            self._write(out / "result.json", result)
            return result
        except Exception as exc:
            self._write(
                out / "failure.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            )
            raise
