"""Method-local architecture and callback contracts; no backend or graph imports."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class ProtocolArchitecture:
    """A message contract, not a graph builder or an arbitrary-topology adapter."""

    name: str
    roles: tuple[tuple[str, str], ...]
    edges: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]
    max_semantic_revisions: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.roles, tuple)
            or not isinstance(self.edges, tuple)
            or any(not isinstance(role, tuple) or len(role) != 2 for role in self.roles)
            or any(
                not isinstance(edge, tuple)
                or len(edge) != 3
                or not isinstance(edge[1], tuple)
                or not isinstance(edge[2], tuple)
                for edge in self.edges
            )
        ):
            raise ValueError("architecture roles/edges must be immutable tuples")
        roles = dict(self.roles)
        if not self.name or len(roles) != len(self.roles) or not roles:
            raise ValueError("architecture needs unique roles and an identifier")
        if type(self.max_semantic_revisions) is not int or self.max_semantic_revisions < 0:
            raise ValueError("revision budget must be nonnegative")
        if any(
            not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", role)
            or "_to_" in role
            or not description.strip()
            for role, description in self.roles
        ):
            raise ValueError("invalid role identifier/description")
        seen = set()
        for edge, allowed, required in self.edges:
            pair = edge.split("_to_")
            if (
                len(pair) != 2
                or any(role not in roles for role in pair)
                or edge in seen
                or not allowed
                or len(set(allowed)) != len(allowed)
                or len(set(required)) != len(required)
                or not set(required) <= set(allowed)
                or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key) for key in allowed)
            ):
                raise ValueError("invalid architecture edge/field interface")
            seen.add(edge)
        if not seen:
            raise ValueError("architecture needs communication edges")

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "roles": dict(self.roles),
            "edges": {
                edge: {"allowed_fields": list(allowed), "required_fields": list(required)}
                for edge, allowed, required in self.edges
            },
            "max_semantic_revisions": self.max_semantic_revisions,
        }

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()

    def check_card(self, raw: dict, task: str, *, require_identity: bool = False) -> None:
        scope = raw.get("task_card", {})
        if not isinstance(scope, dict):
            raise ValueError("card scope must be a mapping")
        roles, edges = scope.get("roles", []), scope.get("edges", [])
        if (
            scope.get("task") != task
            or not isinstance(roles, list)
            or not all(isinstance(role, str) for role in roles)
            or len(roles) != len(self.roles)
            or set(roles) != set(dict(self.roles))
            or not isinstance(edges, list)
            or not all(isinstance(edge, str) for edge in edges)
            or len(edges) != len(self.edges)
            or set(edges) != {edge for edge, _, _ in self.edges}
        ):
            raise ValueError("card task/roles/edges do not match execution architecture")
        identity = scope.get("architecture")
        fingerprint = scope.get("architecture_sha256")
        if (
            identity not in (None, self.name)
            or fingerprint not in (None, self.fingerprint)
            or (require_identity and (identity is None or fingerprint is None))
        ):
            raise ValueError("card architecture identity/fingerprint mismatch")
        for edge, allowed, required in self.edges:
            section = raw.get(edge, {})
            if (
                not isinstance(section, dict)
                or section.get("allowed_fields") != list(allowed)
                or section.get("required_fields") != list(required)
            ):
                raise ValueError(f"card field interface mismatch on {edge}")


@dataclass(frozen=True)
class FieldConstraint:
    """Safe declarative guards, never executable code from a card."""

    name: str
    choices: tuple[str, ...] = ()
    forbidden_values: tuple[str, ...] = ()
    pattern: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("constraint needs a field name")
        for values in (self.choices, self.forbidden_values):
            if not isinstance(values, tuple) or any(not isinstance(v, str) for v in values):
                raise ValueError("constraint values must be string tuples")
        if self.pattern is not None:
            if not isinstance(self.pattern, str):
                raise ValueError("constraint pattern must be a string")
            re.compile(self.pattern)
        if self.error is not None and not isinstance(self.error, str):
            raise ValueError("constraint error must be a string")

    def check(self, fields: dict[str, str]) -> None:
        if self.name not in fields:
            return
        value = fields[self.name]
        if (
            (self.choices and value not in self.choices)
            or value.strip().lower() in self.forbidden_values
            or (self.pattern is not None and re.fullmatch(self.pattern, value) is None)
        ):
            message = (
                self.error.replace("{value}", repr(value))
                if self.error
                else (f"invalid value for field {self.name!r}")
            )
            raise ValueError(message)


@dataclass(frozen=True)
class ConditionalConstraint:
    field: str
    equals: str
    required: tuple[str, ...] = ()
    only: tuple[str, ...] = ()
    error: str | None = None
    forbidden: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.field, str)
            or not isinstance(self.equals, str)
            or not isinstance(self.required, tuple)
            or not isinstance(self.only, tuple)
            or not isinstance(self.forbidden, tuple)
            or any(not isinstance(v, str) for v in (*self.required, *self.only, *self.forbidden))
            or (self.error is not None and not isinstance(self.error, str))
        ):
            raise ValueError("invalid conditional constraint")

    def check(self, fields: dict[str, str]) -> None:
        if fields.get(self.field) == self.equals:
            missing = [name for name in self.required if not fields.get(name)]
            if (
                any(not fields.get(name) for name in self.required)
                or (self.only and set(fields) != set(self.only))
                or any(name in fields for name in self.forbidden)
            ):
                raise ValueError(
                    self.error.replace("{missing}", repr(missing))
                    if self.error
                    else "conditional field constraint violated"
                )


@dataclass(frozen=True)
class EdgeValidation:
    edge: str
    fields: tuple[FieldConstraint, ...] = ()
    conditions: tuple[ConditionalConstraint, ...] = ()
    example_only: tuple[FieldConstraint, ...] = ()
    required_example_fields: tuple[str, ...] = ()
    instruction_lines: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, tuple)
            for value in (
                self.fields,
                self.conditions,
                self.example_only,
                self.required_example_fields,
                self.instruction_lines,
            )
        ):
            raise ValueError("edge validation collections must be immutable tuples")
        if any(
            not isinstance(rule, FieldConstraint) for rule in (*self.fields, *self.example_only)
        ) or any(not isinstance(rule, ConditionalConstraint) for rule in self.conditions):
            raise ValueError("invalid edge validation operations")


@dataclass(frozen=True)
class ProtocolContract:
    """Experiment-supplied semantics; serializable into new cards for standalone loading."""

    task: str
    architecture: ProtocolArchitecture
    field_meanings: tuple[tuple[str, str], ...]
    edge_validation: tuple[EdgeValidation, ...] = ()
    reserved_fields: tuple[str, ...] = ()
    fixed_card_options: tuple[tuple[str, bool | str], ...] = ()
    generation_guidance: str = ""

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, tuple)
            for value in (
                self.field_meanings,
                self.edge_validation,
                self.reserved_fields,
                self.fixed_card_options,
            )
        ) or not isinstance(self.generation_guidance, str):
            raise ValueError("invalid protocol contract collections/guidance")
        if (
            any(
                not isinstance(pair, tuple) or len(pair) != 2
                for pair in (*self.field_meanings, *self.fixed_card_options)
            )
            or any(not isinstance(rule, EdgeValidation) for rule in self.edge_validation)
            or any(not isinstance(name, str) or not name for name in self.reserved_fields)
        ):
            raise ValueError("invalid protocol contract entries")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", self.task):
            raise ValueError("invalid task identifier")
        interfaces = {edge: set(allowed) for edge, allowed, _ in self.architecture.edges}
        meanings = dict(self.field_meanings)
        if (
            len(meanings) != len(self.field_meanings)
            or set(meanings) != set().union(*interfaces.values())
            or any(not isinstance(v, str) or not v.strip() for v in meanings.values())
        ):
            raise ValueError("field meanings must cover exactly the declared fields")
        seen = set()
        for validation in self.edge_validation:
            if validation.edge not in interfaces or validation.edge in seen:
                raise ValueError("validation refers to an unknown/duplicate edge")
            seen.add(validation.edge)
            names = interfaces[validation.edge]
            if (
                any(
                    rule.name not in names
                    for rule in (*validation.fields, *validation.example_only)
                )
                or not set(validation.required_example_fields) <= names
                or any(
                    rule.field not in names
                    or not set((*rule.required, *rule.only, *rule.forbidden)) <= names
                    for rule in validation.conditions
                )
            ):
                raise ValueError("validation refers to an undeclared field")
            if any(not isinstance(line, str) for line in validation.instruction_lines):
                raise ValueError("instruction lines must be strings")
        reserved = {"version", "task_card", "validation_contract", "message_format", "value_policy"}
        if len(dict(self.fixed_card_options)) != len(self.fixed_card_options) or any(
            key in reserved or key in interfaces or not isinstance(value, (str, bool))
            for key, value in self.fixed_card_options
        ):
            raise ValueError("invalid fixed card option")

    def as_dict(self) -> dict:
        validations = [asdict(rule) for rule in self.edge_validation]
        # Preserve historical contract serialization/fingerprints byte-for-byte in JSON.
        for validation in validations:
            for condition in validation["conditions"]:
                if not condition["forbidden"]:
                    del condition["forbidden"]
        return {
            "task": self.task,
            "architecture": self.architecture.as_dict(),
            "field_meanings": dict(self.field_meanings),
            "edge_validation": validations,
            "reserved_fields": list(self.reserved_fields),
            "fixed_card_options": dict(self.fixed_card_options),
            "generation_guidance": self.generation_guidance,
        }

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_dict(cls, value: dict) -> ProtocolContract:
        """Reject unknown operations; convert JSON/YAML lists into immutable rule values."""
        if not isinstance(value, dict) or set(value) != {
            "task",
            "architecture",
            "field_meanings",
            "edge_validation",
            "reserved_fields",
            "fixed_card_options",
            "generation_guidance",
        }:
            raise ValueError("invalid serialized protocol contract")
        try:
            arch = value["architecture"]
            if set(arch) != {"name", "roles", "edges", "max_semantic_revisions"}:
                raise ValueError("invalid serialized architecture")
            architecture = ProtocolArchitecture(
                arch["name"],
                tuple(arch["roles"].items()),
                tuple(
                    (edge, tuple(fields["allowed_fields"]), tuple(fields["required_fields"]))
                    for edge, fields in arch["edges"].items()
                ),
                arch["max_semantic_revisions"],
            )

            def field_rule(item):
                return FieldConstraint(
                    **{
                        k: tuple(v) if k in {"choices", "forbidden_values"} else v
                        for k, v in item.items()
                    }
                )

            rules = []
            for item in value["edge_validation"]:
                if set(item) != {
                    "edge",
                    "fields",
                    "conditions",
                    "example_only",
                    "required_example_fields",
                    "instruction_lines",
                }:
                    raise ValueError("invalid edge validation")
                rules.append(
                    EdgeValidation(
                        edge=item["edge"],
                        fields=tuple(field_rule(v) for v in item["fields"]),
                        conditions=tuple(
                            ConditionalConstraint(
                                **{
                                    k: tuple(v) if k in {"required", "only", "forbidden"} else v
                                    for k, v in c.items()
                                }
                            )
                            for c in item["conditions"]
                        ),
                        example_only=tuple(field_rule(v) for v in item["example_only"]),
                        required_example_fields=tuple(item["required_example_fields"]),
                        instruction_lines=tuple(item["instruction_lines"]),
                    )
                )
            return cls(
                value["task"],
                architecture,
                tuple(value["field_meanings"].items()),
                tuple(rules),
                tuple(value["reserved_fields"]),
                tuple(value["fixed_card_options"].items()),
                value["generation_guidance"],
            )
        except (TypeError, KeyError, AttributeError, re.error) as exc:
            raise ValueError("invalid serialized protocol contract") from exc

    def check_card(self, raw: dict, *, require_identity: bool = False) -> None:
        self.architecture.check_card(raw, self.task, require_identity=require_identity)
        if any(raw.get(key) != value for key, value in self.fixed_card_options):
            raise ValueError("card changed fixed experiment options")
        embedded = raw.get("validation_contract")
        if embedded is not None and self.from_dict(embedded).fingerprint != self.fingerprint:
            raise ValueError("card validation contract mismatch")
        identity = raw["task_card"].get("contract_sha256")
        if identity not in (None, self.fingerprint) or (require_identity and identity is None):
            raise ValueError("card contract fingerprint mismatch")

    def validation_for(self, edge: str) -> EdgeValidation:
        return next(
            (rule for rule in self.edge_validation if rule.edge == edge), EdgeValidation(edge)
        )


def load_protocol_schema(path: str | Path) -> dict:
    """Read design data only; importing the framework does not import YAML."""
    import yaml

    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("protocol schema must be a mapping")
    return value


def build_contract_from_schema(
    schema: dict,
    *,
    task: str,
    architecture_name: str,
    roles: tuple[tuple[str, str], ...],
    edges: tuple[str, ...],
    control_fields: dict[str, str] | None = None,
    runtime_aliases: dict[str, dict[str, str]] | None = None,
    task_field_meanings: dict[str, str] | None = None,
    reserved_fields: tuple[str, ...] = (),
    fixed_card_options: tuple[tuple[str, bool | str], ...] = (),
    generation_guidance: str = "",
    max_semantic_revisions: int = 1,
) -> ProtocolContract:
    """Compile supported declarative schema operations, without task/role constants.

    Prose policies remain design guidance, not executable validation. A field shared
    across edges keeps every edge's meaning instead of last-write-wins flattening.
    Receive-only aliases extend derived guards; they never modify the input schema.
    """
    controls, aliases = control_fields or {}, runtime_aliases or {}
    if (
        not isinstance(controls, dict)
        or not isinstance(aliases, dict)
        or any(not isinstance(value, dict) for value in aliases.values())
    ):
        raise ValueError("control/alias configuration must be mappings")
    if not isinstance(schema, dict) or schema.get("message_format") != {
        "structure": "fields_only",
        "wire_syntax": "direct",
        "field_separator": ";",
    }:
        raise ValueError("unsupported schema message format")
    if set(controls) - set(edges) or set(aliases) - set(edges):
        raise ValueError("control/alias refers to an unknown edge")
    if {key for key in schema if "_to_" in key} != set(edges):
        raise ValueError("schema edges do not match the experiment")

    def names(value: object, label: str) -> tuple[str, ...]:
        if not isinstance(value, list) or any(not isinstance(v, str) or not v for v in value):
            raise ValueError(f"{label} must be a string list")
        if len(value) != len(set(value)):
            raise ValueError(f"duplicate {label}")
        return tuple(value)

    interfaces, validations, meanings = [], [], {}
    for edge in edges:
        spec = schema[edge]
        if not isinstance(spec, dict) or set(spec) - {
            "allowed_fields",
            "required_fields",
            "field_semantics",
            "optional_field_policy",
            "value_constraints",
            "conditional_required",
            "conditional_forbidden",
        }:
            raise ValueError(f"unsupported edge schema: {edge}")
        allowed = names(spec.get("allowed_fields"), "allowed_fields")
        required = names(spec.get("required_fields"), "required_fields")
        if not set(required) <= set(allowed):
            raise ValueError("required fields must be allowed")
        semantics = spec.get("field_semantics", {})
        if not isinstance(semantics, dict) or set(semantics) != set(allowed):
            raise ValueError(f"field semantics must cover exactly {edge}")
        for field, text in semantics.items():
            if not isinstance(text, str) or not text.strip():
                raise ValueError("empty field semantics")
            meanings.setdefault(field, []).append((edge, text))
        policy, values = spec.get("optional_field_policy", {}), spec.get("value_constraints", {})
        if not isinstance(policy, dict) or not isinstance(values, dict):
            raise ValueError("field policies/constraints must be mappings")
        if (set(policy) | set(values)) - set(allowed):
            raise ValueError("constraint refers to an undeclared field")
        fields = []
        for field in allowed:
            option, constraint = policy.get(field, {}), values.get(field, {})
            if (
                not isinstance(option, dict)
                or set(option) - {"omit_when_unused", "forbidden_placeholder_values"}
                or option.get("omit_when_unused", True) is not True
                or not isinstance(constraint, dict)
                or set(constraint) - {"allowed_values", "pattern"}
            ):
                raise ValueError("unsupported field policy/constraint")
            choices = names(constraint.get("allowed_values", []), "allowed_values")
            if "allowed_values" in constraint and not choices:
                raise ValueError("enum must not be empty")
            forbidden = names(option.get("forbidden_placeholder_values", []), "placeholder values")
            if choices or forbidden or "pattern" in constraint:
                fields.append(
                    FieldConstraint(
                        field,
                        choices=choices,
                        forbidden_values=tuple(v.lower() for v in forbidden),
                        pattern=constraint.get("pattern"),
                    )
                )
        conditions = []
        control = controls.get(edge)
        for key, operation in (
            ("conditional_required", "required"),
            ("conditional_forbidden", "forbidden"),
        ):
            table = spec.get(key, {})
            if not isinstance(table, dict):
                raise ValueError("conditional constraints must be mappings")
            for value, dependent in table.items():
                choices = next((r.choices for r in fields if r.name == control), ())
                if control not in allowed or value not in choices:
                    raise ValueError("condition needs a declared control enum")
                conditions.append(
                    ConditionalConstraint(control, value, **{operation: names(dependent, key)})
                )
        for alias, canonical in aliases.get(edge, {}).items():
            rule = next((r for r in fields if r.name == control), None)
            if (
                rule is None
                or not isinstance(alias, str)
                or not alias
                or alias in rule.choices
                or canonical not in rule.choices
                or canonical in aliases.get(edge, {})
            ):
                raise ValueError("alias must point to a canonical control value")
            fields[fields.index(rule)] = replace(rule, choices=(*rule.choices, alias))
            conditions.extend(
                replace(c, equals=alias) for c in tuple(conditions) if c.equals == canonical
            )
        for value in {c.equals for c in conditions}:
            needed = set(required).union(*(c.required for c in conditions if c.equals == value))
            banned = set().union(*(c.forbidden for c in conditions if c.equals == value))
            if needed & banned:
                raise ValueError("contradictory required/forbidden fields")
        interfaces.append((edge, allowed, required))
        validations.append(EdgeValidation(edge, fields=tuple(fields), conditions=tuple(conditions)))
    overrides = task_field_meanings or {}
    if set(overrides) - set(meanings) or any(
        not isinstance(text, str) or not text.strip() for text in overrides.values()
    ):
        raise ValueError("invalid task field meaning override")
    merged = []
    for field, entries in meanings.items():
        text = (
            entries[0][1]
            if len({v for _, v in entries}) == 1
            else "\n".join(f"{edge}: {meaning}" for edge, meaning in entries)
        )
        if field in overrides:
            text += "\nTask requirement: " + overrides[field]
        merged.append((field, text))
    return ProtocolContract(
        task,
        ProtocolArchitecture(architecture_name, roles, tuple(interfaces), max_semantic_revisions),
        tuple(merged),
        tuple(validations),
        reserved_fields,
        fixed_card_options,
        generation_guidance,
    )


def assert_runtime_compatible(expected: ProtocolContract, historical: ProtocolContract) -> None:
    """Explicit predicate comparison; this never substitutes or rewrites a fingerprint."""
    if (
        expected.task != historical.task
        or expected.architecture != historical.architecture
        or expected.fixed_card_options != historical.fixed_card_options
        or expected.reserved_fields != historical.reserved_fields
    ):
        raise ValueError("historical contract architecture/options are incompatible")

    def guards(contract: ProtocolContract, edge: str, allowed: tuple, required: tuple) -> tuple:
        rule = contract.validation_for(edge)

        def field_guards(items):
            return frozenset(
                (v.name, tuple(sorted(v.choices)), tuple(sorted(v.forbidden_values)), v.pattern)
                for v in items
            )

        conditions = {}
        for item in rule.conditions:
            needed, banned = conditions.setdefault((item.field, item.equals), (set(), set()))
            needed.update(item.required)
            needed.update(item.only)
            banned.update(item.forbidden)
            if item.only:
                banned.update(set(allowed) - set(item.only))
        normalized = frozenset(
            (field, value, tuple(sorted(needed - set(required))), tuple(sorted(banned)))
            for (field, value), (needed, banned) in conditions.items()
        )
        return (
            field_guards(rule.fields),
            normalized,
            field_guards(rule.example_only),
            frozenset(rule.required_example_fields),
        )

    for edge, allowed, required in expected.architecture.edges:
        if guards(expected, edge, allowed, required) != guards(historical, edge, allowed, required):
            raise ValueError(f"historical contract validation is incompatible on {edge}")


@dataclass(frozen=True)
class EvolutionSchedule:
    induction: str
    evolution: tuple[str, ...]
    validation: str
    heldout: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.evolution, tuple) or not isinstance(self.heldout, tuple):
            raise ValueError("split schedule collections must be immutable tuples")
        names = (self.induction, *self.evolution, self.validation, *self.heldout)
        if (
            not self.evolution
            or any(not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("split schedule needs unique safe names and evolution rounds")


# Generation returns a backend-shaped result with .text and optional usage.
Generate = Callable[..., Any]
# Execute never receives gold; the caller scores only after execution.
Execute = Callable[[dict, str, Path | None, Path], dict]
Score = Callable[[str, str], float]


class EvidenceDesign(Protocol):
    """Optional experiment-injected legacy-equivalent review/diagnosis/patch strategy."""

    def parse(self, text: str) -> dict: ...
    def initialize(self, traces: list[dict], capability: str) -> str: ...
    def materialize(self, value: dict, version: str) -> dict: ...
    def review(self, card: dict, traces: list[dict], capability: str) -> str: ...
    def feedback(self, nl: list[dict], hybrid: list[dict]) -> dict: ...
    def diagnose(self, card: dict, feedback: dict, capability: str) -> str: ...
    def eligible(self, diagnosis: dict, feedback: dict) -> set[str]: ...
    def evolve(
        self, card: dict, diagnosis: dict, feedback: dict, eligible: set[str], capability: str
    ) -> str: ...
    def apply(
        self, card: dict, decision: dict, eligible: set[str], version: str
    ) -> tuple[dict, list]: ...
