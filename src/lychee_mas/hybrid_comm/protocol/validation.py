"""冷启动与逐代协议更新的结构约束。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .loader import _load_yaml, load_protocol_from_yaml

EDGES = ("planner_to_solver", "solver_to_verifier", "verifier_to_solver")
CORE_REQUIRED = {
    "planner_to_solver": {"G", "P"},
    "solver_to_verifier": {"R"},
    "verifier_to_solver": {"CHK"},
}
ALLOWED_UPDATE_PREFIXES = (
    "planner_to_solver.field_rules",
    "planner_to_solver.required_fields",
    "planner_to_solver.examples",
    "solver_to_verifier.field_rules",
    "solver_to_verifier.required_fields",
    "solver_to_verifier.examples",
    "verifier_to_solver.field_rules",
    "verifier_to_solver.required_fields",
    "verifier_to_solver.examples",
)


def protocol_diff(parent: Any, candidate: Any, prefix: str = "") -> list[str]:
    """返回两个解析后 YAML 对象的叶级递归路径差异。"""
    if isinstance(parent, dict) and isinstance(candidate, dict):
        paths: list[str] = []
        for key in sorted(set(parent) | set(candidate)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in parent or key not in candidate:
                paths.append(path)
            else:
                paths.extend(protocol_diff(parent[key], candidate[key], path))
        return paths
    if parent != candidate:
        return [prefix]
    return []


def validate_protocol_update(
    parent_path: str | Path,
    candidate_path: str | Path,
    *,
    expected_version: str,
    max_changes: int = 3,
    cold_start: bool = False,
) -> tuple[bool, str, list[str]]:
    """验证固定字段结构、fields-only 消息格式和允许修改路径。"""
    parent_path = Path(parent_path)
    candidate_path = Path(candidate_path)
    try:
        parent_suite = load_protocol_from_yaml(str(parent_path))
        candidate_suite = load_protocol_from_yaml(str(candidate_path))
        parent = _load_yaml(str(parent_path))
        candidate = _load_yaml(str(candidate_path))
    except Exception as exc:
        return False, f"YAML load failed: {exc}", []

    if candidate_suite.version != expected_version:
        return False, f"version must be {expected_version}", []
    for edge in EDGES:
        p_card = parent_suite.card_for(edge)
        c_card = candidate_suite.card_for(edge)
        if set(p_card.allowed_fields) != set(c_card.allowed_fields):
            return False, f"{edge}: allowed_fields changed", []
        if not CORE_REQUIRED[edge].issubset(set(c_card.required_fields)):
            return False, f"{edge}: core required fields removed", []
        if not set(c_card.required_fields).issubset(set(c_card.allowed_fields)):
            return False, f"{edge}: required_fields contain unknown fields", []
        if c_card.message_format.get("structure") != "fields_only":
            return False, f"{edge}: message structure must remain fields_only", []
    if candidate_suite.message_format.get("structure") != "fields_only":
        return False, "global message structure must remain fields_only", []

    diffs = [path for path in protocol_diff(parent, candidate) if path != "version"]
    if cold_start:
        return True, "ok", diffs
    illegal = [
        path
        for path in diffs
        if not any(
            path == allowed or path.startswith(f"{allowed}.")
            for allowed in ALLOWED_UPDATE_PREFIXES
        )
    ]
    if illegal:
        return False, f"illegal change paths: {illegal}", diffs
    if len(diffs) > max_changes:
        return False, f"too many change paths: {len(diffs)} > {max_changes}", diffs
    return True, "ok", diffs
