"""协议加载与 schema 测试。"""

from __future__ import annotations

import pytest
from lychee_mas.hybrid_comm.protocol import load_protocol_from_yaml


@pytest.fixture
def protocol_path() -> str:
    return "configs/hybrid_comm/protocols/protocol_p0_fields_only.yaml"


def test_load_protocol_p0(protocol_path: str) -> None:
    suite = load_protocol_from_yaml(protocol_path)
    assert suite.version == "P0-FIELDS-ONLY"
    assert suite.meta.get("protocol_hash", "").startswith("sha256:")


def test_planner_card(protocol_path: str) -> None:
    suite = load_protocol_from_yaml(protocol_path)
    card = suite.card_for("planner_to_solver")
    assert set(card.allowed_fields) >= {"K", "G", "P", "C", "UNIT"}
    assert set(card.required_fields) == {"G", "P"}
    assert card.canonical_order()[:3] == ["K", "G", "P"]
    assert card.message_format["structure"] == "fields_only"


def test_solver_card(protocol_path: str) -> None:
    suite = load_protocol_from_yaml(protocol_path)
    card = suite.card_for("solver_to_verifier")
    assert "R" in card.required_fields
    assert "R" in card.allowed_fields
    assert card.message_format["wire_syntax"] == "direct"


def test_verifier_card(protocol_path: str) -> None:
    suite = load_protocol_from_yaml(protocol_path)
    card = suite.card_for("verifier_to_solver")
    assert set(card.required_fields) == {"CHK"}
    assert "CHK" in card.allowed_fields


def test_legacy_nl_policy_is_rejected(tmp_path) -> None:
    path = tmp_path / "legacy.yaml"
    path.write_text(
        "version: old\nnl_policy:\n  max_sentences: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no longer supported"):
        load_protocol_from_yaml(str(path))
