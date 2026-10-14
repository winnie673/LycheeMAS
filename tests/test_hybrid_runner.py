"""状态机 runner 测试：计数、修正、重试、回退、传输失败。"""

from __future__ import annotations

from typing import Any

from lychee_mas.hybrid_comm.agents import RolePromptBuilder
from lychee_mas.hybrid_comm.backends import FakeBackend
from lychee_mas.hybrid_comm.metrics import TokenCounter
from lychee_mas.hybrid_comm.pipeline import HybridRunner
from lychee_mas.hybrid_comm.protocol import load_protocol_from_yaml
from lychee_mas.hybrid_comm.types import CommunicationMode


def _make_runner(script, mode="hybrid", **kwargs):
    suite = load_protocol_from_yaml("configs/hybrid_comm/protocols/protocol_p0_fields_only.yaml")
    backend = FakeBackend(script=script)
    return (
        HybridRunner(
            backend=backend,
            prompt_builder=RolePromptBuilder(),
            token_counter=TokenCounter(),
            suite=suite,
            mode=CommunicationMode(mode),
            **kwargs,
        ),
        backend,
    )


def _hybrid_solver(answer: str):
    return f"IR:step;R:{answer}"


def _hybrid_verifier(decision: str, error_type: str | None = None):
    if decision == "ACCEPT":
        return "CHK:ACCEPT"
    return f"CHK:FAIL;E:{error_type};REQ:recalculate"


def _hybrid_planner():
    return "G:target;P:plan"


def test_normal_path_three_calls() -> None:
    script = {
        "planner": _hybrid_planner(),
        "solver": _hybrid_solver("18"),
        "verifier": _hybrid_verifier("ACCEPT"),
    }
    runner, _ = _make_runner(script)
    result = runner.run_one("t1", "What is 9+9?", "18")
    assert result.score == 1.0
    assert len(result.model_calls) == 3
    assert not result.revision_triggered
    assert result.run_status == "completed"


def test_math500_keeps_full_expression_for_scoring() -> None:
    script = {
        "planner": _hybrid_planner(),
        "solver": _hybrid_solver(r"\frac{9}{256}"),
        "verifier": _hybrid_verifier("ACCEPT"),
    }
    runner, _ = _make_runner(script, scorer_kind="math500")
    result = runner.run_one("t-math", "Find the exact ratio.", r"\frac{9}{256}")
    assert result.normalized_final_answer == r"\frac{9}{256}"
    assert result.score == 1.0


def test_revise_path_four_calls() -> None:
    def _solver_with_revision(messages: list[dict[str, Any]]) -> str:
        # 第一次 solver 调用无 verifier 反馈；第二次携带 verifier 反馈。
        text = "\n".join(str(m.get("content", "")) for m in messages)
        if "verifier_to_solver" in text:
            return _hybrid_solver("18")
        return _hybrid_solver("17")

    script = {
        "planner": _hybrid_planner(),
        "solver": _solver_with_revision,
        "verifier": _hybrid_verifier("REVISE", "ARITH"),
    }
    runner, _ = _make_runner(script)
    result = runner.run_one("t1", "What is 9+9?", "18")
    assert result.score == 1.0
    assert len(result.model_calls) == 4
    assert result.revision_triggered
    solver_calls = [c for c in result.model_calls if c.role == "solver"]
    assert len(solver_calls) == 2
    assert result.final_answer == "18"


def test_format_retry_counts_extra_call() -> None:
    script = {
        ("solver", 0): "not json",
        ("solver", 1): _hybrid_solver("18"),
        "planner": _hybrid_planner(),
        "verifier": _hybrid_verifier("ACCEPT"),
    }
    runner, _ = _make_runner(script)
    result = runner.run_one("t1", "What is 9+9?", "18")
    assert result.score == 1.0
    solver_calls = [c for c in result.model_calls if c.role == "solver"]
    assert len(solver_calls) == 2
    assert solver_calls[0].parse_failed
    assert not solver_calls[1].parse_failed


def test_fallback_counts_extra_call_and_marks_fallback() -> None:
    script = {
        ("solver", 0): "not json",
        ("solver", 1): "still bad",
        ("solver", 2): _hybrid_solver("18"),
        "planner": _hybrid_planner(),
        "verifier": _hybrid_verifier("ACCEPT"),
    }
    runner, _ = _make_runner(script, fallback_call_limit=1)
    result = runner.run_one("t1", "What is 9+9?", "18")
    assert result.score == 1.0
    solver_msg = [m for m in result.messages if m.edge == "solver_to_verifier"][0]
    assert solver_msg.fallback


def test_transport_failure_counts_and_recover() -> None:
    script = {
        "planner": _hybrid_planner(),
        "solver": _hybrid_solver("0"),
        "verifier": _hybrid_verifier("ACCEPT"),
    }
    backend = FakeBackend(script=script, fail_role_attempts={("solver", 0)})
    suite = load_protocol_from_yaml("configs/hybrid_comm/protocols/protocol_p0_fields_only.yaml")
    runner = HybridRunner(
        backend=backend,
        prompt_builder=RolePromptBuilder(),
        token_counter=TokenCounter(),
        suite=suite,
        mode=CommunicationMode.HYBRID,
    )
    result = runner.run_one("t1", "What is 9+9?", "18")
    # solver 第一次传输失败，第二次使用脚本答案 0，不匹配 gold
    assert result.run_status == "completed"
    solver_calls = [c for c in result.model_calls if c.role == "solver"]
    assert any(c.transport_failed for c in solver_calls)
