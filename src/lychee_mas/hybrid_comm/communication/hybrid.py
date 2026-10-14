"""混合消息（Hybrid）通信的提示模板与输出 schema 辅助。"""

from __future__ import annotations


def solver_output_schema() -> str:
    return "Output direct fields: IR:concise intermediate;R:answer;C:optional constraint"


def verifier_output_schema() -> str:
    return "Output CHK:ACCEPT, or CHK:REVISE;E:ARITH;REQ:recompute X"


def planner_output_schema() -> str:
    return "Output direct fields: K:facts;G:target;P:plan;C:optional constraint"
