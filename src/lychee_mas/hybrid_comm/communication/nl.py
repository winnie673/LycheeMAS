"""纯自然语言（NL）通信的提示模板与输出 schema 辅助。"""

from __future__ import annotations


def solver_output_schema() -> str:
    return (
        'Output JSON: {"message": "给 Verifier 的简洁自然语言推理与候选答案", '
        '"answer": "最终数值答案"}'
    )


def verifier_output_schema() -> str:
    return (
        'Output JSON: {"decision": "ACCEPT 或 REVISE", '
        '"error_type": "ACCEPT 时为 null，REVISE 时从指定枚举中选一个", '
        '"message": "ACCEPT 时可为空，REVISE 时为具体反馈"}'
    )


def planner_output_schema() -> str:
    return 'Output JSON: {"message": "给 Solver 的简洁计划与关键事实"}'
