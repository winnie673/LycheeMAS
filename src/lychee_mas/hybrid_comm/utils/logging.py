"""轻量级事件日志封装。"""

from __future__ import annotations

from typing import Any


class CaseLogger:
    """基于 ``JsonlSpanLogger`` 的 hybrid_comm 事件记录器。"""

    def __init__(self, path: str, *, run_fields: dict[str, Any] | None = None):
        self.path = path
        self._inner = None
        self._run_fields = run_fields or {}

    def _get_logger(self) -> Any:
        if self._inner is None:
            from lychee_mas.backends.spans import JsonlSpanLogger

            self._inner = JsonlSpanLogger(self.path, run_fields=self._run_fields)
        return self._inner

    def set_case(self, case_id: str | None, sample_index: int | None = None) -> None:
        self._get_logger().set_case(case_id, sample_index)

    def log(self, span_type: str, **fields: Any) -> str:
        return self._get_logger().log(span_type, **fields)
