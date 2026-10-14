"""脚本化假后端：确定性输出，用于状态机与计数验证。"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from ..types import GenResult
from .base import Backend, BackendError


class FakeBackend(Backend):
    """通过脚本字典返回固定输出的后端。

    `script` 的键支持 (role, attempt_index) 或 role，值可以是字符串或 Callable。
    未命中脚本时根据 prompt 返回默认 Pure-NL JSON 或 fields-only Hybrid 消息。
    """

    def __init__(
        self,
        script: dict[str | tuple[str, int], str | Callable[..., str]] | None = None,
        *,
        prompt_tokens: int = 10,
        completion_tokens: int = 20,
        latency_s: float = 0.001,
        fail_role_attempts: set[tuple[str, int]] | None = None,
    ):
        self.script = script or {}
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.latency_s = latency_s
        self.fail_role_attempts = fail_role_attempts or set()

    def generate_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> GenResult:
        role = self._extract_role(messages)
        attempt = self._attempt_from_meta(messages)
        key = (role, attempt)

        if key in self.fail_role_attempts:
            raise BackendError(f"scripted transport failure for {role} attempt {attempt}")

        raw = self._lookup(key, role, messages)
        time.sleep(self.latency_s)
        return GenResult(
            text=str(raw),
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            latency_s=self.latency_s,
        )

    @staticmethod
    def _extract_role(messages: list[dict[str, Any]]) -> str:
        """从系统提示中的显式 role= 标记推断当前角色名。"""
        for m in messages:
            content = str(m.get("content", ""))
            if "role=planner" in content:
                return "planner"
            if "role=solver" in content:
                return "solver"
            if "role=verifier" in content:
                return "verifier"
        return "unknown"

    @staticmethod
    def _attempt_from_meta(messages: list[dict[str, Any]]) -> int:
        """允许通过最后一条消息的 meta.attempt 传递当前尝试序号。"""
        if not messages:
            return 0
        meta = messages[-1].get("meta") or {}
        return int(meta.get("attempt", 0))

    def _lookup(
        self,
        key: tuple[str, int],
        role: str,
        messages: list[dict[str, Any]],
    ) -> str:
        value = self.script.get(key)
        if value is None:
            value = self.script.get(role)
        if value is None:
            system_text = "\n".join(
                str(message.get("content", ""))
                for message in messages
                if message.get("role") == "system"
            )
            if "Use structured fields only" in system_text:
                return {
                    "planner": "G:target;P:solve directly",
                    "solver": "IR:default fake calculation;R:0",
                    "verifier": "CHK:ACCEPT",
                }.get(role, "R:0")
            return json.dumps(
                {
                    "planner": {"message": "Solve directly."},
                    "solver": {"message": "Default fake answer is 0.", "answer": "0"},
                    "verifier": {
                        "decision": "ACCEPT",
                        "error_type": None,
                        "message": "",
                    },
                }.get(role, {"message": "default fake output"})
            )
        if callable(value):
            return value(messages)
        return str(value)
