"""OpenAI 兼容 API 后端（默认 DeepSeek）。"""

from __future__ import annotations

import os
from typing import Any

from ..types import GenResult
from .base import Backend, BackendError

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


class DeepSeekBackend(Backend):
    """DeepSeek API 后端。

    密钥只从环境变量 ``DEEPSEEK_API_KEY`` 读取；模型标识可通过 ``DEEPSEEK_MODEL``
    或构造参数覆盖。温度默认 0，对应贪心策略；具体生效性以 API 为准。
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key_env: str = "DEEPSEEK_API_KEY",
        base_url: str | None = None,
        timeout: float = 120.0,
        temperature: float = 0.0,
        max_retries: int = 0,
    ):
        self.model_name = model or os.environ.get("DEEPSEEK_MODEL") or DEFAULT_MODEL
        self.api_key_env = api_key_env
        self.base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL
        self.timeout = timeout
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        # Check for the API key before importing the optional `openai` package so
        # that the absence of the key produces the expected BackendError even when
        # `openai` is not installed.
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise BackendError(
                f"API key not found; set the {self.api_key_env} environment variable. "
                "The key must not be written in code, config, or logs."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - runtime extra
            raise ImportError(
                "API backend requires `openai`; install with `uv pip install -e '.[runtime]'`"
            ) from exc

        self._client = OpenAI(api_key=api_key, base_url=self.base_url, timeout=self.timeout)
        return self._client

    def generate_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> GenResult:
        client = self._get_client()
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "max_tokens": max_new_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if stop:
            payload["stop"] = stop

        try:
            import time

            t0 = time.time()
            response = client.chat.completions.create(**payload)
            latency = time.time() - t0
        except Exception as exc:
            raise BackendError(f"DeepSeek API request failed: {exc}") from exc

        choice = response.choices[0] if response.choices else None
        text = ""
        if choice and choice.message:
            text = choice.message.content or ""

        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        unknown = usage is None

        return GenResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=round(latency, 4),
            usage_unknown=unknown,
        )

    def close(self) -> None:
        self._client = None
