"""生成原语的抽象基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..types import GenResult


class BackendError(Exception):
    """后端调用失败的显式异常。"""


class Backend(ABC):
    """所有 LLM 后端（fake / API / HF）的统一接口。"""

    @abstractmethod
    def generate_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> GenResult:
        """返回生成文本与 usage 元数据。"""

    def close(self) -> None:
        """可选的资源释放钩子。"""
