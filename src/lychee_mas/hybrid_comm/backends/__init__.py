"""统一生成接口：假后端、API 后端、HF 本地后端。"""

from __future__ import annotations

from .api import DeepSeekBackend
from .base import Backend, BackendError
from .fake import FakeBackend

__all__ = ["Backend", "BackendError", "DeepSeekBackend", "FakeBackend"]
