"""通信表示层：NL / Hybrid 消息的解析、校验、序列化与反序列化。"""

from __future__ import annotations

from .parser import ParsedPayload, PayloadParser
from .serializer import PayloadSerializer
from .validator import PayloadValidator

__all__ = ["PayloadParser", "ParsedPayload", "PayloadSerializer", "PayloadValidator"]
