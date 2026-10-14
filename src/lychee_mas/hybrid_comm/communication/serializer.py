"""Hybrid 与 NL payload 的序列化 / 反序列化。"""

from __future__ import annotations

from typing import Optional

from ..protocol.schema import ProtocolCard


class PayloadSerializer:
    """将结构化字段或 NL 文本转成实际传输的 payload 字符串。"""

    _FLAT_ESCAPE_MAP = {
        "\\": "\\\\",
        "\n": "\\n",
        ";": "\\;",
    }
    _ANGLE_ESCAPE_MAP = {
        "\\": "\\\\",
        "\n": "\\n",
        ">": "\\>",
    }
    _UNESCAPE_MAP = {
        "\\": "\\",
        "n": "\n",
        ";": ";",
        ">": ">",
    }

    @classmethod
    def escape_value(cls, value: str, *, fmt: str = "flat") -> str:
        """对值中的分隔符进行可逆转义。flat 格式转义分号；angle 格式转义换行。"""
        mapping = cls._ANGLE_ESCAPE_MAP if fmt == "angle" else cls._FLAT_ESCAPE_MAP
        out: list[str] = []
        for ch in value:
            out.append(mapping.get(ch, ch))
        return "".join(out)

    @classmethod
    def unescape_value(cls, value: str) -> str:
        """还原转义序列。"""
        out: list[str] = []
        i = 0
        n = len(value)
        while i < n:
            if value[i] == "\\" and i + 1 < n and value[i + 1] in cls._UNESCAPE_MAP:
                out.append(cls._UNESCAPE_MAP[value[i + 1]])
                i += 2
            else:
                out.append(value[i])
                i += 1
        return "".join(out)

    @classmethod
    def _line_format(cls, card: ProtocolCard | None) -> str:
        """从卡片 field_rules 中推断物理行格式。"""
        if card is None:
            return "flat"
        fmt = str(card.field_rules.get("line_format", "")).lower()
        if "<field:content>" in fmt:
            return "angle"
        return "flat"

    @classmethod
    def serialize_hybrid(
        cls,
        fields: dict[str, str],
        card: ProtocolCard | None,
        *,
        canonical: bool = True,
    ) -> str:
        """按卡片 canonical_order 输出字段序列。

        默认格式：`KEY:value;KEY:value`。
        若卡片 field_rules 中声明 `line_format: <FIELD:content>`，则每字段独占一行：
            <FIELD:content>
        """
        order = card.canonical_order() if canonical and card else list(fields.keys())
        fmt = cls._line_format(card)
        parts: list[str] = []
        for key in order:
            if key not in fields:
                continue
            value = str(fields[key])
            if value == "":
                continue
            if fmt == "angle":
                parts.append(f"<{key}:{cls.escape_value(value, fmt='angle')}>")
            else:
                parts.append(f"{key}:{cls.escape_value(value, fmt='flat')}")
        if fmt == "angle":
            return "\n".join(parts)
        return ";".join(parts)

    @classmethod
    def deserialize_hybrid(
        cls, payload: str, card: Optional[ProtocolCard] = None
    ) -> dict[str, str]:
        """解析 `KEY:value;KEY:value` 或 `<FIELD:content>` 格式；忽略卡片未声明的字段。"""
        allowed = set(card.allowed_fields) if card else None
        fields: dict[str, str] = {}

        fmt = cls._line_format(card)
        if fmt == "angle":
            raw_parts = [p.strip() for p in payload.splitlines() if p.strip()]
        else:
            # 按未转义的分号切分字段，保留值内部出现的分号
            raw_parts: list[str] = []
            current: list[str] = []
            i = 0
            n = len(payload)
            while i < n:
                if payload[i] == "\\" and i + 1 < n and payload[i + 1] in cls._UNESCAPE_MAP:
                    current.append(payload[i])
                    current.append(payload[i + 1])
                    i += 2
                elif payload[i] == ";":
                    raw_parts.append("".join(current))
                    current = []
                    i += 1
                else:
                    current.append(payload[i])
                    i += 1
            if current:
                raw_parts.append("".join(current))

        for part in raw_parts:
            part = part.strip()
            if not part:
                continue
            if fmt == "angle":
                # 去掉首尾尖括号
                if part.startswith("<") and part.endswith(">"):
                    part = part[1:-1]
                else:
                    continue
            if ":" not in part:
                continue
            key, value = part.split(":", 1)
            key = cls.unescape_value(key).strip()
            value = cls.unescape_value(value).strip()
            if not key or (allowed is not None and key not in allowed):
                continue
            fields[key] = value
        return fields

    @classmethod
    def deserialize_hybrid_strict(
        cls, payload: str, card: ProtocolCard
    ) -> dict[str, str]:
        """严格解析模型直接输出的 wire payload，不忽略任何非法片段。"""
        if cls._line_format(card) != "flat":
            raise ValueError("direct fields-only output currently requires flat wire syntax")
        parts = payload.strip().split(";")
        fields: dict[str, str] = {}
        for part in parts:
            part = part.strip()
            if not part or ":" not in part:
                raise ValueError(f"invalid field segment: {part!r}")
            key, value = part.split(":", 1)
            key, value = key.strip(), value.strip()
            if key not in card.allowed_fields:
                raise ValueError(f"unknown field: {key!r}")
            if key in fields:
                raise ValueError(f"duplicate field: {key!r}")
            if not value:
                raise ValueError(f"empty field value: {key!r}")
            fields[key] = value
        return fields

    @classmethod
    def serialize_nl(cls, message: str) -> str:
        return str(message)

    @classmethod
    def deserialize_nl(cls, payload: str) -> str:
        return str(payload)
