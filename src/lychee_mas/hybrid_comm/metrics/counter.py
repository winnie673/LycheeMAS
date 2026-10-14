"""通信词元计数。"""

from __future__ import annotations

import re
from typing import Any, Optional


class TokenCounter:
    """对 payload 进行 token 计数。

    优先使用 ``transformers.AutoTokenizer``（当提供 tokenizer_name 且本地可加载时）；
    否则退化为按空白切分的近似计数。API 彩排阶段通常无目标 tokenizer，因此近似计数
    被显式记录为 approximate。
    """

    _WHITESPACE_RE = re.compile(r"\s+")

    def __init__(self, tokenizer_name: Optional[str] = None):
        self.tokenizer_name = tokenizer_name
        self._tokenizer = None
        self._approximate = tokenizer_name is None

    def count(self, text: str) -> int:
        if self.tokenizer_name and not self._approximate:
            tok = self._get_tokenizer()
            if tok is not None:
                try:
                    return len(tok.encode(text, add_special_tokens=False))
                except Exception:
                    self._approximate = True
        return len(self._WHITESPACE_RE.split(text.strip())) if text.strip() else 0

    def is_approximate(self) -> bool:
        return self._approximate

    def _get_tokenizer(self) -> Any | None:
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from transformers import AutoTokenizer
        except ImportError:
            self._approximate = True
            return None
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name, trust_remote_code=True
            )
        except Exception:
            self._approximate = True
            return None
        return self._tokenizer
