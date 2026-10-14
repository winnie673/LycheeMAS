"""API 后端冒烟测试（不发起真实网络请求）。"""

from __future__ import annotations

import pytest
from lychee_mas.hybrid_comm.backends import BackendError, DeepSeekBackend


def test_deepseek_backend_requires_env_key(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    backend = DeepSeekBackend()
    with pytest.raises(BackendError, match="API key not found"):
        backend.generate_chat([{"role": "user", "content": "hi"}])
