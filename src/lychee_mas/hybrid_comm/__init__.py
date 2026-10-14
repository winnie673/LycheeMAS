"""hybrid_comm — 混合模态通信协议下的三角色协作研究模块.

本模块是 LycheeMAS v0.3 的独立研究扩展，用于实现：
  - Planner → Solver → Verifier 状态机（最多一次 Solver 修正）
  - 纯自然语言（NL）与混合消息（Hybrid）两种通信模式
  - 三张边级协议卡片（P→S、S→V、V→S）的配置化加载与校验
  - 脚本化假后端、OpenAI 兼容 API 后端、HF 本地后端三种统一生成接口
  - 离线协议进化的控制器骨架（Analyzer / Proposer / Evaluator / Selector / Controller）

所有重依赖（openai、yaml、transformers、sympy 等）均在函数/方法内部惰性导入，
保证 `import lychee_mas.hybrid_comm` 不触发 GPU/API 依赖。
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import backends, communication, metrics, pipeline, protocol

__all__ = ["backends", "communication", "metrics", "pipeline", "protocol"]
