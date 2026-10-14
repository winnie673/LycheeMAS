"""三角色状态机 runner：Planner → Solver → Verifier → 可选 SolverRevision。"""

from __future__ import annotations

import time
from typing import Any

from ..agents.roles import RolePromptBuilder
from ..backends.base import Backend, BackendError
from ..communication.parser import ParsedPayload, PayloadParser
from ..communication.validator import ValidationError
from ..metrics.counter import TokenCounter
from ..protocol.schema import ProtocolSuite
from ..types import CaseResult, CommunicationMode, Decision, ErrorType, MessageRecord, ModelCall


class HybridRunner:
    """按固定状态机执行单题，支持格式重试、回退与失败终止。"""

    def __init__(
        self,
        backend: Backend,
        prompt_builder: RolePromptBuilder,
        token_counter: TokenCounter,
        suite: ProtocolSuite,
        mode: CommunicationMode,
        *,
        max_revision_rounds: int = 1,
        format_retry_limit: int = 1,
        fallback_call_limit: int = 1,
        role_output_limits: dict[str, int] | None = None,
        logger: Any | None = None,
        scorer_kind: str = "exact",
    ):
        self.backend = backend
        self.prompt_builder = prompt_builder
        self.token_counter = token_counter
        self.suite = suite
        self.mode = mode
        self.max_revision_rounds = max_revision_rounds
        self.format_retry_limit = format_retry_limit
        self.fallback_call_limit = fallback_call_limit
        self.role_output_limits = role_output_limits or {
            "planner": 256,
            "solver": 256,
            "verifier": 256,
        }
        self._static_counts = {
            (role, fallback): self.prompt_builder.static_system_token_count(
                role, self.mode, self.suite, self.token_counter, fallback=fallback
            )
            for role in ("planner", "solver", "verifier")
            for fallback in (False, True)
        }
        self.parser = PayloadParser(mode)
        self.logger = logger
        self.scorer_kind = scorer_kind
        self._protocol_hash = suite.meta.get("protocol_hash", "unknown")

    def run_one(
        self,
        task_id: str,
        question: str,
        gold: str,
        context: Any | None = None,
    ) -> CaseResult:
        t_start = time.time()
        result = CaseResult(
            task_id=task_id,
            question=question,
            gold=str(gold),
            mode=self.mode,
            protocol_version=self.suite.version,
        )
        if self.logger:
            self.logger.set_case(task_id)

        # Planner
        planner_msg, planner_fail = self._run_role(
            "planner", "planner_to_solver", result, question, []
        )
        if planner_fail:
            return self._finalize(result, t_start, failed_stage="planner", reason=planner_fail)

        # Solver
        solver_msg, solver_fail = self._run_role(
            "solver", "solver_to_verifier", result, question, [planner_msg]
        )
        if solver_fail:
            return self._finalize(result, t_start, failed_stage="solver", reason=solver_fail)

        # Verifier
        verifier_msg, verifier_fail = self._run_role(
            "verifier", "verifier_to_solver", result, question, [solver_msg]
        )
        if verifier_fail:
            return self._finalize(result, t_start, failed_stage="verifier", reason=verifier_fail)

        decision = self._extract_decision(verifier_msg)
        result.decision = decision
        result.error_type = self._extract_error_type(verifier_msg)

        final_answer = self._answer_from_msg(solver_msg)
        revision_triggered = decision == Decision.REVISE

        if revision_triggered and self.max_revision_rounds > 0:
            result.revision_triggered = True
            revised_solver_msg, revise_fail = self._run_role(
                "solver",
                "solver_to_verifier",
                result,
                question,
                [solver_msg, verifier_msg],
            )
            if revise_fail:
                return self._finalize(
                    result, t_start, failed_stage="solver", reason=revise_fail
                )
            final_answer = self._answer_from_msg(revised_solver_msg)

        return self._finalize(
            result,
            t_start,
            final_answer=final_answer,
            decision=decision,
        )

    def _run_role(
        self,
        role: str,
        edge: str | None,
        result: CaseResult,
        question: str,
        incoming: list[MessageRecord],
    ) -> tuple[MessageRecord | None, str]:
        """执行单个角色并返回交付消息；失败时返回错误原因。"""
        card = self.suite.card_for(edge) if edge else None
        incoming_data = [
            {"edge": m.edge, "payload": m.payload, "mode": str(m.mode)} for m in incoming
        ]
        messages = self.prompt_builder.build_messages(
            role, self.mode, self.suite, question, incoming_data
        )
        max_tokens = self.role_output_limits.get(role, 256)

        attempts: list[ModelCall] = []
        parsed: ParsedPayload | None = None
        attempt = 0
        used_fallback = False
        max_attempts = 1 + self.format_retry_limit + self.fallback_call_limit

        while attempt < max_attempts and parsed is None:
            call = self._call_backend(role, attempt, messages, max_tokens, used_fallback)
            attempts.append(call)
            result.model_calls.append(call)
            if call.transport_failed:
                messages = self._append_error(messages, call.error_message, attempt)
                attempt += 1
                continue

            try:
                if used_fallback:
                    parsed = self.parser.parse_fallback(
                        call.raw_output, edge or "unknown", card, role=role
                    )
                else:
                    parsed = self.parser.parse(call.raw_output, edge or "unknown", card, role=role)
            except (ValidationError, ValueError) as exc:
                call.parse_failed = True
                call.error_message = str(exc)
                if attempt < self.format_retry_limit:
                    target_format = (
                        "valid fields-only syntax"
                        if self.mode == CommunicationMode.HYBRID
                        else "valid JSON"
                    )
                    messages = self._append_error(
                        messages,
                        f"Output did not pass validation: {exc}. "
                        f"Please fix and output {target_format}.",
                        attempt,
                    )
                elif attempt < self.format_retry_limit + self.fallback_call_limit:
                    used_fallback = True
                    messages = self._fallback_messages(
                        role,
                        question,
                        incoming_data,
                        str(exc),
                        card=card,
                        attempt=attempt + 1,
                    )
                attempt += 1
                continue

            call.parsed = parsed.control
            attempt += 1

        for call in attempts:
            result.total_input_tokens += call.prompt_tokens
            result.static_input_tokens += self._static_counts[(call.role, call.fallback)]
            result.total_output_tokens += call.completion_tokens
            result.total_latency_s += call.latency_s

        if parsed is None:
            return None, f"all {len(attempts)} attempts failed for role {role}"

        payload = parsed.payload
        payload_format = parsed.payload_format
        if used_fallback:
            payload_format = "fallback"

        msg = MessageRecord(
            edge=edge or "unknown",
            mode=self.mode,
            payload_format=payload_format,
            fallback=used_fallback,
            payload=payload,
            raw_output=attempts[-1].raw_output,
            protocol_version=self.suite.version,
            protocol_hash=self._protocol_hash,
            sender=role,
            receiver=self._receiver(role),
            call=attempts[-1],
            meta={"attempts": len(attempts)},
        )
        result.messages.append(msg)
        return msg, ""

    def _call_backend(
        self,
        role: str,
        attempt: int,
        messages: list[dict[str, str]],
        max_tokens: int,
        fallback: bool,
    ) -> ModelCall:
        """调用一次后端并返回 ModelCall。"""
        t0 = time.time()
        try:
            gen = self.backend.generate_chat(messages, max_new_tokens=max_tokens)
        except BackendError as exc:
            return ModelCall(
                role=role,
                attempt=attempt,
                raw_output="",
                prompt_tokens=0,
                completion_tokens=0,
                latency_s=round(time.time() - t0, 4),
                transport_failed=True,
                error_message=str(exc),
            )
        return ModelCall(
            role=role,
            attempt=attempt,
            raw_output=gen.text,
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
            latency_s=gen.latency_s or round(time.time() - t0, 4),
            fallback=fallback,
        )

    def _append_error(
        self,
        messages: list[dict[str, str]],
        error_text: str,
        attempt: int,
    ) -> list[dict[str, str]]:
        """为格式重试追加错误提示，保留原始输入证据。"""
        new_messages = list(messages)
        new_messages.append(
            {
                "role": "user",
                "content": f"[format retry {attempt + 1}] {error_text}",
                "meta": {"attempt": attempt + 1},
            }
        )
        return new_messages

    def _fallback_messages(
        self,
        role: str,
        question: str,
        incoming: list[dict[str, Any]],
        error_text: str,
        *,
        card: Any | None = None,
        attempt: int,
    ) -> list[dict[str, str]]:
        """Hybrid 回退保持 fields-only；Pure-NL/终端修订使用简化 JSON。"""
        if self.mode == CommunicationMode.HYBRID and card:
            allowed = ", ".join(card.allowed_fields)
            required = ", ".join(card.required_fields)
            fallback_system = (
                f"role={role}\n\nOutput only FIELD:value segments separated by semicolons. "
                f"Allowed fields: {allowed}. Required fields: {required}. "
                "Do not output JSON, markdown, or separate natural language. "
                "Verifier must output CHK:ACCEPT when no correction is needed."
            )
        else:
            fallback_system = self.prompt_builder.fallback_system_prompt(role)
        incoming_text = "\n\n".join(
            f"Incoming message ({m['edge']}):\n{m['payload']}" for m in incoming
        )
        user = f"Original problem:\n{question}\n\n{incoming_text}\n\nFallback reason: {error_text}"
        return [
            {"role": "system", "content": fallback_system},
            {"role": "user", "content": user, "meta": {"attempt": attempt}},
        ]

    def _receiver(self, role: str) -> str:
        return {"planner": "solver", "solver": "verifier", "verifier": "solver"}.get(
            role, "output"
        )

    def _answer_from_msg(self, msg: MessageRecord) -> str:
        """从已解析消息中提取候选/最终答案。"""
        if not msg.call.parsed:
            return ""
        parsed = msg.call.parsed
        if self.mode == CommunicationMode.HYBRID:
            return parsed.get("fields", {}).get("R", "") or parsed.get("answer", "")
        return parsed.get("answer", "")

    def _extract_decision(self, msg: MessageRecord | None) -> Decision | None:
        if not msg or not msg.call.parsed:
            return None
        parsed = msg.call.parsed
        decision = parsed.get("decision")
        if decision:
            return Decision(decision)
        return None

    def _extract_error_type(self, msg: MessageRecord | None) -> ErrorType | None:
        if not msg or not msg.call.parsed:
            return None
        value = msg.call.parsed.get("error_type")
        if not value:
            return None
        try:
            return ErrorType(value)
        except ValueError:
            return None

    def _finalize(
        self,
        result: CaseResult,
        t_start: float,
        final_answer: str = "",
        failed_stage: str = "",
        reason: str = "",
        decision: Decision | None = None,
    ) -> CaseResult:
        result.final_answer = str(final_answer)
        result.normalized_final_answer = self._normalize_answer(final_answer)
        result.decision = decision or result.decision

        if failed_stage:
            result.run_status = "failed"
            result.failed_stage = failed_stage
            result.failure_reason = reason
        else:
            result.score = self._score(result.normalized_final_answer, result.gold)

        result.communication_tokens = self._communication_tokens(result.messages)
        result.dynamic_input_tokens = result.total_input_tokens - result.static_input_tokens
        result.total_latency_s = round(result.total_latency_s + (time.time() - t_start), 4)
        return result

    def _normalize_answer(self, answer: str) -> str:
        if self.scorer_kind in {"aime", "math500"}:
            return str(answer).strip()
        from lychee_mas.eval.metrics import _last_number

        return _last_number(answer) or answer.strip()

    def _score(self, pred: str, gold: str) -> float:
        from lychee_mas.eval.metrics import score

        return score(self.scorer_kind, pred, gold)

    def _communication_tokens(self, messages: list[MessageRecord]) -> int:
        total = 0
        for m in messages:
            total += self.token_counter.count(m.payload)
        return total
