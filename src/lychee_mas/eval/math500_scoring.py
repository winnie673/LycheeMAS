"""Strict, task-specific equivalence for MATH-500 answer strings."""

from __future__ import annotations

import re
from decimal import Decimal


def _normalize(value: str) -> str:
    from .math_parsing_util import extract_answer, strip_answer_string

    answer = str(value).strip().replace("π", r"\pi")
    if "boxed" in answer:
        answer = extract_answer(answer)
    answer = re.sub(r"^\s*[A-Za-z]\s*\\in\s*", "", answer)
    answer = re.sub(
        r"^([+-]?\d{1,3}(?:,\s*(?:\\!\s*)?\d{3})+)$",
        lambda match: re.sub(r",\s*(?:\\!\s*)?", "", match.group(1)),
        answer,
    )
    answer = re.sub(
        r"\s*(?:degrees?|°|\^?\\circ|\^\{\\circ\})\s*$",
        "",
        answer,
        flags=re.IGNORECASE,
    )
    answer = re.sub(
        r"(?<![A-Za-z\\])(\d+)sqrt(?:\{(\d+)\}|(\d+))",
        lambda match: match.group(1) + r"\sqrt{" + (match.group(2) or match.group(3)) + "}",
        answer,
    )
    if r"\begin{pmatrix}" in answer or r"\begin{bmatrix}" in answer:
        return answer.replace("bmatrix", "pmatrix").replace(" ", "")
    return strip_answer_string(answer)


def _column_vector(value: str) -> list[str] | None:
    match = re.fullmatch(r"\\begin\{pmatrix\}(.*?)\\end\{pmatrix\}", value)
    if not match:
        return None
    rows = [row.strip() for row in match.group(1).split(r"\\")]
    if not rows or any(not row or "&" in row for row in rows):
        return None
    return rows


def _coordinate_tuple(value: str) -> list[str] | None:
    if not (value.startswith("(") and value.endswith(")")):
        return None
    parts = [part.strip() for part in value[1:-1].split(",")]
    return parts if len(parts) >= 2 and all(parts) else None


def _bracketed_sequence(value: str) -> bool:
    return bool(value) and value[0] in "([" and value[-1] in ")]" and "," in value


def score_math500(pred: str, gold: str) -> float:
    """Compare exact mathematical values without AIME's percentage/float tolerance.

    An ordered coordinate tuple is accepted for a one-column vector only. Other
    matrices retain their shape, and intervals retain their open/closed endpoints.
    """
    from .math_parsing_util import math_equal

    prediction, reference = _normalize(pred), _normalize(gold)
    if not prediction or not reference:
        return 0.0
    if prediction == reference:
        return 1.0

    if _bracketed_sequence(prediction) and _bracketed_sequence(reference):
        if prediction[0] != reference[0] or prediction[-1] != reference[-1]:
            return 0.0

    pred_vector, gold_vector = _column_vector(prediction), _column_vector(reference)
    pred_tuple, gold_tuple = _coordinate_tuple(prediction), _coordinate_tuple(reference)
    if (pred_vector and gold_tuple) or (gold_vector and pred_tuple):
        left, right = (pred_vector, gold_tuple) if pred_vector else (pred_tuple, gold_vector)
        assert left is not None and right is not None
        return float(
            len(left) == len(right)
            and all(score_math500(a, b) == 1.0 for a, b in zip(left, right))
        )

    numeric = r"[+-]?\d+(?:\.\d+)?"
    if re.fullmatch(numeric, prediction) and re.fullmatch(numeric, reference):
        return float(Decimal(prediction) == Decimal(reference))
    return float(
        math_equal(
            prediction, reference, include_percentage=False,
            is_close=False, exact_symbolic=True,
        )
    )
