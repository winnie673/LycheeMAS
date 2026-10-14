from __future__ import annotations

import pytest
from lychee_mas.eval.metrics import score, score_aime


@pytest.mark.parametrize(
    ("pred", "gold"),
    [
        ("-√3", r"-\sqrt{3}"),
        ("π", r"\pi"),
        ("180°", r"180^\circ"),
        ("76°", r"76^\circ"),
        ("10080", r"10,\!080"),
    ],
)
def test_math500_equivalent_answer_notation(pred: str, gold: str) -> None:
    assert score_aime(pred, gold) == 1.0


@pytest.mark.parametrize(
    ("pred", "gold"),
    [
        ("[-2, 7]", r"x \in [-2,7]"),
        ("11111111100", r"11,\! 111,\! 111,\! 100"),
        ("120 degrees", r"120^\circ"),
        ("3sqrt3/4", r"\frac{3\sqrt{3}}{4}"),
        ("3sqrt5", r"3\sqrt{5}"),
        ("(-18,-49,96)", r"\begin{pmatrix} -18 \\ -49 \\ 96 \end{pmatrix}"),
        ("(1/5,-18/5)", r"\begin{pmatrix} 1/5 \\ -18/5 \end{pmatrix}"),
    ],
)
def test_math500_strict_scorer_accepts_equivalent_forms(pred: str, gold: str) -> None:
    assert score("math500", pred, gold) == 1.0


@pytest.mark.parametrize(
    ("pred", "gold"),
    [
        ("1", "100"),
        ("1.00005", "1"),
        ("[1,2]", "(1,2)"),
        ("(1,2,3)", r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}"),
    ],
)
def test_math500_strict_scorer_rejects_non_equivalent_forms(pred: str, gold: str) -> None:
    assert score("math500", pred, gold) == 0.0
