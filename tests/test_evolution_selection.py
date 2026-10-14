import json

from scripts.select_evolved_card import candidate_wins, summarize, task_utility


def test_failed_zero_communication_has_zero_utility():
    assert task_utility({"run_status": "failed", "communication_tokens": 0}) == 0.0


def test_completed_utility_is_correct_plus_inverse_tokens():
    assert task_utility(
        {"run_status": "completed", "is_correct": True, "communication_tokens": 100}
    ) == 1.01


def test_candidate_selection_uses_utility_first():
    incumbent = {"n": 2, "mean_utility": 1.005, "accuracy": 1.0,
                 "mean_communication_tokens": 200.0}
    candidate = {"n": 2, "mean_utility": 1.01, "accuracy": 1.0,
                 "mean_communication_tokens": 100.0}
    assert candidate_wins(incumbent, candidate) == (True, "mean_utility")


def test_summary_preserves_unicode_line_separator_inside_json_record(tmp_path):
    path = tmp_path / "outputs.jsonl"
    row = {
        "question": "before\u2028after",
        "run_status": "completed",
        "is_correct": True,
        "communication_tokens": 10,
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    assert summarize(str(path))["correct"] == 1
