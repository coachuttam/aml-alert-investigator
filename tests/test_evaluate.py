"""Checks that saved model answers are scored correctly (tiny made-up examples)."""
import json

from src.evaluate import INVALID, UNFAITHFUL_PROBLEM, load_predictions, model_extras, typology_of


def example(alert_id, risk):
    return {"alert_id": alert_id, "label": {"typology": "STRUCTURING", "risk": risk}}


def answer(alert_id, valid=True, typology="STRUCTURING", risk="high", problems=()):
    return {"alert_id": alert_id, "valid": valid, "typology": typology,
            "risk": risk, "reason": "x", "problems": list(problems)}


def test_a_missing_or_unknown_typology_counts_as_invalid():
    assert typology_of(None) == INVALID
    assert typology_of(answer("a", typology="SMURFING")) == INVALID
    assert typology_of(answer("a")) == "STRUCTURING"


def test_extras_count_valid_answers_faithfulness_and_risk():
    examples = [example("a", "high"), example("b", "high"), example("c", "low"), example("d", "low")]
    answers = {
        "a": answer("a"),
        "b": answer("b", valid=False, risk="low", problems=[UNFAITHFUL_PROBLEM]),
        "c": answer("c", risk="low"),
        # "d" has no answer at all
    }
    extras = model_extras(examples, answers)
    assert extras["valid_answers"] == 0.5          # a and c
    assert extras["faithful_reasons"] == 0.75      # b is unfaithful
    assert extras["risk_accuracy"] == 0.5          # a and c


def test_saved_predictions_load_back_keyed_by_alert_id(tmp_path):
    path = tmp_path / "answers.jsonl"
    path.write_text(json.dumps(answer("a")) + "\n" + json.dumps(answer("b")) + "\n")
    assert set(load_predictions(path)) == {"a", "b"}
