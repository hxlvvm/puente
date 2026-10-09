import json
from pathlib import Path

from bench.build import build
from bench.score import as_sets, check, grounding, score_case, summarize

CASES = build()


def gold_as_prediction(case):
    g = json.loads(json.dumps(case["gold"]))
    g["conditions"] = [{"name": c} for c in g["conditions"]]
    g["allergies"] = [{"substance": a} for a in g["allergies"]]
    g["needs_clarification"] = [{"term": t} for t in g["needs_clarification"]]
    g["culture_bound"] = [{"term": t} for t in g["culture_bound"]]
    return g


def test_benchmark_is_deterministic_and_unique():
    again = build()
    assert [c["text"] for c in again] == [c["text"] for c in CASES]
    assert len({c["text"] for c in CASES}) == len(CASES) > 250


def test_gold_scores_perfectly():
    rows = [score_case(c, gold_as_prediction(c)) for c in CASES]
    s = summarize(rows)
    assert s["pitfall_pass"] == 1.0 and s["exact_case_acc"] == 1.0 and s["red_flag_recall"] == 1.0
    assert s["symptoms"]["f1"] == 1.0 and s["medications"]["f1"] == 1.0 and s["over_ask_rate"] == 0.0


def test_literal_false_friend_fails():
    food = next(c for c in CASES if c["family"] == "intoxicado" and c["variant"] == "food")
    wrong = gold_as_prediction(food)
    wrong["symptoms"].append({"name": "alcohol_intoxication", "status": "present"})
    assert score_case(food, wrong)["pitfall_pass"] is False


def test_always_correcting_fails_the_control():
    ctrl = next(c for c in CASES if c["family"] == "constipado" and c["variant"] == "control")
    wrong = gold_as_prediction(ctrl)
    wrong["symptoms"] = [{"name": "nasal_congestion", "status": "present"}]
    assert score_case(ctrl, wrong)["pitfall_pass"] is False


def test_ambiguous_needs_a_question():
    amb = next(c for c in CASES if c["variant"] == "ambiguous")
    guess = gold_as_prediction(amb)
    guess["needs_clarification"] = []
    guess["symptoms"].append({"name": "alcohol_intoxication", "status": "present"})
    assert score_case(amb, guess)["pitfall_pass"] is False


def test_denied_is_not_present():
    p = as_sets({"symptoms": [{"name": "fever", "status": "denied"}]})
    assert not check(p, {"require_symptoms": ["fever"]})


def test_normalisation_of_meds_allergies_and_terms():
    p = as_sets({"medications": [{"name": "Metformin (Metformina)", "dose": "850 miligramos", "frequency": "twice_daily"}],
                 "allergies": [{"substance": "Sulfa drugs"}], "culture_bound": [{"term": "ataques de nervios"}]})
    assert p["medications"] == {("metformin", "850 mg", "twice_daily")}
    assert p["allergies"] == {"sulfa"} and check(p, {"require_culture": ["ataque de nervios"]})


def test_grounding_counts_verbatim_quotes():
    pred = {"symptoms": [{"name": "fever", "status": "present", "evidence": "tengo fiebre"},
                         {"name": "cough", "status": "present", "evidence": "tengo tos seca"}]}
    assert grounding(pred, "Hola. Tengo  fiebre desde ayer.") == (1, 2)


def test_invalid_prediction_counts_as_empty():
    row = score_case(CASES[0], None)
    assert row["valid"] is False and row["symptoms"][1] == 0


def test_data_file_matches_builder():
    f = Path(__file__).resolve().parent.parent / "data/cases.jsonl"
    if f.exists():
        assert [json.loads(l)["text"] for l in f.open(encoding="utf-8")] == [c["text"] for c in CASES]


def test_guard_matches_verb_forms_and_skips_held_out():
    from puente.lexicon import guard_notes
    for t in ["me intoxiqué con el alcohol", "está intoxicada", "se intoxicó", "tengo azúcar", "estoy embarazada"]:
        assert guard_notes(t), t
    assert guard_notes("estoy constipado") == [] and guard_notes("la boca del estómago") == []
    assert guard_notes("estoy constipado", include_held_out=True)


def test_dose_units_in_both_languages():
    from bench.score import dose
    assert dose("850 miligramos") == dose("850 milligrams") == dose("850mg") == "850 mg"
